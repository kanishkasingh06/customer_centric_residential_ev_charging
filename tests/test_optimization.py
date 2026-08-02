from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

import evmenu.optimization as optimization_module
from evmenu.degradation import DegradationSettings
from evmenu.exceptions import PhysicalConstraintError, SchemaValidationError
from evmenu.menu import MenuCandidate, MenuGenerationSettings, generate_candidate_menu
from evmenu.optimization import (
    FrontierSettings,
    OptimizedProfile,
    SavingFrontier,
    build_least_degradation_profile,
    build_sandwich_saving_frontier,
    build_saving_constrained_profile,
)
from evmenu.schemas import ChargingSession, EVSpec, PlanningSignal


def _ev() -> EVSpec:
    return EVSpec(
        ev_id="ev-1",
        battery_capacity_kwh=40.0,
        minimum_energy_kwh=4.0,
        charger_power_kw=4.0,
        charging_efficiency=1.0,
        chemistry="NMC",
    )


def _session() -> ChargingSession:
    return ChargingSession(
        arrival_step=0,
        departure_step=6,
        initial_energy_kwh=20.0,
        commute_energy_kwh=4.0,
        buffer_energy_kwh=2.0,
    )


def _signal() -> PlanningSignal:
    return PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=(10.0, 9.0, 8.0, 2.0, 1.0, 3.0),
        battery_temperature_c=(30.0,) * 6,
    )


def _candidate() -> tuple[MenuCandidate, float]:
    menu = generate_candidate_menu(ev=_ev(), session=_session(), signal=_signal())
    candidates = [
        item
        for item in menu.candidates
        if item.kind == "minimum_cost" and item.target_soc == 0.8 and item.ready_step == 6
    ]
    assert len(candidates) == 1
    candidate = candidates[0]
    return candidate, candidate.same_target_bau_cost


def _service_like_context(
    *, ev: EVSpec, session: ChargingSession, target_soc: float = 0.8, ready_step: int = 17
) -> tuple[EVSpec, ChargingSession, PlanningSignal, MenuCandidate]:
    prices = tuple(
        4.0
        if (19 * 60 + step * 15) % 1440 < 6 * 60
        else 7.0
        if (19 * 60 + step * 15) % 1440 < 17 * 60
        else 10.0
        if (19 * 60 + step * 15) % 1440 < 23 * 60
        else 5.0
        for step in range(session.departure_step - session.arrival_step)
    )
    signal = PlanningSignal(
        timestep_hours=0.25,
        price_per_kwh=prices,
        battery_temperature_c=(30.0,) * len(prices),
    )
    menu = generate_candidate_menu(ev=ev, session=session, signal=signal)
    candidate = next(
        item
        for item in menu.candidates
        if item.kind == "minimum_cost"
        and item.target_soc == target_soc
        and item.ready_step == ready_step
    )
    return ev, session, signal, candidate


def test_generic_60_nmc_former_service_failure_is_fixed() -> None:
    ev, session, signal, candidate = _service_like_context(
        ev=EVSpec(
            ev_id="generic_60kwh_nmc",
            battery_capacity_kwh=60.0,
            minimum_energy_kwh=3.0,
            charger_power_kw=7.2,
            charging_efficiency=0.9,
            chemistry="NMC",
        ),
        session=ChargingSession(
            arrival_step=0,
            departure_step=48,
            initial_energy_kwh=21.0,
            commute_energy_kwh=7.65,
            buffer_energy_kwh=6.0,
        ),
    )
    frontier = build_sandwich_saving_frontier(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
    )
    assert all(
        right.trajectory_objective + 1e-8 >= left.trajectory_objective
        for left, right in zip(frontier.points, frontier.points[1:], strict=False)
    )


def test_generic_40_soc_half_endpoint_is_clamped_without_relaxing_validation() -> None:
    ev, session, signal, candidate = _service_like_context(
        ev=EVSpec(
            ev_id="generic_40kwh_lfp",
            battery_capacity_kwh=40.0,
            minimum_energy_kwh=2.0,
            charger_power_kw=7.2,
            charging_efficiency=0.9,
            chemistry="LFP",
        ),
        session=ChargingSession(
            arrival_step=0,
            departure_step=48,
            initial_energy_kwh=20.0,
            commute_energy_kwh=0.0,
            buffer_energy_kwh=4.0,
        ),
        target_soc=1.0,
        ready_step=29,
    )
    frontier = build_sandwich_saving_frontier(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
    )
    assert frontier.points
    assert all(point.constructed.validation.is_valid for point in frontier.points)
    assert all(
        0.0 <= soc <= 1.0 for point in frontier.points for soc in point.constructed.profile.soc
    )


def test_frontier_settings_validation() -> None:
    with pytest.raises(PhysicalConstraintError):
        FrontierSettings(maximum_levels=1)
    with pytest.raises(SchemaValidationError):
        FrontierSettings(maximum_levels=True)
    with pytest.raises(PhysicalConstraintError):
        FrontierSettings(saving_band=-1.0)


def test_least_degradation_profile_is_valid_and_delays_charge() -> None:
    candidate, bau_cost = _candidate()
    result = build_least_degradation_profile(
        ev=_ev(), session=_session(), signal=_signal(), candidate=candidate, bau_cost=bau_cost
    )
    assert result.constructed.validation.is_valid
    assert sum(result.constructed.profile.grid_energy_kwh) == pytest.approx(
        candidate.required_grid_energy_kwh
    )
    assert all(value == pytest.approx(0.0) for value in result.constructed.profile.power_kw[6:])
    assert result.trajectory_objective >= 0.0
    assert result.assessment.total_fade >= 0.0


def test_least_degradation_no_worse_objective_than_max_saving() -> None:
    candidate, bau_cost = _candidate()
    least = build_least_degradation_profile(
        ev=_ev(), session=_session(), signal=_signal(), candidate=candidate, bau_cost=bau_cost
    )
    maximum = build_saving_constrained_profile(
        ev=_ev(),
        session=_session(),
        signal=_signal(),
        candidate=candidate,
        bau_cost=bau_cost,
        requested_saving=candidate.saving,
    )
    assert least.trajectory_objective <= maximum.trajectory_objective + 1e-8
    assert maximum.saving == pytest.approx(candidate.saving, abs=2e-6)


def test_saving_constrained_midpoint_respects_band() -> None:
    candidate, bau_cost = _candidate()
    least = build_least_degradation_profile(
        ev=_ev(), session=_session(), signal=_signal(), candidate=candidate, bau_cost=bau_cost
    )
    midpoint = (least.saving + candidate.saving) / 2.0
    result = build_saving_constrained_profile(
        ev=_ev(),
        session=_session(),
        signal=_signal(),
        candidate=candidate,
        bau_cost=bau_cost,
        requested_saving=midpoint,
        frontier_settings=FrontierSettings(saving_band=1e-5),
    )
    assert result.saving == pytest.approx(midpoint, abs=1.1e-5)
    assert result.constructed.validation.is_valid


def test_sandwich_frontier_has_anchors_and_order() -> None:
    candidate, bau_cost = _candidate()
    frontier = build_sandwich_saving_frontier(
        ev=_ev(),
        session=_session(),
        signal=_signal(),
        candidate=candidate,
        bau_cost=bau_cost,
        frontier_settings=FrontierSettings(maximum_levels=5, saving_band=1e-5),
    )
    assert 1 <= len(frontier.points) <= 5
    savings = [point.saving for point in frontier.points]
    assert savings == sorted(savings)
    assert savings[-1] == pytest.approx(candidate.saving, abs=1.1e-5)
    assert all(point.constructed.validation.is_valid for point in frontier.points)


def test_repeated_frontier_is_deterministic() -> None:
    candidate, bau_cost = _candidate()
    settings = FrontierSettings(maximum_levels=4, saving_band=1e-5)
    first = build_sandwich_saving_frontier(
        ev=_ev(),
        session=_session(),
        signal=_signal(),
        candidate=candidate,
        bau_cost=bau_cost,
        frontier_settings=settings,
    )
    second = build_sandwich_saving_frontier(
        ev=_ev(),
        session=_session(),
        signal=_signal(),
        candidate=candidate,
        bau_cost=bau_cost,
        frontier_settings=settings,
    )
    assert first == second


def test_infeasible_requested_saving_fails() -> None:
    candidate, bau_cost = _candidate()
    with pytest.raises(PhysicalConstraintError):
        build_saving_constrained_profile(
            ev=_ev(),
            session=_session(),
            signal=_signal(),
            candidate=candidate,
            bau_cost=bau_cost,
            requested_saving=candidate.saving + 100.0,
        )


def test_invalid_public_types_fail_cleanly() -> None:
    candidate, bau_cost = _candidate()
    with pytest.raises(SchemaValidationError):
        build_least_degradation_profile(
            ev="bad",  # type: ignore[arg-type]
            session=_session(),
            signal=_signal(),
            candidate=candidate,
            bau_cost=bau_cost,
        )


def test_zero_energy_request_is_supported() -> None:
    ev = _ev()
    session = ChargingSession(
        arrival_step=0,
        departure_step=4,
        initial_energy_kwh=40.0,
        commute_energy_kwh=4.0,
        buffer_energy_kwh=2.0,
    )
    signal = PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=(4.0, 3.0, 2.0, 1.0),
        battery_temperature_c=(30.0,) * 4,
    )
    menu = generate_candidate_menu(ev=ev, session=session, signal=signal)
    candidate = next(item for item in menu.candidates if item.kind == "minimum_cost")
    result = build_least_degradation_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        degradation_settings=DegradationSettings(parked_day_hours=0.0),
    )
    assert result.constructed.profile.grid_energy_kwh == (0.0, 0.0, 0.0, 0.0)
    assert result.saving == pytest.approx(0.0)


def _flexible_case(
    *,
    prices: tuple[float, ...] = (1.0, 2.0, 3.0, 10.0, 10.0, 10.0),
    charger_power_kw: float = 10.0,
    charging_efficiency: float = 1.0,
    ready_step: int | None = None,
) -> tuple[EVSpec, ChargingSession, PlanningSignal, MenuCandidate]:
    ev = EVSpec(
        ev_id="flexible-ev",
        battery_capacity_kwh=40.0,
        minimum_energy_kwh=4.0,
        charger_power_kw=charger_power_kw,
        charging_efficiency=charging_efficiency,
        chemistry="NMC",
    )
    session = ChargingSession(
        arrival_step=0,
        departure_step=len(prices),
        initial_energy_kwh=20.0,
        commute_energy_kwh=4.0,
        buffer_energy_kwh=2.0,
    )
    signal = PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=prices,
        battery_temperature_c=(30.0,) * len(prices),
    )
    menu = generate_candidate_menu(
        ev=ev,
        session=session,
        signal=signal,
        generation_settings=MenuGenerationSettings(deduplicate_identical_profiles=False),
    )
    selected_ready_step = len(prices) if ready_step is None else ready_step
    candidate = next(
        item
        for item in menu.candidates
        if item.kind == "minimum_cost"
        and item.target_soc == 0.8
        and item.ready_step == selected_ready_step
    )
    return ev, session, signal, candidate


def test_objective_scale_and_tolerance_validation() -> None:
    with pytest.raises(PhysicalConstraintError):
        FrontierSettings(objective_scale=0.0)
    with pytest.raises(SchemaValidationError):
        FrontierSettings(objective_scale=float("nan"))
    with pytest.raises(PhysicalConstraintError):
        FrontierSettings(saving_tolerance=-1.0)
    settings = FrontierSettings(objective_scale=1_000_000.0, solver_ftol=1e-10)
    assert settings.effective_solver_ftol == pytest.approx(1e-10)


def test_dense_grid_matches_least_degradation_solution() -> None:
    ev, session, signal, candidate = _flexible_case(
        prices=(1.0, 2.0, 3.0, 10.0, 10.0, 10.0), ready_step=3
    )
    result = build_least_degradation_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        frontier_settings=FrontierSettings(plating_guard_weight=0.0),
    )
    required = candidate.required_grid_energy_kwh
    dense_best = float("inf")
    for first_index in range(101):
        first = 10.0 * first_index / 100.0
        for second_index in range(101):
            second = 10.0 * second_index / 100.0
            third = required - first - second
            if -1e-12 <= third <= 10.0 + 1e-12:
                dense_best = min(
                    dense_best,
                    optimization_module._trajectory_objective(
                        [first, second, third],
                        ev=ev,
                        session=session,
                        signal=signal,
                        degradation_settings=DegradationSettings(),
                        frontier_settings=FrontierSettings(plating_guard_weight=0.0),
                    ),
                )
    assert result.trajectory_objective <= dense_best + 1e-7
    assert result.constructed.profile.grid_energy_kwh[2] == pytest.approx(10.0, abs=1e-5)


def test_objective_jacobian_matches_finite_difference() -> None:
    ev, session, signal, _ = _flexible_case()
    settings = FrontierSettings(plating_guard_weight=1e-3)
    x = [3.0, 4.0, 5.0]
    value, jacobian = optimization_module._objective_value_and_jac(
        x,
        ev=ev,
        session=session,
        signal=signal,
        degradation_settings=DegradationSettings(),
        frontier_settings=settings,
    )
    assert value > 0.0
    for index, analytical in enumerate(jacobian):
        step = 1e-6
        plus = x.copy()
        minus = x.copy()
        plus[index] += step
        minus[index] -= step
        numerical = (
            optimization_module._trajectory_objective(
                plus,
                ev=ev,
                session=session,
                signal=signal,
                degradation_settings=DegradationSettings(),
                frontier_settings=settings,
            )
            - optimization_module._trajectory_objective(
                minus,
                ev=ev,
                session=session,
                signal=signal,
                degradation_settings=DegradationSettings(),
                frontier_settings=settings,
            )
        ) / (2.0 * step)
        assert analytical == pytest.approx(numerical, rel=1e-5, abs=1e-9)


def test_strong_plating_guard_prefers_equal_spread() -> None:
    ev, session, signal, candidate = _flexible_case(prices=(1.0,) * 6)
    result = build_least_degradation_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        frontier_settings=FrontierSettings(plating_guard_weight=1.0),
    )
    eligible = result.constructed.profile.grid_energy_kwh[: candidate.ready_step]
    assert max(eligible) - min(eligible) < 1e-3


def test_partial_interval_and_negative_price_constraints() -> None:
    ev, session, signal, candidate = _flexible_case(
        prices=(-5.0, -4.0, -3.0, 2.0, 3.0, 4.0), charging_efficiency=0.8
    )
    result = build_least_degradation_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
    )
    assert sum(result.constructed.profile.grid_energy_kwh) == pytest.approx(
        candidate.required_grid_energy_kwh
    )
    assert result.constructed.validation.is_valid
    constrained = build_saving_constrained_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        requested_saving=candidate.saving,
        frontier_settings=FrontierSettings(saving_band=0.0),
    )
    assert constrained.constructed.validation.is_valid


def test_saving_range_prevalidation_and_zero_width_level() -> None:
    ev, session, signal, candidate = _flexible_case()
    with pytest.raises(PhysicalConstraintError, match="attainable"):
        build_saving_constrained_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=candidate.same_target_bau_cost,
            requested_saving=candidate.saving + 100.0,
        )
    least = build_least_degradation_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        frontier_settings=FrontierSettings(plating_guard_weight=0.0),
    )
    exact = build_saving_constrained_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        requested_saving=(least.saving + candidate.saving) / 2.0,
        frontier_settings=FrontierSettings(saving_band=0.0, plating_guard_weight=0.0),
    )
    assert exact.saving == pytest.approx((least.saving + candidate.saving) / 2.0, abs=1e-7)


def test_bau_and_candidate_saving_metadata_are_validated() -> None:
    ev, session, signal, candidate = _flexible_case()
    with pytest.raises(SchemaValidationError, match="bau_cost"):
        build_least_degradation_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=candidate.same_target_bau_cost + 1.0,
        )
    altered = replace(candidate, saving=candidate.saving + 1.0)
    with pytest.raises(SchemaValidationError, match="saving metadata"):
        build_least_degradation_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=altered,
            bau_cost=altered.same_target_bau_cost,
        )


def test_immediate_bau_candidates_are_rejected_explicitly() -> None:
    ev, session, signal, _ = _flexible_case()
    menu = generate_candidate_menu(ev=ev, session=session, signal=signal)
    candidate = next(item for item in menu.candidates if item.kind == "immediate_bau")
    with pytest.raises(SchemaValidationError, match="minimum_cost"):
        build_least_degradation_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=candidate.same_target_bau_cost,
        )


def test_maximum_endpoint_preserves_analytical_profile_and_identity() -> None:
    ev, session, signal, candidate = _flexible_case()
    result = build_saving_constrained_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
        requested_saving=candidate.saving,
    )
    assert result.constructed.profile == candidate.profile
    assert result.endpoint_role == "maximum_saving"
    assert result.point_id != result.source_candidate_id
    assert result.assessment.candidate_id == result.point_id


@pytest.mark.parametrize(
    "bad_vector",
    [None, [0.0], [float("nan")] * 6, [float("inf")] * 6, [-1.0] * 6, [11.0] * 6, [0.0] * 6],
)
def test_malformed_successful_solver_vectors_are_rejected(
    monkeypatch: pytest.MonkeyPatch, bad_vector: object
) -> None:
    ev, session, signal, candidate = _flexible_case()

    def fake_minimize(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(success=True, message="ok", x=bad_vector, fun=1.0)

    monkeypatch.setattr(optimization_module, "minimize", fake_minimize)
    with pytest.raises(PhysicalConstraintError):
        build_least_degradation_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=candidate.same_target_bau_cost,
        )


def test_frontier_collapses_exact_endpoints_and_assigns_unique_roles() -> None:
    ev, session, signal, candidate = _flexible_case(
        prices=(10.0, 9.0, 8.0, 2.0, 1.0, 3.0), charger_power_kw=4.0
    )
    frontier = build_sandwich_saving_frontier(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
    )
    assert len(frontier.points) == 1
    assert frontier.points[0].endpoint_role == "least_and_maximum"
    assert frontier.points[0].point_id == frontier.points[0].assessment.candidate_id


def test_frontier_direct_validation_rejects_malformed_identity() -> None:
    candidate, bau_cost = _candidate()
    point = build_least_degradation_profile(
        ev=_ev(),
        session=_session(),
        signal=_signal(),
        candidate=candidate,
        bau_cost=bau_cost,
    )
    with pytest.raises(SchemaValidationError):
        SavingFrontier(
            ev_id="ev-1",
            target_soc=0.8,
            ready_step=6,
            bau_cost=bau_cost,
            points=cast(tuple[OptimizedProfile, ...], None),
            source_candidate_id=candidate.candidate_id,
        )
    with pytest.raises(PhysicalConstraintError):
        SavingFrontier(
            ev_id="ev-1",
            target_soc=2.0,
            ready_step=6,
            bau_cost=bau_cost,
            points=(point,),
            source_candidate_id=candidate.candidate_id,
        )


def test_nonzero_arrival_and_frontier_point_ids_are_deterministic() -> None:
    prices = tuple(float(index) for index in range(10))
    ev = EVSpec(
        ev_id="overnight",
        battery_capacity_kwh=40.0,
        minimum_energy_kwh=4.0,
        charger_power_kw=10.0,
        charging_efficiency=1.0,
        chemistry="NMC",
    )
    session = ChargingSession(
        arrival_step=2,
        departure_step=8,
        initial_energy_kwh=20.0,
        commute_energy_kwh=4.0,
        buffer_energy_kwh=2.0,
    )
    signal = PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=prices,
        battery_temperature_c=(30.0,) * len(prices),
    )
    menu = generate_candidate_menu(
        ev=ev,
        session=session,
        signal=signal,
        generation_settings=MenuGenerationSettings(deduplicate_identical_profiles=False),
    )
    candidate = next(
        item
        for item in menu.candidates
        if item.kind == "minimum_cost" and item.target_soc == 0.8 and item.ready_step == 8
    )
    first = build_sandwich_saving_frontier(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
    )
    second = build_sandwich_saving_frontier(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=candidate.same_target_bau_cost,
    )
    assert first == second
    assert first.points[0].constructed.profile.start_step == 2
    assert len({point.point_id for point in first.points}) == len(first.points)
