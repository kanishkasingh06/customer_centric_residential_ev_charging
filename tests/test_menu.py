from __future__ import annotations

from dataclasses import replace

import pytest

import evmenu.menu as menu_module
from evmenu import (
    ChargingSession,
    ConstructedProfile,
    EVSpec,
    GeneratedMenu,
    MenuGenerationSettings,
    MenuSettings,
    PhysicalConstraintError,
    PlanningSignal,
    SchemaValidationError,
    SignalValidationError,
    ValidationCode,
    ValidationIssue,
    ValidationReport,
    ValidationTolerances,
    generate_candidate_menu,
)
from evmenu.profiles import (
    build_minimum_cost_charging_profile as profile_build_minimum_cost_charging_profile,
)


def ev(*, power: float = 4.0, efficiency: float = 1.0) -> EVSpec:
    return EVSpec("EV-1", 40.0, 4.0, power, efficiency, "LFP")


def session(
    *,
    arrival: int = 0,
    departure: int = 6,
    initial: float = 20.0,
    commute: float = 4.0,
    buffer: float = 2.0,
) -> ChargingSession:
    return ChargingSession(arrival, departure, initial, commute, buffer)


def signal(prices: tuple[float, ...], *, dt: float = 1.0) -> PlanningSignal:
    return PlanningSignal(dt, prices)


def test_generates_bau_and_minimum_cost_candidates() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0)),
    )

    assert menu.ev_id == "EV-1"
    assert any(candidate.kind == "immediate_bau" for candidate in menu.candidates)
    assert any(candidate.kind == "minimum_cost" for candidate in menu.candidates)
    assert all(candidate.validation.is_valid for candidate in menu.candidates)
    assert len({candidate.candidate_id for candidate in menu.candidates}) == len(menu.candidates)


def test_every_target_has_exactly_one_bau_reference() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0)),
    )
    targets = {candidate.target_soc for candidate in menu.candidates}
    for target in targets:
        target_candidates = menu.candidates_for_target(target)
        bau = [candidate for candidate in target_candidates if candidate.kind == "immediate_bau"]
        assert len(bau) == 1
        assert bau[0].saving == pytest.approx(0.0)
        assert bau[0].same_target_bau_cost == pytest.approx(bau[0].charging_cost)


def test_savings_use_same_target_bau() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((10.0, 9.0, 8.0, 1.0, 2.0, 3.0)),
    )
    for candidate in menu.candidates:
        assert candidate.saving == pytest.approx(
            candidate.same_target_bau_cost - candidate.charging_cost
        )


def test_deduplication_keeps_earliest_ready_for_identical_profile() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(initial=30.0),
        signal=signal((10.0, 1.0, 5.0, 6.0, 7.0, 8.0)),
    )
    for target in {candidate.target_soc for candidate in menu.candidates}:
        profiles: dict[tuple[float, ...], int] = {}
        for candidate in menu.candidates_for_target(target):
            if candidate.kind != "minimum_cost":
                continue
            key = candidate.profile.grid_energy_kwh
            assert key not in profiles
            profiles[key] = candidate.ready_step


def test_deduplication_can_be_disabled() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(initial=30.0),
        signal=signal((10.0, 1.0, 5.0, 6.0, 7.0, 8.0)),
        generation_settings=MenuGenerationSettings(deduplicate_identical_profiles=False),
    )
    assert len(menu.candidates) > len(
        generate_candidate_menu(
            ev=ev(),
            session=session(initial=30.0),
            signal=signal((10.0, 1.0, 5.0, 6.0, 7.0, 8.0)),
        ).candidates
    )


def test_no_charge_session_produces_only_nonduplicated_bau_per_target() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(initial=40.0),
        signal=signal((5.0,) * 6),
    )
    assert all(sum(candidate.profile.grid_energy_kwh) == 0.0 for candidate in menu.candidates)
    assert all(candidate.ready_step == 0 for candidate in menu.candidates)
    for target in {candidate.target_soc for candidate in menu.candidates}:
        target_candidates = menu.candidates_for_target(target)
        assert sum(candidate.kind == "immediate_bau" for candidate in target_candidates) == 1
        assert sum(candidate.kind == "minimum_cost" for candidate in target_candidates) == 1


def test_negative_prices_do_not_create_overcharging() -> None:
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(initial=34.0),
        signal=signal((-5.0, -4.0, 10.0, 11.0, 12.0, 13.0)),
    )
    for candidate in menu.candidates:
        assert sum(candidate.profile.grid_energy_kwh) == pytest.approx(
            candidate.required_grid_energy_kwh
        )
        terminal = candidate.profile.battery_energy_kwh[candidate.ready_step]
        expected = max(34.0, candidate.target_soc * 40.0)
        assert terminal == pytest.approx(expected)


def test_nonzero_arrival_and_overnight_horizon() -> None:
    prices = tuple(float(index) for index in range(40))
    menu = generate_candidate_menu(
        ev=ev(power=7.2),
        session=session(arrival=18, departure=32, initial=25.0),
        signal=signal(prices, dt=0.25),
    )
    assert all(candidate.profile.start_step == 18 for candidate in menu.candidates)
    assert all(len(candidate.profile.power_kw) == 14 for candidate in menu.candidates)
    assert all(len(candidate.profile.soc) == 15 for candidate in menu.candidates)


def test_candidate_order_is_deterministic() -> None:
    first = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0)),
    )
    second = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0)),
    )
    assert first == second
    assert tuple(candidate.candidate_id for candidate in first.candidates) == tuple(
        candidate.candidate_id for candidate in second.candidates
    )


def test_generation_can_include_only_one_candidate_family() -> None:
    immediate = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0)),
        generation_settings=MenuGenerationSettings(include_minimum_cost=False),
    )
    assert all(candidate.kind == "immediate_bau" for candidate in immediate.candidates)

    with pytest.raises(SchemaValidationError):
        MenuGenerationSettings(include_immediate_bau=False)


def test_generation_settings_reject_invalid_values() -> None:
    with pytest.raises(SchemaValidationError):
        MenuGenerationSettings(include_immediate_bau=False, include_minimum_cost=False)
    with pytest.raises(TypeError):
        MenuGenerationSettings(saving_tolerance=float("nan"))  # type: ignore[call-arg]
    with pytest.raises(SchemaValidationError):
        MenuGenerationSettings(deduplicate_identical_profiles=1)  # type: ignore[arg-type]


def test_invalid_argument_types_are_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        generate_candidate_menu(
            ev="not-an-ev",  # type: ignore[arg-type]
            session=session(),
            signal=signal((1.0,) * 6),
        )


def test_unsupported_standard_targets_are_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        generate_candidate_menu(
            ev=ev(),
            session=session(),
            signal=signal((1.0,) * 6),
            menu_settings=MenuSettings(standard_targets=(0.75, 0.90, 1.0)),
        )


def test_generated_menu_rejects_mismatched_ev() -> None:
    menu = generate_candidate_menu(
        ev=ev(), session=session(), signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0))
    )
    candidate = replace(menu.candidates[0], ev_id="OTHER")
    with pytest.raises(SchemaValidationError):
        GeneratedMenu(ev_id="EV-1", candidates=(candidate,))


def test_menu_candidate_and_generated_menu_validate_direct_construction() -> None:
    menu = generate_candidate_menu(
        ev=ev(), session=session(), signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0))
    )
    candidate = menu.candidates[0]
    with pytest.raises(SchemaValidationError):
        replace(candidate, ready_step=-1)
    with pytest.raises(SchemaValidationError):
        replace(candidate, target_sources=("standard_80", "minimum_required"))
    with pytest.raises(PhysicalConstraintError):
        GeneratedMenu(ev_id="EV-1", candidates=())
    with pytest.raises(SchemaValidationError):
        GeneratedMenu(ev_id="EV-1", candidates=("not-a-candidate",))  # type: ignore[arg-type]

    copied = GeneratedMenu(ev_id="EV-1", candidates=[candidate])  # type: ignore[arg-type]
    assert copied.candidates == (candidate,)


def test_complete_candidate_ordering_key_is_enforced() -> None:
    menu = generate_candidate_menu(
        ev=ev(), session=session(), signal=signal((8.0, 7.0, 6.0, 2.0, 3.0, 5.0))
    )
    expected = tuple(
        sorted(
            menu.candidates,
            key=lambda candidate: (
                candidate.target_soc,
                candidate.ready_step,
                0 if candidate.kind == "immediate_bau" else 1,
                candidate.charging_cost,
                candidate.candidate_id,
            ),
        )
    )
    assert menu.candidates == expected


def test_bau_and_identical_minimum_cost_profile_are_both_retained() -> None:
    menu = generate_candidate_menu(
        ev=ev(), session=session(), signal=signal((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    )
    for target in {candidate.target_soc for candidate in menu.candidates}:
        bau = next(
            candidate
            for candidate in menu.candidates_for_target(target)
            if candidate.kind == "immediate_bau"
        )
        assert any(
            candidate.kind == "minimum_cost" and candidate.profile == bau.profile
            for candidate in menu.candidates_for_target(target)
        )


def test_negative_saving_is_retained_when_early_profile_is_expensive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = profile_build_minimum_cost_charging_profile

    def expensive_early_profile(
        *,
        ev: EVSpec,
        session: ChargingSession,
        signal: PlanningSignal,
        target_soc: float,
        ready_step: int,
        tolerances: ValidationTolerances | None = None,
    ) -> ConstructedProfile:
        constructed = original(
            ev=ev,
            session=session,
            signal=signal,
            target_soc=target_soc,
            ready_step=ready_step,
            tolerances=tolerances,
        )
        if ready_step == 3:
            return replace(constructed, charging_cost=constructed.charging_cost + 100.0)
        return constructed

    monkeypatch.setattr(menu_module, "build_minimum_cost_charging_profile", expensive_early_profile)
    menu = generate_candidate_menu(
        ev=ev(),
        session=session(),
        signal=signal((10.0, 10.0, 10.0, 1.0, 1.0, 1.0)),
    )
    negative = [candidate for candidate in menu.candidates if candidate.saving < 0.0]
    assert negative
    assert all(candidate.charging_cost > candidate.same_target_bau_cost for candidate in negative)
    assert all(candidate.validation.is_valid for candidate in negative)


def test_constructor_physical_failure_is_not_silently_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_constructor(**_: object) -> ConstructedProfile:
        raise PhysicalConstraintError("synthetic constructor failure")

    monkeypatch.setattr(menu_module, "build_minimum_cost_charging_profile", fail_constructor)
    with pytest.raises(PhysicalConstraintError, match="synthetic constructor failure"):
        generate_candidate_menu(ev=ev(), session=session(), signal=signal((1.0,) * 6))


def test_constructor_invalid_report_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = profile_build_minimum_cost_charging_profile(
        ev=ev(),
        session=session(),
        signal=signal((1.0,) * 6),
        target_soc=0.8,
        ready_step=3,
    )
    invalid = object.__new__(ConstructedProfile)
    object.__setattr__(invalid, "profile", valid.profile)
    object.__setattr__(invalid, "target_soc", valid.target_soc)
    object.__setattr__(invalid, "ready_step", valid.ready_step)
    object.__setattr__(invalid, "required_grid_energy_kwh", valid.required_grid_energy_kwh)
    object.__setattr__(invalid, "charging_cost", valid.charging_cost)
    object.__setattr__(
        invalid,
        "validation",
        ValidationReport(
            issues=(ValidationIssue(ValidationCode.POWER_LIMIT, "synthetic invalid report"),)
        ),
    )

    def invalid_constructor(**_: object) -> ConstructedProfile:
        return invalid

    monkeypatch.setattr(menu_module, "build_minimum_cost_charging_profile", invalid_constructor)
    with pytest.raises(PhysicalConstraintError, match="valid validation report"):
        generate_candidate_menu(ev=ev(), session=session(), signal=signal((1.0,) * 6))


def test_signal_must_cover_menu_session() -> None:
    with pytest.raises(SignalValidationError):
        generate_candidate_menu(ev=ev(), session=session(), signal=signal((1.0,) * 5))


def test_impossible_session_requirement_is_rejected() -> None:
    impossible = ChargingSession(0, 6, 20.0, 35.0, 2.0)
    with pytest.raises(PhysicalConstraintError):
        generate_candidate_menu(
            ev=ev(),
            session=impossible,
            signal=signal((1.0,) * 6),
        )
