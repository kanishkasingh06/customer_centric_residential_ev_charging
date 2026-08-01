from __future__ import annotations

import math

import pytest

from evmenu import ChargingSession, EVSpec, MenuSettings, PlanningSignal
from evmenu.exceptions import PhysicalConstraintError, SchemaValidationError, SignalValidationError
from evmenu.feasibility import (
    available_grid_energy_kwh,
    build_target_options,
    compute_buffer_energy,
    evaluate_request_feasibility,
    minimum_required_target_soc,
    required_grid_energy_kwh,
)


def ev(*, efficiency: float = 0.9) -> EVSpec:
    return EVSpec("EV-1", 50.0, 5.0, 7.0, efficiency, "LFP")


def signal(steps: int = 24, dt: float = 1.0) -> PlanningSignal:
    return PlanningSignal(dt, tuple(1.0 for _ in range(steps)))


def test_buffer_rule_uses_larger_component() -> None:
    assert compute_buffer_energy(10.0, base_buffer_kwh=2.0, commute_buffer_fraction=0.3) == 3.0
    assert compute_buffer_energy(2.0, base_buffer_kwh=2.0, commute_buffer_fraction=0.3) == 2.0


@pytest.mark.parametrize(
    ("commute", "base", "fraction"),
    [
        (math.nan, 1.0, 0.2),
        (1.0, math.inf, 0.2),
        (1.0, -math.inf, 0.2),
        (1.0, 1.0, math.inf),
    ],
)
def test_buffer_rule_rejects_nonfinite_scalars(
    commute: float, base: float, fraction: float
) -> None:
    with pytest.raises(SchemaValidationError):
        compute_buffer_energy(commute, base_buffer_kwh=base, commute_buffer_fraction=fraction)


def test_minimum_required_target_is_commute_plus_buffer_above_floor() -> None:
    session = ChargingSession(4, 12, 20.0, 10.0, 3.0)
    assert minimum_required_target_soc(ev(), session) == pytest.approx((5.0 + 10.0 + 3.0) / 50.0)


def test_required_grid_energy_applies_efficiency_once() -> None:
    vehicle = EVSpec("EV", 40.0, 4.0, 10.0, 0.8, "LFP")
    session = ChargingSession(0, 4, 20.0, 0.0, 0.0)
    assert required_grid_energy_kwh(vehicle, session, 0.6) == pytest.approx(5.0)


def test_current_energy_above_target_requires_no_charging() -> None:
    session = ChargingSession(0, 4, 35.0, 5.0, 2.0)
    assert required_grid_energy_kwh(ev(), session, 0.5) == 0.0


def test_ready_time_available_energy_uses_half_open_window() -> None:
    session = ChargingSession(4, 12, 20.0, 5.0, 2.0)
    assert available_grid_energy_kwh(ev(), session, signal(), ready_step=6) == pytest.approx(14.0)


def test_request_feasibility_boundary_and_partial_energy() -> None:
    vehicle = EVSpec("EV", 40.0, 4.0, 3.0, 1.0, "LFP")
    session = ChargingSession(0, 3, 20.0, 0.0, 0.0)
    result = evaluate_request_feasibility(
        vehicle, session, signal(3), target_soc=0.625, ready_step=2
    )
    assert result.required_grid_energy_kwh == pytest.approx(5.0)
    assert result.available_grid_energy_kwh == pytest.approx(6.0)
    assert result.is_feasible


def test_zero_request_is_feasible_at_arrival_when_current_energy_is_sufficient() -> None:
    vehicle = EVSpec("EV", 40.0, 4.0, 3.0, 1.0, "LFP")
    session = ChargingSession(0, 3, 30.0, 0.0, 0.0)
    result = evaluate_request_feasibility(vehicle, session, signal(3), target_soc=0.6, ready_step=0)
    assert result.required_grid_energy_kwh == 0.0
    assert result.available_grid_energy_kwh == 0.0
    assert result.is_feasible


def test_ready_at_departure_is_a_valid_request_boundary() -> None:
    session = ChargingSession(4, 12, 20.0, 5.0, 2.0)
    result = evaluate_request_feasibility(ev(), session, signal(), target_soc=0.8, ready_step=12)
    assert result.is_feasible


def test_nonfinite_target_and_tolerance_are_rejected() -> None:
    session = ChargingSession(0, 4, 20.0, 0.0, 0.0)
    with pytest.raises(SchemaValidationError):
        evaluate_request_feasibility(ev(), session, signal(), target_soc=math.nan, ready_step=1)
    with pytest.raises(SchemaValidationError):
        evaluate_request_feasibility(ev(), session, signal(), target_soc=math.inf, ready_step=1)
    with pytest.raises(SchemaValidationError):
        evaluate_request_feasibility(
            ev(), session, signal(), target_soc=0.8, ready_step=1, tolerance=math.inf
        )
    with pytest.raises(SchemaValidationError):
        evaluate_request_feasibility(
            ev(), session, signal(), target_soc=0.8, ready_step=1, tolerance=math.nan
        )


def test_ready_step_outside_session_is_rejected() -> None:
    session = ChargingSession(4, 12, 20.0, 5.0, 2.0)
    with pytest.raises(PhysicalConstraintError):
        available_grid_energy_kwh(ev(), session, signal(), ready_step=3)
    with pytest.raises(PhysicalConstraintError):
        available_grid_energy_kwh(ev(), session, signal(), ready_step=13)


def test_bool_ready_step_is_rejected() -> None:
    session = ChargingSession(4, 12, 20.0, 5.0, 2.0)
    with pytest.raises(SchemaValidationError):
        available_grid_energy_kwh(ev(), session, signal(), ready_step=True)


def test_target_set_contains_personalized_and_standard_targets() -> None:
    session = ChargingSession(4, 12, 20.0, 10.0, 3.0)
    options = build_target_options(ev(), session, MenuSettings())
    assert [option.target_soc for option in options] == pytest.approx([0.36, 0.8, 0.9, 1.0])
    assert options[0].sources == ("minimum_required",)


def test_target_near_80_percent_is_merged_with_provenance() -> None:
    # (5 + 32 + 2.5) / 50 = 0.79
    session = ChargingSession(0, 12, 20.0, 32.0, 2.5)
    options = build_target_options(ev(), session, MenuSettings(target_merge_tolerance=0.02))
    assert options[0].target_soc == pytest.approx(0.8)
    assert options[0].sources == ("minimum_required", "standard_80")


def test_standard_targets_below_service_requirement_are_omitted() -> None:
    # z_min = 0.86
    session = ChargingSession(0, 12, 20.0, 35.0, 3.0)
    options = build_target_options(ev(), session, MenuSettings())
    assert [option.target_soc for option in options] == pytest.approx([0.86, 0.9, 1.0])
    assert all("standard_80" not in option.sources for option in options)


@pytest.mark.parametrize(
    ("commute", "expected_targets"),
    [
        (13.0, (0.36, 0.8, 0.9, 1.0)),
        (34.75, (0.8, 0.9, 1.0)),
        (35.25, (0.805, 0.9, 1.0)),
        (38.0, (0.86, 0.9, 1.0)),
        (42.0, (0.94, 1.0)),
        (45.0, (1.0,)),
    ],
)
def test_target_options_cover_service_boundary_merge_cases(
    commute: float, expected_targets: tuple[float, ...]
) -> None:
    session = ChargingSession(0, 12, 20.0, commute, 0.0)
    options = build_target_options(ev(), session, MenuSettings())
    assert [option.target_soc for option in options] == pytest.approx(expected_targets)
    assert len({option.target_soc for option in options}) == len(options)
    assert [option.target_soc for option in options] == sorted(
        option.target_soc for option in options
    )


def test_target_options_reject_unsupported_custom_standard_targets() -> None:
    session = ChargingSession(0, 12, 20.0, 10.0, 3.0)
    settings = MenuSettings(standard_targets=(0.75, 0.9, 1.0))
    with pytest.raises(SchemaValidationError):
        build_target_options(ev(), session, settings)


def test_personalized_target_exactly_one_is_valid() -> None:
    session = ChargingSession(0, 12, 20.0, 45.0, 0.0)
    options = build_target_options(ev(), session, MenuSettings())
    assert len(options) == 1
    assert options[0].target_soc == pytest.approx(1.0)


def test_personalized_target_above_one_is_physical_error() -> None:
    session = ChargingSession(0, 12, 20.0, 46.0, 0.0)
    with pytest.raises(PhysicalConstraintError):
        build_target_options(ev(), session, MenuSettings())


def test_signal_must_cover_session_for_request_evaluation() -> None:
    session = ChargingSession(18, 32, 20.0, 5.0, 2.0)
    with pytest.raises(SignalValidationError):
        evaluate_request_feasibility(ev(), session, signal(24), target_soc=0.8, ready_step=30)
