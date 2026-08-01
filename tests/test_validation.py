from __future__ import annotations

from typing import cast

import pytest

from evmenu import ChargingProfile, ChargingSession, EVSpec, PlanningSignal
from evmenu.exceptions import SchemaValidationError
from evmenu.validation import (
    ValidationCode,
    ValidationReport,
    ValidationTolerances,
    validate_charging_profile,
)


def ev(*, efficiency: float = 1.0, charger_power: float = 4.0) -> EVSpec:
    return EVSpec("EV-1", 40.0, 4.0, charger_power, efficiency, "LFP")


def signal(steps: int = 4, dt: float = 1.0) -> PlanningSignal:
    return PlanningSignal(dt, tuple(1.0 for _ in range(steps)))


def valid_profile() -> ChargingProfile:
    return ChargingProfile(
        start_step=0,
        grid_energy_kwh=(4.0, 0.0, 0.0, 0.0),
        battery_energy_kwh=(20.0, 24.0, 24.0, 24.0, 24.0),
        power_kw=(4.0, 0.0, 0.0, 0.0),
        soc=(0.5, 0.6, 0.6, 0.6, 0.6),
    )


def validate(
    profile: ChargingProfile,
    *,
    ready: int = 1,
    target: float = 0.6,
    vehicle: EVSpec | None = None,
) -> ValidationReport:
    return validate_charging_profile(
        ev=vehicle or ev(),
        session=ChargingSession(0, 4, 20.0, 5.0, 2.0),
        signal=signal(),
        target_soc=target,
        ready_step=ready,
        profile=profile,
    )


def test_valid_physically_consistent_profile_passes() -> None:
    report = validate(valid_profile())
    assert report.is_valid
    assert report.errors == ()
    assert report.ready_energy_kwh == pytest.approx(24.0)
    assert report.minimum_commute_margin_kwh == pytest.approx(13.0)


def test_valid_zero_charging_profile_when_current_energy_exceeds_target() -> None:
    profile = ChargingProfile(
        0,
        (0.0, 0.0, 0.0, 0.0),
        (30.0, 30.0, 30.0, 30.0, 30.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.75, 0.75, 0.75, 0.75, 0.75),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 30.0, 5.0, 2.0),
        signal=signal(),
        target_soc=0.6,
        ready_step=0,
        profile=profile,
    )
    assert report.is_valid


def test_wrong_profile_start_step_is_reported() -> None:
    profile = ChargingProfile(
        1,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.6),
    )
    assert any("start_step" in error for error in validate(profile).errors)


def test_power_above_charger_rating_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (5.0, 0.0, 0.0, 0.0),
        (20.0, 25.0, 25.0, 25.0, 25.0),
        (5.0, 0.0, 0.0, 0.0),
        (0.5, 0.625, 0.625, 0.625, 0.625),
    )
    report = validate(profile, target=0.625)
    assert any("charger limit" in error for error in report.errors)


def test_charging_at_or_after_ready_step_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (0.0, 4.0, 0.0, 0.0),
        (20.0, 20.0, 24.0, 24.0, 24.0),
        (0.0, 4.0, 0.0, 0.0),
        (0.5, 0.5, 0.6, 0.6, 0.6),
    )
    report = validate(profile, ready=1)
    assert any("at or after ready_step" in error for error in report.errors)


def test_grid_energy_power_inconsistency_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (3.0, 0.0, 0.0, 0.0),
        (20.0, 23.0, 23.0, 23.0, 23.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.575, 0.575, 0.575, 0.575),
    )
    report = validate(profile, target=0.575)
    assert any("grid energy and power" in error for error in report.errors)


def test_battery_energy_recursion_violation_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 23.0, 23.0, 23.0, 23.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.575, 0.575, 0.575, 0.575),
    )
    assert any("recursion" in error for error in validate(profile, target=0.575).errors)


def test_efficiency_is_checked_in_battery_recursion() -> None:
    profile = ChargingProfile(
        0,
        (5.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (5.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.6),
    )
    report = validate(profile, vehicle=ev(efficiency=0.8, charger_power=5.0))
    assert report.is_valid


def test_battery_bounds_violation_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (0.0, 0.0, 0.0, 0.0),
        (3.0, 3.0, 3.0, 3.0, 3.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.075, 0.075, 0.075, 0.075, 0.075),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 4.0, 0.0, 0.0),
        signal=signal(),
        target_soc=0.1,
        ready_step=0,
        profile=profile,
    )
    assert any("below B_min" in error for error in report.errors)


def test_soc_inconsistency_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.7, 0.7, 0.7, 0.7),
    )
    assert any("SOC is inconsistent" in error for error in validate(profile).errors)


def test_exact_target_violation_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (3.0, 0.0, 0.0, 0.0),
        (20.0, 23.0, 23.0, 23.0, 23.0),
        (3.0, 0.0, 0.0, 0.0),
        (0.5, 0.575, 0.575, 0.575, 0.575),
    )
    assert any("exact delivered target" in error for error in validate(profile).errors)


def test_commute_plus_buffer_violation_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (0.0, 0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0, 20.0, 20.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5, 0.5),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 20.0, 14.0, 3.0),
        signal=signal(),
        target_soc=0.5,
        ready_step=0,
        profile=profile,
    )
    assert any("commute plus buffer" in error for error in report.errors)


def test_profile_must_cover_complete_session() -> None:
    profile = ChargingProfile(0, (4.0,), (20.0, 24.0), (4.0,), (0.5, 0.6))
    assert any("span every interval" in error for error in validate(profile).errors)


@pytest.mark.parametrize("field", ["power_kw", "energy_kwh", "soc"])
@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_validation_tolerances_reject_nonpositive_nonfinite_or_bool(
    field: str, value: object
) -> None:
    with pytest.raises(SchemaValidationError):
        ValidationTolerances(**{field: cast(float, value)})


def test_tolerances_are_applied_by_physical_quantity() -> None:
    profile = ChargingProfile(
        0,
        (4.0 + 5e-7, 0.0, 0.0, 0.0),
        (20.0, 24.0 + 5e-7, 24.0 + 5e-7, 24.0 + 5e-7, 24.0 + 5e-7),
        (4.0 + 5e-7, 0.0, 0.0, 0.0),
        (
            0.5,
            (24.0 + 5e-7) / 40.0,
            (24.0 + 5e-7) / 40.0,
            (24.0 + 5e-7) / 40.0,
            (24.0 + 5e-7) / 40.0,
        ),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 20.0, 5.0, 2.0),
        signal=signal(),
        target_soc=0.6,
        ready_step=1,
        profile=profile,
        tolerances=ValidationTolerances(power_kw=1e-6, energy_kwh=1e-6),
    )
    assert report.is_valid


def test_validation_returns_cross_object_issues_in_stable_order() -> None:
    profile = ChargingProfile(
        0,
        (5.0, 0.0, 0.0, 0.0),
        (3.0, 30.0, 30.0, 30.0, 30.0),
        (5.0, 0.0, 0.0, 0.0),
        (0.5, 0.75, 0.75, 0.75, 0.75),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 20.0, 5.0, 2.0),
        signal=signal(),
        target_soc=0.6,
        ready_step=1,
        profile=profile,
    )
    assert report.issues
    assert report.issues[0].code is ValidationCode.INITIAL_ENERGY
    assert ValidationCode.POWER_LIMIT in {issue.code for issue in report.issues}
    assert ValidationCode.BATTERY_RECURSION in {issue.code for issue in report.issues}
    assert report.is_valid is False


def test_invalid_session_returns_report_instead_of_reraising() -> None:
    profile = valid_profile()
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 20.0, 40.0, 0.0),
        signal=signal(),
        target_soc=0.6,
        ready_step=1,
        profile=profile,
    )
    assert not report.is_valid
    assert report.issues[0].code is ValidationCode.SESSION_INVALID


def test_ready_at_arrival_allows_no_charge_when_target_is_already_met() -> None:
    profile = ChargingProfile(
        0,
        (0.0, 0.0, 0.0, 0.0),
        (30.0, 30.0, 30.0, 30.0, 30.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.75, 0.75, 0.75, 0.75, 0.75),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(0, 4, 30.0, 5.0, 2.0),
        signal=signal(),
        target_soc=0.6,
        ready_step=0,
        profile=profile,
    )
    assert report.is_valid


def test_ready_step_departure_is_represented_by_terminal_state() -> None:
    profile = valid_profile()
    report = validate(profile, ready=4)
    assert report.is_valid
    assert report.ready_energy_kwh == pytest.approx(24.0)


def test_ready_step_before_arrival_is_reported() -> None:
    report = validate(valid_profile(), ready=-1)
    assert ValidationCode.READY_TIME_VIOLATION in {issue.code for issue in report.issues}


def test_ready_step_after_departure_is_reported() -> None:
    report = validate(valid_profile(), ready=5)
    assert ValidationCode.READY_TIME_VIOLATION in {issue.code for issue in report.issues}


def test_profile_start_after_session_is_reported() -> None:
    profile = ChargingProfile(
        1,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.6),
    )
    report = validate(profile)
    assert ValidationCode.PROFILE_ALIGNMENT in {issue.code for issue in report.issues}


def test_profile_start_before_session_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.6),
    )
    session = ChargingSession(1, 5, 20.0, 5.0, 2.0)
    report = validate_charging_profile(
        ev=ev(),
        session=session,
        signal=signal(5),
        target_soc=0.6,
        ready_step=2,
        profile=profile,
    )
    assert ValidationCode.PROFILE_ALIGNMENT in {issue.code for issue in report.issues}


def test_nonzero_global_arrival_profile_is_valid() -> None:
    profile = ChargingProfile(
        3,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.6),
    )
    report = validate_charging_profile(
        ev=ev(),
        session=ChargingSession(3, 7, 20.0, 5.0, 2.0),
        signal=signal(7),
        target_soc=0.6,
        ready_step=4,
        profile=profile,
    )
    assert report.is_valid


def test_terminal_recursion_failure_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 25.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.625),
    )
    report = validate(profile)
    assert any(issue.code is ValidationCode.BATTERY_RECURSION for issue in report.issues)


def test_intermediate_floor_and_terminal_capacity_are_checked() -> None:
    floor_profile = ChargingProfile(
        0,
        (0.0, 0.0, 0.0, 0.0),
        (20.0, 20.0, 3.5, 20.0, 20.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0875, 0.5, 0.5),
    )
    floor_report = validate(floor_profile, target=0.5)
    assert ValidationCode.BATTERY_BELOW_MINIMUM in {issue.code for issue in floor_report.issues}

    capacity_profile = ChargingProfile(
        0,
        (0.0, 0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0, 20.0, 41.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5, 1.0),
    )
    capacity_report = validate(capacity_profile, target=1.0)
    assert ValidationCode.BATTERY_ABOVE_CAPACITY in {issue.code for issue in capacity_report.issues}


def test_target_overcharge_is_reported() -> None:
    profile = ChargingProfile(
        0,
        (4.0, 0.0, 0.0, 0.0),
        (20.0, 24.0, 24.0, 24.0, 24.0),
        (4.0, 0.0, 0.0, 0.0),
        (0.5, 0.6, 0.6, 0.6, 0.6),
    )
    report = validate(profile, target=0.5)
    assert ValidationCode.TARGET_MISMATCH in {issue.code for issue in report.issues}
