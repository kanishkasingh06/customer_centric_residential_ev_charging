from __future__ import annotations

import math

import pytest

import evmenu.profiles as profiles_module
from evmenu import (
    ChargingProfile,
    ChargingSession,
    EVSpec,
    PhysicalConstraintError,
    PlanningSignal,
    SchemaValidationError,
    ValidationCode,
    ValidationTolerances,
    build_immediate_charging_profile,
    build_minimum_cost_charging_profile,
    validate_charging_profile,
)


def ev(*, efficiency: float = 1.0, power: float = 4.0) -> EVSpec:
    return EVSpec("EV-1", 40.0, 4.0, power, efficiency, "LFP")


def session(*, initial: float = 20.0, arrival: int = 0, departure: int = 4) -> ChargingSession:
    return ChargingSession(arrival, departure, initial, 4.0, 2.0)


def signal(prices: tuple[float, ...], *, dt: float = 1.0) -> PlanningSignal:
    return PlanningSignal(dt, prices)


def test_immediate_exact_one_interval() -> None:
    result = build_immediate_charging_profile(
        ev=ev(), session=session(), signal=signal((5.0, 4.0, 3.0, 2.0)), target_soc=0.60
    )
    assert result.profile.grid_energy_kwh == (4.0, 0.0, 0.0, 0.0)
    assert result.profile.power_kw == (4.0, 0.0, 0.0, 0.0)
    assert result.profile.battery_energy_kwh == (20.0, 24.0, 24.0, 24.0, 24.0)
    assert result.ready_step == 1
    assert result.validation.is_valid
    assert result.charging_cost == 20.0


def test_machine_scale_upper_boundary_is_normalized_consistently() -> None:
    result = profiles_module._build_profile_from_grid_energy(
        ev=ev(power=20.0),
        session=ChargingSession(0, 1, 20.0, 0.0, 0.0),
        signal=signal((1.0,)),
        grid_energy_kwh=[20.000000000000004],
        tolerances=ValidationTolerances(),
    )

    assert result.battery_energy_kwh == (20.0, 40.0)
    assert result.soc == (0.5, 1.0)
    assert all(
        abs(energy - soc * 40.0) <= ValidationTolerances().energy_kwh
        for energy, soc in zip(result.battery_energy_kwh, result.soc, strict=True)
    )
    assert (
        abs(result.battery_energy_kwh[1] - (result.battery_energy_kwh[0] + 20.000000000000004))
        <= ValidationTolerances().energy_kwh
    )


def test_machine_scale_lower_boundary_is_normalized() -> None:
    normalized = profiles_module._normalize_battery_energy(
        ev=ev(),
        battery_energy=[-1e-16, 20.0],
        tolerances=ValidationTolerances(),
    )

    assert normalized == (0.0, 20.0)


@pytest.mark.parametrize("energy", [40.0 + 40.0 * 1e-5, -40.0 * 1e-5])
def test_material_battery_boundary_violation_is_rejected(energy: float) -> None:
    with pytest.raises(PhysicalConstraintError):
        profiles_module._normalize_battery_energy(
            ev=ev(),
            battery_energy=[energy],
            tolerances=ValidationTolerances(),
        )


def test_immediate_applies_efficiency_once() -> None:
    result = build_immediate_charging_profile(
        ev=ev(efficiency=0.8, power=5.0),
        session=session(),
        signal=signal((1.0, 1.0, 1.0, 1.0)),
        target_soc=0.60,
    )
    assert result.required_grid_energy_kwh == pytest.approx(5.0)
    assert result.profile.grid_energy_kwh[0] == pytest.approx(5.0)
    assert result.profile.battery_energy_kwh[1] == pytest.approx(24.0)


def test_immediate_uses_partial_final_interval() -> None:
    result = build_immediate_charging_profile(
        ev=ev(power=3.0),
        session=session(),
        signal=signal((1.0, 1.0, 1.0, 1.0)),
        target_soc=0.625,
    )
    assert result.profile.grid_energy_kwh == pytest.approx((3.0, 2.0, 0.0, 0.0))
    assert result.profile.power_kw == pytest.approx((3.0, 2.0, 0.0, 0.0))
    assert result.ready_step == 2


def test_immediate_no_charge_profile_is_full_session() -> None:
    result = build_immediate_charging_profile(
        ev=ev(),
        session=session(initial=28.0),
        signal=signal((1.0, 2.0, 3.0, 4.0)),
        target_soc=0.60,
    )
    assert result.ready_step == 0
    assert result.required_grid_energy_kwh == 0.0
    assert result.profile.power_kw == (0.0, 0.0, 0.0, 0.0)
    assert result.profile.battery_energy_kwh == (28.0, 28.0, 28.0, 28.0, 28.0)
    assert result.validation.is_valid


def test_immediate_rejects_target_unreachable_by_departure() -> None:
    with pytest.raises(PhysicalConstraintError, match="cannot be reached"):
        build_immediate_charging_profile(
            ev=ev(power=2.0),
            session=session(initial=4.0, departure=2),
            signal=signal((1.0, 1.0)),
            target_soc=1.0,
        )


def test_minimum_cost_uses_cheapest_slots() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=2.0),
        session=session(),
        signal=signal((8.0, 2.0, 5.0, 1.0)),
        target_soc=0.60,
        ready_step=3,
    )
    assert result.profile.grid_energy_kwh == pytest.approx((0.0, 2.0, 2.0, 0.0))
    assert result.charging_cost == pytest.approx(14.0)
    assert result.validation.is_valid


def test_minimum_cost_uses_partial_final_cheapest_allocation() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=3.0),
        session=session(),
        signal=signal((5.0, 1.0, 2.0, 9.0)),
        target_soc=0.625,
        ready_step=3,
    )
    assert result.profile.grid_energy_kwh == pytest.approx((0.0, 3.0, 2.0, 0.0))
    assert result.charging_cost == pytest.approx(7.0)


def test_minimum_cost_tie_breaks_by_earlier_step() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=2.0),
        session=session(),
        signal=signal((4.0, 1.0, 1.0, 9.0)),
        target_soc=0.55,
        ready_step=3,
    )
    assert result.profile.grid_energy_kwh == pytest.approx((0.0, 2.0, 0.0, 0.0))


def test_minimum_cost_supports_negative_prices() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=2.0),
        session=session(),
        signal=signal((3.0, -2.0, -1.0, 5.0)),
        target_soc=0.60,
        ready_step=3,
    )
    assert result.profile.grid_energy_kwh == pytest.approx((0.0, 2.0, 2.0, 0.0))
    assert result.charging_cost == pytest.approx(-6.0)


def test_minimum_cost_never_charges_at_or_after_ready_step() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=4.0),
        session=session(),
        signal=signal((10.0, 9.0, -100.0, -200.0)),
        target_soc=0.60,
        ready_step=2,
    )
    assert result.profile.grid_energy_kwh == pytest.approx((0.0, 4.0, 0.0, 0.0))


def test_minimum_cost_zero_energy_is_feasible_at_arrival() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(),
        session=session(initial=28.0),
        signal=signal((1.0, 2.0, 3.0, 4.0)),
        target_soc=0.60,
        ready_step=0,
    )
    assert result.profile.power_kw == (0.0, 0.0, 0.0, 0.0)
    assert result.validation.is_valid


def test_minimum_cost_rejects_infeasible_ready_step() -> None:
    with pytest.raises(PhysicalConstraintError, match="shortfall"):
        build_minimum_cost_charging_profile(
            ev=ev(power=2.0),
            session=session(),
            signal=signal((1.0, 1.0, 1.0, 1.0)),
            target_soc=0.70,
            ready_step=1,
        )


def test_profile_start_step_preserves_nonzero_arrival() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(),
        session=session(arrival=2, departure=6),
        signal=signal((100.0, 100.0, 3.0, 1.0, 2.0, 4.0)),
        target_soc=0.60,
        ready_step=5,
    )
    assert result.profile.start_step == 2
    assert result.profile.grid_energy_kwh == pytest.approx((0.0, 4.0, 0.0, 0.0))


def test_minimum_cost_is_no_more_expensive_than_immediate_for_same_target_and_window() -> None:
    prices = (9.0, 2.0, 5.0, 1.0)
    immediate = build_immediate_charging_profile(
        ev=ev(power=2.0), session=session(), signal=signal(prices), target_soc=0.60
    )
    minimum = build_minimum_cost_charging_profile(
        ev=ev(power=2.0),
        session=session(),
        signal=signal(prices),
        target_soc=0.60,
        ready_step=session().departure_step,
    )
    assert minimum.charging_cost <= immediate.charging_cost


@pytest.mark.parametrize("target", [math.nan, math.inf, -math.inf, True, "0.8"])
def test_constructors_reject_invalid_target(target: object) -> None:
    with pytest.raises((SchemaValidationError, PhysicalConstraintError)):
        build_immediate_charging_profile(
            ev=ev(),
            session=session(),
            signal=signal((1.0, 1.0, 1.0, 1.0)),
            target_soc=target,  # type: ignore[arg-type]
        )


def test_result_metadata_matches_profile() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=4.0),
        session=session(),
        signal=signal((4.0, 3.0, 2.0, 1.0)),
        target_soc=0.60,
        ready_step=4,
    )
    recomputed_cost = sum(
        p * e for p, e in zip((4.0, 3.0, 2.0, 1.0), result.profile.grid_energy_kwh)
    )
    assert result.charging_cost == pytest.approx(recomputed_cost)
    assert sum(result.profile.grid_energy_kwh) == pytest.approx(result.required_grid_energy_kwh)


def test_tiny_infeasible_target_is_rejected_by_both_constructors() -> None:
    target = (36.0 + 5e-11) / 40.0
    vehicle = ev(power=4.0)
    overnight = session(initial=20.0, departure=4)
    planning_signal = signal((1.0, 1.0, 1.0, 1.0))

    with pytest.raises(PhysicalConstraintError):
        build_immediate_charging_profile(
            ev=vehicle,
            session=overnight,
            signal=planning_signal,
            target_soc=target,
        )
    with pytest.raises(PhysicalConstraintError):
        build_minimum_cost_charging_profile(
            ev=vehicle,
            session=overnight,
            signal=planning_signal,
            target_soc=target,
            ready_step=4,
        )


def test_decimal_fraction_capacity_remains_numerically_feasible() -> None:
    vehicle = ev(power=7.2)
    planning_signal = signal((1.0, 1.0, 1.0, 1.0), dt=0.25)
    result = build_immediate_charging_profile(
        ev=vehicle,
        session=session(),
        signal=planning_signal,
        target_soc=0.625,
        tolerances=ValidationTolerances(energy_kwh=1e-9),
    )

    assert result.profile.grid_energy_kwh == pytest.approx((1.8, 1.8, 1.4, 0.0))
    assert sum(result.profile.grid_energy_kwh) == pytest.approx(
        result.required_grid_energy_kwh, abs=1e-9
    )
    assert result.validation.is_valid


def test_exact_current_and_below_current_targets_need_no_charge() -> None:
    planning_signal = signal((1.0, 1.0, 1.0, 1.0))
    exact = build_immediate_charging_profile(
        ev=ev(), session=session(), signal=planning_signal, target_soc=0.5
    )
    below = build_immediate_charging_profile(
        ev=ev(), session=session(), signal=planning_signal, target_soc=0.4
    )

    assert exact.ready_step == 0
    assert below.ready_step == 0
    assert exact.profile.grid_energy_kwh == (0.0, 0.0, 0.0, 0.0)
    assert below.profile.grid_energy_kwh == (0.0, 0.0, 0.0, 0.0)


def test_exact_full_target_and_ready_at_departure() -> None:
    vehicle = ev(power=4.0)
    full_session = session(departure=6)
    planning_signal = signal((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    immediate = build_immediate_charging_profile(
        ev=vehicle, session=full_session, signal=planning_signal, target_soc=1.0
    )
    minimum = build_minimum_cost_charging_profile(
        ev=vehicle,
        session=full_session,
        signal=planning_signal,
        target_soc=0.6,
        ready_step=full_session.departure_step,
    )

    assert immediate.profile.battery_energy_kwh[-1] == pytest.approx(40.0)
    assert immediate.validation.is_valid
    assert minimum.ready_step == full_session.departure_step
    assert minimum.validation.is_valid


def test_negative_price_partial_request_does_not_overcharge() -> None:
    result = build_minimum_cost_charging_profile(
        ev=ev(power=4.0),
        session=ChargingSession(0, 3, 20.0, 0.0, 0.0),
        signal=signal((-5.0, -4.0, 10.0)),
        target_soc=0.55,
        ready_step=3,
    )

    assert result.profile.grid_energy_kwh == pytest.approx((2.0, 0.0, 0.0))
    assert result.charging_cost == pytest.approx(-10.0)
    assert result.profile.battery_energy_kwh[-1] == pytest.approx(22.0)
    assert result.validation.is_valid


def test_overnight_profile_uses_global_price_indices() -> None:
    prices = [100.0] * 32
    prices[18] = 8.0
    prices[19] = 2.0
    overnight = ChargingSession(18, 32, 20.0, 0.0, 0.0)
    result = build_minimum_cost_charging_profile(
        ev=ev(power=4.0),
        session=overnight,
        signal=signal(tuple(prices)),
        target_soc=0.6,
        ready_step=20,
    )

    assert result.profile.start_step == 18
    assert len(result.profile.power_kw) == 14
    assert len(result.profile.battery_energy_kwh) == 15
    assert result.profile.grid_energy_kwh[1] == pytest.approx(4.0)
    assert result.charging_cost == pytest.approx(8.0)
    assert result.validation.is_valid


def test_constructors_do_not_mutate_inputs_or_tolerances() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal((4.0, 3.0, 2.0, 1.0))
    tolerances = ValidationTolerances()
    before = (vehicle, charging_session, planning_signal, tolerances)

    build_immediate_charging_profile(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        target_soc=0.6,
        tolerances=tolerances,
    )
    build_minimum_cost_charging_profile(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        target_soc=0.6,
        ready_step=4,
        tolerances=tolerances,
    )

    assert (vehicle, charging_session, planning_signal, tolerances) == before


def test_independent_validator_catches_corrupted_constructed_profile() -> None:
    result = build_immediate_charging_profile(
        ev=ev(),
        session=session(),
        signal=signal((1.0, 1.0, 1.0, 1.0)),
        target_soc=0.6,
    )
    battery_energy = list(result.profile.battery_energy_kwh)
    battery_energy[1] += 1.0
    corrupted = ChargingProfile(
        start_step=result.profile.start_step,
        grid_energy_kwh=result.profile.grid_energy_kwh,
        battery_energy_kwh=tuple(battery_energy),
        power_kw=result.profile.power_kw,
        soc=result.profile.soc,
    )
    report = validate_charging_profile(
        ev=ev(),
        session=session(),
        signal=signal((1.0, 1.0, 1.0, 1.0)),
        target_soc=0.6,
        ready_step=result.ready_step,
        profile=corrupted,
    )

    assert ValidationCode.BATTERY_RECURSION in {issue.code for issue in report.issues}
    assert not report.is_valid
