"""Analytical charging-profile constructors for one EV session.

Commit 3 deliberately implements only two deterministic, solver-free profiles:

* immediate same-target charging in chronological order; and
* minimum-cost charging by ascending interval price before a requested ready step.

Every returned trajectory spans the complete exact session and is passed through
Commit 2's independent physical validator before it is returned.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .exceptions import PhysicalConstraintError, SchemaValidationError
from .feasibility import (
    evaluate_request_feasibility,
    required_grid_energy_kwh,
)
from .schemas import ChargingProfile, ChargingSession, EVSpec, PlanningSignal
from .validation import ValidationReport, ValidationTolerances, validate_charging_profile

# Absolute floor for "this grid energy is already delivered, not still pending".
# One nanowatt-hour: far below anything physically meaningful.
_ENERGY_RESIDUE_FLOOR_KWH = 1e-12

# Safety factor on the accumulation bound below. The bound counts only the
# subtractions in the allocation loop; the quotient in
# ``required_grid_energy_kwh`` and each interval's capacity are themselves
# rounded before the loop ever runs. A small constant multiple absorbs those
# extra sources without the bound ceasing to scale. 4 is a round number, not a
# derived one. It is NOT what covers the cancellation in
# ``max(B_0, z*B_max) - B_0`` (up to ~5e-14 kWh); the 1e-12 floor absorbs that.
_ENERGY_RESIDUE_SAFETY = 4.0


def _energy_residue_tolerance(required_energy: float, interval_count: int) -> float:
    """Largest grid-energy residue that counts as already delivered.

    ``required_grid_energy_kwh`` is a quotient, ``(max(B_0, z*B_max) - B_0)/eta``,
    and the allocation loop subtracts interval capacities from it one at a time.
    Each subtraction rounds to at most ``eps * |running total|``, so after ``n``
    steps the accumulated residue is bounded by roughly
    ``n * eps * required_energy``. A fixed absolute constant therefore cannot be
    correct for every grid: the residue grows with both the interval count and
    the energy, and a 1-minute grid takes 15x more subtractions over a 24-hour
    horizon than the 15-minute nominal default.

    Scaling with the bound is what makes this guard defensible -- the earlier
    fixed constant rested on a sampled worst-case residue that two independent
    searches did not agree on, and sampling cannot bound a quantity that grows
    with the grid. No measured residue is quoted here on purpose; the guard
    stands on the bound, not on a sample.

    What it returns stays physically meaningless: at least 1e-12 kWh, about
    6.4e-11 kWh for a 1439-interval 50 kWh request. Against the 1e-8 kWh
    battery-side scan in ``degradation`` that leaves 124x-185x of margin across
    the shipped catalogue, and still ~39x for a 200 kWh request on a 1-minute
    grid. The ratio falls as the request grows -- these are request energies,
    not pack sizes, and a large pack charged from empty asks for more than its
    own capacity -- but the tolerance would not reach 1e-8 until about
    7,800 kWh. So it opens no new refusals in that band.

    It does not follow that nothing can be refused between the two. An
    engineered residue above this tolerance but below ~1.1e-8 kWh clears the
    loop here and still trips the battery-side scan. No natural input is known
    to land there; the band is narrow, not absent.

    Deliberately NOT ``ValidationTolerances.energy_kwh``: that is a *validation*
    tolerance a caller may loosen, and routing it into the constructor would let
    a loosened check silently change what gets built.
    """
    if interval_count <= 0 or required_energy <= 0.0:
        return _ENERGY_RESIDUE_FLOOR_KWH
    accumulation_bound = (
        _ENERGY_RESIDUE_SAFETY * interval_count * sys.float_info.epsilon * required_energy
    )
    return max(_ENERGY_RESIDUE_FLOOR_KWH, accumulation_bound)



@dataclass(frozen=True, slots=True)
class ConstructedProfile:
    """A validated profile together with deterministic construction metadata.

    Instances are intended to be created by the public constructors below.
    Direct dataclass construction does not independently validate physical
    consistency.
    """

    profile: ChargingProfile
    target_soc: float
    ready_step: int
    required_grid_energy_kwh: float
    charging_cost: float
    validation: ValidationReport

    def __post_init__(self) -> None:
        if not self.validation.is_valid:
            raise PhysicalConstraintError("ConstructedProfile requires a valid trajectory.")


def _finite_target(target_soc: object) -> float:
    if isinstance(target_soc, bool) or not isinstance(target_soc, Real) or not isfinite(target_soc):
        raise SchemaValidationError("target_soc must be a finite real number.")
    value = float(target_soc)
    if not 0.0 <= value <= 1.0:
        raise PhysicalConstraintError("target_soc must lie in [0, 1].")
    return value


def _validate_common_inputs(
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    target_soc: object,
) -> float:
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec instance.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession instance.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal instance.")
    session.validate_for_ev(ev)
    signal.validate_session_window(session)
    return _finite_target(target_soc)


def _resolve_tolerances(
    tolerances: ValidationTolerances | None,
) -> ValidationTolerances:
    if tolerances is None:
        return ValidationTolerances()
    if not isinstance(tolerances, ValidationTolerances):
        raise SchemaValidationError("tolerances must be a ValidationTolerances instance.")
    return tolerances


def _verify_allocation(
    allocation: list[float],
    required_energy: float,
    tolerances: ValidationTolerances,
) -> None:
    allocated_energy = sum(allocation)
    if abs(allocated_energy - required_energy) > tolerances.energy_kwh:
        raise PhysicalConstraintError(
            "grid-energy allocation does not equal the required energy within tolerance."
        )


def _normalize_battery_energy(
    *,
    ev: EVSpec,
    battery_energy: list[float],
    tolerances: ValidationTolerances,
) -> tuple[float, ...]:
    """Normalize machine-scale battery-boundary noise before schema construction."""
    boundary_energy_tolerance = min(
        tolerances.energy_kwh,
        tolerances.soc * ev.battery_capacity_kwh,
    )
    normalized: list[float] = []
    for index, energy in enumerate(battery_energy):
        if energy < -boundary_energy_tolerance:
            raise PhysicalConstraintError(
                f"battery energy is materially below zero at state {index}."
            )
        if energy > ev.battery_capacity_kwh + boundary_energy_tolerance:
            raise PhysicalConstraintError(
                f"battery energy materially exceeds capacity at state {index}."
            )
        if energy < 0.0:
            normalized.append(0.0)
        elif energy > ev.battery_capacity_kwh:
            normalized.append(ev.battery_capacity_kwh)
        else:
            normalized.append(energy)
    return tuple(normalized)


def _build_profile_from_grid_energy(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    grid_energy_kwh: list[float],
    tolerances: ValidationTolerances,
) -> ChargingProfile:
    """Build exact battery and SOC states from a grid-energy allocation."""
    expected_intervals = session.departure_step - session.arrival_step
    if len(grid_energy_kwh) != expected_intervals:
        raise SchemaValidationError("grid-energy allocation must span the exact session.")

    power_kw = tuple(
        energy / signal.interval_durations[session.arrival_step + local_index]
        for local_index, energy in enumerate(grid_energy_kwh)
    )
    battery_energy = [session.initial_energy_kwh]
    for energy in grid_energy_kwh:
        battery_energy.append(battery_energy[-1] + ev.charging_efficiency * energy)

    # Normalize only machine-scale boundary noise before constructing the
    # immutable profile.  The helper derives SOC from these cleaned energies,
    # preserving exact energy/SOC consistency.
    cleaned_battery_energy = _normalize_battery_energy(
        ev=ev,
        battery_energy=battery_energy,
        tolerances=tolerances,
    )
    soc = tuple(energy / ev.battery_capacity_kwh for energy in cleaned_battery_energy)

    return ChargingProfile(
        start_step=session.arrival_step,
        grid_energy_kwh=tuple(grid_energy_kwh),
        battery_energy_kwh=cleaned_battery_energy,
        power_kw=power_kw,
        soc=soc,
    )


def _charging_cost(
    *,
    profile: ChargingProfile,
    signal: PlanningSignal,
) -> float:
    return sum(
        signal.price_per_kwh[profile.start_step + local_index] * grid_energy
        for local_index, grid_energy in enumerate(profile.grid_energy_kwh)
    )


def _validated_result(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    target_soc: float,
    ready_step: int,
    required_energy: float,
    profile: ChargingProfile,
    tolerances: ValidationTolerances | None,
) -> ConstructedProfile:
    report = validate_charging_profile(
        ev=ev,
        session=session,
        signal=signal,
        target_soc=target_soc,
        ready_step=ready_step,
        profile=profile,
        tolerances=tolerances,
    )
    if not report.is_valid:
        codes = ", ".join(issue.code.value for issue in report.issues)
        raise PhysicalConstraintError(f"constructed profile failed physical validation: {codes}")
    return ConstructedProfile(
        profile=profile,
        target_soc=target_soc,
        ready_step=ready_step,
        required_grid_energy_kwh=required_energy,
        charging_cost=_charging_cost(profile=profile, signal=signal),
        validation=report,
    )


def build_immediate_charging_profile(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    target_soc: float,
    tolerances: ValidationTolerances | None = None,
) -> ConstructedProfile:
    """Charge immediately at the maximum grid-side power until target is met.

    The returned ``ready_step`` is the earliest state boundary at which the
    target is reached. A partial final interval is used when required. If no
    charging is required, ``ready_step == arrival_step`` and the full-session
    trajectory contains zero charging power.
    """
    validation_tolerances = _resolve_tolerances(tolerances)
    target = _validate_common_inputs(ev, session, signal, target_soc)
    required_energy = required_grid_energy_kwh(ev, session, target)
    session_intervals = session.departure_step - session.arrival_step
    capacities = [
        ev.charger_power_kw * signal.interval_durations[session.arrival_step + index]
        for index in range(session_intervals)
    ]
    total_capacity = sum(capacities)
    if required_energy > total_capacity:
        raise PhysicalConstraintError("target cannot be reached before departure.")

    allocation = [0.0] * session_intervals
    remaining = required_energy
    last_used_local: int | None = None
    completion_tolerance = _energy_residue_tolerance(required_energy, session_intervals)
    for local_index in range(session_intervals):
        if remaining <= completion_tolerance:
            break
        energy = min(capacities[local_index], remaining)
        allocation[local_index] = energy
        remaining -= energy
        last_used_local = local_index
        if remaining <= completion_tolerance:
            remaining = 0.0
            break

    _verify_allocation(allocation, required_energy, validation_tolerances)

    ready_step = (
        session.arrival_step
        if last_used_local is None
        else session.arrival_step + last_used_local + 1
    )
    profile = _build_profile_from_grid_energy(
        ev=ev,
        session=session,
        signal=signal,
        grid_energy_kwh=allocation,
        tolerances=validation_tolerances,
    )
    return _validated_result(
        ev=ev,
        session=session,
        signal=signal,
        target_soc=target,
        ready_step=ready_step,
        required_energy=required_energy,
        profile=profile,
        tolerances=validation_tolerances,
    )


def build_minimum_cost_charging_profile(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    target_soc: float,
    ready_step: int,
    tolerances: ValidationTolerances | None = None,
) -> ConstructedProfile:
    """Construct the exact minimum-cost profile before ``ready_step``.

    Since charging efficiency and the charger limit are constant, the linear
    cost problem is solved exactly by filling eligible intervals in ascending
    price order. Each interval's energy capacity is its own charger-power
    limit times its duration. Price ties are resolved by earlier global time
    step, making the constructor deterministic. A partial final interval is
    supported.
    """
    validation_tolerances = _resolve_tolerances(tolerances)
    target = _validate_common_inputs(ev, session, signal, target_soc)
    feasibility = evaluate_request_feasibility(
        ev,
        session,
        signal,
        target_soc=target,
        ready_step=ready_step,
        tolerance=0.0,
    )
    if not feasibility.is_feasible:
        raise PhysicalConstraintError(
            "target is infeasible by ready_step: "
            f"grid-energy shortfall {-feasibility.energy_margin_kwh:.12g} kWh."
        )

    session_intervals = session.departure_step - session.arrival_step
    allocation = [0.0] * session_intervals
    remaining = feasibility.required_grid_energy_kwh

    eligible_steps = list(range(session.arrival_step, ready_step))
    eligible_steps.sort(key=lambda step: (signal.price_per_kwh[step], step))
    for global_step in eligible_steps:
        if remaining <= 0.0:
            break
        interval_capacity = ev.charger_power_kw * signal.interval_durations[global_step]
        energy = min(interval_capacity, remaining)
        allocation[global_step - session.arrival_step] = energy
        remaining -= energy
        if remaining <= 0.0:
            remaining = 0.0
            break

    _verify_allocation(allocation, feasibility.required_grid_energy_kwh, validation_tolerances)

    profile = _build_profile_from_grid_energy(
        ev=ev,
        session=session,
        signal=signal,
        grid_energy_kwh=allocation,
        tolerances=validation_tolerances,
    )
    return _validated_result(
        ev=ev,
        session=session,
        signal=signal,
        target_soc=target,
        ready_step=ready_step,
        required_energy=feasibility.required_grid_energy_kwh,
        profile=profile,
        tolerances=validation_tolerances,
    )
