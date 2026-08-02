"""Convex degradation-aware charging profiles and anchored saving frontiers.

Commit 6 solves one-session, fixed-request trajectory problems.  It does not
compact, Pareto-filter, cap, or run customer choice over frontier points.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from math import exp, isclose, isfinite
from numbers import Real
from typing import Literal, cast

from scipy.optimize import minimize  # type: ignore[import-untyped]

from .degradation import (
    DegradationAssessment,
    DegradationSettings,
    assess_candidate_degradation,
    calendar_soc_stress,
)
from .exceptions import PhysicalConstraintError, SchemaValidationError
from .feasibility import evaluate_request_feasibility, required_grid_energy_kwh
from .menu import MenuCandidate
from .profiles import ConstructedProfile
from .schemas import ChargingProfile, ChargingSession, EVSpec, MenuSettings, PlanningSignal
from .validation import ValidationReport, ValidationTolerances, validate_charging_profile

_HOURS_PER_YEAR = 8760.0
_GAS_CONSTANT = 8.314462618
_DEFAULT_OBJECTIVE_SCALE = 1_000_000.0
EndpointRole = Literal[
    "least_degradation",
    "intermediate",
    "maximum_saving",
    "least_and_maximum",
]


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)


def _canonical_decimal(value: float) -> str:
    try:
        decimal = Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError) as exc:
        raise SchemaValidationError("requested saving cannot be represented canonically.") from exc
    text = format(decimal, "f")
    return "0" if text in ("-0", "0.0") else text


@dataclass(frozen=True, slots=True)
class FrontierSettings:
    """Numerical controls for fixed-request convex frontier construction."""

    plating_guard_weight: float = 1e-6
    saving_band: float = 1e-6
    maximum_levels: int = 5
    solver_tolerance: float = 1e-9
    maximum_iterations: int = 1000
    objective_scale: float = _DEFAULT_OBJECTIVE_SCALE
    bound_tolerance: float = 1e-8
    energy_tolerance: float = 1e-8
    saving_tolerance: float = 1e-8
    objective_tolerance: float = 1e-8
    frontier_gap_tolerance: float = 0.0
    cost_tolerance: float = 1e-8
    solver_ftol: float | None = None
    solver_max_iterations: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("plating_guard_weight", self.plating_guard_weight),
            ("saving_band", self.saving_band),
            ("solver_tolerance", self.solver_tolerance),
            ("objective_scale", self.objective_scale),
            ("bound_tolerance", self.bound_tolerance),
            ("energy_tolerance", self.energy_tolerance),
            ("saving_tolerance", self.saving_tolerance),
            ("objective_tolerance", self.objective_tolerance),
            ("frontier_gap_tolerance", self.frontier_gap_tolerance),
            ("cost_tolerance", self.cost_tolerance),
        ):
            _finite(name, value)
        if self.plating_guard_weight < 0.0:
            raise PhysicalConstraintError("plating_guard_weight must be non-negative.")
        if self.saving_band < 0.0:
            raise PhysicalConstraintError("saving_band must be non-negative.")
        if self.solver_tolerance <= 0.0:
            raise PhysicalConstraintError("solver_tolerance must be positive.")
        if self.objective_scale <= 0.0:
            raise PhysicalConstraintError("objective_scale must be positive.")
        for name, value in (
            ("bound_tolerance", self.bound_tolerance),
            ("energy_tolerance", self.energy_tolerance),
            ("saving_tolerance", self.saving_tolerance),
            ("objective_tolerance", self.objective_tolerance),
            ("frontier_gap_tolerance", self.frontier_gap_tolerance),
            ("cost_tolerance", self.cost_tolerance),
        ):
            if value < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")
        if isinstance(self.maximum_levels, bool) or not isinstance(self.maximum_levels, int):
            raise SchemaValidationError("maximum_levels must be an integer.")
        if self.maximum_levels < 2:
            raise PhysicalConstraintError("maximum_levels must be at least two.")
        if isinstance(self.maximum_iterations, bool) or not isinstance(
            self.maximum_iterations, int
        ):
            raise SchemaValidationError("maximum_iterations must be an integer.")
        if self.maximum_iterations <= 0:
            raise PhysicalConstraintError("maximum_iterations must be positive.")
        if self.solver_ftol is not None:
            _finite("solver_ftol", self.solver_ftol)
            if self.solver_ftol <= 0.0:
                raise PhysicalConstraintError("solver_ftol must be positive.")
        if self.solver_max_iterations is not None:
            if isinstance(self.solver_max_iterations, bool) or not isinstance(
                self.solver_max_iterations, int
            ):
                raise SchemaValidationError("solver_max_iterations must be an integer.")
            if self.solver_max_iterations <= 0:
                raise PhysicalConstraintError("solver_max_iterations must be positive.")

    @property
    def effective_solver_ftol(self) -> float:
        return self.solver_tolerance if self.solver_ftol is None else self.solver_ftol

    @property
    def effective_solver_max_iterations(self) -> int:
        return (
            self.maximum_iterations
            if self.solver_max_iterations is None
            else self.solver_max_iterations
        )


@dataclass(frozen=True, slots=True)
class OptimizedProfile:
    """One validated fixed-request trajectory and its health objective."""

    constructed: ConstructedProfile
    assessment: DegradationAssessment
    saving: float
    trajectory_objective: float
    requested_saving: float | None
    point_id: str
    source_candidate_id: str
    ev_id: str
    endpoint_role: EndpointRole

    def __post_init__(self) -> None:
        if not isinstance(self.constructed, ConstructedProfile):
            raise SchemaValidationError("constructed must be a ConstructedProfile.")
        if not isinstance(self.assessment, DegradationAssessment):
            raise SchemaValidationError("assessment must be a DegradationAssessment.")
        for name, value in (
            ("point_id", self.point_id),
            ("source_candidate_id", self.source_candidate_id),
            ("ev_id", self.ev_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if self.endpoint_role not in (
            "least_degradation",
            "intermediate",
            "maximum_saving",
            "least_and_maximum",
        ):
            raise SchemaValidationError("endpoint_role is unsupported.")
        if self.assessment.candidate_id != self.point_id:
            raise SchemaValidationError("assessment candidate_id must equal point_id.")
        if self.assessment.ev_id != self.ev_id:
            raise SchemaValidationError("assessment ev_id must equal ev_id.")
        if not self.constructed.validation.is_valid:
            raise PhysicalConstraintError("optimized profile requires valid physical validation.")
        if not 0.0 <= self.constructed.target_soc <= 1.0:
            raise PhysicalConstraintError("optimized target SOC must lie in [0, 1].")
        if isinstance(self.constructed.ready_step, bool) or not isinstance(
            self.constructed.ready_step, int
        ):
            raise SchemaValidationError("optimized ready_step must be an integer.")
        _finite("saving", self.saving)
        objective = _finite("trajectory_objective", self.trajectory_objective)
        if objective < 0.0:
            raise PhysicalConstraintError("trajectory_objective must be non-negative.")
        if self.requested_saving is not None:
            _finite("requested_saving", self.requested_saving)

    @property
    def objective_value(self) -> float:
        """Return the original unscaled physical trajectory objective."""
        return self.trajectory_objective


@dataclass(frozen=True, slots=True)
class SavingFrontier:
    """Ordered anchored saving levels for one fixed ready-step/target request."""

    ev_id: str
    target_soc: float
    ready_step: int
    bau_cost: float
    points: tuple[OptimizedProfile, ...]
    source_candidate_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.ev_id, str) or not self.ev_id.strip():
            raise SchemaValidationError("ev_id must be a non-empty string.")
        object.__setattr__(self, "ev_id", self.ev_id.strip())
        target = _finite("target_soc", self.target_soc)
        if not 0.0 <= target <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        if isinstance(self.ready_step, bool) or not isinstance(self.ready_step, int):
            raise SchemaValidationError("ready_step must be an integer.")
        if self.ready_step < 0:
            raise PhysicalConstraintError("ready_step must be non-negative.")
        _finite("bau_cost", self.bau_cost)
        if not isinstance(self.source_candidate_id, str) or not self.source_candidate_id.strip():
            raise SchemaValidationError("source_candidate_id must be a non-empty string.")
        object.__setattr__(self, "source_candidate_id", self.source_candidate_id.strip())
        try:
            points = tuple(self.points)
        except TypeError as exc:
            raise SchemaValidationError(
                "points must be an iterable of OptimizedProfile objects."
            ) from exc
        object.__setattr__(self, "points", points)
        if not points or any(not isinstance(point, OptimizedProfile) for point in points):
            raise SchemaValidationError(
                "points must be a nonempty tuple of OptimizedProfile objects."
            )
        if any(point.ev_id != self.ev_id for point in points):
            raise SchemaValidationError("all points must belong to ev_id.")
        if any(point.source_candidate_id != self.source_candidate_id for point in points):
            raise SchemaValidationError("all points must share source_candidate_id.")
        if any(
            point.constructed.target_soc != target
            or point.constructed.ready_step != self.ready_step
            for point in points
        ):
            raise SchemaValidationError("all points must share target SOC and ready_step.")
        point_ids = tuple(point.point_id for point in points)
        if len(set(point_ids)) != len(point_ids):
            raise SchemaValidationError("frontier point IDs must be unique.")
        assessment_ids = tuple(point.assessment.candidate_id for point in points)
        if len(set(assessment_ids)) != len(assessment_ids):
            raise SchemaValidationError("frontier assessment IDs must be unique.")
        for left, right in pairwise(points):
            if right.saving < left.saving:
                raise PhysicalConstraintError("frontier savings must be nondecreasing.")
            if (
                right.saving == left.saving
                and right.constructed.profile == left.constructed.profile
                and right.constructed.charging_cost == left.constructed.charging_cost
                and right.trajectory_objective == left.trajectory_objective
            ):
                raise PhysicalConstraintError("frontier cannot contain exact duplicate points.")
        roles = {point.endpoint_role for point in points}
        if "least_and_maximum" not in roles and not {
            "least_degradation",
            "maximum_saving",
        }.issubset(roles):
            raise SchemaValidationError("frontier must preserve both endpoint roles.")


def _validate_inputs(
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    bau_cost: float,
    degradation_settings: DegradationSettings,
    frontier_settings: FrontierSettings,
    tolerances: ValidationTolerances,
) -> float:
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec instance.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession instance.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal instance.")
    if not isinstance(candidate, MenuCandidate):
        raise SchemaValidationError("candidate must be a MenuCandidate instance.")
    if not isinstance(degradation_settings, DegradationSettings):
        raise SchemaValidationError("degradation_settings must be a DegradationSettings instance.")
    if not isinstance(frontier_settings, FrontierSettings):
        raise SchemaValidationError("frontier_settings must be a FrontierSettings instance.")
    if not isinstance(tolerances, ValidationTolerances):
        raise SchemaValidationError("tolerances must be a ValidationTolerances instance.")
    if candidate.kind != "minimum_cost":
        raise SchemaValidationError("Commit 6 optimization requires a minimum_cost candidate.")
    numeric_bau_cost = _finite("bau_cost", bau_cost)
    session.validate_for_ev(ev)
    signal.validate_session_window(session)
    if candidate.ev_id != ev.ev_id:
        raise SchemaValidationError("candidate does not belong to ev.")
    feasibility = evaluate_request_feasibility(
        ev,
        session,
        signal,
        target_soc=candidate.target_soc,
        ready_step=candidate.ready_step,
        tolerance=0.0,
    )
    if not feasibility.is_feasible:
        raise PhysicalConstraintError("candidate request is infeasible in the supplied context.")
    required = required_grid_energy_kwh(ev, session, candidate.target_soc)
    report = validate_charging_profile(
        ev=ev,
        session=session,
        signal=signal,
        target_soc=candidate.target_soc,
        ready_step=candidate.ready_step,
        profile=candidate.profile,
        tolerances=tolerances,
    )
    if not report.is_valid:
        raise PhysicalConstraintError("candidate profile failed independent physical validation.")
    direct_cost = _cost(candidate.profile, signal)
    if not isclose(
        candidate.charging_cost,
        direct_cost,
        rel_tol=0.0,
        abs_tol=frontier_settings.cost_tolerance,
    ):
        raise SchemaValidationError("candidate charging_cost does not match its profile cost.")
    if not isclose(
        numeric_bau_cost,
        candidate.same_target_bau_cost,
        rel_tol=0.0,
        abs_tol=frontier_settings.cost_tolerance,
    ):
        raise SchemaValidationError("bau_cost does not match candidate.same_target_bau_cost.")
    if not isclose(
        candidate.saving,
        candidate.same_target_bau_cost - candidate.charging_cost,
        rel_tol=0.0,
        abs_tol=frontier_settings.cost_tolerance,
    ):
        raise SchemaValidationError("candidate saving metadata is inconsistent.")
    if not isclose(
        candidate.required_grid_energy_kwh,
        required,
        rel_tol=0.0,
        abs_tol=frontier_settings.energy_tolerance,
    ):
        raise SchemaValidationError("candidate required energy is inconsistent with its request.")
    return numeric_bau_cost


def _temperature_factor(temperature_c: float, reference_c: float, activation: float) -> float:
    temperature = _finite("temperature_c", temperature_c) + 273.15
    reference = _finite("reference_temperature_c", reference_c) + 273.15
    if temperature <= 0.0 or reference <= 0.0:
        raise PhysicalConstraintError("Kelvin temperatures must be positive.")
    exponent = -activation / _GAS_CONSTANT * (1.0 / temperature - 1.0 / reference)
    if not -745.0 <= exponent <= 709.0:
        raise PhysicalConstraintError("Arrhenius exponent lies outside the supported range.")
    return exp(exponent)


def _relative_age_factor(settings: DegradationSettings, alpha: float) -> float:
    numerator = alpha * settings.battery_age_years ** (alpha - 1.0)
    denominator = alpha * settings.reference_age_years ** (alpha - 1.0)
    factor = float(numerator / denominator)
    if not isfinite(factor):
        raise PhysicalConstraintError("relative age factor must be finite.")
    return factor


def _profile_from_energy(
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    energy: list[float],
    *,
    energy_tolerance: float = 0.0,
) -> ChargingProfile:
    power = tuple(
        value / signal.interval_durations[session.arrival_step + index]
        for index, value in enumerate(energy)
    )
    battery = [session.initial_energy_kwh]
    for value in energy:
        next_energy = battery[-1] + ev.charging_efficiency * value
        if abs(next_energy - ev.battery_capacity_kwh) <= energy_tolerance:
            next_energy = ev.battery_capacity_kwh
        elif abs(next_energy - ev.minimum_energy_kwh) <= energy_tolerance:
            next_energy = ev.minimum_energy_kwh
        battery.append(next_energy)
    return ChargingProfile(
        start_step=session.arrival_step,
        grid_energy_kwh=tuple(energy),
        power_kw=power,
        battery_energy_kwh=tuple(battery),
        soc=tuple(value / ev.battery_capacity_kwh for value in battery),
    )


def _cost(profile: ChargingProfile, signal: PlanningSignal) -> float:
    return sum(
        signal.price_per_kwh[profile.start_step + index] * value
        for index, value in enumerate(profile.grid_energy_kwh)
    )


def _allocate_by_price(
    prices: tuple[float, ...], capacities: tuple[float, ...], required: float, *, ascending: bool
) -> list[float]:
    if len(prices) != len(capacities):
        raise SchemaValidationError("prices and interval capacities must have equal lengths.")
    allocation = [0.0] * len(prices)
    remaining = required
    key = (
        (lambda index: (prices[index], index))
        if ascending
        else (lambda index: (-prices[index], index))
    )
    indices = sorted(range(len(prices)), key=key)
    for index in indices:
        amount = min(capacities[index], remaining)
        allocation[index] = amount
        remaining -= amount
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise PhysicalConstraintError("required energy exceeds eligible interval capacity.")
    return allocation


def _even_allocation(required: float, capacities: tuple[float, ...]) -> list[float]:
    count = len(capacities)
    if required == 0.0:
        return [0.0] * count
    allocation: list[float] = []
    remaining = required
    for index in range(count):
        slots = count - index
        remaining_capacity = sum(capacities[index:])
        if remaining_capacity <= 0.0:
            raise PhysicalConstraintError("even initial allocation has no remaining capacity.")
        future_capacity = sum(capacities[index + 1 :])
        minimum_now = max(0.0, remaining - future_capacity)
        value = min(capacities[index], max(minimum_now, remaining / slots))
        allocation.append(value)
        remaining -= value
    if abs(remaining) > 1e-8:
        raise PhysicalConstraintError("even initial allocation failed energy conservation.")
    return allocation


def _cost_range(
    *, prices: tuple[float, ...], capacities: tuple[float, ...], required: float
) -> tuple[list[float], list[float], float, float]:
    minimum = _allocate_by_price(prices, capacities, required, ascending=True)
    maximum = _allocate_by_price(prices, capacities, required, ascending=False)
    minimum_cost = sum(price * energy for price, energy in zip(prices, minimum, strict=True))
    maximum_cost = sum(price * energy for price, energy in zip(prices, maximum, strict=True))
    return minimum, maximum, minimum_cost, maximum_cost


def _saving_interval(
    *,
    bau_cost: float,
    requested_saving: float,
    saving_band: float,
    minimum_saving: float,
    maximum_saving: float,
) -> tuple[float, float]:
    requested_low = requested_saving - saving_band
    requested_high = requested_saving + saving_band
    if requested_high < minimum_saving or requested_low > maximum_saving:
        raise PhysicalConstraintError(
            "requested saving is unattainable: "
            f"requested={requested_saving}, band={saving_band}, "
            f"attainable=[{minimum_saving}, {maximum_saving}]."
        )
    return requested_low, requested_high


def _saving_initial_allocation(
    *,
    prices: tuple[float, ...],
    capacities: tuple[float, ...],
    required: float,
    bau_cost: float,
    requested_saving: float,
    settings: FrontierSettings,
) -> list[float]:
    minimum, maximum, minimum_cost, maximum_cost = _cost_range(
        prices=prices, capacities=capacities, required=required
    )
    minimum_saving = bau_cost - maximum_cost
    maximum_saving = bau_cost - minimum_cost
    saving_low, saving_high = _saving_interval(
        bau_cost=bau_cost,
        requested_saving=requested_saving,
        saving_band=settings.saving_band,
        minimum_saving=minimum_saving,
        maximum_saving=maximum_saving,
    )
    target_saving = min(max(requested_saving, saving_low), saving_high)
    target_cost = bau_cost - target_saving
    if isclose(maximum_cost, minimum_cost, rel_tol=0.0, abs_tol=settings.cost_tolerance):
        allocation = minimum
    else:
        alpha = (target_cost - minimum_cost) / (maximum_cost - minimum_cost)
        alpha = min(1.0, max(0.0, alpha))
        allocation = [
            low + alpha * (high - low) for low, high in zip(minimum, maximum, strict=True)
        ]
    realized_cost = sum(price * energy for price, energy in zip(prices, allocation, strict=True))
    realized_saving = bau_cost - realized_cost
    if (
        not saving_low - settings.saving_tolerance
        <= realized_saving
        <= saving_high + settings.saving_tolerance
    ):
        raise PhysicalConstraintError("failed to construct a feasible saving-band initial point.")
    return allocation


def _objective_value_and_jac(
    x: list[float],
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    degradation_settings: DegradationSettings,
    frontier_settings: FrontierSettings,
) -> tuple[float, list[float]]:
    params = degradation_settings.parameters_for(ev.chemistry)
    age_factor = _relative_age_factor(degradation_settings, params.calendar_time_exponent)
    count = len(x)
    states = [session.initial_energy_kwh]
    for grid_energy in x:
        states.append(states[-1] + ev.charging_efficiency * grid_energy)
    weights: list[float] = []
    objective = 0.0
    for local_index in range(count):
        global_index = session.arrival_step + local_index
        temperature = (
            degradation_settings.fallback_temperature_c
            if signal.battery_temperature_c is None
            else signal.battery_temperature_c[global_index]
        )
        factor = (
            _temperature_factor(
                temperature,
                degradation_settings.reference_temperature_c,
                params.activation_energy_j_per_mol,
            )
            * age_factor
            * signal.interval_durations[global_index]
            / _HOURS_PER_YEAR
        )
        soc = states[local_index] / ev.battery_capacity_kwh
        stress = calendar_soc_stress(soc, params)
        objective += stress * factor
        weights.append(factor)
        duration_hours = signal.interval_durations[global_index]
        power = x[local_index] / duration_hours
        battery_c_rate = ev.charging_efficiency * power / ev.battery_capacity_kwh
        objective += frontier_settings.plating_guard_weight * battery_c_rate**2 * duration_hours
    jacobian = [0.0] * count
    for variable_index in range(count):
        for state_index in range(variable_index + 1, count):
            soc = states[state_index] / ev.battery_capacity_kwh
            hinge = max(soc - params.calendar_soc_knee, 0.0)
            stress_derivative = params.calendar_a1 + 2.0 * params.calendar_a2 * hinge
            jacobian[variable_index] += (
                stress_derivative
                * weights[state_index]
                * ev.charging_efficiency
                / ev.battery_capacity_kwh
            )
        jacobian[variable_index] += (
            2.0
            * frontier_settings.plating_guard_weight
            * ev.charging_efficiency**2
            * x[variable_index]
            / (
                signal.interval_durations[session.arrival_step + variable_index]
                * ev.battery_capacity_kwh**2
            )
        )
    return float(objective), jacobian


def _trajectory_objective(
    x: list[float],
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    degradation_settings: DegradationSettings,
    frontier_settings: FrontierSettings,
) -> float:
    return _objective_value_and_jac(
        x,
        ev=ev,
        session=session,
        signal=signal,
        degradation_settings=degradation_settings,
        frontier_settings=frontier_settings,
    )[0]


def _validate_solver_vector(
    raw_vector: object,
    *,
    count: int,
    capacities: tuple[float, ...],
    required: float,
    prices: tuple[float, ...],
    bau_cost: float,
    requested_saving: float | None,
    settings: FrontierSettings,
) -> list[float]:
    if len(capacities) != count:
        raise SchemaValidationError("interval capacities must match the solver vector length.")
    if raw_vector is None or isinstance(raw_vector, (str, bytes)):
        raise PhysicalConstraintError("solver returned no usable decision vector.")
    try:
        values: tuple[object, ...] = tuple(cast(Iterable[object], raw_vector))
    except (TypeError, ValueError) as exc:
        raise PhysicalConstraintError("solver returned a non-iterable decision vector.") from exc
    if len(values) != count:
        raise PhysicalConstraintError("solver returned a decision vector of the wrong length.")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise PhysicalConstraintError("solver returned a non-finite decision value.")
        numeric = float(value)
        cap = capacities[len(normalized)]
        if numeric < -settings.bound_tolerance or numeric > cap + settings.bound_tolerance:
            raise PhysicalConstraintError("solver returned a decision outside interval bounds.")
        if abs(numeric) <= settings.bound_tolerance:
            numeric = 0.0
        elif abs(numeric - cap) <= settings.bound_tolerance:
            numeric = cap
        normalized.append(numeric)
    if abs(sum(normalized) - required) > settings.energy_tolerance:
        raise PhysicalConstraintError("solver returned an incorrect total energy.")
    if requested_saving is not None:
        realized_saving = bau_cost - sum(
            price * value for price, value in zip(prices, normalized, strict=True)
        )
        if (
            abs(realized_saving - requested_saving)
            > settings.saving_band + settings.saving_tolerance
        ):
            raise PhysicalConstraintError("solver returned a decision outside the saving band.")
    return normalized


def _coerce_callback_values(values: object) -> list[float]:
    try:
        raw_values = tuple(cast(Iterable[object], values))
    except (TypeError, ValueError) as exc:
        raise PhysicalConstraintError("solver callback received a non-iterable vector.") from exc
    result: list[float] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise PhysicalConstraintError("solver callback received a non-finite value.")
        result.append(float(value))
    return result


def _synthetic_candidate(
    *,
    point_id: str,
    ev: EVSpec,
    candidate: MenuCandidate,
    profile: ChargingProfile,
    charging_cost: float,
    bau_cost: float,
    required: float,
    validation: ValidationReport,
) -> MenuCandidate:
    return MenuCandidate(
        candidate_id=point_id,
        ev_id=ev.ev_id,
        kind="minimum_cost",
        target_soc=candidate.target_soc,
        target_sources=candidate.target_sources,
        target_label=candidate.target_label,
        ready_step=candidate.ready_step,
        charging_cost=charging_cost,
        same_target_bau_cost=bau_cost,
        saving=bau_cost - charging_cost,
        required_grid_energy_kwh=required,
        profile=profile,
        validation=validation,
    )


def _make_optimized_profile(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    bau_cost: float,
    profile: ChargingProfile,
    required: float,
    validation: ValidationReport,
    degradation_settings: DegradationSettings,
    frontier_settings: FrontierSettings,
    requested_saving: float | None,
    endpoint_role: EndpointRole,
) -> OptimizedProfile:
    if endpoint_role == "least_degradation":
        point_id = f"{candidate.candidate_id}:least-degradation"
    elif endpoint_role == "maximum_saving":
        point_id = f"{candidate.candidate_id}:maximum-saving"
    elif endpoint_role == "least_and_maximum":
        point_id = f"{candidate.candidate_id}:least-and-maximum"
    elif requested_saving is not None:
        point_id = f"{candidate.candidate_id}:saving:{_canonical_decimal(requested_saving)}"
    else:
        point_id = f"{candidate.candidate_id}:intermediate"
    charging_cost = _cost(profile, signal)
    synthetic = _synthetic_candidate(
        point_id=candidate.candidate_id,
        ev=ev,
        candidate=candidate,
        profile=profile,
        charging_cost=charging_cost,
        bau_cost=bau_cost,
        required=required,
        validation=validation,
    )
    assessment = assess_candidate_degradation(
        ev=ev,
        session=session,
        signal=signal,
        candidate=synthetic,
        menu_settings=MenuSettings(),
        degradation_settings=degradation_settings,
    )
    assessment = replace(assessment, candidate_id=point_id)
    return OptimizedProfile(
        constructed=ConstructedProfile(
            profile=profile,
            target_soc=candidate.target_soc,
            ready_step=candidate.ready_step,
            required_grid_energy_kwh=required,
            charging_cost=charging_cost,
            validation=validation,
        ),
        assessment=assessment,
        saving=bau_cost - charging_cost,
        trajectory_objective=_trajectory_objective(
            list(profile.grid_energy_kwh[: candidate.ready_step - session.arrival_step]),
            ev=ev,
            session=session,
            signal=signal,
            degradation_settings=degradation_settings,
            frontier_settings=frontier_settings,
        ),
        requested_saving=requested_saving,
        point_id=point_id,
        source_candidate_id=candidate.candidate_id,
        ev_id=ev.ev_id,
        endpoint_role=endpoint_role,
    )


def _solve(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    bau_cost: float,
    requested_saving: float | None,
    degradation_settings: DegradationSettings,
    frontier_settings: FrontierSettings,
    tolerances: ValidationTolerances,
) -> OptimizedProfile:
    n_session = session.departure_step - session.arrival_step
    n_ready = candidate.ready_step - session.arrival_step
    required = required_grid_energy_kwh(ev, session, candidate.target_soc)
    prices = signal.price_per_kwh[session.arrival_step : candidate.ready_step]
    capacities = tuple(
        ev.charger_power_kw * signal.interval_durations[step]
        for step in range(session.arrival_step, candidate.ready_step)
    )
    if required == 0.0:
        x: list[float] = [0.0] * n_ready
    else:
        if n_ready <= 0:
            raise PhysicalConstraintError("positive-energy request has no eligible interval.")
        if requested_saving is None:
            x = _even_allocation(required, capacities)
        else:
            x = _saving_initial_allocation(
                prices=prices,
                capacities=capacities,
                required=required,
                bau_cost=bau_cost,
                requested_saving=requested_saving,
                settings=frontier_settings,
            )
        constraints: list[dict[str, object]] = [
            {
                "type": "eq",
                "fun": lambda values: float(sum(values) - required),
                "jac": lambda values: [1.0] * len(values),
            }
        ]
        if requested_saving is not None:
            target_cost = bau_cost - requested_saving
            band = frontier_settings.saving_band
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda values, p=prices, upper=target_cost + band: float(
                            upper
                            - sum(price * value for price, value in zip(p, values, strict=True))
                        ),
                        "jac": lambda values, p=prices: [-price for price in p],
                    },
                    {
                        "type": "ineq",
                        "fun": lambda values, p=prices, lower=target_cost - band: float(
                            sum(price * value for price, value in zip(p, values, strict=True))
                            - lower
                        ),
                        "jac": lambda values, p=prices: list(p),
                    },
                ]
            )

        def scaled_objective(values: object) -> float:
            numeric = _coerce_callback_values(values)
            return (
                _trajectory_objective(
                    numeric,
                    ev=ev,
                    session=session,
                    signal=signal,
                    degradation_settings=degradation_settings,
                    frontier_settings=frontier_settings,
                )
                * frontier_settings.objective_scale
            )

        def scaled_jacobian(values: object) -> list[float]:
            numeric = _coerce_callback_values(values)
            return [
                value * frontier_settings.objective_scale
                for value in _objective_value_and_jac(
                    numeric,
                    ev=ev,
                    session=session,
                    signal=signal,
                    degradation_settings=degradation_settings,
                    frontier_settings=frontier_settings,
                )[1]
            ]

        result = cast(
            object,
            minimize(
                scaled_objective,
                x,
                jac=scaled_jacobian,
                method="SLSQP",
                bounds=[(0.0, capacity) for capacity in capacities],
                constraints=constraints,
                options={
                    "ftol": frontier_settings.effective_solver_ftol,
                    "maxiter": frontier_settings.effective_solver_max_iterations,
                    "disp": False,
                },
            ),
        )
        if result is None or not bool(getattr(result, "success", False)):
            message = str(getattr(result, "message", "unknown solver failure"))
            raise PhysicalConstraintError(f"degradation optimization failed: {message}")
        x = _validate_solver_vector(
            getattr(result, "x", None),
            count=n_ready,
            capacities=capacities,
            required=required,
            prices=prices,
            bau_cost=bau_cost,
            requested_saving=requested_saving,
            settings=frontier_settings,
        )
        solver_fun = getattr(result, "fun", None)
        if solver_fun is None or isinstance(solver_fun, bool) or not isinstance(solver_fun, Real):
            raise PhysicalConstraintError("solver returned no finite objective value.")
        solver_fun_float = float(solver_fun)
        if not isfinite(solver_fun_float):
            raise PhysicalConstraintError("solver returned a non-finite objective value.")
        recomputed = _trajectory_objective(
            x,
            ev=ev,
            session=session,
            signal=signal,
            degradation_settings=degradation_settings,
            frontier_settings=frontier_settings,
        )
        if (
            abs(recomputed - solver_fun_float / frontier_settings.objective_scale)
            > frontier_settings.objective_tolerance
        ):
            raise PhysicalConstraintError(
                "solver objective does not match independent recomputation."
            )
    full_energy = x + [0.0] * (n_session - n_ready)
    if abs(sum(full_energy) - required) > max(
        tolerances.energy_kwh, frontier_settings.energy_tolerance
    ):
        raise PhysicalConstraintError(
            "optimized profile does not deliver the required grid energy."
        )
    profile = _profile_from_energy(
        ev,
        session,
        signal,
        full_energy,
        energy_tolerance=max(tolerances.energy_kwh, frontier_settings.energy_tolerance),
    )
    report = validate_charging_profile(
        ev=ev,
        session=session,
        signal=signal,
        target_soc=candidate.target_soc,
        ready_step=candidate.ready_step,
        profile=profile,
        tolerances=tolerances,
    )
    if not report.is_valid:
        codes = ", ".join(issue.code.value for issue in report.issues)
        raise PhysicalConstraintError(f"optimized profile failed validation: {codes}")
    charging_cost = _cost(profile, signal)
    saving = bau_cost - charging_cost
    if requested_saving is not None and abs(saving - requested_saving) > (
        frontier_settings.saving_band + frontier_settings.saving_tolerance
    ):
        raise PhysicalConstraintError("optimized profile violates the requested saving band.")
    return _make_optimized_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=bau_cost,
        profile=profile,
        required=required,
        validation=report,
        degradation_settings=degradation_settings,
        frontier_settings=frontier_settings,
        requested_saving=requested_saving,
        endpoint_role="least_degradation" if requested_saving is None else "intermediate",
    )


def build_least_degradation_profile(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    bau_cost: float,
    degradation_settings: DegradationSettings | None = None,
    frontier_settings: FrontierSettings | None = None,
    tolerances: ValidationTolerances | None = None,
) -> OptimizedProfile:
    """Minimize window calendar fade plus the quadratic plating guard."""
    model = DegradationSettings() if degradation_settings is None else degradation_settings
    settings = FrontierSettings() if frontier_settings is None else frontier_settings
    validation = ValidationTolerances() if tolerances is None else tolerances
    numeric_bau_cost = _validate_inputs(
        ev, session, signal, candidate, bau_cost, model, settings, validation
    )
    return _solve(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=numeric_bau_cost,
        requested_saving=None,
        degradation_settings=model,
        frontier_settings=settings,
        tolerances=validation,
    )


def build_saving_constrained_profile(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    bau_cost: float,
    requested_saving: float,
    degradation_settings: DegradationSettings | None = None,
    frontier_settings: FrontierSettings | None = None,
    tolerances: ValidationTolerances | None = None,
) -> OptimizedProfile:
    """Minimize the trajectory objective inside a requested saving band."""
    target_saving = _finite("requested_saving", requested_saving)
    model = DegradationSettings() if degradation_settings is None else degradation_settings
    settings = FrontierSettings() if frontier_settings is None else frontier_settings
    validation = ValidationTolerances() if tolerances is None else tolerances
    numeric_bau_cost = _validate_inputs(
        ev, session, signal, candidate, bau_cost, model, settings, validation
    )
    _, _, minimum_cost, maximum_cost = _cost_range(
        prices=signal.price_per_kwh[session.arrival_step : candidate.ready_step],
        capacities=tuple(
            ev.charger_power_kw * signal.interval_durations[step]
            for step in range(session.arrival_step, candidate.ready_step)
        ),
        required=required_grid_energy_kwh(ev, session, candidate.target_soc),
    )
    _saving_interval(
        bau_cost=numeric_bau_cost,
        requested_saving=target_saving,
        saving_band=settings.saving_band,
        minimum_saving=numeric_bau_cost - maximum_cost,
        maximum_saving=numeric_bau_cost - minimum_cost,
    )
    if isclose(target_saving, candidate.saving, rel_tol=0.0, abs_tol=settings.cost_tolerance):
        return _make_optimized_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=numeric_bau_cost,
            profile=candidate.profile,
            required=candidate.required_grid_energy_kwh,
            validation=candidate.validation,
            degradation_settings=model,
            frontier_settings=settings,
            requested_saving=target_saving,
            endpoint_role="maximum_saving",
        )
    return _solve(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=numeric_bau_cost,
        requested_saving=target_saving,
        degradation_settings=model,
        frontier_settings=settings,
        tolerances=validation,
    )


def _same_point(left: OptimizedProfile, right: OptimizedProfile) -> bool:
    return (
        left.saving == right.saving
        and left.constructed.profile == right.constructed.profile
        and left.constructed.charging_cost == right.constructed.charging_cost
        and left.trajectory_objective == right.trajectory_objective
        and left.assessment.ev_id == right.assessment.ev_id
        and left.assessment.chemistry == right.assessment.chemistry
        and left.assessment.charging_window_calendar_fade
        == right.assessment.charging_window_calendar_fade
        and left.assessment.parked_day_calendar_fade == right.assessment.parked_day_calendar_fade
        and left.assessment.cycle_fade == right.assessment.cycle_fade
        and left.assessment.total_fade == right.assessment.total_fade
        and left.assessment.annualized_degradation_pct
        == right.assessment.annualized_degradation_pct
        and left.assessment.parked_soc == right.assessment.parked_soc
        and left.assessment.peak_c_rate == right.assessment.peak_c_rate
    )


def build_sandwich_saving_frontier(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    bau_cost: float,
    degradation_settings: DegradationSettings | None = None,
    frontier_settings: FrontierSettings | None = None,
    tolerances: ValidationTolerances | None = None,
) -> SavingFrontier:
    """Build endpoint-anchored midpoint levels on the useful saving branch."""
    model = DegradationSettings() if degradation_settings is None else degradation_settings
    settings = FrontierSettings() if frontier_settings is None else frontier_settings
    validation = ValidationTolerances() if tolerances is None else tolerances
    numeric_bau_cost = _validate_inputs(
        ev, session, signal, candidate, bau_cost, model, settings, validation
    )
    least = build_least_degradation_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=numeric_bau_cost,
        degradation_settings=model,
        frontier_settings=settings,
        tolerances=validation,
    )
    maximum = build_saving_constrained_profile(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        bau_cost=numeric_bau_cost,
        requested_saving=numeric_bau_cost - candidate.charging_cost,
        degradation_settings=model,
        frontier_settings=settings,
        tolerances=validation,
    )
    endpoint_tolerance = max(
        settings.saving_band,
        settings.saving_tolerance,
        settings.cost_tolerance,
    )
    if _same_point(least, maximum) or abs(least.saving - maximum.saving) <= endpoint_tolerance:
        endpoint = maximum
        collapsed = _make_optimized_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=numeric_bau_cost,
            profile=endpoint.constructed.profile,
            required=endpoint.constructed.required_grid_energy_kwh,
            validation=endpoint.constructed.validation,
            degradation_settings=model,
            frontier_settings=settings,
            requested_saving=None,
            endpoint_role="least_and_maximum",
        )
        points = [collapsed]
    else:
        points = [least, maximum]
    attempted: set[float] = set()
    while len(points) < settings.maximum_levels and len(points) > 1:
        points.sort(key=lambda point: point.saving)
        gaps = [
            (right.saving - left.saving, left.saving, right.saving)
            for left, right in pairwise(points)
        ]
        eligible = [
            item
            for item in gaps
            if item[0] > max(2.0 * settings.saving_band, settings.frontier_gap_tolerance)
        ]
        if not eligible:
            break
        _, lower, upper = max(eligible, key=lambda item: (item[0], -item[1]))
        midpoint = (lower + upper) / 2.0
        if midpoint in attempted:
            break
        attempted.add(midpoint)
        point = build_saving_constrained_profile(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            bau_cost=numeric_bau_cost,
            requested_saving=midpoint,
            degradation_settings=model,
            frontier_settings=settings,
            tolerances=validation,
        )
        if any(_same_point(point, existing) for existing in points):
            continue
        points.append(point)
    points.sort(key=lambda point: point.saving)
    if len(points) > 1:
        for left, right in pairwise(points):
            if (
                right.trajectory_objective + settings.objective_tolerance
                < left.trajectory_objective
            ):
                raise PhysicalConstraintError("frontier trajectory objective is not monotone.")
    return SavingFrontier(
        ev_id=ev.ev_id,
        target_soc=candidate.target_soc,
        ready_step=candidate.ready_step,
        bau_cost=numeric_bau_cost,
        points=tuple(points),
        source_candidate_id=candidate.candidate_id,
    )
