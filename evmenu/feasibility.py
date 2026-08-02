"""Pure feasibility calculations for one residential EV charging session.

The functions in this module do not construct charging trajectories.  They
implement the deterministic energy and timing calculations used by later menu
construction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real

from .exceptions import PhysicalConstraintError, SchemaValidationError
from .schemas import (
    ChargingSession,
    EVSpec,
    MenuSettings,
    PlanningSignal,
    TargetOption,
    TargetSource,
)


@dataclass(frozen=True, slots=True)
class RequestFeasibility:
    """Diagnostic result for one ready-time/target request."""

    is_feasible: bool
    required_grid_energy_kwh: float
    available_grid_energy_kwh: float
    energy_margin_kwh: float

    @property
    def margin_kwh(self) -> float:
        """Backward-compatible short name for the grid-energy margin."""
        return self.energy_margin_kwh


def _finite_real(name: str, value: object) -> float:
    """Validate a public scalar without coercing arbitrary objects."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)


def compute_buffer_energy(
    commute_energy_kwh: float,
    *,
    base_buffer_kwh: float,
    commute_buffer_fraction: float,
) -> float:
    """Return the population-rule buffer energy.

    Equation
    --------
    ``b = max(b_0, gamma_b * E_com)``.
    """
    numeric_values: dict[str, float] = {}
    for name, value in (
        ("commute_energy_kwh", commute_energy_kwh),
        ("base_buffer_kwh", base_buffer_kwh),
        ("commute_buffer_fraction", commute_buffer_fraction),
    ):
        numeric_value = _finite_real(name, value)
        numeric_values[name] = numeric_value
        if numeric_value < 0.0:
            raise PhysicalConstraintError(f"{name} must be non-negative.")
    return max(
        numeric_values["base_buffer_kwh"],
        numeric_values["commute_buffer_fraction"] * numeric_values["commute_energy_kwh"],
    )


def minimum_required_departure_energy_kwh(
    ev: EVSpec,
    session: ChargingSession,
) -> float:
    """Return battery energy needed at departure for commute plus buffer.

    Equation
    --------
    ``B_req,min = B_min + E_com + b``.
    """
    session.validate_for_ev(ev)
    return ev.minimum_energy_kwh + session.commute_energy_kwh + session.buffer_energy_kwh


def minimum_required_target_soc(ev: EVSpec, session: ChargingSession) -> float:
    """Return the personalized commute-plus-buffer target SOC.

    Equation
    --------
    ``z_min = (B_min + E_com + b) / B_max``.
    """
    return minimum_required_departure_energy_kwh(ev, session) / ev.battery_capacity_kwh


def delivered_target_energy_kwh(
    ev: EVSpec,
    session: ChargingSession,
    target_soc: float,
) -> float:
    """Return exact delivered energy without allowing discharge.

    Equation
    --------
    ``B_target = max(B_0, z * B_max)``.
    """
    target_soc = _finite_real("target_soc", target_soc)
    if not 0.0 <= target_soc <= 1.0:
        raise PhysicalConstraintError("target_soc must lie in [0, 1].")
    session.validate_for_ev(ev)
    return max(session.initial_energy_kwh, target_soc * ev.battery_capacity_kwh)


def required_grid_energy_kwh(
    ev: EVSpec,
    session: ChargingSession,
    target_soc: float,
) -> float:
    """Return grid-side energy required to provide a target exactly.

    Equation
    --------
    ``E_req(z) = max(0, z B_max - B_0) / eta``.
    """
    delivered = delivered_target_energy_kwh(ev, session, target_soc)
    return (delivered - session.initial_energy_kwh) / ev.charging_efficiency


def available_grid_energy_kwh(
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    ready_step: int,
) -> float:
    """Return maximum grid energy deliverable before ``ready_step``.

    Equation
    --------
    ``E_avail(r) = p_max * sum(Delta_t[k] for k before r)``.
    """
    _validate_ready_step(session, signal, ready_step)
    return ev.charger_power_kw * sum(
        signal.interval_durations[step] for step in range(session.arrival_step, ready_step)
    )


def evaluate_request_feasibility(
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    *,
    target_soc: float,
    ready_step: int,
    tolerance: float = 1e-8,
) -> RequestFeasibility:
    """Evaluate whether a target can be reached by a requested ready step."""
    tolerance = _finite_real("tolerance", tolerance)
    if tolerance < 0:
        raise SchemaValidationError("tolerance must be a non-negative real number.")
    session.validate_for_ev(ev)
    signal.validate_session_window(session)
    required = required_grid_energy_kwh(ev, session, target_soc)
    available = available_grid_energy_kwh(ev, session, signal, ready_step)
    margin = available - required
    return RequestFeasibility(
        is_feasible=required <= available + tolerance,
        required_grid_energy_kwh=required,
        available_grid_energy_kwh=available,
        energy_margin_kwh=margin,
    )


def request_is_feasible(
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    *,
    target_soc: float,
    ready_step: int,
    tolerance: float = 1e-8,
) -> bool:
    """Return only the boolean result of :func:`evaluate_request_feasibility`."""
    return evaluate_request_feasibility(
        ev,
        session,
        signal,
        target_soc=target_soc,
        ready_step=ready_step,
        tolerance=tolerance,
    ).is_feasible


def build_target_options(
    ev: EVSpec,
    session: ChargingSession,
    settings: MenuSettings,
) -> tuple[TargetOption, ...]:
    """Build valid personalized and standard target options.

    Standard targets below the minimum service requirement are omitted. Targets
    within ``target_merge_tolerance`` are merged while retaining all provenance.
    The retained numerical target is the larger value, ensuring serviceability.
    """
    if settings.standard_targets != (0.80, 0.90, 1.00):
        raise SchemaValidationError("standard_targets must equal (0.80, 0.90, 1.00) in Commit 2.")
    z_min = minimum_required_target_soc(ev, session)
    if z_min > 1.0:
        raise PhysicalConstraintError("commute-plus-buffer target exceeds 100% SOC.")

    candidates: list[tuple[float, TargetSource]] = [(z_min, "minimum_required")]
    for target in settings.standard_targets:
        source = _standard_target_source(target)
        if target + settings.target_merge_tolerance >= z_min:
            candidates.append((target, source))

    candidates.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, TargetSource]]] = []
    for candidate in candidates:
        if (
            not groups
            or candidate[0] - max(value for value, _ in groups[-1])
            > settings.target_merge_tolerance
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)

    options: list[TargetOption] = []
    for group in groups:
        target_soc = max(value for value, _ in group)
        sources = tuple(source for _, source in group)
        options.append(
            TargetOption(
                target_soc=target_soc,
                sources=sources,
                label=_target_label(sources, target_soc),
            )
        )
    return tuple(options)


def _validate_ready_step(
    session: ChargingSession,
    signal: PlanningSignal,
    ready_step: int,
) -> None:
    if isinstance(ready_step, bool) or not isinstance(ready_step, int):
        raise SchemaValidationError("ready_step must be an integer.")
    signal.validate_session_window(session)
    if ready_step < session.arrival_step:
        raise PhysicalConstraintError("ready_step cannot precede arrival_step.")
    if ready_step > session.departure_step:
        raise PhysicalConstraintError("ready_step cannot exceed departure_step.")


def _standard_target_source(target: float) -> TargetSource:
    supported: tuple[tuple[float, TargetSource], ...] = (
        (0.80, "standard_80"),
        (0.90, "standard_90"),
        (1.00, "standard_100"),
    )
    for expected, source in supported:
        if isclose(target, expected, rel_tol=0.0, abs_tol=0.0):
            return source
    raise SchemaValidationError(
        "Commit 2 supports standard target provenance only for 0.80, 0.90, and 1.00."
    )


def _target_label(sources: tuple[TargetSource, ...], target_soc: float) -> str:
    if "minimum_required" in sources and len(sources) > 1:
        return f"Daily commute + buffer (merged at {100 * target_soc:.0f}%)"
    if sources == ("minimum_required",):
        return "Daily commute + buffer"
    return f"Standard {100 * target_soc:.0f}% target"
