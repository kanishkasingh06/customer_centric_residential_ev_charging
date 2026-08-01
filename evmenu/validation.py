"""Independent cross-object validation of EV charging trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real

from .exceptions import (
    PhysicalConstraintError,
    SchemaValidationError,
    SignalValidationError,
)
from .schemas import ChargingProfile, ChargingSession, EVSpec, PlanningSignal


class ValidationCode(StrEnum):
    """Stable machine-readable identifiers for profile validation issues."""

    SESSION_INVALID = "SESSION_INVALID"
    SIGNAL_COVERAGE = "SIGNAL_COVERAGE"
    PROFILE_ALIGNMENT = "PROFILE_ALIGNMENT"
    PROFILE_LENGTH = "PROFILE_LENGTH"
    INITIAL_ENERGY = "INITIAL_ENERGY"
    POWER_LIMIT = "POWER_LIMIT"
    GRID_ENERGY_MISMATCH = "GRID_ENERGY_MISMATCH"
    BATTERY_RECURSION = "BATTERY_RECURSION"
    BATTERY_BELOW_MINIMUM = "BATTERY_BELOW_MINIMUM"
    BATTERY_ABOVE_CAPACITY = "BATTERY_ABOVE_CAPACITY"
    SOC_MISMATCH = "SOC_MISMATCH"
    READY_TIME_VIOLATION = "READY_TIME_VIOLATION"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    COMMUTE_BUFFER_VIOLATION = "COMMUTE_BUFFER_VIOLATION"


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    numeric_value = float(value)
    if numeric_value <= 0.0:
        raise SchemaValidationError(f"{name} must be strictly positive.")
    return numeric_value


def _finite_target(target_soc: object) -> float:
    if isinstance(target_soc, bool) or not isinstance(target_soc, Real) or not isfinite(target_soc):
        raise SchemaValidationError("target_soc must be a finite real number.")
    numeric_target = float(target_soc)
    if not 0.0 <= numeric_target <= 1.0:
        raise PhysicalConstraintError("target_soc must lie in [0, 1].")
    return numeric_target


@dataclass(frozen=True, slots=True)
class ValidationTolerances:
    """Independent numerical tolerances for physical validation checks."""

    power_kw: float = 1e-8
    energy_kwh: float = 1e-8
    soc: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "power_kw", _finite_positive("power_kw", self.power_kw))
        object.__setattr__(self, "energy_kwh", _finite_positive("energy_kwh", self.energy_kwh))
        object.__setattr__(self, "soc", _finite_positive("soc", self.soc))


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One independently detected physical validation issue."""

    code: ValidationCode
    message: str
    interval_index: int | None = None
    observed: float | None = None
    expected: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Detailed validation result with stable issue codes and diagnostics."""

    issues: tuple[ValidationIssue, ...]
    target_energy_kwh: float = 0.0
    ready_energy_kwh: float | None = None
    terminal_energy_kwh: float = 0.0
    maximum_power_kw: float = 0.0
    maximum_energy_balance_error_kwh: float = 0.0
    minimum_commute_margin_kwh: float | None = None

    @property
    def is_valid(self) -> bool:
        """Whether no validation issues were found."""
        return not self.issues

    @property
    def errors(self) -> tuple[str, ...]:
        """Compatibility view of issue messages; use ``issues`` for new code."""
        return tuple(issue.message for issue in self.issues)

    def require_valid(self) -> None:
        """Raise when validation failed, preserving all diagnostics."""
        if not self.is_valid:
            raise SchemaValidationError("Invalid charging profile: " + "; ".join(self.errors))


def validate_charging_profile(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    target_soc: float,
    ready_step: int,
    profile: ChargingProfile,
    tolerances: ValidationTolerances | None = None,
) -> ValidationReport:
    """Validate a complete, exact-session charging profile.

    Profiles must start at ``session.arrival_step`` and cover exactly the
    half-open interval ``[arrival_step, departure_step)``.  Battery and SOC
    vectors therefore contain one more boundary state than power and grid
    energy vectors.  The function reports physically incompatible, well-formed
    objects rather than raising; malformed direct arguments still raise a
    schema or physical exception.
    """
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec instance.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession instance.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal instance.")
    if not isinstance(profile, ChargingProfile):
        raise SchemaValidationError("profile must be a ChargingProfile instance.")
    if tolerances is None:
        tolerances = ValidationTolerances()
    elif not isinstance(tolerances, ValidationTolerances):
        raise SchemaValidationError("tolerances must be a ValidationTolerances instance.")

    target_soc = _finite_target(target_soc)
    if isinstance(ready_step, bool) or not isinstance(ready_step, int):
        raise SchemaValidationError("ready_step must be an integer.")

    issues: list[ValidationIssue] = []

    def add(
        code: ValidationCode,
        message: str,
        *,
        interval_index: int | None = None,
        observed: float | None = None,
        expected: float | None = None,
    ) -> None:
        issues.append(
            ValidationIssue(
                code,
                message,
                interval_index=interval_index,
                observed=observed,
                expected=expected,
            )
        )

    try:
        session.validate_for_ev(ev)
    except PhysicalConstraintError as exc:
        add(ValidationCode.SESSION_INVALID, str(exc))

    try:
        signal.validate_session_window(session)
    except SignalValidationError as exc:
        add(ValidationCode.SIGNAL_COVERAGE, str(exc))

    if ready_step < session.arrival_step or ready_step > session.departure_step:
        add(
            ValidationCode.READY_TIME_VIOLATION,
            "ready_step must lie in [arrival_step, departure_step].",
            observed=float(ready_step),
            expected=float(session.arrival_step),
        )

    expected_intervals = session.departure_step - session.arrival_step
    profile_aligned = profile.start_step == session.arrival_step
    if not profile_aligned:
        add(
            ValidationCode.PROFILE_ALIGNMENT,
            "profile.start_step must equal session.arrival_step.",
            observed=float(profile.start_step),
            expected=float(session.arrival_step),
        )
    profile_length_valid = len(profile.power_kw) == expected_intervals
    if not profile_length_valid:
        add(
            ValidationCode.PROFILE_LENGTH,
            "profile must span every interval from arrival through departure.",
            observed=float(len(profile.power_kw)),
            expected=float(expected_intervals),
        )
    if profile.start_step + len(profile.power_kw) > signal.number_of_steps:
        add(
            ValidationCode.SIGNAL_COVERAGE,
            "profile extends beyond the planning-signal horizon.",
        )

    target_energy = max(session.initial_energy_kwh, target_soc * ev.battery_capacity_kwh)
    if (
        profile_aligned
        and abs(profile.battery_energy_kwh[0] - session.initial_energy_kwh) > tolerances.energy_kwh
    ):
        add(
            ValidationCode.INITIAL_ENERGY,
            "profile initial battery energy does not equal session initial energy.",
            observed=profile.battery_energy_kwh[0],
            expected=session.initial_energy_kwh,
        )

    max_power = max(profile.power_kw)
    max_balance_error = 0.0
    for local_index, (power, grid_energy) in enumerate(
        zip(profile.power_kw, profile.grid_energy_kwh, strict=True)
    ):
        global_step = profile.start_step + local_index
        if power > ev.charger_power_kw + tolerances.power_kw:
            add(
                ValidationCode.POWER_LIMIT,
                f"power exceeds charger limit at step {global_step}.",
                interval_index=global_step,
                observed=power,
                expected=ev.charger_power_kw,
            )
        expected_grid_energy = power * signal.timestep_hours
        grid_error = abs(grid_energy - expected_grid_energy)
        if grid_error > tolerances.energy_kwh:
            add(
                ValidationCode.GRID_ENERGY_MISMATCH,
                f"grid energy and power are inconsistent at step {global_step}.",
                interval_index=global_step,
                observed=grid_energy,
                expected=expected_grid_energy,
            )

        expected_next = (
            profile.battery_energy_kwh[local_index] + ev.charging_efficiency * grid_energy
        )
        balance_error = abs(profile.battery_energy_kwh[local_index + 1] - expected_next)
        max_balance_error = max(max_balance_error, balance_error)
        if balance_error > tolerances.energy_kwh:
            add(
                ValidationCode.BATTERY_RECURSION,
                f"battery energy recursion is violated at step {global_step}.",
                interval_index=global_step,
                observed=profile.battery_energy_kwh[local_index + 1],
                expected=expected_next,
            )

        if global_step >= ready_step and power > tolerances.power_kw:
            add(
                ValidationCode.READY_TIME_VIOLATION,
                f"charging occurs at or after ready_step at step {global_step}.",
                interval_index=global_step,
                observed=power,
                expected=0.0,
            )

    for local_index, energy in enumerate(profile.battery_energy_kwh):
        state_step = profile.start_step + local_index
        if energy < ev.minimum_energy_kwh - tolerances.energy_kwh:
            add(
                ValidationCode.BATTERY_BELOW_MINIMUM,
                f"battery energy is below B_min at state step {state_step}.",
                interval_index=state_step,
                observed=energy,
                expected=ev.minimum_energy_kwh,
            )
        if energy > ev.battery_capacity_kwh + tolerances.energy_kwh:
            add(
                ValidationCode.BATTERY_ABOVE_CAPACITY,
                f"battery energy exceeds B_max at state step {state_step}.",
                interval_index=state_step,
                observed=energy,
                expected=ev.battery_capacity_kwh,
            )
        expected_soc = energy / ev.battery_capacity_kwh
        if abs(profile.soc[local_index] - expected_soc) > tolerances.soc:
            add(
                ValidationCode.SOC_MISMATCH,
                f"SOC is inconsistent with battery energy at state step {state_step}.",
                interval_index=state_step,
                observed=profile.soc[local_index],
                expected=expected_soc,
            )

    ready_energy: float | None = None
    commute_margin: float | None = None
    if session.arrival_step <= ready_step <= session.departure_step and profile_aligned:
        ready_index = ready_step - profile.start_step
        if 0 <= ready_index < len(profile.battery_energy_kwh):
            ready_energy = profile.battery_energy_kwh[ready_index]
            if abs(ready_energy - target_energy) > tolerances.energy_kwh:
                add(
                    ValidationCode.TARGET_MISMATCH,
                    "battery energy at ready_step does not equal the exact delivered target.",
                    interval_index=ready_step,
                    observed=ready_energy,
                    expected=target_energy,
                )
            commute_margin = (
                ready_energy
                - session.commute_energy_kwh
                - ev.minimum_energy_kwh
                - session.buffer_energy_kwh
            )
            if commute_margin < -tolerances.energy_kwh:
                add(
                    ValidationCode.COMMUTE_BUFFER_VIOLATION,
                    "ready energy does not cover commute plus buffer above B_min.",
                    interval_index=ready_step,
                    observed=commute_margin,
                    expected=0.0,
                )

    return ValidationReport(
        issues=tuple(issues),
        target_energy_kwh=target_energy,
        ready_energy_kwh=ready_energy,
        terminal_energy_kwh=profile.battery_energy_kwh[-1],
        maximum_power_kw=max_power,
        maximum_energy_balance_error_kwh=max_balance_error,
        minimum_commute_margin_kwh=commute_margin,
    )
