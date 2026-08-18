"""Immutable data contracts for the residential EV menu generator.

This module deliberately contains no menu construction, optimization,
customer-choice, Monte Carlo, power-flow, degradation, plotting, or file-I/O
logic. It defines the contracts exchanged by those later layers and validates
states that are locally knowable without constructing a charging trajectory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal, cast

from .exceptions import (
    EVMenuError,
    PhysicalConstraintError,
    SchemaValidationError,
    SignalValidationError,
)

Chemistry = Literal["LFP", "NMC"]
TargetSource = Literal[
    "minimum_required",
    "standard_80",
    "standard_90",
    "standard_100",
]

_VALID_TARGET_SOURCES: tuple[TargetSource, ...] = (
    "minimum_required",
    "standard_80",
    "standard_90",
    "standard_100",
)
_TARGET_SOURCE_ORDER = {source: index for index, source in enumerate(_VALID_TARGET_SOURCES)}


def _require_finite(
    name: str,
    value: object,
    *,
    error_type: type[EVMenuError] = SchemaValidationError,
) -> None:
    """Require a finite real value while rejecting bool and non-real objects."""
    _finite_real(name, value, error_type=error_type)


def _finite_real(
    name: str,
    value: object,
    *,
    error_type: type[EVMenuError] = SchemaValidationError,
) -> float:
    """Return a validated real value without coercing its runtime representation."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise error_type(f"{name} must be a finite real number; received {value!r}.")
    return cast(float, value)


def _require_nonnegative(
    name: str,
    value: object,
    *,
    error_type: type[EVMenuError] = SchemaValidationError,
) -> None:
    numeric_value = _finite_real(name, value, error_type=error_type)
    if numeric_value < 0.0:
        raise error_type(f"{name} must be non-negative; received {numeric_value}.")


def _require_positive(
    name: str,
    value: object,
    *,
    error_type: type[EVMenuError] = SchemaValidationError,
) -> None:
    numeric_value = _finite_real(name, value, error_type=error_type)
    if numeric_value <= 0.0:
        raise error_type(f"{name} must be positive; received {numeric_value}.")


def _canonical_text(name: str, value: object) -> str:
    """Return a nonempty, stripped identifier or label."""
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a string.")
    canonical = value.strip()
    if not canonical:
        raise SchemaValidationError(f"{name} must be a non-empty string.")
    return canonical


def _require_step(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{name} must be an integer.")


def _freeze_numeric_tuple(
    name: str,
    values: object,
    *,
    error_type: type[EVMenuError] = SchemaValidationError,
) -> tuple[float, ...]:
    """Copy a numeric sequence to an immutable tuple and validate finiteness."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise error_type(f"{name} must be a sequence of finite real numbers.")
    frozen = tuple(value for value in values)
    for index, value in enumerate(frozen):
        _require_finite(f"{name}[{index}]", value, error_type=error_type)
    return cast(tuple[float, ...], frozen)


def _freeze_target_sources(name: str, values: object) -> tuple[TargetSource, ...]:
    """Copy, validate, deduplicate-check, and canonically order target sources."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SchemaValidationError(f"{name} must be a sequence of target sources.")
    frozen = tuple(value for value in values)
    if not frozen:
        raise SchemaValidationError(f"{name} cannot be empty.")
    if any(source not in _VALID_TARGET_SOURCES for source in frozen):
        raise SchemaValidationError(f"{name} contains an unsupported target source.")
    if len(set(frozen)) != len(frozen):
        raise SchemaValidationError(f"{name} cannot contain duplicate target sources.")
    ordered = tuple(sorted(frozen, key=_TARGET_SOURCE_ORDER.__getitem__))
    return cast(tuple[TargetSource, ...], ordered)


@dataclass(frozen=True, slots=True)
class EVSpec:
    """Static physical specification of one electric vehicle.

    ``battery_capacity_kwh`` is battery-side usable maximum energy ``B_max``;
    it is not nominal/nameplate capacity. ``minimum_energy_kwh`` is the
    absolute battery-energy floor ``B_min``. ``charging_efficiency`` is the
    grid-to-battery efficiency ``eta = battery energy increase / grid energy
    drawn``.
    """

    ev_id: str
    battery_capacity_kwh: float
    minimum_energy_kwh: float
    charger_power_kw: float
    charging_efficiency: float
    chemistry: Chemistry

    def __post_init__(self) -> None:
        object.__setattr__(self, "ev_id", _canonical_text("ev_id", self.ev_id))
        _require_positive("battery_capacity_kwh", self.battery_capacity_kwh)
        _require_nonnegative("minimum_energy_kwh", self.minimum_energy_kwh)
        _require_positive("charger_power_kw", self.charger_power_kw)
        _require_finite("charging_efficiency", self.charging_efficiency)

        if self.minimum_energy_kwh >= self.battery_capacity_kwh:
            raise PhysicalConstraintError(
                "minimum_energy_kwh must be strictly below battery_capacity_kwh."
            )
        if not 0.0 < self.charging_efficiency <= 1.0:
            raise PhysicalConstraintError("charging_efficiency must lie in (0, 1].")
        if self.chemistry not in ("LFP", "NMC"):
            raise SchemaValidationError(
                f"chemistry must be 'LFP' or 'NMC'; received {self.chemistry!r}."
            )


@dataclass(frozen=True, slots=True)
class ChargingSession:
    """One plug-in-to-departure charging session on a planning time grid.

    ``arrival_step`` is inclusive and ``departure_step`` is exclusive. A
    later cross-object validator must ensure the session fits its
    :class:`PlanningSignal` horizon.
    """

    arrival_step: int
    departure_step: int
    initial_energy_kwh: float
    commute_energy_kwh: float
    buffer_energy_kwh: float

    def __post_init__(self) -> None:
        _require_step("arrival_step", self.arrival_step)
        _require_step("departure_step", self.departure_step)
        if self.arrival_step < 0:
            raise SchemaValidationError("arrival_step must be non-negative.")
        if self.departure_step <= self.arrival_step:
            raise PhysicalConstraintError(
                "departure_step must be strictly greater than arrival_step."
            )
        _require_nonnegative("initial_energy_kwh", self.initial_energy_kwh)
        _require_nonnegative("commute_energy_kwh", self.commute_energy_kwh)
        _require_nonnegative("buffer_energy_kwh", self.buffer_energy_kwh)

    def validate_for_ev(self, ev: EVSpec) -> None:
        """Validate session requirements that depend only on an EV specification."""
        if self.initial_energy_kwh < ev.minimum_energy_kwh:
            raise PhysicalConstraintError(
                "initial_energy_kwh is below the EV minimum-energy floor."
            )
        if self.initial_energy_kwh > ev.battery_capacity_kwh:
            raise PhysicalConstraintError("initial_energy_kwh exceeds the usable battery capacity.")
        if (
            ev.minimum_energy_kwh + self.commute_energy_kwh + self.buffer_energy_kwh
            > ev.battery_capacity_kwh
        ):
            raise PhysicalConstraintError(
                "minimum energy, commute energy, and buffer energy exceed usable capacity."
            )


@dataclass(frozen=True, slots=True)
class PlanningSignal:
    """Exogenous inputs for an arbitrary contiguous planning horizon.

    Step 0 is the first represented interval, not necessarily midnight.
    ``price_per_kwh[t]`` is the grid-energy price for interval ``t`` and may
    be negative. ``base_load_kw`` is non-negative active load. Battery
    temperature is representative battery/pack temperature in degrees Celsius,
    not ambient temperature.
    """

    timestep_hours: float
    price_per_kwh: tuple[float, ...]
    base_load_kw: tuple[float, ...] | None = None
    battery_temperature_c: tuple[float, ...] | None = None
    interval_duration_hours: tuple[float, ...] | None = None
    interval_start_minutes: tuple[int, ...] | None = None
    interval_end_minutes: tuple[int, ...] | None = None
    nominal_timestep_minutes: int | None = None

    def __post_init__(self) -> None:
        _require_positive("timestep_hours", self.timestep_hours, error_type=SignalValidationError)
        price = _freeze_numeric_tuple(
            "price_per_kwh", self.price_per_kwh, error_type=SignalValidationError
        )
        if not price:
            raise SignalValidationError("price_per_kwh must contain at least one step.")

        base_load = (
            None
            if self.base_load_kw is None
            else _freeze_numeric_tuple(
                "base_load_kw", self.base_load_kw, error_type=SignalValidationError
            )
        )
        temperature = (
            None
            if self.battery_temperature_c is None
            else _freeze_numeric_tuple(
                "battery_temperature_c",
                self.battery_temperature_c,
                error_type=SignalValidationError,
            )
        )

        expected_length = len(price)
        if self.interval_duration_hours is None:
            durations = (float(self.timestep_hours),) * expected_length
        else:
            durations = _freeze_numeric_tuple(
                "interval_duration_hours",
                self.interval_duration_hours,
                error_type=SignalValidationError,
            )
            if len(durations) != expected_length:
                raise SignalValidationError(
                    "interval_duration_hours must have the same length as price_per_kwh."
                )
            if any(value <= 0.0 for value in durations):
                raise SignalValidationError("interval_duration_hours must be positive.")

        if self.nominal_timestep_minutes is None:
            nominal_minutes = max(1, round(float(self.timestep_hours) * 60.0))
        else:
            _require_step("nominal_timestep_minutes", self.nominal_timestep_minutes)
            nominal_minutes = self.nominal_timestep_minutes
        if nominal_minutes <= 0:
            raise SignalValidationError("nominal_timestep_minutes must be positive.")

        starts = self.interval_start_minutes
        ends = self.interval_end_minutes
        if (starts is None) != (ends is None):
            raise SignalValidationError(
                "interval_start_minutes and interval_end_minutes must be supplied together."
            )
        if starts is None:
            # Boundary fields are absolute minutes, even when callers omit
            # explicit metadata.  Preserve the declared nominal interval
            # length instead of silently treating each step as one minute.
            start_values = tuple(index * nominal_minutes for index in range(expected_length))
            end_values = tuple((index + 1) * nominal_minutes for index in range(expected_length))
        else:
            if isinstance(starts, (str, bytes)) or not isinstance(starts, Sequence):
                raise SignalValidationError(
                    "interval_start_minutes must be a sequence of integers."
                )
            if isinstance(ends, (str, bytes)) or not isinstance(ends, Sequence):
                raise SignalValidationError("interval_end_minutes must be a sequence of integers.")
            start_values = tuple(starts)
            end_values = tuple(ends)
            if len(start_values) != expected_length or len(end_values) != expected_length:
                raise SignalValidationError(
                    "interval boundaries must have the same length as price_per_kwh."
                )
            for index, (start, end) in enumerate(zip(start_values, end_values, strict=True)):
                if isinstance(start, bool) or not isinstance(start, int):
                    raise SignalValidationError(
                        f"interval_start_minutes[{index}] must be an integer."
                    )
                if isinstance(end, bool) or not isinstance(end, int):
                    raise SignalValidationError(
                        f"interval_end_minutes[{index}] must be an integer."
                    )
                if end <= start:
                    raise SignalValidationError(f"interval {index} must have positive duration.")
                if index and start != end_values[index - 1]:
                    raise SignalValidationError("interval boundaries must be continuous.")

        if base_load is not None:
            if len(base_load) != expected_length:
                raise SignalValidationError(
                    "base_load_kw must have the same length as price_per_kwh."
                )
            for index, load in enumerate(base_load):
                _require_nonnegative(
                    f"base_load_kw[{index}]", load, error_type=SignalValidationError
                )
        if temperature is not None:
            if len(temperature) != expected_length:
                raise SignalValidationError(
                    "battery_temperature_c must have the same length as price_per_kwh."
                )
            for index, value in enumerate(temperature):
                if value <= -273.15:
                    raise SignalValidationError(
                        f"battery_temperature_c[{index}] must be above absolute zero."
                    )

        object.__setattr__(self, "price_per_kwh", price)
        object.__setattr__(self, "base_load_kw", base_load)
        object.__setattr__(self, "battery_temperature_c", temperature)
        object.__setattr__(self, "interval_duration_hours", durations)
        object.__setattr__(self, "interval_start_minutes", start_values)
        object.__setattr__(self, "interval_end_minutes", end_values)
        object.__setattr__(self, "nominal_timestep_minutes", nominal_minutes)

    @property
    def number_of_steps(self) -> int:
        """Number of charging intervals in the planning horizon."""
        return len(self.price_per_kwh)

    @property
    def interval_duration_minutes(self) -> tuple[float, ...]:
        """Interval durations in minutes, aligned with prices and temperatures."""
        return tuple(hours * 60.0 for hours in self.interval_durations)

    @property
    def interval_durations(self) -> tuple[float, ...]:
        """Canonical non-optional interval durations for scientific consumers."""
        durations = self.interval_duration_hours
        if durations is None:  # pragma: no cover - post-init always canonicalizes it
            raise SignalValidationError("interval durations were not initialized.")
        return durations

    def validate_session_window(self, session: ChargingSession) -> None:
        """Require the complete half-open session window to fit this horizon."""
        if session.departure_step > self.number_of_steps:
            raise SignalValidationError(
                "departure_step exceeds the available planning-signal horizon."
            )


@dataclass(frozen=True, slots=True)
class TargetOption:
    """A target SOC and the semantic sources that produced it.

    Multiple sources preserve provenance after target merging, for example
    ``("minimum_required", "standard_80")``. Commute and buffer data belong
    exclusively to :class:`ChargingSession`.
    """

    target_soc: float
    sources: tuple[TargetSource, ...]
    label: str

    def __post_init__(self) -> None:
        _require_finite("target_soc", self.target_soc)
        if not 0.0 <= self.target_soc <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        object.__setattr__(self, "sources", _freeze_target_sources("sources", self.sources))
        object.__setattr__(self, "label", _canonical_text("label", self.label))


@dataclass(frozen=True, slots=True)
class ChargingProfile:
    """A locally valid, time-anchored charging-trajectory representation.

    ``start_step`` locates the first interval on a later planning signal. For
    ``N`` intervals, power and grid energy have ``N`` entries; battery energy
    and SOC contain boundary states and have ``N + 1`` entries. A no-charge
    option uses a nonempty, time-aligned all-zero power and grid-energy profile.

    Charger limits, battery capacity, energy recursion, SOC consistency,
    planning-signal alignment, and ready-time restrictions require a later
    cross-object physical-trajectory validator and are intentionally not
    checked here.
    """

    start_step: int
    grid_energy_kwh: tuple[float, ...]
    battery_energy_kwh: tuple[float, ...]
    power_kw: tuple[float, ...]
    soc: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_step("start_step", self.start_step)
        if self.start_step < 0:
            raise SchemaValidationError("start_step must be non-negative.")

        grid_energy = _freeze_numeric_tuple("grid_energy_kwh", self.grid_energy_kwh)
        battery_energy = _freeze_numeric_tuple("battery_energy_kwh", self.battery_energy_kwh)
        power = _freeze_numeric_tuple("power_kw", self.power_kw)
        soc = _freeze_numeric_tuple("soc", self.soc)
        object.__setattr__(self, "grid_energy_kwh", grid_energy)
        object.__setattr__(self, "battery_energy_kwh", battery_energy)
        object.__setattr__(self, "power_kw", power)
        object.__setattr__(self, "soc", soc)

        number_of_steps = len(power)
        if number_of_steps == 0:
            raise SchemaValidationError("ChargingProfile must contain at least one interval.")
        if len(grid_energy) != number_of_steps:
            raise SchemaValidationError("grid_energy_kwh and power_kw must have identical lengths.")
        if len(battery_energy) != number_of_steps + 1:
            raise SchemaValidationError(
                "battery_energy_kwh must contain one more entry than power_kw."
            )
        if len(soc) != number_of_steps + 1:
            raise SchemaValidationError("soc must contain one more entry than power_kw.")

        for index, value in enumerate(grid_energy):
            if value < 0.0:
                raise PhysicalConstraintError(f"grid_energy_kwh[{index}] must be non-negative.")
        for index, value in enumerate(power):
            if value < 0.0:
                raise PhysicalConstraintError(f"power_kw[{index}] must be non-negative.")
        for index, value in enumerate(battery_energy):
            if value < 0.0:
                raise PhysicalConstraintError(f"battery_energy_kwh[{index}] must be non-negative.")
        for index, value in enumerate(soc):
            if not 0.0 <= value <= 1.0:
                raise PhysicalConstraintError(f"soc[{index}] must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class MenuOffer:
    """One customer-facing offer and its embedded charging profile.

    Costs and ``advertised_saving`` may be negative because planning prices may
    be negative. A later cross-object validator must verify that
    ``advertised_saving`` approximately equals ``same_target_bau_cost -
    charging_cost``.
    """

    offer_id: str
    ev_id: str
    target_sources: tuple[TargetSource, ...]
    ready_step: int
    target_soc: float
    charging_cost: float
    same_target_bau_cost: float
    advertised_saving: float
    incremental_degradation: float
    annualized_degradation_pct: float
    charging_health_score: float
    profile: ChargingProfile
    # ``charging_health_score`` is retained as a compatibility field for
    # earlier commits.  Scientific consumers must use the absolute stress
    # value below; it is never normalized against the current menu.
    raw_battery_stress: float | None = None
    estimated_capacity_loss: float | None = None
    estimated_rul_years: float | None = None
    requested_saving: float | None = None
    saving_band_lower: float | None = None
    saving_band_upper: float | None = None
    saving_band_violation: float = 0.0
    battery_metric_model_id: str = "semi_empirical_total_fade_v1"
    battery_metric_comparison_scope: str = "same EV model and degradation parameterization"
    # Canonical degradation provenance and decomposition.  The two
    # ``battery_metric_*`` fields above are retained as compatibility aliases
    # for earlier commits; scientific consumers should use these fields.
    degradation_model_id: str | None = None
    degradation_model_version: str = "1"
    parameter_set_id: str | None = None
    parameter_status: str = "legacy_compatibility"
    battery_chemistry: Chemistry | None = None
    calendar_capacity_fade: float | None = None
    cycle_capacity_fade: float | None = None
    total_capacity_fade: float | None = None
    capacity_fade_percent: float | None = None
    calendar_fade_connected_window: float | None = None
    calendar_fade_parked_period: float | None = None
    battery_temperature_c: float | None = None
    battery_age_years: float | None = None
    accumulated_fec_at_start: float | None = None
    effective_fec_at_start: float | None = None
    minimum_effective_fec: float | None = None
    fec_regularization_applied: bool = False
    fec_regularization_source: str = "DegradationSettings.minimum_effective_fec"
    parked_period_hours: float | None = None
    parked_period_source: str = "deferred_not_modeled"
    calendar_scope: str = "connected_window_only"
    temperature_source: str = "planning_signal_or_assumed_default"
    battery_age_source: str = "degradation_settings"
    accumulated_fec_source: str = "degradation_settings"
    provenance_flags: tuple[str, ...] = ()
    ready_boundary_absolute_minute: int | None = None
    target_battery_energy_kwh: float | None = None
    scientific_role: str | None = None
    # These flags describe derived customer-facing pipeline stages.  They are
    # deliberately false on the immutable rich generated snapshot and are set
    # only on derived copies returned by compaction/Pareto stages.
    is_pareto_efficient: bool = False
    is_preserved_bau_anchor: bool = False
    is_selected_bau_anchor: bool = False
    is_selected_maximum_saving: bool = False
    is_selected_least_degradation: bool = False
    is_selected_intermediate: bool = False
    selection_reason: str | None = None
    selection_rank_within_target: int | None = None
    diversity_score: float | None = None

    @property
    def battery_stress(self) -> float:
        """Deprecated compatibility alias for ``total_capacity_fade``."""
        return float(
            self.total_capacity_fade
            if self.total_capacity_fade is not None
            else self.raw_battery_stress or 0.0
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "offer_id", _canonical_text("offer_id", self.offer_id))
        object.__setattr__(self, "ev_id", _canonical_text("ev_id", self.ev_id))
        object.__setattr__(
            self,
            "target_sources",
            _freeze_target_sources("target_sources", self.target_sources),
        )
        _require_step("ready_step", self.ready_step)
        if self.ready_step < 0:
            raise SchemaValidationError("ready_step must be non-negative.")
        _require_finite("target_soc", self.target_soc)
        if not 0.0 <= self.target_soc <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        for name, value in (
            ("charging_cost", self.charging_cost),
            ("same_target_bau_cost", self.same_target_bau_cost),
            ("advertised_saving", self.advertised_saving),
        ):
            _require_finite(name, value)
        _require_nonnegative("incremental_degradation", self.incremental_degradation)
        _require_nonnegative("annualized_degradation_pct", self.annualized_degradation_pct)
        _require_finite("charging_health_score", self.charging_health_score)
        if not 0.0 <= self.charging_health_score <= 100.0:
            raise SchemaValidationError("charging_health_score must lie in [0, 100].")
        if not isinstance(self.profile, ChargingProfile):
            raise SchemaValidationError("profile must be a ChargingProfile instance.")
        stress = (
            self.incremental_degradation
            if self.raw_battery_stress is None
            else self.raw_battery_stress
        )
        _require_nonnegative("raw_battery_stress", stress)
        object.__setattr__(self, "raw_battery_stress", float(stress))
        optional_values: tuple[tuple[str, float | None], ...] = (
            ("estimated_capacity_loss", self.estimated_capacity_loss),
            ("estimated_rul_years", self.estimated_rul_years),
            ("requested_saving", self.requested_saving),
            ("saving_band_lower", self.saving_band_lower),
            ("saving_band_upper", self.saving_band_upper),
        )
        for optional_name, optional_value in optional_values:
            name = optional_name
            numeric_value: float | None = optional_value
            if numeric_value is not None:
                _require_finite(name, numeric_value)
                if name != "requested_saving" and numeric_value < 0.0:
                    raise PhysicalConstraintError(f"{name} must be non-negative.")
        _require_nonnegative("saving_band_violation", self.saving_band_violation)
        if (self.saving_band_lower is None) != (self.saving_band_upper is None):
            raise SchemaValidationError(
                "saving_band_lower and saving_band_upper must be supplied together."
            )
        if (
            self.saving_band_lower is not None
            and self.saving_band_upper is not None
            and self.saving_band_lower > self.saving_band_upper
        ):
            raise SchemaValidationError("saving band lower bound cannot exceed upper bound.")
        for metadata_name, metadata_value in (
            ("battery_metric_model_id", self.battery_metric_model_id),
            ("battery_metric_comparison_scope", self.battery_metric_comparison_scope),
            ("degradation_model_version", self.degradation_model_version),
            ("parameter_status", self.parameter_status),
            ("calendar_scope", self.calendar_scope),
            ("temperature_source", self.temperature_source),
            ("battery_age_source", self.battery_age_source),
            ("accumulated_fec_source", self.accumulated_fec_source),
            ("fec_regularization_source", self.fec_regularization_source),
            ("parked_period_source", self.parked_period_source),
        ):
            if not isinstance(metadata_value, str) or not metadata_value.strip():
                raise SchemaValidationError(f"{metadata_name} must be a non-empty string.")
            object.__setattr__(self, metadata_name, metadata_value.strip())
        model_id = (
            self.battery_metric_model_id
            if self.degradation_model_id is None
            else self.degradation_model_id
        )
        if not isinstance(model_id, str) or not model_id.strip():
            raise SchemaValidationError("degradation_model_id must be a non-empty string.")
        object.__setattr__(self, "degradation_model_id", model_id.strip())
        if not isinstance(self.fec_regularization_applied, bool):
            raise SchemaValidationError("fec_regularization_applied must be bool.")
        if self.parameter_set_id is not None:
            if not isinstance(self.parameter_set_id, str) or not self.parameter_set_id.strip():
                raise SchemaValidationError("parameter_set_id must be non-empty when supplied.")
            object.__setattr__(self, "parameter_set_id", self.parameter_set_id.strip())
        if self.battery_chemistry is not None and self.battery_chemistry not in ("LFP", "NMC"):
            raise SchemaValidationError("battery_chemistry must be 'LFP' or 'NMC'.")
        optional_fade_values: tuple[tuple[str, float | None], ...] = (
            ("calendar_capacity_fade", self.calendar_capacity_fade),
            ("cycle_capacity_fade", self.cycle_capacity_fade),
            ("total_capacity_fade", self.total_capacity_fade),
            ("capacity_fade_percent", self.capacity_fade_percent),
            ("calendar_fade_connected_window", self.calendar_fade_connected_window),
            ("calendar_fade_parked_period", self.calendar_fade_parked_period),
            ("battery_temperature_c", self.battery_temperature_c),
            ("battery_age_years", self.battery_age_years),
            ("accumulated_fec_at_start", self.accumulated_fec_at_start),
            ("effective_fec_at_start", self.effective_fec_at_start),
            ("minimum_effective_fec", self.minimum_effective_fec),
            ("parked_period_hours", self.parked_period_hours),
        )
        for fade_name, fade_value in optional_fade_values:
            if fade_value is not None:
                _require_finite(fade_name, fade_value)
                if fade_name not in ("battery_temperature_c",) and fade_value < 0.0:
                    raise PhysicalConstraintError(f"{fade_name} must be non-negative.")
        total_fade = (
            self.incremental_degradation
            if self.total_capacity_fade is None
            else self.total_capacity_fade
        )
        object.__setattr__(self, "total_capacity_fade", float(total_fade))
        object.__setattr__(self, "capacity_fade_percent", float(total_fade * 100.0))
        if (
            self.calendar_fade_parked_period is not None
            and self.calendar_fade_connected_window is not None
        ):
            calendar = self.calendar_fade_connected_window + self.calendar_fade_parked_period
            if (
                self.calendar_capacity_fade is not None
                and abs(calendar - self.calendar_capacity_fade) > 1e-12
            ):
                raise PhysicalConstraintError("calendar_capacity_fade must equal its components.")
            object.__setattr__(self, "calendar_capacity_fade", float(calendar))
        if (
            self.cycle_capacity_fade is not None
            and self.calendar_capacity_fade is not None
            and abs(self.calendar_capacity_fade + self.cycle_capacity_fade - total_fade) > 1e-12
        ):
            raise PhysicalConstraintError(
                "total_capacity_fade must equal calendar plus cycle fade."
            )
        flags = tuple(self.provenance_flags)
        if any(not isinstance(flag, str) or not flag.strip() for flag in flags):
            raise SchemaValidationError("provenance_flags must contain non-empty strings.")
        if len(set(flags)) != len(flags):
            raise SchemaValidationError("provenance_flags must be unique.")
        object.__setattr__(self, "provenance_flags", tuple(sorted(flags)))
        if self.ready_boundary_absolute_minute is not None and (
            isinstance(self.ready_boundary_absolute_minute, bool)
            or not isinstance(self.ready_boundary_absolute_minute, int)
        ):
            raise SchemaValidationError("ready_boundary_absolute_minute must be an integer.")
        target_energy = (
            self.target_soc * 0.0
            if self.target_battery_energy_kwh is None
            else self.target_battery_energy_kwh
        )
        if self.target_battery_energy_kwh is not None:
            _require_nonnegative("target_battery_energy_kwh", target_energy)
        if self.scientific_role is not None:
            if not isinstance(self.scientific_role, str) or not self.scientific_role.strip():
                raise SchemaValidationError("scientific_role must be non-empty when supplied.")
            object.__setattr__(self, "scientific_role", self.scientific_role.strip())
        for name, value in (
            ("is_pareto_efficient", self.is_pareto_efficient),
            ("is_preserved_bau_anchor", self.is_preserved_bau_anchor),
            ("is_selected_bau_anchor", self.is_selected_bau_anchor),
            ("is_selected_maximum_saving", self.is_selected_maximum_saving),
            ("is_selected_least_degradation", self.is_selected_least_degradation),
            ("is_selected_intermediate", self.is_selected_intermediate),
        ):
            if not isinstance(value, bool):
                raise SchemaValidationError(f"{name} must be bool.")
        if self.selection_reason is not None:
            if not isinstance(self.selection_reason, str) or not self.selection_reason.strip():
                raise SchemaValidationError("selection_reason must be non-empty when supplied.")
            allowed_reasons = {
                "bau_anchor",
                "target_maximum_saving",
                "target_least_degradation",
                "intermediate_diversity",
                "close_ready_saving_stress_tradeoff",
                "target_positive_saving_coverage",
                "per_target_limit",
                "global_limit",
                "too_close_in_ready_time",
                "insufficient_saving_difference",
                "insufficient_stress_difference",
                "lower_diversity_contribution",
                "target_coverage_anchor",
                "maximum_saving_anchor",
                "least_degradation_anchor",
                "farthest_point_diversity",
                "anchor_suppressed_similarity",
            }
            if self.selection_reason not in allowed_reasons:
                raise SchemaValidationError("unsupported selection_reason.")
            object.__setattr__(self, "selection_reason", self.selection_reason.strip())
        if self.selection_rank_within_target is not None and (
            isinstance(self.selection_rank_within_target, bool)
            or not isinstance(self.selection_rank_within_target, int)
            or self.selection_rank_within_target < 1
        ):
            raise SchemaValidationError(
                "selection_rank_within_target must be a positive integer when supplied."
            )
        if self.diversity_score is not None:
            _require_finite("diversity_score", self.diversity_score)
            if self.diversity_score < 0.0:
                raise PhysicalConstraintError("diversity_score must be non-negative.")


@dataclass(frozen=True, slots=True)
class MenuSettings:
    """Configuration values controlling later deterministic menu construction."""

    standard_targets: tuple[float, ...] = (0.80, 0.90, 1.00)
    target_merge_tolerance: float = 0.01
    numerical_tolerance: float = 1e-8
    equivalent_sessions_per_year: int = 300
    reference_degradation_pct: float = 2.0
    # Algorithm 1 rich-generation controls.  Keeping these on the immutable
    # scenario object gives every layer one source of truth for the defaults.
    saving_step: float = 5.0
    maximum_saving_levels_per_request: int = 5
    saving_band_tolerance: float = 0.50
    saving_zero_tolerance: float | None = None

    def __post_init__(self) -> None:
        targets = _freeze_numeric_tuple("standard_targets", self.standard_targets)
        object.__setattr__(self, "standard_targets", targets)
        if not targets:
            raise SchemaValidationError("standard_targets cannot be empty.")
        previous = -1.0
        for index, target in enumerate(targets):
            if not 0.0 < target <= 1.0:
                raise PhysicalConstraintError(f"standard_targets[{index}] must lie in (0, 1].")
            if target <= previous:
                raise SchemaValidationError(
                    "standard_targets must be strictly increasing and unique."
                )
            previous = target
        _require_nonnegative("target_merge_tolerance", self.target_merge_tolerance)
        if self.target_merge_tolerance >= 1.0:
            raise SchemaValidationError("target_merge_tolerance must lie in [0, 1).")
        _require_positive("numerical_tolerance", self.numerical_tolerance)
        _require_step("equivalent_sessions_per_year", self.equivalent_sessions_per_year)
        if self.equivalent_sessions_per_year <= 0:
            raise SchemaValidationError("equivalent_sessions_per_year must be positive.")
        _require_positive("reference_degradation_pct", self.reference_degradation_pct)
        _require_positive("saving_step", self.saving_step)
        if isinstance(self.maximum_saving_levels_per_request, bool) or not isinstance(
            self.maximum_saving_levels_per_request, int
        ):
            raise SchemaValidationError("maximum_saving_levels_per_request must be an integer.")
        if self.maximum_saving_levels_per_request < 1:
            raise PhysicalConstraintError("maximum_saving_levels_per_request must be positive.")
        _require_nonnegative("saving_band_tolerance", self.saving_band_tolerance)
        zero = (
            self.numerical_tolerance
            if self.saving_zero_tolerance is None
            else self.saving_zero_tolerance
        )
        _require_nonnegative("saving_zero_tolerance", zero)
        object.__setattr__(self, "saving_zero_tolerance", float(zero))
