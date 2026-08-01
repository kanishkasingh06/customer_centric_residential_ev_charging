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
    frozen = tuple([*values])
    for index, value in enumerate(frozen):
        _require_finite(f"{name}[{index}]", value, error_type=error_type)
    return cast(tuple[float, ...], frozen)


def _freeze_target_sources(name: str, values: object) -> tuple[TargetSource, ...]:
    """Copy, validate, deduplicate-check, and canonically order target sources."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SchemaValidationError(f"{name} must be a sequence of target sources.")
    frozen = tuple([*values])
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
            raise PhysicalConstraintError(
                "initial_energy_kwh exceeds the usable battery capacity."
            )
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

    def __post_init__(self) -> None:
        _require_positive(
            "timestep_hours", self.timestep_hours, error_type=SignalValidationError
        )
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

    @property
    def number_of_steps(self) -> int:
        """Number of charging intervals in the planning horizon."""
        return len(self.price_per_kwh)

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
        battery_energy = _freeze_numeric_tuple(
            "battery_energy_kwh", self.battery_energy_kwh
        )
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
            raise SchemaValidationError(
                "grid_energy_kwh and power_kw must have identical lengths."
            )
        if len(battery_energy) != number_of_steps + 1:
            raise SchemaValidationError(
                "battery_energy_kwh must contain one more entry than power_kw."
            )
        if len(soc) != number_of_steps + 1:
            raise SchemaValidationError("soc must contain one more entry than power_kw.")

        for index, value in enumerate(grid_energy):
            if value < 0.0:
                raise PhysicalConstraintError(
                    f"grid_energy_kwh[{index}] must be non-negative."
                )
        for index, value in enumerate(power):
            if value < 0.0:
                raise PhysicalConstraintError(f"power_kw[{index}] must be non-negative.")
        for index, value in enumerate(battery_energy):
            if value < 0.0:
                raise PhysicalConstraintError(
                    f"battery_energy_kwh[{index}] must be non-negative."
                )
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


@dataclass(frozen=True, slots=True)
class MenuSettings:
    """Configuration values controlling later deterministic menu construction."""

    standard_targets: tuple[float, ...] = (0.80, 0.90, 1.00)
    target_merge_tolerance: float = 0.01
    numerical_tolerance: float = 1e-8
    equivalent_sessions_per_year: int = 300
    reference_degradation_pct: float = 2.0

    def __post_init__(self) -> None:
        targets = _freeze_numeric_tuple("standard_targets", self.standard_targets)
        object.__setattr__(self, "standard_targets", targets)
        if not targets:
            raise SchemaValidationError("standard_targets cannot be empty.")
        previous = -1.0
        for index, target in enumerate(targets):
            if not 0.0 < target <= 1.0:
                raise PhysicalConstraintError(
                    f"standard_targets[{index}] must lie in (0, 1]."
                )
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
