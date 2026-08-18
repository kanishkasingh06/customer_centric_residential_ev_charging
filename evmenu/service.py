"""High-level single-EV menu generation from user-facing inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from math import isclose, isfinite
from numbers import Real
from typing import Literal, cast

from .assembly import (
    AssembledMenu,
    MenuAssemblySettings,
    MenuStage,
    OfferSource,
    PipelineDiagnostics,
    TargetGenerationSummary,
    assemble_customer_menu,
)
from .catalog import EVModel, get_ev_model
from .degradation import PARKED_PERIOD_USER_SOURCE, DegradationSettings
from .exceptions import PhysicalConstraintError, SchemaValidationError
from .menu import MenuGenerationSettings, RequestGenerationDiagnostics, generate_candidate_menu
from .optimization import (
    DegradationObjectiveDiagnostics,
    FrontierSettings,
    OptimizationDiagnostics,
    SavingLevelFailure,
)
from .pricing import (
    TimestampedPriceProfile,
    WeeklyPriceProfile,
)
from .schemas import ChargingSession, MenuOffer, MenuSettings, PlanningSignal
from .timegrid import build_time_intervals, recurring_daily_boundaries
from .validation import ValidationTolerances

TariffName = Literal["research_tou", "flat", "custom"]
_CUSTOMER_ROLES = frozenset(
    {
        "bau",
        "low_saving",
        "least_degradation",
        "intermediate",
        "maximum_saving",
        "least_and_maximum",
    }
)
_ROLE_ORDER = {
    "bau": 0,
    "low_saving": 1,
    "intermediate": 2,
    "least_degradation": 3,
    "maximum_saving": 4,
    "least_and_maximum": 5,
}


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)


def _parse_clock(name: str, value: object) -> int:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a HH:MM string.")
    if (
        len(value) != 5
        or value[2] != ":"
        or not all("0" <= character <= "9" for character in value[:2] + value[3:])
    ):
        raise SchemaValidationError(f"{name} must use strict HH:MM format.")
    hour = int(value[:2])
    minute = int(value[3:])
    if hour > 23 or minute > 59:
        raise SchemaValidationError(f"{name} must use 24-hour HH:MM format.")
    return hour * 60 + minute


def _format_clock(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def _currency_label(value: object) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError("currency_label must be a string.")
    label = value.strip()
    if not label:
        raise SchemaValidationError("currency_label must be non-empty.")
    if len(label) > 64 or any(ord(character) < 32 or ord(character) == 127 for character in label):
        raise SchemaValidationError(
            "currency_label must be at most 64 characters without control characters."
        )
    return label


def _metadata_values(name: str, value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{name} must be a sequence.")
    return tuple(value)


def _tariff_price(name: TariffName, minute_of_day: int, flat_price: float) -> float:
    if name == "flat":
        return flat_price
    # Illustrative research TOU, not a regulated retail tariff.
    if minute_of_day < 6 * 60:
        return 4.0
    if minute_of_day < 17 * 60:
        return 7.0
    if minute_of_day < 23 * 60:
        return 10.0
    return 5.0


@dataclass(frozen=True, slots=True)
class CustomerMenuRow:
    """Serializable customer-facing view of one assembled offer."""

    offer_id: str
    ready_time: str
    target_soc_percent: float
    charging_cost: float
    saving: float
    health_score: float
    energy_drawn_kwh: float
    role: str
    charging_schedule_kw: tuple[float, ...]
    requested_saving: float | None = None
    actual_saving: float | None = None
    actual_cost: float | None = None
    raw_battery_stress: float | None = None
    estimated_capacity_loss: float | None = None
    estimated_rul_years: float | None = None
    provenance_flags: tuple[str, ...] = ()
    ready_boundary_absolute_minute: int | None = None
    target_battery_energy_kwh: float | None = None
    same_target_bau_cost: float | None = None
    saving_band_lower: float | None = None
    saving_band_upper: float | None = None
    saving_band_violation: float = 0.0
    battery_metric_model_id: str = "semi_empirical_total_fade_v1"
    battery_metric_comparison_scope: str = "same EV model and degradation parameterization"
    degradation_model_id: str | None = None
    degradation_model_version: str = "1"
    parameter_set_id: str | None = None
    parameter_status: str = "legacy_compatibility"
    battery_chemistry: str | None = None
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

    @property
    def battery_stress(self) -> float:
        """Deprecated compatibility alias for ``total_capacity_fade``."""
        return float(
            self.total_capacity_fade
            if self.total_capacity_fade is not None
            else self.raw_battery_stress or 0.0
        )

    is_pareto_efficient: bool = False
    is_preserved_bau_anchor: bool = False
    is_selected_bau_anchor: bool = False
    is_selected_maximum_saving: bool = False
    is_selected_least_degradation: bool = False
    is_selected_intermediate: bool = False
    selection_reason: str | None = None
    selection_rank_within_target: int | None = None
    diversity_score: float | None = None

    def __post_init__(self) -> None:
        for name in ("offer_id", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        _parse_clock("ready_time", self.ready_time)
        if self.role not in _CUSTOMER_ROLES:
            raise SchemaValidationError(f"role must be one of {sorted(_CUSTOMER_ROLES)}.")
        for name in (
            "target_soc_percent",
            "charging_cost",
            "saving",
            "health_score",
            "energy_drawn_kwh",
        ):
            _finite(name, getattr(self, name))
        if not 0.0 <= self.target_soc_percent <= 100.0:
            raise PhysicalConstraintError("target_soc_percent must lie in [0, 100].")
        if not 0.0 <= self.health_score <= 100.0:
            raise PhysicalConstraintError("health_score must lie in [0, 100].")
        if self.energy_drawn_kwh < 0.0:
            raise PhysicalConstraintError("energy_drawn_kwh must be non-negative.")
        if self.actual_saving is None:
            object.__setattr__(self, "actual_saving", self.saving)
        if self.actual_cost is None:
            object.__setattr__(self, "actual_cost", self.charging_cost)
        if self.raw_battery_stress is None:
            object.__setattr__(self, "raw_battery_stress", self.health_score)
        for name, value in (
            ("requested_saving", self.requested_saving),
            ("actual_saving", self.actual_saving),
            ("actual_cost", self.actual_cost),
            ("raw_battery_stress", self.raw_battery_stress),
            ("estimated_capacity_loss", self.estimated_capacity_loss),
            ("estimated_rul_years", self.estimated_rul_years),
            ("target_battery_energy_kwh", self.target_battery_energy_kwh),
            ("same_target_bau_cost", self.same_target_bau_cost),
            ("saving_band_lower", self.saving_band_lower),
            ("saving_band_upper", self.saving_band_upper),
        ):
            if value is not None:
                _finite(name, value)
                if (
                    name
                    in (
                        "raw_battery_stress",
                        "estimated_capacity_loss",
                        "estimated_rul_years",
                        "target_battery_energy_kwh",
                    )
                    and value < 0.0
                ):
                    raise PhysicalConstraintError(f"{name} must be non-negative.")
        _finite("saving_band_violation", self.saving_band_violation)
        if self.saving_band_violation < 0.0:
            raise PhysicalConstraintError("saving_band_violation must be non-negative.")
        if (self.saving_band_lower is None) != (self.saving_band_upper is None):
            raise SchemaValidationError("saving band bounds must be supplied together.")
        for name, value in (
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
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        model_id = (
            self.battery_metric_model_id
            if self.degradation_model_id is None
            else self.degradation_model_id
        )
        if not isinstance(model_id, str) or not model_id.strip():
            raise SchemaValidationError("degradation_model_id must be non-empty.")
        object.__setattr__(self, "degradation_model_id", model_id.strip())
        if not isinstance(self.fec_regularization_applied, bool):
            raise SchemaValidationError("fec_regularization_applied must be bool.")
        if self.parameter_set_id is not None:
            if not isinstance(self.parameter_set_id, str) or not self.parameter_set_id.strip():
                raise SchemaValidationError("parameter_set_id must be non-empty when supplied.")
            object.__setattr__(self, "parameter_set_id", self.parameter_set_id.strip())
        if self.battery_chemistry is not None and self.battery_chemistry not in ("LFP", "NMC"):
            raise SchemaValidationError("battery_chemistry must be LFP or NMC.")
        total_fade = (
            self.raw_battery_stress
            if self.total_capacity_fade is None
            else self.total_capacity_fade
        )
        if total_fade is None:
            total_fade = 0.0
        object.__setattr__(self, "total_capacity_fade", float(total_fade))
        object.__setattr__(self, "capacity_fade_percent", float(total_fade * 100.0))
        for name, value in (
            ("calendar_capacity_fade", self.calendar_capacity_fade),
            ("cycle_capacity_fade", self.cycle_capacity_fade),
            ("calendar_fade_connected_window", self.calendar_fade_connected_window),
            ("calendar_fade_parked_period", self.calendar_fade_parked_period),
            ("battery_temperature_c", self.battery_temperature_c),
            ("battery_age_years", self.battery_age_years),
            ("accumulated_fec_at_start", self.accumulated_fec_at_start),
            ("effective_fec_at_start", self.effective_fec_at_start),
            ("minimum_effective_fec", self.minimum_effective_fec),
            ("parked_period_hours", self.parked_period_hours),
        ):
            if value is not None:
                _finite(name, value)
                if name != "battery_temperature_c" and value < 0.0:
                    raise PhysicalConstraintError(f"{name} must be non-negative.")
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
        if self.selection_reason is not None and (
            not isinstance(self.selection_reason, str) or not self.selection_reason.strip()
        ):
            raise SchemaValidationError("selection_reason must be non-empty when supplied.")
        if self.selection_rank_within_target is not None and (
            isinstance(self.selection_rank_within_target, bool)
            or not isinstance(self.selection_rank_within_target, int)
            or self.selection_rank_within_target < 1
        ):
            raise SchemaValidationError("selection_rank_within_target must be positive.")
        if self.diversity_score is not None:
            _finite("diversity_score", self.diversity_score)
            if self.diversity_score < 0.0:
                raise PhysicalConstraintError("diversity_score must be non-negative.")
        flags = tuple(self.provenance_flags)
        if any(not isinstance(flag, str) or not flag.strip() for flag in flags):
            raise SchemaValidationError("provenance_flags must contain non-empty strings.")
        object.__setattr__(self, "provenance_flags", tuple(sorted(set(flags))))
        if self.ready_boundary_absolute_minute is not None and (
            isinstance(self.ready_boundary_absolute_minute, bool)
            or not isinstance(self.ready_boundary_absolute_minute, int)
        ):
            raise SchemaValidationError("ready_boundary_absolute_minute must be an integer.")
        if self.charging_schedule_kw is None or not isinstance(self.charging_schedule_kw, tuple):
            raise SchemaValidationError("charging_schedule_kw must be a nonempty tuple.")
        if not self.charging_schedule_kw:
            raise SchemaValidationError("charging_schedule_kw must be nonempty.")
        for index, value in enumerate(self.charging_schedule_kw):
            power = _finite(f"charging_schedule_kw[{index}]", value)
            if power < 0.0:
                raise PhysicalConstraintError("charging_schedule_kw cannot contain negatives.")


@dataclass(frozen=True, slots=True)
class GeneratedCustomerMenu:
    """High-level deterministic result and its auditable core objects."""

    ev_model: EVModel
    arrival_time: str
    departure_time: str
    timestep_minutes: int
    current_soc: float
    next_trip_distance_km: float
    tariff_name: str
    tariff_is_illustrative: bool
    offers: tuple[CustomerMenuRow, ...]
    assembled_menu: AssembledMenu
    profile_id: str | None = None
    currency_label: str = "currency"
    arrival_day: str | None = None
    arrival_date: str | None = None
    interval_start_minutes: tuple[int, ...] = ()
    interval_end_minutes: tuple[int, ...] = ()
    interval_duration_minutes: tuple[int, ...] = ()
    interval_price_per_kwh: tuple[float, ...] = ()
    generated_offers: tuple[CustomerMenuRow, ...] = ()
    retained_offers: tuple[CustomerMenuRow, ...] = ()
    compacted_offers: tuple[CustomerMenuRow, ...] = ()
    pareto_efficient_offers: tuple[CustomerMenuRow, ...] = ()
    preserved_bau_anchors: tuple[CustomerMenuRow, ...] = ()
    bau_reference_offers: tuple[CustomerMenuRow, ...] = ()
    pareto_stage_offers: tuple[CustomerMenuRow, ...] = ()
    displayed_stage_offers: tuple[CustomerMenuRow, ...] = ()
    menu_stage: MenuStage = "displayed"
    pipeline_diagnostics: PipelineDiagnostics = field(default_factory=PipelineDiagnostics)
    raw_offer_count: int = 0
    scientific_duplicate_count: int = 0
    generated_offer_count: int = 0
    request_diagnostics: RequestGenerationDiagnostics = field(
        default_factory=RequestGenerationDiagnostics
    )
    optimization_diagnostics: OptimizationDiagnostics = field(
        default_factory=OptimizationDiagnostics
    )
    degradation_objective_diagnostics: DegradationObjectiveDiagnostics | None = None
    target_summaries: tuple[TargetGenerationSummary, ...] = ()
    saving_level_failures: tuple[SavingLevelFailure, ...] = ()

    @property
    def pareto_offers(self) -> tuple[CustomerMenuRow, ...]:
        """Customer-facing Pareto stage including preserved BAU anchors."""
        return self.pareto_stage_offers or self.offers

    @property
    def retained_offer_count(self) -> int:
        return self.pipeline_diagnostics.retained_offer_count

    @property
    def compacted_offer_count(self) -> int:
        return self.pipeline_diagnostics.compacted_offer_count

    @property
    def pareto_efficient_offer_count(self) -> int:
        return self.pipeline_diagnostics.pareto_efficient_offer_count

    @property
    def pareto_offer_count(self) -> int:
        return self.pipeline_diagnostics.pareto_offer_count

    @property
    def displayed_offer_count(self) -> int:
        return self.pipeline_diagnostics.displayed_offer_count

    @property
    def nonpositive_removed_count(self) -> int:
        return self.pipeline_diagnostics.nonpositive_removed_count

    @property
    def compaction_removed_count(self) -> int:
        return self.pipeline_diagnostics.compaction_removed_count

    @property
    def pareto_dominated_removed_count(self) -> int:
        return self.pipeline_diagnostics.pareto_dominated_removed_count

    @property
    def display_selection_removed_count(self) -> int:
        return self.pipeline_diagnostics.display_selection_removed_count

    @property
    def displayed_offers(self) -> tuple[CustomerMenuRow, ...]:
        return self.displayed_stage_offers or self.offers

    @property
    def bau_references(self) -> tuple[CustomerMenuRow, ...]:
        """Scientific BAU baselines, never customer-facing choices."""
        return self.bau_reference_offers or self.preserved_bau_anchors

    def __post_init__(self) -> None:
        if not isinstance(self.ev_model, EVModel):
            raise SchemaValidationError("ev_model must be an EVModel.")
        arrival_minute = _parse_clock("arrival_time", self.arrival_time)
        departure_minute = _parse_clock("departure_time", self.departure_time)
        if arrival_minute == departure_minute:
            raise PhysicalConstraintError("arrival_time and departure_time must differ.")
        if isinstance(self.timestep_minutes, bool) or not isinstance(self.timestep_minutes, int):
            raise SchemaValidationError("timestep_minutes must be an integer.")
        if self.timestep_minutes <= 0 or 1440 % self.timestep_minutes != 0:
            raise PhysicalConstraintError("timestep_minutes must be a positive divisor of 1440.")
        current = _finite("current_soc", self.current_soc)
        distance = _finite("next_trip_distance_km", self.next_trip_distance_km)
        if not 0.0 <= current <= 1.0:
            raise PhysicalConstraintError("current_soc must lie in [0, 1].")
        if distance < 0.0:
            raise PhysicalConstraintError("next_trip_distance_km must be non-negative.")
        if not isinstance(self.tariff_name, str) or self.tariff_name not in (
            "research_tou",
            "flat",
            "custom",
        ):
            raise SchemaValidationError("tariff_name must be 'research_tou', 'flat', or 'custom'.")
        if not isinstance(self.assembled_menu, AssembledMenu):
            raise SchemaValidationError("assembled_menu must be an AssembledMenu.")
        if self.assembled_menu.ev_id != self.ev_model.model_id:
            raise SchemaValidationError("assembled_menu EV does not match ev_model.")
        if not isinstance(self.tariff_is_illustrative, bool):
            raise SchemaValidationError("tariff_is_illustrative must be bool.")
        if self.tariff_is_illustrative != (self.tariff_name == "research_tou"):
            raise SchemaValidationError("tariff_is_illustrative does not match tariff_name.")
        if self.menu_stage not in ("generated", "compacted", "pareto", "displayed"):
            raise SchemaValidationError(
                "menu_stage must be generated, compacted, pareto, or displayed."
            )
        if not isinstance(self.pipeline_diagnostics, PipelineDiagnostics):
            raise SchemaValidationError("pipeline_diagnostics has an invalid type.")
        if self.degradation_objective_diagnostics is not None and not isinstance(
            self.degradation_objective_diagnostics, DegradationObjectiveDiagnostics
        ):
            raise SchemaValidationError("invalid degradation objective diagnostics.")
        if self.offers is None or not isinstance(self.offers, tuple):
            raise SchemaValidationError("offers must be a non-empty tuple.")
        offers = self.offers
        if any(not isinstance(row, CustomerMenuRow) for row in offers):
            raise SchemaValidationError("offers must contain CustomerMenuRow objects.")
        if len(offers) != len(self.assembled_menu.offers):
            raise SchemaValidationError("customer rows must align with assembled offers.")
        row_ids = tuple(row.offer_id for row in offers)
        offer_ids = tuple(offer.offer_id for offer in self.assembled_menu.offers)
        if len(set(row_ids)) != len(row_ids):
            raise SchemaValidationError("customer row IDs must be unique.")
        if set(row_ids) != set(offer_ids):
            raise SchemaValidationError("customer rows are not aligned with assembled offers.")
        rich_rows = self.generated_offers or offers
        if not isinstance(rich_rows, tuple) or not rich_rows:
            raise SchemaValidationError("generated_offers must be a non-empty tuple.")
        rich_ids = tuple(row.offer_id for row in rich_rows)
        assembled_rich_ids = tuple(row.offer_id for row in self.assembled_menu.generated_offers)
        if self.generated_offers and set(rich_ids) != set(assembled_rich_ids):
            raise SchemaValidationError("generated customer rows are not aligned with rich offers.")
        stage_rows = {
            "retained_offers": self.retained_offers or rich_rows,
            "compacted_offers": self.compacted_offers or rich_rows,
            "pareto_efficient_offers": self.pareto_efficient_offers or rich_rows,
            "preserved_bau_anchors": self.preserved_bau_anchors,
            "bau_reference_offers": self.bau_reference_offers or self.preserved_bau_anchors,
            "pareto_stage_offers": self.pareto_stage_offers or offers,
            "displayed_stage_offers": self.displayed_stage_offers or offers,
        }
        stage_offer_sets = {
            "retained_offers": {item.offer_id for item in self.assembled_menu.retained_offers},
            "compacted_offers": {item.offer_id for item in self.assembled_menu.compacted_offers},
            "pareto_efficient_offers": {
                item.offer_id for item in self.assembled_menu.pareto_efficient_offers
            },
            "preserved_bau_anchors": {
                item.offer_id for item in self.assembled_menu.preserved_bau_anchors
            },
            "bau_reference_offers": {
                item.offer_id for item in self.assembled_menu.bau_reference_offers
            },
            "pareto_stage_offers": {item.offer_id for item in self.assembled_menu.pareto_offers},
            "displayed_stage_offers": {
                item.offer_id for item in self.assembled_menu.displayed_offers
            },
        }
        for name, rows_for_stage in stage_rows.items():
            if not isinstance(rows_for_stage, tuple):
                raise SchemaValidationError(f"{name} must be a tuple.")
            if any(not isinstance(row, CustomerMenuRow) for row in rows_for_stage):
                raise SchemaValidationError(f"{name} contains an invalid row.")
            explicitly_supplied = bool(getattr(self, name))
            if (
                explicitly_supplied
                and rows_for_stage
                and {row.offer_id for row in rows_for_stage} != stage_offer_sets[name]
            ):
                raise SchemaValidationError(f"{name} rows are not aligned with assembled stage.")
            object.__setattr__(self, name, rows_for_stage)
        duration_minutes = (departure_minute - arrival_minute) % 1440
        departure_absolute = arrival_minute + duration_minutes
        metadata = (
            self.interval_start_minutes,
            self.interval_end_minutes,
            self.interval_duration_minutes,
        )
        if all(value == () for value in metadata):
            intervals = build_time_intervals(
                arrival_minute=arrival_minute,
                departure_minute=departure_absolute,
                nominal_timestep_minutes=self.timestep_minutes,
            )
            starts = tuple(interval.start_minute for interval in intervals)
            ends = tuple(interval.end_minute for interval in intervals)
            durations = tuple(interval.duration_minutes for interval in intervals)
        else:
            starts_values = _metadata_values("interval_start_minutes", self.interval_start_minutes)
            ends_values = _metadata_values("interval_end_minutes", self.interval_end_minutes)
            durations_values = _metadata_values(
                "interval_duration_minutes", self.interval_duration_minutes
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for values in (starts_values, ends_values, durations_values)
                for value in values
            ):
                raise SchemaValidationError("interval minute metadata must contain integers only.")
            starts = cast(tuple[int, ...], starts_values)
            ends = cast(tuple[int, ...], ends_values)
            durations = cast(tuple[int, ...], durations_values)
            if not starts or len(starts) != len(ends) or len(starts) != len(durations):
                raise SchemaValidationError("interval metadata must be aligned and nonempty.")
            if starts[0] % 1440 != arrival_minute or ends[-1] - starts[0] != duration_minutes:
                raise SchemaValidationError("interval metadata must preserve exact session bounds.")
            if any(end <= start for start, end in zip(starts, ends, strict=True)):
                raise SchemaValidationError("interval metadata must have positive durations.")
            if any(start != previous for previous, start in zip(ends, starts[1:])):
                raise SchemaValidationError("interval metadata must be continuous.")
            if any(
                end - start != duration
                for start, end, duration in zip(starts, ends, durations, strict=True)
            ):
                raise SchemaValidationError("interval duration metadata is inconsistent.")
        expected_steps = len(starts)
        price_values = _metadata_values("interval_price_per_kwh", self.interval_price_per_kwh)
        prices = tuple(
            _finite(f"interval_price_per_kwh[{index}]", value)
            for index, value in enumerate(price_values)
        )
        if prices and len(prices) != expected_steps:
            raise SchemaValidationError("interval prices must align with interval metadata.")
        if self.profile_id is not None and (
            not isinstance(self.profile_id, str) or not self.profile_id.strip()
        ):
            raise SchemaValidationError("profile_id must be non-empty when supplied.")
        currency_label = _currency_label(self.currency_label)
        offer_by_id = {offer.offer_id: offer for offer in self.assembled_menu.offers}
        source_by_id = {source.offer_id: source for source in self.assembled_menu.source_metadata}
        for row in offers:
            offer = offer_by_id[row.offer_id]
            if (
                len(row.charging_schedule_kw) != expected_steps
                or len(offer.profile.power_kw) != expected_steps
            ):
                raise SchemaValidationError("customer schedules must match session intervals.")
            if offer.ready_step > expected_steps:
                raise SchemaValidationError("offer ready_step exceeds interval boundaries.")
            expected_ready = _format_clock(
                starts[offer.ready_step] if offer.ready_step < expected_steps else ends[-1]
            )
            source = source_by_id[offer.offer_id]
            checks = (
                ("target_soc", row.target_soc_percent / 100.0, offer.target_soc),
                ("charging_cost", row.charging_cost, offer.charging_cost),
                ("saving", row.saving, offer.advertised_saving),
                ("health_score", row.health_score, offer.charging_health_score),
                ("actual_cost", row.actual_cost or 0.0, offer.charging_cost),
                ("actual_saving", row.actual_saving or 0.0, offer.advertised_saving),
                (
                    "same_target_bau_cost",
                    row.same_target_bau_cost or 0.0,
                    offer.same_target_bau_cost,
                ),
                ("saving_band_violation", row.saving_band_violation, offer.saving_band_violation),
                (
                    "raw_battery_stress",
                    row.raw_battery_stress or 0.0,
                    offer.raw_battery_stress or 0.0,
                ),
                ("energy_drawn_kwh", row.energy_drawn_kwh, sum(offer.profile.grid_energy_kwh)),
            )
            for name, observed, expected in checks:
                if not isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9):
                    raise SchemaValidationError(f"customer row {name} does not match offer.")
            if row.is_pareto_efficient != offer.is_pareto_efficient:
                raise SchemaValidationError("customer row Pareto status does not match offer.")
            if row.is_preserved_bau_anchor != offer.is_preserved_bau_anchor:
                raise SchemaValidationError("customer row BAU-anchor status does not match offer.")
            if (
                row.is_selected_bau_anchor != offer.is_selected_bau_anchor
                or row.is_selected_maximum_saving != offer.is_selected_maximum_saving
                or row.is_selected_least_degradation != offer.is_selected_least_degradation
                or row.is_selected_intermediate != offer.is_selected_intermediate
                or row.selection_reason != offer.selection_reason
                or row.selection_rank_within_target != offer.selection_rank_within_target
                or row.diversity_score != offer.diversity_score
            ):
                raise SchemaValidationError("customer row selection metadata does not match offer.")
            if row.ready_time != expected_ready:
                raise SchemaValidationError("customer row ready_time does not match offer.")
            if row.charging_schedule_kw != offer.profile.power_kw:
                raise SchemaValidationError("customer row schedule does not match offer.")
            if row.role != source.endpoint_role:
                raise SchemaValidationError("customer row role does not match offer provenance.")
            if row.requested_saving is not None and source.source_kind == "bau":
                raise SchemaValidationError("BAU requested saving must be null.")
            if row.requested_saving != offer.requested_saving:
                raise SchemaValidationError("customer requested_saving does not match offer.")
            if row.provenance_flags != source.provenance_flags:
                raise SchemaValidationError("customer row provenance flags do not match source.")
            if (
                row.saving_band_lower != offer.saving_band_lower
                or row.saving_band_upper != offer.saving_band_upper
            ):
                raise SchemaValidationError("customer saving band metadata does not match offer.")
            if (
                row.battery_metric_model_id != offer.battery_metric_model_id
                or row.battery_metric_comparison_scope != offer.battery_metric_comparison_scope
            ):
                raise SchemaValidationError(
                    "customer battery metric metadata does not match offer."
                )
            if (
                row.degradation_model_id != offer.degradation_model_id
                or row.parameter_set_id != offer.parameter_set_id
                or row.parameter_status != offer.parameter_status
                or row.effective_fec_at_start != offer.effective_fec_at_start
                or row.minimum_effective_fec != offer.minimum_effective_fec
                or row.fec_regularization_applied != offer.fec_regularization_applied
                or row.fec_regularization_source != offer.fec_regularization_source
                or row.parked_period_hours != offer.parked_period_hours
                or row.parked_period_source != offer.parked_period_source
            ):
                raise SchemaValidationError("customer degradation provenance does not match offer.")
        ordered_offers = tuple(
            sorted(
                offers,
                key=lambda row: (
                    starts[offer_by_id[row.offer_id].ready_step]
                    if offer_by_id[row.offer_id].ready_step < expected_steps
                    else ends[-1],
                    row.target_soc_percent,
                    row.actual_saving if row.actual_saving is not None else row.saving,
                    _ROLE_ORDER[row.role],
                    row.offer_id,
                ),
            )
        )
        rich_offer_by_id = {offer.offer_id: offer for offer in self.assembled_menu.generated_offers}
        ordered_rich = tuple(
            sorted(
                rich_rows,
                key=lambda row: (
                    row.ready_boundary_absolute_minute
                    if row.ready_boundary_absolute_minute is not None
                    else (
                        starts[rich_offer_by_id[row.offer_id].ready_step]
                        if rich_offer_by_id[row.offer_id].ready_step < expected_steps
                        else ends[-1]
                    ),
                    row.target_soc_percent,
                    row.actual_saving if row.actual_saving is not None else row.saving,
                    _ROLE_ORDER[row.role],
                    row.offer_id,
                ),
            )
        )
        object.__setattr__(self, "current_soc", current)
        object.__setattr__(self, "next_trip_distance_km", distance)
        object.__setattr__(self, "offers", ordered_offers)
        object.__setattr__(self, "generated_offers", ordered_rich)
        object.__setattr__(self, "interval_start_minutes", starts)
        object.__setattr__(self, "interval_end_minutes", ends)
        object.__setattr__(self, "interval_duration_minutes", durations)
        object.__setattr__(self, "interval_price_per_kwh", prices)
        object.__setattr__(self, "currency_label", currency_label)
        if (
            self.raw_offer_count < 0
            or self.scientific_duplicate_count < 0
            or self.generated_offer_count < 0
        ):
            raise SchemaValidationError("offer counts must be non-negative.")
        if self.generated_offer_count not in (0, len(ordered_rich)):
            raise SchemaValidationError("generated_offer_count does not match generated_offers.")
        object.__setattr__(self, "generated_offer_count", len(ordered_rich))
        request_diagnostics = self.request_diagnostics or self.assembled_menu.request_diagnostics
        optimization_diagnostics = (
            self.optimization_diagnostics or self.assembled_menu.optimization_diagnostics
        )
        if not isinstance(request_diagnostics, RequestGenerationDiagnostics):
            raise SchemaValidationError("request_diagnostics has an invalid type.")
        if not isinstance(optimization_diagnostics, OptimizationDiagnostics):
            raise SchemaValidationError("optimization_diagnostics has an invalid type.")
        summaries = self.target_summaries or self.assembled_menu.target_summaries
        failures = self.saving_level_failures or self.assembled_menu.saving_level_failures
        if any(not isinstance(item, TargetGenerationSummary) for item in summaries):
            raise SchemaValidationError("target_summaries has an invalid item.")
        if any(not isinstance(item, SavingLevelFailure) for item in failures):
            raise SchemaValidationError("saving_level_failures has an invalid item.")
        object.__setattr__(self, "request_diagnostics", request_diagnostics)
        object.__setattr__(self, "optimization_diagnostics", optimization_diagnostics)
        object.__setattr__(self, "target_summaries", tuple(summaries))
        object.__setattr__(self, "saving_level_failures", tuple(failures))

    @property
    def interval_duration_hours(self) -> tuple[float, ...]:
        return tuple(value / 60.0 for value in self.interval_duration_minutes)

    @property
    def request_count_total(self) -> int:
        return self.request_diagnostics.request_count_total

    @property
    def request_count_feasible(self) -> int:
        return self.request_diagnostics.request_count_feasible

    @property
    def request_count_positive_saving(self) -> int:
        return self.request_diagnostics.request_count_positive_saving

    @property
    def request_count_no_saving(self) -> int:
        return self.request_diagnostics.request_count_no_saving

    @property
    def request_count_infeasible(self) -> int:
        return self.request_diagnostics.request_count_infeasible

    @property
    def interval_start_times(self) -> tuple[str, ...]:
        return tuple(_format_clock(value) for value in self.interval_start_minutes)

    @property
    def interval_end_times(self) -> tuple[str, ...]:
        return tuple(_format_clock(value) for value in self.interval_end_minutes)


def generate_ev_menu(
    *,
    ev_model: str | EVModel,
    arrival_time: str,
    departure_time: str,
    current_soc: float,
    next_trip_distance_km: float,
    buffer_soc: float = 0.10,
    tariff_name: TariffName = "research_tou",
    flat_price_per_kwh: float = 7.0,
    battery_temperature_c: float = 25.0,
    battery_age_years: float | None = None,
    accumulated_equivalent_full_cycles: float | None = None,
    daytime_parked_hours: float | None = None,
    timestep_minutes: int = 15,
    menu_settings: MenuSettings | None = None,
    saving_step: float | None = None,
    maximum_saving_levels_per_request: int | None = None,
    saving_band_tolerance: float | None = None,
    saving_zero_tolerance: float | None = None,
    generation_settings: MenuGenerationSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
    frontier_settings: FrontierSettings | None = None,
    assembly_settings: MenuAssemblySettings | None = None,
    menu_stage: MenuStage | None = None,
    validation_tolerances: ValidationTolerances | None = None,
    custom_price_profile: WeeklyPriceProfile | TimestampedPriceProfile | None = None,
    arrival_day: str | None = None,
    arrival_date: str | None = None,
    tariff: TariffName | None = None,
) -> GeneratedCustomerMenu:
    """Generate a deterministic customer menu without low-level schema construction.

    Times use strict local 24-hour ``HH:MM`` notation. Arrival and departure
    may be arbitrary minute values; no rounding is performed. With a custom
    profile, pass a validated immutable weekly or timestamped profile and the
    corresponding arrival day/date. Catalogue and built-in tariff values are
    research assumptions and are explicitly exposed in the returned result.
    """
    model = get_ev_model(ev_model)
    if menu_settings is None:
        menu_settings = MenuSettings()
    if menu_stage is not None:
        if assembly_settings is None:
            assembly_settings = MenuAssemblySettings(menu_stage=menu_stage)
        else:
            assembly_settings = replace(assembly_settings, menu_stage=menu_stage)
    if any(
        value is not None
        for value in (
            saving_step,
            maximum_saving_levels_per_request,
            saving_band_tolerance,
            saving_zero_tolerance,
        )
    ):
        menu_settings = replace(
            menu_settings,
            saving_step=menu_settings.saving_step if saving_step is None else saving_step,
            maximum_saving_levels_per_request=(
                menu_settings.maximum_saving_levels_per_request
                if maximum_saving_levels_per_request is None
                else maximum_saving_levels_per_request
            ),
            saving_band_tolerance=(
                menu_settings.saving_band_tolerance
                if saving_band_tolerance is None
                else saving_band_tolerance
            ),
            saving_zero_tolerance=(
                menu_settings.saving_zero_tolerance
                if saving_zero_tolerance is None
                else saving_zero_tolerance
            ),
        )
    current = _finite("current_soc", current_soc)
    distance = _finite("next_trip_distance_km", next_trip_distance_km)
    buffer = _finite("buffer_soc", buffer_soc)
    temperature = _finite("battery_temperature_c", battery_temperature_c)
    if battery_age_years is not None:
        age = _finite("battery_age_years", battery_age_years)
        if age <= 0.0:
            raise PhysicalConstraintError("battery_age_years must be positive.")
    else:
        age = None
    if accumulated_equivalent_full_cycles is not None:
        accumulated_fec = _finite(
            "accumulated_equivalent_full_cycles", accumulated_equivalent_full_cycles
        )
        if accumulated_fec < 0.0:
            raise PhysicalConstraintError(
                "accumulated_equivalent_full_cycles must be non-negative."
            )
    else:
        accumulated_fec = None
    if daytime_parked_hours is not None:
        parked_hours = _finite("daytime_parked_hours", daytime_parked_hours)
        if parked_hours < 0.0:
            raise PhysicalConstraintError("daytime_parked_hours must be non-negative.")
    else:
        parked_hours = None
    flat_price = _finite("flat_price_per_kwh", flat_price_per_kwh)
    if not 0.0 <= current <= 1.0:
        raise PhysicalConstraintError("current_soc must lie in [0, 1].")
    if distance < 0.0:
        raise PhysicalConstraintError("next_trip_distance_km must be non-negative.")
    if not 0.0 <= buffer <= 1.0:
        raise PhysicalConstraintError("buffer_soc must lie in [0, 1].")
    if temperature <= -273.15:
        raise PhysicalConstraintError("battery_temperature_c must exceed absolute zero.")
    if isinstance(timestep_minutes, bool) or not isinstance(timestep_minutes, int):
        raise SchemaValidationError("timestep_minutes must be an integer.")
    if timestep_minutes <= 0 or 1440 % timestep_minutes != 0:
        raise PhysicalConstraintError("timestep_minutes must be a positive divisor of 1440.")
    if tariff is not None:
        if tariff_name != "research_tou" and tariff_name != tariff:
            raise SchemaValidationError("tariff and tariff_name disagree.")
        tariff_name = tariff
    if tariff_name not in ("research_tou", "flat", "custom"):
        raise SchemaValidationError("tariff_name must be 'research_tou', 'flat', or 'custom'.")

    arrival_minute = _parse_clock("arrival_time", arrival_time)
    departure_minute = _parse_clock("departure_time", departure_time)
    duration_minutes = (departure_minute - arrival_minute) % 1440
    if duration_minutes == 0:
        raise PhysicalConstraintError("arrival_time and departure_time must differ.")
    if tariff_name == "custom" and custom_price_profile is None:
        raise SchemaValidationError("custom tariff requires custom_price_profile.")
    if tariff_name != "custom" and custom_price_profile is not None:
        raise SchemaValidationError("custom_price_profile requires tariff_name='custom'.")

    profile_id: str | None = None
    currency_label = "currency"
    planning_arrival = arrival_minute
    planning_departure = arrival_minute + duration_minutes
    timestamped_start: datetime | None = None
    additional_boundaries: tuple[int, ...] = ()
    if tariff_name == "research_tou":
        additional_boundaries = recurring_daily_boundaries(
            start_minute=planning_arrival,
            end_minute=planning_departure,
            boundaries_of_day=(0, 6 * 60, 17 * 60, 23 * 60, 1440),
        )
        profile_id = "research_tou"
    elif tariff_name == "custom":
        if isinstance(custom_price_profile, WeeklyPriceProfile):
            if arrival_day is None:
                raise SchemaValidationError("weekly custom profiles require arrival_day.")
            if not isinstance(arrival_day, str):
                raise SchemaValidationError("arrival_day must be a weekday string.")
            day = arrival_day.strip().title()
            day_index = {
                name: index
                for index, name in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
            }.get(day)
            if day_index is None:
                raise SchemaValidationError(
                    "arrival_day must be Mon, Tue, Wed, Thu, Fri, Sat, or Sun."
                )
            planning_arrival = day_index * 1440 + arrival_minute
            planning_departure = planning_arrival + duration_minutes
            additional_boundaries = custom_price_profile.absolute_boundaries(
                start_minute=planning_arrival,
                end_minute=planning_departure,
            )
            arrival_day = day
            profile_id = custom_price_profile.profile_id
            currency_label = custom_price_profile.currency_label
        elif isinstance(custom_price_profile, TimestampedPriceProfile):
            if arrival_date is None:
                raise SchemaValidationError("timestamped custom profiles require arrival_date.")
            if not isinstance(arrival_date, str):
                raise SchemaValidationError("arrival_date must use YYYY-MM-DD format.")
            try:
                parsed_date = date.fromisoformat(arrival_date)
            except ValueError as exc:
                raise SchemaValidationError("arrival_date must use YYYY-MM-DD format.") from exc
            profile_timezone = custom_price_profile.periods[0].start.tzinfo
            timestamped_start = datetime.combine(
                parsed_date, datetime.min.time(), tzinfo=profile_timezone
            ) + timedelta(minutes=arrival_minute)
            timestamped_end = timestamped_start + timedelta(minutes=duration_minutes)
            additional_boundaries = custom_price_profile.boundaries_for_session(
                timestamped_start,
                timestamped_end,
            )
            planning_arrival = arrival_minute
            planning_departure = arrival_minute + duration_minutes
            additional_boundaries = tuple(arrival_minute + value for value in additional_boundaries)
            profile_id = custom_price_profile.profile_id
            currency_label = custom_price_profile.currency_label
        else:
            raise SchemaValidationError("custom_price_profile has an unsupported type.")

    intervals = build_time_intervals(
        arrival_minute=planning_arrival,
        departure_minute=planning_departure,
        nominal_timestep_minutes=timestep_minutes,
        additional_boundaries=additional_boundaries,
    )
    steps = len(intervals)

    ev = model.to_ev_spec()
    initial_energy = current * ev.battery_capacity_kwh
    commute_energy = distance * model.consumption_kwh_per_km
    buffer_energy = buffer * ev.battery_capacity_kwh
    energy_tolerance = (
        validation_tolerances.energy_kwh
        if isinstance(validation_tolerances, ValidationTolerances)
        else 1e-8
    )
    if initial_energy < ev.minimum_energy_kwh:
        raise PhysicalConstraintError(
            f"Request is below the minimum reserve for model {model.model_id}: "
            f"current_soc={current}, initial_energy={initial_energy} kWh, "
            f"reserve={ev.minimum_energy_kwh} kWh, capacity={ev.battery_capacity_kwh} kWh."
        )
    minimum_required_energy = ev.minimum_energy_kwh + commute_energy + buffer_energy
    if minimum_required_energy > ev.battery_capacity_kwh + energy_tolerance:
        raise PhysicalConstraintError(
            f"Request is physically impossible for model {model.model_id}: "
            f"capacity={ev.battery_capacity_kwh} kWh, reserve={ev.minimum_energy_kwh} kWh, "
            f"trip_distance={distance} km, trip_energy={commute_energy} kWh, "
            f"buffer_soc={buffer} ({buffer_energy} kWh), "
            f"required={minimum_required_energy} kWh."
        )
    session = ChargingSession(
        arrival_step=0,
        departure_step=steps,
        initial_energy_kwh=initial_energy,
        commute_energy_kwh=commute_energy,
        buffer_energy_kwh=buffer_energy,
    )
    if tariff_name == "research_tou":
        prices = tuple(
            _tariff_price("research_tou", interval.start_minute % 1440, flat_price)
            for interval in intervals
        )
    elif tariff_name == "flat":
        prices = (flat_price,) * steps
    elif isinstance(custom_price_profile, WeeklyPriceProfile):
        prices = tuple(
            custom_price_profile.price_at(interval.start_minute) for interval in intervals
        )
    else:
        if timestamped_start is None or not isinstance(
            custom_price_profile, TimestampedPriceProfile
        ):
            raise SchemaValidationError("timestamped custom profile is not configured.")
        prices = tuple(
            custom_price_profile.price_at(
                timestamped_start + timedelta(minutes=interval.start_minute - arrival_minute)
            )
            for interval in intervals
        )
    signal = PlanningSignal(
        timestep_hours=timestep_minutes / 60.0,
        price_per_kwh=prices,
        battery_temperature_c=(temperature,) * steps,
        interval_duration_hours=tuple(interval.duration_hours for interval in intervals),
        interval_start_minutes=tuple(interval.start_minute for interval in intervals),
        interval_end_minutes=tuple(interval.end_minute for interval in intervals),
        nominal_timestep_minutes=timestep_minutes,
    )
    dsettings = degradation_settings
    if dsettings is None:
        dsettings = DegradationSettings()
    dsettings = replace(
        dsettings,
        battery_age_years=dsettings.battery_age_years if age is None else age,
        cumulative_equivalent_full_cycles=(
            dsettings.cumulative_equivalent_full_cycles
            if accumulated_fec is None
            else accumulated_fec
        ),
        parked_day_hours=(dsettings.parked_day_hours if parked_hours is None else parked_hours),
        parked_period_source=(
            dsettings.parked_period_source if parked_hours is None else PARKED_PERIOD_USER_SOURCE
        ),
        fallback_temperature_c=temperature,
    )
    generated = generate_candidate_menu(
        ev=ev,
        session=session,
        signal=signal,
        menu_settings=menu_settings,
        generation_settings=generation_settings,
        validation_tolerances=validation_tolerances,
    )
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=generated,
        menu_settings=menu_settings,
        degradation_settings=dsettings,
        frontier_settings=frontier_settings,
        assembly_settings=assembly_settings,
        validation_tolerances=validation_tolerances,
    )
    generated_source_by_id = {source.offer_id: source for source in assembled.generated_metadata}
    stage_source_by_id: dict[str, dict[str, OfferSource]] = {
        "generated": generated_source_by_id,
        "retained": {source.offer_id: source for source in assembled.retained_metadata},
        "compacted": {source.offer_id: source for source in assembled.compacted_metadata},
        "pareto_efficient": {
            source.offer_id: source for source in assembled.pareto_efficient_metadata
        },
        "preserved_bau_anchors": {
            source.offer_id: source for source in assembled.preserved_bau_anchor_metadata
        },
        "bau_reference_offers": {
            source.offer_id: source for source in assembled.bau_reference_metadata
        },
        "pareto": {source.offer_id: source for source in assembled.pareto_metadata},
        "displayed": {source.offer_id: source for source in assembled.displayed_stage_metadata},
    }

    def make_rows(
        offers: tuple[MenuOffer, ...],
        stage: str,
    ) -> tuple[CustomerMenuRow, ...]:
        result: list[CustomerMenuRow] = []
        stage_sources = stage_source_by_id[stage]
        for offer in offers:
            source = stage_sources[offer.offer_id]
            ready_absolute = (
                intervals[offer.ready_step].start_minute
                if offer.ready_step < len(intervals)
                else intervals[-1].end_minute
            )
            result.append(
                CustomerMenuRow(
                    offer_id=offer.offer_id,
                    ready_time=_format_clock(ready_absolute),
                    target_soc_percent=offer.target_soc * 100.0,
                    charging_cost=offer.charging_cost,
                    saving=offer.advertised_saving,
                    health_score=offer.charging_health_score,
                    energy_drawn_kwh=sum(offer.profile.grid_energy_kwh),
                    role=source.endpoint_role,
                    charging_schedule_kw=offer.profile.power_kw,
                    requested_saving=source.saving_provenance[0]
                    if source.saving_provenance
                    else None,
                    actual_saving=offer.advertised_saving,
                    actual_cost=offer.charging_cost,
                    raw_battery_stress=offer.raw_battery_stress,
                    estimated_capacity_loss=offer.estimated_capacity_loss,
                    estimated_rul_years=offer.estimated_rul_years,
                    provenance_flags=source.provenance_flags,
                    ready_boundary_absolute_minute=ready_absolute,
                    target_battery_energy_kwh=offer.target_battery_energy_kwh,
                    same_target_bau_cost=offer.same_target_bau_cost,
                    saving_band_lower=offer.saving_band_lower,
                    saving_band_upper=offer.saving_band_upper,
                    saving_band_violation=offer.saving_band_violation,
                    battery_metric_model_id=offer.battery_metric_model_id,
                    battery_metric_comparison_scope=offer.battery_metric_comparison_scope,
                    degradation_model_id=offer.degradation_model_id,
                    degradation_model_version=offer.degradation_model_version,
                    parameter_set_id=offer.parameter_set_id,
                    parameter_status=offer.parameter_status,
                    battery_chemistry=offer.battery_chemistry,
                    calendar_capacity_fade=offer.calendar_capacity_fade,
                    cycle_capacity_fade=offer.cycle_capacity_fade,
                    total_capacity_fade=offer.total_capacity_fade,
                    capacity_fade_percent=offer.capacity_fade_percent,
                    calendar_fade_connected_window=offer.calendar_fade_connected_window,
                    calendar_fade_parked_period=offer.calendar_fade_parked_period,
                    battery_temperature_c=offer.battery_temperature_c,
                    battery_age_years=offer.battery_age_years,
                    accumulated_fec_at_start=offer.accumulated_fec_at_start,
                    effective_fec_at_start=offer.effective_fec_at_start,
                    minimum_effective_fec=offer.minimum_effective_fec,
                    fec_regularization_applied=offer.fec_regularization_applied,
                    fec_regularization_source=offer.fec_regularization_source,
                    parked_period_hours=offer.parked_period_hours,
                    parked_period_source=offer.parked_period_source,
                    calendar_scope=offer.calendar_scope,
                    temperature_source=offer.temperature_source,
                    battery_age_source=offer.battery_age_source,
                    accumulated_fec_source=offer.accumulated_fec_source,
                    is_pareto_efficient=offer.is_pareto_efficient,
                    is_preserved_bau_anchor=offer.is_preserved_bau_anchor,
                    is_selected_bau_anchor=offer.is_selected_bau_anchor,
                    is_selected_maximum_saving=offer.is_selected_maximum_saving,
                    is_selected_least_degradation=offer.is_selected_least_degradation,
                    is_selected_intermediate=offer.is_selected_intermediate,
                    selection_reason=offer.selection_reason,
                    selection_rank_within_target=offer.selection_rank_within_target,
                    diversity_score=offer.diversity_score,
                )
            )
        return tuple(result)

    rows = make_rows(assembled.offers, assembled.menu_stage)
    generated_rows = make_rows(assembled.generated_offers, "generated")
    retained_rows = make_rows(assembled.retained_offers, "retained")
    compacted_rows = make_rows(assembled.compacted_offers, "compacted")
    efficient_rows = make_rows(assembled.pareto_efficient_offers, "pareto_efficient")
    anchor_rows = make_rows(assembled.preserved_bau_anchors, "preserved_bau_anchors")
    bau_reference_rows = make_rows(assembled.bau_reference_offers, "bau_reference_offers")
    pareto_rows = make_rows(assembled.pareto_offers, "pareto")
    displayed_stage_rows = make_rows(assembled.displayed_offers, "displayed")
    return GeneratedCustomerMenu(
        ev_model=model,
        arrival_time=_format_clock(arrival_minute),
        departure_time=_format_clock(departure_minute),
        timestep_minutes=timestep_minutes,
        current_soc=current,
        next_trip_distance_km=distance,
        tariff_name=tariff_name,
        tariff_is_illustrative=tariff_name == "research_tou",
        offers=rows,
        assembled_menu=assembled,
        profile_id=profile_id,
        currency_label=currency_label,
        arrival_day=arrival_day,
        arrival_date=arrival_date,
        interval_start_minutes=tuple(interval.start_minute for interval in intervals),
        interval_end_minutes=tuple(interval.end_minute for interval in intervals),
        interval_duration_minutes=tuple(interval.duration_minutes for interval in intervals),
        interval_price_per_kwh=prices,
        generated_offers=generated_rows,
        retained_offers=retained_rows,
        compacted_offers=compacted_rows,
        pareto_efficient_offers=efficient_rows,
        preserved_bau_anchors=anchor_rows,
        bau_reference_offers=bau_reference_rows,
        pareto_stage_offers=pareto_rows,
        displayed_stage_offers=displayed_stage_rows,
        menu_stage=assembled.menu_stage,
        pipeline_diagnostics=assembled.pipeline_diagnostics,
        raw_offer_count=assembled.raw_offer_count,
        scientific_duplicate_count=assembled.scientific_duplicate_count,
        generated_offer_count=assembled.generated_offer_count,
        request_diagnostics=assembled.request_diagnostics,
        optimization_diagnostics=assembled.optimization_diagnostics,
        degradation_objective_diagnostics=assembled.degradation_objective_diagnostics,
        target_summaries=assembled.target_summaries,
        saving_level_failures=assembled.saving_level_failures,
    )
