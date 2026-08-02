"""High-level single-EV menu generation from user-facing inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isclose, isfinite
from numbers import Real
from typing import Literal, cast

from .assembly import AssembledMenu, MenuAssemblySettings, OfferSource, assemble_customer_menu
from .catalog import EVModel, get_ev_model
from .degradation import DegradationSettings
from .exceptions import PhysicalConstraintError, SchemaValidationError
from .menu import MenuGenerationSettings, generate_candidate_menu
from .optimization import FrontierSettings
from .pricing import (
    TimestampedPriceProfile,
    WeeklyPriceProfile,
)
from .schemas import ChargingSession, MenuSettings, PlanningSignal
from .timegrid import build_time_intervals, recurring_daily_boundaries
from .validation import ValidationTolerances

TariffName = Literal["research_tou", "flat", "custom"]
_CUSTOMER_ROLES = frozenset(
    {
        "bau",
        "least_degradation",
        "intermediate",
        "maximum_saving",
        "least_and_maximum",
    }
)


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
        if self.offers is None or not isinstance(self.offers, tuple):
            raise SchemaValidationError("offers must be a non-empty tuple.")
        offers = self.offers
        if not offers or any(not isinstance(row, CustomerMenuRow) for row in offers):
            raise SchemaValidationError("offers must be a non-empty sequence of CustomerMenuRow.")
        if len(offers) != len(self.assembled_menu.offers):
            raise SchemaValidationError("customer rows must align with assembled offers.")
        row_ids = tuple(row.offer_id for row in offers)
        offer_ids = tuple(offer.offer_id for offer in self.assembled_menu.offers)
        if len(set(row_ids)) != len(row_ids):
            raise SchemaValidationError("customer row IDs must be unique.")
        if row_ids != offer_ids:
            raise SchemaValidationError("customer rows are not aligned with assembled offers.")
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
        source_by_id = {source.offer_id: source for source in self.assembled_menu.source_metadata}
        for row, offer in zip(offers, self.assembled_menu.offers, strict=True):
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
                ("energy_drawn_kwh", row.energy_drawn_kwh, sum(offer.profile.grid_energy_kwh)),
            )
            for name, observed, expected in checks:
                if not isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9):
                    raise SchemaValidationError(f"customer row {name} does not match offer.")
            if row.ready_time != expected_ready:
                raise SchemaValidationError("customer row ready_time does not match offer.")
            if row.charging_schedule_kw != offer.profile.power_kw:
                raise SchemaValidationError("customer row schedule does not match offer.")
            if row.role != source.endpoint_role:
                raise SchemaValidationError("customer row role does not match offer provenance.")
        object.__setattr__(self, "current_soc", current)
        object.__setattr__(self, "next_trip_distance_km", distance)
        object.__setattr__(self, "offers", offers)
        object.__setattr__(self, "interval_start_minutes", starts)
        object.__setattr__(self, "interval_end_minutes", ends)
        object.__setattr__(self, "interval_duration_minutes", durations)
        object.__setattr__(self, "interval_price_per_kwh", prices)
        object.__setattr__(self, "currency_label", currency_label)

    @property
    def interval_duration_hours(self) -> tuple[float, ...]:
        return tuple(value / 60.0 for value in self.interval_duration_minutes)

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
    battery_temperature_c: float = 30.0,
    timestep_minutes: int = 15,
    menu_settings: MenuSettings | None = None,
    generation_settings: MenuGenerationSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
    frontier_settings: FrontierSettings | None = None,
    assembly_settings: MenuAssemblySettings | None = None,
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
    current = _finite("current_soc", current_soc)
    distance = _finite("next_trip_distance_km", next_trip_distance_km)
    buffer = _finite("buffer_soc", buffer_soc)
    temperature = _finite("battery_temperature_c", battery_temperature_c)
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
        degradation_settings=degradation_settings,
        frontier_settings=frontier_settings,
        assembly_settings=assembly_settings,
        validation_tolerances=validation_tolerances,
    )
    source_by_id: dict[str, OfferSource] = {
        source.offer_id: source for source in assembled.source_metadata
    }
    rows = tuple(
        CustomerMenuRow(
            offer_id=offer.offer_id,
            ready_time=_format_clock(
                intervals[offer.ready_step].start_minute
                if offer.ready_step < len(intervals)
                else intervals[-1].end_minute
            ),
            target_soc_percent=offer.target_soc * 100.0,
            charging_cost=offer.charging_cost,
            saving=offer.advertised_saving,
            health_score=offer.charging_health_score,
            energy_drawn_kwh=sum(offer.profile.grid_energy_kwh),
            role=source_by_id[offer.offer_id].endpoint_role,
            charging_schedule_kw=offer.profile.power_kw,
        )
        for offer in assembled.offers
    )
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
    )
