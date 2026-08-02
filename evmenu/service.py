"""High-level single-EV menu generation from user-facing inputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from typing import Literal

from .assembly import AssembledMenu, MenuAssemblySettings, OfferSource, assemble_customer_menu
from .catalog import EVModel, get_ev_model
from .degradation import DegradationSettings
from .exceptions import PhysicalConstraintError, SchemaValidationError
from .menu import MenuGenerationSettings, generate_candidate_menu
from .optimization import FrontierSettings
from .schemas import ChargingSession, MenuSettings, PlanningSignal
from .validation import ValidationTolerances

TariffName = Literal["research_tou", "flat"]
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
        if arrival_minute % self.timestep_minutes or departure_minute % self.timestep_minutes:
            raise PhysicalConstraintError("arrival and departure must align to timestep_minutes.")
        current = _finite("current_soc", self.current_soc)
        distance = _finite("next_trip_distance_km", self.next_trip_distance_km)
        if not 0.0 <= current <= 1.0:
            raise PhysicalConstraintError("current_soc must lie in [0, 1].")
        if distance < 0.0:
            raise PhysicalConstraintError("next_trip_distance_km must be non-negative.")
        if not isinstance(self.tariff_name, str) or self.tariff_name not in (
            "research_tou",
            "flat",
        ):
            raise SchemaValidationError("tariff_name must be 'research_tou' or 'flat'.")
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
        expected_steps = duration_minutes // self.timestep_minutes
        source_by_id = {source.offer_id: source for source in self.assembled_menu.source_metadata}
        for row, offer in zip(offers, self.assembled_menu.offers, strict=True):
            if (
                len(row.charging_schedule_kw) != expected_steps
                or len(offer.profile.power_kw) != expected_steps
            ):
                raise SchemaValidationError("customer schedules must match session intervals.")
            expected_ready = _format_clock(
                arrival_minute + offer.ready_step * self.timestep_minutes
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
) -> GeneratedCustomerMenu:
    """Generate a deterministic customer menu without low-level schema construction.

    Times use local 24-hour ``HH:MM`` notation. Arrival and departure must differ; explicit dates are deferred to the CLI/configuration layer. Catalogue and default tariff values are
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
    if tariff_name not in ("research_tou", "flat"):
        raise SchemaValidationError("tariff_name must be 'research_tou' or 'flat'.")

    arrival_minute = _parse_clock("arrival_time", arrival_time)
    departure_minute = _parse_clock("departure_time", departure_time)
    if arrival_minute % timestep_minutes or departure_minute % timestep_minutes:
        raise PhysicalConstraintError("arrival and departure must align to timestep_minutes.")
    duration_minutes = (departure_minute - arrival_minute) % 1440
    if duration_minutes == 0:
        raise PhysicalConstraintError("arrival_time and departure_time must differ.")
    steps = duration_minutes // timestep_minutes

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
    prices = tuple(
        _tariff_price(
            tariff_name,
            (arrival_minute + step * timestep_minutes) % 1440,
            flat_price,
        )
        for step in range(steps)
    )
    signal = PlanningSignal(
        timestep_hours=timestep_minutes / 60.0,
        price_per_kwh=prices,
        battery_temperature_c=(temperature,) * steps,
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
            ready_time=_format_clock(arrival_minute + offer.ready_step * timestep_minutes),
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
    )
