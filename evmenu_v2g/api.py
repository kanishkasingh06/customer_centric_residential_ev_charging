"""Public entry point: ``generate_ev_menu_v2g``.

Mirrors ``evmenu.service.generate_ev_menu`` and adds the V2G consent
envelope.  With ``envelope=V1G_ENVELOPE`` (the default) no export is
permitted and every result reduces to the unidirectional case.
"""

from __future__ import annotations

from dataclasses import dataclass

from evmenu.catalog import get_ev_model
from evmenu.degradation import DegradationSettings
from evmenu.exceptions import PhysicalConstraintError
from evmenu.pricing import (
    MINUTES_PER_WEEK,
    WeeklyPriceProfile,
    load_weekly_price_profile_csv,
)
from evmenu.schemas import ChargingSession, EVSpec, PlanningSignal

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

from .envelope import V1G_ENVELOPE, V2GEnvelope
from .menu import (
    V2GOffer,
    compact,
    generate_offers,
    is_realizable,
    pareto_filter,
)


@dataclass(frozen=True, slots=True)
class V2GMenu:
    """A generated V2G menu plus the diagnostics the formulation asks for."""

    ev_id: str
    ev_model: str
    chemistry: str
    arrival_minute: int
    departure_minute: int
    timestep_minutes: int
    envelope: V2GEnvelope
    sell_to_buy_ratio: float
    generated: tuple[V2GOffer, ...]
    compacted: tuple[V2GOffer, ...]
    displayed: tuple[V2GOffer, ...]
    simultaneous_offer_count: int
    export_offer_count: int
    generated_offer_count: int = 0
    withheld_offer_count: int = 0

    @property
    def export_share(self) -> float:
        """Fraction of displayed offers that actually use export."""
        if not self.displayed:
            return 0.0
        return self.export_offer_count / len(self.displayed)


def _parse_clock(value: str) -> int:
    hours, _, minutes = value.partition(":")
    return int(hours) * 60 + int(minutes)


def build_session_and_signal(
    *,
    ev: EVSpec,
    arrival_time: str,
    departure_time: str,
    current_soc: float,
    next_trip_distance_km: float,
    consumption_kwh_per_km: float,
    buffer_soc: float,
    timestep_minutes: int,
    price_per_kwh: list[float] | tuple[float, ...] | None = None,
    weekly_profile: WeeklyPriceProfile | None = None,
    arrival_day: str = 'Wed',
    battery_temperature_c: float = 25.0,
) -> tuple[ChargingSession, PlanningSignal]:
    """Build a single-session planning signal on a uniform grid.

    ``price_per_kwh`` is a 24-hour profile at ``timestep_minutes``
    resolution, indexed from midnight; it is tiled across two days so an
    overnight session that crosses midnight is covered.
    """
    per_day = (24 * 60) // timestep_minutes
    if weekly_profile is not None:
        # Price by minute-of-week from the repo's own weekly profile, over a
        # two-week grid so a session crossing midnight (or Sunday) is covered.
        day = WEEKDAYS.index(arrival_day[:3].title())
        steps = per_day * 14
        prices = tuple(
            weekly_profile.price_at((day * 1440 + i * timestep_minutes) % MINUTES_PER_WEEK)
            for i in range(steps)
        )
        offset_days = day
    else:
        if price_per_kwh is None or len(price_per_kwh) != per_day:
            raise PhysicalConstraintError(
                f"price_per_kwh must hold {per_day} values at "
                f"{timestep_minutes}-minute resolution."
            )
        prices = tuple(price_per_kwh) * 2
        steps = len(prices)
        offset_days = 0
    hours = timestep_minutes / 60.0
    signal = PlanningSignal(
        timestep_hours=hours,
        price_per_kwh=prices,
        battery_temperature_c=tuple([battery_temperature_c] * steps),
        interval_duration_hours=tuple([hours] * steps),
        interval_start_minutes=tuple(i * timestep_minutes for i in range(steps)),
        interval_end_minutes=tuple((i + 1) * timestep_minutes for i in range(steps)),
        nominal_timestep_minutes=timestep_minutes,
    )
    arrival_minute = _parse_clock(arrival_time)
    departure_minute = _parse_clock(departure_time)
    if departure_minute <= arrival_minute:
        departure_minute += 24 * 60
    base = offset_days * per_day
    arrival_step = base + arrival_minute // timestep_minutes
    departure_step = base + departure_minute // timestep_minutes
    commute = next_trip_distance_km * consumption_kwh_per_km
    session = ChargingSession(
        arrival_step=arrival_step,
        departure_step=departure_step,
        initial_energy_kwh=current_soc * ev.battery_capacity_kwh,
        commute_energy_kwh=commute,
        buffer_energy_kwh=buffer_soc * ev.battery_capacity_kwh,
    )
    return session, signal


def default_tou_profile(timestep_minutes: int = 30) -> list[float]:
    """Illustrative duck-curve ToU tariff in currency units per kWh."""
    per_hour = 60 // timestep_minutes
    hourly = [
        3.0, 2.8, 2.6, 2.5, 2.5, 2.7, 3.5, 5.0,
        6.5, 6.0, 5.0, 4.5, 4.2, 4.0, 4.2, 5.0,
        6.5, 8.0, 9.5, 10.0, 9.0, 7.0, 5.0, 3.8,
    ]
    return [price for price in hourly for _ in range(per_hour)]


def generate_ev_menu_v2g(
    *,
    ev_model: str = "generic_60kwh_nmc",
    ev_id: str | None = None,
    arrival_time: str = "19:30",
    departure_time: str = "08:00",
    current_soc: float = 0.35,
    next_trip_distance_km: float = 40.0,
    consumption_kwh_per_km: float | None = None,
    buffer_soc: float = 0.10,
    timestep_minutes: int = 30,
    price_profile: list[float] | None = None,
    tariff_csv: str | None = None,
    weekly_profile: WeeklyPriceProfile | None = None,
    arrival_day: str = 'Wed',
    sell_to_buy_ratio: float = 0.80,
    battery_temperature_c: float = 25.0,
    battery_age_years: float = 3.0,
    accumulated_equivalent_full_cycles: float = 500.0,
    envelope: V2GEnvelope = V1G_ENVELOPE,
    discharge_severity: float = 1.0,
    levels: int = 4,
    band_tolerance: float = 0.5,
    guard_weight: float = 0.0,
    restarts: int = 1,
    maximum_ready_steps: int = 4,
) -> V2GMenu:
    """Generate a V2G customer menu for one EV and one connected session."""
    model = get_ev_model(ev_model)
    ev = EVSpec(
        ev_id=ev_id or f"{ev_model}-1",
        battery_capacity_kwh=model.usable_battery_kwh,
        minimum_energy_kwh=model.minimum_soc * model.usable_battery_kwh,
        charger_power_kw=model.onboard_ac_power_kw,
        charging_efficiency=model.charging_efficiency,
        chemistry=model.chemistry,
    )
    if consumption_kwh_per_km is None:
        consumption_kwh_per_km = model.consumption_kwh_per_km
    if weekly_profile is None and tariff_csv is not None:
        weekly_profile = load_weekly_price_profile_csv(
            tariff_csv, profile_format='hour_of_week'
        )
    prices = None if weekly_profile is not None else (
        price_profile or default_tou_profile(timestep_minutes)
    )
    session, signal = build_session_and_signal(
        ev=ev,
        arrival_time=arrival_time,
        departure_time=departure_time,
        current_soc=current_soc,
        next_trip_distance_km=next_trip_distance_km,
        consumption_kwh_per_km=consumption_kwh_per_km,
        buffer_soc=buffer_soc,
        timestep_minutes=timestep_minutes,
        price_per_kwh=prices,
        weekly_profile=weekly_profile,
        arrival_day=arrival_day,
        battery_temperature_c=battery_temperature_c,
    )
    if not 0.0 <= sell_to_buy_ratio <= 1.0:
        raise PhysicalConstraintError(
            "sell_to_buy_ratio must lie in [0, 1]; assumption (A1) requires "
            "P^sell <= P."
        )
    n = session.departure_step - session.arrival_step
    sell_price = tuple(
        signal.price_per_kwh[session.arrival_step + i] * sell_to_buy_ratio for i in range(n)
    )
    settings = DegradationSettings(
        battery_age_years=battery_age_years,
        cumulative_equivalent_full_cycles=accumulated_equivalent_full_cycles,
        fallback_temperature_c=battery_temperature_c,
    )
    offers = generate_offers(
        ev=ev,
        session=session,
        signal=signal,
        envelope=envelope,
        settings=settings,
        sell_price=sell_price,
        discharge_severity=discharge_severity,
        levels=levels,
        band_tolerance=band_tolerance,
        guard_weight=guard_weight,
        restarts=restarts,
        maximum_ready_steps=maximum_ready_steps,
    )
    compacted = compact(offers)
    efficient = pareto_filter(compacted)
    # An offer that still imports and exports in the same interval cannot be
    # run by a single inverter and its quoted fade counts both directions, so
    # it is not a thing to put in front of a customer.  It stays in
    # `generated` as the reported Prop 3 residue; `withheld_offer_count` says
    # how many reached the display stage and were held back.
    displayed = [offer for offer in efficient if is_realizable(offer)]
    withheld = len(efficient) - len(displayed)
    displayed = sorted(displayed, key=lambda o: (o.target_soc, o.ready_step, -o.saving))
    return V2GMenu(
        ev_id=ev.ev_id,
        ev_model=ev_model,
        chemistry=ev.chemistry,
        arrival_minute=_parse_clock(arrival_time),
        departure_minute=_parse_clock(departure_time),
        timestep_minutes=timestep_minutes,
        envelope=envelope,
        sell_to_buy_ratio=sell_to_buy_ratio,
        generated=tuple(offers),
        compacted=tuple(compacted),
        displayed=tuple(displayed),
        simultaneous_offer_count=sum(1 for o in offers if o.simultaneous_periods > 0),
        export_offer_count=sum(1 for o in displayed if o.uses_export),
        generated_offer_count=len(offers),
        withheld_offer_count=withheld,
    )


def format_menu(menu: V2GMenu) -> str:
    """Render a menu as a fixed-width table."""
    lines = [
        f"EV {menu.ev_id}  ({menu.chemistry}, {menu.ev_model})",
        (
            f"  session {menu.arrival_minute // 60:02d}:{menu.arrival_minute % 60:02d}"
            f" -> {menu.departure_minute // 60:02d}:{menu.departure_minute % 60:02d}"
            f"   dt={menu.timestep_minutes}min"
        ),
        (
            f"  envelope: discharge {menu.envelope.discharge_power_kw:.1f} kW,"
            f" reserve {menu.envelope.reserve_soc:.0%},"
            f" allowance {menu.envelope.session_export_allowance_kwh}"
            f"   P_sell/P = {menu.sell_to_buy_ratio:.2f}"
        ),
        (
            f"  generated {len(menu.generated)} -> compacted {len(menu.compacted)}"
            f" -> displayed {len(menu.displayed)}"
        ),
        "",
        "  ready   target   saving    fade %    import   export   EFC",
        "  " + "-" * 60,
    ]
    for offer in menu.displayed:
        ready = offer.ready_minute % 1440
        lines.append(
            f"  {ready // 60:02d}:{ready % 60:02d}"
            f"   {offer.target_soc:6.0%}"
            f"  {offer.saving:7.2f}"
            f"  {offer.capacity_fade * 100:8.5f}"
            f"  {offer.import_energy_kwh:7.2f}"
            f"  {offer.export_energy_kwh:7.2f}"
            f"  {offer.equivalent_full_cycles:5.3f}"
        )
    if menu.simultaneous_offer_count:
        lines.append(
            f"  ! {menu.simultaneous_offer_count} of {menu.generated_offer_count}"
            f" generated offer(s) used simultaneous import/export"
            f" -- binaries needed (eq. 63);"
            f" {menu.withheld_offer_count} withheld from the display"
        )
    return "\n".join(lines)
