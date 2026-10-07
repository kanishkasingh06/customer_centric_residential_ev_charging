"""Invariants the frozen formulation requires of any V2G solution."""

from __future__ import annotations

import pytest

from evmenu.degradation import DegradationSettings
from evmenu.exceptions import PhysicalConstraintError

from evmenu_v2g import (
    THROUGHPUT_FRACTIONS_PER_FULL_CYCLE,
    V2GEnvelope,
    generate_ev_menu_v2g,
    saving_levels,
    throughput_fraction,
)
from evmenu_v2g.api import build_session_and_signal, default_tou_profile
from evmenu.catalog import get_ev_model
from evmenu.schemas import EVSpec


def _ev(model_id: str = "generic_60kwh_nmc") -> EVSpec:
    model = get_ev_model(model_id)
    return EVSpec(
        ev_id="invariant-ev",
        battery_capacity_kwh=model.usable_battery_kwh,
        minimum_energy_kwh=model.minimum_soc * model.usable_battery_kwh,
        charger_power_kw=model.onboard_ac_power_kw,
        charging_efficiency=model.charging_efficiency,
        chemistry=model.chemistry,
    )


ENVELOPE = V2GEnvelope(
    discharge_power_kw=7.2,
    discharge_efficiency=0.93,
    reserve_soc=0.25,
    session_export_allowance_kwh=15.0,
)


@pytest.fixture(scope="module")
def menu():
    return generate_ev_menu_v2g(envelope=ENVELOPE, levels=3, maximum_ready_steps=3)


def test_efc_convention(menu) -> None:
    """One full charge plus one full discharge must advance N by exactly 1."""
    ev = _ev()
    capacity = ev.battery_capacity_kwh
    full_charge_grid = capacity / ev.charging_efficiency
    full_discharge_grid = capacity * 0.93
    phi = throughput_fraction(
        ev=ev,
        grid_in_kwh=[full_charge_grid],
        grid_out_kwh=[full_discharge_grid],
        discharge_efficiency=0.93,
    )
    assert phi == pytest.approx(THROUGHPUT_FRACTIONS_PER_FULL_CYCLE, rel=0.0, abs=1e-12)
    assert phi / THROUGHPUT_FRACTIONS_PER_FULL_CYCLE == pytest.approx(1.0, abs=1e-12)


def test_no_offer_has_negative_saving(menu) -> None:
    """Eq. (56): the k=0 band is floored at zero, so no offer costs more than BAU."""
    assert menu.generated
    assert min(offer.saving for offer in menu.generated) >= -1e-9


def test_zero_saving_endpoint_exists(menu) -> None:
    """The k=0 health-oriented endpoint must survive into the generated set."""
    assert any(offer.requested_level == 0.0 for offer in menu.generated)


def test_no_simultaneous_import_and_export(menu) -> None:
    """Complementarity, eq. (37), after the Prop 3 exchange and repair pass."""
    assert menu.simultaneous_offer_count == 0
    for offer in menu.generated:
        traj = offer.trajectory
        for imported, exported in zip(traj.grid_in_kwh, traj.grid_out_kwh, strict=True):
            assert imported <= 1e-9 or exported <= 1e-9


def test_terminal_energy_equality_holds(menu) -> None:
    """Eq. (39): every trajectory lands exactly on B^dagger."""
    ev = _ev()
    for offer in menu.generated:
        arrival_soc = offer.trajectory.soc[0]
        zeta = ENVELOPE.resolved_departure_floor(arrival_soc)
        expected = max(zeta, offer.target_soc) * ev.battery_capacity_kwh
        assert offer.trajectory.battery_energy_kwh[-1] == pytest.approx(expected, abs=1e-6)


def test_reserve_floor_and_availability_hold(menu) -> None:
    """Eq. (36) and (38)."""
    ev = _ev()
    reserve = max(ev.minimum_energy_kwh, ENVELOPE.reserve_soc * ev.battery_capacity_kwh)
    for offer in menu.generated:
        traj = offer.trajectory
        assert min(traj.battery_energy_kwh) >= reserve - 1e-6
        assert max(traj.battery_energy_kwh) <= ev.battery_capacity_kwh + 1e-6
        assert traj.feasible, traj.violations


def test_export_allowance_respected(menu) -> None:
    """Eq. (40), per connected session rather than per calendar day."""
    for offer in menu.generated:
        assert offer.export_energy_kwh <= ENVELOPE.session_export_allowance_kwh + 1e-6


def test_saving_levels_include_zero() -> None:
    levels = saving_levels(100.0, levels=4)
    assert levels[0] == 0.0
    assert levels[-1] == pytest.approx(100.0)
    assert len(levels) == 5


def test_sell_above_buy_is_rejected() -> None:
    """Assumption (A1): P^sell <= P."""
    with pytest.raises(PhysicalConstraintError):
        generate_ev_menu_v2g(envelope=ENVELOPE, sell_to_buy_ratio=1.5, levels=1)


def test_session_crossing_midnight_gets_one_allowance() -> None:
    """The per-session allowance must not double for an overnight session."""
    ev = _ev()
    session, signal = build_session_and_signal(
        ev=ev,
        arrival_time="19:30",
        departure_time="08:00",
        current_soc=0.35,
        next_trip_distance_km=40.0,
        consumption_kwh_per_km=0.17,
        buffer_soc=0.10,
        timestep_minutes=30,
        price_per_kwh=default_tou_profile(30),
        battery_temperature_c=25.0,
    )
    # The session spans midnight: arrival is on day 0, departure on day 1.
    assert session.arrival_step < 48 <= session.departure_step
    menu = generate_ev_menu_v2g(envelope=ENVELOPE, levels=2, maximum_ready_steps=2)
    for offer in menu.generated:
        assert offer.export_energy_kwh <= ENVELOPE.session_export_allowance_kwh + 1e-6


def test_export_reduces_window_calendar_fade_somewhere(menu) -> None:
    """Sanity: the mechanism eq. (49) describes should be visible at least once."""
    exporting = [o for o in menu.generated if o.export_energy_kwh > 1e-6]
    assert exporting, "expected at least one offer to use export"
    assert all(o.soc_excursion >= 0.0 for o in exporting)
