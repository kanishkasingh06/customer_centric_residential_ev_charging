"""Invariants the frozen formulation requires of any V2G solution."""

from __future__ import annotations

import pytest

from evmenu.catalog import get_ev_model
from evmenu.degradation import DegradationSettings
from evmenu.exceptions import PhysicalConstraintError
from evmenu.schemas import EVSpec
from evmenu_v2g import (
    THROUGHPUT_FRACTIONS_PER_FULL_CYCLE,
    V2GEnvelope,
    generate_ev_menu_v2g,
    saving_levels,
    service_minimum_soc,
    throughput_fraction,
)
from evmenu_v2g.api import build_session_and_signal, default_tou_profile, format_menu
from evmenu_v2g.menu import bau_cost_for
from evmenu_v2g.trajectory import RequestContext, solve


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


def test_simultaneity_diagnostic_is_truthful(menu) -> None:
    """The reported overlap count must equal the real one.

    Proposition 3 makes the relaxation exact for the cost-minimising solve
    but NOT for the band-constrained ones, so a residue of overlapping
    trajectories is expected and is reported rather than asserted away.
    What must hold is that the diagnostic sees what it claims to protect:
    a counter that under-reports is worse than no counter.
    """
    actual = 0
    for offer in menu.generated:
        traj = offer.trajectory
        overlapping = sum(
            1
            for imported, exported in zip(traj.grid_in_kwh, traj.grid_out_kwh, strict=True)
            if imported > 1e-9 and exported > 1e-9
        )
        assert overlapping == traj.simultaneous_periods
        if overlapping:
            actual += 1
    assert actual == menu.simultaneous_offer_count


def test_simultaneity_residue_stays_small(menu) -> None:
    """Regression guard on the overlap rate, not a correctness claim."""
    rate = menu.simultaneous_offer_count / len(menu.generated)
    assert rate < 0.20, f"overlap rate {rate:.1%} -- the exchange/repair pass regressed"


def test_service_minimum_includes_the_reserve(menu) -> None:
    """The lowest target must cover trip + buffer AND the reserve floor."""
    ev = _ev()
    session, _ = build_session_and_signal(
        ev=ev,
        arrival_time="19:30",
        departure_time="08:00",
        current_soc=0.35,
        next_trip_distance_km=40.0,
        consumption_kwh_per_km=0.17,
        buffer_soc=0.10,
        timestep_minutes=30,
        price_per_kwh=default_tou_profile(30),
    )
    expected = service_minimum_soc(
        ev=ev, session=session, reserve_soc=ENVELOPE.reserve_soc
    )
    usable = (session.commute_energy_kwh + session.buffer_energy_kwh) / ev.battery_capacity_kwh
    assert expected == pytest.approx(usable + ENVELOPE.reserve_soc)
    assert min(o.target_soc for o in menu.generated) >= expected - 1e-9


def test_no_export_after_ready_by_default(menu) -> None:
    """Once the car is advertised as ready, its charge is not for sale."""
    n = len(menu.generated[0].trajectory.grid_in_kwh)
    departure_step = max(o.ready_step for o in menu.generated)
    arrival_step = departure_step - n
    for offer in menu.generated:
        traj = offer.trajectory
        ready_local = max(0, min(n, offer.ready_step - arrival_step))
        assert sum(traj.grid_out_kwh[ready_local:]) <= 1e-9, offer.offer_id


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

    # Built the session and signal, then asserted against a DIFFERENT menu
    # and re-checked the bound test_export_allowance_respected already
    # covers.  That tested nothing about doubling.  What doubling would
    # look like is a session drawing its full allowance on each side of
    # midnight, so split the export at the boundary and require the two
    # halves to share one allocation rather than hold one each.
    allowance = ENVELOPE.session_export_allowance_kwh
    assert allowance is not None
    ctx = RequestContext(
        ev=ev, session=session, signal=signal, envelope=ENVELOPE,
        settings=DegradationSettings(), ready_step=session.departure_step,
        target_soc=0.8,
        sell_price=tuple(
            signal.price_per_kwh[session.arrival_step + i] * 0.8
            for i in range(session.departure_step - session.arrival_step)
        ),
        discharge_severity=1.0,
    )
    bau = bau_cost_for(
        ev=ev, session=session, signal=signal,
        terminal_energy_kwh=ctx.terminal_energy_kwh,
    )
    # Maximise export by maximising saving: the allowance is what stops it.
    solution = solve(ctx, objective="cost", bau_cost=bau, restarts=1)
    assert solution is not None
    before, after = 0.0, 0.0
    for index, energy in enumerate(solution.grid_out_kwh):
        minute = signal.interval_start_minutes[session.arrival_step + index]
        (before, after) = (
            (before + energy, after) if minute // 1440 == session.arrival_step // 48
            else (before, after + energy)
        )
    assert before > 1e-6 and after > 1e-6, (
        "expected export on both sides of midnight, else this proves nothing"
    )
    assert before + after <= allowance + 1e-6, (
        f"session drew {before:.3f} before midnight and {after:.3f} after, "
        f"totalling more than the one allowance of {allowance}"
    )


def test_export_reduces_window_calendar_fade_somewhere(menu) -> None:
    """Sanity: the mechanism eq. (49) describes should be visible at least once."""
    exporting = [o for o in menu.generated if o.export_energy_kwh > 1e-6]
    assert exporting, "expected at least one offer to use export"
    assert all(o.soc_excursion >= 0.0 for o in exporting)


def test_displayed_offers_are_realizable(menu) -> None:
    """No simultaneous import/export in anything shown to a customer.

    Proposition 3 does not make the relaxation exact under a saving band,
    so a residue in ``generated`` is expected and is reported rather than
    asserted away.  What must not happen is that residue reaching the
    display: a single inverter cannot run it, and the fade quoted against
    it counts throughput in both directions, so the advertised number is
    unachievable by any real schedule.
    """
    assert all(o.simultaneous_periods == 0 for o in menu.displayed)
    efficient_overlaps = menu.withheld_offer_count
    assert efficient_overlaps >= 0
    assert menu.generated_offer_count == len(menu.generated)


def test_offer_ids_are_unique(menu) -> None:
    """Formatting the level value collided whenever S^max was small."""
    ids = [o.offer_id for o in menu.generated]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate offer ids: {sorted(duplicates)}"


def test_export_window_crossing_midnight_is_honoured() -> None:
    """A 22:00-02:00 window must permit export on both sides of midnight.

    Written by continuing past 1440, not by wrapping the numbers round.
    Comparing a wrapped clock minute against a non-wrapping window
    silently dropped the post-midnight half -- the same failure class as
    the per-day allowance this envelope deliberately avoids, and for an
    overnight residential session it is the normal case.
    """
    envelope = V2GEnvelope(discharge_power_kw=7.2, export_window=(1320, 1560))
    assert envelope.permits_minute(1320)        # 22:00
    assert envelope.permits_minute(1430)        # 23:50
    assert envelope.permits_minute(0)           # 00:00, the far side
    assert envelope.permits_minute(90)          # 01:30
    assert not envelope.permits_minute(120)     # 02:00, the exclusive edge
    assert not envelope.permits_minute(1200)    # 20:00, before the window


def test_export_window_capacities_span_midnight() -> None:
    """The same thing where it bites: the per-interval export caps."""
    from evmenu_v2g.trajectory import capacities

    ev = _ev()
    session, signal = build_session_and_signal(
        ev=ev, arrival_time="19:30", departure_time="08:00", current_soc=0.35,
        next_trip_distance_km=40.0, consumption_kwh_per_km=0.17, buffer_soc=0.10,
        timestep_minutes=30, price_per_kwh=default_tou_profile(30),
    )
    envelope = V2GEnvelope(
        discharge_power_kw=7.2, export_window=(1320, 1560),
        allow_export_after_ready=True,
    )
    ctx = RequestContext(
        ev=ev, session=session, signal=signal, envelope=envelope,
        settings=DegradationSettings(), ready_step=session.departure_step,
        target_soc=0.8,
        sell_price=(0.0,) * (session.departure_step - session.arrival_step),
        discharge_severity=1.0,
    )
    _, cap_out = capacities(ctx)
    open_minutes = {
        signal.interval_start_minutes[session.arrival_step + i] % 1440
        for i, cap in enumerate(cap_out) if cap > 0.0
    }
    assert 1320 in open_minutes and 1410 in open_minutes   # before midnight
    assert 0 in open_minutes and 90 in open_minutes        # after midnight
    assert 120 not in open_minutes and 1200 not in open_minutes


def test_ambiguous_wrapped_export_window_is_rejected() -> None:
    """(1320, 120) could mean 22:00-02:00 or nothing; refuse to guess."""
    for bad in ((1320, 120), (0, 2000), (1500, 1600)):
        with pytest.raises(PhysicalConstraintError):
            V2GEnvelope(discharge_power_kw=7.2, export_window=bad)


def test_displayed_menu_shows_delivered_soc_not_the_requested_tier(menu) -> None:
    """The SOC column a customer reads must be what they receive.

    Terminal energy is pinned to ``max(zeta, z) B_max``, so every tier
    beneath the arrival SOC delivers the arrival SOC.  format_menu
    printed ``target_soc``, the request, which showed two offers of
    fifteen as "39%" when both hand back a car at 65%.  Nothing
    computed was wrong; the only number the customer acts on was.
    """
    rendered = format_menu(menu)
    for offer in menu.displayed:
        assert f"{offer.delivered_soc:8.0%}" in rendered, offer.offer_id
    mislabelled = [
        o for o in menu.displayed if abs(o.target_soc - o.delivered_soc) > 1e-6
    ]
    for offer in mislabelled:
        # the request must NOT appear in the SOC column for these rows
        assert f"   {offer.target_soc:8.0%}" not in rendered, offer.offer_id
