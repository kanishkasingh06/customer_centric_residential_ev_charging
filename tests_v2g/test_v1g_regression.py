"""The regression that must stay green: with no export, V2G == V1G.

Proposition 2 makes this a STRICT equality, not a tolerance question.  If
anything here drifts it is a defect in the bidirectional fade model, not a
numerical artefact.
"""

from __future__ import annotations

import pytest

from evmenu.degradation import DegradationSettings, assess_candidate_degradation
from evmenu.menu import MenuCandidate
from evmenu.profiles import build_immediate_charging_profile
from evmenu.schemas import ChargingSession, EVSpec, MenuSettings, PlanningSignal
from evmenu.validation import ValidationReport
from evmenu_v2g import V1G_ENVELOPE, V2GEnvelope, assess_trajectory, generate_ev_menu_v2g
from evmenu_v2g.fade import soc_excursion, throughput_fraction


def _fixture(chemistry: str = "NMC") -> tuple[EVSpec, ChargingSession, PlanningSignal]:
    ev = EVSpec(
        ev_id=f"regression-{chemistry}",
        battery_capacity_kwh=60.0,
        minimum_energy_kwh=3.0,
        charger_power_kw=7.2,
        charging_efficiency=0.9,
        chemistry=chemistry,  # type: ignore[arg-type]
    )
    steps = 48
    session = ChargingSession(
        arrival_step=2,
        departure_step=28,
        initial_energy_kwh=0.35 * 60.0,
        commute_energy_kwh=6.8,
        buffer_energy_kwh=6.0,
    )
    prices = tuple(3.0 + 4.0 * ((i * 7) % 11) / 11.0 for i in range(steps))
    signal = PlanningSignal(
        timestep_hours=0.5,
        price_per_kwh=prices,
        battery_temperature_c=(27.0,) * steps,
        interval_duration_hours=(0.5,) * steps,
        interval_start_minutes=tuple(i * 30 for i in range(steps)),
        interval_end_minutes=tuple((i + 1) * 30 for i in range(steps)),
        nominal_timestep_minutes=30,
    )
    return ev, session, signal


@pytest.mark.parametrize("chemistry", ["NMC", "LFP"])
@pytest.mark.parametrize("target", [0.8, 0.9, 1.0])
def test_unidirectional_fade_matches_evmenu_exactly(chemistry: str, target: float) -> None:
    """Prop 2: delta == phi when E_out == 0, so total fade must agree bitwise."""
    ev, session, signal = _fixture(chemistry)
    settings = DegradationSettings(
        battery_age_years=3.0,
        cumulative_equivalent_full_cycles=500.0,
        fallback_temperature_c=27.0,
    )
    built = build_immediate_charging_profile(
        ev=ev, session=session, signal=signal, target_soc=target
    )
    candidate = MenuCandidate(
        candidate_id=f"{ev.ev_id}-{target}-bau",
        ev_id=ev.ev_id,
        kind="immediate_bau",
        target_soc=target,
        target_sources=("minimum_required",),
        target_label="regression target",
        ready_step=built.ready_step,
        charging_cost=built.charging_cost,
        same_target_bau_cost=built.charging_cost,
        saving=0.0,
        required_grid_energy_kwh=built.required_grid_energy_kwh,
        profile=built.profile,
        validation=ValidationReport(issues=()),
    )
    reference = assess_candidate_degradation(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        menu_settings=MenuSettings(),
        degradation_settings=settings,
    )
    terminal = max(session.initial_energy_kwh, target * ev.battery_capacity_kwh)
    mine = assess_trajectory(
        ev=ev,
        session=session,
        signal=signal,
        grid_in_kwh=built.profile.grid_energy_kwh,
        grid_out_kwh=(0.0,) * len(built.profile.grid_energy_kwh),
        soc=built.profile.soc,
        terminal_energy_kwh=terminal,
        discharge_efficiency=0.93,
        discharge_severity=1.0,
        settings=settings,
    )
    assert mine.calendar_window == pytest.approx(
        reference.charging_window_calendar_fade, rel=0.0, abs=1e-18
    )
    assert mine.calendar_parked == pytest.approx(
        reference.parked_day_calendar_fade, rel=0.0, abs=1e-18
    )
    assert mine.cycle == pytest.approx(reference.cycle_fade, rel=0.0, abs=1e-18)
    assert mine.total == pytest.approx(reference.total_fade, rel=0.0, abs=1e-18)


@pytest.mark.parametrize("target", [0.8, 1.0])
def test_soc_excursion_equals_throughput_when_unidirectional(target: float) -> None:
    """The identity that makes the depth substitution backward compatible."""
    ev, session, signal = _fixture()
    built = build_immediate_charging_profile(
        ev=ev, session=session, signal=signal, target_soc=target
    )
    zeros = (0.0,) * len(built.profile.grid_energy_kwh)
    phi = throughput_fraction(
        ev=ev,
        grid_in_kwh=built.profile.grid_energy_kwh,
        grid_out_kwh=zeros,
        discharge_efficiency=0.93,
    )
    delta = soc_excursion(built.profile.soc)
    assert delta == pytest.approx(phi, rel=0.0, abs=1e-15)


def test_v1g_envelope_produces_no_export() -> None:
    """A disabled envelope must yield a strictly unidirectional menu."""
    menu = generate_ev_menu_v2g(envelope=V1G_ENVELOPE, levels=2, maximum_ready_steps=2)
    assert menu.displayed
    assert all(offer.export_energy_kwh == 0.0 for offer in menu.generated)
    assert menu.export_offer_count == 0
    assert menu.simultaneous_offer_count == 0


def test_zero_allowance_envelope_matches_v1g_menu() -> None:
    """A nonzero converter with a zero allowance must also reduce to V1G."""
    blocked = V2GEnvelope(discharge_power_kw=7.2, session_export_allowance_kwh=0.0)
    a = generate_ev_menu_v2g(envelope=V1G_ENVELOPE, levels=2, maximum_ready_steps=2)
    b = generate_ev_menu_v2g(envelope=blocked, levels=2, maximum_ready_steps=2)
    assert len(a.displayed) == len(b.displayed)
    for left, right in zip(a.displayed, b.displayed, strict=True):
        assert left.saving == pytest.approx(right.saving, rel=0.0, abs=1e-9)
        assert left.capacity_fade == pytest.approx(right.capacity_fade, rel=0.0, abs=1e-12)


def test_cost_solve_matches_evmenu_minimum_cost_exactly() -> None:
    """The end-to-end E_out == 0 agreement test Block 9 asks for.

    ``test_unidirectional_fade_matches_evmenu_exactly`` compares the fade
    FUNCTION on a profile evmenu built, which is tight but cannot see a
    solver that returns the wrong trajectory.  This compares the solver:
    with export disabled the feasible set is exactly evmenu's, so the
    cost-minimising solve must reproduce ``build_minimum_cost_charging_
    profile`` -- which is exact by construction, since a constant charger
    limit makes cheapest-first filling optimal.

    Two defects lived in the gap between those two tests.  SLSQP stopped
    at ``ftol`` short of the vertex, and ``solve`` discarded points whose
    ``result.success`` was False even though they were feasible and
    optimal, collapsing whole saving ladders to zero.  Worst error across
    this grid was 76.8 currency units before the LP route; assert it in
    the noise.
    """
    from evmenu.exceptions import PhysicalConstraintError
    from evmenu.profiles import build_minimum_cost_charging_profile
    from evmenu_v2g.menu import bau_cost_for
    from evmenu_v2g.trajectory import RequestContext, solve

    ev, session, signal = _fixture()
    # reserve matched to the pack's own minimum so the envelope adds nothing
    envelope = V2GEnvelope(discharge_power_kw=0.0, reserve_soc=0.05)
    n = session.departure_step - session.arrival_step
    compared = 0
    for target in (0.5, 0.6, 0.8, 0.9, 1.0):
        for ready in range(session.arrival_step + 4, session.departure_step + 1, 3):
            try:
                reference = build_minimum_cost_charging_profile(
                    ev=ev, session=session, signal=signal,
                    target_soc=target, ready_step=ready,
                )
            except PhysicalConstraintError:
                continue  # evmenu cannot serve it either
            ctx = RequestContext(
                ev=ev, session=session, signal=signal, envelope=envelope,
                settings=DegradationSettings(), ready_step=ready,
                target_soc=target, sell_price=(0.0,) * n, discharge_severity=1.0,
            )
            bau = bau_cost_for(
                ev=ev, session=session, signal=signal,
                terminal_energy_kwh=ctx.terminal_energy_kwh,
            )
            got = solve(ctx, objective="cost", bau_cost=bau, restarts=1)
            assert got is not None, f"no solution for target={target} ready={ready}"
            assert got.cost == pytest.approx(
                reference.charging_cost, rel=0.0, abs=1e-6
            ), f"target={target} ready={ready}"
            compared += 1
    assert compared >= 15, f"only {compared} requests compared -- grid went stale"


def test_low_arrival_soc_is_still_served() -> None:
    """A car arriving below the declared reserve must still get a menu.

    Eq. (36) says export may never drive the pack below the reserve.
    Imposing that as a state constraint at every boundary instead said
    the pack is above it throughout, so a car arriving beneath it had no
    feasible trajectory and every request was dropped -- an empty menu,
    silently, for the customer who most needs to charge.  evmenu serves
    this case, so the V1G envelope must too.
    """
    for arrival in (0.08, 0.10, 0.15):
        menu = generate_ev_menu_v2g(
            envelope=V1G_ENVELOPE, current_soc=arrival,
            levels=2, maximum_ready_steps=2,
        )
        assert menu.generated, f"empty menu at arrival SOC {arrival}"
        assert menu.displayed, f"nothing displayable at arrival SOC {arrival}"


def test_export_never_takes_a_low_pack_below_its_arrival_energy() -> None:
    """Relaxing the floor must not license discharging the pack further."""
    envelope = V2GEnvelope(
        discharge_power_kw=7.2, discharge_efficiency=0.93, reserve_soc=0.25,
        session_export_allowance_kwh=15.0,
    )
    arrival_soc = 0.10
    menu = generate_ev_menu_v2g(
        envelope=envelope, current_soc=arrival_soc, levels=2, maximum_ready_steps=2,
    )
    assert menu.generated
    arrival_energy = arrival_soc * 60.0
    for offer in menu.generated:
        assert min(offer.trajectory.battery_energy_kwh) >= arrival_energy - 1e-6, (
            offer.offer_id
        )
