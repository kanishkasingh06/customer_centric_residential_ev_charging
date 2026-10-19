"""V2G menu generation: requests, saving levels, offers, Pareto filter.

Formulation reference: Rev 4.2 Block 3 (eq. 45), Block 4 (eq. 56-57),
Block 5 (eq. 61-62).

Offer tuple is ``(ready_by, target_soc, saving, capacity_fade)``.  Export
energy rides along as an INFORMATIONAL field: it tells the customer how
much of their consent an offer spends, but it is not a dominance axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from evmenu.degradation import DegradationSettings
from evmenu.exceptions import PhysicalConstraintError
from evmenu.schemas import ChargingSession, EVSpec, PlanningSignal

from .envelope import V2GEnvelope
from .trajectory import (
    RequestContext,
    Trajectory,
    solve,
)

#: Absolute floor on the realization band half-width, eq. (62).  Without it
#: a zero-saving offer gets the degenerate band [0, 0].
DEFAULT_BAND_ABSOLUTE_FLOOR = 0.50


@dataclass(frozen=True, slots=True)
class V2GOffer:
    """One customer-facing offer."""

    offer_id: str
    ready_step: int
    ready_minute: int
    target_soc: float
    saving: float
    capacity_fade: float
    charging_cost: float
    bau_cost: float
    export_energy_kwh: float
    import_energy_kwh: float
    calendar_window_fade: float
    calendar_parked_fade: float
    cycle_fade: float
    soc_excursion: float
    peak_c_rate: float
    equivalent_full_cycles: float
    requested_level: float
    simultaneous_periods: int
    trajectory: Trajectory

    @property
    def uses_export(self) -> bool:
        return self.export_energy_kwh > 1e-9

    @property
    def delivered_soc(self) -> float:
        """Terminal state of charge actually delivered.

        ``target_soc`` is the REQUESTED tier.  Terminal energy is pinned
        to ``max(zeta, z) B_max``, so whenever the declared departure
        floor exceeds the tier -- which it does for every tier below the
        arrival SOC -- the request understates what the car leaves with.
        Two offers labelled 39% and 65% can deliver identical terminal
        SOC, and a consumer scoring the label reads a difference that is
        not there.  A choice model weighting ``b_soc * target`` is
        exactly such a consumer, so it must read this instead.

        A property rather than a field: it cannot be left unset, and it
        cannot drift from the trajectory it describes.
        """
        states = self.trajectory.battery_energy_kwh
        capacity = self._battery_capacity_kwh
        if capacity <= 0.0:
            # Guard the divisor, not a different quantity: an earlier
            # version tested max(states) and then divided by capacity,
            # so when capacity fell back to 1.0 this returned terminal
            # kWh presented as a fraction.
            return self.target_soc
        return states[-1] / capacity

    @property
    def _battery_capacity_kwh(self) -> float:
        """Pack capacity, recovered from the trajectory's own SOC series."""
        states = self.trajectory.battery_energy_kwh
        socs = self.trajectory.soc
        for energy, soc in zip(states, socs, strict=False):
            if soc > 1e-9:
                return energy / soc
        return 1.0


def bau_cost_for(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    terminal_energy_kwh: float,
) -> float:
    """Same-service BAU: unidirectional immediate charging to ``B^dagger``.

    Where the departure floor sits below the arrival SOC the reference
    requires no purchased energy and the cost is zero; the saving is then
    pure net export revenue, which the caller should flag.
    """
    required = max(0.0, terminal_energy_kwh - session.initial_energy_kwh) / ev.charging_efficiency
    n = session.departure_step - session.arrival_step
    cost = 0.0
    for index in range(n):
        if required <= 0.0:
            break
        step = session.arrival_step + index
        cap = ev.charger_power_kw * signal.interval_durations[step]
        take = min(cap, required)
        cost += signal.price_per_kwh[step] * take
        required -= take
    return cost


def service_minimum_soc(
    *, ev: EVSpec, session: ChargingSession, reserve_soc: float
) -> float:
    """The lowest SOC that is a defensible service promise.

    ``(trip energy + buffer) / B_max + beta``.  The trip and its buffer are
    what the customer needs to *use*; the reserve floor is what they are
    never asked to dip into.  Stacking them is the point: a target that
    covers only the trip leaves the car at the reserve floor on arrival at
    the far end, with nothing in hand.
    """
    usable = (
        session.commute_energy_kwh + session.buffer_energy_kwh
    ) / ev.battery_capacity_kwh
    return min(1.0, max(0.0, usable + max(0.0, reserve_soc)))


def candidate_targets(
    *,
    ev: EVSpec,
    session: ChargingSession,
    reserve_soc: float,
    standard: tuple[float, ...] | None = (0.8, 1.0),
    tier_count: int | None = None,
) -> list[float]:
    """Service minimum plus tiers above it.

    Two ways to pick the tiers.  ``standard`` names them absolutely, which
    is what 0.8 and 1.0 have always been.  ``tier_count`` instead spaces
    ``K`` tiers uniformly from the floor to full, mirroring the
    ``k/K * S^max`` convention the formulation already froze for saving
    levels, and takes precedence when given.

    The floor is ``max(service minimum, arrival SOC)``, not the service
    minimum alone.  A target beneath what the car arrived holding is not
    a service level -- terminal energy is ``max(arrival, target)``, so
    every such tier collapses onto the same "decline to charge"
    trajectory.  Absolute tiers hit this constantly: a car plugged in at
    65% has nothing between there and 0.8, so the health end of its menu
    is one row.
    """
    minimum = service_minimum_soc(
        ev=ev, session=session, reserve_soc=reserve_soc
    )
    # Round the service minimum UP, never to nearest: a displayed target that
    # rounds below its own floor promises very slightly less than the floor
    # it was derived from.
    quantised = min(1.0, ceil(minimum * 1e6) / 1e6)
    targets = {quantised}
    arrival = session.initial_energy_kwh / ev.battery_capacity_kwh
    floor = min(1.0, max(quantised, arrival))
    if tier_count is not None:
        if tier_count < 1:
            raise PhysicalConstraintError("tier_count must be at least 1.")
        span = 1.0 - floor
        for k in range(tier_count + 1):
            targets.add(round(min(1.0, floor + span * k / tier_count), 6))
    else:
        for value in standard or ():
            if value >= floor - 1e-9:
                targets.add(round(value, 6))
    return sorted(t for t in targets if 0.0 <= t <= 1.0)


def candidate_ready_steps(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    terminal_energy_kwh: float,
    maximum: int = 4,
) -> list[int]:
    """Ready-by boundaries from the earliest achievable through departure."""
    required = max(0.0, terminal_energy_kwh - session.initial_energy_kwh) / ev.charging_efficiency
    n = session.departure_step - session.arrival_step
    earliest = session.arrival_step
    remaining = required
    for index in range(n):
        if remaining <= 1e-12:
            break
        step = session.arrival_step + index
        remaining -= min(ev.charger_power_kw * signal.interval_durations[step], remaining)
        earliest = step + 1
    latest = session.departure_step
    if earliest >= latest:
        return [latest]
    span = latest - earliest
    if span <= maximum:
        return list(range(earliest, latest + 1))
    stride = span / maximum
    steps = sorted({earliest + round(i * stride) for i in range(maximum + 1)})
    return [min(latest, s) for s in steps]


def saving_levels(maximum_saving: float, *, levels: int) -> list[float]:
    """``k/K * S_max`` for ``k = 0..K``, eq. (45).

    ``k = 0`` is the health-oriented endpoint: no positive saving is
    required, and the generation band's floored lower edge keeps it from
    becoming a negative-saving offer.
    """
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if maximum_saving <= 0.0:
        return [0.0]
    return [maximum_saving * k / levels for k in range(levels + 1)]


def generate_offers(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    envelope: V2GEnvelope,
    settings: DegradationSettings,
    sell_price: tuple[float, ...],
    discharge_severity: float = 1.0,
    levels: int = 4,
    band_tolerance: float = 0.5,
    guard_weight: float = 0.0,
    restarts: int = 1,
    maximum_ready_steps: int = 4,
    target_tiers: tuple[float, ...] | None = (0.8, 1.0),
    target_tier_count: int | None = None,
) -> list[V2GOffer]:
    """Generate every feasible offer across the request space."""
    offers: list[V2GOffer] = []
    starts = signal.interval_start_minutes
    for tier_index, target in enumerate(candidate_targets(
        ev=ev, session=session, reserve_soc=envelope.reserve_soc, standard=target_tiers,
        tier_count=target_tier_count,
    )):
        probe = RequestContext(
            ev=ev,
            session=session,
            signal=signal,
            envelope=envelope,
            settings=settings,
            ready_step=session.departure_step,
            target_soc=target,
            sell_price=sell_price,
            discharge_severity=discharge_severity,
        )
        terminal = probe.terminal_energy_kwh
        if terminal > ev.battery_capacity_kwh + 1e-9:
            continue
        bau = bau_cost_for(
            ev=ev, session=session, signal=signal, terminal_energy_kwh=terminal
        )
        for ready_step in candidate_ready_steps(
            ev=ev,
            session=session,
            signal=signal,
            terminal_energy_kwh=terminal,
            maximum=maximum_ready_steps,
        ):
            ctx = RequestContext(
                ev=ev,
                session=session,
                signal=signal,
                envelope=envelope,
                settings=settings,
                ready_step=ready_step,
                target_soc=target,
                sell_price=sell_price,
                discharge_severity=discharge_severity,
            )
            # Proposition 3: the relaxation is exact here, so the maximum-
            # saving endpoint needs neither binaries nor restarts.
            cheapest = solve(ctx, objective="cost", bau_cost=bau, restarts=1)
            if cheapest is None:
                continue
            maximum_saving = cheapest.saving
            for index, level in enumerate(saving_levels(maximum_saving, levels=levels)):
                low = max(0.0, level - band_tolerance)
                high = level + band_tolerance
                found = solve(
                    ctx,
                    objective="fade",
                    bau_cost=bau,
                    saving_low=low,
                    saving_high=high,
                    guard_weight=guard_weight,
                    restarts=restarts,
                )
                if found is None:
                    continue
                minute = (
                    starts[ready_step] if starts is not None and ready_step < len(starts)
                    else ready_step
                )
                offers.append(
                    V2GOffer(
                        # The level INDEX, not its formatted value: two
                        # levels closer than the 0.01 display precision
                        # produced colliding ids, which every request with
                        # a small S^max does.
                        offer_id=(
                            f"{ev.ev_id}-z{tier_index}"
                            f"-r{ready_step}-k{index}"
                        ),
                        ready_step=ready_step,
                        ready_minute=int(minute),
                        target_soc=target,
                        saving=found.saving,
                        capacity_fade=found.fade.total,
                        charging_cost=found.cost,
                        bau_cost=bau,
                        export_energy_kwh=found.fade.export_energy_kwh,
                        import_energy_kwh=float(sum(found.grid_in_kwh)),
                        calendar_window_fade=found.fade.calendar_window,
                        calendar_parked_fade=found.fade.calendar_parked,
                        cycle_fade=found.fade.cycle,
                        soc_excursion=found.fade.soc_excursion,
                        peak_c_rate=found.fade.peak_c_rate,
                        equivalent_full_cycles=found.fade.equivalent_full_cycles,
                        requested_level=level,
                        simultaneous_periods=found.simultaneous_periods,
                        trajectory=found,
                    )
                )
    return offers


def dominates(a: V2GOffer, b: V2GOffer, *, tol: float = 1e-9) -> bool:
    """Four-attribute dominance with the fade inequality REVERSED.

    ``a > b`` iff a is sooner-or-equal, at-least-as-much SOC, at-least-as-
    much saving and at-least-as-low fade, with one strict.  The sign flip
    relative to a 0-100 health score is the point: lower fade is better.

    SOC is the DELIVERED terminal state, not the requested tier.  Tiers
    beneath the arrival SOC all deliver the arrival SOC, so comparing
    labels makes offers that deliver the same car look different and
    stops dominance removing them.
    """
    not_worse = (
        a.ready_step <= b.ready_step + tol
        and a.delivered_soc >= b.delivered_soc - tol
        and a.saving >= b.saving - tol
        and a.capacity_fade <= b.capacity_fade + tol
    )
    if not not_worse:
        return False
    return (
        a.ready_step < b.ready_step - tol
        or a.delivered_soc > b.delivered_soc + tol
        or a.saving > b.saving + tol
        or a.capacity_fade < b.capacity_fade - tol
    )


def pareto_filter(offers: list[V2GOffer]) -> list[V2GOffer]:
    """Retain offers not dominated by any other."""
    return [a for a in offers if not any(dominates(b, a) for b in offers if b is not a)]


def _merge_rank(offer: V2GOffer) -> tuple[bool, float]:
    """Representative of a merged run: realizable first, then least fade."""
    return (is_realizable(offer), -offer.capacity_fade)


def is_realizable(offer: V2GOffer) -> bool:
    """Whether a single inverter could actually run this offer.

    An offer left with simultaneous import and export is a relaxation
    artefact: no residential charger imports and exports at once, and the
    fade quoted against it counts throughput in both directions, so the
    advertised number is not achievable by any real schedule.  Such an
    offer stays in ``generated`` as a reported residue and is kept out of
    the customer-facing menu.
    """
    return offer.simultaneous_periods == 0


def compact(offers: list[V2GOffer], *, merge_tolerance: float = 5.0) -> list[V2GOffer]:
    """Within each (ready, target) group, keep the lowest-fade offer per band.

    A realizable offer beats an unrealizable one outright, whatever the
    fade: picking the overlapping trajectory as a bucket's representative
    discards a schedule that can be run in favour of one that cannot.
    """
    groups: dict[tuple[int, float], list[V2GOffer]] = {}
    for offer in offers:
        groups.setdefault((offer.ready_step, round(offer.target_soc, 6)), []).append(offer)

    # Merge on the GAP between consecutive savings, not on a fixed grid.
    # `int(saving // merge_tolerance)` bins against zero, so two offers a
    # thousandth apart survive if they straddle a bin edge while two
    # nearly a whole tolerance apart inside one bin are merged: the
    # behaviour depends on where savings sit relative to zero rather
    # than on how close they are to each other.
    kept: list[V2GOffer] = []
    for members in groups.values():
        members.sort(key=lambda o: (o.saving, o.offer_id))
        run: list[V2GOffer] = []
        for offer in members:
            if run and merge_tolerance > 0 and offer.saving - run[0].saving > merge_tolerance:
                kept.append(max(run, key=_merge_rank))
                run = []
            run.append(offer)
        if run:
            kept.append(max(run, key=_merge_rank))

    return sorted(
        kept, key=lambda o: (o.target_soc, o.ready_step, -o.saving)
    )
