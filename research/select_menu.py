"""Pick a fixed-size customer menu that spans health, amenity and savings.

Selection is two-stage, following `evmenu/assembly.py`:

1. **Intent anchors.**  Each of the three intents gets its true optimum.
   An anchor is reserved, not competed for: greedy diversity alone can
   miss an extreme point, and the extreme points are exactly what a
   customer scanning for "cheapest" or "gentlest" is looking for.

   Only three.  Reserving one anchor per target SOC as well was measured
   and dropped: with the dense tier ladder it spends eight of fifteen
   slots on near-duplicates and drives the minimum separation DOWN, from
   0.337 to 0.133 on the V2G menu.  Target is already one of the four
   distance axes, so farthest-point covers the tiers on its own.

2. **Farthest-point fill.**  The remaining slots go to whichever
   candidate maximises its minimum distance to everything already
   chosen (greedy k-center) under a range-normalised weighted Euclidean
   metric over (ready minute, target SOC, saving, fade).

Why not KL divergence: offers are points, not distributions.  The only
natural distribution to attach is the charging profile over time, which
discards target, saving and fade; is asymmetric, so pairwise "distance"
is ill-defined; and diverges whenever one profile is zero where another
is not, which is most pairs.  It would also spread the menu along the
one axis Phase 11 showed carries almost no health signal -- at fixed
target, intra-window scheduling leaves 78% of degradation untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import sqrt

# research/ is this file's own directory, so the Phase 11 choice model
# imports directly.  Reuse it rather than restate the conversion: the
# pack costs are sourced and the +/-25% sensitivity band lives there.
#: End of life, as a fraction of initial capacity lost.  Wear is priced
#: as a write-down of the WHOLE pack over the fade it survives, not of
#: the capacity increment lost:
#:
#:     W = (dQ / dQ_EoL) * B_max * pi
#:
#: Reaching end of life costs an owner a pack, not the 20% of capacity
#: that was lost getting there.  Preis and Biedenbach (2023) review the
#: V2G optimisation literature and find the whole-pack form is what it
#: uses, with 20% the standard threshold.  ``research/choice.py``, the
#: Phase 11 V1G model, omits the divisor and so undercharges wear by
#: exactly this factor.
#:
#: Lived in choice_v2g.py until that module was removed; it is a
#: pricing assumption of the menu economics, not of any choice model.
END_OF_LIFE_CAPACITY_LOSS = 0.20

# ---------------------------------------------------------------------------
# money basis: one currency, no FX step
# ---------------------------------------------------------------------------
#
# Earlier versions compared savings from a rupee tariff against a pack
# cost quoted in USD and converted at market exchange.  That ignores
# purchasing power parity, and the two sides of the comparison sit on
# opposite sides of it: a battery pack is a globally traded good, so
# market FX is right for it, while a residential electricity tariff is
# non-traded and domestic, so it is not.  Converting the pack at market
# FX against India's PPP factor of roughly a quarter of it inflated the
# battery side about fourfold, and made most of the menu look
# net-negative when it is not.
#
# Everything is therefore in USD.  The tariff SHAPE is untouched -- the
# duck curve and its 4.0x peak-to-trough ratio are what the optimiser
# responds to -- and only the level is set, to a plausible residential
# mean.  Savings scale linearly with this, so it is a reported
# assumption, not a hidden one.
# Shape statistics of the inherited profile, used to place the mapped
# mean without importing it (the imports sit below these constants).
_RUPEE_TARIFF_MEAN = 5.117              # mean of default_tou_profile(30)
_RUPEE_TARIFF_MIN = 2.500
_RUPEE_TARIFF_MAX = 10.000


#: The saving machinery carries ABSOLUTE currency tolerances: offers are
#: banded to within `band_tolerance` of a saving level, and compacted by
#: merging any two within `merge_tolerance`.  Scaling the tariff without
#: scaling these changes the menu rather than just relabelling it --
#: rupee-sized tolerances against dollar savings merge offers a rupee run
#: keeps apart, and generating natively in USD with unscaled tolerances
#: gave 38 displayed offers where the rupee run gave 149.  Both are
#: therefore scaled by the same factor, so the discretisation stays
#: proportional to the spread it is discretising.

#: Tariff endpoints, USD per kWh.  Set directly rather than as a ratio
#: about a mean: a tariff is published as a cheapest rate and a dearest
#: one, and those are the numbers a customer faces.  The duck-curve
#: SHAPE is inherited unchanged -- which hour is dear and which is cheap
#: is untouched -- and only the endpoints are mapped.
#:
#: The spread is the arbitrage opportunity, so V2G value scales with it.
#: Measured on the worked menu at the 20 USD/kWh residual pack basis,
#: mean held fixed while only the spread moved:
#:     spread   V2G offers   best net   wear as % of saving
#:      4.0x      11 of 15     $4.06            11%
#:      3.0x      10 of 15     $2.99            14%
#:      2.0x       8 of 15     $1.51            20%
#:      1.6x       0 of 15     $0.84            29%
#: Below about 1.6x nothing exports at all: the spread stops covering
#: the round-trip loss plus the fade the export costs.
#:
#: Converter efficiencies, each direction.  These and the tariff spread
#: are not independent: export pays only where the within-session price
#: ratio exceeds
#:
#:     (1 / (eta_in * eta_out)) / sell_to_buy_ratio
#:
#: so efficiency and the export tariff trade off directly against the
#: spread.  At 0.90/0.93 and a sell ratio of 0.80 that threshold is
#: 1.493x; at 0.95/0.95 it is 1.385x.  Efficiency alone cannot rescue a
#: narrow tariff -- a 1.29x session would need a round-trip above 0.970,
#: about 0.985 each way, which no real charger reaches.  The sell ratio
#: is the stronger lever and is still an unexamined inherited
#: assumption.
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

TARIFF_MIN_USD_PER_KWH = 0.08
TARIFF_MAX_USD_PER_KWH = 0.30


def _tariff_usd(
    low: float = TARIFF_MIN_USD_PER_KWH,
    high: float = TARIFF_MAX_USD_PER_KWH,
) -> tuple[float, ...]:
    """The inherited duck-curve shape mapped affinely onto [low, high].

    Affine in the price, so every hour's position relative to the others
    is preserved exactly and only the endpoints move.
    """
    if not 0.0 < low < high:
        raise ValueError("tariff endpoints must satisfy 0 < low < high")
    base = [float(x) for x in default_tou_profile(30)]
    bottom, top = min(base), max(base)
    return tuple(low + (x - bottom) / (top - bottom) * (high - low) for x in base)


#: Band and merge tolerances are ABSOLUTE currency amounts, so they must
#: track the tariff LEVEL the endpoints set.  Left at a value sized for
#: a different currency or level they change the menu rather than
#: relabelling it: rupee-sized tolerances against dollar savings merged
#: offers a rupee run kept apart, giving 38 displayed offers where the
#: rupee run gave 149.
#: Scaled against the MAPPED profile's own mean, not the midpoint of the
#: endpoints over the legacy rupee mean -- mixing those two bases put the
#: tolerances 21% out, and bugs 1 and 3 made tolerance load-bearing.
_MEAN_POSITION = (_RUPEE_TARIFF_MEAN - _RUPEE_TARIFF_MIN) / (
    _RUPEE_TARIFF_MAX - _RUPEE_TARIFF_MIN
)
_MAPPED_TARIFF_MEAN = TARIFF_MIN_USD_PER_KWH + _MEAN_POSITION * (
    TARIFF_MAX_USD_PER_KWH - TARIFF_MIN_USD_PER_KWH
)
_TOLERANCE_SCALE = _MAPPED_TARIFF_MEAN / _RUPEE_TARIFF_MEAN
BAND_TOLERANCE_USD = 0.5 * _TOLERANCE_SCALE
MERGE_TOLERANCE_USD = 5.0 * _TOLERANCE_SCALE

#: Pack cost, USD per kWh.  The RESIDUAL-VALUE basis, not a new-pack
#: price.  What an owner loses to marginal fade is the value of used
#: capacity, which Hein et al. (2012) model endogenously at roughly
#: EUR 18/kWh; a new-pack price is an upper bound by construction, and
#: one that technology makes worse over time, since the replacement a
#: customer eventually buys costs what packs cost THEN, not now.
#: Sweep against the alternatives below before quoting any figure.
PACK_COST_USD_PER_KWH = 20.0
PACK_COST_ALTERNATIVES = {
    "residual (Hein 2012)": 20.0,
    "BNEF BEV pack average": 99.0,
    "BNEF NMC new pack": 128.0,
}

from scipy.optimize import brentq

from evmenu.catalog import EVModel
from evmenu.degradation import (
    NAUMANN_LFP_2018_PARAMETERS,
    NAUMANN_LFP_2020_CYCLE_PARAMETERS,
    SCHMALSTIEG_NMC_2014_PARAMETERS,
    DegradationSettings,
)
from evmenu.schemas import EVSpec
from evmenu_v2g import THROUGHPUT_FRACTIONS_PER_FULL_CYCLE, V2GEnvelope, generate_ev_menu_v2g
from evmenu_v2g.api import default_tou_profile
from evmenu_v2g.fade import cycle_fade
from evmenu_v2g.menu import V2GOffer, compact, pareto_filter

MENU_SIZE = 15

#: Equal weights over the four attributes a customer actually chooses on.
#: target_soc is already on [0, 1] so it is compared raw, as evmenu does;
#: the other three are divided by their range across the candidate set.
#: Relative weights on the four distance axes.  Note that these are NOT
#: the whole story: `_ranges` divides three axes by their observed range
#: and leaves the target axis raw, so target carries more weight than a
#: value of 1.0 here suggests.  That special case cannot move into this
#: dict -- raw comparison means dividing by one, which expressed as a
#: weight would be the square of the menu's target range and so would
#: change with every menu.  Change the behaviour in `_ranges`; this dict
#: only scales it.
WEIGHTS = {"ready": 1.0, "target": 1.0, "saving": 1.0, "fade": 1.0}
_EPSILON = 1e-12

KIA = EVModel(
    model_id="kia_80kwh_nmc",
    display_name="Kia EV6 / EV3 Long Range (~80 kWh NMC)",
    usable_battery_kwh=78.0,
    minimum_soc=0.05,
    onboard_ac_power_kw=7.2,
    # 0.95 each way, a modern bidirectional charger rather than the
    # conservative 0.90/0.93 AC onboard figures inherited from the V1G
    # work.  Round-trip rises from 0.837 to 0.9025.
    charging_efficiency=CHARGE_EFFICIENCY,
    chemistry="NMC",
    consumption_kwh_per_km=0.18,
    assumption_note="EV6/EV3 long-range class; verify usable capacity before deployment.",
)

#: One reserve for both modes.  beta is a declared consent parameter, not a
#: property of the direction of power flow, so holding it fixed is what
#: makes the V1G and V2G menus comparable.
RESERVE_SOC = 0.20

SESSION = {
    "ev_model": KIA,
    "ev_id": "Kia-EV6",
    "arrival_time": "20:00",
    "departure_time": "07:00",
    "current_soc": 0.65,
    "next_trip_distance_km": 40.0,
    "battery_age_years": 3.0,
    "accumulated_equivalent_full_cycles": 500.0,
    "levels": 6,
    "maximum_ready_steps": 6,
    # Uniform tiers from max(service minimum, arrival SOC) to full, rather
    # than the absolute (0.8, 1.0).  Absolute tiers leave a 65%-arrival car
    # nothing between arrival and 0.8, so the health end of its menu is a
    # single "decline" row.
    "target_tier_count": 6,
    "price_profile": _tariff_usd(),
    "band_tolerance": BAND_TOLERANCE_USD,
    "merge_tolerance": MERGE_TOLERANCE_USD,
    # The published parameter sets, not the shipped literature-anchor
    # surrogates.  Naumann et al. (2018) for LFP, Schmalstieg et al.
    # (2014) for NMC.  Set both to None to reproduce the older numbers.
    "lfp_parameters": NAUMANN_LFP_2018_PARAMETERS,
    "nmc_parameters": SCHMALSTIEG_NMC_2014_PARAMETERS,
}


#: An offer leads an axis only if it beats the runner-up by this much, on
#: the 0-1 normalised scale.  Below it the offer is genuinely balanced and
#: saying so is more honest than naming whichever axis won by 0.01.
LEAD_MARGIN = 0.08

#: A customer's stated ready-by deadline.  Amenity is defined against this
#: rather than against "earliest of whatever happens to be on the menu":
#: relative earliness has no meaning to someone who only cares that the car
#: is ready before they leave, and scoring it that way made the amenity
#: axis track target SOC instead, so it selected the fullest offers
#: regardless of what they cost.
AMENITY_DEADLINE_MINUTE_OF_DAY = 6 * 60          # 06:00
#: Hours past the deadline at which amenity has decayed to zero.
AMENITY_DECAY_HOURS = 3.0


def _deadline_absolute(offers: Sequence[V2GOffer]) -> float:
    """The ready-by deadline on the session's ABSOLUTE clock.

    Ready minutes run on a multi-day absolute clock (1200 to 1860 for a
    20:00-07:00 session) while the deadline is a minute of day.  Taking
    `ready_minute % 1440` to compare them wrapped every pre-midnight
    offer to the far side of the deadline, so the twelve earliest
    offers on the menu scored as maximally late and the amenity anchor
    could only ever come from a post-midnight row.

    Known limitation: the deadline day is inferred from the earliest
    ready minute, which is right for an overnight session and wrong for
    a same-day one -- an 08:00-18:00 session would put the deadline at
    06:00 the following morning and score every offer punctual.  The
    deadline should come from the session's declared departure.  Latent
    while every worked session is overnight.
    """
    if not offers:
        return float(AMENITY_DEADLINE_MINUTE_OF_DAY)
    day = min(o.ready_minute for o in offers) // 1440
    deadline = (day + 1) * 1440 + AMENITY_DEADLINE_MINUTE_OF_DAY
    return float(deadline)


def saving_usd(offer: V2GOffer) -> float:
    """The offer's saving, already in USD.

    The menu is now GENERATED against the scaled tariff, so no
    conversion happens here.  An earlier version generated in rupees and
    multiplied afterwards; that is exact per offer, because saving is
    linear in the price vector, but it is not the same menu -- the
    absolute band and merge tolerances were still rupee-sized, so the
    composition came from a finer discretisation than the currency
    warranted.
    """
    return offer.saving


def wear_usd(
    offer: V2GOffer,
    pack_cost: float = PACK_COST_USD_PER_KWH,
    capacity_kwh: float = KIA.usable_battery_kwh,
) -> float:
    """Dollars of pack life this session consumes, via the Phase 11 mapping.

    ``capacity_fade`` is ``total_capacity_fade``: the FRACTION of usable
    capacity spent, so 3.95e-5 is 0.00395% of the pack, not 0.00395.
    Multiplying by capacity and pack cost puts battery wear in the same
    unit as the bill, which is the whole point -- it is what let the
    choice model weigh a dollar of pack against a dollar of electricity,
    and it is the only form of "health" a customer can act on.
    """
    return (
        offer.capacity_fade
        / END_OF_LIFE_CAPACITY_LOSS
        * capacity_kwh
        * pack_cost
    )


def marginal_wear_usd(
    offer: V2GOffer,
    baseline_fade: float,
    pack_cost: float = PACK_COST_USD_PER_KWH,
    capacity_kwh: float = KIA.usable_battery_kwh,
) -> float:
    """Wear attributable to CHOOSING this offer.

    Parked-day calendar fade happens whether or not the customer plugs
    in, so charging it to the offer overstates what the decision costs.
    The baseline is the menu's own least-fade row -- in practice the
    decline-to-charge option -- which makes this "what you pay in
    battery, over and above doing nothing".
    """
    return (
        max(0.0, offer.capacity_fade - baseline_fade)
        / END_OF_LIFE_CAPACITY_LOSS
        * capacity_kwh
        * pack_cost
    )


def _attributes(offer: V2GOffer) -> tuple[float, float, float, float]:
    return (
        float(offer.ready_minute),
        offer.delivered_soc,
        saving_usd(offer),
        float(offer.capacity_fade),
    )


def _ranges(universe: Sequence[V2GOffer]) -> tuple[float, float, float, float]:
    columns = list(zip(*(_attributes(o) for o in universe), strict=True))
    spans = [max(col) - min(col) for col in columns]
    # Delivered SOC is already a fraction, so it is compared raw while
    # the other three are divided by their observed range.  This keeps a
    # 65%-vs-100% gap meaningful on a menu whose targets cluster, at the
    # cost of giving the axis more weight than WEIGHTS["target"] implies
    # -- on the worked menu the target range is 0.35, so raw comparison
    # weights it about 2.9x what range-normalising would.  Measured: the
    # choice changes nothing on this menu, same separation and the same
    # offers selected, so it is a stated preference rather than a tuned
    # one.
    spans[1] = 1.0
    return tuple(max(s, _EPSILON) for s in spans)  # type: ignore[return-value]


def distance(a: V2GOffer, b: V2GOffer, spans: tuple[float, float, float, float]) -> float:
    """Range-normalised weighted Euclidean distance over the four attributes."""
    left, right = _attributes(a), _attributes(b)
    weights = (WEIGHTS["ready"], WEIGHTS["target"], WEIGHTS["saving"], WEIGHTS["fade"])
    return sqrt(
        sum(
            w * ((left[i] - right[i]) / spans[i]) ** 2
            for i, w in enumerate(weights)
        )
    )


def orientation_scores(
    offers: Sequence[V2GOffer],
) -> dict[str, tuple[float, float, float]]:
    """Score every offer on health, amenity and savings, each in [0, 1].

    health  -- how little pack it spends, against the menu's own range.
    savings -- how much money it returns.
    amenity -- the car is ready by the stated deadline, with range in
               hand.  Meeting the deadline is worth full marks; being
               earlier than needed is worth nothing extra, which is the
               point -- scoring relative earliness turned this axis into
               a second target-SOC axis.
    """
    fades = [o.capacity_fade for o in offers]
    savings = [saving_usd(o) for o in offers]
    targets = [o.delivered_soc for o in offers]

    def unit(value: float, column: Sequence[float]) -> float:
        low, high = min(column), max(column)
        return 0.0 if high - low < _EPSILON else (value - low) / (high - low)

    # Amenity is built first, then rescaled onto [0, 1] like the other
    # two.  Punctuality saturates at 1 for anything meeting the deadline,
    # so without rescaling the axis sits in [0.5, 1] and wins the argmax
    # for most of the menu -- it took four labels to savings' one.
    deadline = _deadline_absolute(offers)
    raw_amenity = {}
    for offer in offers:
        minutes_late = offer.ready_minute - deadline
        punctuality = (
            1.0 if minutes_late <= 0.0
            else max(0.0, 1.0 - minutes_late / (AMENITY_DECAY_HOURS * 60.0))
        )
        raw_amenity[offer.offer_id] = 0.5 * punctuality + 0.5 * unit(
            offer.delivered_soc, targets
        )
    amenity_column = list(raw_amenity.values())

    scores = {}
    for offer in offers:
        health = 1.0 - unit(offer.capacity_fade, fades)
        saving = unit(saving_usd(offer), savings)
        amenity = unit(raw_amenity[offer.offer_id], amenity_column)
        scores[offer.offer_id] = (health, amenity, saving)
    return scores


def orientation_label(scores: tuple[float, float, float]) -> str:
    """Name the axis an offer leads, or call it balanced when none does."""
    names = ("health", "amenity", "savings")
    ranked = sorted(zip(scores, names, strict=True), reverse=True)
    if ranked[0][0] - ranked[1][0] < LEAD_MARGIN:
        return "balanced"
    return ranked[0][1]


def intent_anchors(
    offers: Sequence[V2GOffer], baseline_fade: float | None = None
) -> list[tuple[V2GOffer, str]]:
    """The representative offer for each intent.

    Health is least fade.  Savings is best NET value, not largest
    advertised saving.  Amenity is the fullest offer that meets the
    stated ready-by deadline.  None of the three may be an offer whose
    saving is smaller than the pack it consumes.
    """
    anchors: list[tuple[V2GOffer, str]] = []
    seen: set[str] = set()
    floor = min(o.capacity_fade for o in offers) if baseline_fade is None else baseline_fade

    def net(offer: V2GOffer) -> float:
        return saving_usd(offer) - marginal_wear_usd(offer, floor)

    # An offer that destroys value should never REPRESENT an intent, even
    # if it is the extreme point on that axis.  Without this the amenity
    # anchor picked a row returning Rs 11.90 against Rs 42.41 of pack.
    worthwhile = [o for o in offers if net(o) > 0.0] or list(offers)

    def take(offer: V2GOffer, label: str) -> None:
        if offer.offer_id not in seen:
            seen.add(offer.offer_id)
            anchors.append((offer, label))

    take(min(offers, key=lambda o: (o.capacity_fade, -net(o), o.offer_id)), "health")
    # Max NET, not max saving.  advertised_saving is measured against a
    # per-target BAU baseline, so argmax(saving) selects the highest target
    # by construction -- the Phase 11 menu-design defect.  On this session
    # it picks 100% at Rs 143.72 netting Rs 81.86 over 88% at Rs 137.62
    # netting Rs 94.46: six rupees more saving for nineteen more pack.
    take(max(worthwhile, key=lambda o: (net(o), -o.capacity_fade, o.offer_id)), "savings")
    on_time = [
        o for o in worthwhile
        if o.ready_minute <= _deadline_absolute(worthwhile)
    ] or worthwhile
    take(max(on_time, key=lambda o: (o.delivered_soc, net(o), o.offer_id)), "amenity")
    return anchors


def select(offers: Sequence[V2GOffer], size: int = MENU_SIZE) -> list[tuple[V2GOffer, str]]:
    """Intent anchors, then greedy farthest-point fill to `size`."""
    if not offers:
        return []
    spans = _ranges(offers)
    chosen = intent_anchors(offers)[:size]
    taken = {o.offer_id for o, _ in chosen}

    while len(chosen) < min(size, len(offers)):
        remaining = [o for o in offers if o.offer_id not in taken]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda cand: (
                min(distance(cand, picked, spans) for picked, _ in chosen),
                cand.saving,
                -cand.capacity_fade,
                cand.offer_id,
            ),
        )
        chosen.append((best, "diversity"))
        taken.add(best.offer_id)
    return chosen


def minimum_separation(chosen: Sequence[tuple[V2GOffer, str]], spans) -> float:
    """The quantity greedy k-center maximises; report it, do not assume it."""
    pairs = [
        distance(a, b, spans)
        for i, (a, _) in enumerate(chosen)
        for b, _ in chosen[i + 1:]
    ]
    return min(pairs) if pairs else float("nan")


V1G_ENVELOPE_SPEC = V2GEnvelope(discharge_power_kw=0.0, reserve_soc=RESERVE_SOC)
V2G_ENVELOPE_SPEC = V2GEnvelope(
    discharge_power_kw=7.2,
    discharge_efficiency=DISCHARGE_EFFICIENCY,
    reserve_soc=RESERVE_SOC,
    session_export_allowance_kwh=15.0,
)


def candidates_for(envelope: V2GEnvelope, tag: str) -> list[V2GOffer]:
    """The Pareto front for one envelope, tagged so ids stay distinct.

    Compaction and the Pareto filter are applied here rather than taken
    from ``menu.displayed`` so that a single-mode front is built exactly
    as the pooled one is -- same tolerance, same dominance -- and the
    three are therefore comparable.
    """
    menu = generate_ev_menu_v2g(**SESSION, envelope=envelope)
    tagged = [replace(o, offer_id=f"{tag}:{o.offer_id}") for o in menu.displayed]
    return pareto_filter(compact(tagged, merge_tolerance=MERGE_TOLERANCE_USD))


def v1g_candidates() -> list[V2GOffer]:
    """Front for a charge-only customer: no export permitted at all."""
    return candidates_for(V1G_ENVELOPE_SPEC, "V1G")


def v2g_candidates() -> list[V2GOffer]:
    """Front for a V2G-enabled customer.

    This is the menu such a customer would actually see, so it includes
    charge-only offers: the envelope PERMITS export, it does not compel
    it, and `E_out = 0` stays feasible throughout.  Filtering to only
    the exporting offers would answer a different question -- what if
    export were mandatory -- and would make the grid comparison one
    between a free choice and a forced one.
    """
    return candidates_for(V2G_ENVELOPE_SPEC, "V2G")


def pooled_candidates() -> list[V2GOffer]:
    """One candidate set spanning both modes.

    A V2G envelope contains V1G as a feasible subset, so in principle the
    V2G run alone would do.  In practice the two runs explore different
    local optima -- Remark 1's nonconvexity -- so each finds offers the
    other misses.  Pooling and re-filtering costs one extra generation
    pass and is the honest way to get one menu rather than two.

    For a V1G-versus-V2G grid comparison use ``v1g_candidates`` and
    ``v2g_candidates`` instead, and select the same number of offers
    from each: menu length changes the load distribution, so comparing
    fronts of different sizes confounds the export option with the
    number of rows on offer.
    """
    pool: list[V2GOffer] = []
    for tag, envelope in (("V1G", V1G_ENVELOPE_SPEC), ("V2G", V2G_ENVELOPE_SPEC)):
        menu = generate_ev_menu_v2g(**SESSION, envelope=envelope)
        # ids collide across runs; keep them distinct so dedup is by
        # attributes, not by a name two different offers happen to share.
        pool.extend(replace(o, offer_id=f"{tag}:{o.offer_id}") for o in menu.displayed)
    # A V2G offer can dominate a V1G one and vice versa, so the union has
    # to be re-filtered; neither input menu's Pareto front survives union.
    # merge_tolerance is NOT optional here.  Left at compact's default of
    # 5.0 against savings spanning 0 to 4.3 it puts the whole menu in one
    # bucket, degrading compaction to "keep one offer per (ready, target)"
    # -- which keeps the lowest-fade row, almost always the one exporting
    # least.  Measured: 177 pooled offers give a 49-offer front with 7 V2G
    # rows at the default, against 116 and 74 at the intended tolerance.
    return pareto_filter(compact(pool, merge_tolerance=MERGE_TOLERANCE_USD))


def render(chosen: Sequence[tuple[V2GOffer, str]], universe: Sequence[V2GOffer]) -> None:
    spans = _ranges(universe)
    scores = orientation_scores(universe)
    v2g = sum(1 for o, _ in chosen if o.export_energy_kwh > 1e-6)
    print(f"\n{'=' * 86}\nCommon menu -- {len(chosen)} of {len(universe)} "
          f"Pareto-efficient offers pooled across V1G and V2G\n{'=' * 86}")
    baseline = min(o.capacity_fade for o in universe)
    print(f"{'ready':>7} {'target':>7} {'saving':>8} {'battery':>8} {'net':>8} "
          f"{'import':>7} {'export':>7}  {'mode':<4} {'oriented':<9} {'H/A/S':<14}")
    print("-" * 86)
    for offer, why in sorted(
        chosen, key=lambda item: (item[0].delivered_soc, item[0].ready_minute)
    ):
        hour, minute = divmod(offer.ready_minute % 1440, 60)
        health, amenity, saving = scores[offer.offer_id]
        mode = "V2G" if offer.export_energy_kwh > 1e-6 else "V1G"
        wear = marginal_wear_usd(offer, baseline)
        print(f"{hour:02d}:{minute:02d}   {offer.delivered_soc:6.0%} "
              f"{saving_usd(offer):8.2f} "
              f"{wear:8.2f} {saving_usd(offer) - wear:8.2f} "
              f"{offer.import_energy_kwh:7.2f} {offer.export_energy_kwh:7.2f}  {mode:<4} "
              f"{orientation_label((health, amenity, saving)):<9} "
              f"{health:.2f}/{amenity:.2f}/{saving:.2f}"
              f"{'   <- costs more pack than it returns' if saving_usd(offer) - wear < 0 else ''}")
    counts: dict[str, int] = {}
    for offer, _ in chosen:
        counts[orientation_label(scores[offer.offer_id])] = counts.get(
            orientation_label(scores[offer.offer_id]), 0) + 1
    print("\n  orientation mix: " + ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items())))
    negative = sum(
        1 for o, _ in chosen if saving_usd(o) - marginal_wear_usd(o, baseline) < 0
    )
    print(f"  modes: V1G {len(chosen) - v2g}, V2G {v2g};"
          f" {negative} offer(s) net-negative")
    print(f"  minimum pairwise separation: {minimum_separation(chosen, spans):.4f}")
    print(f"  battery = marginal pack spend vs the least-fade row "
          f"(${wear_usd(min(universe, key=lambda o: o.capacity_fade)):.2f} absolute), "
          f"at ${PACK_COST_USD_PER_KWH:,.0f}/kWh residual basis "
          f"({KIA.chemistry}); net = saving - battery, all USD")


def main() -> None:
    universe = pooled_candidates()
    render(select(universe), universe)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# service life
# ---------------------------------------------------------------------------
#
# Health is reported to the customer as years to 80% capacity under daily
# repetition of the chosen session.  That needs no pack price, which is
# why it replaced the monetary form: the price spanned an order of
# magnitude between a used-pack residual and a new-pack quote.
#
# What it does need is care with three referrals, each of which was wrong
# in a first attempt and each of which failed silently:
#
#   1. The commute.  An offer's throughput counts only grid exchanges, so
#      the energy the customer drives away and uses was absent from the
#      cycle term.  A life quoted as "years if you do this every day" has
#      to include the drive.  Omitting it overstated life by 2% on the
#      lightest offer and 12% on the heaviest.
#
#   2. The per-EFC rate.  One session is a fraction of a cycle, so its
#      cycle fade is NOT the per-EFC rate.  Treating it as such
#      understated the cumulative coefficient by 1/EFC_day and inflated
#      every life by roughly threefold.
#
#   3. The calendar anchor.  An offer's calendar fade is measured at the
#      scenario's pack age, where psi < 1.  The cumulative integral
#      assumes the rate at the anchor age, so the measured rate must be
#      divided by psi first.  Omitting that understated calendar fade by
#      1/psi, here 1.32x.
#
# Both laws decelerate, so the cumulative loss is an integral and not a
# multiplication.  Dividing the threshold by a daily rate treats both as
# linear and overstates life by about 45%: a daily rate is a derivative.

#: Capacity lost at end of life.  A threshold, not a failure.
SERVICE_LIFE_THRESHOLD = END_OF_LIFE_CAPACITY_LOSS


def trip_throughput_fraction(
    capacity_kwh: float = KIA.usable_battery_kwh,
    distance_km: float | None = None,
    consumption_kwh_per_km: float = KIA.consumption_kwh_per_km,
) -> float:
    """Battery-side throughput of the commute, as a fraction of capacity.

    This is the same quantity for every offer of a menu -- the trip does
    not depend on which schedule was chosen -- so it changes no
    comparison WITHIN a menu.  It is not therefore optional: it shifts
    every service life, and by different amounts, because the constant
    is a larger fraction of a light schedule's throughput than of a
    heavy one's.
    """
    if distance_km is None:
        distance_km = float(SESSION["next_trip_distance_km"])
    return consumption_kwh_per_km * distance_km / capacity_kwh


def _default_settings() -> DegradationSettings:
    """The published parameter sets at the worked scenario's age and cycles."""
    return DegradationSettings(
        nmc=SCHMALSTIEG_NMC_2014_PARAMETERS,
        lfp=NAUMANN_LFP_2020_CYCLE_PARAMETERS,
        battery_age_years=float(SESSION["battery_age_years"]),
        cumulative_equivalent_full_cycles=float(
            SESSION["accumulated_equivalent_full_cycles"]
        ),
    )


def daily_fade(
    offer: V2GOffer,
    settings: DegradationSettings | None = None,
    capacity_kwh: float = KIA.usable_battery_kwh,
) -> tuple[float, float, float]:
    """One day's (calendar fade, cycle fade, equivalent full cycles).

    Session plus commute.  The commute extends three inputs to the cycle
    term and all three are recomputed rather than scaled:

      throughput  gains the trip energy;
      excursion   gains whatever the post-trip SOC adds below the
                  session minimum, which on this scenario is only the
                  lightest offers, since most already dip lower;
      mean SOC    is averaged over session and post-trip state, which
                  matters because the NMC cycle term carries a
                  mean-voltage dependence.

    Peak C-rate is left at the session value.  Driving 7.2 kWh over an
    hour is about 0.09C on a 78 kWh pack, comparable to the charger, so
    the trip does not raise the session peak.  A fast trip on a small
    pack would, and this would then need revisiting.

    Calendar fade is unchanged by the trip: the parked term already
    prices the post-trip state of charge, and the window term covers
    only the connected session.
    """
    if settings is None:
        settings = _default_settings()
    trip = trip_throughput_fraction(capacity_kwh)
    soc = [e / capacity_kwh for e in offer.trajectory.battery_energy_kwh]
    post_trip = offer.delivered_soc - trip

    phi = offer.trajectory.fade.throughput_fraction + trip
    delta = max(max(soc), offer.delivered_soc) - min(min(soc), post_trip)
    mean_soc = (sum(soc) / len(soc) + post_trip) / 2.0

    ev = EVSpec(
        ev_id=KIA.model_id,
        battery_capacity_kwh=capacity_kwh,
        minimum_energy_kwh=KIA.minimum_soc * capacity_kwh,
        charger_power_kw=KIA.onboard_ac_power_kw,
        charging_efficiency=CHARGE_EFFICIENCY,
        chemistry=KIA.chemistry,
    )
    cycle = cycle_fade(
        ev=ev,
        settings=settings,
        phi=phi,
        delta=delta,
        chat=offer.peak_c_rate,
        mean_soc=mean_soc,
    )
    calendar = offer.calendar_window_fade + offer.calendar_parked_fade
    return calendar, cycle, phi / THROUGHPUT_FRACTIONS_PER_FULL_CYCLE


def service_life_years(
    offer: V2GOffer,
    settings: DegradationSettings | None = None,
    capacity_kwh: float = KIA.usable_battery_kwh,
    threshold: float = SERVICE_LIFE_THRESHOLD,
) -> float:
    """Years to `threshold` capacity loss, repeating this session daily.

    LIFE FROM NEW, not remaining life: both laws are integrated from
    zero, so the answer is "a pack driven and charged this way every day
    from new reaches 80% after T years".  Reporting remaining life would
    need the same integrals evaluated between the current state and the
    threshold, and would answer a different question.

    Each rate is first referred back to its own anchor, then integrated:

        calendar   rate at anchor age  = cal_day / psi(age)
        cycle      rate per EFC at N   = cyc_day / EFC_day
                   cumulative constant = that / (beta N^(beta-1))

        Q(T) = 365 (cal_day/psi) T^alpha / alpha
               + k (365 EFC_day T)^beta

    and T solves Q(T) = threshold.
    """
    if settings is None:
        settings = _default_settings()
    params = settings.parameters_for(KIA.chemistry)
    alpha = params.calendar_time_exponent
    beta = params.cycle_time_exponent
    cal_day, cyc_day, efc_day = daily_fade(offer, settings, capacity_kwh)

    age = max(settings.battery_age_years, 1e-9)
    psi = (age / settings.reference_age_years) ** (alpha - 1.0)
    cal_anchor = cal_day / psi

    reference_n = max(
        settings.cumulative_equivalent_full_cycles, settings.minimum_effective_fec
    )
    k_cycle = 0.0
    if efc_day > 0.0 and cyc_day > 0.0:
        per_efc = cyc_day / efc_day
        k_cycle = per_efc / (beta * reference_n ** (beta - 1.0))

    def accumulated(years: float) -> float:
        calendar = 365.0 * cal_anchor * years**alpha / alpha
        cycle = k_cycle * (365.0 * efc_day * years) ** beta
        return calendar + cycle - threshold

    if accumulated(_SERVICE_LIFE_MAX_YEARS) < 0.0:
        return float("inf")
    return float(brentq(accumulated, 1e-3, _SERVICE_LIFE_MAX_YEARS))


#: Upper bracket for the root find.  A life beyond this is reported as
#: infinite rather than extrapolated: the parameterisation is fitted
#: over months to a few years and says nothing useful at a century.
_SERVICE_LIFE_MAX_YEARS = 500.0
