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
from choice_v2g import END_OF_LIFE_CAPACITY_LOSS

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
TARIFF_MEAN_USD_PER_KWH = 0.16          # US residential order of magnitude
_RUPEE_TARIFF_MEAN = 5.117              # mean of default_tou_profile(30)
TARIFF_SCALE = TARIFF_MEAN_USD_PER_KWH / _RUPEE_TARIFF_MEAN

#: The saving machinery carries ABSOLUTE currency tolerances: offers are
#: banded to within `band_tolerance` of a saving level, and compacted by
#: merging any two within `merge_tolerance`.  Scaling the tariff without
#: scaling these changes the menu rather than just relabelling it --
#: rupee-sized tolerances against dollar savings merge offers a rupee run
#: keeps apart, and generating natively in USD with unscaled tolerances
#: gave 38 displayed offers where the rupee run gave 149.  Both are
#: therefore scaled by the same factor, so the discretisation stays
#: proportional to the spread it is discretising.
BAND_TOLERANCE_USD = 0.5 * TARIFF_SCALE
MERGE_TOLERANCE_USD = 5.0 * TARIFF_SCALE

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

from evmenu.catalog import EVModel
from evmenu.degradation import (
    NAUMANN_LFP_2018_PARAMETERS,
    SCHMALSTIEG_NMC_2014_PARAMETERS,
)
from evmenu_v2g import V2GEnvelope, generate_ev_menu_v2g
from evmenu_v2g.api import default_tou_profile
from evmenu_v2g.menu import V2GOffer, compact, pareto_filter

MENU_SIZE = 15

#: Equal weights over the four attributes a customer actually chooses on.
#: target_soc is already on [0, 1] so it is compared raw, as evmenu does;
#: the other three are divided by their range across the candidate set.
WEIGHTS = {"ready": 1.0, "target": 1.0, "saving": 1.0, "fade": 1.0}
_EPSILON = 1e-12

KIA = EVModel(
    model_id="kia_80kwh_nmc",
    display_name="Kia EV6 / EV3 Long Range (~80 kWh NMC)",
    usable_battery_kwh=78.0,
    minimum_soc=0.05,
    onboard_ac_power_kw=7.2,
    charging_efficiency=0.90,
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
    "price_profile": tuple(p * TARIFF_SCALE for p in default_tou_profile(30)),
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
AMENITY_DEADLINE_MINUTE = 6 * 60          # 06:00
#: Hours past the deadline at which amenity has decayed to zero.
AMENITY_DECAY_HOURS = 3.0


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


def wear_rs(offer: V2GOffer) -> float:
    """Rupees of pack this session consumes, via the Phase 11 mapping.

    ``capacity_fade`` is ``total_capacity_fade``: the FRACTION of usable
    capacity spent, so 3.95e-5 is 0.00395% of the pack, not 0.00395.
    Multiplying by capacity and pack cost puts battery wear in the same
    unit as the bill, which is the whole point -- it is what let the
    choice model weigh a rupee of pack against a rupee of electricity,
    and it is the only form of "health" a customer can act on.
    """
    return (
        offer.capacity_fade
        / END_OF_LIFE_CAPACITY_LOSS
        * KIA.usable_battery_kwh
        * PACK_COST_USD_PER_KWH
    )


def marginal_wear_rs(offer: V2GOffer, baseline_fade: float) -> float:
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
        * KIA.usable_battery_kwh
        * PACK_COST_USD_PER_KWH
    )


def _attributes(offer: V2GOffer) -> tuple[float, float, float, float]:
    return (
        float(offer.ready_minute),
        float(offer.target_soc),
        saving_usd(offer),
        float(offer.capacity_fade),
    )


def _ranges(universe: Sequence[V2GOffer]) -> tuple[float, float, float, float]:
    columns = list(zip(*(_attributes(o) for o in universe), strict=True))
    spans = [max(col) - min(col) for col in columns]
    # target_soc is already a fraction; comparing it raw keeps a 39% vs 100%
    # gap meaningful even when every displayed offer shares a target.
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
    targets = [o.target_soc for o in offers]

    def unit(value: float, column: Sequence[float]) -> float:
        low, high = min(column), max(column)
        return 0.0 if high - low < _EPSILON else (value - low) / (high - low)

    # Amenity is built first, then rescaled onto [0, 1] like the other
    # two.  Punctuality saturates at 1 for anything meeting the deadline,
    # so without rescaling the axis sits in [0.5, 1] and wins the argmax
    # for most of the menu -- it took four labels to savings' one.
    raw_amenity = {}
    for offer in offers:
        minutes_late = (offer.ready_minute % 1440) - AMENITY_DEADLINE_MINUTE
        punctuality = (
            1.0 if minutes_late <= 0.0
            else max(0.0, 1.0 - minutes_late / (AMENITY_DECAY_HOURS * 60.0))
        )
        raw_amenity[offer.offer_id] = 0.5 * punctuality + 0.5 * unit(
            offer.target_soc, targets
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
        return saving_usd(offer) - marginal_wear_rs(offer, floor)

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
        if (o.ready_minute % 1440) <= AMENITY_DEADLINE_MINUTE
    ] or worthwhile
    take(max(on_time, key=lambda o: (o.target_soc, net(o), o.offer_id)), "amenity")
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


def pooled_candidates() -> list[V2GOffer]:
    """One candidate set spanning both modes.

    A V2G envelope contains V1G as a feasible subset, so in principle the
    V2G run alone would do.  In practice the two runs explore different
    local optima -- Remark 1's nonconvexity -- so each finds offers the
    other misses.  Pooling and re-filtering costs one extra generation
    pass and is the honest way to get one menu rather than two.
    """
    pool: list[V2GOffer] = []
    for mode, envelope in (
        ("V1G", V2GEnvelope(discharge_power_kw=0.0, reserve_soc=RESERVE_SOC)),
        ("V2G", V2GEnvelope(
            discharge_power_kw=7.2, discharge_efficiency=0.93,
            reserve_soc=RESERVE_SOC, session_export_allowance_kwh=15.0)),
    ):
        menu = generate_ev_menu_v2g(**SESSION, envelope=envelope)
        # ids collide across runs; keep them distinct so dedup is by
        # attributes, not by a name two different offers happen to share.
        pool.extend(replace(o, offer_id=f"{mode}:{o.offer_id}") for o in menu.displayed)
    # A V2G offer can dominate a V1G one and vice versa, so the union has
    # to be re-filtered; neither input menu's Pareto front survives union.
    return pareto_filter(compact(pool))


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
        chosen, key=lambda item: (item[0].target_soc, item[0].ready_minute)
    ):
        hour, minute = divmod(offer.ready_minute % 1440, 60)
        health, amenity, saving = scores[offer.offer_id]
        mode = "V2G" if offer.export_energy_kwh > 1e-6 else "V1G"
        wear = marginal_wear_rs(offer, baseline)
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
        1 for o, _ in chosen if saving_usd(o) - marginal_wear_rs(o, baseline) < 0
    )
    print(f"  modes: V1G {len(chosen) - v2g}, V2G {v2g};"
          f" {negative} offer(s) net-negative")
    print(f"  minimum pairwise separation: {minimum_separation(chosen, spans):.4f}")
    print(f"  battery = marginal pack spend vs the least-fade row "
          f"(${wear_rs(min(universe, key=lambda o: o.capacity_fade)):.2f} absolute), "
          f"at ${PACK_COST_USD_PER_KWH:,.0f}/kWh residual basis "
          f"({KIA.chemistry}); net = saving - battery, all USD")


def main() -> None:
    universe = pooled_candidates()
    render(select(universe), universe)


if __name__ == "__main__":
    main()
