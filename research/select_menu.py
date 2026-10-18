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
from choice import PACK_COST_RS_PER_KWH, wear_cost_rs

from evmenu.catalog import EVModel
from evmenu_v2g import V2GEnvelope, generate_ev_menu_v2g
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
}


#: An offer leads an axis only if it beats the runner-up by this much, on
#: the 0-1 normalised scale.  Below it the offer is genuinely balanced and
#: saying so is more honest than naming whichever axis won by 0.01.
LEAD_MARGIN = 0.08


def wear_rs(offer: V2GOffer) -> float:
    """Rupees of pack this session consumes, via the Phase 11 mapping.

    ``capacity_fade`` is ``total_capacity_fade``: the FRACTION of usable
    capacity spent, so 3.95e-5 is 0.00395% of the pack, not 0.00395.
    Multiplying by capacity and pack cost puts battery wear in the same
    unit as the bill, which is the whole point -- it is what let the
    choice model weigh a rupee of pack against a rupee of electricity,
    and it is the only form of "health" a customer can act on.
    """
    return wear_cost_rs(
        offer.capacity_fade, KIA.usable_battery_kwh, KIA.chemistry
    )


def marginal_wear_rs(offer: V2GOffer, baseline_fade: float) -> float:
    """Wear attributable to CHOOSING this offer.

    Parked-day calendar fade happens whether or not the customer plugs
    in, so charging it to the offer overstates what the decision costs.
    The baseline is the menu's own least-fade row -- in practice the
    decline-to-charge option -- which makes this "what you pay in
    battery, over and above doing nothing".
    """
    return wear_cost_rs(
        max(0.0, offer.capacity_fade - baseline_fade),
        KIA.usable_battery_kwh,
        KIA.chemistry,
    )


def _attributes(offer: V2GOffer) -> tuple[float, float, float, float]:
    return (
        float(offer.ready_minute),
        float(offer.target_soc),
        float(offer.saving),
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
    amenity -- ready early AND with range in hand.  Those are the two
               halves of the convenience-driven customer's utility in
               Phase 11 (delay aversion and target preference), and they
               move together, so they are averaged rather than ranked.
    """
    fades = [o.capacity_fade for o in offers]
    savings = [o.saving for o in offers]
    readies = [float(o.ready_minute) for o in offers]
    targets = [o.target_soc for o in offers]

    def unit(value: float, column: Sequence[float]) -> float:
        low, high = min(column), max(column)
        return 0.0 if high - low < _EPSILON else (value - low) / (high - low)

    scores = {}
    for offer in offers:
        health = 1.0 - unit(offer.capacity_fade, fades)
        saving = unit(offer.saving, savings)
        amenity = 0.5 * (1.0 - unit(float(offer.ready_minute), readies)) + 0.5 * unit(
            offer.target_soc, targets
        )
        scores[offer.offer_id] = (health, amenity, saving)
    return scores


def orientation_label(scores: tuple[float, float, float]) -> str:
    """Name the axis an offer leads, or call it balanced when none does."""
    names = ("health", "amenity", "savings")
    ranked = sorted(zip(scores, names, strict=True), reverse=True)
    if ranked[0][0] - ranked[1][0] < LEAD_MARGIN:
        return "balanced"
    return ranked[0][1]


def intent_anchors(offers: Sequence[V2GOffer]) -> list[tuple[V2GOffer, str]]:
    """The true optimum of each intent, plus one anchor per target SOC.

    Amenity is "ready soonest, then fullest": the convenience-driven
    customer of Phase 11 wants the car usable early and with range in
    hand, and those two move together.
    """
    anchors: list[tuple[V2GOffer, str]] = []
    seen: set[str] = set()

    def take(offer: V2GOffer, label: str) -> None:
        if offer.offer_id not in seen:
            seen.add(offer.offer_id)
            anchors.append((offer, label))

    take(min(offers, key=lambda o: (o.capacity_fade, -o.saving, o.offer_id)), "health")
    take(max(offers, key=lambda o: (o.saving, -o.capacity_fade, o.offer_id)), "savings")
    take(min(offers, key=lambda o: (o.ready_minute, -o.target_soc, o.offer_id)), "amenity")
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
        print(f"{hour:02d}:{minute:02d}   {offer.target_soc:6.0%} {offer.saving:8.2f} "
              f"{wear:8.2f} {offer.saving - wear:8.2f} "
              f"{offer.import_energy_kwh:7.2f} {offer.export_energy_kwh:7.2f}  {mode:<4} "
              f"{orientation_label((health, amenity, saving)):<9} "
              f"{health:.2f}/{amenity:.2f}/{saving:.2f}")
    counts: dict[str, int] = {}
    for offer, _ in chosen:
        counts[orientation_label(scores[offer.offer_id])] = counts.get(
            orientation_label(scores[offer.offer_id]), 0) + 1
    print("\n  orientation mix: " + ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"  modes: V1G {len(chosen) - v2g}, V2G {v2g}")
    print(f"  minimum pairwise separation: {minimum_separation(chosen, spans):.4f}")
    print(f"  battery = marginal pack spend vs the least-fade row "
          f"({wear_rs(min(universe, key=lambda o: o.capacity_fade)):.2f} Rs absolute), "
          f"at {PACK_COST_RS_PER_KWH[KIA.chemistry]:,.0f} Rs/kWh "
          f"({KIA.chemistry}, BNEF 2025); net = saving - battery")


def main() -> None:
    universe = pooled_candidates()
    render(select(universe), universe)


if __name__ == "__main__":
    main()
