"""Reference policies against which the choice-model load profiles are read.

Deterministic policies on the same fleets and menus as the sweep:

* ``max_saving``   every customer takes the highest-saving offer on their menu.
* ``min_cost``     every customer takes the cheapest offer -- the genuinely
                   cost-minimising choice, which is NOT the same thing (see below).
* ``uncontrolled`` every customer takes the BAU (immediate charging) at the SAME
                   target SOC as their max-saving offer. Identical delivered
                   energy, so max_saving vs uncontrolled isolates timing alone.
* ``best_health``  every customer takes the highest-health offer.

Two findings these policies exist to record:

1. ``advertised_saving`` is measured against a PER-TARGET BAU baseline, so a
   higher target carries a larger achievable saving. argmax(saving) therefore
   selects the 100% target in the large majority of menus and is never the
   cheapest offer. A customer optimising the "you save Rs X" column is steered
   toward maximum consumption, not minimum spend. This is a menu-design issue,
   not just an analysis artefact.

2. Every deterministic policy drives the coincidence factor toward 1.0: perfect
   price response synchronises the fleet on the tariff trough. The mixed-logit
   cells hold it far lower purely because of preference heterogeneity.
"""

from __future__ import annotations

import json

import numpy as np
from choice import PREFERENCE_MIXES
from fleet import ARRIVAL_PATTERNS, GRID_MIN, MenuBank, clock_labels, rasterise, sample_fleet
from scenarios import MAX_FLEET, RESEARCH_DIR, SEEDS, run_cell

POLICIES = ("uncontrolled", "max_saving", "min_cost", "best_health")


def pick(record, policy: str) -> int:
    """Return the index into ``record.choice_set`` this policy selects.

    Indices address ``record.choice_set`` -- displayed offers followed by the
    BAU references -- which is what ``fleet.rasterise`` and
    ``choice.menu_attribute_matrix`` also use.
    """
    offers = record.choice_set
    displayed = record.offers
    if not displayed:
        raise ValueError("menu has no displayed offers")
    best_saving = int(np.argmax([o.advertised_saving for o in displayed]))
    if policy == "max_saving":
        return best_saving
    # min_cost and best_health characterise THE MENU -- the best option a
    # customer could take from what they were offered -- so they range over the
    # displayed offers only. Letting them see the BAU references makes both
    # collapse to "decline to charge", which is true but says nothing about
    # menu design. The choice model does see the full set; these do not.
    if policy == "min_cost":
        return int(np.argmin([o.charging_cost for o in displayed]))
    if policy == "best_health":
        # battery_stress is capacity LOST, so least damage is the MINIMUM.
        # This was argmax over a field that e4bea57 turned into a damage
        # measure, so the policy was selecting the WORST offer on every menu.
        return int(np.argmin([o.battery_stress for o in displayed]))
    if policy == "uncontrolled":
        # The immediate-charging BAU at the same target as the max-saving
        # offer. Identical delivered energy, so max_saving vs uncontrolled
        # isolates timing alone -- which only holds if the target really does
        # match, hence the hard failure below rather than a fallback.
        target = displayed[best_saving].target_soc
        for index, offer in enumerate(offers):
            if offer.role == "bau" and abs(offer.target_soc - target) <= 1e-9:
                return index
        raise ValueError(
            f"no BAU reference at target {target:.3f}; the uncontrolled baseline "
            f"is undefined for this menu. Available BAU targets: "
            f"{sorted(o.target_soc for o in record.bau_offers)}. "
            "An earlier version silently fell back to the max-saving offer here, "
            "which made uncontrolled and max_saving identical in every table."
        )
    raise ValueError(policy)


def run_policy(pattern: str, policy: str, size: int, seed: int, bank: MenuBank):
    fleet = sample_fleet(MAX_FLEET, ARRIVAL_PATTERNS[pattern], np.random.default_rng(seed))[:size]
    load = np.zeros(len(clock_labels()), dtype=float)
    peak_individual = 0.0
    targets: list[float] = []
    for customer in fleet:
        record = bank.get(customer.key)
        if record is None:
            continue
        index = pick(record, policy)
        offer = record.choice_set[index]
        if offer.energy_kwh > 1e-9:
            peak_individual += max(offer.power_kw)
        targets.append(offer.target_soc * 100.0)
        load += rasterise(record, index)
    return {
        "peak_kw": float(load.max()),
        "peak_time": clock_labels()[int(np.argmax(load))],
        "energy_kwh": float(load.sum() * GRID_MIN / 60.0),
        "coincidence_factor": float(load.max() / peak_individual) if peak_individual else 0.0,
        "mean_target_soc": float(np.mean(targets)),
    }, load


def saving_column_diagnostic(bank: MenuBank) -> dict:
    """Quantify the per-target baseline problem in the saving column."""
    fleet = sample_fleet(MAX_FLEET, ARRIVAL_PATTERNS["concentrated"], np.random.default_rng(11))
    at_full = total = agree = 0
    for customer in fleet:
        record = bank.get(customer.key)
        if record is None:
            continue
        offers = record.offers      # displayed options only: what a customer sees
        total += 1
        best_saving = int(np.argmax([o.advertised_saving for o in offers]))
        cheapest = int(np.argmin([o.charging_cost for o in offers]))
        if offers[best_saving].target_soc >= 0.999:
            at_full += 1
        if best_saving == cheapest:
            agree += 1
    return {
        "menus": total,
        "argmax_saving_picks_100pct": at_full,
        "argmax_saving_equals_cheapest": agree,
    }


def main() -> None:
    bank = MenuBank()
    if not bank.menus:
        raise SystemExit("menu bank is empty -- run scenarios.py first")

    diag = saving_column_diagnostic(bank)
    print(f"Saving-column diagnostic over {diag['menus']} menus:")
    print(f"  argmax(saving) selects the 100% target : "
          f"{diag['argmax_saving_picks_100pct']}/{diag['menus']}")
    print(f"  argmax(saving) is also the cheapest    : "
          f"{diag['argmax_saving_equals_cheapest']}/{diag['menus']}")

    rows = []
    for pattern in ARRIVAL_PATTERNS:
        for policy in POLICIES:
            cells = [run_policy(pattern, policy, 150, s, bank)[0] for s in SEEDS]
            times = [c["peak_time"] for c in cells]
            rows.append({
                "pattern": pattern, "policy": policy, "kind": "policy",
                "peak_kw": float(np.mean([c["peak_kw"] for c in cells])),
                "peak_time": max(set(times), key=times.count),
                "energy_kwh": float(np.mean([c["energy_kwh"] for c in cells])),
                "coincidence_factor": float(np.mean([c["coincidence_factor"] for c in cells])),
                "mean_target_soc": float(np.mean([c["mean_target_soc"] for c in cells])),
            })
        for preference in PREFERENCE_MIXES:
            cells = [run_cell(pattern, preference, 150, s, bank)[0] for s in SEEDS]
            times = [c.peak_time for c in cells]
            rows.append({
                "pattern": pattern, "policy": preference, "kind": "choice",
                "peak_kw": float(np.mean([c.peak_kw for c in cells])),
                "peak_time": max(set(times), key=times.count),
                "energy_kwh": float(np.mean([c.energy_kwh for c in cells])),
                "coincidence_factor": float(np.mean([c.coincidence_factor for c in cells])),
                "mean_target_soc": float(np.mean([c.mean_target_soc for c in cells])),
            })

    # Beside the other outputs, not in whatever directory this was launched
    # from. Writing to the cwd left a stale copy in the repo root.
    with (RESEARCH_DIR / "counterfactual.json").open("w", encoding="utf-8") as handle:
        json.dump({"diagnostic": diag, "rows": rows}, handle, indent=1)

    for pattern in ARRIVAL_PATTERNS:
        sub = [r for r in rows if r["pattern"] == pattern]
        base = next(r for r in sub if r["policy"] == "uncontrolled")["peak_kw"]
        print(f"\n=== {pattern} arrivals, 150 EVs, seed-averaged ===")
        print(f"{'policy':22s} {'kind':7s} {'peak kW':>8s} {'at':>6s} {'kWh':>6s} "
              f"{'coinc':>6s} {'tgt%':>5s} {'vs uncontrolled':>16s}")
        for r in sorted(sub, key=lambda x: -x["peak_kw"]):
            print(f"{r['policy']:22s} {r['kind']:7s} {r['peak_kw']:8.1f} {r['peak_time']:>6s} "
                  f"{r['energy_kwh']:6.0f} {r['coincidence_factor']:6.2f} "
                  f"{r['mean_target_soc']:5.1f} {100*(r['peak_kw']/base-1):+15.1f}%")


if __name__ == "__main__":
    main()
