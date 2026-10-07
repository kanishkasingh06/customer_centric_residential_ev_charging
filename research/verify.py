"""Verification checks for the choice + load-profile pipeline.

These are the claims the load profiles rest on. Each one fails loudly.

1. Energy conservation -- the rasterised fleet profile carries exactly the
   energy of the chosen offers, so the 5-minute regrid of variable-duration
   intervals loses nothing.
2. Readiness -- no rasterised power appears at or after an offer's ready time.
3. Probability normalisation -- every customer's choice distribution sums to 1.
4. Seed stability -- peak load is a property of the scenario, not of the draw.
5. Quantisation -- the load shape survives dropping the request quantisation
   that makes the menu bank cacheable.
"""

from __future__ import annotations

import sys

import numpy as np
from choice import PREFERENCE_MIXES, choice_probabilities, menu_attribute_matrix, utilities
from fleet import (
    ARRIVAL_PATTERNS,
    GRID_MIN,
    MenuBank,
    build_record,
    rasterise,
    sample_fleet,
)
from scenarios import MAX_FLEET, SEEDS, run_cell

FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name:24s} {detail}")
    if not passed:
        FAILURES.append(name)


def main() -> None:
    bank = MenuBank()
    if not bank.menus:
        sys.exit("menu bank is empty -- run scenarios.py first")
    rng = np.random.default_rng(2024)
    draws = sample_fleet(MAX_FLEET, ARRIVAL_PATTERNS["concentrated"], np.random.default_rng(11))

    print("\n0. Choice sets are complete (BAU reference present at every target)")
    missing_bau = 0
    no_decline = 0
    inspected = 0
    for customer in draws[:60]:
        record = bank.get(customer.key)
        if record is None:
            continue
        inspected += 1
        if not record.bau_offers:
            missing_bau += 1
        displayed_targets = {round(o.target_soc, 6) for o in record.offers}
        bau_targets = {round(o.target_soc, 6) for o in record.bau_offers}
        if not displayed_targets <= bau_targets:
            no_decline += 1
    check("BAU references present", missing_bau == 0,
          f"{missing_bau} of {inspected} menus carry none")
    check("BAU covers every displayed target", no_decline == 0,
          f"{no_decline} of {inspected} menus leave a displayed target without "
          "an uncontrolled baseline")
    # This check exists because its absence was expensive. e4bea57 moved the BAU
    # offers out of the displayed menu; this layer kept reading only the
    # displayed offers, so every choice set silently lost "decline to charge"
    # and counterfactual.py silently compared max_saving against itself. Checks
    # 1-5 below all passed throughout, because each of them only inspects the
    # offers that reached it.

    print("\n1. Energy conservation (raster vs source offers)")
    worst = 0.0
    for customer in draws[:60]:
        record = bank.get(customer.key)
        if record is None:
            continue
        for index, offer in enumerate(record.choice_set):
            raster = rasterise(record, index).sum() * GRID_MIN / 60.0
            worst = max(worst, abs(offer.energy_kwh - raster))
    check("max energy error", worst < 1e-9, f"{worst:.3e} kWh over 60 customers x all offers")

    print("\n2. Readiness (no rasterised power at or after ready time)")
    violations = 0
    checked = 0
    for customer in draws[:60]:
        record = bank.get(customer.key)
        if record is None:
            continue
        starts = record.interval_start_minutes
        for offer in record.choice_set:
            checked += 1
            if offer.ready_step >= len(starts):
                continue
            for start, kilowatts in zip(starts, offer.power_kw):
                if start >= offer.ready_minute and kilowatts > 1e-9:
                    violations += 1
    check("post-ready charging", violations == 0, f"{violations} violations in {checked} offers")

    print("\n3. Choice probability normalisation")
    worst_p = 0.0
    for customer in draws[:60]:
        record = bank.get(customer.key)
        if record is None:
            continue
        attributes = menu_attribute_matrix(record)
        for distribution in PREFERENCE_MIXES.values():
            probabilities = choice_probabilities(utilities(attributes, distribution.draw(rng, 40)))
            worst_p = max(worst_p, float(np.abs(probabilities.sum(axis=1) - 1.0).max()))
    check("max |sum(p) - 1|", worst_p < 1e-9, f"{worst_p:.3e}")

    print("\n4. Seed stability of peak load (concentrated / balanced / 150)")
    peaks = [run_cell("concentrated", "balanced", 150, seed, bank)[0].peak_kw for seed in SEEDS]
    spread = float(np.std(peaks) / np.mean(peaks))
    check(
        "peak CV across seeds",
        spread < 0.20,
        f"{spread*100:.1f}% (peaks {[round(p,1) for p in peaks]} kW)",
    )

    print("\n5. Quantisation sensitivity (40 customers, quantised vs exact requests)")
    subset = draws[:40]
    exact_bank: dict[tuple, object] = {}
    for key in sorted({c.exact_key for c in subset}):
        record = build_record(key)
        if record is not None:
            exact_bank[key] = record

    def profile_for(keyfunc, lookup) -> np.ndarray:
        local = np.random.default_rng(777)
        betas = PREFERENCE_MIXES["balanced"].draw(local, len(subset))
        total = None
        for i, customer in enumerate(subset):
            record = lookup(keyfunc(customer))
            if record is None:
                continue
            attributes = menu_attribute_matrix(record)
            single = {k: np.asarray([v[i]]) for k, v in betas.items()}
            probabilities = choice_probabilities(utilities(attributes, single))[0]
            cumulative = np.cumsum(probabilities)
            cumulative[-1] = 1.0
            chosen = int((local.random() > cumulative).sum())
            load = rasterise(record, chosen)
            total = load if total is None else total + load
        return total

    quantised = profile_for(lambda c: c.key, bank.get)
    exact = profile_for(lambda c: c.exact_key, exact_bank.get)
    peak_delta = abs(quantised.max() - exact.max()) / exact.max()
    energy_delta = abs(quantised.sum() - exact.sum()) / exact.sum()
    check("peak difference", peak_delta < 0.15, f"{peak_delta*100:.1f}%")
    check("energy difference", energy_delta < 0.15, f"{energy_delta*100:.1f}%")

    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
    print("All verification checks passed.")


if __name__ == "__main__":
    main()
