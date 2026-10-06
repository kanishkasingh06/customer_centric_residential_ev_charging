"""Monte Carlo scenario sweep: preference mix x fleet size x arrival distribution.

The hourly tariff is held fixed (it is the environment, not a swept factor).
Menus depend only on the physical request, so one menu bank serves the whole
sweep; fleet sizes are nested subsets of the largest draw and preference mixes
reuse menus untouched.

Outputs
-------
``results.json``  per-cell metrics
``profiles.npz``  the fleet load profile (kW per 5-minute cell) for each cell
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from choice import PREFERENCE_MIXES, choice_probabilities, menu_attribute_matrix, utilities
from fleet import (
    ARRIVAL_PATTERNS,
    GRID_MIN,
    N_CELLS,
    MenuBank,
    clock_labels,
    rasterise,
    sample_fleet,
)

RESEARCH_DIR = Path(__file__).parent
MAX_FLEET = 150
FLEET_SIZES = (40, 80, 150)
SEEDS = (11, 22, 33)


@dataclass
class CellResult:
    """Metrics for one scenario cell."""

    pattern: str
    preference: str
    fleet_size: int
    seed: int
    n_served: int
    peak_kw: float
    peak_time: str
    energy_kwh: float
    coincidence_factor: float
    mean_cost_rs: float
    mean_health: float
    mean_target_soc: float
    share_no_charge: float
    mean_ready_delay_h: float


def _choose(attributes: np.ndarray, betas: dict[str, np.ndarray], rng: np.random.Generator) -> int:
    """Return the index of the offer this customer picks.

    ``betas`` holds one customer's taste vector, each value already shaped (1,).
    """
    probabilities = choice_probabilities(utilities(attributes, betas))[0]
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return int((rng.random() > cumulative).sum())


def run_cell(
    pattern_name: str,
    preference_name: str,
    fleet_size: int,
    seed: int,
    bank: MenuBank,
    draws=None,
) -> tuple[CellResult, np.ndarray]:
    """Simulate one scenario cell and return its metrics plus load profile."""
    pattern = ARRIVAL_PATTERNS[pattern_name]
    distribution = PREFERENCE_MIXES[preference_name]
    if draws is None:
        draws = sample_fleet(MAX_FLEET, pattern, np.random.default_rng(seed))
    fleet = draws[:fleet_size]

    # Preference draws and choice noise get their own stream so that changing
    # the preference mix does not reshuffle the physical fleet.
    rng = np.random.default_rng((seed * 1000 + fleet_size) % (2**32))
    betas = distribution.draw(rng, len(fleet))

    load = np.zeros(N_CELLS, dtype=float)
    costs, healths, targets, delays = [], [], [], []
    served = 0
    no_charge = 0
    peak_individual = 0.0

    for index, customer in enumerate(fleet):
        record = bank.get(customer.key)
        if record is None:
            continue  # physically infeasible request; excluded from the cell
        served += 1
        customer_betas = {k: np.asarray([v[index]]) for k, v in betas.items()}
        attributes = menu_attribute_matrix(record)
        chosen = _choose(attributes, customer_betas, rng)
        offer = record.offers[chosen]
        if offer.energy_kwh <= 1e-9:
            no_charge += 1
        else:
            peak_individual += max(offer.power_kw)
        load += rasterise(record, chosen)
        costs.append(offer.charging_cost)
        healths.append(offer.health_score)
        targets.append(offer.target_soc * 100.0)
        delays.append(attributes[chosen, 2])

    if served == 0:
        raise RuntimeError(f"no feasible customers in cell {pattern_name}/{preference_name}")

    peak_cell = int(np.argmax(load))
    result = CellResult(
        pattern=pattern_name,
        preference=preference_name,
        fleet_size=fleet_size,
        seed=seed,
        n_served=served,
        peak_kw=float(load.max()),
        peak_time=clock_labels()[peak_cell],
        energy_kwh=float(load.sum() * GRID_MIN / 60.0),
        # Coincidence factor: fleet peak divided by the sum of the individual
        # peaks of the customers who actually charge. 1.0 = fully synchronised.
        coincidence_factor=float(load.max() / peak_individual) if peak_individual > 0 else 0.0,
        mean_cost_rs=float(np.mean(costs)),
        mean_health=float(np.mean(healths)),
        mean_target_soc=float(np.mean(targets)),
        share_no_charge=no_charge / served,
        mean_ready_delay_h=float(np.mean(delays)),
    )
    return result, load


def build_bank(limit: int | None = None) -> tuple[MenuBank, int]:
    """Ensure the menu bank covers the whole sweep. Returns (bank, still missing).

    Resumable: call repeatedly until it reports 0 missing. The bank saves
    atomically every 100 menus, so an interrupted pass loses at most that many.
    """
    bank = MenuBank()
    keys = sorted({c.key for draws in sweep_draws().values() for c in draws})
    remaining = bank.ensure(keys, limit=limit)
    print(f"  bank: {len(bank.menus)} cached, {len(bank.infeasible)} infeasible, "
          f"{remaining} still missing of {len(keys)}")
    return bank, remaining


def sweep_draws() -> dict[tuple[str, int], list]:
    """One shared fleet draw per (pattern, seed); fleet sizes are nested prefixes."""
    return {
        (p, s): sample_fleet(MAX_FLEET, ARRIVAL_PATTERNS[p], np.random.default_rng(s))
        for p in ARRIVAL_PATTERNS
        for s in SEEDS
    }


def main() -> None:
    bank, remaining = build_bank()
    if remaining:
        raise SystemExit(f"{remaining} menus still missing -- re-run to continue the build")

    draws_by = sweep_draws()
    results: list[CellResult] = []
    profiles: dict[str, np.ndarray] = {}
    for pattern in ARRIVAL_PATTERNS:
        for preference in PREFERENCE_MIXES:
            for size in FLEET_SIZES:
                for seed in SEEDS:
                    result, load = run_cell(
                        pattern, preference, size, seed, bank, draws=draws_by[(pattern, seed)]
                    )
                    results.append(result)
                    profiles[f"{pattern}|{preference}|{size}|{seed}"] = load
        print(f"  {pattern}: done", flush=True)

    (RESEARCH_DIR / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8"
    )
    np.savez_compressed(RESEARCH_DIR / "profiles.npz", **profiles)
    print(f"\nwrote {len(results)} cells -> results.json, profiles.npz")


if __name__ == "__main__":
    main()
