# Daily EV Menu — Commit 7

Commit 7 assembles Commit 6 fixed-request saving frontiers into one immutable,
deterministic customer menu.  The pipeline is:

1. validate the complete Commit 4 `GeneratedMenu`;
2. prune ready-step change points;
3. build exactly one Commit 6 frontier per retained request;
4. convert frontier points and BAUs into offers with explicit provenance;
5. normalize health across the complete structural offer pool;
6. filter non-positive savings, remove exact duplicates, compact requests,
   Pareto-filter, and select the bounded display menu.

## Request pruning

For each exact target SOC, only `minimum_cost` candidates are considered. Ready
steps are sorted and duplicate ready steps are rejected. Candidates with saving
at or below `positive_saving_tolerance` are excluded. The remaining maximum
savings must be nondecreasing within `pruning_saving_tolerance`; a material
decrease is rejected. A running maximum retains every strict increase. Each
equal-saving plateau retains its latest ready step, and the latest positive
request is always retained. Savings are currency values.

## BAU and provenance

Every target has exactly one immediate same-target BAU offer with zero saving.
Optimized offers carry immutable `OfferSource` metadata mapping:

```text
final offer ID -> frontier point ID -> source candidate ID -> endpoint role
```

BAUs use source role `bau`; optimized points preserve their Commit 6 endpoint
role. `source_frontiers` contains exactly the frontiers built for retained
requests and is also available as `AssembledMenu.frontiers`.

## Health stage order

All BAUs and all points from retained frontiers are scored together using total
degradation fade, Commit 5's tiny-spread policy, resolution, and half-up
quantization. Health is normalized once over that complete structural pool.
Only then are positive-saving filtering, exact-duplicate removal, request-local
compaction, Pareto filtering, and display selection applied. Health is not
renormalized after reduction, so removing an extreme-fade offer after this
stage does not change another offer's already-advertised score.

Non-BAUs are retained only when:

```text
saving > positive_saving_tolerance
```

Values are never clipped to zero. Targets without positive optimized offers
retain their BAU only. A no-charge target therefore has one zero-energy BAU and
no duplicate zero-saving optimized offer.

## Exact duplicates and compaction

Exact duplicate non-BAUs are removed first, including equal target, ready step,
saving, health, cost, profile, and degradation assessment. The lexicographically
smallest offer ID wins. This applies even when `saving_merge_gap == 0` and does
not cross request keys.

Compaction then operates only within `(target_soc, ready_step)`. Offers are
sorted by saving and ID and use deterministic chain-connected clusters: the next
offer joins the current cluster when its saving difference from the previous
offer is strictly less than `saving_merge_gap`. Thus `0.00, 0.09, 0.18` is one
cluster for a `0.10` gap. A difference exactly equal to the gap does not merge.

The cluster winner is selected by higher quantized health; health differences
within `health_tie_tolerance` use higher realized saving; a further tie uses the
lexicographically smaller offer ID. The winner keeps its own profile,
assessment, and provenance.

## Pareto rule

Offer `a` dominates `b` when it is no later, has no lower target, no lower
saving, and no lower health, with at least one strict improvement:

```text
ready_a <= ready_b
target_a >= target_b - target_dominance_tolerance
saving_a >= saving_b - saving_dominance_tolerance
health_a >= health_b - health_dominance_tolerance
```

Ready steps use exact integer comparison. SOC, saving, and health use their
separate tolerances and strict `>` comparisons beyond those tolerances. Exact
duplicates are removed before Pareto filtering. BAUs are policy-protected and
are never removed.

## Display selection

`display_cap` includes BAUs. The mandatory set contains every BAU, the global
maximum-saving non-BAU, and the global highest-health non-BAU. Anchor ties use:

- maximum saving, then higher health, earlier ready step, higher target, lower ID;
- maximum health, then higher saving, earlier ready step, higher target, lower ID.

If the distinct mandatory set cannot fit, assembly raises
`PhysicalConstraintError`; it never silently deletes a BAU or anchor. After
mandatory anchors are reserved, ready-step diversity is best-effort and is
processed in ascending ready-step order. Remaining slots use saving descending,
health descending, readiness ascending, target descending, and ID ascending.

Final presentation order is target SOC ascending, BAU before optimized, ready
step ascending, saving ascending, health descending, and offer ID ascending.
Selection priority and presentation order are intentionally separate.

Assembly settings use separate finite, nonnegative tolerances for pruning,
positive-saving filtering, compaction, saving dominance, health dominance,
target dominance, and health ties. Saving units are currency, target units are
SOC fractions, and health units are score points.

## Deferred beyond Commit 7

Commit 7 does not add customer-choice modelling, preference estimation,
stochastic realization, Monte Carlo, multi-day state coupling, fleet
aggregation, network simulation, plotting, reporting, file I/O, or experiment
scripts.

# Commit 8 — High-level single-EV menu generation

Commit 8 adds a user-facing service that converts an EV model, local arrival and
departure times, current SOC, and next-trip distance into the validated Commit
1–7 pipeline. Callers no longer need to construct `EVSpec`, `ChargingSession`,
`PlanningSignal`, or invoke the candidate and assembly layers manually.

```python
from evmenu import generate_ev_menu

menu = generate_ev_menu(
    ev_model="generic_40kwh_lfp",
    arrival_time="19:00",
    departure_time="07:00",
    current_soc=0.35,
    next_trip_distance_km=45.0,
)
```

The service uses a 15-minute grid by default and supports one overnight window.
Clock strings are strict five-character `HH:MM` values: whitespace, one-digit
fields, and invalid 24-hour values are rejected. Equal arrival/departure clock
times are rejected as ambiguous. Arrival and departure must align to the
selected timestep. Current SOC is fractional (`0.35` means 35%) and is converted to
battery energy, next-trip distance is converted using the selected model's
consumption assumption, and the default buffer is 10% of usable battery
capacity.

The built-in `research_tou` price schedule and generic EV catalogue entries are
**illustrative research assumptions**, not manufacturer specifications or a
regulated retail tariff. Deployment code should pass a custom `EVModel` with
verified values. Model lookup is case-sensitive and trims surrounding whitespace;
model search is case-insensitive, trims surrounding whitespace, and rejects empty
queries. A flat tariff is also supported,
including finite negative prices because the underlying research signal permits
them. The returned
`GeneratedCustomerMenu` retains the complete `AssembledMenu` for auditability
and exposes aligned `CustomerMenuRow` objects containing ready time, target SOC,
cost, saving, health, energy drawn, source role, and the full charging schedule.
`charging_schedule_kw` is grid-side charging power for each interval, whose
duration is given by `timestep_minutes`. Roles are `bau`, `least_degradation`,
`intermediate`, `maximum_saving`, and `least_and_maximum`.

Commit 8 does not add a CLI, external configuration files, manufacturer-data
scraping, customer-choice modelling, Monte Carlo realization, multi-day state
coupling, fleet/network simulation, plotting, or report generation.
Dates, time zones, and sessions longer than one overnight cycle are deferred.
