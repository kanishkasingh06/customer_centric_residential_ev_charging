# Daily EV Menu — Commit 10

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

The service uses a 15-minute nominal wall-clock grid by default and supports
arbitrary minute-level arrival and departure values, including overnight
windows. Clock strings are strict five-character `HH:MM` values: whitespace,
one-digit fields, and invalid 24-hour values are rejected. Equal
arrival/departure clock times are rejected as ambiguous. Boundaries are never
rounded. A partial first or final interval is represented explicitly, and
intervals are split again at every tariff boundary. Current SOC is fractional
(`0.35` means 35%) and is converted to battery energy, next-trip distance is
converted using the selected model's consumption assumption, and the default
buffer is 10% of usable battery capacity.

For example, both of these built-in runs preserve their exact clock boundaries:

```bash
evmenu generate --ev-model generic_40kwh_lfp \
  --arrival 11:07 --departure 18:52 --current-soc 35 --next-trip-km 45
evmenu generate --ev-model generic_40kwh_lfp \
  --arrival 23:53 --departure 07:08 --current-soc 35 --next-trip-km 45
```

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
`charging_schedule_kw` is grid-side charging power for each interval. Its
aligned `interval_start_times`, `interval_end_times`, and
`interval_duration_minutes` make the schedule self-describing; the nominal
`timestep_minutes` is not necessarily each interval's duration. Roles are `bau`, `least_degradation`,
`intermediate`, `maximum_saving`, and `least_and_maximum`.

Commit 8 did not add a CLI, external configuration files, manufacturer-data
scraping, customer-choice modelling, Monte Carlo realization, multi-day state
coupling, fleet/network simulation, plotting, or report generation. Commit 10
adds exact local clock-minute handling and optional explicit arrival
weekday/date metadata; it still does not add time-zone conversion or multi-day
battery-state coupling.

## Command-line interface

Commit 9 adds a standard-library command-line interface. The installed console command and
module entry point are equivalent:

Install the project from its checkout with:

```bash
python -m pip install .
```

```bash
evmenu generate \
  --ev-model generic_40kwh_lfp \
  --arrival 19:00 \
  --departure 07:00 \
  --current-soc 35 \
  --next-trip-km 45
```

```bash
python -m evmenu generate \
  --ev-model generic_40kwh_lfp \
  --arrival 19:00 \
  --departure 07:00 \
  --current-soc 35 \
  --next-trip-km 45
```

Unlike the Python service API, CLI SOC values are percentages: `35` means 35%, and the default
`--buffer-soc 10` means 10% of usable capacity. Clock values remain strict local `HH:MM` strings.

The required `generate` arguments are `--ev-model`, `--arrival`, `--departure`,
`--current-soc`, and `--next-trip-km`. Optional defaults are:

- `--buffer-soc 10` (percentage of usable capacity);
- `--tariff research_tou`;
- `--flat-price 7.0` currency/kWh when `--tariff flat` is selected;
- `--temperature-c 30` degrees Celsius;
- `--timestep-minutes 15`;
- no `--display-cap` limit;
- `--format text`.

`--flat-price` is valid only with `--tariff flat`; finite negative flat prices are supported.
`--include-schedule` and `--include-intervals` are valid only with `--format json`. Text cost and
saving values are in currency units, health is a score from 0 to 100, and schedules are grid-side
kW values aligned to the returned interval metadata. Add `--include-intervals` to expose exact
boundaries, durations, and prices in JSON. The default text output intentionally stays compact.

Machine-readable output is available without exposing internal schema objects:

```bash
evmenu generate ... --format json
```

Schedules are omitted from JSON by default to keep output compact. Add `--include-schedule` to
include `charging_schedule_kw`, the grid-side power for each generated interval.
JSON numbers retain the unrounded service values; the text table rounds only for display.

Other commands:

```bash
evmenu models
evmenu tariffs
```

`evmenu models` reports each model's capacity, charger power, chemistry, consumption, and
illustrative-assumption note. `evmenu tariffs` reports the illustrative research TOU period
boundaries and prices plus the caller-configurable flat tariff. Neither command makes a
manufacturer, official, current, or regulated-price claim. Domain and physical errors are printed
to standard error and return exit status 2. Argument syntax errors use the same exit status.

## Commit 10 — exact times and numerical custom prices

The canonical interval representation is the immutable `TimeInterval` with
integer absolute-minute `start_minute` and `end_minute`. `build_time_intervals`
adds exact arrival/departure boundaries, wall-clock nominal-grid boundaries,
and tariff/profile boundaries, then constructs continuous half-open intervals
`[start, end)`. A 23:53→07:08 request therefore has a 435-minute exact
connection and 7-minute/8-minute boundary intervals; it is not converted to a
rounded 15-minute request. There is no one-minute simulation: interval count
remains approximately connection duration divided by the nominal grid plus
only boundary splits.

Built-in tariffs are `research_tou` (illustrative 00:00–06:00=4,
06:00–17:00=7, 17:00–23:00=10, 23:00–24:00=5) and `flat`. Every optimization
interval has one constant price. Prices are multiplied by interval grid energy
only; no extra duration factor is used. Grid-side energy is bounded by
`charger_power_kw * interval_duration_hours[k]`, and power is reconstructed as
`energy / interval_duration_hours[k]`.

### Numerical custom profiles

Custom prices must be supplied as machine-readable numerical CSV. The service
API accepts immutable `WeeklyPriceProfile` or `TimestampedPriceProfile`
objects; CSV parsing is kept at the CLI/helper boundary. Negative finite prices
are valid. A recurring weekly profile must cover every minute of every weekday,
and requires an explicit weekday (do not guess it):

```csv
day_of_week,start_time,end_time,price
Mon,00:00,01:00,3.5
Mon,01:00,02:00,3.0
...
Sun,23:00,24:00,4.3
```

The compact hourly form is also supported and must contain all 168 hours exactly
once:

```csv
hour_of_week,price
0,3.5
1,3.0
...
167,4.3
```

Use it from the CLI with an explicit arrival day:

```bash
evmenu generate --ev-model generic_40kwh_lfp \
  --arrival 11:07 --departure 18:52 --current-soc 35 --next-trip-km 45 \
  --tariff custom --price-profile weekly_prices.csv \
  --price-profile-format hour_of_week --arrival-day Mon \
  --format json --include-intervals
```

Timestamped CSV uses the explicit, unambiguous schema
`start_time,end_time,price` with ISO-8601 datetimes and contiguous coverage;
the CLI selects it with `--price-profile-format timestamped` and requires
`--arrival-date YYYY-MM-DD`. Timestamped files must use either all naive
datetimes or all timezone-aware datetimes, and the session query must use the
same semantics. Commit 10 performs no timezone conversion. Recurring weekly
profiles use a weekday plus local clock time and do not require a timezone.
Weekly sessions map each boundary to absolute
minute-of-week, including overnight weekday crossings. Malformed, incomplete,
overlapping, or uncovered profiles fail before any menu output is written.

Custom CSV profiles are intentionally small tabular inputs; files larger than
10 MiB are rejected before parsing. Headers and column order are exact, and
trailing or missing fields are rejected rather than ignored.
Custom profiles may carry a validated nonempty currency label (for example,
`Rs`, `USD`, or `EUR`); it is preserved in JSON and used by the text table
without inferring an official currency symbol.

A plotted image cannot be used as the numerical tariff input. The project does
not digitize the uploaded plot or guess values from pixels; export the
underlying numerical series to CSV so runs are reproducible and auditable.

Readiness follows Policy A (boundary-ready semantics): ready times are exact
generated interval boundaries, and charging is allowed only for intervals
before the ready-step boundary. Continuous within-interval completion times
are intentionally deferred until a future commit rather than inferred from an
average interval power.

Commit 10 does not add configuration-file input, manufacturer-verified data, live tariff retrieval,
customer-choice modelling, Monte Carlo realization, multi-day coupling, fleet/network simulation,
plotting, or reporting.
