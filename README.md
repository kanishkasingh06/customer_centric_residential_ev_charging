# Daily EV Menu — Algorithm 1 rich generation and paper filtering

The current stage implements Algorithm 1 through the rich generated offer set:

1. parse exact session and tariff intervals;
2. construct one immediate, same-target BAU for every target SOC;
3. evaluate every feasible ready-boundary/target request and construct its
   minimum-cost profile;
4. calculate request-level maximum saving (`Smax`);
5. select deterministic value-spaced saving levels (`saving_step`, `Kmax`);
6. solve battery-stress minimization inside each saving band;
7. independently validate trajectories, assign roles and provenance, retain
   BAU plus positive-saving offers, and remove exact scientific duplicates.

`MenuSettings` is immutable and owns the Algorithm 1 defaults: saving step
`5.0`, at most five levels per request, a `0.50` currency-unit saving band,
and the existing numerical zero tolerance.  `select_saving_levels` preserves
the lowest positive region, intermediate regions, and the exact maximum
endpoint; when Kmax is smaller it selects deterministic indices over the full
raw range (for example `(5, 15, 30, 40)`).

Each offer retains requested and achieved saving, same-target BAU cost, actual
cost, target energy, charging trajectory, raw battery stress, optional RUL/loss
fields, and immutable provenance flags. `BatteryStress` is schedule-intrinsic
and lower is better; it is never normalized against the other offers in a
menu, a Pareto set, or a display cap. The legacy `charging_health_score` field
is retained only for source compatibility and is not serialized or used by the
rich scientific logic.

`generated_offers` means BAU plus every valid positive-saving Algorithm 1 offer
after exact scientific duplicate removal. It is immutable and remains
available even after customer-facing filtering. The paper-consistent pipeline
is:

```text
generated_offers (Fi)
  -> retained_offers (Fbar_i: BAU + positive saving)
  -> compacted_offers (Ftilde_i: savings near-duplicate compaction)
  -> pareto_offers (Pi: Pareto-efficient offers + preserved BAU anchors)
  -> displayed_offers (Mi: deterministic customer-facing diversity selection)
```

The default `menu_stage` is `displayed`, so the default customer menu is a
deterministic diversity-selected subset of the compacted Pareto set containing
only positive-saving, non-BAU customer options. One same-target BAU reference
per feasible target remains available separately as `bau_reference_offers`; it
never consumes a customer-option slot. The rich set is still exposed as
`generated_offers`; it is never overwritten by a derived stage. Use
`menu_stage="generated"`, `"compacted"`, `"pareto"`, or `"displayed"` in
Python, or `--menu-stage` in the CLI. `--menu-stage pareto` exposes all Pareto
offers (including preserved BAU anchors); only the displayed stage applies the
customer-facing diversity limits.

Compaction uses the actual achieved saving, not requested or display-rounded
saving. It is performed only within an exact fixed request group identified by
absolute ready-by minute and target battery energy. The default
`delta_saving_merge` is `5.0` currency units, independent of the `0.50`
currency-unit saving-band feasibility tolerance. Saving buckets use the
deterministic convention
`floor((actual_saving + numerical_tolerance) / delta_saving_merge)`; a zero
threshold disables near-duplicate compaction. The lowest-stress offer in each
bucket is retained, and the true maximum-saving endpoint is always preserved.
Compaction never merges different ready promises or different target SOCs.

Pareto dominance uses absolute ready minute, target energy, actual saving, and
raw schedule-intrinsic battery stress. Earlier readiness, higher target, higher
saving, and lower stress are better. Values are compared with explicit
tolerances and without requested saving, display rounding, normalized health,
role labels, or weighted sums. A BAU reference can be mathematically
dominated; it is still preserved and marked `is_preserved_bau_anchor`, while
`is_pareto_efficient` independently reports mathematical efficiency.

### Final customer-facing diversity selection

The final stage selects `M_i` from `P_i` without changing any scientific offer
stage. `maximum_displayed_offers=12` means at most 12 non-BAU customer options;
BAU references are separate scientific baselines. A generous safety cap of 12
offers per target prevents pathological concentration, but the allocation is
not mechanically `four targets × three offers`: after one maximum-saving and,
when meaningfully distinct, one least-degradation anchor per target, remaining
slots are selected globally by deterministic farthest-point marginal distance.
Tie-breaking is larger distance, larger saving, lower raw fade, earlier ready
time, higher target SOC, then scientific ID. No BAU is selected by this stage.

Distinctness uses the equal-weight Euclidean feature vector
`[ready_delay, target_soc, actual_saving, total_capacity_fade]`. Ready delay,
saving, and fade use dynamic min–max scaling over the candidate/display set;
target SOC uses its fixed physical `[0,1]` domain; zero ranges map to zero.
The raw scientific attributes are never changed and no normalized health score
is introduced. Pairwise component distances, overall distances, nearest
neighbours, marginal selection contributions, labels (`highly_distinct`,
`moderately_distinct`, `weakly_distinct`, `near_duplicate`), ranges, target
coverage/entropy, and near-duplicate pairs are exposed in Python and optional
diagnostic JSON. These geometric diagnostics are not utility or behavioural
claims. The minimum-overall-distance guard is transparent and configurable;
it does not turn readiness spacing into an automatic rejection rule.

For manual inspection, this checkout includes deterministic, unversioned
inspection artifacts under `artifacts/menu_distinctness/`: complete displayed
and BAU-reference CSVs, pairwise-distance CSVs, per-EV JSON summaries, and a
Markdown report for three representative profiles. These are generated
inspection outputs, not runtime inputs or a new reporting subsystem.

Compaction removes near-duplicate saving choices only within the same
ready-by time and target SOC request. It does not merge offers across
different readiness promises or different SOC targets. The current displayed
menu is the compacted Pareto set's globally diversity-selected non-BAU options;
same-target BAU anchors are available only in `bau_reference_offers`. A later
display-diversity stage may further reduce closely spaced ready-time choices
only if this policy is intentionally changed; it must not change the
scientific offer set.

Saving levels and saving bands use absolute currency units. The optional
relative-step policy is intentionally disabled; `saving_step` is never silently
scaled by BAU cost. Every returned optimized offer records
`requested_saving`, `saving_band_lower`, `saving_band_upper`, actual saving, and
`saving_band_violation`. Request enumeration diagnostics reconcile
`total = feasible + infeasible` and `feasible = positive-saving + no-saving`.
Optimizer diagnostics separately count attempts, successes, infeasible bands,
validation failures, and solver failures. Use the Python result's diagnostics
or `--include-diagnostics` in JSON for these details; normal text output does
not expose solver internals.

The active degradation quantity is incremental capacity fade, not a
menu-relative health score:

```text
DeltaQ_total = DeltaQ_calendar + DeltaQ_cycle
DeltaQ_calendar = DeltaQ_connected_window + DeltaQ_parked_period
```

All terms are fractions of nominal usable capacity (`2e-5` means `0.002%`).
LFP offers use the Naumann parameter family and NMC offers use the Schmalstieg
parameter family. Calendar aging uses the convex chemistry-specific SOC shape,
Arrhenius temperature scaling, and the local calendar-age slope. Cycle aging
uses battery-side throughput, DoD, peak C-rate, and the local accumulated-FEC
slope. Active IDs are `naumann_lfp_capacity_fade_anchor_v1` and
`schmalstieg_nmc_capacity_fade_anchor_v1`; “anchor” explicitly means a
literature-family calibration, not an exact reproduction of published fitted
coefficients.

The reporting model includes the exact peak-C-rate cycle factor after each
trajectory is solved. The SLSQP frontier objective uses a labelled convex
optimization surrogate (`*_optimization_surrogate_v1`) consisting of the
trajectory-dependent calendar term and a convex C-rate guard; fixed throughput
and parked-period constants are added back by reporting. Diagnostics expose
both model IDs and state that the exact reporting model is not silently claimed
to be the solver objective.

The supplied checkout does not contain the cited papers' fitted coefficient
tables, so immutable defaults are labelled `literature_anchor_calibrated` with
citations; they are not silently presented as exact published fits.
Temperature, battery age, and accumulated FEC are carried in every evaluation.
At zero accumulated FEC the square-root cycle derivative is regularized with
the explicit immutable `minimum_effective_fec` floor (default `1.0` FEC).
The older `minimum_reference_fec` spelling is only a compatibility alias and
cannot define a second floor.
Diagnostics expose the raw starting FEC, effective FEC, floor, whether the
floor was active, and its source; the floor changes only the local derivative,
not the modeled fade components. The standalone service uses an explicit
16-hour post-commute parked-period assumption by default and serializes both
`parked_period_hours` and `parked_period_source` (`assumed_default` versus
`user_input`). Pass `daytime_parked_hours` (or
`DegradationSettings(parked_day_hours=None)`) when that period is not
defensible; disabled parked fade is marked `deferred_not_modeled`. Connected-
window and parked-period fade are calendar-aging subcomponents, not separate
physical mechanisms.

The legacy `battery_stress`/`BatteryStress` field is retained as a deprecated
alias of `total_capacity_fade`; it cannot diverge and is lower-is-better.
`charging_health_score` remains only for source compatibility and is a
deterministic capacity-fade display alias; the removed `health_score_resolution`
and min–max helper are rejected/deleted, so no normalized-health path remains.
The model does not
apply an artificial low-SOC aging penalty, does not claim vehicle-specific RUL
or warranty outcomes, and does not use a weighted cost/degradation objective.
Provenance flags include `is_bau`, `is_low_saving`,
`is_intermediate`, `is_maximum_saving`, `is_least_degradation`, and
`is_endpoint`; the primary role is deterministic (`bau`, `least_and_maximum`,
`least_degradation`, `maximum_saving`, `low_saving`, `intermediate`) and is not
inferred from row position.

Scientific duplicates use unrounded, tolerance-quantized absolute ready
boundary, target energy, actual saving, raw stress, and charging-energy
trajectory. Display duplicate-looking rows are collapsed deterministically at
serialization time, preferring lower stress, lower cost, and then the
lexicographically smallest trajectory. Internal provenance is preserved.

The nominal 15-minute grid controls physical scheduling resolution. It does not
determine final customer-facing spacing of menu options. This stage applies
paper-consistent saving compaction, Pareto filtering, and the deterministic
customer-facing diversity selection described above; it does not add
customer-choice modelling.

## Deferred beyond the filtering stage

This filtering stage does not add customer-choice modelling, preference estimation,
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
requested/actual saving, cost, raw battery stress, provenance flags, and the
full charging schedule.
`charging_schedule_kw` is grid-side charging power for each interval. Its
aligned `interval_start_times`, `interval_end_times`, and
`interval_duration_minutes` make the schedule self-describing; the nominal
`timestep_minutes` is not necessarily each interval's duration. Roles are
`bau`, `low_saving`, `intermediate`, `least_degradation`, `maximum_saving`, and
`least_and_maximum`.

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
- `--battery-temperature-c 25` degrees Celsius (`--temperature-c` remains an alias);
- `--battery-age-years 1` and `--accumulated-fec 0`;
- optional `--daytime-parked-hours` for an explicit post-commute horizon;
- `--timestep-minutes 15`;
- `--menu-stage displayed` (the diversity-selected non-BAU Pareto stage);
- `--max-displayed-offers 12` (or `--maximum-displayed-offers`; non-BAU options only);
- `--max-offers-per-target 12` (a generous safety cap, not a fixed allocation);
- `--exclude-bau-from-display` (the default; BAU rows are references, not choices);
- `--include-bau-references` to include separate same-target BAU rows in JSON;
- `--include-distinctness-diagnostics` to include feature scaling, pairwise distances,
  nearest-neighbour metrics, and selection diagnostics;
- `--min-ready-separation-minutes 120`, `--min-saving-difference 5`,
  `--min-saving-fraction-of-bau 0.05`, and `--min-relative-stress-difference 0.02`;
- `--format text`.

`--flat-price` is valid only with `--tariff flat`; finite negative flat prices are supported.
`--include-schedule` and `--include-intervals` are valid only with `--format json`. Text cost and
saving values are in currency units, and `TotalFade` is an absolute incremental capacity-fade
fraction (lower is better). Schedules are grid-side kW values aligned to the returned interval metadata. Add `--include-intervals` to expose exact
boundaries, durations, and prices in JSON. The default text output intentionally stays compact.
Use `--include-diagnostics` with JSON to include reconciled request counts,
optimizer outcome counts, per-target completeness summaries, and structured
saving-level failures. `--include-pipeline-stages` additionally serializes all
generated, retained, compacted, Pareto-efficient, preserved-BAU, Pareto,
displayed, and BAU-reference arrays. Uncapped text output warns when more than 100 rich offers
were generated; it is never silently truncated. Text output reports every
pipeline count and notes that BAU references may be mathematically dominated.

Machine-readable output is available without exposing internal schema objects:

```bash
evmenu generate ... --format json
```

Schedules are omitted from JSON by default to keep output compact. Add `--include-schedule` to
include `charging_schedule_kw`, the grid-side power for each generated interval.
JSON numbers retain the unrounded service values; the text table rounds only for display.
Diagnostics expose the selected/rejected reason, target rank, diversity score,
pairwise distinctness, nearest option, and per-target selection summaries. The
text table includes `Role` and `SelectedAs`; `TotalFade` is raw incremental
capacity fade, where lower is better. BAU references are never customer
options and may be mathematically dominated.

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
