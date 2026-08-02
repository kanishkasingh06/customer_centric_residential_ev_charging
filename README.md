# Daily EV Menu — Commit 5

Commit 5 annotates the existing Commit 4 candidate menu with chemistry-aware,
additive degradation assessments and a deterministic relative health score. It
does not construct new charging trajectories or remove existing candidates.

## Degradation units and model

Every fade value is a fraction of usable battery capacity: `0.01` means 1%
capacity fade. For each existing candidate:

```text
total_fade = window_calendar_fade + parked_day_fade + cycle_fade
```

Calendar SOC stress is chemistry-specific:

```text
g(s) = a0 + a1*s + a2*max(s - s_knee, 0)^2
```

LFP and NMC parameters are separate, immutable, case-sensitive parameter sets.
The defaults reproduce representative 30°C, one-year calendar anchors:

- LFP: 1.00%/year at 50% SOC and 1.24%/year at 100% SOC;
- NMC: 1.78%, 2.41%, and 3.02%/year at 50%, 80%, and 100% SOC.

These are representative calibrated scenario defaults, not publication-grade
cell-specific fitted coefficients.

Calendar temperature scaling uses Celsius inputs converted once to Kelvin and
the Arrhenius expression. Calendar age uses the local time-power-law slope
relative to a one-year reference age. Age is measured in years.

The charging-window term samples exactly the beginning-of-interval SOC states
(`profile.soc[:-1]`) for the `N` session intervals. The terminal state is not
counted as an additional interval. The parked-day temperature proxy is the
final in-session signal temperature at `session.departure_step - 1`.

Parked SOC is calculated as:

```text
parked_energy = max(target_soc * B_max, initial_energy) - commute_energy
parked_soc = parked_energy / B_max
```

Commute energy is battery-side, charging efficiency is not applied to it, and
buffer energy is not subtracted again. Values outside `[0, 1]` are rejected;
they are never silently clipped.

Cycle fade uses battery-side session throughput, depth fraction, and battery-
side peak C-rate:

```text
throughput = eta * sum(grid_energy)
peak_c_rate = eta * max(grid_power) / B_max
```

C-rate has units of `h^-1`, conventionally written as C. Zero cumulative FEC is
allowed. The local cycle-age slope uses a configurable positive
`minimum_reference_fec` regularization so a new battery remains finite; zero
throughput still produces zero cycle fade.

Annualized degradation is scenario-based:

```text
annualized_degradation_pct = total_fade * equivalent_sessions_per_year * 100
```

The default is 300 equivalent service sessions per year. Each assessment
represents one charging window, the configured parked-day dwell, and one cycle
contribution. Parked fade is included once per equivalent session. This is not
an absolute laboratory life prediction.

## Health score

For one EV/session, all generated candidates are normalized together:

```text
spread = max_fade - min_fade
normalized = (fade - min_fade) / spread
raw_health = 100 * (1 - normalized)
```

If `spread` is at or below `degradation_comparison_tolerance`, every candidate
receives 100. Otherwise scores are quantized with explicit half-up rounding:

```text
quantized = resolution * floor(raw_health / resolution + 0.5)
```

A tiny floating-point epsilon is used at exact boundaries, then scores are
clamped to `[0, 100]`. The default resolution is 5 points. Health is a
within-menu relative score, not absolute state of health, remaining battery
life, or a cross-EV comparison metric.

## Public workflow

```python
candidate_menu = generate_candidate_menu(
    ev=ev,
    session=session,
    signal=signal,
)

scored_menu = score_generated_menu(
    ev=ev,
    session=session,
    signal=signal,
    menu=candidate_menu,
)
```

`scored_menu.offers` preserves candidate count, order, IDs, candidate metadata,
costs, savings, profiles, and validation reports. Every offer has one matching
immutable degradation assessment.

## Still deferred to Commit 6

- least-degradation trajectory optimization;
- intermediate saving-health profiles;
- anchored saving levels;
- Pareto filtering;
- display caps;
- customer-choice modelling;
- Monte Carlo simulation;
- distribution-network simulation;
- plotting; and
- file I/O.
