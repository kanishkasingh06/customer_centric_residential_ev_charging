# Daily EV Menu — Commit 2

Commit 2 adds deterministic feasibility calculations and an independent
cross-object validator on top of the immutable Commit 1 schemas. It does not
construct or optimize charging profiles.

## Conventions

- `PlanningSignal` is an arbitrary contiguous horizon: step `0` is its first
  represented interval, not necessarily midnight, and it must cover the full
  session window.
- Charging windows are half-open: `[arrival_step, departure_step)`. Arrival is
  inclusive and departure is exclusive.
- A Commit 2 profile must start exactly at `session.arrival_step` and cover
  exactly every interval through, but not including, `session.departure_step`.
  For `N = departure_step - arrival_step`, `power_kw` and
  `grid_energy_kwh` have `N` entries, while `battery_energy_kwh` and `soc` have
  `N + 1` boundary states.
- Local profile interval `k` maps to global planning step
  `profile.start_step + k`; with the exact-session rule this is
  `session.arrival_step + k`.
- `ready_step` is a boundary state. Charging is forbidden for every interval
  whose step is at or after `ready_step`; therefore a profile ready at arrival
  is valid only when no charging is needed, and `ready_step == departure_step`
  is represented by the terminal state.
- Grid-side power and interval energy are distinct from battery-side energy.
  Battery increases by `charging_efficiency * grid_energy_kwh`.
- `battery_capacity_kwh` is usable battery-side capacity (`B_max`), and
  `minimum_energy_kwh` is the absolute battery floor (`B_min`).

## Feasibility

The population buffer rule is applied once, before constructing a
`ChargingSession`:

```text
b = max(base_buffer_kwh, commute_buffer_fraction * commute_energy_kwh)
```

The resulting `buffer_energy_kwh` is finalized session data. Feasibility and
validation consume that value directly and never apply the rule again. Public
numeric inputs reject booleans, strings, NaN, and infinities.

Commit 2 uses the fixed standard targets `(0.80, 0.90, 1.00)`. Personalized
`z_min = (B_min + commute + buffer) / B_max` is never clipped: values above
`1.0` are a physical error, while exactly `1.0` is valid. Standard targets
below the service requirement are omitted. Targets within the configurable
`target_merge_tolerance` are merged, retaining the larger numeric target and
all deterministic provenance sources.

## Independent validation

`validate_charging_profile` returns a `ValidationReport` for physically
incompatible but well-formed EV/session/signal/profile combinations. The
report's primary API is an ordered tuple of frozen `ValidationIssue` objects
with stable `ValidationCode` values, interval indices where applicable, and
observed/expected diagnostics. `report.is_valid` is derived from whether the
issue tuple is empty; `report.errors` remains a compatibility view of messages.

`ValidationTolerances` separates strictly positive finite tolerances for
charger power (`power_kw`), energy identities and bounds (`energy_kwh`), and
SOC consistency (`soc`).

## Deferred to Commit 3

Charging-profile construction, optimization, menu generation, cost and
degradation calculations, customer choice modelling, Monte Carlo simulation,
and distribution-network power flow are intentionally not implemented.

## Run checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pytest --cov=evmenu --cov-report=term-missing
ruff check .
ruff format --check .
mypy evmenu tests
```
