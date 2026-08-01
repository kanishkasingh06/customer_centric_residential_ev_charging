# Daily EV Menu — Commit 3

Commit 3 builds on the immutable Commit 1 schemas and Commit 2 feasibility and
validation contracts. It adds only deterministic, analytical charging-profile
construction; it does not add menu generation or customer-choice logic.

## Conventions

- `PlanningSignal` is an arbitrary contiguous horizon. Step `0` is its first
  represented interval, not necessarily midnight, and it must cover the full
  session window.
- Charging sessions use the half-open interval
  `[arrival_step, departure_step)`.
- Every constructed profile follows the exact-session convention. If
  `N = departure_step - arrival_step`, then power and grid-energy vectors have
  `N` entries and battery-energy and SOC vectors have `N + 1` boundary states.
  Local interval `k` maps to global step `session.arrival_step + k`.
- Charging is allowed only before `ready_step`; all intervals from
  `ready_step` through departure remain in the profile with zero power.
- Grid energy is `power_kw * timestep_hours`, and battery energy increases by
  `charging_efficiency * grid_energy_kwh`.
- `battery_capacity_kwh` is usable battery-side capacity (`B_max`), while
  `minimum_energy_kwh` is the battery floor (`B_min`).

## Commit 3 profile constructors

`build_immediate_charging_profile` allocates the exact required grid energy
chronologically from arrival, filling each interval up to charger capacity and
using a partial final interval when needed. Its ready step is the boundary at
which the target is reached. A no-charge request returns a full-session
zero-power profile ready at arrival.

`build_minimum_cost_charging_profile` solves the analytical linear allocation
problem before a requested ready step. It fills intervals in ascending
`(price_per_kwh, global_step)` order, so equal-price ties deterministically use
the earlier interval. Negative prices are supported, but the target remains
exact: the constructors never charge beyond the required energy merely to earn
revenue.

Both constructors support partial intervals and return a frozen
`ConstructedProfile` containing:

- the complete exact-session `ChargingProfile`;
- target SOC and ready-step metadata;
- required grid energy;
- charging cost;
- the independent Commit 2 `ValidationReport`.

Cost is calculated on grid energy using global signal indices:

```text
C = sum(price[global_step] * E_grid[local_step])
```

Every internally constructed profile is passed through the independent Commit
2 validator before being returned. A failed validation raises
`PhysicalConstraintError`; an invalid profile is never returned as successful.
`ConstructedProfile` is intended to be created by the public constructors;
direct manual dataclass construction does not itself perform physical
validation.

The optional `compute_buffer_energy(...)` helper is preprocessing for
constructing a `ChargingSession`. Once created, `session.buffer_energy_kwh` is
finalized data and is consumed directly by feasibility and validation.

## Deferred to Commit 4

The following are intentionally not implemented:

- menu generation, Pareto filtering, and intermediate saving levels;
- battery degradation or degradation-aware profiles;
- customer-choice modelling;
- Monte Carlo simulation;
- distribution-network simulation;
- plotting, file loading, or experiment scripts.

## Run checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pytest --cov=evmenu --cov-report=term-missing
ruff check .
ruff format --check .
mypy evmenu tests
```
