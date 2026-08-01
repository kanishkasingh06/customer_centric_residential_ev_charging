# Daily EV Menu — Commit 1

Commit 1 establishes immutable schemas and domain exceptions only. It does not
yet construct or physically validate charging trajectories.

Included:

- `EVSpec`
- `ChargingSession`
- `PlanningSignal`
- `TargetOption`
- `ChargingProfile`
- `MenuOffer`
- `MenuSettings`
- domain-specific exceptions and schema tests

Not included yet:

- physical trajectory validation
- charging-profile construction or optimization
- menu generation or customer choice
- Monte Carlo simulation
- distribution-network simulation
- degradation equations

## Conventions

- `PlanningSignal` represents an arbitrary contiguous planning horizon. Step 0
  is its first represented interval; it is not necessarily midnight.
- Residential overnight sessions are supported when the planning horizon
  includes every interval through `departure_step`.
- Charging windows are half-open: `[arrival_step, departure_step)`. Arrival is
  inclusive and departure is exclusive.
- Ready-step restrictions are not enforced in Commit 1. A later cross-object
  physical validator will enforce charging only before an offer's ready step.
- Grid-side charging power is measured in kW and grid-side interval energy in
  kWh. Battery energy and SOC are battery-side state quantities.
- For `N` charging intervals, `power_kw` and `grid_energy_kwh` have `N` values;
  `battery_energy_kwh` and `soc` have `N + 1` boundary-state values.
- `battery_capacity_kwh` is usable battery-side maximum energy `B_max`, not
  nominal/nameplate capacity. `minimum_energy_kwh` is the absolute battery
  floor `B_min`.
- `charging_efficiency` is grid-to-battery efficiency:
  `battery energy increase / grid energy drawn`.
- Prices are finite and may be negative. Base load is non-negative.
- `battery_temperature_c` is representative battery/pack temperature in
  degrees Celsius, not ambient temperature.
- Schemas reject locally invalid values without clipping. Cross-object checks
  such as charger limits, energy recursion, SOC consistency, signal alignment,
  and ready-time restrictions remain intentionally unimplemented.

## Run checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pytest --cov=evmenu --cov-report=term-missing
ruff check .
ruff format --check .
mypy evmenu tests
```
