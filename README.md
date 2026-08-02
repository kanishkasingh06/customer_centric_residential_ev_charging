# Daily EV Menu — Commit 6

Commit 6 adds deterministic, fixed-request degradation-aware trajectory
optimization and saving-frontier construction. It does not add Commit 7
cross-request menu assembly or customer-choice functionality.

## Fixed-request model

For one Commit 4 `minimum_cost` candidate, the decision variables are grid-side
energy values `E_grid[k]` for intervals
`arrival_step <= k < ready_step`:

```text
0 <= E_grid[k] <= charger_power_kw * timestep_hours
sum(E_grid[k]) = max(0, target_soc * B_max - B_initial) / eta
```

The returned profile spans the complete session, has zero charging at and
after `ready_step`, and is independently checked by the Commit 2 physical
validator.

The optimized trajectory objective is:

```text
J = charging-window calendar fade
    + plating_guard_weight
      * sum((eta * p_grid[k] / B_max)^2 * timestep_hours)
```

The calendar term uses beginning-of-interval SOC and the signal's global
temperature indices. Parked-day and cycle fade remain in the Commit 5
assessment after optimization; `trajectory_objective`/`objective_value` is not
full total degradation. Peak C-rate can therefore make total-fade ordering
different from trajectory-objective ordering.

SciPy SLSQP receives `objective_scale * J` and an analytical Jacobian. Scaling
is numerical only: returned objective values remain the original physical `J`.
Solver success is never trusted without raw-vector checks, profile validation,
cost recomputation, saving-band checks, and an independent objective
recalculation.

## Saving constraints

The analytical Commit 4 minimum-cost profile is preserved exactly as the
maximum-saving endpoint:

```text
S_max = C_BAU - C_min_cost
```

Before solving, Commit 6 computes the attainable saving interval using
ascending-price and descending-price allocations. A requested band must
intersect that interval. Negative savings and negative prices are valid.

For requested saving `s` and band `delta`:

```text
s - delta <= C_BAU - C(profile) <= s + delta
```

Zero-width bands are supported when the requested cost is exactly attainable.
Least-degradation uses a deterministic evenly spread feasible start; constrained
solves interpolate between feasible minimum- and maximum-cost allocations so
SLSQP never starts outside the saving band.

## Results and frontier

Each `OptimizedProfile` has:

- immutable validated profile and Commit 5 assessment;
- unscaled `trajectory_objective` plus `objective_value` alias;
- realized and requested savings;
- source candidate ID, deterministic `point_id`, EV identity, and endpoint role.

Endpoint roles are `least_degradation`, `intermediate`, `maximum_saving`, or
`least_and_maximum`. Exact duplicate endpoint/profile points collapse to one
`least_and_maximum` point; this is exact duplicate handling, not Commit 7
display compaction.

`build_sandwich_saving_frontier` preserves the analytical maximum endpoint,
then repeatedly requests the midpoint of the largest unresolved realized
saving gap. Ties are deterministic. `maximum_levels` includes endpoints. The
algorithm stops at the level cap or when every gap is no larger than
`max(2 * saving_band, frontier_gap_tolerance)`, and rejects material
trajectory-objective decreases along the useful branch.

## Public API

- `FrontierSettings`
- `OptimizedProfile`
- `SavingFrontier`
- `build_least_degradation_profile(...)`
- `build_saving_constrained_profile(...)`
- `build_sandwich_saving_frontier(...)`

`FrontierSettings` separates objective scaling, solver controls, bound/energy/
saving/objective tolerances, frontier-gap tolerance, cost tolerance, and
plating-guard weight. All numeric values reject booleans, NaN, and infinity.

## Deferred to Commit 7

- ready-step change-point pruning;
- cross-request menu assembly;
- savings-gap display compaction;
- Pareto filtering;
- display caps;
- customer choice;
- Monte Carlo simulation;
- distribution-network simulation;
- plotting; and
- file I/O.
