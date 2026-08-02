# Daily EV Menu — Commit 4

Commit 4 adds deterministic pre-degradation candidate-menu generation on top
of the validated schemas, feasibility equations, physical validator, and
analytical profile constructors from Commits 1–3.

## Candidate-menu flow

For each EV/session, the generator uses target options from Commit 2's
feasibility layer and:

1. constructs the personalized commute-plus-buffer target and valid 80%, 90%,
   and 100% standard targets;
2. constructs exactly one immediate same-target charging profile as the BAU
   reference for every target;
3. enumerates every exactly feasible ready boundary;
4. constructs the analytical minimum-cost profile for each feasible request;
5. calculates same-target saving
   `S = C_BAU(target) - C_candidate`;
6. independently validates every trajectory; and
7. applies exact deterministic deduplication to minimum-cost candidates.

The public entry point is:

```python
menu = generate_candidate_menu(
    ev=ev,
    session=session,
    signal=signal,
)
```

It returns a frozen `GeneratedMenu` containing frozen `MenuCandidate` objects.
Each candidate stores target provenance, ready step, charging cost, its
same-target BAU cost, saving, required grid energy, the full-session profile,
and an independent validation report.

## Candidate semantics

- BAU is always included. Every target has exactly one `immediate_bau`
  candidate, with `saving == 0.0` and
  `same_target_bau_cost == charging_cost`. The BAU reference is never removed,
  even when a minimum-cost profile is physically identical.
- Minimum-cost schedules use ascending `(price, global_step)` order and are
  considered at every exactly feasible ready boundary.
- Savings are always measured against immediate charging to the **same target**.
  A negative saving is valid and is retained exactly: it means an earlier or
  otherwise tighter ready promise costs more than the full-session BAU
  reference. Non-finite savings are rejected; savings are never clipped to
  zero.
- Negative prices are supported, but charging remains target-exact; candidates
  never overcharge to earn revenue.
- A profile spans the complete session `[arrival_step, departure_step)`.

## Exact deduplication and ordering

By default, only `minimum_cost` candidates within one target are deduplicated.
Two such candidates are duplicates only when exact equality holds for target
SOC, target provenance, candidate type, every profile vector
(`grid_energy_kwh`, `power_kw`, `battery_energy_kwh`, and `soc`), charging cost,
and saving. No tolerance-based profile matching is used. This exact rule is
deliberate because the Commit 3 constructors are deterministic. When repeated
minimum-cost requests produce the same key, ready steps are enumerated in
ascending order, so the earliest customer promise is retained.

Candidates are sorted after deduplication by:

1. target SOC ascending;
2. ready step ascending;
3. candidate type, with `immediate_bau` before `minimum_cost`;
4. charging cost ascending; and
5. candidate ID ascending.

Candidate IDs are deterministic, contain no Python hash or memory address, and
distinguish target, candidate type, and ready step where needed.

## Deferred

The following remain intentionally outside Commit 4:

- literature-based battery degradation;
- health scoring and degradation-aware/intermediate-saving profile
  construction;
- final Pareto filtering and display-cap selection;
- customer-choice modelling;
- Monte Carlo simulation;
- distribution-network simulation; and
- plotting and file I/O.

## Run checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pytest --cov=evmenu --cov-report=term-missing
ruff check .
ruff format --check .
mypy evmenu tests
```
