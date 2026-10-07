# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A residential EV charging **menu generator**: given one vehicle, a plug-in
window, a current state of charge and tomorrow's trip, it produces a small,
deterministic set of customer-facing offers that trade cost, battery health,
readiness time and target SOC against each other. `research/` then puts a
customer-choice model on top of those menus and simulates the resulting
distribution-feeder load.

The project is research code for the DER Integration Lab. Numerical results are
meant to be defensible in a paper, so **correctness and auditability outrank
convenience everywhere in `evmenu/`**.

## Layout and the two standards

```
evmenu/     the validated physical core  -- strict standard
tests/      pytest suite for evmenu      -- strict standard
research/   Monte Carlo analysis layer   -- looser standard
```

`evmenu/` and `tests/` hold a **strict** bar:

- `pytest` (528 tests) must pass.
- `ruff check .` must pass; `mypy` runs `strict = true` over `evmenu`.
- Every public type is a frozen, slotted dataclass that validates in
  `__post_init__`. Invalid states must be unconstructible, not merely unused.
- Every constructor re-validates its own output through
  `validation.validate_charging_profile`, which is deliberately independent of
  the code that built the profile. Do not remove or bypass that second check to
  make something faster.
- Tolerances are explicit fields on settings objects, never inline literals.
- Physical impossibility raises `PhysicalConstraintError`; malformed input
  raises `SchemaValidationError`. Keep that distinction.

`research/` is exploratory analysis. It must lint clean and its own
`verify.py` checks must pass, but it is not under `mypy --strict` (which runs
`packages = ["evmenu"]`) and it is not part of the distribution
(`[tool.setuptools.packages.find] include = ["evmenu*"]` in `pyproject.toml`
keeps auto-discovery off it). Its numpy dependency lives in the `research`
extra, not in the package's own requirements.

The exception is `tests/test_research_fleet.py`, which runs in the normal gate
and skips if numpy is absent. The quantiser and the raster decide where a
vehicle's energy lands on the fleet clock, and a mistake there produces a
plausible-looking profile rather than an error, so those two are held to the
strict standard even though the rest of `research/` is not.

## Running things

```bash
pip install -e ".[dev,research]"

python -m pytest                    # 528 tests, the gate for evmenu changes
ruff check . && mypy                # lint + types
```

Use `python -m pytest`, not a bare `pytest`: a `pytest` installed as a
standalone tool runs in its own isolated environment and will not see the
editable `evmenu` install, producing 13 collection errors that look like a code
problem but are not.

```bash
python research/make_tariff.py      # regenerate hourly_tariff.csv
python research/scenarios.py        # build menu bank, then the 108-cell sweep
python research/verify.py           # the five correctness checks
python research/counterfactual.py   # deterministic reference policies
python research/plots.py            # dashboard_built.html from the sweep outputs
```

`research/scenarios.py` is **resumable**. The menu bank saves atomically every
100 menus, so an interrupted build loses at most that many and re-running
continues from disk. A cold build is roughly **9 hours** on two cores
(measured: 26.4 s per menu, ~1,200 menus). An earlier version of this file said
10 minutes; that was never timed and is wrong by about fifty-fold. Plan to
leave it running.

**Nothing the pipeline generates is committed.** `menu_bank.pkl`,
`profiles.npz`, `results.json`, `counterfactual.json`, `viz_data.json` and
`dashboard_built.html` are all gitignored. They used to be committed, on the
reasoning that they were "the small record of a specific run" -- and that is
exactly how the repo came to hold a results table produced by a version of
`fleet.py` that no longer existed, with a day-shift bug in it that nothing in
the tree disclosed. A result file in git is a claim the code has to keep
earning, and this code takes nine hours to re-earn it. Regenerate instead.

## Things that are easy to get wrong here

**Saving is not cost.** `MenuOffer.advertised_saving` is measured against the
BAU *for that same target SOC*. A 100% target has a larger baseline and so a
larger achievable saving. `argmax(advertised_saving)` therefore selects the
100% target in the large majority of menus and is **never** the cheapest offer.
Anything that ranks or recommends offers by the saving column is steering the
customer toward maximum consumption. Use `charging_cost` when you mean cheap.
`research/counterfactual.py` quantifies this.

**Health barely varies within a request.** At fixed target SOC and ready step,
parked-day calendar fade and cycle fade are exactly constant, so ~78% of total
degradation cannot move; the median within-request fade spread is ~1e-21.
Battery health is essentially a function of *target SOC* (27% of total
variation) and weakly of ready time (2%). Do not add per-target health
normalization — it destroys the only real signal. Do not expect intra-window
scheduling to buy health.

**Intermediate frontier points are physically identical to the endpoints.**
`FrontierSettings.maximum_levels > 2` costs solve time and produces offers the
Pareto filter correctly deletes.

**`plating_guard_weight` is mis-scaled at its default 1e-6.** It is ~40x larger
than the window-fade differences it competes with and inverts their ordering,
so the point labelled `least_degradation` can have the highest window fade.
`research/` runs at 1e-9. The package default is unchanged; changing it would
alter published menu outputs, so treat it as a deliberate open item.

**Time is absolute minutes, not clock time.** `TimeInterval` uses an absolute
minute clock so overnight and weekly sessions do not wrap. Anything comparing
or differencing times must use `interval_start_minutes` / `interval_end_minutes`,
never the `HH:MM` strings.

**Intervals have variable duration.** The nominal timestep is not each
interval's length — exact arrival/departure and tariff boundaries create
7-minute and 8-minute slices. Never multiply by a single timestep; use
`PlanningSignal.interval_durations[k]`, and when regridding, conserve energy
rather than resampling power (`research/fleet.py:rasterise` is the reference).

**Illustrative values are labelled as such and must stay labelled.** The EV
catalogue, the `research_tou` tariff, `research/hourly_tariff.csv` and every
choice coefficient in `research/choice.py` are research assumptions, not
manufacturer specifications, regulated tariffs, or estimates from
revealed-preference data. Do not quote a willingness-to-pay number derived from
them as empirical, and do not quietly drop the disclaimers.

## Conventions

- Python 3.11+, line length 100.
- `from __future__ import annotations` at the top of every module.
- Keyword-only arguments for anything with more than two parameters.
- Prefer adding a validated field on a settings dataclass over threading a new
  positional argument through the call chain.
- Do not encode invariants in identifier strings. `degradation.py` currently
  validates a candidate's ready step by checking the suffix of
  `candidate_id` — that is existing debt, not a pattern to copy.

## Reproducibility note

The sweep is fully seeded, so the menu bank rebuilds to the same set of keys
on the same code: **1,202 keys** across the three arrival patterns and three
seeds at the maximum fleet size. The split between feasible and infeasible is
NOT stable across evmenu versions -- it was 1,183/19 on 4d109c5 and is 1,186/16
on 9c60b98, because the saving-band and ready_step fixes serve requests that
used to be refused. Only the total is a fixed property of the sampler.

The midnight fix does not change that total -- measured both ways, 1,202 before
and after -- but it does change **9 of the keys**: the ones whose arrival had
been collapsed onto bucket 0. A bank built before that commit therefore carries
9 entries nothing will ever look up and is missing 9 it needs. It still
resumes; it is simply 9 menus short and a little larger than necessary.

Individual peaks can move by ~2% across SciPy versions, because SLSQP solutions
differ in their last digits and that can flip a marginal choice. Do not expect
bit-identical numbers across environments; do expect the conclusions to hold.

Seed-to-seed variation of peak load is **10.3%** (concentrated / balanced /
150 EVs, peaks 167.5 / 138.2 / 133.2 kW), measured on 44d0b70. The 2.3% figure
quoted in earlier write-ups came from a sweep run before the midnight,
choice-set and wear fixes; it is superseded, not merely unverified. Quantisation
sensitivity is 2.3% on peak and 9.1% on energy against an unquantised re-run --
the energy figure is large enough to be worth stating alongside any result.
