"""Fleet sampling, menu caching, and energy-conserving load aggregation.

The expensive object in this phase is a generated menu (~0.5 s). Menus depend
ONLY on the physical request -- EV model, plug-in and departure clock times,
current SOC, trip distance, weekday -- and not at all on customer preferences
or fleet size. So the preference-mix and fleet-size sweeps reuse one menu bank
entirely, and only the arrival-distribution scenarios add new menus.

Requests are quantised before lookup (see ``QUANTISATION``) so that a fleet of
150 customers collapses onto fewer distinct menus. ``verify.py`` re-runs one
cell without quantisation to confirm the load shape is not an artefact of that
grid.
"""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evmenu import (
    FrontierSettings,
    MenuAssemblySettings,
    generate_ev_menu,
    load_weekly_price_profile_csv,
)
from evmenu.exceptions import EVMenuError

RESEARCH_DIR = Path(__file__).parent
TARIFF_CSV = RESEARCH_DIR / "hourly_tariff.csv"
BANK_PATH = RESEARCH_DIR / "menu_bank.pkl"

# Frontier config established in the phase-11 diagnosis: the plating guard was
# inverting the ordering of an already-negligible within-request signal, and
# intermediate frontier levels are physically identical to the endpoints.
#
# An earlier version of this comment claimed these overrides produce
# "byte-identical displayed menus". That is FALSE and was never measured: a
# direct comparison found 0 of 12 sampled menus identical to the package
# defaults. The overrides change which offers are displayed. Everything this
# tree reports is therefore a result FOR THIS CONFIGURATION, and how far it
# diverges from the defaults has not been quantified. Do not describe the two
# as equivalent.
FRONTIER = FrontierSettings(plating_guard_weight=1e-9, maximum_levels=2)

# Ready-step pruning tolerance in Rs. The default (1e-8) retains a separate
# request for every ready step whose saving improves at all, building ~87
# frontiers to fill a 12-row menu. At Rs 40 the menu keeps the same 12 rows
# with 8-9 distinct ready times and all 4 targets, but builds ~25 frontiers --
# 3x faster. It is also the more defensible product rule: do not offer two
# options that differ by less than Rs 40.
ASSEMBLY = MenuAssemblySettings(pruning_saving_tolerance=40.0)

ARRIVAL_DAY = "Wed"
NOMINAL_TIMESTEP_MIN = 15
GRID_MIN = 5  # load-profile raster resolution

# Coarse on purpose. Menu generation costs seconds, and a 5-D continuous draw
# produces near-unique requests, so fine quantisation gives a ~0% cache hit
# rate. These bucket widths are below the resolution at which the load profile
# responds (SOC to 5 points, trips to 10 km, clock to half an hour) and are
# checked against an unquantised re-run in verify.py.
QUANTISATION = {"clock_min": 30, "soc": 0.05, "trip_km": 10.0}

# The sampling windows, as minutes past midnight. Arrivals sit on ARRIVAL_DAY,
# departures on the following morning. These bound BOTH the draw and its
# quantised bucket: see _quantise_minutes for why the upper arrival bound
# matters more than it looks.
ARRIVAL_WINDOW_MIN = (15 * 60, 23 * 60 + 55)       # 15:00 .. 23:55
DEPARTURE_WINDOW_MIN = (5 * 60, 10 * 60)           # 05:00 .. 10:00

# Fleet composition: illustrative research assumption.
MODEL_MIX = (("generic_40kwh_lfp", 0.60), ("generic_60kwh_nmc", 0.40))


@dataclass(frozen=True)
class ArrivalPattern:
    """Plug-in / departure timing distribution for one scenario."""

    name: str
    arrival_components: tuple[tuple[float, float, float], ...]  # (weight, mean_min, sd_min)
    departure_mean_min: float
    departure_sd_min: float

    def __post_init__(self) -> None:
        total = sum(weight for weight, _, _ in self.arrival_components)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"arrival component weights must sum to 1; got {total}")
        if self.departure_sd_min < 0:
            raise ValueError("departure_sd_min must be non-negative")


ARRIVAL_PATTERNS: dict[str, ArrivalPattern] = {
    # Tight commuter peak: everyone home within about an hour of each other.
    "concentrated": ArrivalPattern("concentrated", ((1.0, 18 * 60 + 30, 35.0),), 7 * 60 + 45, 25.0),
    # Wide spread of plug-in times: shift work, errands, mixed households.
    "dispersed": ArrivalPattern("dispersed", ((1.0, 19 * 60, 130.0),), 7 * 60 + 30, 75.0),
    # Two waves: early commuters and a later evening-activity group.
    "bimodal": ArrivalPattern(
        "bimodal", ((0.55, 17 * 60 + 30, 45.0), (0.45, 21 * 60 + 30, 60.0)), 7 * 60 + 30, 60.0
    ),
}


def _quantise_minutes(minutes: int, step: int, window: tuple[int, int]) -> int:
    """Round a time of day onto the bucket grid without leaving ``window``.

    Plain ``round(m / step) * step`` can land on 1440, which is midnight of the
    FOLLOWING day. The obvious repair -- ``% 1440`` -- is a trap: it relabels
    that as midnight of the arrival day, 24 hours early. The menu is then built
    for a session that ends before the fleet clock starts, so ``rasterise``
    clips every interval away and the vehicle contributes exactly zero with no
    error raised. Arrivals from 23:45 on were lost this way, which is 1.4% of
    the dispersed pattern and 0.5% of bimodal.

    Clamping into the window instead keeps the bucket on the correct day. The
    cost is a wider top bucket: 23:45-23:55 is represented by 23:30, a shift of
    at most 25 minutes, which sits inside the half-hour quantisation this cache
    already accepts and is covered by the unquantised re-run in verify.py.
    """
    low, high = window
    lowest_bucket = -(-low // step) * step   # first grid point at or above low
    highest_bucket = (high // step) * step   # last grid point at or below high
    bucket = round(minutes / step) * step
    return int(min(max(bucket, lowest_bucket), highest_bucket))


@dataclass(frozen=True)
class CustomerDraw:
    """One sampled customer's physical charging request."""

    model_id: str
    arrival_min: int      # minutes past midnight on ARRIVAL_DAY
    departure_min: int    # minutes past midnight next morning
    current_soc: float
    trip_km: float

    def __post_init__(self) -> None:
        if not ARRIVAL_WINDOW_MIN[0] <= self.arrival_min <= ARRIVAL_WINDOW_MIN[1]:
            raise ValueError(
                f"arrival_min {self.arrival_min} outside {ARRIVAL_WINDOW_MIN}; "
                "a draw outside the window cannot be placed on the fleet clock"
            )
        if not DEPARTURE_WINDOW_MIN[0] <= self.departure_min <= DEPARTURE_WINDOW_MIN[1]:
            raise ValueError(
                f"departure_min {self.departure_min} outside {DEPARTURE_WINDOW_MIN}"
            )

    @property
    def key(self) -> tuple:
        """Quantised menu-bank lookup key."""
        q = QUANTISATION
        step = int(q["clock_min"])
        return (
            self.model_id,
            _quantise_minutes(self.arrival_min, step, ARRIVAL_WINDOW_MIN),
            _quantise_minutes(self.departure_min, step, DEPARTURE_WINDOW_MIN),
            round(round(self.current_soc / q["soc"]) * q["soc"], 4),
            round(round(self.trip_km / q["trip_km"]) * q["trip_km"], 2),
        )

    @property
    def exact_key(self) -> tuple:
        # No modulo here: __post_init__ already guarantees both times are inside
        # their windows, so there is nothing to wrap and nothing to lose.
        return (self.model_id, self.arrival_min, self.departure_min,
                round(self.current_soc, 4), round(self.trip_km, 2))


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def sample_fleet(n: int, pattern: ArrivalPattern, rng: np.random.Generator) -> list[CustomerDraw]:
    """Sample ``n`` customer requests from the population distributions."""
    if n <= 0:
        raise ValueError("fleet size must be positive")
    weights = np.array([w for w, _, _ in pattern.arrival_components])
    component = rng.choice(len(weights), size=n, p=weights)
    means = np.array([m for _, m, _ in pattern.arrival_components])[component]
    sds = np.array([s for _, _, s in pattern.arrival_components])[component]
    arrivals = rng.normal(means, sds)
    departures = rng.normal(pattern.departure_mean_min, pattern.departure_sd_min, size=n)
    socs = rng.normal(0.38, 0.14, size=n)
    trips = np.exp(np.log(38.0) + 0.50 * rng.standard_normal(n))
    model_ids = rng.choice([m for m, _ in MODEL_MIX], size=n, p=[p for _, p in MODEL_MIX])

    draws: list[CustomerDraw] = []
    for i in range(n):
        draws.append(
            CustomerDraw(
                model_id=str(model_ids[i]),
                arrival_min=round(_clip(arrivals[i], *ARRIVAL_WINDOW_MIN)),
                departure_min=round(_clip(departures[i], *DEPARTURE_WINDOW_MIN)),
                current_soc=_clip(socs[i], 0.12, 0.80),
                trip_km=_clip(trips[i], 5.0, 120.0),
            )
        )
    return draws


def _fmt(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


@dataclass(frozen=True, slots=True)
class OfferRecord:
    """The part of one menu offer the load simulation actually reads."""

    offer_id: str
    role: str
    target_soc: float          # fraction
    ready_step: int
    ready_minute: int          # absolute minute of the ready boundary
    charging_cost: float       # Rs
    advertised_saving: float   # Rs, against the SAME-TARGET BAU (see counterfactual.py)
    health_score: float        # 0-100
    energy_kwh: float
    power_kw: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MenuRecord:
    """A generated menu reduced to its simulation-relevant fields.

    A full ``GeneratedCustomerMenu`` is ~109 KB, of which 88% is
    ``AssembledMenu.source_frontiers`` -- the provenance trail kept for
    auditability, which nothing here reads. Storing the full object makes the
    bank ~128 MB; this record makes it ~8 MB. The full object for any key is
    always one ``build_menu(key)`` call away when an audit is wanted.
    """

    interval_start_minutes: tuple[int, ...]
    interval_end_minutes: tuple[int, ...]
    offers: tuple[OfferRecord, ...]

    @classmethod
    def from_menu(cls, menu) -> MenuRecord:
        starts = tuple(menu.interval_start_minutes)
        ends = tuple(menu.interval_end_minutes)
        source = {s.offer_id: s.endpoint_role for s in menu.assembled_menu.source_metadata}
        offers = tuple(
            OfferRecord(
                offer_id=offer.offer_id,
                role=source[offer.offer_id],
                target_soc=float(offer.target_soc),
                ready_step=int(offer.ready_step),
                ready_minute=int(
                    starts[offer.ready_step] if offer.ready_step < len(starts) else ends[-1]
                ),
                charging_cost=float(offer.charging_cost),
                advertised_saving=float(offer.advertised_saving),
                health_score=float(offer.charging_health_score),
                energy_kwh=float(sum(offer.profile.grid_energy_kwh)),
                power_kw=tuple(float(p) for p in offer.profile.power_kw),
            )
            for offer in menu.assembled_menu.offers
        )
        return cls(interval_start_minutes=starts, interval_end_minutes=ends, offers=offers)


def build_menu(key: tuple):
    """Generate one menu for a quantised request key. Returns None if infeasible."""
    model_id, arrival_min, departure_min, soc, trip_km = key
    profile = load_weekly_price_profile_csv(TARIFF_CSV, profile_format="hour_of_week")
    try:
        return generate_ev_menu(
            ev_model=model_id,
            arrival_time=_fmt(arrival_min),
            departure_time=_fmt(departure_min),
            current_soc=soc,
            next_trip_distance_km=trip_km,
            tariff_name="custom",
            custom_price_profile=profile,
            arrival_day=ARRIVAL_DAY,
            timestep_minutes=NOMINAL_TIMESTEP_MIN,
            frontier_settings=FRONTIER,
            assembly_settings=ASSEMBLY,
        )
    except EVMenuError:
        return None


def build_record(key: tuple) -> MenuRecord | None:
    """Generate a menu and reduce it to a MenuRecord (what the bank stores)."""
    menu = build_menu(key)
    return None if menu is None else MenuRecord.from_menu(menu)


class MenuBank:
    """Disk-backed, resumable cache of menu records keyed by quantised request."""

    def __init__(self, path: Path = BANK_PATH) -> None:
        self.path = path
        self.menus: dict[tuple, MenuRecord] = {}
        self.infeasible: set[tuple] = set()
        if path.exists():
            try:
                with path.open("rb") as handle:
                    state = pickle.load(handle)
                self.menus = state["menus"]
                self.infeasible = state["infeasible"]
            except (EOFError, pickle.UnpicklingError) as exc:
                raise SystemExit(
                    f"menu bank at {path} is corrupt ({exc}). Delete it and re-run."
                ) from exc

    def save(self) -> None:
        """Write atomically: an interrupted save must never corrupt the bank."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=".menu_bank-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump(
                    {"menus": self.menus, "infeasible": self.infeasible},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def known(self, key: tuple) -> bool:
        return key in self.menus or key in self.infeasible

    def ensure(
        self, keys: list[tuple], workers: int = 2, verbose: bool = True, limit: int | None = None
    ) -> int:
        """Generate menus not already cached. Returns how many are still missing.

        Saves in chunks and resumes from disk, so a long build can be run in
        several passes and survives an interrupted one.
        """
        missing = sorted({k for k in keys if not self.known(k)})
        if not missing:
            return 0
        batch = missing if limit is None else missing[:limit]
        if verbose:
            print(f"  building {len(batch)} of {len(missing)} missing menus...", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for key, record in zip(batch, pool.map(build_record, batch, chunksize=4)):
                if record is None:
                    self.infeasible.add(key)
                else:
                    self.menus[key] = record
                done += 1
                if done % 100 == 0:
                    self.save()
                    if verbose:
                        print(f"    {done}/{len(batch)}", flush=True)
        self.save()
        return len(missing) - done

    def get(self, key: tuple) -> MenuRecord | None:
        return self.menus.get(key)


# --------------------------------------------------------------------------
# Load aggregation
# --------------------------------------------------------------------------

# Common absolute-minute clock for the fleet profile: ARRIVAL_DAY 12:00 to the
# following day 12:00. ARRIVAL_DAY is index 2 (Wed) in the weekly minute clock.
DAY_INDEX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
CLOCK_START = DAY_INDEX[ARRIVAL_DAY] * 1440 + 12 * 60
CLOCK_END = CLOCK_START + 1440
N_CELLS = (CLOCK_END - CLOCK_START) // GRID_MIN


def rasterise(record, offer_index: int) -> np.ndarray:
    """Map one chosen offer's schedule onto the common clock as kW per cell.

    Energy-conserving: each source interval's energy is distributed across the
    grid cells it overlaps in proportion to the overlap, then converted back to
    power. This handles the variable-duration intervals (7-minute boundary
    slices and so on) that the exact-time grid produces.
    """
    starts = record.interval_start_minutes
    ends = record.interval_end_minutes
    power = record.offers[offer_index].power_kw
    cell_kwh = np.zeros(N_CELLS, dtype=float)
    offered_kwh = 0.0
    for start, end, kilowatts in zip(starts, ends, power):
        if kilowatts <= 0.0:
            continue
        offered_kwh += kilowatts * (end - start) / 60.0
        first = max(start, CLOCK_START)
        last = min(end, CLOCK_END)
        if last <= first:
            continue
        lo = (first - CLOCK_START) // GRID_MIN
        hi = (last - 1 - CLOCK_START) // GRID_MIN
        for cell in range(lo, hi + 1):
            cell_lo = CLOCK_START + cell * GRID_MIN
            overlap = min(last, cell_lo + GRID_MIN) - max(first, cell_lo)
            if overlap > 0:
                cell_kwh[cell] += kilowatts * overlap / 60.0

    # Every sampled session lies inside [CLOCK_START, CLOCK_END] by construction
    # (arrivals 15:00-23:55 on ARRIVAL_DAY, departures 05:00-10:00 the morning
    # after), so any energy the clipping above removed means the session was
    # placed on the wrong day. That used to happen silently; it is the single
    # most expensive class of bug in this file, because the output still looks
    # like a plausible load profile. Refuse instead.
    placed_kwh = float(cell_kwh.sum())
    if not math.isclose(placed_kwh, offered_kwh, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"rasterise dropped energy: offer carries {offered_kwh:.6f} kWh but only "
            f"{placed_kwh:.6f} kWh landed on the clock "
            f"[{CLOCK_START}, {CLOCK_END}); session spans "
            f"[{starts[0]}, {ends[-1]}]. The session is outside the fleet window."
        )
    return cell_kwh / (GRID_MIN / 60.0)


def clock_labels() -> list[str]:
    return [_fmt((CLOCK_START + i * GRID_MIN) % 1440) for i in range(N_CELLS)]
