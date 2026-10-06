"""Tests for the research fleet sampler's clock handling.

``research/`` is not under the strict standard that ``evmenu/`` is held to, but
the quantiser and the raster decide where a vehicle's energy lands on the fleet
clock, and a mistake there is invisible in the output. These are the checks
that would have caught the midnight wraparound.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="research/ needs the 'research' extra")

RESEARCH_DIR = Path(__file__).resolve().parent.parent / "research"


def _load_fleet():
    """Import research/fleet.py without putting research/ on sys.path globally."""
    if str(RESEARCH_DIR) not in sys.path:
        sys.path.insert(0, str(RESEARCH_DIR))
    spec = importlib.util.spec_from_file_location("research_fleet", RESEARCH_DIR / "fleet.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so the module has to
    # be registered before it is executed or CustomerDraw fails to build.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fleet = _load_fleet()


def _draw(arrival_min: int, departure_min: int = 450):
    return fleet.CustomerDraw(
        model_id="generic_40kwh_lfp",
        arrival_min=arrival_min,
        departure_min=departure_min,
        current_soc=0.35,
        trip_km=40.0,
    )


def test_every_arrival_minute_buckets_inside_the_arrival_window() -> None:
    """No draw may quantise to a time outside the window it was drawn from.

    This is the regression test for the midnight wraparound. Before the fix,
    `round(m / 30) * 30 % 1440` sent every arrival from 23:45 on to bucket 0 --
    midnight of the arrival day rather than the following one -- and the whole
    session was then placed 24 hours early.
    """
    low, high = fleet.ARRIVAL_WINDOW_MIN
    offenders = [
        (minute, _draw(minute).key[1])
        for minute in range(low, high + 1)
        if not low <= _draw(minute).key[1] <= high
    ]
    assert offenders == [], f"{len(offenders)} arrivals quantised outside the window"


def test_late_arrivals_bucket_to_the_last_same_day_slot() -> None:
    """23:45-23:55 must land on 23:30, not wrap to midnight."""
    assert _draw(23 * 60 + 44).key[1] == 23 * 60 + 30
    assert _draw(23 * 60 + 45).key[1] == 23 * 60 + 30
    assert _draw(23 * 60 + 55).key[1] == 23 * 60 + 30
    # And the shift stays inside the quantisation step the cache already accepts.
    assert (23 * 60 + 55) - (23 * 60 + 30) <= fleet.QUANTISATION["clock_min"]


def test_quantised_arrival_never_reaches_a_full_day() -> None:
    low, high = fleet.ARRIVAL_WINDOW_MIN
    step = int(fleet.QUANTISATION["clock_min"])
    assert all(
        fleet._quantise_minutes(minute, step, fleet.ARRIVAL_WINDOW_MIN) < 1440
        for minute in range(low, high + 1)
    )


def test_draw_outside_the_window_is_unconstructible() -> None:
    with pytest.raises(ValueError, match="arrival_min"):
        _draw(fleet.ARRIVAL_WINDOW_MIN[1] + 1)
    with pytest.raises(ValueError, match="arrival_min"):
        _draw(0)
    with pytest.raises(ValueError, match="departure_min"):
        _draw(900, departure_min=1200)


def _record(start: int, end: int, kilowatts: float = 7.0):
    """Minimal stand-in for a MenuRecord carrying one powered interval."""
    return SimpleNamespace(
        interval_start_minutes=(start,),
        interval_end_minutes=(end,),
        offers=(SimpleNamespace(power_kw=(kilowatts,)),),
    )


def test_rasterise_conserves_energy_for_a_session_on_the_clock() -> None:
    start = fleet.CLOCK_START + 60
    record = _record(start, start + 120, 7.0)
    profile = fleet.rasterise(record, 0)
    placed = profile.sum() * (fleet.GRID_MIN / 60.0)
    assert placed == pytest.approx(7.0 * 2.0, rel=1e-12)


def test_rasterise_refuses_a_session_placed_off_the_clock() -> None:
    """The guard that makes the wraparound loud instead of silent.

    A session a day early lands entirely before CLOCK_START. The old code
    clipped it away and returned a profile of zeros, which still looks like a
    valid answer.
    """
    start = fleet.CLOCK_START - 720
    record = _record(start, start + 120, 7.0)
    with pytest.raises(ValueError, match="dropped energy"):
        fleet.rasterise(record, 0)


def test_rasterise_refuses_a_session_that_only_partly_overlaps() -> None:
    record = _record(fleet.CLOCK_END - 30, fleet.CLOCK_END + 90, 7.0)
    with pytest.raises(ValueError, match="dropped energy"):
        fleet.rasterise(record, 0)
