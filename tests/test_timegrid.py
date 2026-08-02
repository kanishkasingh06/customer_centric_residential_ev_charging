from __future__ import annotations

from typing import cast

import pytest

from evmenu import TimeInterval, build_time_intervals
from evmenu.exceptions import PhysicalConstraintError, SchemaValidationError


def test_wall_clock_grid_preserves_both_partial_boundaries() -> None:
    intervals = build_time_intervals(
        arrival_minute=11 * 60 + 7,
        departure_minute=12 * 60 + 2,
        nominal_timestep_minutes=15,
    )
    assert [(item.start_minute, item.end_minute) for item in intervals] == [
        (667, 675),
        (675, 690),
        (690, 705),
        (705, 720),
        (720, 722),
    ]
    assert sum(item.duration_minutes for item in intervals) == 55


def test_overnight_grid_uses_absolute_wall_clock_boundaries() -> None:
    intervals = build_time_intervals(
        arrival_minute=23 * 60 + 53,
        departure_minute=24 * 60 + 7 * 60 + 8,
        nominal_timestep_minutes=15,
    )
    assert intervals[0].duration_minutes == 7
    assert intervals[-1].duration_minutes == 8
    assert intervals[1].start_minute == 1440
    assert intervals[-1].end_minute == 1868
    assert sum(item.duration_minutes for item in intervals) == 435


def test_time_interval_rejects_invalid_types() -> None:
    with pytest.raises(SchemaValidationError):
        TimeInterval(cast(int, True), 2)
    with pytest.raises(PhysicalConstraintError):
        TimeInterval(2, 1)


def test_additional_boundaries_split_without_zero_length_intervals() -> None:
    intervals = build_time_intervals(
        arrival_minute=10,
        departure_minute=40,
        nominal_timestep_minutes=15,
        additional_boundaries=(17, 30),
    )
    assert [(item.start_minute, item.end_minute) for item in intervals] == [
        (10, 15),
        (15, 17),
        (17, 30),
        (30, 40),
    ]
    assert all(item.duration_minutes > 0 for item in intervals)
