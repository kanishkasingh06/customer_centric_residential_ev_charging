"""Exact minute-level session boundaries and variable-duration intervals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from .exceptions import PhysicalConstraintError, SchemaValidationError


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """One half-open interval in an absolute minute clock."""

    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        for name, value in (("start_minute", self.start_minute), ("end_minute", self.end_minute)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaValidationError(f"{name} must be an integer.")
        if self.end_minute <= self.start_minute:
            raise PhysicalConstraintError("TimeInterval end must be strictly after start.")

    @property
    def duration_minutes(self) -> int:
        return self.end_minute - self.start_minute

    @property
    def duration_hours(self) -> float:
        return self.duration_minutes / 60.0


def build_time_intervals(
    *,
    arrival_minute: int,
    departure_minute: int,
    nominal_timestep_minutes: int,
    additional_boundaries: Iterable[int] = (),
) -> tuple[TimeInterval, ...]:
    """Build exact, continuous intervals with wall-clock grid alignment.

    ``arrival_minute`` and ``departure_minute`` are absolute minutes, with
    departure strictly later than arrival. Nominal boundaries are multiples of
    the nominal timestep on the wall clock rather than offsets from arrival.
    """
    for name, value in (("arrival_minute", arrival_minute), ("departure_minute", departure_minute)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaValidationError(f"{name} must be an integer.")
    if departure_minute <= arrival_minute:
        raise PhysicalConstraintError("departure_minute must be strictly after arrival_minute.")
    if isinstance(nominal_timestep_minutes, bool) or not isinstance(nominal_timestep_minutes, int):
        raise SchemaValidationError("nominal_timestep_minutes must be an integer.")
    if nominal_timestep_minutes <= 0:
        raise PhysicalConstraintError("nominal_timestep_minutes must be positive.")

    boundaries = {arrival_minute, departure_minute}
    first_grid = ((arrival_minute // nominal_timestep_minutes) + 1) * nominal_timestep_minutes
    boundary = first_grid
    while boundary < departure_minute:
        if boundary > arrival_minute:
            boundaries.add(boundary)
        boundary += nominal_timestep_minutes

    try:
        for value in additional_boundaries:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaValidationError("additional boundaries must be integers.")
            if arrival_minute < value < departure_minute:
                boundaries.add(value)
    except TypeError as exc:
        raise SchemaValidationError("additional_boundaries must be iterable.") from exc

    ordered = sorted(boundaries)
    intervals = tuple(TimeInterval(start, end) for start, end in pairwise(ordered))
    if not intervals or intervals[0].start_minute != arrival_minute:
        raise SchemaValidationError("time intervals must start at exact arrival.")
    if intervals[-1].end_minute != departure_minute:
        raise SchemaValidationError("time intervals must end at exact departure.")
    return intervals


def recurring_daily_boundaries(
    *,
    start_minute: int,
    end_minute: int,
    boundaries_of_day: Iterable[int],
) -> tuple[int, ...]:
    """Return recurring wall-clock boundaries inside an absolute session."""
    boundaries = tuple(boundaries_of_day)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
        raise SchemaValidationError("recurring boundaries must be integers.")
    result: set[int] = set()
    first_day = start_minute // 1440
    last_day = (end_minute - 1) // 1440
    for day in range(first_day, last_day + 1):
        for minute in boundaries:
            absolute = day * 1440 + minute
            if start_minute < absolute < end_minute:
                result.add(absolute)
    return tuple(sorted(result))
