"""Immutable machine-readable electricity-price profiles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import cast

from .exceptions import PhysicalConstraintError, SchemaValidationError

MINUTES_PER_WEEK = 7 * 1440
MAX_PROFILE_FILE_BYTES = 10 * 1024 * 1024
_DAY_INDEX = {
    name: index for index, name in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
}


def _finite_price(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError("price must be a finite real number.")
    return float(value)


def _currency_label(value: object) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError("currency_label must be a string.")
    label = value.strip()
    if not label:
        raise SchemaValidationError("currency_label must be non-empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in label):
        raise SchemaValidationError(
            "currency_label must be at most 64 characters without control characters."
        )
    if len(label) > 64:
        raise SchemaValidationError(
            "currency_label must be at most 64 characters without control characters."
        )
    return label


def _clock_minute(value: str, *, allow_24: bool = False) -> int:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise SchemaValidationError("profile times must use strict HH:MM format.")
    if not all("0" <= character <= "9" for character in value[:2] + value[3:]):
        raise SchemaValidationError("profile times must use strict HH:MM format.")
    hour = int(value[:2])
    minute = int(value[3:])
    if minute > 59 or hour > (24 if allow_24 else 23) or (hour == 24 and minute != 0):
        raise SchemaValidationError("profile times must use valid 24-hour values.")
    if hour == 24:
        return 1440
    return hour * 60 + minute


def _is_timezone_aware(value: datetime) -> bool:
    """Return the standard-library awareness classification for a datetime."""
    try:
        return value.tzinfo is not None and value.utcoffset() is not None
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(
            "timestamped profile datetime has an invalid timezone."
        ) from exc


@dataclass(frozen=True, slots=True)
class PricePeriod:
    """One half-open recurring weekly price period."""

    start_minute_of_week: int
    end_minute_of_week: int
    price_per_kwh: float

    def __post_init__(self) -> None:
        for name, value in (
            ("start_minute_of_week", self.start_minute_of_week),
            ("end_minute_of_week", self.end_minute_of_week),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaValidationError(f"{name} must be an integer.")
        if not 0 <= self.start_minute_of_week < MINUTES_PER_WEEK:
            raise PhysicalConstraintError("price period start must lie within the week.")
        if not 0 < self.end_minute_of_week <= MINUTES_PER_WEEK:
            raise PhysicalConstraintError("price period end must lie within the week.")
        if self.end_minute_of_week <= self.start_minute_of_week:
            raise PhysicalConstraintError("price period end must be after start.")
        object.__setattr__(self, "price_per_kwh", _finite_price(self.price_per_kwh))


@dataclass(frozen=True, slots=True)
class WeeklyPriceProfile:
    """A complete, non-overlapping weekly recurring price profile."""

    profile_id: str
    periods: tuple[PricePeriod, ...]
    currency_label: str = "currency"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise SchemaValidationError("profile_id must be a non-empty string.")
        object.__setattr__(self, "currency_label", _currency_label(self.currency_label))
        if self.periods is None or isinstance(self.periods, (str, bytes)):
            raise SchemaValidationError("periods must be an iterable of PricePeriod objects.")
        try:
            periods = tuple(self.periods)
        except TypeError as exc:
            raise SchemaValidationError(
                "periods must be an iterable of PricePeriod objects."
            ) from exc
        if not periods or any(not isinstance(period, PricePeriod) for period in periods):
            raise SchemaValidationError("periods must be a nonempty tuple of PricePeriod objects.")
        if periods != tuple(sorted(periods, key=lambda period: period.start_minute_of_week)):
            raise SchemaValidationError("weekly price periods must be ordered by start time.")
        cursor = 0
        for period in periods:
            if period.start_minute_of_week != cursor:
                raise PhysicalConstraintError("weekly price periods must cover the week exactly.")
            cursor = period.end_minute_of_week
        if cursor != MINUTES_PER_WEEK:
            raise PhysicalConstraintError("weekly price periods must end at the end of Sunday.")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "currency_label", _currency_label(self.currency_label))
        object.__setattr__(self, "periods", periods)

    @property
    def boundaries(self) -> tuple[int, ...]:
        return tuple(period.start_minute_of_week for period in self.periods[1:])

    def price_at(self, minute_of_week: int) -> float:
        minute = minute_of_week % MINUTES_PER_WEEK
        for period in self.periods:
            if period.start_minute_of_week <= minute < period.end_minute_of_week:
                return period.price_per_kwh
        raise PhysicalConstraintError("weekly price profile does not cover the requested minute.")

    def absolute_boundaries(self, *, start_minute: int, end_minute: int) -> tuple[int, ...]:
        result: set[int] = set()
        first_week = start_minute // MINUTES_PER_WEEK - 1
        last_week = (end_minute - 1) // MINUTES_PER_WEEK + 1
        for week in range(first_week, last_week + 1):
            for period in self.periods[1:]:
                absolute = week * MINUTES_PER_WEEK + period.start_minute_of_week
                if start_minute < absolute < end_minute:
                    result.add(absolute)
        return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class TimestampedPricePeriod:
    start: datetime
    end: datetime
    price_per_kwh: float

    def __post_init__(self) -> None:
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise SchemaValidationError("timestamped period bounds must be datetimes.")
        if any(value.second != 0 or value.microsecond != 0 for value in (self.start, self.end)):
            raise SchemaValidationError("timestamped period bounds must fall on whole minutes.")
        try:
            invalid_order = self.end <= self.start
        except TypeError as exc:
            raise SchemaValidationError("timestamped period datetimes must be comparable.") from exc
        if invalid_order:
            raise PhysicalConstraintError("timestamped price period end must be after start.")
        object.__setattr__(self, "price_per_kwh", _finite_price(self.price_per_kwh))


@dataclass(frozen=True, slots=True)
class TimestampedPriceProfile:
    profile_id: str
    periods: tuple[TimestampedPricePeriod, ...]
    currency_label: str = "currency"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise SchemaValidationError("profile_id must be a non-empty string.")
        if self.periods is None or isinstance(self.periods, (str, bytes)):
            raise SchemaValidationError(
                "periods must be an iterable of TimestampedPricePeriod objects."
            )
        try:
            periods = tuple(self.periods)
        except TypeError as exc:
            raise SchemaValidationError(
                "periods must be an iterable of TimestampedPricePeriod objects."
            ) from exc
        if not periods or any(not isinstance(period, TimestampedPricePeriod) for period in periods):
            raise SchemaValidationError("periods must be a nonempty tuple of timestamped periods.")
        aware = _is_timezone_aware(periods[0].start)
        if any(
            _is_timezone_aware(period.start) != aware or _is_timezone_aware(period.end) != aware
            for period in periods
        ):
            raise SchemaValidationError(
                "timestamped profile datetimes must all be naive or all timezone-aware."
            )
        try:
            ordered = tuple(sorted(periods, key=lambda period: period.start))
        except TypeError as exc:
            raise SchemaValidationError(
                "timestamped profile datetimes are not comparable."
            ) from exc
        if periods != ordered:
            raise SchemaValidationError("timestamped price periods must be ordered.")
        for left, right in pairwise(periods):
            if left.end != right.start:
                raise PhysicalConstraintError("timestamped price periods must be continuous.")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "currency_label", _currency_label(self.currency_label))
        object.__setattr__(self, "periods", periods)

    def _validate_query_datetime(self, value: object) -> datetime:
        if not isinstance(value, datetime):
            raise SchemaValidationError("timestamped profile queries must use datetime values.")
        profile_aware = _is_timezone_aware(self.periods[0].start)
        query_aware = _is_timezone_aware(value)
        if profile_aware and not query_aware:
            raise SchemaValidationError(
                "Timestamped profile uses timezone-aware datetimes but the session datetime is naive."
            )
        if not profile_aware and query_aware:
            raise SchemaValidationError(
                "Timestamped profile uses naive datetimes but the session datetime is timezone-aware."
            )
        return value

    def boundaries_for_session(self, start: datetime, end: datetime) -> tuple[int, ...]:
        start = self._validate_query_datetime(start)
        end = self._validate_query_datetime(end)
        if end <= start:
            raise PhysicalConstraintError("timestamped profile session end must be after start.")
        if start < self.periods[0].start or end > self.periods[-1].end:
            raise PhysicalConstraintError("timestamped price profile does not cover the session.")
        return tuple(
            int((period.start - start).total_seconds() // 60)
            for period in self.periods[1:]
            if start < period.start < end
        )

    def price_at(self, timestamp: datetime) -> float:
        timestamp = self._validate_query_datetime(timestamp)
        for period in self.periods:
            if period.start <= timestamp < period.end:
                return period.price_per_kwh
        raise PhysicalConstraintError(
            "timestamped price profile does not cover the requested time."
        )


def _csv_rows(
    path: str | Path,
    *,
    profile_format: str,
    expected_headers: list[str] | None = None,
    expected_header_options: tuple[list[str], ...] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    path_object = Path(path)
    line_number: int | str = "?"
    try:
        size = path_object.stat().st_size
    except OSError as exc:
        raise SchemaValidationError(
            f"cannot read price profile '{path}' (format: {profile_format}): {exc}"
        ) from exc
    if size > MAX_PROFILE_FILE_BYTES:
        raise SchemaValidationError(
            f"{profile_format} price profile '{path}' exceeds the 10 MiB file-size limit."
        )
    try:
        with path_object.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None:
                raise SchemaValidationError(
                    f"{profile_format} price profile '{path}' is missing a header row."
                )
            fieldnames = list(reader.fieldnames)
            header_options = expected_header_options
            if header_options is None and expected_headers is not None:
                header_options = (expected_headers,)
            if header_options is not None and fieldnames not in header_options:
                raise SchemaValidationError(
                    f"{profile_format} price profile '{path}' has headers {fieldnames!r}; "
                    f"expected {header_options!r}."
                )
            rows: list[dict[str, str]] = []
            for row in reader:
                line_number = reader.line_num
                if None in row:
                    raise SchemaValidationError(
                        f"{profile_format} price profile '{path}' row {line_number} has "
                        f"unexpected values {row[None]!r}; expected headers {fieldnames!r}."
                    )
                if any(value is None or value == "" for value in row.values()):
                    raise SchemaValidationError(
                        f"{profile_format} price profile '{path}' row {line_number} has a "
                        f"missing required field; expected headers {fieldnames!r}."
                    )
                rows.append(cast(dict[str, str], row))
            if not rows:
                raise SchemaValidationError(
                    f"{profile_format} price profile '{path}' contains no data rows."
                )
    except UnicodeDecodeError as exc:
        raise SchemaValidationError(
            f"cannot read price profile '{path}' (format: {profile_format}): "
            "file is not valid UTF-8."
        ) from exc
    except csv.Error as exc:
        raise SchemaValidationError(
            f"Could not parse {profile_format} price profile '{path}' near CSV row "
            f"{line_number}: {exc}."
        ) from exc
    except OSError as exc:
        raise SchemaValidationError(
            f"cannot read price profile '{path}' (format: {profile_format}): {exc}"
        ) from exc
    return fieldnames, rows


def _require_headers(
    fieldnames: list[str],
    *,
    expected: list[list[str]],
    path: str | Path,
    profile_format: str,
) -> None:
    if fieldnames not in expected:
        expected_text = " or ".join(repr(headers) for headers in expected)
        raise SchemaValidationError(
            f"{profile_format} price profile '{path}' has headers {fieldnames!r}; "
            f"expected {expected_text}."
        )


def load_weekly_price_profile_csv(
    path: str | Path,
    *,
    profile_id: str | None = None,
    profile_format: str = "weekly",
) -> WeeklyPriceProfile:
    """Load a complete weekly profile from hour-of-week or day-period CSV."""
    fieldnames, rows = _csv_rows(
        path,
        profile_format=profile_format,
        expected_headers=(["hour_of_week", "price"] if profile_format == "hour_of_week" else None),
        expected_header_options=(
            (["hour_of_week", "price"], ["day_of_week", "start_time", "end_time", "price"])
            if profile_format == "weekly"
            else None
        ),
    )
    identifier = profile_id or Path(path).stem
    _require_headers(
        fieldnames,
        expected=[
            ["hour_of_week", "price"],
            ["day_of_week", "start_time", "end_time", "price"],
        ],
        path=path,
        profile_format=profile_format,
    )
    if fieldnames == ["hour_of_week", "price"]:
        if len(rows) != 168:
            raise SchemaValidationError(
                f"hour_of_week price profile '{path}' (format: {profile_format}) must "
                "contain exactly 168 rows."
            )
        values: dict[int, float] = {}
        for row in rows:
            try:
                hour = int(row["hour_of_week"])
                price = _finite_price(float(row["price"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaValidationError(
                    f"hour_of_week price profile '{path}' (format: {profile_format}) "
                    "contains an invalid row."
                ) from exc
            if hour in values or not 0 <= hour < 168:
                raise SchemaValidationError(
                    f"hour_of_week price profile '{path}' (format: {profile_format}) "
                    "must contain unique integer values from 0 to 167."
                )
            values[hour] = price
        if set(values) != set(range(168)):
            raise SchemaValidationError(
                f"hour_of_week price profile '{path}' (format: {profile_format}) must "
                "cover every hour 0 through 167."
            )
        periods = tuple(
            PricePeriod(hour * 60, (hour + 1) * 60, values[hour]) for hour in range(168)
        )
        return WeeklyPriceProfile(identifier, periods)
    parsed: list[PricePeriod] = []
    for row in rows:
        try:
            day = _DAY_INDEX[row["day_of_week"].strip().title()]
            start = _clock_minute(row["start_time"])
            end = _clock_minute(row["end_time"], allow_24=True)
            price = _finite_price(float(row["price"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"weekly price profile '{path}' (format: {profile_format}) contains an invalid row."
            ) from exc
        if end <= start:
            raise PhysicalConstraintError("weekly price period must end after it starts.")
        parsed.append(PricePeriod(day * 1440 + start, day * 1440 + end, price))
    parsed.sort(key=lambda period: period.start_minute_of_week)
    return WeeklyPriceProfile(identifier, tuple(parsed))


def load_timestamped_price_profile_csv(
    path: str | Path,
    *,
    profile_id: str | None = None,
) -> TimestampedPriceProfile:
    _fieldnames, rows = _csv_rows(
        path,
        profile_format="timestamped",
        expected_headers=["start_time", "end_time", "price"],
    )
    periods: list[TimestampedPricePeriod] = []
    for row in rows:
        try:
            start = datetime.fromisoformat(row["start_time"])
            end = datetime.fromisoformat(row["end_time"])
            price = _finite_price(float(row["price"]))
            periods.append(TimestampedPricePeriod(start, end, price))
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"timestamped price profile '{path}' (format: timestamped) contains an invalid row."
            ) from exc
    return TimestampedPriceProfile(profile_id or Path(path).stem, tuple(periods))


def load_price_profile_csv(
    path: str | Path,
    *,
    profile_format: str,
    profile_id: str | None = None,
) -> WeeklyPriceProfile | TimestampedPriceProfile:
    if profile_format in ("weekly", "hour_of_week"):
        return load_weekly_price_profile_csv(
            path, profile_id=profile_id, profile_format=profile_format
        )
    if profile_format == "timestamped":
        return load_timestamped_price_profile_csv(path, profile_id=profile_id)
    raise SchemaValidationError(
        "price-profile-format must be weekly, hour_of_week, or timestamped."
    )
