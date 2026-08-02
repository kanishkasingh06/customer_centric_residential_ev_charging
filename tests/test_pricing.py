from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evmenu import (
    PricePeriod,
    TimestampedPricePeriod,
    TimestampedPriceProfile,
    WeeklyPriceProfile,
    load_timestamped_price_profile_csv,
    load_weekly_price_profile_csv,
)
from evmenu.exceptions import PhysicalConstraintError, SchemaValidationError


def hourly_csv() -> str:
    return "hour_of_week,price\n" + "".join(
        f"{hour},{-1.0 if hour == 5 else 2.5}\n" for hour in range(168)
    )


def test_hourly_weekly_csv_is_complete_and_supports_negative_prices(tmp_path: Path) -> None:
    path = tmp_path / "weekly.csv"
    path.write_text(hourly_csv(), encoding="utf-8")
    profile = load_weekly_price_profile_csv(path, profile_id="weekly-test")
    assert profile.profile_id == "weekly-test"
    assert profile.price_at(5 * 60) == -1.0
    assert profile.price_at(7 * 1440 + 167 * 60) == 2.5
    assert profile.absolute_boundaries(start_minute=1433, end_minute=1868) == tuple(
        range(1440, 1868, 60)
    )


def test_weekly_period_csv_rejects_gaps_and_overlaps(tmp_path: Path) -> None:
    path = tmp_path / "periods.csv"
    path.write_text(
        "day_of_week,start_time,end_time,price\nMon,00:00,12:00,1\nMon,13:00,24:00,2\n",
        encoding="utf-8",
    )
    with pytest.raises(PhysicalConstraintError, match="cover the week"):
        load_weekly_price_profile_csv(path)


def test_weekly_profile_rejects_duplicate_or_missing_hour_rows(tmp_path: Path) -> None:
    path = tmp_path / "hours.csv"
    rows = ["hour_of_week,price"] + [f"{hour},1" for hour in range(167)] + ["166,1"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="exactly 168|unique"):
        load_weekly_price_profile_csv(path)


def test_timestamped_csv_requires_continuous_periods(tmp_path: Path) -> None:
    path = tmp_path / "timestamped.csv"
    path.write_text(
        "start_time,end_time,price\n"
        "2026-08-03 00:00,2026-08-03 01:00,3.0\n"
        "2026-08-03 01:30,2026-08-03 02:00,2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(PhysicalConstraintError, match="continuous"):
        load_timestamped_price_profile_csv(path)


@pytest.mark.parametrize(
    ("profile_format", "header", "loader"),
    [
        ("hour_of_week", b"hour_of_week,price\n0,\xff\n", "hourly"),
        ("weekly", b"day_of_week,start_time,end_time,price\nMon,00:00,24:00,\xff\n", "weekly"),
        (
            "timestamped",
            b"start_time,end_time,price\n2026-08-03 00:00,2026-08-03 01:00,\xff\n",
            "timestamped",
        ),
    ],
)
def test_csv_decoding_errors_are_domain_errors(
    tmp_path: Path, profile_format: str, header: bytes, loader: str
) -> None:
    path = tmp_path / f"bad-{profile_format}.csv"
    path.write_bytes(header)
    with pytest.raises(SchemaValidationError, match="not valid UTF-8") as raised:
        if loader == "hourly" or loader == "weekly":
            load_weekly_price_profile_csv(path, profile_format=profile_format)
        else:
            load_timestamped_price_profile_csv(path)
    assert str(path) in str(raised.value)
    assert profile_format in str(raised.value)
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    ("path_name", "content", "loader"),
    [
        ("hourly.csv", 'hour_of_week,price\n0,"3.5\n', "hourly"),
        (
            "timestamped.csv",
            'start_time,end_time,price\n2026-08-03 00:00,"2026-08-03 01:00,3.5\n',
            "timestamped",
        ),
    ],
)
def test_malformed_csv_quoting_is_wrapped(
    tmp_path: Path, path_name: str, content: str, loader: str
) -> None:
    path = tmp_path / path_name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="CSV row") as raised:
        if loader == "hourly":
            load_weekly_price_profile_csv(path, profile_format="hour_of_week")
        else:
            load_timestamped_price_profile_csv(path)
    assert str(path) in str(raised.value)
    assert isinstance(raised.value.__cause__, csv.Error)


@pytest.mark.parametrize(
    "content",
    [
        "hour_of_week,price\n0,3.5,unexpected\n",
        "hour_of_week,price\n0,3.5,unexpected,another\n",
        "hour_of_week,price\n0,3.5,\n",
    ],
)
def test_trailing_csv_columns_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "trailing.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="row .*unexpected values"):
        load_weekly_price_profile_csv(path, profile_format="hour_of_week")


@pytest.mark.parametrize(
    ("content", "loader"),
    [
        ("hour_of_week,price\n0\n", "hourly"),
        ("start_time,end_time,price\n2026-08-03 00:00,,3.5\n", "timestamped"),
        ("start_time,end_time,price\n,2026-08-03 01:00,3.5\n", "timestamped"),
    ],
)
def test_missing_csv_fields_are_rejected(tmp_path: Path, content: str, loader: str) -> None:
    path = tmp_path / "missing-field.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="row .*missing required field"):
        if loader == "hourly":
            load_weekly_price_profile_csv(path, profile_format="hour_of_week")
        else:
            load_timestamped_price_profile_csv(path)


@pytest.mark.parametrize(
    "header",
    [
        "price,hour_of_week",
        "hour_of_week,price,extra",
        "Hour_of_week,price",
        " hour_of_week,price",
        "hour_of_week,price ",
    ],
)
def test_csv_headers_are_exact_and_ordered(tmp_path: Path, header: str) -> None:
    path = tmp_path / "headers.csv"
    path.write_text(f"{header}\n0,3.5\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="headers .*expected"):
        load_weekly_price_profile_csv(path, profile_format="hour_of_week")


@pytest.mark.parametrize("content", ["", "   \n\t\n", "hour_of_week,price\n"])
def test_empty_blank_and_header_only_csv_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_weekly_price_profile_csv(path, profile_format="hour_of_week")


def test_oversized_csv_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "oversized.csv"
    with path.open("wb") as handle:
        handle.seek(10 * 1024 * 1024 + 1)
        handle.write(b"\n")
    with pytest.raises(SchemaValidationError, match="10 MiB"):
        load_weekly_price_profile_csv(path, profile_format="hour_of_week")


def test_weekly_period_and_timestamped_csv_valid_profiles_load(tmp_path: Path) -> None:
    weekly_path = tmp_path / "weekly-periods.csv"
    weekly_path.write_text(
        "day_of_week,start_time,end_time,price\n"
        + "".join(
            f"{day},00:00,24:00,{index + 1}\n"
            for index, day in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
        ),
        encoding="utf-8",
    )
    weekly = load_weekly_price_profile_csv(weekly_path)
    assert weekly.price_at(6 * 1440 + 1) == 7.0

    timestamped_path = tmp_path / "timestamped-valid.csv"
    timestamped_path.write_text(
        "start_time,end_time,price\n"
        "2026-08-03T00:00,2026-08-03T01:00,3.0\n"
        "2026-08-03T01:00,2026-08-03T02:00,-1.0\n",
        encoding="utf-8",
    )
    timestamped = load_timestamped_price_profile_csv(timestamped_path)
    assert timestamped.price_at(datetime.fromisoformat("2026-08-03T01:00")) == -1.0


@pytest.mark.parametrize("periods", [None, "not periods", b"not periods", (object(),), ()])
def test_profile_constructors_validate_period_collections(periods: object) -> None:
    with pytest.raises(SchemaValidationError):
        WeeklyPriceProfile("weekly", periods)  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError):
        TimestampedPriceProfile("timestamped", periods)  # type: ignore[arg-type]


def test_timestamped_profiles_require_consistent_query_datetime_semantics() -> None:
    naive = TimestampedPriceProfile(
        "naive",
        (
            TimestampedPricePeriod(
                datetime.fromisoformat("2026-08-03T00:00"),
                datetime.fromisoformat("2026-08-03T01:00"),
                1.0,
            ),
            TimestampedPricePeriod(
                datetime.fromisoformat("2026-08-03T01:00"),
                datetime.fromisoformat("2026-08-03T02:00"),
                2.0,
            ),
        ),
    )
    aware = TimestampedPriceProfile(
        "aware",
        (
            TimestampedPricePeriod(
                datetime(2026, 8, 3, 0, tzinfo=UTC),
                datetime(2026, 8, 3, 1, tzinfo=UTC),
                1.0,
            ),
            TimestampedPricePeriod(
                datetime(2026, 8, 3, 1, tzinfo=UTC),
                datetime(2026, 8, 3, 2, tzinfo=UTC),
                2.0,
            ),
        ),
    )
    assert naive.price_at(datetime.fromisoformat("2026-08-03T01:00")) == 2.0
    assert aware.boundaries_for_session(
        datetime(2026, 8, 3, 0, tzinfo=UTC), datetime(2026, 8, 3, 2, tzinfo=UTC)
    ) == (60,)
    with pytest.raises(SchemaValidationError, match="timezone-aware.*naive"):
        aware.price_at(datetime.fromisoformat("2026-08-03T01:00"))
    with pytest.raises(SchemaValidationError, match="naive.*timezone-aware"):
        naive.boundaries_for_session(
            datetime(2026, 8, 3, 0, tzinfo=UTC), datetime(2026, 8, 3, 1, tzinfo=UTC)
        )
    with pytest.raises(SchemaValidationError, match="all be naive or all timezone-aware"):
        TimestampedPriceProfile(
            "mixed",
            (
                TimestampedPricePeriod(
                    datetime.fromisoformat("2026-08-03T00:00"),
                    datetime.fromisoformat("2026-08-03T01:00"),
                    1.0,
                ),
                TimestampedPricePeriod(
                    datetime(2026, 8, 3, 1, tzinfo=UTC),
                    datetime(2026, 8, 3, 2, tzinfo=UTC),
                    2.0,
                ),
            ),
        )


def test_immutable_profiles_lookup_exact_boundaries() -> None:
    weekly = WeeklyPriceProfile(
        "x",
        tuple(PricePeriod(hour * 60, (hour + 1) * 60, float(hour)) for hour in range(168)),
    )
    assert weekly.price_at(0) == 0.0
    assert weekly.price_at(10080) == 0.0
    timestamped = TimestampedPriceProfile(
        "t",
        (
            TimestampedPricePeriod(
                datetime(2026, 8, 3, tzinfo=UTC),
                datetime(2026, 8, 3, 1, tzinfo=UTC),
                3.0,
            ),
            TimestampedPricePeriod(
                datetime(2026, 8, 3, 1, tzinfo=UTC),
                datetime(2026, 8, 3, 2, tzinfo=UTC),
                2.0,
            ),
        ),
    )
    assert timestamped.price_at(datetime(2026, 8, 3, 1, tzinfo=UTC)) == 2.0
