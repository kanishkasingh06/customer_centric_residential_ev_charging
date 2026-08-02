from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from evmenu import (
    CustomerMenuRow,
    EVModel,
    GeneratedCustomerMenu,
    PhysicalConstraintError,
    PricePeriod,
    SchemaValidationError,
    TimestampedPricePeriod,
    TimestampedPriceProfile,
    WeeklyPriceProfile,
    generate_ev_menu,
)


def _menu() -> GeneratedCustomerMenu:
    return generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="19:00",
        departure_time="07:00",
        current_soc=0.35,
        next_trip_distance_km=45.0,
    )


def _generate_dynamic(**kwargs: object) -> GeneratedCustomerMenu:
    return generate_ev_menu(**kwargs)  # type: ignore[arg-type]


def test_high_level_service_generates_aligned_customer_rows() -> None:
    menu = _menu()
    assert menu.arrival_time == "19:00"
    assert menu.departure_time == "07:00"
    assert menu.tariff_is_illustrative
    assert menu.offers
    assert tuple(row.offer_id for row in menu.offers) == tuple(
        offer.offer_id for offer in menu.assembled_menu.offers
    )
    assert all(isinstance(row, CustomerMenuRow) for row in menu.offers)
    assert all(len(row.charging_schedule_kw) == 48 for row in menu.offers)
    assert menu.timestep_minutes == 15


def test_service_is_deterministic() -> None:
    assert _menu() == _menu()


def test_overnight_ready_times_are_clock_times() -> None:
    menu = _menu()
    assert all(len(row.ready_time) == 5 and row.ready_time[2] == ":" for row in menu.offers)
    assert any(row.ready_time.startswith("0") for row in menu.offers)


def test_equal_times_are_rejected_as_ambiguous() -> None:
    with pytest.raises(PhysicalConstraintError, match="must differ"):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="08:00",
            departure_time="08:00",
            current_soc=0.5,
            next_trip_distance_km=10.0,
        )


def test_flat_tariff_is_supported() -> None:
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="19:00",
        departure_time="23:00",
        current_soc=0.6,
        next_trip_distance_km=10.0,
        tariff_name="flat",
        flat_price_per_kwh=6.0,
    )
    assert not menu.tariff_is_illustrative
    assert menu.tariff_name == "flat"


@pytest.mark.parametrize(
    ("model", "soc", "distance"),
    [
        ("generic_60kwh_nmc", 0.35, 45.0),
        ("generic_40kwh_lfp", 0.50, 0.0),
    ],
)
def test_former_frontier_failures_generate_menus(model: str, soc: float, distance: float) -> None:
    menu = generate_ev_menu(
        ev_model=model,
        arrival_time="19:00",
        departure_time="07:00",
        current_soc=soc,
        next_trip_distance_km=distance,
    )
    assert menu.offers


@pytest.mark.parametrize(
    "clock", ["7:00", "07:0", " 07:00", "07:00 ", "24:00", "07:60", "０７:００"]
)
def test_clock_parser_is_strict(clock: str) -> None:
    with pytest.raises(SchemaValidationError):
        _generate_dynamic(**{**_base_kwargs(), "departure_time": clock})


@pytest.mark.parametrize("timestep,steps", [(15, 48), (20, 36), (30, 24), (60, 12)])
def test_supported_timesteps_preserve_interval_count(timestep: int, steps: int) -> None:
    menu = _generate_dynamic(**_base_kwargs(), timestep_minutes=timestep)
    assert menu.timestep_minutes == timestep
    assert all(len(row.charging_schedule_kw) == steps for row in menu.offers)


def test_thirty_minute_nonzero_charge_generates_menu() -> None:
    menu = _generate_dynamic(**_base_kwargs(), timestep_minutes=30)

    assert menu.offers
    assert all(len(row.charging_schedule_kw) == 24 for row in menu.offers)
    assert all(
        0.0 <= soc <= 1.0 for offer in menu.assembled_menu.offers for soc in offer.profile.soc
    )


def test_same_day_and_midnight_windows_have_exact_lengths() -> None:
    same_day_kwargs = _base_kwargs()
    same_day_kwargs.update(arrival_time="09:00", departure_time="17:00")
    same_day = _generate_dynamic(**same_day_kwargs)
    midnight = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="23:45",
        departure_time="00:15",
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
    )
    assert all(len(row.charging_schedule_kw) == 32 for row in same_day.offers)
    assert all(len(row.charging_schedule_kw) == 2 for row in midnight.offers)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"arrival_time": "7pm"}, SchemaValidationError),
        ({"current_soc": 1.1}, PhysicalConstraintError),
        ({"next_trip_distance_km": -1.0}, PhysicalConstraintError),
        ({"buffer_soc": -0.1}, PhysicalConstraintError),
        ({"timestep_minutes": True}, SchemaValidationError),
        ({"timestep_minutes": 7}, PhysicalConstraintError),
        ({"timestep_minutes": 17}, PhysicalConstraintError),
        ({"tariff_name": "unknown"}, SchemaValidationError),
    ],
)
def test_service_rejects_invalid_user_inputs(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    base: dict[str, object] = {
        "ev_model": "generic_40kwh_lfp",
        "arrival_time": "19:00",
        "departure_time": "07:00",
        "current_soc": 0.35,
        "next_trip_distance_km": 45.0,
    }
    base.update(kwargs)
    with pytest.raises(error):
        generate_ev_menu(**base)  # type: ignore[arg-type]


def test_arbitrary_minute_boundaries_are_preserved() -> None:
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="11:07",
        departure_time="18:52",
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
    )
    assert menu.arrival_time == "11:07"
    assert menu.departure_time == "18:52"
    assert menu.interval_start_times[0] == "11:07"
    assert menu.interval_end_times[-1] == "18:52"
    assert menu.interval_duration_minutes[0] == 8
    assert menu.interval_duration_minutes[-1] == 7
    assert sum(menu.interval_duration_minutes) == 465


def test_one_minute_service_session_preserves_exact_interval_and_energy_cap() -> None:
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="19:02",
        departure_time="19:03",
        current_soc=0.999,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
    )
    assert menu.interval_start_minutes == (19 * 60 + 2,)
    assert menu.interval_end_minutes == (19 * 60 + 3,)
    assert menu.interval_duration_minutes == (1,)
    assert len(menu.interval_duration_hours) == 1
    assert menu.interval_duration_hours[0] * menu.ev_model.onboard_ac_power_kw == pytest.approx(
        menu.ev_model.onboard_ac_power_kw / 60.0
    )
    assert all(len(row.charging_schedule_kw) == 1 for row in menu.offers)
    assert max(max(offer.profile.grid_energy_kwh) for offer in menu.assembled_menu.offers) <= (
        menu.ev_model.onboard_ac_power_kw / 60.0 + 1e-12
    )


@pytest.mark.parametrize(
    ("arrival", "departure", "expected_prices"),
    [
        ("05:59", "06:01", (4.0, 7.0)),
        ("06:00", "06:01", (7.0,)),
        ("16:59", "17:01", (7.0, 10.0)),
        ("17:00", "17:01", (10.0,)),
        ("22:59", "23:01", (10.0, 5.0)),
        ("23:00", "23:01", (5.0,)),
        ("23:59", "00:01", (5.0, 4.0)),
    ],
)
def test_builtin_tariff_boundaries_split_exactly(
    arrival: str, departure: str, expected_prices: tuple[float, ...]
) -> None:
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time=arrival,
        departure_time=departure,
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
    )
    assert menu.interval_price_per_kwh == expected_prices
    assert (
        sum(menu.interval_duration_minutes)
        == (
            int(departure[:2]) * 60
            + int(departure[3:])
            - (int(arrival[:2]) * 60 + int(arrival[3:]))
        )
        % 1440
    )


def test_generated_menu_interval_metadata_rejects_malformed_aligned_vectors() -> None:
    menu = _menu()
    count = len(menu.interval_start_minutes)
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_start_minutes=("bad",) * count)  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_end_minutes=(1.5,) * count)  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_duration_minutes=(True,) * count)
    inconsistent = list(menu.interval_duration_minutes)
    inconsistent[0] += 1
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_duration_minutes=tuple(inconsistent))

    gap = list(menu.interval_start_minutes)
    gap[1] += 1
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_start_minutes=tuple(gap))
    overlap = list(menu.interval_start_minutes)
    overlap[1] -= 1
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_start_minutes=tuple(overlap))
    reversed_end = list(menu.interval_end_minutes)
    reversed_end[0] = menu.interval_start_minutes[0] - 1
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_end_minutes=tuple(reversed_end))
    with pytest.raises(SchemaValidationError):
        replace(menu, interval_price_per_kwh=menu.interval_price_per_kwh[:-1])
    altered_row = replace(menu.offers[0], charging_schedule_kw=(0.0,))
    with pytest.raises(SchemaValidationError):
        replace(menu, offers=(altered_row, *menu.offers[1:]))


@pytest.mark.parametrize("label", ["", "  ", "Rs\x00", "x" * 65])
def test_generated_menu_currency_label_is_validated(label: str) -> None:
    with pytest.raises(SchemaValidationError):
        replace(_menu(), currency_label=label)


def test_weekly_custom_profile_splits_at_price_boundaries_and_preserves_metadata() -> None:
    profile = WeeklyPriceProfile(
        "weekly-demo",
        (
            PricePeriod(0, 12 * 60, 1.0),
            PricePeriod(12 * 60, 18 * 60, -2.0),
            PricePeriod(18 * 60, 10080, 3.0),
        ),
        currency_label="USD/kWh",
    )
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="11:07",
        departure_time="13:02",
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
        tariff="custom",
        custom_price_profile=profile,
        arrival_day="Mon",
    )
    assert menu.tariff_name == "custom"
    assert menu.profile_id == "weekly-demo"
    assert menu.currency_label == "USD/kWh"
    assert menu.arrival_day == "Mon"
    assert (menu.interval_start_times[0], menu.interval_end_times[-1]) == ("11:07", "13:02")
    assert 12 * 60 in menu.interval_start_minutes
    assert all(
        len(row.charging_schedule_kw) == len(menu.interval_duration_minutes) for row in menu.offers
    )
    prices_at_noon = [
        price
        for start, price in zip(menu.interval_start_minutes, menu.interval_price_per_kwh)
        if start == 12 * 60
    ]
    assert prices_at_noon == [-2.0]


def test_custom_profile_requires_explicit_weekday() -> None:
    profile = WeeklyPriceProfile("weekly", (PricePeriod(0, 10080, 1.0),))
    with pytest.raises(SchemaValidationError, match="arrival_day"):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="19:02",
            departure_time="19:17",
            current_soc=1.0,
            next_trip_distance_km=0.0,
            buffer_soc=0.0,
            tariff="custom",
            custom_price_profile=profile,
        )


def test_timestamped_custom_profile_crosses_midnight() -> None:
    start = datetime(2026, 8, 3, 23, 0, tzinfo=UTC)
    profile = TimestampedPriceProfile(
        "timestamped-demo",
        tuple(
            TimestampedPricePeriod(
                start + timedelta(hours=index), start + timedelta(hours=index + 1), price
            )
            for index, price in enumerate((1.0, 2.0, 3.0))
        ),
    )
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="23:53",
        departure_time="01:08",
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
        tariff="custom",
        custom_price_profile=profile,
        arrival_date="2026-08-03",
    )
    assert menu.profile_id == "timestamped-demo"
    assert menu.arrival_date == "2026-08-03"
    assert menu.interval_duration_minutes[0] == 7
    assert menu.interval_duration_minutes[-1] == 8
    assert 1440 in menu.interval_start_minutes
    assert 1.0 in menu.interval_price_per_kwh
    assert 2.0 in menu.interval_price_per_kwh


def test_trip_requirement_that_exceeds_capacity_is_rejected() -> None:
    with pytest.raises(PhysicalConstraintError, match="trip_energy=.*required="):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="19:00",
            departure_time="07:00",
            current_soc=0.35,
            next_trip_distance_km=300.0,
        )


def _base_kwargs() -> dict[str, object]:
    return {
        "ev_model": "generic_40kwh_lfp",
        "arrival_time": "19:00",
        "departure_time": "07:00",
        "current_soc": 0.35,
        "next_trip_distance_km": 45.0,
    }


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, 35.0])
def test_invalid_soc_variants_are_rejected(value: object) -> None:
    with pytest.raises((SchemaValidationError, PhysicalConstraintError)):
        _generate_dynamic(**{**_base_kwargs(), "current_soc": value})


def test_soc_floor_and_buffer_capacity_errors_have_context() -> None:
    with pytest.raises(PhysicalConstraintError, match="minimum reserve"):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="19:00",
            departure_time="07:00",
            current_soc=0.0,
            next_trip_distance_km=0.0,
            buffer_soc=0.0,
        )
    with pytest.raises(PhysicalConstraintError, match="buffer_soc=.*required="):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="19:00",
            departure_time="07:00",
            current_soc=0.35,
            next_trip_distance_km=0.0,
            buffer_soc=1.0,
        )


def test_exact_capacity_boundary_is_allowed_and_just_above_is_rejected() -> None:
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="19:00",
        departure_time="07:00",
        current_soc=0.5,
        next_trip_distance_km=0.0,
        buffer_soc=0.95,
    )
    assert menu.offers
    with pytest.raises(PhysicalConstraintError, match="physically impossible"):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="19:00",
            departure_time="07:00",
            current_soc=0.5,
            next_trip_distance_km=0.0,
            buffer_soc=0.950001,
        )


def test_negative_flat_prices_are_supported() -> None:
    menu = _generate_dynamic(**_base_kwargs(), tariff_name="flat", flat_price_per_kwh=-2.0)
    assert menu.tariff_name == "flat"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flat_price_per_kwh": True},
        {"flat_price_per_kwh": math.nan},
        {"flat_price_per_kwh": math.inf},
        {"battery_temperature_c": True},
        {"battery_temperature_c": math.nan},
        {"battery_temperature_c": -273.15},
    ],
)
def test_tariff_and_temperature_values_are_finite_and_physical(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((SchemaValidationError, PhysicalConstraintError)):
        _generate_dynamic(**_base_kwargs(), **kwargs)


def test_custom_model_generates_menu() -> None:
    custom = EVModel(
        model_id="custom",
        display_name="Custom EV",
        usable_battery_kwh=40.0,
        minimum_soc=0.05,
        onboard_ac_power_kw=7.2,
        charging_efficiency=0.9,
        chemistry="LFP",
        consumption_kwh_per_km=0.15,
        assumption_note="Caller supplied.",
    )
    menu = generate_ev_menu(
        ev_model=custom,
        arrival_time="19:00",
        departure_time="07:00",
        current_soc=0.35,
        next_trip_distance_km=45.0,
    )
    assert menu.ev_model is custom


def test_zero_trip_and_zero_charging_requirement() -> None:
    zero_trip = _generate_dynamic(**{**_base_kwargs(), "next_trip_distance_km": 0.0})
    zero_charge = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="09:00",
        departure_time="09:15",
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
    )
    assert zero_trip.offers
    assert all(row.energy_drawn_kwh == 0.0 for row in zero_charge.offers)


def test_insufficient_window_is_rejected() -> None:
    with pytest.raises(PhysicalConstraintError, match="target cannot be reached"):
        generate_ev_menu(
            ev_model="generic_40kwh_lfp",
            arrival_time="09:00",
            departure_time="09:15",
            current_soc=0.05,
            next_trip_distance_km=0.0,
            buffer_soc=0.0,
        )


def test_customer_row_direct_validation_never_leaks_type_error() -> None:
    row = _menu().offers[0]
    with pytest.raises(SchemaValidationError):
        replace(row, charging_schedule_kw=None)  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError):
        replace(row, charging_schedule_kw=())
    with pytest.raises(SchemaValidationError):
        replace(row, charging_schedule_kw=list(row.charging_schedule_kw))  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError):
        replace(row, role="unsupported")
    with pytest.raises(SchemaValidationError):
        replace(row, ready_time="7:00")


def test_generated_menu_direct_validation_and_exact_row_correspondence() -> None:
    menu = _menu()

    def rebuild(**changes: object) -> GeneratedCustomerMenu:
        values: dict[str, object] = {
            "ev_model": menu.ev_model,
            "arrival_time": menu.arrival_time,
            "departure_time": menu.departure_time,
            "timestep_minutes": menu.timestep_minutes,
            "current_soc": menu.current_soc,
            "next_trip_distance_km": menu.next_trip_distance_km,
            "tariff_name": menu.tariff_name,
            "tariff_is_illustrative": menu.tariff_is_illustrative,
            "offers": menu.offers,
            "assembled_menu": menu.assembled_menu,
        }
        values.update(changes)
        return GeneratedCustomerMenu(**values)  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError):
        rebuild(offers=None)
    with pytest.raises(SchemaValidationError):
        rebuild(offers=list(menu.offers))
    with pytest.raises(PhysicalConstraintError):
        rebuild(current_soc=2.0)
    with pytest.raises(PhysicalConstraintError):
        rebuild(next_trip_distance_km=-1.0)
    with pytest.raises(SchemaValidationError):
        rebuild(arrival_time="7:00")
    with pytest.raises(SchemaValidationError):
        rebuild(tariff_name="unsupported")
    with pytest.raises(SchemaValidationError):
        rebuild(timestep_minutes=True)
    with pytest.raises(PhysicalConstraintError):
        rebuild(timestep_minutes=7)
    with pytest.raises(SchemaValidationError):
        rebuild(ev_model=replace(menu.ev_model, model_id="other"))
    with pytest.raises(SchemaValidationError):
        rebuild(offers=(menu.offers[0],) * len(menu.offers))
    with pytest.raises(SchemaValidationError):
        rebuild(offers=menu.offers[:-1])
    altered = replace(menu.offers[0], charging_cost=menu.offers[0].charging_cost + 1.0)
    with pytest.raises(SchemaValidationError, match="charging_cost"):
        rebuild(offers=(altered, *menu.offers[1:]))

    row = menu.offers[0]
    changed_rows = (
        {"saving": row.saving + 1.0},
        {"health_score": row.health_score + 1.0 if row.health_score < 99.0 else 0.0},
        {
            "target_soc_percent": (
                row.target_soc_percent + 1.0
                if row.target_soc_percent < 99.0
                else row.target_soc_percent - 1.0
            )
        },
        {"ready_time": "7:00"},
        {"energy_drawn_kwh": row.energy_drawn_kwh + 1.0},
        {
            "charging_schedule_kw": (
                row.charging_schedule_kw[0] + 1.0,
                *row.charging_schedule_kw[1:],
            )
        },
        {"role": "intermediate" if row.role == "bau" else "bau"},
    )
    for changes in changed_rows:
        with pytest.raises(SchemaValidationError):
            changed = replace(row, **changes)
            rebuild(offers=(changed, *menu.offers[1:]))


def test_rows_match_every_assembled_offer_exactly() -> None:
    menu = _menu()
    source = {item.offer_id: item for item in menu.assembled_menu.source_metadata}
    for row, offer in zip(menu.offers, menu.assembled_menu.offers, strict=True):
        assert row.offer_id == offer.offer_id
        assert row.target_soc_percent == pytest.approx(offer.target_soc * 100.0)
        assert row.energy_drawn_kwh == pytest.approx(sum(offer.profile.grid_energy_kwh))
        assert row.charging_schedule_kw == offer.profile.power_kw
        assert row.role == source[offer.offer_id].endpoint_role
