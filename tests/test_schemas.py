"""Behavioural tests for immutable Commit 1 data contracts."""

from collections.abc import Callable
from typing import TypeVar, cast

import pytest

from evmenu import (
    ChargingProfile,
    ChargingSession,
    Chemistry,
    EVSpec,
    MenuOffer,
    MenuSettings,
    PhysicalConstraintError,
    PlanningSignal,
    SchemaValidationError,
    SignalValidationError,
    TargetOption,
    TargetSource,
)

T = TypeVar("T")


def as_tuple(values: list[T]) -> tuple[T, ...]:
    """Pass a deliberately wrong runtime list through a tuple-typed API."""
    return cast(tuple[T, ...], values)


def valid_ev() -> EVSpec:
    return EVSpec(
        ev_id="EV001",
        battery_capacity_kwh=50.0,
        minimum_energy_kwh=5.0,
        charger_power_kw=7.2,
        charging_efficiency=0.92,
        chemistry="NMC",
    )


def valid_signal(number_of_steps: int = 4) -> PlanningSignal:
    return PlanningSignal(
        timestep_hours=0.5,
        price_per_kwh=tuple(0.2 for _ in range(number_of_steps)),
    )


def valid_profile() -> ChargingProfile:
    return ChargingProfile(
        start_step=4,
        grid_energy_kwh=(2.0, 0.0),
        battery_energy_kwh=(20.0, 21.8, 21.8),
        power_kw=(2.0, 0.0),
        soc=(0.4, 0.436, 0.436),
    )


def valid_offer(profile: ChargingProfile | None = None) -> MenuOffer:
    return MenuOffer(
        offer_id="offer-1",
        ev_id="EV001",
        target_sources=("standard_80",),
        ready_step=8,
        target_soc=0.8,
        charging_cost=2.0,
        same_target_bau_cost=3.0,
        advertised_saving=1.0,
        incremental_degradation=0.01,
        annualized_degradation_pct=2.0,
        charging_health_score=90.0,
        profile=valid_profile() if profile is None else profile,
    )


def test_valid_schemas_are_created_with_canonical_identifiers() -> None:
    ev = EVSpec("  EV001  ", 50.0, 5.0, 7.2, 0.92, "NMC")
    target = TargetOption(0.8, ("standard_80",), "  Standard 80  ")
    offer = MenuOffer(
        "  offer-1  ",
        "  EV001  ",
        ("standard_80",),
        8,
        0.8,
        2.0,
        3.0,
        1.0,
        0.01,
        2.0,
        90.0,
        valid_profile(),
    )

    assert ev.ev_id == "EV001"
    assert target.label == "Standard 80"
    assert offer.offer_id == "offer-1"
    assert offer.ev_id == "EV001"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf"), "x"])
@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EVSpec("EV", cast(float, value), 0.0, 1.0, 1.0, "LFP"),
        lambda value: EVSpec("EV", 50.0, cast(float, value), 1.0, 1.0, "LFP"),
        lambda value: EVSpec("EV", 50.0, 0.0, cast(float, value), 1.0, "LFP"),
        lambda value: EVSpec("EV", 50.0, 0.0, 1.0, cast(float, value), "LFP"),
        lambda value: ChargingSession(1, 2, cast(float, value), 1.0, 1.0),
        lambda value: ChargingSession(1, 2, 1.0, cast(float, value), 1.0),
        lambda value: ChargingSession(1, 2, 1.0, 1.0, cast(float, value)),
        lambda value: PlanningSignal(cast(float, value), (0.2,)),
        lambda value: TargetOption(cast(float, value), ("standard_80",), "Standard"),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            cast(float, value),
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            50.0,
            valid_profile(),
        ),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            0.8,
            cast(float, value),
            1.0,
            0.0,
            0.0,
            0.0,
            50.0,
            valid_profile(),
        ),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            0.8,
            1.0,
            cast(float, value),
            0.0,
            0.0,
            0.0,
            50.0,
            valid_profile(),
        ),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            0.8,
            1.0,
            1.0,
            cast(float, value),
            0.0,
            0.0,
            50.0,
            valid_profile(),
        ),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            0.8,
            1.0,
            1.0,
            0.0,
            cast(float, value),
            0.0,
            50.0,
            valid_profile(),
        ),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            0.8,
            1.0,
            1.0,
            0.0,
            0.0,
            cast(float, value),
            50.0,
            valid_profile(),
        ),
        lambda value: MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            1,
            0.8,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            cast(float, value),
            valid_profile(),
        ),
        lambda value: MenuSettings(target_merge_tolerance=cast(float, value)),
        lambda value: MenuSettings(numerical_tolerance=cast(float, value)),
        lambda value: MenuSettings(reference_degradation_pct=cast(float, value)),
    ],
)
def test_numeric_scalar_fields_reject_bool_nonreal_and_nonfinite(
    factory: Callable[[object], object], value: object
) -> None:
    with pytest.raises((SchemaValidationError, SignalValidationError, PhysicalConstraintError)):
        factory(value)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_step_fields_reject_bool_and_noninteger_values(value: object) -> None:
    with pytest.raises(SchemaValidationError):
        ChargingSession(cast(int, value), 2, 1.0, 0.0, 0.0)
    with pytest.raises(SchemaValidationError):
        ChargingSession(1, cast(int, value), 1.0, 0.0, 0.0)
    with pytest.raises(SchemaValidationError):
        ChargingProfile(cast(int, value), (0.0,), (1.0, 1.0), (0.0,), (0.2, 0.2))
    with pytest.raises(SchemaValidationError):
        MenuOffer(
            "offer",
            "EV",
            ("standard_80",),
            cast(int, value),
            0.8,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            50.0,
            valid_profile(),
        )
    with pytest.raises(SchemaValidationError):
        MenuSettings(equivalent_sessions_per_year=cast(int, value))


def test_ev_physical_constraints_and_chemistry_are_enforced() -> None:
    with pytest.raises(PhysicalConstraintError):
        EVSpec("EV", 50.0, 50.0, 7.2, 0.9, "LFP")
    with pytest.raises(PhysicalConstraintError):
        EVSpec("EV", 50.0, 5.0, 7.2, 0.0, "LFP")
    with pytest.raises(SchemaValidationError):
        EVSpec("EV", 50.0, 5.0, 7.2, 0.9, cast(Chemistry, "other"))
    assert EVSpec("EV", 50.0, 5.0, 7.2, 1.0, "LFP").charging_efficiency == 1.0


def test_session_rejects_invalid_window_and_unserviceable_commute() -> None:
    with pytest.raises(PhysicalConstraintError):
        ChargingSession(10, 10, 20.0, 5.0, 2.0)
    session = ChargingSession(4, 10, 20.0, 44.0, 2.0)
    with pytest.raises(PhysicalConstraintError):
        session.validate_for_ev(valid_ev())


def test_session_validation_checks_initial_energy_against_ev_bounds() -> None:
    with pytest.raises(PhysicalConstraintError):
        ChargingSession(4, 10, 4.0, 0.0, 0.0).validate_for_ev(valid_ev())
    with pytest.raises(PhysicalConstraintError):
        ChargingSession(4, 10, 51.0, 0.0, 0.0).validate_for_ev(valid_ev())


def test_planning_signal_accepts_negative_prices_and_arbitrary_horizons() -> None:
    signal = PlanningSignal(0.5, (-0.25, 0.0, 0.4))
    assert signal.number_of_steps == 3
    assert signal.price_per_kwh[0] == -0.25


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_planning_signal_rejects_nonfinite_prices(value: float) -> None:
    with pytest.raises(SignalValidationError):
        PlanningSignal(1.0, (value,))


def test_planning_signal_optional_vectors_must_match_horizon_and_constraints() -> None:
    with pytest.raises(SignalValidationError):
        PlanningSignal(1.0, (0.2, 0.3), base_load_kw=(10.0,))
    with pytest.raises(SignalValidationError):
        PlanningSignal(1.0, (0.2, 0.3), battery_temperature_c=(20.0,))
    with pytest.raises(SignalValidationError):
        PlanningSignal(1.0, (0.2,), base_load_kw=(-1.0,))
    with pytest.raises(SignalValidationError):
        PlanningSignal(1.0, (0.2,), battery_temperature_c=(-273.15,))


def test_planning_signal_and_profiles_copy_caller_owned_lists() -> None:
    prices = [-0.1, 0.2]
    base_load = [10.0, 11.0]
    temperatures = [20.0, 21.0]
    signal = PlanningSignal(
        1.0,
        as_tuple(prices),
        as_tuple(base_load),
        as_tuple(temperatures),
    )
    prices[0] = float("nan")
    base_load[0] = -1.0
    temperatures[0] = -300.0

    grid_energy = [0.0]
    battery_energy = [20.0, 20.0]
    power = [0.0]
    soc = [0.4, 0.4]
    profile = ChargingProfile(
        0,
        as_tuple(grid_energy),
        as_tuple(battery_energy),
        as_tuple(power),
        as_tuple(soc),
    )
    grid_energy[0] = -1.0
    battery_energy[0] = -1.0
    power[0] = -1.0
    soc[0] = 2.0

    assert signal.price_per_kwh == (-0.1, 0.2)
    assert signal.base_load_kw == (10.0, 11.0)
    assert signal.battery_temperature_c == (20.0, 21.0)
    assert profile.grid_energy_kwh == (0.0,)
    assert profile.battery_energy_kwh == (20.0, 20.0)
    assert profile.power_kw == (0.0,)
    assert profile.soc == (0.4, 0.4)


def test_tuple_inputs_are_materialized_as_fresh_tuples() -> None:
    prices = (0.2, 0.3)
    signal = PlanningSignal(1.0, prices)
    targets = (0.8, 0.9, 1.0)
    settings = MenuSettings(targets)

    assert signal.price_per_kwh is not prices
    assert settings.standard_targets is not targets


def test_overnight_session_is_covered_by_extended_horizon() -> None:
    signal = valid_signal(32)
    session = ChargingSession(18, 32, 20.0, 0.0, 0.0)
    signal.validate_session_window(session)


def test_session_window_rejects_departure_outside_horizon() -> None:
    signal = valid_signal(32)
    with pytest.raises(SignalValidationError):
        signal.validate_session_window(ChargingSession(18, 33, 20.0, 0.0, 0.0))


def test_target_option_preserves_merged_sources_in_deterministic_order() -> None:
    target = TargetOption(0.8, ("standard_80", "minimum_required"), "Daily need")
    assert target.sources == ("minimum_required", "standard_80")


@pytest.mark.parametrize(
    "sources",
    [(), ("standard_80", "standard_80"), ("not-a-source",)],
)
def test_target_option_rejects_empty_duplicate_or_unknown_sources(
    sources: tuple[str, ...],
) -> None:
    with pytest.raises(SchemaValidationError):
        TargetOption(0.8, cast(tuple[TargetSource, ...], sources), "Target")


def test_target_option_source_list_is_frozen() -> None:
    sources = ["minimum_required", "standard_80"]
    target = TargetOption(0.8, cast(tuple[TargetSource, ...], sources), "Target")
    sources.append("standard_90")
    assert target.sources == ("minimum_required", "standard_80")


def test_target_soc_zero_and_one_are_valid() -> None:
    assert TargetOption(0.0, ("minimum_required",), "No charge").target_soc == 0.0
    assert TargetOption(1.0, ("standard_100",), "Full").target_soc == 1.0


def test_charging_profile_uses_n_interval_and_n_plus_one_state_values() -> None:
    profile = valid_profile()
    assert profile.start_step == 4
    assert len(profile.power_kw) == 2
    assert len(profile.grid_energy_kwh) == 2
    assert len(profile.battery_energy_kwh) == 3
    assert len(profile.soc) == 3


def test_charging_profile_rejects_empty_and_negative_battery_energy() -> None:
    with pytest.raises(SchemaValidationError):
        ChargingProfile(0, (), (20.0,), (), (0.4,))
    with pytest.raises(PhysicalConstraintError):
        ChargingProfile(0, (0.0,), (-1.0, 0.0), (0.0,), (0.4, 0.4))


def test_charging_profile_rejects_negative_start_step() -> None:
    with pytest.raises(SchemaValidationError):
        ChargingProfile(-1, (0.0,), (20.0, 20.0), (0.0,), (0.4, 0.4))


def test_no_charge_profile_is_valid_when_it_has_time_aligned_intervals() -> None:
    profile = ChargingProfile(
        start_step=4,
        grid_energy_kwh=(0.0, 0.0),
        battery_energy_kwh=(20.0, 20.0, 20.0),
        power_kw=(0.0, 0.0),
        soc=(0.4, 0.4, 0.4),
    )
    assert profile.power_kw == (0.0, 0.0)


def test_menu_offer_validates_profile_and_allows_negative_costs() -> None:
    offer = MenuOffer(
        "offer",
        "EV",
        ("standard_80",),
        1,
        0.8,
        -1.0,
        -0.5,
        0.5,
        0.0,
        0.0,
        50.0,
        valid_profile(),
    )
    assert offer.charging_cost == -1.0
    with pytest.raises(SchemaValidationError):
        valid_offer(cast(ChargingProfile, object()))


def test_menu_offer_target_sources_are_frozen_and_canonical() -> None:
    sources = ["standard_80", "minimum_required"]
    offer = MenuOffer(
        "offer",
        "EV",
        cast(tuple[TargetSource, ...], sources),
        1,
        0.8,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        50.0,
        valid_profile(),
    )
    sources.clear()
    assert offer.target_sources == ("minimum_required", "standard_80")


def test_menu_settings_targets_are_frozen_and_merge_tolerance_is_bounded() -> None:
    targets = [0.8, 0.9, 1.0]
    settings = MenuSettings(standard_targets=as_tuple(targets))
    targets[0] = -1.0
    assert settings.standard_targets == (0.8, 0.9, 1.0)
    with pytest.raises(SchemaValidationError):
        MenuSettings(target_merge_tolerance=1.0)


def test_menu_settings_rejects_nonincreasing_targets() -> None:
    with pytest.raises(SchemaValidationError):
        MenuSettings(standard_targets=(0.9, 0.8, 1.0))
