from __future__ import annotations

import math

import pytest

from evmenu import (
    EVModel,
    SchemaValidationError,
    generate_ev_menu,
    get_ev_model,
    list_ev_models,
    search_ev_models,
)


def test_catalog_is_deterministic_and_convertible() -> None:
    models = list_ev_models()
    assert models == tuple(sorted(models, key=lambda model: model.model_id))
    assert models
    assert all(model.to_ev_spec().ev_id == model.model_id for model in models)


def test_get_and_search_models() -> None:
    model = get_ev_model("generic_40kwh_lfp")
    assert model.chemistry == "LFP"
    assert search_ev_models("40 KWH") == (model,)
    assert get_ev_model(model) is model


def test_unknown_model_is_clear() -> None:
    with pytest.raises(SchemaValidationError, match="Unknown ev_model"):
        get_ev_model("missing")


def test_custom_model_validation() -> None:
    custom = EVModel(
        model_id="custom",
        display_name="Custom EV",
        usable_battery_kwh=50.0,
        minimum_soc=0.05,
        onboard_ac_power_kw=7.0,
        charging_efficiency=0.9,
        chemistry="NMC",
        consumption_kwh_per_km=0.16,
        assumption_note="User supplied.",
    )
    assert get_ev_model(custom) is custom


def test_catalogue_ids_are_unique_and_lookup_is_case_sensitive() -> None:
    models = list_ev_models()
    assert len({model.model_id for model in models}) == len(models)
    assert get_ev_model(" generic_40kwh_lfp ").model_id == "generic_40kwh_lfp"
    with pytest.raises(SchemaValidationError, match="Unknown ev_model"):
        get_ev_model("GENERIC_40KWH_LFP")


def test_search_policy_is_case_insensitive_and_rejects_empty_queries() -> None:
    assert [model.model_id for model in search_ev_models("  GENERIC  ")] == [
        "generic_40kwh_lfp",
        "generic_60kwh_nmc",
    ]
    assert search_ev_models("does-not-match") == ()
    with pytest.raises(SchemaValidationError):
        search_ev_models(" ")
    with pytest.raises(SchemaValidationError):
        search_ev_models(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("usable_battery_kwh", True),
        ("usable_battery_kwh", 0.0),
        ("minimum_soc", 1.0),
        ("onboard_ac_power_kw", math.nan),
        ("charging_efficiency", 0.0),
        ("consumption_kwh_per_km", math.inf),
        ("chemistry", "unknown"),
    ],
)
def test_malformed_custom_models_are_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "model_id": "custom",
        "display_name": "Custom EV",
        "usable_battery_kwh": 50.0,
        "minimum_soc": 0.05,
        "onboard_ac_power_kw": 7.0,
        "charging_efficiency": 0.9,
        "chemistry": "NMC",
        "consumption_kwh_per_km": 0.16,
        "assumption_note": "User supplied.",
    }
    values[field] = value
    with pytest.raises((SchemaValidationError, ValueError)):
        EVModel(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_id", [model.model_id for model in list_ev_models()])
def test_each_catalogue_model_generates_the_documented_menu(model_id: str) -> None:
    menu = generate_ev_menu(
        ev_model=model_id,
        arrival_time="19:00",
        departure_time="07:00",
        current_soc=0.35,
        next_trip_distance_km=45.0,
    )
    assert menu.ev_model.model_id == model_id
    assert menu.offers
