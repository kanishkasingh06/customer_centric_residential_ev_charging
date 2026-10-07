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


def _load_sibling(name: str):
    """Import a research/ module without putting research/ on sys.path globally."""
    if str(RESEARCH_DIR) not in sys.path:
        sys.path.insert(0, str(RESEARCH_DIR))
    spec = importlib.util.spec_from_file_location(f"research_{name}", RESEARCH_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so the module has to
    # be registered before it is executed or CustomerDraw fails to build.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


import numpy as np

fleet = _load_sibling("fleet")


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
    offer = SimpleNamespace(power_kw=(kilowatts,))
    return SimpleNamespace(
        interval_start_minutes=(start,),
        interval_end_minutes=(end,),
        offers=(offer,),
        bau_offers=(),
        choice_set=(offer,),
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


# ---------------------------------------------------------------------------
# Choice-set composition and the monetary wear term.
#
# e4bea57 moved the BAU offers out of the displayed menu and repurposed
# charging_health_score into a damage measure. This layer tracked neither, which
# removed "decline to charge" from every choice set, inverted the best_health
# policy, and left the uncontrolled baseline silently comparing max_saving
# against itself. These tests pin the three contracts that were broken.
# ---------------------------------------------------------------------------

choice = _load_sibling("choice")


def _offer(name: str, *, stress: float, cost: float, target: float, role: str, energy: float):
    return fleet.OfferRecord(
        offer_id=name, role=role, target_soc=target, ready_step=1, ready_minute=100,
        charging_cost=cost, advertised_saving=0.0 if role == "bau" else 10.0,
        battery_stress=stress, energy_kwh=energy, power_kw=(7.0,),
    )


def _menu_record():
    return fleet.MenuRecord(
        interval_start_minutes=(60,), interval_end_minutes=(120,),
        offers=(
            _offer("hi", stress=4.0e-4, cost=100.0, target=1.0, role="maximum_saving", energy=28.0),
            _offer("lo", stress=2.0e-4, cost=180.0, target=0.8, role="least_degradation", energy=20.0),
        ),
        bau_offers=(
            _offer("bau8", stress=2.5e-4, cost=210.0, target=0.8, role="bau", energy=20.0),
            _offer("bau0", stress=0.0, cost=0.0, target=0.3, role="bau", energy=0.0),
        ),
        model_id="generic_40kwh_lfp", usable_battery_kwh=40.0, chemistry="LFP",
    )


def test_choice_set_includes_the_bau_references() -> None:
    record = _menu_record()
    assert len(record.choice_set) == 4
    assert record.choice_set[: len(record.offers)] == record.offers
    # The decline-to-charge option must be reachable, or share_no_charge is
    # zero by construction rather than by preference.
    assert any(o.energy_kwh <= 1e-9 for o in record.choice_set)
    assert not any(o.energy_kwh <= 1e-9 for o in record.offers)


def test_wear_cost_is_priced_per_chemistry_and_capacity() -> None:
    lfp = choice.wear_cost_rs(2.939e-4, 40.0, "LFP")
    nmc = choice.wear_cost_rs(3.875e-4, 60.0, "NMC")
    assert lfp == pytest.approx(2.939e-4 * 40.0 * 7850.0)
    assert nmc == pytest.approx(3.875e-4 * 60.0 * 12400.0)
    # Same order as the charging bill, with no fitted coefficient.
    assert 50.0 < lfp < 150.0
    assert 200.0 < nmc < 400.0
    assert nmc > lfp, "the denser, dearer NMC pack must cost more to wear"
    with pytest.raises(ValueError, match="no pack cost for chemistry"):
        choice.wear_cost_rs(1e-4, 40.0, "SODIUM_ION")


def test_pack_cost_sensitivity_scales_linearly() -> None:
    base = choice.wear_cost_rs(3.0e-4, 40.0, "LFP")
    up = choice.wear_cost_rs(3.0e-4, 40.0, "LFP", choice.scaled_pack_costs(1.25))
    assert up == pytest.approx(base * 1.25)
    with pytest.raises(ValueError):
        choice.scaled_pack_costs(0.0)


def test_wear_enters_utility_as_a_cost_not_a_benefit() -> None:
    """More battery damage must never raise utility, whatever the mix."""
    record = _menu_record()
    attributes = choice.menu_attribute_matrix(record)
    assert attributes.shape == (4, 4)
    for name, mix in choice.PREFERENCE_MIXES.items():
        betas = mix.draw(np.random.default_rng(7), 1)
        cheap = attributes.copy()
        damaged = attributes.copy()
        damaged[:, 1] = cheap[:, 1] + 50.0          # 50 rupees more wear
        worse = choice.utilities(damaged, betas)
        better = choice.utilities(cheap, betas)
        assert np.all(worse < better), f"{name}: extra wear increased utility"


def test_preference_mixes_price_wear_against_money() -> None:
    for mix in choice.PREFERENCE_MIXES.values():
        assert mix.median_wear == pytest.approx(
            choice.REFERENCE_MONEY_WEIGHT * mix.wear_internalisation
        )
    assert choice.PREFERENCE_MIXES["health_driven"].wear_internalisation > 1.0
    assert choice.PREFERENCE_MIXES["convenience_driven"].wear_internalisation < 1.0
    with pytest.raises(ValueError, match="wear_internalisation"):
        choice.PreferenceDistribution(wear_internalisation=0.0)


def test_bank_refuses_a_stale_schema(tmp_path) -> None:
    """A bank from an older record shape must fail loudly, not half-load."""
    import pickle

    path = tmp_path / "menu_bank.pkl"
    with path.open("wb") as handle:
        pickle.dump({"schema_version": 1, "menus": {}, "infeasible": set()}, handle)
    with pytest.raises(SystemExit, match="schema version"):
        fleet.MenuBank(path)


def test_wear_weight_orders_by_internalisation_not_by_price_sensitivity() -> None:
    """A segment defined by caring about battery life must weigh wear more.

    Regression test. The first version anchored b_wear to each segment's OWN
    median_cost, so health_driven (0.008 x 2.50) and balanced (0.020 x 1.00)
    both came out at 0.020 -- identical absolute weight. A rupee of wear then
    moved the two segments equally, and health_driven differed only by caring
    less about money. In the sweep it wore batteries MORE than balanced
    (48.9-59.0 Rs against 40.2-47.7 Rs), which is the opposite of its name.
    """
    mixes = choice.PREFERENCE_MIXES
    balanced = mixes["balanced"]
    health = mixes["health_driven"]
    cost = mixes["cost_driven"]

    assert health.median_wear > balanced.median_wear, (
        "health_driven must weigh wear more in ABSOLUTE utils, not just "
        "relative to its own cost coefficient"
    )
    assert cost.median_wear < balanced.median_wear

    # The ordering must survive wildly different price sensitivities, which is
    # exactly what the broken version failed to do.
    assert health.median_cost < cost.median_cost
    for name, mix in mixes.items():
        assert mix.median_wear == pytest.approx(
            choice.REFERENCE_MONEY_WEIGHT * mix.wear_internalisation
        ), f"{name}: wear weight must be anchored to the reference money scale"


def test_a_wear_sensitive_segment_actually_responds_to_wear() -> None:
    """Utility must move more for health_driven than for cost_driven."""
    extra_wear_rs = 50.0
    health = choice.PREFERENCE_MIXES["health_driven"]
    cost = choice.PREFERENCE_MIXES["cost_driven"]
    assert health.median_wear * extra_wear_rs > cost.median_wear * extra_wear_rs * 2.0
