"""Smoke tests for the bidirectional choice model.

Deliberately cheap: these check the things that silently produce
plausible-looking nonsense, not the numerics.  Phase 11's lesson was
that a fluent wrong answer is the dangerous one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from choice_v2g import (
    ATTRIBUTES,
    SEGMENTS,
    attribute_matrix,
    choice_probabilities,
    draw_coefficients,
    preference_coverage,
    utilities,
    with_export_aversion,
)


class _Trajectory:
    def __init__(self, terminal_soc: float, capacity: float = 78.0) -> None:
        self.battery_energy_kwh = [0.5 * capacity, terminal_soc * capacity]
        self.soc = [0.5, terminal_soc]


class _Offer:
    """Minimal stand-in carrying only what the attribute matrix reads."""

    def __init__(
        self, *, target: float, delivered: float, cost: float,
        fade: float, ready: int, export: float = 0.0,
    ) -> None:
        self.target_soc = target
        self.delivered_soc = delivered
        self.charging_cost = cost
        self.capacity_fade = fade
        self.ready_minute = ready
        self.export_energy_kwh = export


def _menu() -> list[_Offer]:
    return [
        _Offer(target=0.40, delivered=0.65, cost=0.0, fade=4e-5, ready=1200),
        _Offer(target=0.65, delivered=0.65, cost=40.0, fade=5e-5, ready=1400),
        _Offer(target=1.00, delivered=1.00, cost=120.0, fade=9e-5, ready=1860, export=10.0),
    ]


def test_matrix_reads_delivered_soc_not_the_requested_tier() -> None:
    """The label understates delivery for tiers below arrival SOC.

    Scoring ``b_soc * target_soc`` would see the first two offers as
    0.40 and 0.65 when both deliver 0.65 -- a utility difference that
    does not exist.
    """
    matrix = attribute_matrix(
        _menu(), pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
        deadline_minute=1860,
    )
    assert matrix[0, 3] == pytest.approx(0.65)      # delivered, not 0.40
    assert matrix[0, 3] == matrix[1, 3]


def test_delay_is_lateness_against_the_deadline_not_time_since_plug_in() -> None:
    """Anything ready by the deadline is equally on time."""
    matrix = attribute_matrix(
        _menu(), pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
        deadline_minute=1860,
    )
    assert matrix[0, 2] == pytest.approx(0.0)
    assert matrix[1, 2] == pytest.approx(0.0)
    assert matrix[2, 2] == pytest.approx(0.0)
    late = attribute_matrix(
        _menu(), pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
        deadline_minute=1300,
    )
    assert late[0, 2] == pytest.approx(0.0)         # early, floored at zero
    assert late[2, 2] == pytest.approx((1860 - 1300) / 60.0)


def test_wear_is_in_the_same_currency_as_cost() -> None:
    matrix = attribute_matrix(
        _menu(), pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
        deadline_minute=1860,
    )
    assert matrix[0, 1] == pytest.approx(4e-5 * 78.0 * 12400.0)


def test_zero_export_weight_reproduces_the_four_attribute_model() -> None:
    """The default must not silently change the existing model."""
    rng = np.random.default_rng(0)
    betas = draw_coefficients(SEGMENTS[0], 64, rng)
    assert np.all(betas[:, ATTRIBUTES.index("export")] == 0.0)


def test_export_aversion_penalises_exporting_offers() -> None:
    matrix = attribute_matrix(
        _menu(), pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
        deadline_minute=1860,
    )
    plain = preference_coverage(matrix, SEGMENTS, draws=600, seed=1)
    averse = preference_coverage(
        matrix, with_export_aversion(SEGMENTS, 0.5), draws=600, seed=1
    )
    exporting = 2
    assert averse.argmax_share[exporting] <= plain.argmax_share[exporting]


def test_scale_normalisation_equalises_decisiveness() -> None:
    """Phase 11's confound: a flat segment chooses more randomly, and
    the model reports that as preference rather than scale."""
    rng = np.random.default_rng(3)
    norms = [
        float(np.linalg.norm(draw_coefficients(s, 400, rng), axis=1).mean())
        for s in SEGMENTS
    ]
    assert max(norms) - min(norms) < 1e-6
    raw = [
        float(np.linalg.norm(
            draw_coefficients(s, 400, rng, normalise_scale=False), axis=1
        ).mean())
        for s in SEGMENTS
    ]
    assert max(raw) / min(raw) > 1.5        # the confound is real without it


def test_probabilities_normalise_and_coverage_sums_to_one() -> None:
    matrix = attribute_matrix(
        _menu(), pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
        deadline_minute=1860,
    )
    rng = np.random.default_rng(5)
    probs = choice_probabilities(utilities(matrix, draw_coefficients(SEGMENTS[0], 32, rng)))
    assert np.allclose(probs.sum(axis=1), 1.0)
    coverage = preference_coverage(matrix, draws=400, seed=5)
    assert sum(coverage.argmax_share) == pytest.approx(1.0)
    assert 1.0 <= coverage.effective_menu_size <= len(matrix)


def test_attribute_validation_rejects_nonsense() -> None:
    bad = _menu()
    bad[0].capacity_fade = float("nan")
    with pytest.raises(ValueError):
        attribute_matrix(
            bad, pack_cost_per_kwh=12400.0, battery_capacity_kwh=78.0,
            deadline_minute=1860,
        )
