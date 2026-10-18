"""Customer choice over a bidirectional menu, and preference coverage.

Separate from ``choice.py`` for the same reason ``evmenu_v2g`` is
separate from ``evmenu``: ``fleet.py`` and the existing research tests
depend on the unidirectional model, and changing it underneath them
would mix two questions.  Four things differ here.

**Delivered state of charge, not the requested tier.**  ``choice.py``
scores ``b_soc * offer.target_soc``.  On a bidirectional menu the tier
is the REQUEST; terminal energy is pinned to ``max(zeta, z) B_max``, so
two offers labelled 39% and 65% can deliver the same car.  Scoring the
label invents a utility difference.  This module reads
``offer.delivered_soc``.

**An export term.**  A V2G offer lends the customer's battery out.  The
four-attribute utility can only see that through cost and wear, which
assumes the customer is indifferent to export as such -- no range
anxiety mid-session, no loss-of-control, no warranty worry.  That is an
assumption, not a finding, so ``b_export`` exists and defaults to ZERO:
the default reproduces the four-attribute model exactly, and any
non-zero value is a declared sensitivity.

**Currency-neutral.**  ``choice.py`` is denominated in rupees with the
money weight anchored to a rupee.  Money here is whatever unit the
caller's menu and pack cost are in; the anchor is expressed per unit of
that currency, so the LFP/NMC ratio and the wear-to-cost ratio are what
carry meaning, not the absolute numbers.

**Scale-normalised segments.**  Phase 11 left a known confound: the
segments differ in overall coefficient magnitude, so a "flat" segment
chooses more randomly and `preference` is partly `decisiveness`.
``normalise_scale`` divides each draw by its own coefficient norm,
leaving the DIRECTION of preference and equalising decisiveness.  It
defaults to on here, which is a deliberate change from ``choice.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

ATTRIBUTES = ("cost", "wear", "delay", "soc", "export")

#: Utils per unit of money for the reference customer.  Unit-free: it
#: means the same thing in pounds as it did in rupees, provided the menu
#: and the pack cost are in the same unit.
REFERENCE_MONEY_WEIGHT = 0.020

#: End of life, as a fraction of initial capacity lost.  Wear is priced
#: as a write-down of the WHOLE pack over the fade it survives, not of
#: the capacity increment lost:
#:
#:     W = (dQ / dQ_EoL) * B_max * pi
#:
#: Reaching end of life costs an owner a pack, not the 20% of capacity
#: that was lost getting there.  ``choice.py`` omits the divisor and so
#: undercharges wear by exactly this factor; Preis and Biedenbach (2023)
#: review the V2G optimisation literature and find the whole-pack form
#: is what it uses, with 20% the standard threshold.
END_OF_LIFE_CAPACITY_LOSS = 0.20


@dataclass(frozen=True)
class SegmentSpec:
    """One preference segment, as multiples of the reference weights."""

    name: str
    money: float
    #: How much of a unit of consumed pack the customer feels as a unit
    #: out of pocket.  1.0 means a currency unit of wear hurts exactly
    #: as much as a currency unit of electricity.
    wear_internalisation: float
    delay: float
    soc: float
    #: Disutility per kWh exported.  Zero reproduces the four-attribute
    #: model; positive means the customer dislikes export beyond its
    #: effect on the bill and on fade.
    export: float = 0.0


#: Illustrative research assumptions, not estimated from stated- or
#: revealed-preference data.  The wear multiples are Phase 11's.
SEGMENTS: tuple[SegmentSpec, ...] = (
    SegmentSpec("balanced", 1.00, 1.00, 1.00, 1.00),
    SegmentSpec("health_driven", 0.40, 2.50, 0.80, 0.80),
    SegmentSpec("cost_driven", 2.00, 0.40, 0.60, 0.60),
    SegmentSpec("convenience_driven", 0.25, 0.25, 2.50, 2.00),
)


def attribute_matrix(
    offers: Sequence,
    *,
    pack_cost_per_kwh: float,
    battery_capacity_kwh: float,
    deadline_minute: int,
) -> np.ndarray:
    """The (n_offers, 5) attribute matrix for a bidirectional menu.

    Columns are cost, LATENESS in hours, wear, delivered SOC, exported
    kWh -- in the declared ATTRIBUTES order.  Wear is in the same
    currency as cost, so their ratio is the only free quantity, which
    is the point of pricing fade at all.  It is divided by
    END_OF_LIFE_CAPACITY_LOSS, making it a write-down of the whole pack
    rather than of the capacity increment; see that constant.

    Delay is **lateness against the customer's stated ready-by
    deadline**, floored at zero, not elapsed time since plug-in.  The
    distinction decides the model.  Measured from plug-in, an overnight
    session spans eleven hours, so at a weight of one util per hour the
    delay term contributes eleven utils of spread against four for
    cost, one for wear and a third for state of charge: it swamps
    everything and the menu collapses onto whichever offer is ready
    soonest.  A customer asleep between 20:00 and 07:00 is not
    indifferent between those offers by a margin of eleven utils; they
    are indifferent outright, until the deadline passes.
    """
    rows = []
    for offer in offers:
        rows.append(
            (
                offer.charging_cost,
                offer.capacity_fade
                / END_OF_LIFE_CAPACITY_LOSS
                * battery_capacity_kwh
                * pack_cost_per_kwh,
                max(0.0, (offer.ready_minute - deadline_minute) / 60.0),
                offer.delivered_soc,          # NOT target_soc; see module docstring
                offer.export_energy_kwh,
            )
        )
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(ATTRIBUTES):
        raise ValueError(f"attribute matrix must be (n_offers, {len(ATTRIBUTES)})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("menu attributes contain non-finite values")
    if np.any(matrix[:, 2] < -1e-9):
        raise ValueError("negative lateness: floor at zero failed")
    if np.any(matrix[:, 4] < -1e-9):
        raise ValueError("negative export energy")
    return matrix


def draw_coefficients(
    segment: SegmentSpec,
    draws: int,
    rng: np.random.Generator,
    *,
    sigma: float = 0.4,
    normalise_scale: bool = True,
) -> np.ndarray:
    """Lognormal coefficient draws, shape (draws, 5).

    Signs are applied at the utility step, so every draw here is
    positive.  With ``normalise_scale`` each draw is divided by its own
    L2 norm, which fixes the Phase 11 confound: without it a segment
    with small coefficients chooses more randomly, and the model reports
    that as a preference when it is only a scale.
    """
    means = np.array(
        [
            segment.money * REFERENCE_MONEY_WEIGHT,
            segment.wear_internalisation * REFERENCE_MONEY_WEIGHT,
            segment.delay,
            segment.soc,
            segment.export,
        ],
        dtype=float,
    )
    out = np.empty((draws, len(ATTRIBUTES)), dtype=float)
    for index, mean in enumerate(means):
        if mean <= 0.0:
            out[:, index] = 0.0          # an inactive attribute stays inactive
            continue
        mu = np.log(mean) - 0.5 * sigma**2
        out[:, index] = rng.lognormal(mu, sigma, size=draws)
    if normalise_scale:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        out = out / norms
    return out


def utilities(attributes: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Utility of every offer under every draw, shape (draws, n_offers).

    ``U = -b_cost c - b_wear w - b_delay d + b_soc s - b_export x``.
    Export carries a MINUS sign: a positive ``b_export`` means the
    customer dislikes exporting, over and above the bill and the fade.
    """
    signs = np.array([-1.0, -1.0, -1.0, +1.0, -1.0])
    return (betas * signs) @ attributes.T


def choice_probabilities(values: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability."""
    shifted = values - values.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Coverage:
    """How much of the preference population each offer actually wins."""

    argmax_share: tuple[float, ...]
    probability_share: tuple[float, ...]
    dead_offers: tuple[int, ...]
    effective_menu_size: float


def preference_coverage(
    attributes: np.ndarray,
    segments: Sequence[SegmentSpec] = SEGMENTS,
    *,
    draws: int = 4000,
    seed: int = 0,
    normalise_scale: bool = True,
    dead_threshold: float = 0.005,
) -> Coverage:
    """Share of the preference population each offer wins.

    This is the quantity a menu should be optimised against.  Geometric
    spread -- farthest-point on normalised attributes -- maximises how
    different the offers LOOK; it makes no claim that anyone wants them.
    An offer that is never the utility maximum for any plausible
    coefficient draw is dead weight however distant it is.

    ``argmax_share`` is the fraction of draws for which each offer is
    the deterministic best; ``probability_share`` is the mean logit
    probability, which also credits offers that are frequently a close
    second.  ``effective_menu_size`` is the inverse Simpson index of the
    argmax shares -- the number of offers the menu behaves as if it had.
    """
    rng = np.random.default_rng(seed)
    per_segment = []
    for segment in segments:
        betas = draw_coefficients(
            segment, draws, rng, normalise_scale=normalise_scale
        )
        values = utilities(attributes, betas)
        winners = np.argmax(values, axis=1)
        argmax = np.bincount(winners, minlength=attributes.shape[0]) / draws
        probs = choice_probabilities(values).mean(axis=0)
        per_segment.append((argmax, probs))
    argmax_share = np.mean([a for a, _ in per_segment], axis=0)
    probability_share = np.mean([p for _, p in per_segment], axis=0)
    dead = tuple(int(i) for i, s in enumerate(argmax_share) if s < dead_threshold)
    simpson = float(np.sum(argmax_share**2))
    return Coverage(
        argmax_share=tuple(float(x) for x in argmax_share),
        probability_share=tuple(float(x) for x in probability_share),
        dead_offers=dead,
        effective_menu_size=(1.0 / simpson) if simpson > 0 else 0.0,
    )


def with_export_aversion(
    segments: Sequence[SegmentSpec], export: float
) -> tuple[SegmentSpec, ...]:
    """The same segments with a declared export disutility, for sweeps."""
    return tuple(replace(s, export=export) for s in segments)
