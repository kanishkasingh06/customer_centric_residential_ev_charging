"""Mixed-logit customer choice over a generated EV charging menu.

Model
-----
Customer ``n`` facing menu offers ``j`` draws a personal preference vector and
picks the offer maximising random utility

    U_nj = - b_cost,n * cost_j
           - b_wear,n * wear_j
           - b_delay,n * delay_j
           + b_soc,n   * soc_j
           + eps_nj,        eps ~ iid Gumbel(0, 1)

which gives the conditional-logit probability ``softmax(V_n)`` for that draw.
Averaging over the population distribution of ``b`` makes this a mixed logit.

All four coefficients are drawn from lognormal distributions so signs are
guaranteed -- nobody in the population prefers paying more, waiting longer, or
degrading their battery faster.

The wear term, and why it is money
----------------------------------
This used to be a ``health`` term reading ``MenuOffer.charging_health_score``
on a 0-100 "higher is healthier" scale, entering utility with a PLUS sign.
``e4bea57`` repurposed that field: ``degradation.py:1157`` now sets it to
``total_capacity_fade * 100``, so it became a damage measure where higher is
WORSE, roughly 1e-2 instead of ~90. The old term therefore had the wrong sign
and was about four orders of magnitude too small to influence any choice. The
health-driven preference mix was, in effect, a low-cost-sensitivity mix.

Rather than invent a replacement coefficient, wear is now priced:

    wear_j  =  battery_stress_j  x  usable_battery_kwh  x  pack_cost_per_kwh

``battery_stress`` is ``raw_battery_stress`` = ``total_capacity_fade``, the
fraction of pack capacity lost in that session, so the product is rupees of
battery consumed. ``b_wear`` is then in the same units as ``b_cost`` -- utils
per rupee -- and their ratio is the only free quantity: how much of a rupee of
battery wear a customer treats as a rupee out of pocket. That is
``wear_internalisation`` below, and it is an interpretable modelling assumption
rather than a fitted scale.

The resulting term needs no tuning to be comparable with the bill. Measured on
one menu per model: 2.939e-4 x 40 kWh x Rs 7,850 = Rs 92 per session for the
LFP, and 3.875e-4 x 60 kWh x Rs 12,400 = Rs 288 for the NMC, against charging
costs of Rs 81-286 in the same menus.

Attribute scaling
-----------------
* ``cost``   Rs, as charged (menu range is roughly 0-300)
* ``wear``   Rs of battery capacity consumed (see above)
* ``delay``  hours between plug-in and the offer's ready time, in [0, ~14]
* ``soc``    target state of charge as a fraction, in [0, 1]

Coefficient values are illustrative research assumptions. They are not
estimated from revealed- or stated-preference data, and no willingness-to-pay
number derived from them should be quoted as empirical. The pack costs are
sourced (see PACK_COST_RS_PER_KWH) but are a global benchmark, not an Indian
retail replacement quote.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

CHOICE_NOTES = __doc__

# Battery pack cost by chemistry, Rs per kWh of pack capacity.
#
# Source (supplied by the project owner, not independently verified here --
# this sandbox has no web access): BloombergNEF's 2025 chemistry split, $81/kWh
# for LFP and $128/kWh for NMC, converted at about Rs 96.81/USD, giving
# Rs 7,842 and Rs 12,392. Rounded to the figures below. The IEA's 2026 battery
# price dataset uses the same chemistry-level distinction and cites BNEF as its
# source.
#
# CAVEAT for any write-up: these are GLOBAL VOLUME-WEIGHTED PACK PRICES ACROSS
# APPLICATIONS, not Indian EV replacement-pack retail quotes. BNEF separately
# reports BEV packs averaging $99/kWh. Describe them as a chemistry-specific
# pack-cost benchmark. Using them preserves the empirically observed LFP/NMC
# differential instead of inventing separate Indian multipliers.
#
# This is the only exogenous monetary assumption in the choice model. Vary it
# with scaled_pack_costs() for the sensitivity analysis; +/-25% is the
# recommended band.
PACK_COST_RS_PER_KWH: dict[str, float] = {
    "LFP": 7850.0,
    "NMC": 12400.0,
}


def scaled_pack_costs(factor: float) -> dict[str, float]:
    """Pack costs scaled by ``factor``, for the sensitivity sweep."""
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"factor must be positive and finite; got {factor!r}")
    return {k: v * factor for k, v in PACK_COST_RS_PER_KWH.items()}


def wear_cost_rs(stress: float, capacity_kwh: float, chemistry: str,
                 pack_costs: dict[str, float] | None = None) -> float:
    """Rupees of pack capacity consumed by a session of this stress.

    ``stress`` is ``raw_battery_stress`` -- the fraction of usable capacity the
    session costs -- so this is a straight monetary conversion, not a utility.
    """
    costs = PACK_COST_RS_PER_KWH if pack_costs is None else pack_costs
    try:
        per_kwh = costs[chemistry]
    except KeyError:
        raise ValueError(
            f"no pack cost for chemistry {chemistry!r}; known: {sorted(costs)}"
        ) from None
    return float(stress) * float(capacity_kwh) * float(per_kwh)

_ATTRIBUTES = ("cost", "wear", "delay", "soc")


@dataclass(frozen=True)
class PreferenceDistribution:
    """Population distribution of mixed-logit taste coefficients.

    Each coefficient is lognormal: ``b = exp(mu + sigma * z)``, so ``median_*``
    is the population median and ``sigma_*`` the log-scale dispersion.
    ``sigma = 0`` collapses the mixed logit to a plain MNL, which is what the
    fixed-coefficient baseline uses.
    """

    median_cost: float = 0.020      # utils per Rs of charging cost
    # Utils per Rs of battery wear, expressed as a MULTIPLE of median_cost.
    # 1.0 = a rupee of pack capacity consumed feels exactly like a rupee spent
    # on electricity. Below 1 discounts future battery cost against today's
    # bill; above 1 over-weights it. This replaces the old free-floating
    # median_health, whose scale was never examined and turned out to be
    # meaningless once the underlying field changed.
    wear_internalisation: float = 1.00
    median_delay: float = 0.25      # per hour of waiting
    median_soc: float = 4.00        # per unit target SOC (0-1)
    sigma_cost: float = 0.60
    sigma_wear: float = 0.70
    sigma_delay: float = 0.60
    sigma_soc: float = 0.45
    label: str = "balanced"

    @property
    def median_wear(self) -> float:
        """Utils per Rs of wear. Derived, never set directly."""
        return self.median_cost * self.wear_internalisation

    def __post_init__(self) -> None:
        if not np.isfinite(self.wear_internalisation) or self.wear_internalisation <= 0.0:
            raise ValueError(
                f"wear_internalisation must be positive and finite; "
                f"got {self.wear_internalisation!r}"
            )
        for name in _ATTRIBUTES:
            median = getattr(self, f"median_{name}")
            sigma = getattr(self, f"sigma_{name}")
            if not np.isfinite(median) or median <= 0.0:
                raise ValueError(f"median_{name} must be positive and finite; got {median!r}")
            if not np.isfinite(sigma) or sigma < 0.0:
                raise ValueError(f"sigma_{name} must be non-negative and finite; got {sigma!r}")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

    def draw(self, rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
        """Draw ``size`` independent customer taste vectors."""
        if size <= 0:
            raise ValueError("size must be positive")
        out: dict[str, np.ndarray] = {}
        for name in _ATTRIBUTES:
            median = float(getattr(self, f"median_{name}"))
            sigma = float(getattr(self, f"sigma_{name}"))
            mu = np.log(median)
            out[name] = (
                np.exp(mu + sigma * rng.standard_normal(size))
                if sigma > 0
                else np.full(size, median, dtype=float)
            )
        return out

    def as_fixed(self) -> PreferenceDistribution:
        """Return the plain-MNL analogue: same medians, zero dispersion."""
        return replace(
            self,
            sigma_cost=0.0,
            sigma_wear=0.0,
            sigma_delay=0.0,
            sigma_soc=0.0,
            label=f"{self.label}-fixed",
        )


# Scenario dimension: who the customers are.
PREFERENCE_MIXES: dict[str, PreferenceDistribution] = {
    # wear_internalisation is the share of a rupee of battery wear the customer
    # treats as a rupee of money. The old median_health values are NOT carried
    # over: they were utils per unit of a 0-1 "health score" that no longer
    # exists, so there is no conversion. These are fresh assumptions.
    "balanced": PreferenceDistribution(label="balanced"),
    "cost_driven": PreferenceDistribution(
        # Watches the bill; discounts a cost that lands years away.
        median_cost=0.055, wear_internalisation=0.40, median_delay=0.14, median_soc=3.00,
        label="cost_driven",
    ),
    "health_driven": PreferenceDistribution(
        # Treats pack life as worth more than its replacement cost -- range
        # anxiety, resale value, and dislike of the replacement itself.
        median_cost=0.008, wear_internalisation=2.50, median_delay=0.30, median_soc=3.40,
        label="health_driven",
    ),
    "convenience_driven": PreferenceDistribution(
        # Wants the car ready; battery wear barely registers.
        median_cost=0.007, wear_internalisation=0.25, median_delay=1.10, median_soc=5.20,
        label="convenience_driven",
    ),
}


def menu_attribute_matrix(record, pack_costs: dict[str, float] | None = None) -> np.ndarray:
    """Extract the (n_offers, 4) attribute matrix from a MenuRecord.

    Rows follow ``record.choice_set`` -- the displayed offers followed by the
    BAU references -- so an index returned by a choice function indexes that,
    NOT ``record.offers``.

    ``delay`` is measured from plug-in, using the menu's own absolute interval
    clock so that overnight sessions do not wrap. A BAU that needs no charging
    is ready at arrival and therefore has zero delay.
    """
    arrival_abs = record.interval_start_minutes[0]
    offers = record.choice_set
    matrix = np.asarray(
        [
            (
                offer.charging_cost,
                wear_cost_rs(
                    offer.battery_stress,
                    record.usable_battery_kwh,
                    record.chemistry,
                    pack_costs,
                ),
                (offer.ready_minute - arrival_abs) / 60.0,
                offer.target_soc,
            )
            for offer in offers
        ],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("attribute matrix must be (n_offers, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("menu attributes contain non-finite values")
    if np.any(matrix[:, 2] < -1e-9):
        raise ValueError("negative delay: ready time precedes plug-in")
    return matrix


def utilities(attributes: np.ndarray, betas: dict[str, np.ndarray]) -> np.ndarray:
    """Return the (n_customers, n_offers) deterministic utility matrix."""
    cost, wear, delay, soc = (attributes[:, i] for i in range(4))
    return (
        -np.outer(betas["cost"], cost)
        - np.outer(betas["wear"], wear)      # wear is a COST: minus, not plus
        - np.outer(betas["delay"], delay)
        + np.outer(betas["soc"], soc)
    )


def choice_probabilities(values: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("choice probabilities do not sum to one")
    return probabilities


def sample_choices(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one offer index per customer via inverse-CDF on each row."""
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0  # guard against float drift at the top of the CDF
    draws = rng.random((probabilities.shape[0], 1))
    return (draws > cumulative).sum(axis=1).astype(int)
