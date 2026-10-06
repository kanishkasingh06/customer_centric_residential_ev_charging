"""Mixed-logit customer choice over a generated EV charging menu.

Model
-----
Customer ``n`` facing menu offers ``j`` draws a personal preference vector and
picks the offer maximising random utility

    U_nj = - b_cost,n  * cost_j
           + b_health,n * health_j
           - b_delay,n  * delay_j
           + b_soc,n    * soc_j
           + eps_nj,        eps ~ iid Gumbel(0, 1)

which gives the conditional-logit probability ``softmax(V_n)`` for that draw.
Averaging over the population distribution of ``b`` makes this a mixed logit:
choice diversity comes from preference heterogeneity, not just the Gumbel noise.

All four coefficients are drawn from lognormal distributions so signs are
guaranteed -- nobody in the population prefers paying more, waiting longer, or
degrading their battery faster.

Attribute scaling (deliberate, see ``CHOICE_NOTES``)
---------------------------------------------------
* ``cost``   Rs, as charged (menu range is roughly 0-300)
* ``health`` menu health score / 100, in [0, 1]
* ``delay``  hours between plug-in and the offer's ready time, in [0, ~14]
* ``soc``    target state of charge as a fraction, in [0, 1]

Default medians are chosen so each term spans a comparable utility range over a
typical menu (about 2-5 utils). If one term's range dwarfed the others the
model would degenerate into a single-attribute rule and the scenario sweep
would be measuring the scaling, not the preferences.

Coefficient values are illustrative research assumptions. They are not
estimated from revealed- or stated-preference data, and no willingness-to-pay
number derived from them should be quoted as empirical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

CHOICE_NOTES = __doc__

_ATTRIBUTES = ("cost", "health", "delay", "soc")


@dataclass(frozen=True)
class PreferenceDistribution:
    """Population distribution of mixed-logit taste coefficients.

    Each coefficient is lognormal: ``b = exp(mu + sigma * z)``, so ``median_*``
    is the population median and ``sigma_*`` the log-scale dispersion.
    ``sigma = 0`` collapses the mixed logit to a plain MNL, which is what the
    fixed-coefficient baseline uses.
    """

    median_cost: float = 0.020      # per Rs
    median_health: float = 2.00     # per unit health (0-1)
    median_delay: float = 0.25      # per hour of waiting
    median_soc: float = 4.00        # per unit target SOC (0-1)
    sigma_cost: float = 0.60
    sigma_health: float = 0.70
    sigma_delay: float = 0.60
    sigma_soc: float = 0.45
    label: str = "balanced"

    def __post_init__(self) -> None:
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
            sigma_health=0.0,
            sigma_delay=0.0,
            sigma_soc=0.0,
            label=f"{self.label}-fixed",
        )


# Scenario dimension: who the customers are.
PREFERENCE_MIXES: dict[str, PreferenceDistribution] = {
    "balanced": PreferenceDistribution(label="balanced"),
    "cost_driven": PreferenceDistribution(
        median_cost=0.055, median_health=0.70, median_delay=0.14, median_soc=3.00,
        label="cost_driven",
    ),
    "health_driven": PreferenceDistribution(
        median_cost=0.008, median_health=6.00, median_delay=0.30, median_soc=3.40,
        label="health_driven",
    ),
    "convenience_driven": PreferenceDistribution(
        median_cost=0.007, median_health=1.20, median_delay=1.10, median_soc=5.20,
        label="convenience_driven",
    ),
}


def menu_attribute_matrix(record) -> np.ndarray:
    """Extract the (n_offers, 4) attribute matrix from a MenuRecord.

    ``delay`` is measured from plug-in, using the menu's own absolute interval
    clock so that overnight sessions do not wrap. A BAU that needs no charging
    is ready at arrival and therefore has zero delay.
    """
    arrival_abs = record.interval_start_minutes[0]
    matrix = np.asarray(
        [
            (
                offer.charging_cost,
                offer.health_score / 100.0,
                (offer.ready_minute - arrival_abs) / 60.0,
                offer.target_soc,
            )
            for offer in record.offers
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
    cost, health, delay, soc = (attributes[:, i] for i in range(4))
    return (
        -np.outer(betas["cost"], cost)
        + np.outer(betas["health"], health)
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
