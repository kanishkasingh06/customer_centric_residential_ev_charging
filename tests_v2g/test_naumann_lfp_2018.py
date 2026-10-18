"""The published LFP calendar model, checked against the paper itself.

Naumann et al. (2018), "Analysis and modeling of calendar aging of a
commercial LiFePO4/graphite cell", J. Energy Storage 17, 153-169,
doi:10.1016/j.est.2018.01.019.

These tests exist so the parameter set can be cited rather than defended.
Each one restates a number that appears in the paper and checks the
implementation reproduces it; if a coefficient is edited, the citation
stops being true and a test fails in the same commit.
"""

from __future__ import annotations

from itertools import pairwise
from math import exp, sqrt

import pytest

from evmenu.degradation import (
    NAUMANN_LFP_2018_PARAMETERS,
    DegradationSettings,
    calendar_soc_stress,
)

R = 8.314  # the paper's own rounding of the gas constant, its Section 3.2.4
SECONDS_PER_YEAR = 8760 * 3600


def _published_qloss_percent(temperature_c: float, soc: float, years: float) -> float:
    """Naumann Eq. (3) with Table 7, evaluated directly from the paper."""
    k_ref, activation, c, d, t_ref_k = 0.0012571, 17126.0, 2.8575, 0.60225, 298.15
    k_temp = k_ref * exp(-activation / R * (1.0 / (temperature_c + 273.15) - 1.0 / t_ref_k))
    k_soc = c * (soc - 0.5) ** 3 + d
    return k_temp * k_soc * sqrt(years * SECONDS_PER_YEAR)


def _implemented_qloss_percent(temperature_c: float, soc: float, years: float) -> float:
    """The same quantity through this package's differential form."""
    params = NAUMANN_LFP_2018_PARAMETERS
    reference = params.calendar_reference_temperature_c
    assert reference is not None
    arrhenius = exp(
        -params.activation_energy_j_per_mol
        / R
        * (1.0 / (temperature_c + 273.15) - 1.0 / (reference + 273.15))
    )
    # integral of g * t^(alpha-1) dt from 0 to `years`, alpha = 1/2
    return calendar_soc_stress(soc, params) * arrhenius * 2.0 * sqrt(years) * 100.0


def test_reproduces_the_papers_own_twenty_year_projection() -> None:
    """Section 4: "about 19.0% decrease ... at 50% SOC and 25 C" over 20 years."""
    assert _implemented_qloss_percent(25.0, 0.50, 20.0) == pytest.approx(19.0, abs=0.05)


@pytest.mark.parametrize("temperature_c", [0.0, 25.0, 40.0, 60.0])
@pytest.mark.parametrize("soc", [0.0, 0.25, 0.50, 0.875, 1.0])
@pytest.mark.parametrize("years", [0.5, 2.0, 20.0])
def test_matches_equation_three_everywhere(
    temperature_c: float, soc: float, years: float
) -> None:
    """The differential form must agree with the paper's closed form."""
    assert _implemented_qloss_percent(temperature_c, soc, years) == pytest.approx(
        _published_qloss_percent(temperature_c, soc, years), rel=1e-6
    )


def test_table_seven_constants_are_what_the_paper_prints() -> None:
    params = NAUMANN_LFP_2018_PARAMETERS
    assert params.activation_energy_j_per_mol == 17126.0
    assert params.calendar_reference_temperature_c == 25.0
    assert params.calendar_time_exponent == 0.50      # sqrt(t), Eq. (3)
    assert params.calendar_cubic_center == 0.5        # cubic centred on mid-SOC
    assert params.calendar_a2 == 0.0                  # no knee in the published form


def test_soc_sensitivity_is_the_published_factor_not_a_hinge() -> None:
    """Empty-to-full spread is 3.92x in the paper, not the ~1.2x of a knee.

    The shipped default uses a quadratic hinge at SOC 0.70, which is flat
    beneath the knee.  Naumann's cubic is still rising there.  This is the
    difference that decides whether exporting an LFP pack below 70% has a
    calendar benefit, so it is asserted rather than left to inspection.
    """
    params = NAUMANN_LFP_2018_PARAMETERS
    full, empty = calendar_soc_stress(1.0, params), calendar_soc_stress(0.0, params)
    assert full / empty == pytest.approx(3.9151, abs=0.001)
    # strictly increasing, with no flat region
    grid = [i / 20 for i in range(21)]
    values = [calendar_soc_stress(s, params) for s in grid]
    assert all(b > a for a, b in pairwise(values))


def test_shipped_default_is_untouched() -> None:
    """Adopting the published set must not silently move the V1G baseline."""
    default = DegradationSettings().parameters_for("LFP")
    assert default.calendar_cubic_coefficient == 0.0
    assert default.calendar_reference_temperature_c is None
    assert calendar_soc_stress(0.5, default) == pytest.approx(0.01)
