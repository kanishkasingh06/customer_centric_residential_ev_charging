"""The NMC coefficients, checked against Schmalstieg et al. (2014).

J. Schmalstieg, S. Kaebitz, M. Ecker, D. U. Sauer, "A holistic aging
model for Li(NiMnCo)O2 based 18650 lithium-ion batteries", J. Power
Sources 257, 325-334, doi:10.1016/j.jpowsour.2014.02.012.

As with the LFP tests, each assertion restates a number printed in the
paper so the citation cannot drift away from the code.  The tests also
pin what is NOT theirs, so a later reader does not over-attribute.
"""

from __future__ import annotations

from itertools import pairwise
from math import exp, sqrt

import pytest

from evmenu.degradation import _GAS_CONSTANT_J_PER_MOL_K as R
from evmenu.degradation import (
    DEFAULT_NMC_PARAMETERS,
    SCHMALSTIEG_NMC_2014_PARAMETERS,
    calendar_soc_stress,
)
from evmenu.ocv import SANYO_UR18650E_OCV, interpolate_ocv

PARAMS = SCHMALSTIEG_NMC_2014_PARAMETERS


def test_activation_energy_is_their_arrhenius_exponent() -> None:
    """Eq. (14) carries exp(-6976/T); Section 3.4 states 58.0 kJ/mol.

    Stored so that ``Ea / R`` reproduces their exponent under THIS
    package's gas constant.  The paper rounds R to 8.314, so matching
    their printed 58.0 kJ/mol and matching their exponent are not quite
    the same thing; the exponent is what actually gets applied.
    """
    assert PARAMS.activation_energy_j_per_mol / R == pytest.approx(6976.0, rel=1e-12)
    assert PARAMS.activation_energy_j_per_mol / 1000.0 == pytest.approx(58.0, abs=0.05)


def test_temperature_sensitivity_matches_equation_fourteen() -> None:
    """Our Arrhenius factor must track theirs over the tested 35-50 C range."""
    reference = PARAMS.calendar_reference_temperature_c
    assert reference is not None
    for temperature_c in (0.0, 25.0, 35.0, 40.0, 50.0):
        theirs = exp(-6976.0 / (temperature_c + 273.15)) / exp(-6976.0 / (reference + 273.15))
        ours = exp(
            -PARAMS.activation_energy_j_per_mol
            / R
            * (1.0 / (temperature_c + 273.15) - 1.0 / (reference + 273.15))
        )
        assert ours == pytest.approx(theirs, rel=1e-9)


def test_calendar_time_exponent_is_their_choice_of_t_to_the_three_quarters() -> None:
    """Section 3.2: sqrt(t) fitted badly, so t^0.75 was adopted."""
    assert PARAMS.calendar_time_exponent == 0.75


def test_cycle_terms_are_equation_twenty_scaled_per_full_cycle() -> None:
    """b_cap = 7.348e-3(Vbar-3.667)^2 + 7.600e-4 + 4.081e-3*dDOD.

    Eq. (20) applies to sqrt(charge throughput in Ah); matching cumulative
    fade per equivalent full cycle for their 2.05 Ah cell scales every
    term by sqrt(2*2.05)/2.  The depth form is ADDITIVE, as they fit it.
    """
    scale = sqrt(2 * 2.05) / 2.0
    assert PARAMS.cycle_depth_is_additive is True
    assert PARAMS.cycle_reference_coefficient == pytest.approx(scale * 7.600e-4, rel=1e-9)
    assert PARAMS.cycle_dod_coefficient == pytest.approx(scale * 4.081e-3, rel=1e-9)
    assert PARAMS.cycle_mean_voltage_coefficient == pytest.approx(scale * 7.348e-3, rel=1e-9)
    assert PARAMS.cycle_mean_voltage_center == 3.667
    # the depth-to-baseline ratio is what the paper pins, and it survives
    # the scaling whatever the throughput convention
    assert PARAMS.cycle_dod_coefficient / PARAMS.cycle_reference_coefficient == pytest.approx(
        4.081e-3 / 7.600e-4, rel=1e-9
    )


def test_cycle_throughput_exponent_is_their_square_root() -> None:
    """Eq. (18): capacity loss goes as sqrt(charge throughput)."""
    assert PARAMS.cycle_time_exponent == 0.50


def test_c_rate_term_is_zero_because_the_study_cannot_support_one() -> None:
    """Every cycle test ran at 1C and 35 C (Section 4), so there is no
    C-rate and no cycle-temperature dependence to take.  Zero is the
    honest default; the shipped surrogate used 1.0, which is invented."""
    assert PARAMS.cycle_c_rate_coefficient == 0.0
    assert DEFAULT_NMC_PARAMETERS.cycle_c_rate_coefficient == 1.0


def test_calendar_is_now_in_voltage_space_as_published() -> None:
    """Eq. (14) is linear in cell voltage; the SOC polynomial is unused."""
    assert PARAMS.calendar_voltage_scale > 0.0
    assert PARAMS.calendar_voltage_slope == 7.543
    assert PARAMS.calendar_voltage_intercept == -23.75
    assert PARAMS.ocv_curve == SANYO_UR18650E_OCV
    assert (PARAMS.calendar_a0, PARAMS.calendar_a1, PARAMS.calendar_a2) == (0.0, 0.0, 0.0)


def test_ocv_curve_hits_both_voltages_the_paper_states() -> None:
    """3.699 V at 50% SOC and 4.162 V at 100%, from their Section 3."""
    assert interpolate_ocv(0.50) == pytest.approx(3.699, abs=0.005)
    assert interpolate_ocv(1.00) == pytest.approx(4.162, abs=0.005)


def test_ocv_curve_is_strictly_increasing_and_clears_the_zero_crossing() -> None:
    """a_cap goes negative below V = 23.75/7.543 = 3.1486 V.

    The curve bottoms out at 3.3346 V, so the whole 0-100% range is
    usable with no floor.  A re-digitisation that dipped under the
    crossing would make calendar ageing negative, so it is asserted.
    """
    voltages = [v for _, v in SANYO_UR18650E_OCV]
    assert all(b > a for a, b in pairwise(voltages))
    assert min(voltages) > 23.75 / 7.543


def test_reproduces_equation_fourteen_at_the_reference_point() -> None:
    """One year at 25 C and 50% SOC, against Eq. (14) evaluated directly."""
    theirs = (7.543 * 3.699 - 23.75) * 1e6 * exp(-6976.0 / 298.15) * 365 ** 0.75
    ours = calendar_soc_stress(0.50, PARAMS) / 0.75   # integral of g t^(a-1), a = 0.75
    assert ours == pytest.approx(theirs, rel=0.005)


def test_shipped_default_is_untouched() -> None:
    assert DEFAULT_NMC_PARAMETERS.activation_energy_j_per_mol == 24000.0
    assert DEFAULT_NMC_PARAMETERS.cycle_dod_coefficient == 1.0
