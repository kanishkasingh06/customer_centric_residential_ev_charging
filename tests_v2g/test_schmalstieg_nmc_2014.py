"""The NMC coefficients, checked against Schmalstieg et al. (2014).

J. Schmalstieg, S. Kaebitz, M. Ecker, D. U. Sauer, "A holistic aging
model for Li(NiMnCo)O2 based 18650 lithium-ion batteries", J. Power
Sources 257, 325-334, doi:10.1016/j.jpowsour.2014.02.012.

As with the LFP tests, each assertion restates a number printed in the
paper so the citation cannot drift away from the code.  The tests also
pin what is NOT theirs, so a later reader does not over-attribute.
"""

from __future__ import annotations

from math import exp

import pytest

from evmenu.degradation import _GAS_CONSTANT_J_PER_MOL_K as R
from evmenu.degradation import (
    DEFAULT_NMC_PARAMETERS,
    SCHMALSTIEG_NMC_2014_PARAMETERS,
)

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


def test_depth_coefficient_is_the_ratio_printed_in_equation_twenty() -> None:
    """b_cap = 7.348e-3(Vbar-3.667)^2 + 7.600e-4 + 4.081e-3*dDOD.

    The depth-to-baseline ratio is what this package's multiplicative
    ``(1 + c_D * delta)`` has to reproduce, and it is invariant to whether
    their charge throughput counts one or both directions.
    """
    assert PARAMS.cycle_dod_coefficient == pytest.approx(4.081e-3 / 7.600e-4, rel=1e-4)
    assert PARAMS.cycle_dod_coefficient == pytest.approx(5.3697, abs=1e-4)


def test_cycle_throughput_exponent_is_their_square_root() -> None:
    """Eq. (18): capacity loss goes as sqrt(charge throughput)."""
    assert PARAMS.cycle_time_exponent == 0.50


def test_c_rate_term_is_zero_because_the_study_cannot_support_one() -> None:
    """Every cycle test ran at 1C and 35 C (Section 4), so there is no
    C-rate and no cycle-temperature dependence to take.  Zero is the
    honest default; the shipped surrogate used 1.0, which is invented."""
    assert PARAMS.cycle_c_rate_coefficient == 0.0
    assert DEFAULT_NMC_PARAMETERS.cycle_c_rate_coefficient == 1.0


def test_calendar_soc_shape_is_still_the_unattributed_surrogate() -> None:
    """Schmalstieg parameterises calendar ageing on VOLTAGE, linearly.

    Converting to state of charge needs the OCV curve of their Fig. 1,
    which the paper plots but does not tabulate.  Until it is digitised
    the hinge shape stays, and stays unattributed; this test records that
    so nobody later reads the parameter-set name as covering it.
    """
    assert PARAMS.calendar_a0 == DEFAULT_NMC_PARAMETERS.calendar_a0
    assert PARAMS.calendar_a1 == DEFAULT_NMC_PARAMETERS.calendar_a1
    assert PARAMS.calendar_a2 == DEFAULT_NMC_PARAMETERS.calendar_a2
    assert PARAMS.calendar_soc_knee == DEFAULT_NMC_PARAMETERS.calendar_soc_knee


def test_shipped_default_is_untouched() -> None:
    assert DEFAULT_NMC_PARAMETERS.activation_energy_j_per_mol == 24000.0
    assert DEFAULT_NMC_PARAMETERS.cycle_dod_coefficient == 1.0
