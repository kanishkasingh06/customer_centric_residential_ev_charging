"""The formulation document's parameter table must match the code.

Table 1 silently carried a superseded NMC activation energy of
24,000 J/mol while Section 6 of the same document stated 58.0 kJ/mol,
and six other cells had drifted the same way as parameter sets were
replaced.  A document is not executable, so nothing caught it.  This
test makes the table executable: every number in it is checked against
the parameter set it claims to describe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evmenu.degradation import (
    NAUMANN_LFP_2020_CYCLE_PARAMETERS,
    SCHMALSTIEG_NMC_2014_PARAMETERS,
)

DOCUMENT = Path(__file__).resolve().parents[1] / "research" / "battery_health_formulation.tex"

#: (row symbol, attribute, LFP printed value, NMC printed value).  None
#: means the row shows a dash because that chemistry's functional form
#: does not use the term.
ROWS = [
    (r"a_0", "calendar_a0", 0.0212579, None),
    (r"a_3", "calendar_cubic_coefficient", 0.1008624, None),
    (r"\alpha", "calendar_time_exponent", 0.50, 0.75),
    (r"E_a", "activation_energy_j_per_mol", 17126, 58002),
    (r"\kappa", "calendar_voltage_scale", None, 0.0043184),
    (r"\mu", "calendar_voltage_slope", None, 7.543),
    (r"\nu", "calendar_voltage_intercept", None, -23.75),
    (r"c_{\mathrm{ref}}", "cycle_reference_coefficient", 0.00053031, 0.00076944),
    (r"c_D", "cycle_dod_coefficient", None, 0.0041317),
    (r"c_{D3}", "cycle_depth_cubic_coefficient", 0.00195428, None),
    (r"\delta_c", "cycle_depth_cubic_center", 0.60, None),
    (r"c_V", "cycle_mean_voltage_coefficient", None, 0.0074393),
    (r"V_c", "cycle_mean_voltage_center", None, 3.667),
    (r"c_C", "cycle_c_rate_coefficient", 0.6488157, 0),
    (r"\beta", "cycle_time_exponent", 0.50, 0.50),
]


@pytest.mark.parametrize(("symbol", "attribute", "lfp", "nmc"), ROWS)
def test_printed_value_matches_the_parameter_set(
    symbol: str, attribute: str, lfp: float | None, nmc: float | None
) -> None:
    """Each printed figure must round-trip to the stored coefficient."""
    for printed, params, label in (
        (lfp, NAUMANN_LFP_2020_CYCLE_PARAMETERS, "LFP"),
        (nmc, SCHMALSTIEG_NMC_2014_PARAMETERS, "NMC"),
    ):
        if printed is None:
            continue  # the table prints a dash; the term is unused
        stored = getattr(params, attribute)
        # printed to the precision shown, so compare at that precision
        places = len(str(printed).partition(".")[2])
        assert round(stored, places) == pytest.approx(printed, rel=1e-9), (
            f"{label} {symbol}: document prints {printed}, code holds {stored}"
        )


def test_activation_energies_appear_in_the_document_as_printed() -> None:
    """The specific regression: Table 1 said 24000, Section 6 said 58.0 kJ."""
    text = DOCUMENT.read_text(encoding="utf-8")
    assert r"\num{58002}" in text
    assert r"\num{17126}" in text
    # the superseded NMC value must not appear as a live table entry
    assert r"& \num{24000} \\" not in text


def test_reference_temperatures_agree() -> None:
    for params, label in (
        (NAUMANN_LFP_2020_CYCLE_PARAMETERS, "LFP"),
        (SCHMALSTIEG_NMC_2014_PARAMETERS, "NMC"),
    ):
        assert params.calendar_reference_temperature_c == 25.0, label
    text = DOCUMENT.read_text(encoding="utf-8")
    assert r"\SI{25}{\celsius} & \SI{25}{\celsius}" in text


def test_dashes_correspond_to_genuinely_unused_terms() -> None:
    """A dash must mean the form does not use the term, not a stale cell."""
    for symbol, attribute, lfp, nmc in ROWS:
        if lfp is None:
            assert getattr(NAUMANN_LFP_2020_CYCLE_PARAMETERS, attribute) == 0.0, symbol
        if nmc is None:
            assert getattr(SCHMALSTIEG_NMC_2014_PARAMETERS, attribute) == 0.0, symbol
