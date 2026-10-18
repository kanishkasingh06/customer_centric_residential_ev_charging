"""LFP cycle ageing, checked against Naumann, Spingler and Jossen (2020).

J. Power Sources 451, 227666, doi:10.1016/j.jpowsour.2019.227666 -- the
companion cycle study to the 2018 calendar paper, same cell.  With it
the LFP branch is a reproduction end to end.
"""

from __future__ import annotations

from math import sqrt

import pytest

from evmenu.catalog import get_ev_model
from evmenu.degradation import (
    DEFAULT_LFP_PARAMETERS,
    NAUMANN_LFP_2020_CYCLE_PARAMETERS,
    DegradationSettings,
)
from evmenu.schemas import EVSpec
from evmenu_v2g.fade import cycle_fade

PARAMS = NAUMANN_LFP_2020_CYCLE_PARAMETERS
SCALE = 0.0971 / 200.0


def _published_qloss_percent(c_rate: float, doc: float, fec: float) -> float:
    """Their Eq. (5) with Eqs. (8), (10) and Table 6, k_T = 1."""
    k_c_rate = 0.0630 * c_rate + 0.0971
    k_doc = 4.0253 * (doc - 0.6) ** 3 + 1.0923
    return k_c_rate * k_doc * sqrt(fec)


def test_table_six_coefficients() -> None:
    assert PARAMS.cycle_depth_is_additive is True
    assert PARAMS.cycle_time_exponent == 0.50                       # z = 0.5, Eq. (5)
    assert PARAMS.cycle_depth_cubic_center == 0.6                   # Eq. (10)
    assert PARAMS.cycle_reference_coefficient == pytest.approx(SCALE * 1.0923, rel=1e-9)
    assert PARAMS.cycle_depth_cubic_coefficient == pytest.approx(SCALE * 4.0253, rel=1e-9)
    assert PARAMS.cycle_dod_coefficient == 0.0                      # cubic, not linear


def test_c_rate_coefficient_is_now_sourced() -> None:
    """Eq. (8) k_Crate = a*C + b, so normalising on b gives c_C = a/b.

    This is the coefficient that had no empirical support anywhere
    before this paper; Schmalstieg ran every NMC cycle test at 1C, so
    the NMC branch still has none.
    """
    assert PARAMS.cycle_c_rate_coefficient == pytest.approx(0.0630 / 0.0971, rel=1e-9)
    assert DEFAULT_LFP_PARAMETERS.cycle_c_rate_coefficient == 1.0   # the old invention


@pytest.mark.parametrize("c_rate", [0.0, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("doc", [0.0, 0.2, 0.6, 0.8, 1.0])
@pytest.mark.parametrize("fec", [100.0, 1000.0, 10600.0])
def test_cumulative_fade_matches_equation_five(
    c_rate: float, doc: float, fec: float
) -> None:
    """Cumulative fade per FEC must reproduce their closed form.

    Integrating this package's incremental cycle term over N cycles at
    fixed conditions gives Gamma * (1 + c_C * C) * 2 * sqrt(N).
    """
    model = get_ev_model("generic_40kwh_lfp")
    ev = EVSpec(
        ev_id="lfp", battery_capacity_kwh=model.usable_battery_kwh,
        minimum_energy_kwh=model.minimum_soc * model.usable_battery_kwh,
        charger_power_kw=model.onboard_ac_power_kw,
        charging_efficiency=model.charging_efficiency, chemistry="LFP",
    )
    settings = DegradationSettings(lfp=PARAMS, cumulative_equivalent_full_cycles=fec)
    # incremental form at N = fec, integrated: Gamma (1 + c_C C) 2 sqrt(N)
    increment = cycle_fade(
        ev=ev, settings=settings, phi=2.0, delta=doc, chat=c_rate, mean_soc=None
    )
    # increment = Gamma (1 + c_C C) * sigma(N) * phi, so recover the
    # coefficient and integrate it: cumulative = coefficient * 2 sqrt(N).
    sigma = 0.5 * fec ** -0.5
    coefficient = increment / (sigma * 2.0)
    cumulative = coefficient * 2.0 * sqrt(fec) * 100.0
    assert cumulative == pytest.approx(
        _published_qloss_percent(c_rate, doc, fec), rel=1e-6
    )


def test_no_cycle_temperature_term() -> None:
    """Section 3.2.1: temperature influence on PURE cycle ageing between
    25 and 40 C was negligible, so their model carries none."""
    model = get_ev_model("generic_40kwh_lfp")
    ev = EVSpec(
        ev_id="lfp", battery_capacity_kwh=model.usable_battery_kwh,
        minimum_energy_kwh=model.minimum_soc * model.usable_battery_kwh,
        charger_power_kw=model.onboard_ac_power_kw,
        charging_efficiency=model.charging_efficiency, chemistry="LFP",
    )
    values = {
        cycle_fade(
            ev=ev,
            settings=DegradationSettings(
                lfp=PARAMS, cumulative_equivalent_full_cycles=500.0,
                fallback_temperature_c=t,
            ),
            phi=1.0, delta=0.5, chat=0.3,
        )
        for t in (0.0, 25.0, 40.0, 60.0)
    }
    assert len(values) == 1


def test_depth_factor_is_cubic_and_stays_positive() -> None:
    """Their Fig. 7a: flat through mid-depth, rising at both ends."""
    gamma = lambda d: (
        PARAMS.cycle_reference_coefficient
        + PARAMS.cycle_depth_cubic_coefficient * (d - 0.6) ** 3
    )
    assert all(gamma(d / 20) > 0.0 for d in range(21))
    assert gamma(1.0) / gamma(0.0) == pytest.approx(6.06, abs=0.02)


def test_shipped_default_is_untouched() -> None:
    assert DEFAULT_LFP_PARAMETERS.cycle_depth_cubic_coefficient == 0.0
    assert DEFAULT_LFP_PARAMETERS.cycle_depth_is_additive is False
