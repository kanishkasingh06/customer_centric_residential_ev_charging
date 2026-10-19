"""The hardcoded tariff statistics must track the profile they describe.

``select_menu`` hardcodes the min, mean and max of
``default_tou_profile(30)`` because its imports sit below its constants,
and nothing otherwise binds the three numbers to the profile.  If the
profile changes they drift silently, and they set the band and merge
tolerances -- which is the quantity a review found had discarded
nine-tenths of the Pareto front when it was wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import select_menu as S

from evmenu_v2g.api import default_tou_profile


def test_hardcoded_profile_statistics_match_the_profile() -> None:
    profile = [float(x) for x in default_tou_profile(30)]
    assert S._RUPEE_TARIFF_MIN == pytest.approx(min(profile), abs=5e-4)
    assert S._RUPEE_TARIFF_MAX == pytest.approx(max(profile), abs=5e-4)
    assert S._RUPEE_TARIFF_MEAN == pytest.approx(
        sum(profile) / len(profile), abs=5e-4
    )


def test_mapped_mean_matches_the_mapped_profile() -> None:
    """The tolerance base must equal the mean of the tariff actually used."""
    mapped = S._tariff_usd()
    assert S._MAPPED_TARIFF_MEAN == pytest.approx(
        sum(mapped) / len(mapped), abs=5e-4
    )


def test_tariff_endpoints_are_honoured() -> None:
    mapped = S._tariff_usd()
    assert min(mapped) == pytest.approx(S.TARIFF_MIN_USD_PER_KWH)
    assert max(mapped) == pytest.approx(S.TARIFF_MAX_USD_PER_KWH)


def test_compact_requires_an_explicit_tolerance() -> None:
    """No default: an absolute currency amount cannot have a safe one.

    A 5.0 default against savings spanning 0 to 4.3 put the whole menu
    in one bucket and discarded most of the Pareto front.
    """
    import inspect

    from evmenu_v2g.menu import compact

    parameter = inspect.signature(compact).parameters["merge_tolerance"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
