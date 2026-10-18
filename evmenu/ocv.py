"""Open-circuit voltage curves, for parameter sets defined on cell voltage.

Some published ageing models are parameterised on cell voltage rather
than state of charge.  Schmalstieg et al. (2014) is one: their calendar
term is linear in V, and their cycle term carries a quadratic in the
mean voltage of the cycle.  Using those models from a state-of-charge
trajectory needs the cell's OCV curve, so it lives here rather than
being folded into a coefficient.
"""

from __future__ import annotations

from .exceptions import PhysicalConstraintError

__all__ = ["SANYO_UR18650E_OCV", "interpolate_ocv"]

#: Sanyo UR18650E open-circuit voltage against state of charge, digitised
#: at 1% intervals from Fig. 1 of Schmalstieg et al. (2014), measured at
#: 35 C.  Two points (SOC 34 and 37) dipped by under a millivolt against
#: their predecessor -- digitisation jitter on a flat part of the curve --
#: and were raised by 1e-4 V to keep the curve strictly increasing, since
#: a non-monotone map puts spurious local minima into anything that
#: differentiates through it.
#:
#: Validation: the paper states two cell voltages in its text, 3.699 V at
#: 50% SOC and 4.162 V at 100%.  This curve gives 3.7000 and 4.1596, which
#: is +0.18% and -0.24% in their calendar ageing factor.  Both are
#: asserted in the test suite.
SANYO_UR18650E_OCV: tuple[tuple[int, float], ...] = (
    (0, 3.3346), (1, 3.35), (2, 3.3692), (3, 3.3885),
    (4, 3.4077), (5, 3.4231), (6, 3.4385), (7, 3.4538),
    (8, 3.4692), (9, 3.4808), (10, 3.4962), (11, 3.5038),
    (12, 3.5154), (13, 3.5269), (14, 3.5346), (15, 3.5462),
    (16, 3.5538), (17, 3.5615), (18, 3.5692), (19, 3.5769),
    (20, 3.5846), (21, 3.5885), (22, 3.5962), (23, 3.6),
    (24, 3.6058), (25, 3.6096), (26, 3.6154), (27, 3.6192),
    (28, 3.6231), (29, 3.625), (30, 3.6269), (31, 3.6308),
    (32, 3.6346), (33, 3.6385), (34, 3.6386), (35, 3.6423),
    (36, 3.6462), (37, 3.6463), (38, 3.65), (39, 3.6538),
    (40, 3.6577), (41, 3.6596), (42, 3.6615), (43, 3.6654),
    (44, 3.6692), (45, 3.6731), (46, 3.6769), (47, 3.6827),
    (48, 3.6885), (49, 3.6923), (50, 3.7), (51, 3.7038),
    (52, 3.7115), (53, 3.7192), (54, 3.7231), (55, 3.7308),
    (56, 3.7385), (57, 3.75), (58, 3.7577), (59, 3.7654),
    (60, 3.7769), (61, 3.7846), (62, 3.7923), (63, 3.8038),
    (64, 3.8115), (65, 3.8231), (66, 3.8308), (67, 3.8423),
    (68, 3.85), (69, 3.8615), (70, 3.8692), (71, 3.8769),
    (72, 3.8885), (73, 3.8962), (74, 3.9077), (75, 3.9154),
    (76, 3.9231), (77, 3.9346), (78, 3.9423), (79, 3.9538),
    (80, 3.9654), (81, 3.975), (82, 3.9846), (83, 3.9962),
    (84, 4.0077), (85, 4.0192), (86, 4.0308), (87, 4.0423),
    (88, 4.0538), (89, 4.0615), (90, 4.0731), (91, 4.0808),
    (92, 4.0923), (93, 4.1), (94, 4.1077), (95, 4.1154),
    (96, 4.1269), (97, 4.1346), (98, 4.1423), (99, 4.1538),
    (100, 4.1596),)


def interpolate_ocv(
    soc: float, curve: tuple[tuple[int, float], ...] = SANYO_UR18650E_OCV
) -> float:
    """Open-circuit voltage at a fractional state of charge, by linear
    interpolation on a strictly increasing 0-100% curve."""
    if not curve:
        raise PhysicalConstraintError("OCV curve must not be empty.")
    if not 0.0 <= soc <= 1.0:
        raise PhysicalConstraintError("soc must lie in [0, 1].")
    position = soc * (len(curve) - 1)
    lower = int(position)
    if lower >= len(curve) - 1:
        return curve[-1][1]
    weight = position - lower
    return curve[lower][1] * (1.0 - weight) + curve[lower + 1][1] * weight
