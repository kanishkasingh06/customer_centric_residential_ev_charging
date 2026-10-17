"""Bidirectional (V2G) extension of the residential EV customer-menu generator.

Implements the frozen formulation Rev 4.2.  Built as a separate package so
that ``evmenu`` and its test suite are untouched, which is what makes the
``E_out == 0`` regression a genuine agreement test between two independent
code paths rather than a tautology.
"""

from .api import (
    V2GMenu,
    build_session_and_signal,
    default_tou_profile,
    format_menu,
    generate_ev_menu_v2g,
)
from .envelope import V1G_ENVELOPE, V2GEnvelope
from .fade import (
    THROUGHPUT_FRACTIONS_PER_FULL_CYCLE,
    BidirectionalFade,
    assess_trajectory,
    calendar_parked_fade,
    calendar_window_fade,
    cycle_fade,
    peak_c_rate,
    soc_excursion,
    throughput_fraction,
)
from .menu import (
    DEFAULT_BAND_ABSOLUTE_FLOOR,
    V2GOffer,
    bau_cost_for,
    candidate_ready_steps,
    candidate_targets,
    compact,
    dominates,
    generate_offers,
    is_realizable,
    pareto_filter,
    saving_levels,
    service_minimum_soc,
)
from .trajectory import (
    RequestContext,
    Trajectory,
    battery_states,
    capacities,
    evaluate,
    require_feasible,
    solve,
    solve_cost_linear,
    warm_start,
)

__all__ = [
    "DEFAULT_BAND_ABSOLUTE_FLOOR",
    "THROUGHPUT_FRACTIONS_PER_FULL_CYCLE",
    "V1G_ENVELOPE",
    "BidirectionalFade",
    "RequestContext",
    "Trajectory",
    "V2GEnvelope",
    "V2GMenu",
    "V2GOffer",
    "assess_trajectory",
    "battery_states",
    "bau_cost_for",
    "build_session_and_signal",
    "calendar_parked_fade",
    "calendar_window_fade",
    "candidate_ready_steps",
    "candidate_targets",
    "capacities",
    "compact",
    "cycle_fade",
    "default_tou_profile",
    "dominates",
    "evaluate",
    "format_menu",
    "generate_ev_menu_v2g",
    "generate_offers",
    "is_realizable",
    "pareto_filter",
    "peak_c_rate",
    "require_feasible",
    "saving_levels",
    "service_minimum_soc",
    "soc_excursion",
    "solve",
    "solve_cost_linear",
    "throughput_fraction",
    "warm_start",
]
