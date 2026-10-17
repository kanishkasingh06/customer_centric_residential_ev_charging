"""Bidirectional incremental capacity fade.

Formulation reference: Rev 4.2 Block 4, eq. (46)-(55).

    dQ_tot = dQ_cal_window + dQ_cal_parked + dQ_cycle

Relative to the unidirectional model in ``evmenu.degradation`` there are
exactly three changes, and all three are inert when ``E_out == 0``:

1. throughput ``phi`` counts battery-side energy in BOTH directions;
2. the depth factor uses the SOC excursion ``delta``, not ``phi``;
3. the peak C-rate takes the max over charge and (lambda_d-weighted)
   discharge rates.

Proposition 2: with ``E_out == 0`` and no trip inside the connected window
the SOC trajectory is non-decreasing, so ``delta == phi`` identically and
this module reproduces ``evmenu.degradation`` exactly -- not approximately.
``tests_v2g/test_v1g_regression.py`` asserts that as a strict equality.
"""

from __future__ import annotations

from dataclasses import dataclass

from evmenu.degradation import (
    _HOURS_PER_YEAR,
    DegradationSettings,
    _relative_age_factor,
    _temperature_factor,
    calendar_soc_stress,
    cycle_age_slope,
)
from evmenu.schemas import ChargingSession, EVSpec, PlanningSignal

#: One equivalent full cycle is 2 * B_max of battery-side throughput, so a
#: full charge plus a full discharge advances the count by exactly one.
#: The package itself never advances ``cumulative_equivalent_full_cycles``;
#: it is a read-only scenario input there.  This constant defines the
#: convention the formulation adds.  See ``test_efc_convention``.
THROUGHPUT_FRACTIONS_PER_FULL_CYCLE = 2.0


@dataclass(frozen=True, slots=True)
class BidirectionalFade:
    """Decomposed incremental fade for one bidirectional trajectory."""

    calendar_window: float
    calendar_parked: float
    cycle: float
    total: float
    throughput_fraction: float
    soc_excursion: float
    peak_c_rate: float
    export_energy_kwh: float

    @property
    def equivalent_full_cycles(self) -> float:
        """Cycle count this session advances, ``dN = phi / 2`` (eq. 55)."""
        return self.throughput_fraction / THROUGHPUT_FRACTIONS_PER_FULL_CYCLE


def calendar_window_fade(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    soc: tuple[float, ...] | list[float],
    settings: DegradationSettings,
) -> float:
    """Calendar fade over the connected window, eq. (49).

    ``soc`` holds the state-boundary SOC values; the final boundary is the
    departure state and is excluded, matching ``evmenu.degradation``.
    """
    params = settings.parameters_for(ev.chemistry)
    age_factor = _relative_age_factor(
        age_years=settings.battery_age_years,
        alpha_time=params.calendar_time_exponent,
        reference_age_years=settings.reference_age_years,
    )
    total = 0.0
    for local_step, state in enumerate(soc[:-1]):
        global_step = session.arrival_step + local_step
        temperature = (
            settings.fallback_temperature_c
            if signal.battery_temperature_c is None
            else signal.battery_temperature_c[global_step]
        )
        clamped = min(1.0, max(0.0, state))
        total += (
            calendar_soc_stress(clamped, params)
            * _temperature_factor(
                temperature,
                settings.reference_temperature_c,
                params.activation_energy_j_per_mol,
            )
            * age_factor
            * signal.interval_durations[global_step]
            / _HOURS_PER_YEAR
        )
    return total


def calendar_parked_fade(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    terminal_energy_kwh: float,
    settings: DegradationSettings,
) -> float:
    """Calendar fade over the parked period, eq. (50).

    Proposition 1: this depends on the trajectory only through the terminal
    energy, which the terminal equality pins to the request.  It is a
    constant of ``(r, z, zeta)`` and is excluded from the solver objective.
    """
    if settings.parked_day_hours is None:
        return 0.0
    params = settings.parameters_for(ev.chemistry)
    age_factor = _relative_age_factor(
        age_years=settings.battery_age_years,
        alpha_time=params.calendar_time_exponent,
        reference_age_years=settings.reference_age_years,
    )
    parked_soc = (terminal_energy_kwh - session.commute_energy_kwh) / ev.battery_capacity_kwh
    if not 0.0 <= parked_soc <= 1.0:
        raise ValueError(f"computed parked SOC {parked_soc!r} lies outside [0, 1].")
    temperature = (
        settings.fallback_temperature_c
        if signal.battery_temperature_c is None
        else signal.battery_temperature_c[session.departure_step - 1]
    )
    return (
        calendar_soc_stress(parked_soc, params)
        * _temperature_factor(
            temperature,
            settings.reference_temperature_c,
            params.activation_energy_j_per_mol,
        )
        * age_factor
        * settings.parked_day_hours
        / _HOURS_PER_YEAR
    )


def throughput_fraction(
    *,
    ev: EVSpec,
    grid_in_kwh: tuple[float, ...] | list[float],
    grid_out_kwh: tuple[float, ...] | list[float],
    discharge_efficiency: float,
) -> float:
    """Bidirectional battery-side throughput fraction, eq. (51).

    This is a SUM over the horizon of absolute battery-side energy, not a
    maximum.  ``dN = phi / 2`` depends on it being cumulative.
    """
    charged = ev.charging_efficiency * sum(grid_in_kwh)
    discharged = sum(grid_out_kwh) / discharge_efficiency
    return (charged + discharged) / ev.battery_capacity_kwh


def soc_excursion(soc: tuple[float, ...] | list[float]) -> float:
    """Session SOC excursion ``max_t s_t - min_t s_t``, eq. (52).

    Equal to ``phi`` whenever the trajectory is non-decreasing, which is
    why substituting it for ``phi`` in the depth factor is backward
    compatible (Proposition 2).
    """
    if not len(soc):
        return 0.0
    return max(soc) - min(soc)


def peak_c_rate(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    grid_in_kwh: tuple[float, ...] | list[float],
    grid_out_kwh: tuple[float, ...] | list[float],
    discharge_efficiency: float,
    discharge_severity: float,
) -> float:
    """Peak battery-side C-rate over both directions, eq. (53)."""
    peak = 0.0
    for index, energy_in in enumerate(grid_in_kwh):
        duration = signal.interval_durations[session.arrival_step + index]
        charge_rate = ev.charging_efficiency * energy_in / duration
        discharge_rate = (
            discharge_severity * grid_out_kwh[index] / (discharge_efficiency * duration)
        )
        peak = max(peak, charge_rate, discharge_rate)
    return peak / ev.battery_capacity_kwh


def cycle_fade(
    *,
    ev: EVSpec,
    settings: DegradationSettings,
    phi: float,
    delta: float,
    chat: float,
) -> float:
    """Cycle fade, eq. (54).

    Identical in form to ``evmenu.degradation``; the depth factor takes the
    SOC excursion ``delta`` where the unidirectional model takes ``phi``.
    """
    params = settings.parameters_for(ev.chemistry)
    slope = cycle_age_slope(
        settings.cumulative_equivalent_full_cycles,
        throughput_exponent=params.cycle_time_exponent,
        minimum_fec=settings.minimum_effective_fec,
    )
    return (
        params.cycle_reference_coefficient
        * (1.0 + params.cycle_dod_coefficient * delta)
        * (1.0 + params.cycle_c_rate_coefficient * chat)
        * slope
        * phi
    )


def assess_trajectory(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    grid_in_kwh: tuple[float, ...] | list[float],
    grid_out_kwh: tuple[float, ...] | list[float],
    soc: tuple[float, ...] | list[float],
    terminal_energy_kwh: float,
    discharge_efficiency: float,
    discharge_severity: float,
    settings: DegradationSettings,
) -> BidirectionalFade:
    """Assess total incremental fade for one bidirectional trajectory."""
    phi = throughput_fraction(
        ev=ev,
        grid_in_kwh=grid_in_kwh,
        grid_out_kwh=grid_out_kwh,
        discharge_efficiency=discharge_efficiency,
    )
    delta = soc_excursion(soc)
    chat = peak_c_rate(
        ev=ev,
        session=session,
        signal=signal,
        grid_in_kwh=grid_in_kwh,
        grid_out_kwh=grid_out_kwh,
        discharge_efficiency=discharge_efficiency,
        discharge_severity=discharge_severity,
    )
    window = calendar_window_fade(
        ev=ev, session=session, signal=signal, soc=soc, settings=settings
    )
    parked = calendar_parked_fade(
        ev=ev,
        session=session,
        signal=signal,
        terminal_energy_kwh=terminal_energy_kwh,
        settings=settings,
    )
    cycle = cycle_fade(ev=ev, settings=settings, phi=phi, delta=delta, chat=chat)
    return BidirectionalFade(
        calendar_window=window,
        calendar_parked=parked,
        cycle=cycle,
        total=window + parked + cycle,
        throughput_fraction=phi,
        soc_excursion=delta,
        peak_c_rate=chat,
        export_energy_kwh=float(sum(grid_out_kwh)),
    )
