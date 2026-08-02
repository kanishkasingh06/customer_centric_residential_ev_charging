from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest

import evmenu.degradation as degradation_module
from evmenu.degradation import (
    DEFAULT_LFP_PARAMETERS,
    DEFAULT_NMC_PARAMETERS,
    ChemistryDegradationParameters,
    DegradationScoredMenu,
    DegradationSettings,
    assess_candidate_degradation,
    calendar_soc_stress,
    score_generated_menu,
)
from evmenu.exceptions import PhysicalConstraintError, SchemaValidationError
from evmenu.menu import MenuCandidate, generate_candidate_menu
from evmenu.schemas import (
    ChargingProfile,
    ChargingSession,
    Chemistry,
    EVSpec,
    MenuSettings,
    PlanningSignal,
)
from evmenu.validation import ValidationReport


def ev(chemistry: Chemistry = "NMC") -> EVSpec:
    return EVSpec(
        ev_id="ev-1",
        battery_capacity_kwh=40.0,
        minimum_energy_kwh=4.0,
        charger_power_kw=4.0,
        charging_efficiency=1.0,
        chemistry=chemistry,
    )


def session() -> ChargingSession:
    return ChargingSession(
        arrival_step=0,
        departure_step=6,
        initial_energy_kwh=20.0,
        commute_energy_kwh=8.0,
        buffer_energy_kwh=2.0,
    )


def signal(temperatures: tuple[float, ...] | None = None) -> PlanningSignal:
    return PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=(10.0, 8.0, 6.0, 4.0, 2.0, 1.0),
        battery_temperature_c=temperatures,
    )


def direct_zero_charge_candidate(
    *,
    chemistry: Chemistry,
    soc: float,
    initial_energy_kwh: float | None = None,
    commute_energy_kwh: float = 0.0,
    timestep_hours: float = 1e-12,
    arrival_step: int = 0,
) -> tuple[EVSpec, ChargingSession, PlanningSignal, MenuCandidate]:
    initial = soc * 40.0 if initial_energy_kwh is None else initial_energy_kwh
    vehicle = EVSpec(
        ev_id=f"{chemistry}-{soc}",
        battery_capacity_kwh=40.0,
        minimum_energy_kwh=0.0,
        charger_power_kw=4.0,
        charging_efficiency=1.0,
        chemistry=chemistry,
    )
    charging_session = ChargingSession(
        arrival_step=arrival_step,
        departure_step=arrival_step + 1,
        initial_energy_kwh=initial,
        commute_energy_kwh=commute_energy_kwh,
        buffer_energy_kwh=0.0,
    )
    planning_signal = PlanningSignal(
        timestep_hours=timestep_hours,
        price_per_kwh=(1.0,) * (arrival_step + 1),
        battery_temperature_c=(30.0,) * (arrival_step + 1),
    )
    profile = ChargingProfile(
        start_step=arrival_step,
        grid_energy_kwh=(0.0,),
        battery_energy_kwh=(initial, initial),
        power_kw=(0.0,),
        soc=(soc, soc),
    )
    candidate = MenuCandidate(
        candidate_id=f"{chemistry}-{soc}-bau",
        ev_id=vehicle.ev_id,
        kind="immediate_bau",
        target_soc=soc,
        target_sources=("minimum_required",),
        target_label="direct test target",
        ready_step=arrival_step,
        charging_cost=0.0,
        same_target_bau_cost=0.0,
        saving=0.0,
        required_grid_energy_kwh=0.0,
        profile=profile,
        validation=ValidationReport(issues=()),
    )
    return vehicle, charging_session, planning_signal, candidate


def test_calendar_anchor_values() -> None:
    assert calendar_soc_stress(0.5, DEFAULT_LFP_PARAMETERS) == pytest.approx(0.0100)
    assert calendar_soc_stress(1.0, DEFAULT_LFP_PARAMETERS) == pytest.approx(0.0124)
    assert calendar_soc_stress(0.5, DEFAULT_NMC_PARAMETERS) == pytest.approx(0.0178)
    assert calendar_soc_stress(0.8, DEFAULT_NMC_PARAMETERS) == pytest.approx(0.0241)
    assert calendar_soc_stress(1.0, DEFAULT_NMC_PARAMETERS) == pytest.approx(0.0302)


@pytest.mark.parametrize(
    ("parameters", "alpha"),
    ((DEFAULT_LFP_PARAMETERS, 0.5), (DEFAULT_NMC_PARAMETERS, 0.75)),
)
def test_relative_calendar_age_factor_matches_local_slope(
    parameters: ChemistryDegradationParameters,
    alpha: float,
) -> None:
    for age in (1.0, 3.0, 10.0):
        expected = (alpha * age ** (alpha - 1.0)) / (alpha * 1.0 ** (alpha - 1.0))
        assert degradation_module._relative_age_factor(
            age_years=age,
            alpha_time=parameters.calendar_time_exponent,
            reference_age_years=1.0,
        ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("chemistry", "soc", "expected"),
    (
        ("LFP", 0.50, 0.0100),
        ("LFP", 1.00, 0.0124),
        ("NMC", 0.50, 0.0178),
        ("NMC", 0.80, 0.0241),
        ("NMC", 1.00, 0.0302),
    ),
)
def test_end_to_end_calendar_anchor_values(
    chemistry: Chemistry,
    soc: float,
    expected: float,
) -> None:
    vehicle, charging_session, planning_signal, candidate = direct_zero_charge_candidate(
        chemistry=chemistry,
        soc=soc,
    )
    assessment = assess_candidate_degradation(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        candidate=candidate,
        menu_settings=MenuSettings(equivalent_sessions_per_year=1),
        degradation_settings=DegradationSettings(
            battery_age_years=1.0,
            cumulative_equivalent_full_cycles=0.0,
            parked_day_hours=8760.0,
        ),
    )
    assert assessment.parked_day_calendar_fade == pytest.approx(expected, abs=1e-12)


def test_nmc_has_steeper_high_soc_stress_than_lfp() -> None:
    lfp_increase = calendar_soc_stress(1.0, DEFAULT_LFP_PARAMETERS) - calendar_soc_stress(
        0.5, DEFAULT_LFP_PARAMETERS
    )
    nmc_increase = calendar_soc_stress(1.0, DEFAULT_NMC_PARAMETERS) - calendar_soc_stress(
        0.5, DEFAULT_NMC_PARAMETERS
    )
    assert nmc_increase > lfp_increase


def test_assessment_is_additive_and_annualized() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    assessment = assess_candidate_degradation(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        candidate=menu.candidates[0],
    )
    assert assessment.total_fade == pytest.approx(
        assessment.charging_window_calendar_fade
        + assessment.parked_day_calendar_fade
        + assessment.cycle_fade
    )
    assert assessment.annualized_degradation_pct == pytest.approx(assessment.total_fade * 300 * 100)


def test_zero_fec_is_regularized_without_nan_or_infinity() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    charged = next(candidate for candidate in menu.candidates if candidate.target_soc == 1.0)
    assessment = assess_candidate_degradation(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        candidate=charged,
        degradation_settings=DegradationSettings(cumulative_equivalent_full_cycles=0.0),
    )
    assert assessment.cycle_fade > 0.0
    assert isfinite(assessment.cycle_fade)

    zero_vehicle, zero_session, zero_signal, zero_candidate = direct_zero_charge_candidate(
        chemistry="LFP", soc=0.5
    )
    zero_assessment = assess_candidate_degradation(
        ev=zero_vehicle,
        session=zero_session,
        signal=zero_signal,
        candidate=zero_candidate,
        degradation_settings=DegradationSettings(cumulative_equivalent_full_cycles=0.0),
    )
    assert zero_assessment.cycle_fade == 0.0

    with pytest.raises(PhysicalConstraintError):
        DegradationSettings(cumulative_equivalent_full_cycles=-1.0)


def test_battery_side_c_rate_uses_efficiency_once() -> None:
    charging_session = session()
    planning_signal = signal()
    full_efficiency = EVSpec("ev-1", 40.0, 4.0, 8.0, 1.0, "NMC")
    lower_efficiency = EVSpec("ev-1", 40.0, 4.0, 8.0, 0.5, "NMC")
    full_candidate = next(
        candidate
        for candidate in generate_candidate_menu(
            ev=full_efficiency, session=charging_session, signal=planning_signal
        ).candidates
        if candidate.target_soc == 0.8 and candidate.kind == "immediate_bau"
    )
    lower_candidate = next(
        candidate
        for candidate in generate_candidate_menu(
            ev=lower_efficiency, session=charging_session, signal=planning_signal
        ).candidates
        if candidate.target_soc == 0.8 and candidate.kind == "immediate_bau"
    )
    full_assessment = assess_candidate_degradation(
        ev=full_efficiency,
        session=charging_session,
        signal=planning_signal,
        candidate=full_candidate,
    )
    lower_assessment = assess_candidate_degradation(
        ev=lower_efficiency,
        session=charging_session,
        signal=planning_signal,
        candidate=lower_candidate,
    )
    assert lower_assessment.peak_c_rate == pytest.approx(full_assessment.peak_c_rate * 0.5)

    larger_ev = EVSpec("ev-1", 80.0, 4.0, 4.0, 1.0, "NMC")
    larger_session = ChargingSession(0, 20, 20.0, 8.0, 2.0)
    larger_signal = PlanningSignal(1.0, (1.0,) * 20)
    larger_menu = generate_candidate_menu(
        ev=larger_ev, session=larger_session, signal=larger_signal
    )
    larger_candidate = max(larger_menu.candidates, key=lambda item: item.target_soc)
    larger_assessment = assess_candidate_degradation(
        ev=larger_ev,
        session=larger_session,
        signal=larger_signal,
        candidate=larger_candidate,
    )
    assert larger_assessment.peak_c_rate < full_assessment.peak_c_rate


def test_arrhenius_reference_ordering_and_overflow_error() -> None:
    assert degradation_module._temperature_factor(30.0, 30.0, 24000.0) == pytest.approx(1.0)
    assert degradation_module._temperature_factor(40.0, 30.0, 24000.0) > 1.0
    assert degradation_module._temperature_factor(30.0, 30.0, 24000.0) > (
        degradation_module._temperature_factor(20.0, 30.0, 24000.0)
    )
    with pytest.raises(PhysicalConstraintError):
        degradation_module._temperature_factor(40.0, 20.0, 1e308)


def test_hotter_temperature_increases_calendar_fade() -> None:
    vehicle = ev()
    charging_session = session()
    cool = signal((20.0,) * 6)
    hot = signal((40.0,) * 6)
    candidate = generate_candidate_menu(
        ev=vehicle, session=charging_session, signal=cool
    ).candidates[0]
    cool_assessment = assess_candidate_degradation(
        ev=vehicle, session=charging_session, signal=cool, candidate=candidate
    )
    hot_assessment = assess_candidate_degradation(
        ev=vehicle, session=charging_session, signal=hot, candidate=candidate
    )
    assert hot_assessment.charging_window_calendar_fade > (
        cool_assessment.charging_window_calendar_fade
    )
    assert hot_assessment.parked_day_calendar_fade > cool_assessment.parked_day_calendar_fade


def test_window_uses_beginning_of_interval_soc_once() -> None:
    vehicle = EVSpec("window", 40.0, 0.0, 24.0, 1.0, "LFP")
    charging_session = ChargingSession(0, 1, 8.0, 0.0, 0.0)
    planning_signal = PlanningSignal(1.0, (1.0,), battery_temperature_c=(30.0,))
    profile = ChargingProfile(
        start_step=0,
        grid_energy_kwh=(24.0,),
        battery_energy_kwh=(8.0, 32.0),
        power_kw=(24.0,),
        soc=(0.2, 0.8),
    )
    candidate = MenuCandidate(
        candidate_id="window-bau",
        ev_id="window",
        kind="immediate_bau",
        target_soc=0.8,
        target_sources=("minimum_required",),
        target_label="window",
        ready_step=1,
        charging_cost=24.0,
        same_target_bau_cost=24.0,
        saving=0.0,
        required_grid_energy_kwh=24.0,
        profile=profile,
        validation=ValidationReport(issues=()),
    )
    assessment = assess_candidate_degradation(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        candidate=candidate,
        menu_settings=MenuSettings(equivalent_sessions_per_year=1),
        degradation_settings=DegradationSettings(
            battery_age_years=1.0,
            cumulative_equivalent_full_cycles=0.0,
            parked_day_hours=0.0,
        ),
    )
    expected = calendar_soc_stress(0.2, DEFAULT_LFP_PARAMETERS) / 8760.0
    assert assessment.charging_window_calendar_fade == pytest.approx(expected)


def test_final_session_temperature_is_parked_proxy() -> None:
    vehicle = EVSpec("temp", 40.0, 0.0, 4.0, 1.0, "LFP")
    charging_session = ChargingSession(1, 3, 20.0, 0.0, 0.0)
    planning_signal = PlanningSignal(
        1.0,
        (1.0, 1.0, 1.0),
        battery_temperature_c=(20.0, 30.0, 40.0),
    )
    profile = ChargingProfile(
        start_step=1,
        grid_energy_kwh=(0.0, 0.0),
        battery_energy_kwh=(20.0, 20.0, 20.0),
        power_kw=(0.0, 0.0),
        soc=(0.5, 0.5, 0.5),
    )
    candidate = MenuCandidate(
        candidate_id="temp-bau",
        ev_id="temp",
        kind="immediate_bau",
        target_soc=0.5,
        target_sources=("minimum_required",),
        target_label="temperature",
        ready_step=1,
        charging_cost=0.0,
        same_target_bau_cost=0.0,
        saving=0.0,
        required_grid_energy_kwh=0.0,
        profile=profile,
        validation=ValidationReport(issues=()),
    )
    assessment = assess_candidate_degradation(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        candidate=candidate,
        menu_settings=MenuSettings(equivalent_sessions_per_year=1),
        degradation_settings=DegradationSettings(
            battery_age_years=1.0,
            parked_day_hours=8760.0,
        ),
    )
    expected = calendar_soc_stress(
        0.5, DEFAULT_LFP_PARAMETERS
    ) * degradation_module._temperature_factor(40.0, 30.0, 24000.0)
    assert assessment.parked_day_calendar_fade == pytest.approx(expected)


def test_parked_soc_rejection_and_zero_duration() -> None:
    vehicle, charging_session, planning_signal, candidate = direct_zero_charge_candidate(
        chemistry="LFP", soc=0.5, commute_energy_kwh=0.0
    )
    zero_duration = assess_candidate_degradation(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        candidate=candidate,
        degradation_settings=DegradationSettings(parked_day_hours=0.0),
    )
    assert zero_duration.parked_day_calendar_fade == 0.0

    with pytest.raises(PhysicalConstraintError):
        assess_candidate_degradation(
            ev=vehicle,
            session=ChargingSession(0, 1, 20.0, 25.0, 0.0),
            signal=planning_signal,
            candidate=candidate,
        )


def test_higher_target_has_more_parked_fade() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    low = min(menu.candidates, key=lambda item: item.target_soc)
    high = max(menu.candidates, key=lambda item: item.target_soc)
    low_result = assess_candidate_degradation(
        ev=vehicle, session=charging_session, signal=planning_signal, candidate=low
    )
    high_result = assess_candidate_degradation(
        ev=vehicle, session=charging_session, signal=planning_signal, candidate=high
    )
    assert high_result.parked_soc > low_result.parked_soc
    assert high_result.parked_day_calendar_fade > low_result.parked_day_calendar_fade


def test_same_target_cycle_fade_is_profile_sensitive_only_to_peak_c_rate() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    target = menu.candidates[0].target_soc
    candidates = [item for item in menu.candidates if item.target_soc == target]
    assessments = [
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate=item,
        )
        for item in candidates
    ]
    assert all(item.total_fade >= 0.0 for item in assessments)
    assert len({round(item.parked_day_calendar_fade, 16) for item in assessments}) == 1


def test_scoring_is_within_menu_quantized_and_order_preserving() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    scored = score_generated_menu(
        ev=vehicle, session=charging_session, signal=planning_signal, menu=menu
    )
    assert len(scored.offers) == len(menu.candidates) == len(scored.assessments)
    scores = [offer.charging_health_score for offer in scored.offers]
    assert max(scores) == 100.0
    assert min(scores) == 0.0
    assert all(score % 5.0 == 0.0 for score in scores)
    ranked = sorted(
        zip(scored.assessments, scored.offers, strict=True),
        key=lambda pair: pair[0].total_fade,
    )
    assert [offer.charging_health_score for _, offer in ranked] == sorted(
        (offer.charging_health_score for _, offer in ranked), reverse=True
    )


@pytest.mark.parametrize(
    ("raw_health", "expected"),
    ((0.0, 0.0), (2.4, 0.0), (2.5, 5.0), (7.5, 10.0), (97.5, 100.0), (100.0, 100.0)),
)
def test_health_quantization_is_explicit_half_up(
    raw_health: float,
    expected: float,
) -> None:
    fade = 1.0 - raw_health / 100.0
    assert degradation_module._health_score(fade, 0.0, 1.0, 5.0, 0.0) == expected


def test_tiny_health_spread_receives_equal_health() -> None:
    assert degradation_module._health_score(1.0, 1.0, 1.0 + 1e-13, 5.0, 1e-12) == 100.0
    assert degradation_module._health_score(1.0, 1.0, 1.0 + 1e-8, 5.0, 1e-12) == 100.0
    assert degradation_module._health_score(1.0 + 1e-8, 1.0, 1.0 + 1e-8, 5.0, 1e-12) == 0.0


def test_all_equal_fades_receive_health_100() -> None:
    vehicle = ev("LFP")
    charging_session = ChargingSession(
        arrival_step=0,
        departure_step=2,
        initial_energy_kwh=40.0,
        commute_energy_kwh=0.0,
        buffer_energy_kwh=0.0,
    )
    planning_signal = PlanningSignal(timestep_hours=1.0, price_per_kwh=(1.0, 1.0))
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    scored = score_generated_menu(
        ev=vehicle, session=charging_session, signal=planning_signal, menu=menu
    )
    assert all(offer.charging_health_score == 100.0 for offer in scored.offers)


def test_offer_metadata_is_preserved() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    scored = score_generated_menu(
        ev=vehicle, session=charging_session, signal=planning_signal, menu=menu
    )
    for candidate, offer in zip(menu.candidates, scored.offers, strict=True):
        assert offer.offer_id == candidate.candidate_id
        assert offer.ev_id == candidate.ev_id
        assert offer.target_sources == candidate.target_sources
        assert offer.ready_step == candidate.ready_step
        assert offer.target_soc == candidate.target_soc
        assert offer.charging_cost == candidate.charging_cost
        assert offer.same_target_bau_cost == candidate.same_target_bau_cost
        assert offer.advertised_saving == candidate.saving
        assert offer.profile == candidate.profile
    assert tuple(offer.offer_id for offer in scored.offers) == tuple(
        candidate.candidate_id for candidate in menu.candidates
    )


def test_invalid_model_parameters_are_rejected() -> None:
    with pytest.raises(PhysicalConstraintError):
        DegradationSettings(battery_age_years=0.0)
    with pytest.raises(PhysicalConstraintError):
        DegradationSettings(reference_age_years=0.0)
    with pytest.raises(PhysicalConstraintError):
        DegradationSettings(minimum_reference_fec=0.0)
    with pytest.raises(SchemaValidationError):
        DegradationSettings(battery_age_years=float("nan"))
    with pytest.raises(SchemaValidationError):
        DegradationSettings(parked_day_hours=float("inf"))
    with pytest.raises(SchemaValidationError):
        DegradationSettings(health_score_resolution=True)
    with pytest.raises(SchemaValidationError):
        DegradationSettings(degradation_comparison_tolerance=True)
    with pytest.raises(PhysicalConstraintError):
        replace(DEFAULT_LFP_PARAMETERS, calendar_soc_knee=1.1)
    with pytest.raises(PhysicalConstraintError):
        ChemistryDegradationParameters(
            calendar_a0=0.0,
            calendar_a1=0.0,
            calendar_a2=0.0,
            calendar_soc_knee=0.5,
            calendar_time_exponent=0.5,
            activation_energy_j_per_mol=0.0,
            cycle_reference_coefficient=-1.0,
        )


def test_candidate_ev_mismatch_is_rejected() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    candidate = generate_candidate_menu(
        ev=vehicle, session=charging_session, signal=planning_signal
    ).candidates[0]
    other = replace(vehicle, ev_id="other")
    with pytest.raises(SchemaValidationError):
        assess_candidate_degradation(
            ev=other,
            session=charging_session,
            signal=planning_signal,
            candidate=candidate,
        )


def test_public_argument_types_are_rejected_without_incidental_errors() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    candidate = menu.candidates[0]
    with pytest.raises(SchemaValidationError):
        assess_candidate_degradation(
            ev="bad",  # type: ignore[arg-type]
            session=charging_session,
            signal=planning_signal,
            candidate=candidate,
        )
    with pytest.raises(SchemaValidationError):
        assess_candidate_degradation(
            ev=vehicle,
            session="bad",  # type: ignore[arg-type]
            signal=planning_signal,
            candidate=candidate,
        )
    with pytest.raises(SchemaValidationError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal="bad",  # type: ignore[arg-type]
            candidate=candidate,
        )
    with pytest.raises(SchemaValidationError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(SchemaValidationError):
        score_generated_menu(ev="bad", session=charging_session, signal=planning_signal, menu=menu)  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError):
        score_generated_menu(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            menu="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(SchemaValidationError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate=candidate,
            degradation_settings="bad",  # type: ignore[arg-type]
        )


def test_candidate_context_mismatches_are_rejected() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    candidate = next(
        item for item in menu.candidates if item.kind == "immediate_bau" and item.target_soc == 0.8
    )

    with pytest.raises(PhysicalConstraintError):
        assess_candidate_degradation(
            ev=vehicle,
            session=ChargingSession(0, 5, 20.0, 8.0, 2.0),
            signal=PlanningSignal(1.0, (1.0,) * 5),
            candidate=candidate,
        )
    with pytest.raises(PhysicalConstraintError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate=replace(candidate, target_soc=0.9),
        )
    with pytest.raises(PhysicalConstraintError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate=replace(candidate, required_grid_energy_kwh=1.0),
        )
    with pytest.raises(PhysicalConstraintError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate=replace(candidate, profile=replace(candidate.profile, start_step=1)),
        )
    with pytest.raises(PhysicalConstraintError):
        assess_candidate_degradation(
            ev=vehicle,
            session=charging_session,
            signal=planning_signal,
            candidate=replace(candidate, ready_step=6),
        )


def test_assessment_identity_and_additive_validation() -> None:
    vehicle = ev()
    charging_session = session()
    planning_signal = signal()
    menu = generate_candidate_menu(ev=vehicle, session=charging_session, signal=planning_signal)
    scored = score_generated_menu(
        ev=vehicle,
        session=charging_session,
        signal=planning_signal,
        menu=menu,
    )
    for candidate, assessment, offer in zip(
        menu.candidates, scored.assessments, scored.offers, strict=True
    ):
        assert assessment.candidate_id == candidate.candidate_id == offer.offer_id
        assert assessment.ev_id == vehicle.ev_id == offer.ev_id
        assert assessment.chemistry == vehicle.chemistry

    assessment = scored.assessments[0]
    with pytest.raises(PhysicalConstraintError):
        replace(assessment, total_fade=assessment.total_fade + 1.0)
    with pytest.raises(SchemaValidationError):
        replace(assessment, candidate_id="")
    with pytest.raises(SchemaValidationError):
        DegradationScoredMenu(
            ev_id=scored.ev_id,
            offers=(scored.offers[0], scored.offers[0]),
            assessments=(scored.assessments[0], scored.assessments[0]),
        )


def test_unsupported_chemistry_and_stress_parameters_are_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        DegradationSettings().parameters_for("nmc")
    with pytest.raises(SchemaValidationError):
        DegradationSettings().parameters_for("unsupported")
    with pytest.raises(SchemaValidationError):
        calendar_soc_stress(0.5, object())  # type: ignore[arg-type]
