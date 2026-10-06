from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any, cast

import pytest

import evmenu.assembly as assembly_module
from evmenu import (
    AssembledMenu,
    ChargingProfile,
    ChargingSession,
    DegradationSettings,
    DisplayDiversityParameters,
    EVSpec,
    FrontierSettings,
    MenuAssemblySettings,
    MenuGenerationSettings,
    MenuOffer,
    MenuSettings,
    PhysicalConstraintError,
    PlanningSignal,
    SchemaValidationError,
    ValidationCode,
    ValidationIssue,
    ValidationReport,
    assemble_customer_menu,
    generate_candidate_menu,
    prune_ready_step_change_points,
)
from evmenu.menu import GeneratedMenu, MenuCandidate
from evmenu.optimization import SavingFrontier, build_sandwich_saving_frontier


def _context() -> tuple[EVSpec, ChargingSession, PlanningSignal]:
    ev = EVSpec(
        ev_id="ev-7",
        battery_capacity_kwh=60.0,
        minimum_energy_kwh=6.0,
        charger_power_kw=7.0,
        charging_efficiency=0.9,
        chemistry="NMC",
    )
    session = ChargingSession(
        arrival_step=0,
        departure_step=8,
        initial_energy_kwh=18.0,
        commute_energy_kwh=8.0,
        buffer_energy_kwh=4.0,
    )
    signal = PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=(10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 3.0, 5.0),
        battery_temperature_c=(30.0,) * 8,
    )
    return ev, session, signal


def _generated() -> tuple[EVSpec, ChargingSession, PlanningSignal, GeneratedMenu]:
    ev, session, signal = _context()
    return ev, session, signal, generate_candidate_menu(ev=ev, session=session, signal=signal)


def test_assembly_settings_validate() -> None:
    with pytest.raises(SchemaValidationError):
        MenuAssemblySettings(display_cap=True)
    with pytest.raises(PhysicalConstraintError):
        MenuAssemblySettings(display_cap=0)
    with pytest.raises(PhysicalConstraintError):
        MenuAssemblySettings(saving_merge_gap=-1.0)
    with pytest.raises(SchemaValidationError):
        MenuAssemblySettings(delta_saving_merge=True)
    assert MenuAssemblySettings(delta_saving_merge=0.0).delta_saving_merge == 0.0
    assert (
        MenuAssemblySettings(
            target_energy_dominance_tolerance_kwh=0.0
        ).target_energy_dominance_tolerance_kwh
        == 0.0
    )


def test_display_diversity_parameters_validate_and_are_documented_defaults() -> None:
    defaults = DisplayDiversityParameters()
    assert defaults.maximum_displayed_offers == 12
    assert defaults.maximum_offers_per_target == 12
    assert defaults.minimum_ready_separation_minutes == 120
    with pytest.raises(SchemaValidationError):
        DisplayDiversityParameters(maximum_displayed_offers=True)
    with pytest.raises(PhysicalConstraintError):
        DisplayDiversityParameters(minimum_saving_difference=-1.0)
    with pytest.raises(PhysicalConstraintError):
        DisplayDiversityParameters(minimum_relative_stress_difference=1.1)


def test_displayed_stage_is_diverse_subset_and_other_stages_remain_immutable() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
    )
    displayed_ids = {offer.offer_id for offer in assembled.displayed_offers}
    pareto_ids = {offer.offer_id for offer in assembled.pareto_offers}
    generated_snapshot = tuple(assembled.generated_offers)
    assert displayed_ids <= pareto_ids
    assert tuple(assembled.generated_offers) == generated_snapshot
    assert len(assembled.displayed_offers) <= 12
    assert all(offer.advertised_saving > 0.0 for offer in assembled.displayed_offers)
    by_target: dict[float, list[MenuOffer]] = {}
    for offer in assembled.displayed_offers:
        by_target.setdefault(offer.target_soc, []).append(offer)
    for offers in by_target.values():
        assert len(offers) <= 12
    covered_targets = sum(
        any(offer.advertised_saving > 0.0 for offer in offers) for offers in by_target.values()
    )
    summary = assembled.pipeline_diagnostics.display_selection_summary
    assert summary is not None
    assert covered_targets == summary.positive_target_coverage_count


def test_pareto_stage_exposes_all_pareto_offers_not_diversity_subset() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(menu_stage="pareto"),
    )
    assert assembled.menu_stage == "pareto"
    assert len(assembled.offers) == len(assembled.pareto_offers)
    assert len(assembled.displayed_offers) <= 12
    assert len(assembled.generated_offers) == len(
        {offer.target_soc for offer in assembled.generated_offers}
    ) + sum(1 for offer in assembled.generated_offers if offer.advertised_saving > 0.0)


def test_close_ready_tradeoff_requires_both_raw_stress_and_saving_difference() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
    )
    bau = next(
        offer
        for offer in assembled.compacted_offers
        if offer.target_soc == 0.8 and offer.advertised_saving == 0.0
    )
    base = next(
        offer
        for offer in assembled.compacted_offers
        if offer.target_soc == bau.target_soc and offer.advertised_saving > 0.0
    )
    profile_a = replace(
        base.profile,
        grid_energy_kwh=tuple(value + 0.01 for value in base.profile.grid_energy_kwh),
    )
    profile_b = replace(
        base.profile,
        grid_energy_kwh=tuple(value + 0.02 for value in base.profile.grid_energy_kwh),
    )
    anchor = replace(
        bau,
        offer_id="display-bau",
        ready_boundary_absolute_minute=1000,
        raw_battery_stress=10.0,
    )
    lower_stress = replace(
        base,
        offer_id="display-lower-stress",
        ready_boundary_absolute_minute=1100,
        advertised_saving=10.0,
        charging_cost=base.same_target_bau_cost - 10.0,
        raw_battery_stress=20.0,
        profile=profile_a,
    )
    higher_saving = replace(
        base,
        offer_id="display-higher-saving",
        ready_boundary_absolute_minute=1200,
        advertised_saving=20.0,
        charging_cost=base.same_target_bau_cost - 20.0,
        raw_battery_stress=30.0,
        profile=profile_b,
    )
    source = next(
        source for source in assembled.compacted_metadata if source.offer_id == bau.offer_id
    )
    optimized_source = next(
        source for source in assembled.compacted_metadata if source.offer_id == base.offer_id
    )
    source_by_id = {
        "display-bau": replace(source, offer_id="display-bau"),
        "display-lower-stress": replace(
            optimized_source,
            offer_id="display-lower-stress",
            endpoint_role="least_degradation",
        ),
        "display-higher-saving": replace(
            optimized_source,
            offer_id="display-higher-saving",
            endpoint_role="maximum_saving",
        ),
    }
    displayed, _summary, _targets, _decisions, _distinctness = (
        assembly_module._display_diversity_selection(
            (anchor, lower_stress, higher_saving),
            source_by_id,
            DisplayDiversityParameters(
                maximum_displayed_offers=3, minimum_saving_fraction_of_bau=0.0
            ),
            numerical_tolerance=1e-8,
            positive_saving_tolerance=1e-8,
            saving_tolerance=1e-8,
            target_energy_tolerance=1e-8,
            battery_stress_tolerance=1e-8,
            ready_tolerance_minutes=0,
        )
    )
    assert {offer.offer_id for offer in displayed} == {
        "display-lower-stress",
        "display-higher-saving",
    }


def test_change_point_pruning_is_positive_and_deterministic() -> None:
    _, _, _, menu = _generated()
    first = prune_ready_step_change_points(menu)
    second = prune_ready_step_change_points(menu)
    assert first == second
    assert all(candidate.kind == "minimum_cost" for candidate in first)
    assert all(candidate.saving > 0.0 for candidate in first)
    for target in {candidate.target_soc for candidate in first}:
        group = [candidate for candidate in first if candidate.target_soc == target]
        assert group == sorted(group, key=lambda item: item.ready_step)


def test_assembled_menu_is_bounded_aligned_and_deterministic() -> None:
    ev, session, signal, menu = _generated()
    settings = MenuAssemblySettings(display_cap=8, saving_merge_gap=0.01)
    first = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        degradation_settings=DegradationSettings(parked_day_hours=8.0),
        frontier_settings=FrontierSettings(maximum_levels=3),
        assembly_settings=settings,
    )
    second = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        degradation_settings=DegradationSettings(parked_day_hours=8.0),
        frontier_settings=FrontierSettings(maximum_levels=3),
        assembly_settings=settings,
    )
    assert first == second
    assert settings.display_cap is not None
    assert 1 <= len(first.offers) <= settings.display_cap
    assert len(first.offers) == len(first.assessments)
    assert len({offer.offer_id for offer in first.offers}) == len(first.offers)
    assert [offer.offer_id for offer in first.offers] == [
        assessment.candidate_id for assessment in first.assessments
    ]
    assert all(offer.profile.grid_energy_kwh for offer in first.offers)


def test_all_bau_offers_are_preserved() -> None:
    ev, session, signal, menu = _generated()
    bau_ids = {
        candidate.candidate_id for candidate in menu.candidates if candidate.kind == "immediate_bau"
    }
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=12),
    )
    assert bau_ids == {offer.offer_id for offer in assembled.bau_reference_offers}
    assert not any(offer.offer_id in bau_ids for offer in assembled.displayed_offers)


def test_display_cap_applies_only_to_non_bau_options() -> None:
    ev, session, signal, menu = _generated()
    bau_count = sum(candidate.kind == "immediate_bau" for candidate in menu.candidates)
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        assembly_settings=MenuAssemblySettings(display_cap=12),
    )
    assert len(assembled.displayed_offers) <= 12
    assert len(assembled.bau_reference_offers) == bau_count


def test_generated_menu_ev_mismatch_is_rejected() -> None:
    ev, session, signal, menu = _generated()
    with pytest.raises(SchemaValidationError):
        assemble_customer_menu(
            ev=replace(ev, ev_id="other"),
            session=session,
            signal=signal,
            generated_menu=menu,
        )


def test_invalid_public_types_are_rejected() -> None:
    _ev, session, signal, menu = _generated()
    with pytest.raises(SchemaValidationError):
        assemble_customer_menu(
            ev="bad",  # type: ignore[arg-type]
            session=session,
            signal=signal,
            generated_menu=menu,
        )
    with pytest.raises(SchemaValidationError):
        prune_ready_step_change_points("bad")  # type: ignore[arg-type]


def test_negative_or_zero_saving_frontiers_are_not_required() -> None:
    ev, session, _ = _context()
    signal = PlanningSignal(
        timestep_hours=1.0,
        price_per_kwh=(1.0,) * 8,
        battery_temperature_c=(30.0,) * 8,
    )
    menu = generate_candidate_menu(
        ev=ev,
        session=session,
        signal=signal,
        generation_settings=MenuGenerationSettings(deduplicate_identical_profiles=False),
    )
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        menu_settings=MenuSettings(),
        assembly_settings=MenuAssemblySettings(display_cap=12),
    )
    assert not assembled.offers
    assert assembled.bau_reference_offers
    assert all(offer.advertised_saving == 0.0 for offer in assembled.bau_reference_offers)


def _prune_case(values: list[float], *, duplicate_ready: bool = False) -> GeneratedMenu:
    ev, _session, _signal, menu = _generated()
    source = next(
        candidate
        for candidate in menu.candidates
        if candidate.kind == "minimum_cost" and candidate.target_soc == 0.8
    )
    replacements = tuple(
        replace(
            source,
            candidate_id=f"prune-case-{index}",
            ready_step=source.ready_step
            if duplicate_ready and index
            else source.ready_step + index,
            same_target_bau_cost=source.charging_cost + saving,
            saving=saving,
        )
        for index, saving in enumerate(values)
    )
    retained = tuple(
        candidate
        for candidate in menu.candidates
        if not (candidate.kind == "minimum_cost" and candidate.target_soc == 0.8)
    )
    return GeneratedMenu(ev_id=ev.ev_id, candidates=retained + replacements)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], [5, 6, 7]),
        ([1.0, 1.0, 1.0], [5, 7]),
        ([1.0, 2.0, 2.0, 2.0], [5, 6, 8]),
        ([1.0, 1.0, 2.0, 2.0, 3.0], [5, 6, 7, 8, 9]),
    ],
)
def test_pruning_strict_increases_and_plateau_tails(
    values: list[float], expected: list[int]
) -> None:
    menu = _prune_case(values)
    retained = prune_ready_step_change_points(menu)
    assert [
        candidate.ready_step for candidate in retained if candidate.target_soc == 0.8
    ] == expected


def test_pruning_rejects_nonmonotone_savings() -> None:
    with pytest.raises(SchemaValidationError, match="nondecreasing"):
        prune_ready_step_change_points(_prune_case([1.0, 2.0, 1.9, 3.0]))


def test_pruning_tolerance_and_degenerate_inputs() -> None:
    near_plateau = prune_ready_step_change_points(
        _prune_case([1.0, 1.0 + 5e-9]), pruning_tolerance=1e-8
    )
    assert [candidate.ready_step for candidate in near_plateau if candidate.target_soc == 0.8] == [
        5,
        6,
    ]
    assert not [
        candidate
        for candidate in prune_ready_step_change_points(_prune_case([0.0, -1.0]))
        if candidate.target_soc == 0.8
    ]
    assert [
        candidate.ready_step
        for candidate in prune_ready_step_change_points(_prune_case([1.0]))
        if candidate.target_soc == 0.8
    ] == [5]


def test_pruning_rejects_duplicate_ready_steps_and_is_order_independent() -> None:
    with pytest.raises(SchemaValidationError, match="duplicate minimum-cost ready_step"):
        prune_ready_step_change_points(_prune_case([1.0, 2.0], duplicate_ready=True))
    menu = _prune_case([1.0, 2.0, 2.0, 2.0])
    shuffled = list(menu.candidates)
    Random(7).shuffle(shuffled)
    assert prune_ready_step_change_points(menu) == prune_ready_step_change_points(
        GeneratedMenu(ev_id=menu.ev_id, candidates=tuple(shuffled))
    )


def test_generated_menu_preflight_rejects_duplicate_or_missing_bau() -> None:
    ev, session, signal, menu = _generated()
    bau = next(candidate for candidate in menu.candidates if candidate.kind == "immediate_bau")
    duplicate = replace(bau, candidate_id=f"{bau.candidate_id}-duplicate")
    with pytest.raises(SchemaValidationError, match="duplicate BAU"):
        assemble_customer_menu(
            ev=ev,
            session=session,
            signal=signal,
            generated_menu=GeneratedMenu(ev_id=ev.ev_id, candidates=menu.candidates + (duplicate,)),
            assembly_settings=MenuAssemblySettings(display_cap=20),
        )
    missing = tuple(
        candidate
        for candidate in menu.candidates
        if not (candidate.kind == "immediate_bau" and candidate.target_soc == 0.8)
    )
    with pytest.raises(PhysicalConstraintError, match="missing its BAU"):
        assemble_customer_menu(
            ev=ev,
            session=session,
            signal=signal,
            generated_menu=GeneratedMenu(ev_id=ev.ev_id, candidates=missing),
        )


@pytest.mark.parametrize("field", ["saving", "charging_cost", "profile", "validation"])
def test_generated_menu_preflight_rejects_altered_candidate(field: str) -> None:
    ev, session, signal, menu = _generated()
    candidate = next(
        item for item in menu.candidates if item.kind == "minimum_cost" and item.target_soc == 0.8
    )
    if field == "saving":
        altered = replace(candidate, saving=candidate.saving + 1.0)
    elif field == "charging_cost":
        altered = replace(candidate, charging_cost=candidate.charging_cost + 1.0)
    elif field == "profile":
        bau = next(item for item in menu.candidates if item.kind == "immediate_bau")
        altered = replace(candidate, profile=bau.profile)
    else:
        altered = replace(
            candidate,
            validation=candidate.validation,
        )
        object.__setattr__(
            altered,
            "validation",
            ValidationReport(issues=(ValidationIssue(ValidationCode.TARGET_MISMATCH, "invalid"),)),
        )
    candidates = tuple(
        altered if item.candidate_id == candidate.candidate_id else item for item in menu.candidates
    )
    with pytest.raises((SchemaValidationError, PhysicalConstraintError)):
        assemble_customer_menu(
            ev=ev,
            session=session,
            signal=signal,
            generated_menu=GeneratedMenu(ev_id=ev.ev_id, candidates=candidates),
        )


def test_frontier_invocation_count_and_order(monkeypatch: pytest.MonkeyPatch) -> None:
    ev, session, signal, menu = _generated()
    calls: list[tuple[float, int]] = []
    original = build_sandwich_saving_frontier

    def wrapped(**kwargs: object) -> SavingFrontier:
        candidate = cast(MenuCandidate, kwargs["candidate"])
        calls.append((candidate.target_soc, candidate.ready_step))
        return cast(SavingFrontier, cast(Any, original)(**kwargs))

    monkeypatch.setattr(assembly_module, "build_sandwich_saving_frontier", wrapped)
    assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    expected = [
        (candidate.target_soc, candidate.ready_step)
        for candidate in prune_ready_step_change_points(menu)
    ]
    assert calls == expected


def test_frontier_failure_is_contextual_and_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    ev, session, signal, menu = _generated()
    calls: list[str] = []

    def failed(**kwargs: object) -> SavingFrontier:
        candidate = cast(MenuCandidate, kwargs["candidate"])
        calls.append(candidate.candidate_id)
        raise ValueError("boom")

    monkeypatch.setattr(assembly_module, "build_sandwich_saving_frontier", failed)
    with pytest.raises(
        PhysicalConstraintError, match="candidate=.*target_soc=.*ready_step"
    ) as error:
        assemble_customer_menu(ev=ev, session=session, signal=signal, generated_menu=menu)
    assert isinstance(error.value.__cause__, ValueError)
    assert len(calls) == 1


def test_source_metadata_and_assessment_alignment() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    assert len(assembled.source_metadata) == len(assembled.offers)
    assert [item.offer_id for item in assembled.source_metadata] == [
        offer.offer_id for offer in assembled.offers
    ]
    assert all(
        item.endpoint_role == "bau"
        for item in assembled.source_metadata
        if item.source_kind == "bau"
    )
    assert all(
        item.source_point_id is not None
        for item in assembled.source_metadata
        if item.source_kind == "optimized"
    )


def test_health_scores_are_stable_after_post_normalization_display_reduction() -> None:
    ev, session, signal, menu = _generated()
    full = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    reduced = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=8),
    )
    full_health = {offer.offer_id: offer.charging_health_score for offer in full.offers}
    reduced_health = {offer.offer_id: offer.charging_health_score for offer in reduced.offers}
    assert reduced_health
    assert all(full_health[offer_id] == score for offer_id, score in reduced_health.items())


def test_exact_duplicate_removal_works_with_zero_gap() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20, saving_merge_gap=0.0),
    )
    offer = next(item for item in assembled.compacted_offers if item.advertised_saving > 0.0)
    duplicate_a = replace(offer, offer_id="duplicate-a")
    duplicate_z = replace(offer, offer_id="duplicate-z")
    assessments = {item.candidate_id: item for item in assembled.compacted_assessments}
    assessments["duplicate-a"] = replace(assessments[offer.offer_id], candidate_id="duplicate-a")
    assessments["duplicate-z"] = replace(assessments[offer.offer_id], candidate_id="duplicate-z")
    result = assembly_module._remove_exact_duplicates(
        (duplicate_z, duplicate_a), assessments, set()
    )
    assert [item.offer_id for item in result] == ["duplicate-a"]


def test_compaction_boundaries_chain_and_tie_breaks() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    base = next(item for item in assembled.compacted_offers if item.advertised_saving > 0.0)
    chain = tuple(
        replace(
            base,
            offer_id=f"chain-{index}",
            advertised_saving=value,
            charging_cost=base.same_target_bau_cost - value,
            charging_health_score=50.0 + index,
        )
        for index, value in enumerate((0.00, 0.09, 0.18))
    )
    merged = assembly_module._compact_within_request(chain, 0.10, 1e-8, set())
    assert [item.offer_id for item in merged] == ["chain-2"]
    exact = tuple(
        replace(
            base,
            offer_id=f"exact-{index}",
            advertised_saving=value,
            charging_cost=base.same_target_bau_cost - value,
        )
        for index, value in enumerate((1.0, 1.1))
    )
    assert len(assembly_module._compact_within_request(exact, 0.1, 1e-8, set())) == 2


def test_pareto_directions_and_strictness() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    base = next(item for item in assembled.compacted_offers if item.advertised_saving > 0.0)
    improved = replace(
        base,
        offer_id="improved",
        ready_step=max(0, base.ready_step - 1),
        target_soc=min(1.0, base.target_soc + 0.05),
        advertised_saving=base.advertised_saving + 1.0,
        charging_cost=base.same_target_bau_cost - base.advertised_saving - 1.0,
        charging_health_score=min(100.0, base.charging_health_score + 5.0),
    )
    assert assembly_module._dominates(
        improved,
        base,
        saving_tolerance=1e-8,
        health_tolerance=1e-8,
        target_tolerance=1e-8,
    )
    assert not assembly_module._dominates(
        replace(improved, offer_id="equal"),
        improved,
        saving_tolerance=1e-8,
        health_tolerance=1e-8,
        target_tolerance=1e-8,
    )


def test_tight_display_cap_preserves_collision_or_rejects_distinct_anchors() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    base = next(item for item in assembled.compacted_offers if item.advertised_saving > 0.0)
    high = replace(
        base,
        offer_id="high",
        advertised_saving=9.0,
        charging_cost=base.same_target_bau_cost - 9.0,
        charging_health_score=90.0,
    )
    low = replace(
        base,
        offer_id="low",
        advertised_saving=10.0,
        charging_cost=base.same_target_bau_cost - 10.0,
        charging_health_score=10.0,
    )
    selected = assembly_module._select_displayed((high, low), set(), 1)
    assert [item.offer_id for item in selected] == ["low"]
    bau = next(item for item in assembled.bau_reference_offers if item.advertised_saving == 0.0)
    collision = replace(
        base,
        offer_id="collision",
        advertised_saving=10.0,
        charging_cost=base.same_target_bau_cost - 10.0,
        charging_health_score=100.0,
    )
    other = replace(
        base,
        offer_id="other",
        advertised_saving=5.0,
        charging_cost=base.same_target_bau_cost - 5.0,
        charging_health_score=20.0,
    )
    selected = assembly_module._select_displayed((bau, collision, other), {bau.offer_id}, 2)
    assert {item.offer_id for item in selected} == {bau.offer_id, "collision"}


def test_no_charge_target_is_bau_only() -> None:
    ev, session, signal, menu = _generated()
    no_charge = GeneratedMenu(
        ev_id=ev.ev_id,
        candidates=tuple(candidate for candidate in menu.candidates if candidate.target_soc == 0.3),
    )
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=no_charge,
        assembly_settings=MenuAssemblySettings(display_cap=2),
    )
    assert not assembled.offers
    assert len(assembled.bau_reference_offers) == 1
    assert assembled.bau_reference_offers[0].advertised_saving == 0.0
    assert not assembled.source_frontiers


def test_paper_filtering_pipeline_preserves_generated_snapshot_and_exposes_counts() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=3),
        assembly_settings=MenuAssemblySettings(delta_saving_merge=5.0),
    )
    assert assembled.menu_stage == "displayed"
    assert assembled.generated_offer_count == len(assembled.generated_offers)
    assert len(assembled.generated_offers) == assembled.pipeline_diagnostics.generated_offer_count
    assert len(assembled.retained_offers) == assembled.pipeline_diagnostics.retained_offer_count
    assert len(assembled.compacted_offers) == assembled.pipeline_diagnostics.compacted_offer_count
    assert len(assembled.pareto_offers) == assembled.pipeline_diagnostics.pareto_offer_count
    assert len(assembled.offers) == assembled.pipeline_diagnostics.displayed_offer_count
    generated_by_id = {offer.offer_id: offer for offer in assembled.generated_offers}
    assert all(
        generated_by_id[offer.offer_id].profile == offer.profile
        for offer in assembled.retained_offers
    )
    assert all(offer.offer_id in generated_by_id for offer in assembled.compacted_offers)
    assert assembled.pipeline_diagnostics.compaction_removed_count == (
        len(assembled.retained_offers) - len(assembled.compacted_offers)
    )


def test_compaction_uses_actual_saving_and_does_not_cross_requests() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(delta_saving_merge=5.0),
    )
    base = next(offer for offer in assembled.compacted_offers if offer.advertised_saving > 0.0)
    source = assembled.compacted_metadata[assembled.compacted_offers.index(base)]
    source_by_id = {source.offer_id: source}
    same_group = tuple(
        replace(
            base,
            offer_id=f"compact-{index}",
            advertised_saving=saving,
            charging_cost=base.same_target_bau_cost - saving,
            raw_battery_stress=stress,
            provenance_flags=("is_intermediate",),
            scientific_role="intermediate",
        )
        for index, (saving, stress) in enumerate(((1.0, 4.0), (4.0, 1.0), (8.0, 3.0)))
    )
    other_request = replace(
        same_group[0],
        offer_id="compact-other-request",
        ready_boundary_absolute_minute=(base.ready_boundary_absolute_minute or base.ready_step) + 1,
    )
    other_target = replace(
        same_group[0],
        offer_id="compact-other-target",
        target_soc=same_group[0].target_soc + 0.1,
    )
    all_offers = same_group + (other_request, other_target)
    source_by_id = {
        offer.offer_id: replace(source, offer_id=offer.offer_id) for offer in all_offers
    }
    compacted, groups, _decisions = assembly_module._compact_offer_stages(
        all_offers,
        source_by_id,
        delta_saving_merge=5.0,
        numerical_tolerance=1e-8,
        endpoint_tolerance=1e-8,
    )
    ids = {offer.offer_id for offer in compacted}
    assert "compact-other-request" in ids
    assert "compact-other-target" in ids
    assert "compact-2" in ids  # true maximum endpoint is preserved
    assert "compact-1" in ids  # lower raw stress wins the first bucket
    assert "compact-0" not in ids
    assert groups[0].input_offer_count == 3


def test_pareto_uses_absolute_ready_target_saving_and_raw_stress() -> None:
    ev, session, signal, menu = _generated()
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
    )
    base = next(offer for offer in assembled.compacted_offers if offer.advertised_saving > 0.0)
    source = assembled.compacted_metadata[assembled.compacted_offers.index(base)]
    offers = tuple(
        replace(
            base,
            offer_id=offer_id,
            ready_boundary_absolute_minute=ready,
            target_battery_energy_kwh=target_energy,
            target_soc=target_energy / 60.0,
            advertised_saving=saving,
            charging_cost=base.same_target_bau_cost - saving,
            raw_battery_stress=stress,
        )
        for offer_id, ready, target_energy, saving, stress in (
            ("pareto-a", 10, 30.0, 5.0, 2.0),
            ("pareto-b", 11, 29.0, 4.0, 3.0),
            ("pareto-c", 10, 30.0, 6.0, 1.0),
        )
    )
    source_by_id = {offer.offer_id: replace(source, offer_id=offer.offer_id) for offer in offers}
    efficient, anchors, pairs = assembly_module._pareto_offer_stages(
        offers,
        source_by_id,
        saving_tolerance=1e-8,
        target_energy_tolerance=1e-8,
        battery_stress_tolerance=1e-8,
        ready_tolerance_minutes=0,
    )
    assert {offer.offer_id for offer in efficient} == {"pareto-c"}
    assert not anchors
    assert ("pareto-c", "pareto-a") in pairs
    assert ("pareto-c", "pareto-b") in pairs


def test_assembled_menu_rejects_missing_bau_and_preserves_snapshots() -> None:
    ev, session, signal, menu = _generated()
    original_menu = menu
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=FrontierSettings(maximum_levels=2),
        assembly_settings=MenuAssemblySettings(display_cap=20),
    )
    kept = [
        (offer, assessment, source)
        for offer, assessment, source in zip(
            assembled.offers, assembled.assessments, assembled.source_metadata, strict=True
        )
        if not (source.source_kind == "bau" and offer.target_soc == 0.8)
    ]
    with pytest.raises(SchemaValidationError, match="exactly one BAU"):
        AssembledMenu(
            ev_id=assembled.ev_id,
            offers=tuple(item[0] for item in kept),
            assessments=tuple(item[1] for item in kept),
            source_frontiers=assembled.source_frontiers,
            source_metadata=tuple(item[2] for item in kept),
        )
    assert menu == original_menu


def test_saving_band_lower_is_floored_at_zero() -> None:
    """A requested saving below the band width must not produce a negative edge.

    ``select_saving_levels`` always includes the exact maximum saving, however
    small. When that maximum is below the +/- band, the raw lower edge
    ``requested - band`` goes negative, and ``MenuOffer`` rejects a negative
    ``saving_band_lower`` with a ``PhysicalConstraintError`` -- which aborted the
    whole menu over a single offer. Measured on e4bea57, that refused roughly an
    eighth of sampled requests outright.
    """
    band = 0.50

    # The realistic trigger: a maximum-saving point worth less than the band.
    assert assembly_module._saving_band_lower(0.26, band) == 0.0
    assert assembly_module._saving_band_lower(0.009, band) == 0.0
    # Unaffected when the requested saving clears the band.
    assert assembly_module._saving_band_lower(2.0, band) == pytest.approx(1.5)
    # Never negative, for any requested saving the optimizer can produce.
    for requested in (0.0, 1e-9, 0.01, 0.2, 0.49999, 0.5, 0.5001, 1.0, 250.0):
        assert assembly_module._saving_band_lower(requested, band) >= 0.0

    # The floored edge is what MenuOffer accepts; the raw edge is not.
    with pytest.raises(PhysicalConstraintError):
        _menu_offer_with_band(lower=0.26 - band, upper=0.26 + band)
    offer = _menu_offer_with_band(
        lower=assembly_module._saving_band_lower(0.26, band), upper=0.26 + band
    )
    assert offer.saving_band_lower == 0.0
    # A positive realized saving sits inside the floored band, so no violation.
    assert offer.saving_band_violation == 0.0


def _menu_offer_with_band(*, lower: float, upper: float) -> MenuOffer:
    """Build a minimal MenuOffer carrying a saving band, for the test above."""
    profile = ChargingProfile(
        start_step=0,
        grid_energy_kwh=(1.0,),
        battery_energy_kwh=(10.0, 10.9),
        power_kw=(4.0,),
        soc=(0.25, 0.2725),
    )
    return MenuOffer(
        offer_id="band-probe",
        ev_id="ev",
        target_sources=("standard_80",),
        ready_step=1,
        target_soc=0.80,
        charging_cost=10.0,
        same_target_bau_cost=10.26,
        advertised_saving=0.26,
        incremental_degradation=1e-6,
        annualized_degradation_pct=0.01,
        charging_health_score=50.0,
        raw_battery_stress=1e-6,
        profile=profile,
        saving_band_lower=lower,
        saving_band_upper=upper,
    )


def test_assembly_survives_a_requested_saving_below_the_band() -> None:
    """A saving smaller than the band must not abort the whole menu.

    This exercises the real assembly path rather than the helper in isolation:
    with a band wide enough to swallow every achievable saving, every optimized
    offer's raw lower edge ``requested - band`` is negative. Before the floor,
    ``MenuOffer`` rejected the first such edge with a ``PhysicalConstraintError``
    and the entire menu -- BAU anchors included -- was lost.
    """
    ev, session, signal, menu = _generated()

    # Band far larger than any saving this session can produce.
    huge_band = FrontierSettings(maximum_levels=2, saving_band_tolerance=10_000.0)
    assembled = assemble_customer_menu(
        ev=ev,
        session=session,
        signal=signal,
        generated_menu=menu,
        frontier_settings=huge_band,
    )

    assert assembled.displayed_offers, "menu aborted instead of completing"
    bands = [
        offer.saving_band_lower
        for offer in assembled.generated_offers
        if offer.saving_band_lower is not None
    ]
    assert bands, "no offer carried a saving band, so this exercised nothing"
    assert min(bands) == 0.0, "the band floor was not applied on the assembly path"
    assert all(edge >= 0.0 for edge in bands)
    # Every realized saving still sits inside the floored band, so no violation.
    assert all(offer.saving_band_violation == 0.0 for offer in assembled.generated_offers)
