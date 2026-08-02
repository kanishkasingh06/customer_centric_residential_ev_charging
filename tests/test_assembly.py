from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any, cast

import pytest

import evmenu.assembly as assembly_module
from evmenu import (
    AssembledMenu,
    ChargingSession,
    DegradationSettings,
    EVSpec,
    FrontierSettings,
    MenuAssemblySettings,
    MenuGenerationSettings,
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
    assert bau_ids <= {offer.offer_id for offer in assembled.offers}


def test_display_cap_must_fit_bau_references() -> None:
    ev, session, signal, menu = _generated()
    bau_count = sum(candidate.kind == "immediate_bau" for candidate in menu.candidates)
    with pytest.raises(PhysicalConstraintError, match="BAU"):
        assemble_customer_menu(
            ev=ev,
            session=session,
            signal=signal,
            generated_menu=menu,
            assembly_settings=MenuAssemblySettings(display_cap=max(1, bau_count - 1)),
        )


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
    assert assembled.offers
    assert all(offer.advertised_saving == 0.0 for offer in assembled.offers)


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
    offer = next(item for item in assembled.offers if item.advertised_saving > 0.0)
    duplicate_a = replace(offer, offer_id="duplicate-a")
    duplicate_z = replace(offer, offer_id="duplicate-z")
    assessments = {item.candidate_id: item for item in assembled.assessments}
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
    base = next(item for item in assembled.offers if item.advertised_saving > 0.0)
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
    base = next(item for item in assembled.offers if item.advertised_saving > 0.0)
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
    base = next(item for item in assembled.offers if item.advertised_saving > 0.0)
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
    with pytest.raises(PhysicalConstraintError, match="mandatory anchors"):
        assembly_module._select_displayed((high, low), set(), 1)
    bau = next(item for item in assembled.offers if item.advertised_saving == 0.0)
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
    assert len(assembled.offers) == 1
    assert assembled.offers[0].advertised_saving == 0.0
    assert not assembled.source_frontiers


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
