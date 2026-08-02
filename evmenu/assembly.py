"""Deterministic cross-request customer-menu assembly.

Commit 7 combines Commit 4 candidates, Commit 6 saving frontiers, and Commit 5
health scoring.  It deliberately stops before customer choice, stochastic
realization, fleet/network simulation, plotting, reporting, and file I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal, TypeVar

from .degradation import DegradationAssessment, DegradationSettings, score_generated_menu
from .exceptions import PhysicalConstraintError, SchemaValidationError
from .menu import GeneratedMenu, MenuCandidate
from .optimization import (
    FrontierSettings,
    OptimizedProfile,
    SavingFrontier,
    build_sandwich_saving_frontier,
)
from .schemas import (
    ChargingProfile,
    ChargingSession,
    EVSpec,
    MenuOffer,
    MenuSettings,
    PlanningSignal,
)
from .validation import ValidationTolerances, validate_charging_profile

SourceKind = Literal["bau", "optimized"]
_TupleItem = TypeVar("_TupleItem")


@dataclass(frozen=True, slots=True)
class OfferSource:
    """Immutable provenance for one assembled offer."""

    offer_id: str
    source_point_id: str | None
    source_candidate_id: str
    endpoint_role: str
    source_kind: SourceKind

    def __post_init__(self) -> None:
        if not isinstance(self.offer_id, str) or not self.offer_id.strip():
            raise SchemaValidationError("offer_id must be a non-empty string.")
        object.__setattr__(self, "offer_id", self.offer_id.strip())
        if self.source_point_id is not None:
            if not isinstance(self.source_point_id, str) or not self.source_point_id.strip():
                raise SchemaValidationError("source_point_id must be non-empty when supplied.")
            object.__setattr__(self, "source_point_id", self.source_point_id.strip())
        if not isinstance(self.source_candidate_id, str) or not self.source_candidate_id.strip():
            raise SchemaValidationError("source_candidate_id must be a non-empty string.")
        object.__setattr__(self, "source_candidate_id", self.source_candidate_id.strip())
        if not isinstance(self.endpoint_role, str) or not self.endpoint_role.strip():
            raise SchemaValidationError("endpoint_role must be a non-empty string.")
        object.__setattr__(self, "endpoint_role", self.endpoint_role.strip())
        if self.source_kind not in ("bau", "optimized"):
            raise SchemaValidationError("source_kind must be 'bau' or 'optimized'.")
        if self.source_kind == "bau":
            if self.source_point_id is not None or self.endpoint_role != "bau":
                raise SchemaValidationError("BAU provenance must use a null point and 'bau' role.")
        elif self.source_point_id is None or self.endpoint_role == "bau":
            raise SchemaValidationError(
                "optimized provenance must include a point ID and non-BAU endpoint role."
            )


@dataclass(frozen=True, slots=True)
class MenuAssemblySettings:
    """Deterministic controls for Commit 7 reduction and display.

    Saving tolerances are in currency units, target tolerances are SOC
    fractions, and health tolerances are score points.
    """

    pruning_saving_tolerance: float = 1e-8
    positive_saving_tolerance: float = 1e-8
    saving_merge_gap: float = 0.05
    saving_dominance_tolerance: float = 1e-8
    health_dominance_tolerance: float = 1e-8
    target_dominance_tolerance: float = 1e-8
    health_tie_tolerance: float = 1e-8
    display_cap: int = 12
    dominance_tolerance: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("pruning_saving_tolerance", self.pruning_saving_tolerance),
            ("positive_saving_tolerance", self.positive_saving_tolerance),
            ("saving_merge_gap", self.saving_merge_gap),
            ("saving_dominance_tolerance", self.saving_dominance_tolerance),
            ("health_dominance_tolerance", self.health_dominance_tolerance),
            ("target_dominance_tolerance", self.target_dominance_tolerance),
            ("health_tie_tolerance", self.health_tie_tolerance),
        ):
            numeric = _finite(name, value)
            if numeric < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")
        if self.dominance_tolerance is not None:
            alias = _finite("dominance_tolerance", self.dominance_tolerance)
            if alias < 0.0:
                raise PhysicalConstraintError("dominance_tolerance must be non-negative.")
            object.__setattr__(self, "saving_dominance_tolerance", alias)
            object.__setattr__(self, "health_dominance_tolerance", alias)
            object.__setattr__(self, "target_dominance_tolerance", alias)
        if isinstance(self.display_cap, bool) or not isinstance(self.display_cap, int):
            raise SchemaValidationError("display_cap must be an integer.")
        if self.display_cap <= 0:
            raise PhysicalConstraintError("display_cap must be positive.")


@dataclass(frozen=True, slots=True)
class AssembledMenu:
    """Final deterministic customer-facing menu for one session."""

    ev_id: str
    offers: tuple[MenuOffer, ...]
    assessments: tuple[DegradationAssessment, ...]
    source_frontiers: tuple[SavingFrontier, ...]
    source_metadata: tuple[OfferSource, ...] = ()
    display_cap: int | None = None

    @property
    def frontiers(self) -> tuple[SavingFrontier, ...]:
        """Compatibility alias for the auditable source frontiers."""
        return self.source_frontiers

    def __post_init__(self) -> None:
        if not isinstance(self.ev_id, str) or not self.ev_id.strip():
            raise SchemaValidationError("ev_id must be a non-empty string.")
        object.__setattr__(self, "ev_id", self.ev_id.strip())
        offers = _tuple_field("offers", self.offers)
        assessments = _tuple_field("assessments", self.assessments)
        frontiers = _tuple_field("source_frontiers", self.source_frontiers)
        metadata = _tuple_field("source_metadata", self.source_metadata)
        object.__setattr__(self, "offers", offers)
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "source_frontiers", frontiers)
        object.__setattr__(self, "source_metadata", metadata)
        if self.display_cap is not None:
            if isinstance(self.display_cap, bool) or not isinstance(self.display_cap, int):
                raise SchemaValidationError("display_cap must be an integer when supplied.")
            if self.display_cap <= 0:
                raise PhysicalConstraintError("display_cap must be positive.")
        if not offers or len(offers) != len(assessments):
            raise SchemaValidationError("offers and assessments must be nonempty and aligned.")
        if any(not isinstance(offer, MenuOffer) for offer in offers):
            raise SchemaValidationError("offers must contain MenuOffer objects.")
        if any(not isinstance(item, DegradationAssessment) for item in assessments):
            raise SchemaValidationError("assessments must contain DegradationAssessment objects.")
        if any(not isinstance(item, SavingFrontier) for item in frontiers):
            raise SchemaValidationError("source_frontiers must contain SavingFrontier objects.")
        if any(not isinstance(item, OfferSource) for item in metadata):
            raise SchemaValidationError("source_metadata must contain OfferSource objects.")
        if len(metadata) != len(offers):
            raise SchemaValidationError("source_metadata must align one-to-one with offers.")
        if self.display_cap is not None and len(offers) > self.display_cap:
            raise PhysicalConstraintError("assembled menu exceeds display_cap.")
        if any(offer.ev_id != self.ev_id for offer in offers):
            raise SchemaValidationError("all offers must belong to ev_id.")
        offer_ids = tuple(offer.offer_id for offer in offers)
        assessment_ids = tuple(item.candidate_id for item in assessments)
        metadata_ids = tuple(item.offer_id for item in metadata)
        if len(set(offer_ids)) != len(offer_ids):
            raise SchemaValidationError("offer IDs must be unique.")
        if len(set(assessment_ids)) != len(assessment_ids):
            raise SchemaValidationError("assessment IDs must be unique.")
        if len(set(metadata_ids)) != len(metadata_ids):
            raise SchemaValidationError("source metadata IDs must be unique.")
        if offer_ids != assessment_ids or offer_ids != metadata_ids:
            raise SchemaValidationError("offers, assessments, and source metadata must align.")
        frontier_keys: set[tuple[float, int]] = set()
        frontier_points: dict[str, tuple[SavingFrontier, OptimizedProfile]] = {}
        for frontier in frontiers:
            key = (frontier.target_soc, frontier.ready_step)
            if key in frontier_keys:
                raise SchemaValidationError("source frontier request keys must be unique.")
            frontier_keys.add(key)
            if frontier.ev_id != self.ev_id:
                raise SchemaValidationError("all source frontiers must belong to ev_id.")
            for point in frontier.points:
                if point.point_id in frontier_points:
                    raise SchemaValidationError("source frontier point IDs must be unique.")
                frontier_points[point.point_id] = (frontier, point)
        target_bau_count: dict[float, int] = {}
        metadata_by_id = {item.offer_id: item for item in metadata}
        for offer, assessment, source in zip(offers, assessments, metadata, strict=True):
            if assessment.ev_id != self.ev_id or source.offer_id != offer.offer_id:
                raise SchemaValidationError("offer, assessment, and source identities must align.")
            if source.source_kind == "bau":
                target_bau_count[offer.target_soc] = target_bau_count.get(offer.target_soc, 0) + 1
                if abs(offer.advertised_saving) > 1e-8:
                    raise SchemaValidationError("BAU advertised saving must be zero.")
            else:
                if source.source_point_id not in frontier_points:
                    raise SchemaValidationError("optimized source point is not auditable.")
                frontier, point = frontier_points[source.source_point_id]
                if (
                    frontier.target_soc != offer.target_soc
                    or frontier.ready_step != offer.ready_step
                ):
                    raise SchemaValidationError("source point request does not match offer.")
                if point.source_candidate_id != source.source_candidate_id:
                    raise SchemaValidationError("source candidate identity does not match point.")
        all_targets = set(target_bau_count) | {frontier.target_soc for frontier in frontiers}
        if any(target_bau_count.get(target, 0) != 1 for target in all_targets):
            raise SchemaValidationError("each target must have exactly one BAU offer.")
        expected_order = tuple(
            sorted(
                offers,
                key=lambda offer: _display_sort_key(
                    offer, metadata_by_id[offer.offer_id].source_kind
                ),
            )
        )
        if offers != expected_order:
            raise SchemaValidationError("offers must use deterministic display ordering.")


def prune_ready_step_change_points(
    menu: GeneratedMenu,
    *,
    positive_saving_tolerance: float = 1e-8,
    pruning_tolerance: float | None = None,
) -> tuple[MenuCandidate, ...]:
    """Retain strict maximum-saving changes and equal-saving plateau tails."""
    if not isinstance(menu, GeneratedMenu):
        raise SchemaValidationError("menu must be a GeneratedMenu.")
    positive_tolerance = _nonnegative("positive_saving_tolerance", positive_saving_tolerance)
    plateau_tolerance = (
        positive_tolerance
        if pruning_tolerance is None
        else _nonnegative("pruning_tolerance", pruning_tolerance)
    )
    groups: dict[float, list[MenuCandidate]] = {}
    for candidate in menu.candidates:
        if candidate.kind == "minimum_cost":
            groups.setdefault(candidate.target_soc, []).append(candidate)
    retained: list[MenuCandidate] = []
    for target in sorted(groups):
        candidates = sorted(groups[target], key=lambda item: (item.ready_step, item.candidate_id))
        seen_ready: set[int] = set()
        for candidate in candidates:
            if candidate.ready_step in seen_ready:
                raise SchemaValidationError(
                    f"duplicate minimum-cost ready_step={candidate.ready_step} for target={target}."
                )
            seen_ready.add(candidate.ready_step)
        positive = [candidate for candidate in candidates if candidate.saving > positive_tolerance]
        if not positive:
            continue
        kept_ids: set[str] = set()
        running_max = float("-inf")
        index = 0
        while index < len(positive):
            candidate = positive[index]
            if candidate.saving < running_max - plateau_tolerance:
                raise SchemaValidationError(
                    "minimum-cost savings must be nondecreasing within "
                    f"pruning_tolerance for target={target}, ready_step={candidate.ready_step}."
                )
            if candidate.saving > running_max + plateau_tolerance:
                kept_ids.add(candidate.candidate_id)
                running_max = candidate.saving
            end = index
            while (
                end + 1 < len(positive)
                and abs(positive[end + 1].saving - candidate.saving) <= plateau_tolerance
            ):
                end += 1
            if end > index:
                kept_ids.add(positive[end].candidate_id)
            index = end + 1
        kept_ids.add(positive[-1].candidate_id)
        retained.extend(candidate for candidate in positive if candidate.candidate_id in kept_ids)
    retained.sort(key=lambda item: (item.target_soc, item.ready_step, item.candidate_id))
    return tuple(retained)


def assemble_customer_menu(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    generated_menu: GeneratedMenu,
    menu_settings: MenuSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
    frontier_settings: FrontierSettings | None = None,
    assembly_settings: MenuAssemblySettings | None = None,
    validation_tolerances: ValidationTolerances | None = None,
) -> AssembledMenu:
    """Build a validated, compact, deterministic customer menu."""
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal.")
    if not isinstance(generated_menu, GeneratedMenu):
        raise SchemaValidationError("generated_menu must be a GeneratedMenu.")
    msettings = MenuSettings() if menu_settings is None else menu_settings
    dsettings = DegradationSettings() if degradation_settings is None else degradation_settings
    fsettings = FrontierSettings() if frontier_settings is None else frontier_settings
    asettings = MenuAssemblySettings() if assembly_settings is None else assembly_settings
    tolerances = ValidationTolerances() if validation_tolerances is None else validation_tolerances
    for name, value, expected in (
        ("menu_settings", msettings, MenuSettings),
        ("degradation_settings", dsettings, DegradationSettings),
        ("frontier_settings", fsettings, FrontierSettings),
        ("assembly_settings", asettings, MenuAssemblySettings),
        ("validation_tolerances", tolerances, ValidationTolerances),
    ):
        if not isinstance(value, expected):
            raise SchemaValidationError(f"{name} has an invalid type.")
    if generated_menu.ev_id != ev.ev_id:
        raise SchemaValidationError("generated_menu does not belong to ev.")
    session.validate_for_ev(ev)
    signal.validate_session_window(session)
    bau_by_target = _preflight_generated_menu(
        ev=ev,
        session=session,
        signal=signal,
        menu=generated_menu,
        menu_settings=msettings,
        validation_tolerances=tolerances,
    )
    if len(bau_by_target) > asettings.display_cap:
        raise PhysicalConstraintError(
            "display_cap must be at least the number of BAU offers: "
            f"cap={asettings.display_cap}, bau_count={len(bau_by_target)}."
        )

    requests = prune_ready_step_change_points(
        generated_menu,
        positive_saving_tolerance=asettings.positive_saving_tolerance,
        pruning_tolerance=asettings.pruning_saving_tolerance,
    )
    frontiers: list[SavingFrontier] = []
    synthetic: list[MenuCandidate] = [bau_by_target[target] for target in sorted(bau_by_target)]
    source_by_candidate_id: dict[str, OfferSource] = {
        candidate.candidate_id: OfferSource(
            offer_id=candidate.candidate_id,
            source_point_id=None,
            source_candidate_id=candidate.candidate_id,
            endpoint_role="bau",
            source_kind="bau",
        )
        for candidate in synthetic
    }
    for candidate in requests:
        try:
            frontier = build_sandwich_saving_frontier(
                ev=ev,
                session=session,
                signal=signal,
                candidate=candidate,
                bau_cost=candidate.same_target_bau_cost,
                degradation_settings=dsettings,
                frontier_settings=fsettings,
                tolerances=tolerances,
            )
        except Exception as exc:
            raise PhysicalConstraintError(
                "Frontier construction failed for "
                f"candidate={candidate.candidate_id}, target_soc={candidate.target_soc}, "
                f"ready_step={candidate.ready_step}."
            ) from exc
        frontiers.append(frontier)
        for point in frontier.points:
            offer_id = f"{point.point_id}-mc-r{candidate.ready_step}"
            synthetic.append(
                MenuCandidate(
                    candidate_id=offer_id,
                    ev_id=ev.ev_id,
                    kind="minimum_cost",
                    target_soc=candidate.target_soc,
                    target_sources=candidate.target_sources,
                    target_label=candidate.target_label,
                    ready_step=candidate.ready_step,
                    charging_cost=point.constructed.charging_cost,
                    same_target_bau_cost=candidate.same_target_bau_cost,
                    saving=point.saving,
                    required_grid_energy_kwh=point.constructed.required_grid_energy_kwh,
                    profile=point.constructed.profile,
                    validation=point.constructed.validation,
                )
            )
            source_by_candidate_id[offer_id] = OfferSource(
                offer_id=offer_id,
                source_point_id=point.point_id,
                source_candidate_id=point.source_candidate_id,
                endpoint_role=point.endpoint_role,
                source_kind="optimized",
            )

    scored = score_generated_menu(
        ev=ev,
        session=session,
        signal=signal,
        menu=GeneratedMenu(ev_id=ev.ev_id, candidates=tuple(synthetic)),
        degradation_settings=dsettings,
        menu_settings=msettings,
    )
    assessment_by_id = {assessment.candidate_id: assessment for assessment in scored.assessments}
    bau_ids = {candidate.candidate_id for candidate in bau_by_target.values()}
    positive_or_bau = tuple(
        offer
        for offer in scored.offers
        if offer.offer_id in bau_ids
        or offer.advertised_saving > asettings.positive_saving_tolerance
    )
    deduplicated = _remove_exact_duplicates(positive_or_bau, assessment_by_id, bau_ids)
    compacted = _compact_within_request(
        deduplicated,
        asettings.saving_merge_gap,
        asettings.health_tie_tolerance,
        bau_ids,
    )
    pareto = _pareto_filter(
        compacted,
        saving_tolerance=asettings.saving_dominance_tolerance,
        health_tolerance=asettings.health_dominance_tolerance,
        target_tolerance=asettings.target_dominance_tolerance,
        protected_ids=bau_ids,
    )
    displayed = _select_displayed(pareto, bau_ids, asettings.display_cap)
    displayed = tuple(
        sorted(
            displayed,
            key=lambda offer: _display_sort_key(
                offer, source_by_candidate_id[offer.offer_id].source_kind
            ),
        )
    )
    assessments = tuple(assessment_by_id[offer.offer_id] for offer in displayed)
    metadata = tuple(source_by_candidate_id[offer.offer_id] for offer in displayed)
    return AssembledMenu(
        ev_id=ev.ev_id,
        offers=displayed,
        assessments=assessments,
        source_frontiers=tuple(frontiers),
        source_metadata=metadata,
        display_cap=asettings.display_cap,
    )


def _preflight_generated_menu(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    menu: GeneratedMenu,
    menu_settings: MenuSettings,
    validation_tolerances: ValidationTolerances,
) -> dict[float, MenuCandidate]:
    ids: set[str] = set()
    by_target: dict[float, list[MenuCandidate]] = {}
    for candidate in menu.candidates:
        context = f"candidate={candidate.candidate_id}, target_soc={candidate.target_soc}"
        if candidate.candidate_id in ids:
            raise SchemaValidationError(f"duplicate candidate ID: {context}.")
        ids.add(candidate.candidate_id)
        if candidate.ev_id != ev.ev_id:
            raise SchemaValidationError(f"candidate EV mismatch: {context}.")
        if candidate.kind not in ("immediate_bau", "minimum_cost"):
            raise SchemaValidationError(f"unsupported candidate kind: {context}.")
        if not candidate.validation.is_valid:
            raise PhysicalConstraintError(f"candidate validation report is invalid: {context}.")
        report = validate_charging_profile(
            ev=ev,
            session=session,
            signal=signal,
            target_soc=candidate.target_soc,
            ready_step=candidate.ready_step,
            profile=candidate.profile,
            tolerances=validation_tolerances,
        )
        if not report.is_valid:
            details = "; ".join(report.errors)
            raise PhysicalConstraintError(f"candidate profile is invalid: {context}: {details}")
        direct_cost = _profile_cost(candidate.profile, signal)
        numerical_tolerance = menu_settings.numerical_tolerance
        if abs(candidate.charging_cost - direct_cost) > numerical_tolerance:
            raise SchemaValidationError(f"candidate charging cost is inconsistent: {context}.")
        if abs(candidate.saving - (candidate.same_target_bau_cost - candidate.charging_cost)) > (
            numerical_tolerance
        ):
            raise SchemaValidationError(f"candidate saving is inconsistent: {context}.")
        if abs(candidate.required_grid_energy_kwh - sum(candidate.profile.grid_energy_kwh)) > (
            numerical_tolerance
        ):
            raise SchemaValidationError(f"candidate required energy is inconsistent: {context}.")
        expected_terminal = max(
            session.initial_energy_kwh, candidate.target_soc * ev.battery_capacity_kwh
        )
        if abs(candidate.profile.battery_energy_kwh[-1] - expected_terminal) > numerical_tolerance:
            raise PhysicalConstraintError(f"candidate terminal state is inconsistent: {context}.")
        by_target.setdefault(candidate.target_soc, []).append(candidate)

    bau_by_target: dict[float, MenuCandidate] = {}
    request_keys: set[tuple[float, int, str]] = set()
    for target, candidates in by_target.items():
        provenance = (candidates[0].target_sources, candidates[0].target_label)
        for candidate in candidates:
            if (candidate.target_sources, candidate.target_label) != provenance:
                raise SchemaValidationError(
                    f"target provenance is inconsistent: target_soc={target}."
                )
            if candidate.kind == "immediate_bau" and target in bau_by_target:
                raise SchemaValidationError(
                    f"duplicate BAU candidate: target_soc={target}, candidate={candidate.candidate_id}."
                )
            key = (candidate.target_soc, candidate.ready_step, candidate.kind)
            if key in request_keys:
                raise SchemaValidationError(
                    "duplicate candidate request key: "
                    f"target_soc={target}, ready_step={candidate.ready_step}, kind={candidate.kind}."
                )
            request_keys.add(key)
            if candidate.kind == "immediate_bau":
                if abs(candidate.saving) > menu_settings.numerical_tolerance:
                    raise SchemaValidationError(
                        f"BAU saving must be zero: candidate={candidate.candidate_id}."
                    )
                if abs(candidate.charging_cost - candidate.same_target_bau_cost) > (
                    menu_settings.numerical_tolerance
                ):
                    raise SchemaValidationError(
                        f"BAU cost is inconsistent: candidate={candidate.candidate_id}."
                    )
                bau_by_target[target] = candidate
    if not bau_by_target:
        raise PhysicalConstraintError("generated_menu must contain BAU candidates.")
    for target, candidates in by_target.items():
        if target not in bau_by_target:
            raise PhysicalConstraintError(f"target_soc={target} is missing its BAU candidate.")
        bau_cost = bau_by_target[target].charging_cost
        for candidate in candidates:
            if abs(candidate.same_target_bau_cost - bau_cost) > menu_settings.numerical_tolerance:
                raise SchemaValidationError(
                    f"candidate BAU cost does not match target baseline: candidate={candidate.candidate_id}."
                )
    return bau_by_target


def _remove_exact_duplicates(
    offers: tuple[MenuOffer, ...],
    assessment_by_id: dict[str, DegradationAssessment],
    protected_ids: set[str],
) -> tuple[MenuOffer, ...]:
    groups: dict[tuple[object, ...], list[MenuOffer]] = {}
    retained: list[MenuOffer] = []
    for offer in offers:
        if offer.offer_id in protected_ids:
            retained.append(offer)
            continue
        assessment = assessment_by_id[offer.offer_id]
        key = (
            offer.target_soc,
            offer.ready_step,
            offer.advertised_saving,
            offer.charging_health_score,
            offer.charging_cost,
            offer.profile,
            _assessment_value_key(assessment),
        )
        groups.setdefault(key, []).append(offer)
    retained.extend(min(group, key=lambda item: item.offer_id) for group in groups.values())
    return tuple(sorted(retained, key=lambda item: item.offer_id))


def _assessment_value_key(assessment: DegradationAssessment) -> tuple[object, ...]:
    return (
        assessment.ev_id,
        assessment.chemistry,
        assessment.charging_window_calendar_fade,
        assessment.parked_day_calendar_fade,
        assessment.cycle_fade,
        assessment.total_fade,
        assessment.annualized_degradation_pct,
        assessment.parked_soc,
        assessment.peak_c_rate,
    )


def _compact_within_request(
    offers: tuple[MenuOffer, ...],
    gap: float,
    health_tie_tolerance: float,
    protected_ids: set[str],
) -> tuple[MenuOffer, ...]:
    retained: list[MenuOffer] = []
    groups: dict[tuple[float, int], list[MenuOffer]] = {}
    for offer in offers:
        if offer.offer_id in protected_ids:
            retained.append(offer)
        else:
            groups.setdefault((offer.target_soc, offer.ready_step), []).append(offer)
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda item: (item.advertised_saving, item.offer_id))
        cluster: list[MenuOffer] = []
        for offer in ordered:
            if not cluster or offer.advertised_saving - cluster[-1].advertised_saving < gap:
                cluster.append(offer)
            else:
                retained.append(_best_cluster_offer(cluster, health_tie_tolerance))
                cluster = [offer]
        if cluster:
            retained.append(_best_cluster_offer(cluster, health_tie_tolerance))
    return tuple(sorted(retained, key=lambda item: item.offer_id))


def _best_cluster_offer(cluster: list[MenuOffer], health_tie_tolerance: float) -> MenuOffer:
    highest_health = max(item.charging_health_score for item in cluster)
    health_tied = [
        item
        for item in cluster
        if highest_health - item.charging_health_score <= health_tie_tolerance
    ]
    highest_saving = max(item.advertised_saving for item in health_tied)
    saving_tied = [item for item in health_tied if item.advertised_saving == highest_saving]
    return min(saving_tied, key=lambda item: item.offer_id)


def _pareto_filter(
    offers: tuple[MenuOffer, ...],
    *,
    saving_tolerance: float,
    health_tolerance: float,
    target_tolerance: float,
    protected_ids: set[str],
) -> tuple[MenuOffer, ...]:
    ordered = tuple(sorted(offers, key=lambda item: item.offer_id))
    retained: list[MenuOffer] = []
    for offer in ordered:
        if offer.offer_id in protected_ids:
            retained.append(offer)
            continue
        if not any(
            other.offer_id != offer.offer_id
            and _dominates(
                other,
                offer,
                saving_tolerance=saving_tolerance,
                health_tolerance=health_tolerance,
                target_tolerance=target_tolerance,
            )
            for other in ordered
        ):
            retained.append(offer)
    return tuple(retained)


def _dominates(
    a: MenuOffer,
    b: MenuOffer,
    *,
    saving_tolerance: float,
    health_tolerance: float,
    target_tolerance: float,
) -> bool:
    weak = (
        a.ready_step <= b.ready_step
        and a.target_soc >= b.target_soc - target_tolerance
        and a.advertised_saving >= b.advertised_saving - saving_tolerance
        and a.charging_health_score >= b.charging_health_score - health_tolerance
    )
    strict = (
        a.ready_step < b.ready_step
        or a.target_soc > b.target_soc + target_tolerance
        or a.advertised_saving > b.advertised_saving + saving_tolerance
        or a.charging_health_score > b.charging_health_score + health_tolerance
    )
    return weak and strict


def _select_displayed(
    offers: tuple[MenuOffer, ...], protected_ids: set[str], cap: int
) -> tuple[MenuOffer, ...]:
    if len(protected_ids) > cap:
        raise PhysicalConstraintError(f"display_cap={cap} is below BAU count={len(protected_ids)}.")
    if len(offers) <= cap:
        return offers
    mandatory: dict[str, MenuOffer] = {
        offer.offer_id: offer
        for offer in sorted(offers, key=lambda item: item.offer_id)
        if offer.offer_id in protected_ids
    }
    non_bau = [offer for offer in offers if offer.offer_id not in protected_ids]
    if non_bau:
        max_saving = min(
            non_bau,
            key=lambda item: (
                -item.advertised_saving,
                -item.charging_health_score,
                item.ready_step,
                -item.target_soc,
                item.offer_id,
            ),
        )
        max_health = min(
            non_bau,
            key=lambda item: (
                -item.charging_health_score,
                -item.advertised_saving,
                item.ready_step,
                -item.target_soc,
                item.offer_id,
            ),
        )
        mandatory[max_saving.offer_id] = max_saving
        mandatory[max_health.offer_id] = max_health
        if len(mandatory) > cap:
            raise PhysicalConstraintError(
                "display cap cannot retain mandatory anchors: "
                f"cap={cap}, bau_count={len(protected_ids)}, "
                f"mandatory_count={len(mandatory)}."
            )
        for ready in sorted({offer.ready_step for offer in non_bau}):
            if len(mandatory) >= cap:
                break
            group = [offer for offer in non_bau if offer.ready_step == ready]
            best = min(
                group,
                key=lambda item: (
                    -item.advertised_saving,
                    -item.charging_health_score,
                    -item.target_soc,
                    item.offer_id,
                ),
            )
            mandatory.setdefault(best.offer_id, best)
    selected = dict(mandatory)
    remaining = [offer for offer in offers if offer.offer_id not in selected]
    remaining.sort(
        key=lambda item: (
            -item.advertised_saving,
            -item.charging_health_score,
            item.ready_step,
            -item.target_soc,
            item.offer_id,
        )
    )
    for offer in remaining:
        if len(selected) >= cap:
            break
        selected[offer.offer_id] = offer
    return tuple(selected.values())


def _display_sort_key(offer: MenuOffer, source_kind: str | None = None) -> tuple[object, ...]:
    is_bau = source_kind == "bau" or (source_kind is None and abs(offer.advertised_saving) <= 1e-8)
    return (
        offer.target_soc,
        0 if is_bau else 1,
        offer.ready_step,
        offer.advertised_saving,
        -offer.charging_health_score,
        offer.offer_id,
    )


def _profile_cost(profile: ChargingProfile, signal: PlanningSignal) -> float:
    try:
        return float(
            sum(
                signal.price_per_kwh[profile.start_step + index] * energy
                for index, energy in enumerate(profile.grid_energy_kwh)
            )
        )
    except (AttributeError, IndexError, TypeError) as exc:
        raise SchemaValidationError(
            "candidate profile cannot be priced in the supplied signal."
        ) from exc


def _tuple_field(name: str, value: Iterable[_TupleItem] | None) -> tuple[_TupleItem, ...]:
    if value is None:
        raise SchemaValidationError(f"{name} cannot be None.")
    try:
        return tuple(value)
    except TypeError as exc:
        raise SchemaValidationError(f"{name} must be iterable.") from exc


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)


def _nonnegative(name: str, value: object) -> float:
    numeric = _finite(name, value)
    if numeric < 0.0:
        raise PhysicalConstraintError(f"{name} must be non-negative.")
    return numeric
