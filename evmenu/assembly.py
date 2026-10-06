"""Deterministic cross-request customer-menu assembly.

The assembly layer combines validated Algorithm 1 candidates, saving frontiers,
raw battery-stress scoring, paper filtering, and deterministic final display
diversity. It deliberately stops before customer choice, stochastic
realization, fleet/network simulation, plotting, reporting, and file I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from math import isfinite, log, sqrt
from numbers import Real
from statistics import median
from typing import Literal, TypeVar

from .degradation import (
    BATTERY_METRIC_COMPARISON_SCOPE,
    BATTERY_METRIC_MODEL_ID,
    DegradationAssessment,
    DegradationSettings,
    score_generated_menu,
)
from .exceptions import PhysicalConstraintError, SchemaValidationError
from .menu import GeneratedMenu, MenuCandidate, RequestGenerationDiagnostics
from .optimization import (
    DegradationObjectiveDiagnostics,
    FailureReason,
    FrontierSettings,
    OptimizationDiagnostics,
    OptimizedProfile,
    SavingFrontier,
    SavingLevelFailure,
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
MenuStage = Literal["generated", "compacted", "pareto", "displayed"]
SelectionReason = Literal[
    "bau_anchor",
    "target_maximum_saving",
    "target_least_degradation",
    "intermediate_diversity",
    "close_ready_saving_stress_tradeoff",
    "target_positive_saving_coverage",
    "per_target_limit",
    "global_limit",
    "too_close_in_ready_time",
    "insufficient_saving_difference",
    "insufficient_stress_difference",
    "lower_diversity_contribution",
    "target_coverage_anchor",
    "maximum_saving_anchor",
    "least_degradation_anchor",
    "farthest_point_diversity",
    "anchor_suppressed_similarity",
]
_TupleItem = TypeVar("_TupleItem")


def _saving_band_lower(requested_saving: float, band: float) -> float:
    """Lower edge of an offer's saving band, floored at zero.

    The band is +/- ``band`` around the saving that was requested of the
    optimizer. When the requested saving is smaller than the band -- which
    ``select_saving_levels`` produces routinely, because it always includes the
    exact maximum saving however small that is -- the raw lower edge goes
    negative. ``MenuOffer`` rejects a negative ``saving_band_lower`` with a
    ``PhysicalConstraintError``, which aborted the ENTIRE menu over one offer.

    Flooring at zero is the physically correct edge, not a workaround: a non-BAU
    offer is only retained when its saving is positive, so no realizable saving
    lies below zero and the unreachable part of the band carries no information.
    The violation metric uses this same floored edge so the reported bound and
    the violation measured against it stay consistent.
    """
    return max(0.0, requested_saving - band)


def _flags_for_role(role: str) -> tuple[str, ...]:
    """Canonical provenance flags shared by assembly and serialization."""
    flags = {"is_bau"} if role == "bau" else set()
    for name in ("low_saving", "intermediate", "maximum_saving", "least_degradation"):
        if role == name:
            flags.add(f"is_{name}")
    if role in ("maximum_saving", "least_degradation", "least_and_maximum"):
        flags.add("is_endpoint")
    if role == "least_and_maximum":
        flags.update(("is_least_and_maximum", "is_least_degradation", "is_maximum_saving"))
    return tuple(sorted(flags))


@dataclass(frozen=True, slots=True)
class OfferSource:
    """Immutable provenance for one assembled offer."""

    offer_id: str
    source_point_id: str | None
    source_candidate_id: str
    endpoint_role: str
    source_kind: SourceKind
    provenance_flags: tuple[str, ...] = ()
    saving_provenance: tuple[float, ...] = ()

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
        flags = tuple(self.provenance_flags)
        if any(not isinstance(flag, str) or not flag.strip() for flag in flags):
            raise SchemaValidationError("provenance_flags must contain non-empty strings.")
        if len(set(flags)) != len(flags):
            raise SchemaValidationError("provenance_flags must be unique.")
        object.__setattr__(self, "provenance_flags", tuple(sorted(flags)))
        savings = tuple(float(value) for value in self.saving_provenance)
        if any(not isfinite(value) for value in savings):
            raise SchemaValidationError("saving_provenance must contain finite values.")
        object.__setattr__(self, "saving_provenance", savings)


@dataclass(frozen=True, slots=True)
class TargetGenerationSummary:
    """Completeness summary for one target SOC."""

    target_soc: float
    bau_ready: int
    request_count: int
    positive_saving_request_count: int
    selected_saving_level_count: int
    optimization_success_count: int
    generated_offer_count: int
    duplicate_count: int
    roles_present: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= _finite("target_soc", self.target_soc) <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        if (
            isinstance(self.bau_ready, bool)
            or not isinstance(self.bau_ready, int)
            or self.bau_ready < 0
        ):
            raise SchemaValidationError("bau_ready must be a non-negative integer.")
        counts = (
            self.request_count,
            self.positive_saving_request_count,
            self.selected_saving_level_count,
            self.optimization_success_count,
            self.generated_offer_count,
            self.duplicate_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise SchemaValidationError("target summary counts must be non-negative integers.")
        if self.positive_saving_request_count > self.request_count:
            raise SchemaValidationError("positive request count cannot exceed request count.")
        roles = tuple(self.roles_present)
        if any(not isinstance(role, str) or not role.strip() for role in roles):
            raise SchemaValidationError("roles_present must contain non-empty strings.")
        object.__setattr__(self, "roles_present", tuple(sorted(set(roles))))


@dataclass(frozen=True, slots=True)
class CompactionGroupSummary:
    """Immutable audit record for one fixed ready-by/target request group."""

    ready_absolute_minute: int
    target_soc: float
    input_offer_count: int
    output_offer_count: int
    saving_min: float
    saving_max: float
    representative_offer_ids: tuple[str, ...]
    removed_offer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.ready_absolute_minute, bool) or not isinstance(
            self.ready_absolute_minute, int
        ):
            raise SchemaValidationError("ready_absolute_minute must be an integer.")
        if not 0.0 <= _finite("target_soc", self.target_soc) <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        for name, value in (
            ("input_offer_count", self.input_offer_count),
            ("output_offer_count", self.output_offer_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        if self.output_offer_count > self.input_offer_count:
            raise SchemaValidationError("compaction output cannot exceed input.")
        low = _finite("saving_min", self.saving_min)
        high = _finite("saving_max", self.saving_max)
        if low > high:
            raise SchemaValidationError("saving_min cannot exceed saving_max.")
        reps = tuple(self.representative_offer_ids)
        removed = tuple(self.removed_offer_ids)
        if any(not isinstance(item, str) or not item.strip() for item in reps + removed):
            raise SchemaValidationError("compaction IDs must be non-empty strings.")
        if len(set(reps)) != len(reps) or len(set(removed)) != len(removed):
            raise SchemaValidationError("compaction IDs must be unique.")
        if set(reps) & set(removed):
            raise SchemaValidationError("compaction representatives cannot be removed.")
        object.__setattr__(self, "representative_offer_ids", tuple(sorted(reps)))
        object.__setattr__(self, "removed_offer_ids", tuple(sorted(removed)))


@dataclass(frozen=True, slots=True)
class CompactionDecision:
    """Immutable explanation for one saving-region representative."""

    retained_offer_id: str
    removed_offer_ids: tuple[str, ...]
    saving_interval_lower: float
    saving_interval_upper: float
    reason: Literal[
        "same_saving_interval_higher_stress",
        "same_saving_interval_endpoint_preserved",
        "same_customer_value_representative",
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.retained_offer_id, str) or not self.retained_offer_id.strip():
            raise SchemaValidationError("retained_offer_id must be non-empty.")
        removed = tuple(self.removed_offer_ids)
        if any(not isinstance(item, str) or not item.strip() for item in removed):
            raise SchemaValidationError("removed_offer_ids must contain non-empty strings.")
        if self.retained_offer_id in removed:
            raise SchemaValidationError("retained offer cannot be listed as removed.")
        lower = _finite("saving_interval_lower", self.saving_interval_lower)
        upper = _finite("saving_interval_upper", self.saving_interval_upper)
        if lower > upper:
            raise SchemaValidationError("saving interval lower bound cannot exceed upper bound.")
        if self.reason not in (
            "same_saving_interval_higher_stress",
            "same_saving_interval_endpoint_preserved",
            "same_customer_value_representative",
        ):
            raise SchemaValidationError("unsupported compaction decision reason.")
        object.__setattr__(self, "removed_offer_ids", tuple(sorted(set(removed))))


@dataclass(frozen=True, slots=True)
class PipelineTargetSummary:
    """Per-target counts and roles across all derived pipeline stages."""

    target_soc: float
    generated_offer_count: int
    retained_offer_count: int
    compacted_offer_count: int
    pareto_efficient_offer_count: int
    preserved_bau_anchor_count: int
    pareto_offer_count: int
    displayed_offer_count: int
    roles_present: tuple[str, ...]
    generated_roles: tuple[str, ...] = ()
    retained_roles: tuple[str, ...] = ()
    compacted_roles: tuple[str, ...] = ()
    pareto_efficient_roles: tuple[str, ...] = ()
    preserved_bau_anchor_roles: tuple[str, ...] = ()
    pareto_roles: tuple[str, ...] = ()
    displayed_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= _finite("target_soc", self.target_soc) <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        for name in (
            "generated_offer_count",
            "retained_offer_count",
            "compacted_offer_count",
            "pareto_efficient_offer_count",
            "preserved_bau_anchor_count",
            "pareto_offer_count",
            "displayed_offer_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        role_fields = (
            "roles_present",
            "generated_roles",
            "retained_roles",
            "compacted_roles",
            "pareto_efficient_roles",
            "preserved_bau_anchor_roles",
            "pareto_roles",
            "displayed_roles",
        )
        for name in role_fields:
            roles = tuple(getattr(self, name))
            if any(not isinstance(role, str) or not role.strip() for role in roles):
                raise SchemaValidationError(f"{name} must contain non-empty strings.")
            object.__setattr__(self, name, tuple(sorted(set(roles))))


@dataclass(frozen=True, slots=True)
class PipelineDiagnostics:
    """Immutable counts and audit references for all derived menu stages."""

    generated_offer_count: int = 0
    retained_offer_count: int = 0
    compacted_offer_count: int = 0
    pareto_efficient_offer_count: int = 0
    pareto_offer_count: int = 0
    preserved_bau_anchor_count: int = 0
    displayed_offer_count: int = 0
    nonpositive_removed_count: int = 0
    compaction_removed_count: int = 0
    pareto_dominated_removed_count: int = 0
    display_selection_removed_count: int = 0
    compaction_groups: tuple[CompactionGroupSummary, ...] = ()
    compaction_decisions: tuple[CompactionDecision, ...] = ()
    target_summaries: tuple[PipelineTargetSummary, ...] = ()
    pareto_dominated_offer_ids: tuple[str, ...] = ()
    pareto_dominated_bau_anchor_ids: tuple[str, ...] = ()
    pareto_dominance_pairs: tuple[tuple[str, str], ...] = ()
    display_selection_summary: DisplaySelectionSummary | None = None
    target_display_summaries: tuple[TargetDisplaySelectionSummary, ...] = ()
    display_selection_decisions: tuple[DisplaySelectionDecision, ...] = ()
    distinctness_diagnostics: MenuDistinctnessDiagnostics | None = None

    def __post_init__(self) -> None:
        names = (
            "generated_offer_count",
            "retained_offer_count",
            "compacted_offer_count",
            "pareto_efficient_offer_count",
            "pareto_offer_count",
            "preserved_bau_anchor_count",
            "displayed_offer_count",
            "nonpositive_removed_count",
            "compaction_removed_count",
            "pareto_dominated_removed_count",
            "display_selection_removed_count",
        )
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        if self.pareto_efficient_offer_count > self.compacted_offer_count:
            raise SchemaValidationError("Pareto-efficient count cannot exceed compacted count.")
        if self.pareto_offer_count < self.pareto_efficient_offer_count:
            raise SchemaValidationError("Pareto stage cannot lose efficient offers.")
        groups = tuple(self.compaction_groups)
        decisions = tuple(self.compaction_decisions)
        if any(not isinstance(item, CompactionGroupSummary) for item in groups):
            raise SchemaValidationError("compaction_groups contains an invalid item.")
        if any(not isinstance(item, CompactionDecision) for item in decisions):
            raise SchemaValidationError("compaction_decisions contains an invalid item.")
        target_summaries = tuple(self.target_summaries)
        if any(not isinstance(item, PipelineTargetSummary) for item in target_summaries):
            raise SchemaValidationError("target_summaries contains an invalid item.")
        object.__setattr__(self, "compaction_groups", groups)
        object.__setattr__(self, "compaction_decisions", decisions)
        object.__setattr__(self, "target_summaries", target_summaries)
        if self.display_selection_summary is not None and not isinstance(
            self.display_selection_summary, DisplaySelectionSummary
        ):
            raise SchemaValidationError("display_selection_summary has an invalid type.")
        target_display_summaries = tuple(self.target_display_summaries)
        if any(
            not isinstance(item, TargetDisplaySelectionSummary) for item in target_display_summaries
        ):
            raise SchemaValidationError("target_display_summaries contains an invalid item.")
        display_selection_decisions = tuple(self.display_selection_decisions)
        if any(
            not isinstance(item, DisplaySelectionDecision) for item in display_selection_decisions
        ):
            raise SchemaValidationError("display_selection_decisions contains an invalid item.")
        if self.distinctness_diagnostics is not None and not isinstance(
            self.distinctness_diagnostics, MenuDistinctnessDiagnostics
        ):
            raise SchemaValidationError("distinctness_diagnostics has an invalid type.")
        object.__setattr__(self, "target_display_summaries", target_display_summaries)
        object.__setattr__(self, "display_selection_decisions", display_selection_decisions)
        dominated = tuple(self.pareto_dominated_offer_ids)
        dominated_bau = tuple(self.pareto_dominated_bau_anchor_ids)
        pairs = tuple(self.pareto_dominance_pairs)
        if any(not isinstance(item, str) or not item.strip() for item in dominated):
            raise SchemaValidationError("pareto_dominated_offer_ids contains an invalid item.")
        if any(not isinstance(item, str) or not item.strip() for item in dominated_bau):
            raise SchemaValidationError("pareto_dominated_bau_anchor_ids contains an invalid item.")
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(item, str) or not item.strip() for item in pair)
            for pair in pairs
        ):
            raise SchemaValidationError("pareto_dominance_pairs contains an invalid item.")
        object.__setattr__(self, "pareto_dominated_offer_ids", tuple(sorted(set(dominated))))
        object.__setattr__(
            self,
            "pareto_dominated_bau_anchor_ids",
            tuple(sorted(set(dominated_bau))),
        )
        object.__setattr__(self, "pareto_dominance_pairs", tuple(sorted(set(pairs))))


@dataclass(frozen=True, slots=True)
class DistinctnessParameters:
    """Immutable feature-space controls for display distinctness diagnostics."""

    ready_weight: float = 1.0
    target_weight: float = 1.0
    saving_weight: float = 1.0
    fade_weight: float = 1.0
    minimum_overall_distinctness: float = 0.0
    distinctness_warning_threshold: float = 0.15
    minimum_meaningful_saving_difference: float = 5.0
    minimum_meaningful_fade_difference: float = 0.0
    minimum_meaningful_ready_difference_minutes: int = 120
    minimum_meaningful_target_difference: float = 0.01
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name, value in (
            ("ready_weight", self.ready_weight),
            ("target_weight", self.target_weight),
            ("saving_weight", self.saving_weight),
            ("fade_weight", self.fade_weight),
            ("minimum_overall_distinctness", self.minimum_overall_distinctness),
            ("distinctness_warning_threshold", self.distinctness_warning_threshold),
            ("minimum_meaningful_saving_difference", self.minimum_meaningful_saving_difference),
            ("minimum_meaningful_fade_difference", self.minimum_meaningful_fade_difference),
            ("minimum_meaningful_target_difference", self.minimum_meaningful_target_difference),
            ("epsilon", self.epsilon),
        ):
            numeric = _finite(name, value)
            if numeric < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")
        if self.epsilon <= 0.0:
            raise PhysicalConstraintError("epsilon must be positive.")
        if (
            isinstance(self.minimum_meaningful_ready_difference_minutes, bool)
            or not isinstance(self.minimum_meaningful_ready_difference_minutes, int)
            or self.minimum_meaningful_ready_difference_minutes < 0
        ):
            raise SchemaValidationError(
                "minimum_meaningful_ready_difference_minutes must be non-negative."
            )


@dataclass(frozen=True, slots=True)
class PairwiseDistinctness:
    """Auditable pairwise distances in the display feature space."""

    offer_a: str
    offer_b: str
    ready_distance: float
    target_distance: float
    saving_distance: float
    fade_distance: float
    overall_distance: float

    def __post_init__(self) -> None:
        for name in ("offer_a", "offer_b"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be non-empty.")
        if self.offer_a == self.offer_b:
            raise SchemaValidationError(
                "pairwise distinctness cannot compare an offer with itself."
            )
        for name in (
            "ready_distance",
            "target_distance",
            "saving_distance",
            "fade_distance",
            "overall_distance",
        ):
            if _finite(name, getattr(self, name)) < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class OfferDistinctnessDiagnostics:
    """Nearest-neighbour and selection diagnostics for one displayed offer."""

    offer_id: str
    nearest_offer_id: str | None
    nearest_offer_distance: float | None
    nearest_ready_distance: float | None
    nearest_target_distance: float | None
    nearest_saving_distance: float | None
    nearest_fade_distance: float | None
    marginal_diversity_contribution: float | None
    selection_order: int | None
    selection_reason: str
    distinctness_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.offer_id, str) or not self.offer_id.strip():
            raise SchemaValidationError("offer_id must be non-empty.")
        if self.nearest_offer_id is not None and (
            not isinstance(self.nearest_offer_id, str) or not self.nearest_offer_id.strip()
        ):
            raise SchemaValidationError("nearest_offer_id must be non-empty when supplied.")
        for name in (
            "nearest_offer_distance",
            "nearest_ready_distance",
            "nearest_target_distance",
            "nearest_saving_distance",
            "nearest_fade_distance",
            "marginal_diversity_contribution",
        ):
            value = getattr(self, name)
            if value is not None and _finite(name, value) < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")
        if self.selection_order is not None and (
            isinstance(self.selection_order, bool)
            or not isinstance(self.selection_order, int)
            or self.selection_order < 1
        ):
            raise SchemaValidationError("selection_order must be positive when supplied.")
        if self.distinctness_label not in (
            "highly_distinct",
            "moderately_distinct",
            "weakly_distinct",
            "near_duplicate",
        ):
            raise SchemaValidationError("unsupported distinctness label.")


@dataclass(frozen=True, slots=True)
class MenuDistinctnessDiagnostics:
    """Complete immutable menu-level distinctness report."""

    feature_definitions: tuple[tuple[str, str], ...]
    pairwise_metrics: tuple[PairwiseDistinctness, ...]
    minimum_pairwise_distance: float | None
    mean_pairwise_distance: float | None
    median_pairwise_distance: float | None
    mean_nearest_neighbour_distance: float | None
    minimum_nearest_neighbour_distance: float | None
    ready_range: tuple[float, float] | None
    target_range: tuple[float, float] | None
    saving_range: tuple[float, float] | None
    fade_range: tuple[float, float] | None
    offers_per_target: tuple[tuple[float, int], ...]
    target_coverage_count: int
    target_share_entropy: float
    near_duplicate_pair_count: int
    warning_threshold: float
    option_metrics: tuple[OfferDistinctnessDiagnostics, ...]
    closest_pairs: tuple[PairwiseDistinctness, ...]
    most_distinct_pairs: tuple[PairwiseDistinctness, ...]

    def __post_init__(self) -> None:
        if self.target_coverage_count < 0 or isinstance(self.target_coverage_count, bool):
            raise SchemaValidationError("target_coverage_count must be non-negative.")
        if self.near_duplicate_pair_count < 0 or isinstance(self.near_duplicate_pair_count, bool):
            raise SchemaValidationError("near_duplicate_pair_count must be non-negative.")
        if _finite("target_share_entropy", self.target_share_entropy) < 0.0:
            raise PhysicalConstraintError("target_share_entropy must be non-negative.")
        if _finite("warning_threshold", self.warning_threshold) < 0.0:
            raise PhysicalConstraintError("warning_threshold must be non-negative.")
        for name in (
            "minimum_pairwise_distance",
            "mean_pairwise_distance",
            "median_pairwise_distance",
            "mean_nearest_neighbour_distance",
            "minimum_nearest_neighbour_distance",
        ):
            value = getattr(self, name)
            if value is not None and _finite(name, value) < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class DisplayDiversityParameters:
    """Immutable controls for the final customer-facing diversity stage."""

    maximum_displayed_offers: int | None = 12
    # Safety cap only; global diversity selection determines allocation.
    maximum_offers_per_target: int = 12
    minimum_ready_separation_minutes: int = 120
    minimum_saving_difference: float = 5.0
    minimum_saving_fraction_of_bau: float = 0.05
    minimum_relative_stress_difference: float = 0.02
    distinctness: DistinctnessParameters = field(default_factory=DistinctnessParameters)

    def __post_init__(self) -> None:
        if self.maximum_displayed_offers is not None and (
            isinstance(self.maximum_displayed_offers, bool)
            or not isinstance(self.maximum_displayed_offers, int)
            or self.maximum_displayed_offers < 1
        ):
            raise SchemaValidationError(
                "maximum_displayed_offers must be None or a positive integer."
            )
        for name, value in (
            ("maximum_offers_per_target", self.maximum_offers_per_target),
            ("minimum_ready_separation_minutes", self.minimum_ready_separation_minutes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        if self.maximum_offers_per_target < 1:
            raise PhysicalConstraintError("maximum_offers_per_target must be at least one.")
        if not isinstance(self.distinctness, DistinctnessParameters):
            raise SchemaValidationError("distinctness has an invalid type.")
        numeric_parameters: tuple[tuple[str, float], ...] = (
            ("minimum_saving_difference", self.minimum_saving_difference),
            ("minimum_saving_fraction_of_bau", self.minimum_saving_fraction_of_bau),
            ("minimum_relative_stress_difference", self.minimum_relative_stress_difference),
        )
        for numeric_name, numeric_input in numeric_parameters:
            validated_value: float = _finite(numeric_name, numeric_input)
            if validated_value < 0.0:
                raise PhysicalConstraintError(f"{numeric_name} must be non-negative.")
            if numeric_name == "minimum_relative_stress_difference" and validated_value > 1.0:
                raise PhysicalConstraintError(
                    "minimum_relative_stress_difference must lie in [0, 1]."
                )
            if numeric_name == "minimum_saving_fraction_of_bau" and validated_value > 1.0:
                raise PhysicalConstraintError("minimum_saving_fraction_of_bau must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class DisplaySelectionSummary:
    """Immutable aggregate audit for the final diversity selection."""

    pareto_input_count: int
    target_count: int
    bau_anchor_count: int
    positive_target_coverage_count: int
    selected_count: int
    removed_count: int
    minimum_ready_separation_minutes: int
    maximum_offers_per_target: int
    maximum_displayed_offers: int | None

    def __post_init__(self) -> None:
        for name in (
            "pareto_input_count",
            "target_count",
            "bau_anchor_count",
            "positive_target_coverage_count",
            "selected_count",
            "removed_count",
            "minimum_ready_separation_minutes",
            "maximum_offers_per_target",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        if self.maximum_offers_per_target < 1:
            raise SchemaValidationError("maximum_offers_per_target must be positive.")
        if self.maximum_displayed_offers is not None and (
            isinstance(self.maximum_displayed_offers, bool)
            or not isinstance(self.maximum_displayed_offers, int)
            or self.maximum_displayed_offers < 1
        ):
            raise SchemaValidationError("maximum_displayed_offers must be None or positive.")


@dataclass(frozen=True, slots=True)
class TargetDisplaySelectionSummary:
    """Immutable per-target diversity-selection audit."""

    target_soc: float
    target_energy_kwh: float
    input_pareto_count: int
    bau_count: int
    selected_count: int
    selected_offer_ids: tuple[str, ...]
    selected_ready_times: tuple[int, ...]
    roles_present: tuple[str, ...]
    minimum_pairwise_ready_separation: int | None

    def __post_init__(self) -> None:
        if not 0.0 <= _finite("target_soc", self.target_soc) <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        if _finite("target_energy_kwh", self.target_energy_kwh) < 0.0:
            raise PhysicalConstraintError("target_energy_kwh must be non-negative.")
        for name in ("input_pareto_count", "bau_count", "selected_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        ids = tuple(self.selected_offer_ids)
        ready = tuple(self.selected_ready_times)
        roles = tuple(self.roles_present)
        if len(ids) != len(ready):
            raise SchemaValidationError("selected IDs and ready times must align.")
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise SchemaValidationError("selected_offer_ids must contain non-empty strings.")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in ready):
            raise SchemaValidationError("selected_ready_times must contain integers.")
        if any(not isinstance(item, str) or not item.strip() for item in roles):
            raise SchemaValidationError("roles_present must contain non-empty strings.")
        if self.minimum_pairwise_ready_separation is not None and (
            isinstance(self.minimum_pairwise_ready_separation, bool)
            or not isinstance(self.minimum_pairwise_ready_separation, int)
            or self.minimum_pairwise_ready_separation < 0
        ):
            raise SchemaValidationError("minimum_pairwise_ready_separation must be non-negative.")
        object.__setattr__(self, "selected_offer_ids", ids)
        object.__setattr__(self, "selected_ready_times", ready)
        object.__setattr__(self, "roles_present", tuple(sorted(set(roles))))


@dataclass(frozen=True, slots=True)
class DisplaySelectionDecision:
    """Immutable decision record for each Pareto-stage offer."""

    offer_id: str
    selected: bool
    reason: SelectionReason
    nearest_selected_offer_id: str | None = None
    ready_difference_minutes: int | None = None
    saving_difference: float | None = None
    relative_stress_difference: float | None = None
    diversity_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.offer_id, str) or not self.offer_id.strip():
            raise SchemaValidationError("offer_id must be non-empty.")
        if not isinstance(self.selected, bool):
            raise SchemaValidationError("selected must be bool.")
        allowed = {
            "bau_anchor",
            "target_maximum_saving",
            "target_least_degradation",
            "intermediate_diversity",
            "close_ready_saving_stress_tradeoff",
            "target_positive_saving_coverage",
            "per_target_limit",
            "global_limit",
            "too_close_in_ready_time",
            "insufficient_saving_difference",
            "insufficient_stress_difference",
            "lower_diversity_contribution",
            "target_coverage_anchor",
            "maximum_saving_anchor",
            "least_degradation_anchor",
            "farthest_point_diversity",
            "anchor_suppressed_similarity",
        }
        if self.reason not in allowed:
            raise SchemaValidationError("unsupported display selection reason.")
        if self.nearest_selected_offer_id is not None and (
            not isinstance(self.nearest_selected_offer_id, str)
            or not self.nearest_selected_offer_id.strip()
        ):
            raise SchemaValidationError(
                "nearest_selected_offer_id must be non-empty when supplied."
            )
        if self.ready_difference_minutes is not None and (
            isinstance(self.ready_difference_minutes, bool)
            or not isinstance(self.ready_difference_minutes, int)
            or self.ready_difference_minutes < 0
        ):
            raise SchemaValidationError("ready_difference_minutes must be non-negative.")
        for name, value in (
            ("saving_difference", self.saving_difference),
            ("relative_stress_difference", self.relative_stress_difference),
            ("diversity_score", self.diversity_score),
        ):
            if value is not None and _finite(name, value) < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class MenuAssemblySettings:
    """Deterministic controls for paper-consistent filtering and display.

    Saving tolerances are in currency units, target tolerances are SOC
    fractions.  ``delta_saving_merge`` is a separate customer-value
    compaction threshold in tariff currency units; it is not the saving-band
    feasibility tolerance.  Saving buckets use
    ``floor((actual_saving + numerical_tolerance) / delta_saving_merge)`` and
    are formed independently within each absolute-ready/target-energy group.
    """

    pruning_saving_tolerance: float = 1e-8
    positive_saving_tolerance: float = 1e-8
    delta_saving_merge: float = 5.0
    # Compatibility alias for the former Commit 7 setting name.  When
    # supplied it overrides ``delta_saving_merge`` deterministically.
    saving_merge_gap: float | None = None
    saving_dominance_tolerance: float = 1e-8
    battery_stress_dominance_tolerance: float = 1e-8
    health_dominance_tolerance: float = 1e-8
    target_dominance_tolerance: float = 1e-8
    health_tie_tolerance: float = 1e-8
    target_energy_dominance_tolerance_kwh: float | None = None
    ready_tolerance_minutes: int = 0
    display_diversity: DisplayDiversityParameters = field(
        default_factory=DisplayDiversityParameters
    )
    display_cap: int | None = None
    dominance_tolerance: float | None = None
    menu_stage: MenuStage = "displayed"

    def __post_init__(self) -> None:
        for name, value in (
            ("pruning_saving_tolerance", self.pruning_saving_tolerance),
            ("positive_saving_tolerance", self.positive_saving_tolerance),
            ("delta_saving_merge", self.delta_saving_merge),
            ("saving_dominance_tolerance", self.saving_dominance_tolerance),
            (
                "battery_stress_dominance_tolerance",
                self.battery_stress_dominance_tolerance,
            ),
            ("health_dominance_tolerance", self.health_dominance_tolerance),
            ("target_dominance_tolerance", self.target_dominance_tolerance),
            ("health_tie_tolerance", self.health_tie_tolerance),
        ):
            numeric = _finite(name, value)
            if numeric < 0.0:
                raise PhysicalConstraintError(f"{name} must be non-negative.")
        if self.saving_merge_gap is not None:
            alias = _finite("saving_merge_gap", self.saving_merge_gap)
            if alias < 0.0:
                raise PhysicalConstraintError("saving_merge_gap must be non-negative.")
            object.__setattr__(self, "delta_saving_merge", alias)
        object.__setattr__(self, "saving_merge_gap", self.delta_saving_merge)
        target_energy_tolerance = self.target_energy_dominance_tolerance_kwh
        if target_energy_tolerance is None:
            target_energy_tolerance = self.target_dominance_tolerance
        else:
            target_energy_tolerance = _finite(
                "target_energy_dominance_tolerance_kwh", target_energy_tolerance
            )
            if target_energy_tolerance < 0.0:
                raise PhysicalConstraintError(
                    "target_energy_dominance_tolerance_kwh must be non-negative."
                )
            object.__setattr__(self, "target_dominance_tolerance", target_energy_tolerance)
        object.__setattr__(self, "target_energy_dominance_tolerance_kwh", target_energy_tolerance)
        if isinstance(self.ready_tolerance_minutes, bool) or not isinstance(
            self.ready_tolerance_minutes, int
        ):
            raise SchemaValidationError("ready_tolerance_minutes must be an integer.")
        if self.ready_tolerance_minutes < 0:
            raise PhysicalConstraintError("ready_tolerance_minutes must be non-negative.")
        if not isinstance(self.display_diversity, DisplayDiversityParameters):
            raise SchemaValidationError("display_diversity has an invalid type.")
        if self.dominance_tolerance is not None:
            alias = _finite("dominance_tolerance", self.dominance_tolerance)
            if alias < 0.0:
                raise PhysicalConstraintError("dominance_tolerance must be non-negative.")
            object.__setattr__(self, "saving_dominance_tolerance", alias)
            object.__setattr__(self, "battery_stress_dominance_tolerance", alias)
            object.__setattr__(self, "health_dominance_tolerance", alias)
            object.__setattr__(self, "target_dominance_tolerance", alias)
            object.__setattr__(self, "target_energy_dominance_tolerance_kwh", alias)
        elif (
            self.battery_stress_dominance_tolerance == 1e-8
            and self.health_dominance_tolerance != 1e-8
        ):
            # ``health_dominance_tolerance`` was the public name before raw
            # The legacy stress field is retained as a compatibility alias;
            # the canonical absolute fade remains the scientific quantity.
            object.__setattr__(
                self,
                "battery_stress_dominance_tolerance",
                self.health_dominance_tolerance,
            )
        if self.display_cap is not None:
            if isinstance(self.display_cap, bool) or not isinstance(self.display_cap, int):
                raise SchemaValidationError("display_cap must be an integer when supplied.")
            if self.display_cap <= 0:
                raise PhysicalConstraintError("display_cap must be positive.")
            object.__setattr__(
                self,
                "display_diversity",
                replace(
                    self.display_diversity,
                    maximum_displayed_offers=self.display_cap,
                ),
            )
        if self.menu_stage not in ("generated", "compacted", "pareto", "displayed"):
            raise SchemaValidationError(
                "menu_stage must be generated, compacted, pareto, or displayed."
            )


@dataclass(frozen=True, slots=True)
class AssembledMenu:
    """Immutable generated-to-displayed menu pipeline for one session."""

    ev_id: str
    offers: tuple[MenuOffer, ...]
    assessments: tuple[DegradationAssessment, ...]
    source_frontiers: tuple[SavingFrontier, ...]
    source_metadata: tuple[OfferSource, ...] = ()
    display_cap: int | None = None
    all_generated_offers: tuple[MenuOffer, ...] = ()
    all_generated_assessments: tuple[DegradationAssessment, ...] = ()
    all_generated_metadata: tuple[OfferSource, ...] = ()
    retained_offers: tuple[MenuOffer, ...] = ()
    retained_assessments: tuple[DegradationAssessment, ...] = ()
    retained_metadata: tuple[OfferSource, ...] = ()
    compacted_offers: tuple[MenuOffer, ...] = ()
    compacted_assessments: tuple[DegradationAssessment, ...] = ()
    compacted_metadata: tuple[OfferSource, ...] = ()
    pareto_efficient_offers: tuple[MenuOffer, ...] = ()
    pareto_efficient_assessments: tuple[DegradationAssessment, ...] = ()
    pareto_efficient_metadata: tuple[OfferSource, ...] = ()
    preserved_bau_anchors: tuple[MenuOffer, ...] = ()
    preserved_bau_anchor_assessments: tuple[DegradationAssessment, ...] = ()
    preserved_bau_anchor_metadata: tuple[OfferSource, ...] = ()
    pareto_offers: tuple[MenuOffer, ...] = ()
    pareto_assessments: tuple[DegradationAssessment, ...] = ()
    pareto_metadata: tuple[OfferSource, ...] = ()
    displayed_stage_offers: tuple[MenuOffer, ...] = ()
    displayed_stage_assessments: tuple[DegradationAssessment, ...] = ()
    displayed_stage_metadata: tuple[OfferSource, ...] = ()
    bau_reference_offers: tuple[MenuOffer, ...] = ()
    bau_reference_assessments: tuple[DegradationAssessment, ...] = ()
    bau_reference_metadata: tuple[OfferSource, ...] = ()
    menu_stage: MenuStage = "displayed"
    pipeline_diagnostics: PipelineDiagnostics = field(default_factory=PipelineDiagnostics)
    raw_offer_count: int = 0
    scientific_duplicate_count: int = 0
    generated_offer_count: int = 0
    request_diagnostics: RequestGenerationDiagnostics = field(
        default_factory=RequestGenerationDiagnostics
    )
    optimization_diagnostics: OptimizationDiagnostics = field(
        default_factory=OptimizationDiagnostics
    )
    degradation_objective_diagnostics: DegradationObjectiveDiagnostics | None = None
    target_summaries: tuple[TargetGenerationSummary, ...] = ()
    saving_level_failures: tuple[SavingLevelFailure, ...] = ()

    @property
    def frontiers(self) -> tuple[SavingFrontier, ...]:
        """Compatibility alias for the auditable source frontiers."""
        return self.source_frontiers

    @property
    def generated_offers(self) -> tuple[MenuOffer, ...]:
        """Rich Algorithm 1 offers before optional display capping."""
        return self.all_generated_offers or self.offers

    @property
    def generated_assessments(self) -> tuple[DegradationAssessment, ...]:
        return self.all_generated_assessments or self.assessments

    @property
    def generated_metadata(self) -> tuple[OfferSource, ...]:
        return self.all_generated_metadata or self.source_metadata

    @property
    def displayed_offers(self) -> tuple[MenuOffer, ...]:
        return self.displayed_stage_offers or self.offers

    @property
    def retained(self) -> tuple[MenuOffer, ...]:
        """Compatibility-friendly alias for the positive-saving retained set."""
        return self.retained_offers

    def __post_init__(self) -> None:
        if not isinstance(self.ev_id, str) or not self.ev_id.strip():
            raise SchemaValidationError("ev_id must be a non-empty string.")
        object.__setattr__(self, "ev_id", self.ev_id.strip())
        offers = _tuple_field("offers", self.offers)
        assessments = _tuple_field("assessments", self.assessments)
        frontiers = _tuple_field("source_frontiers", self.source_frontiers)
        metadata = _tuple_field("source_metadata", self.source_metadata)
        all_offers = _tuple_field("all_generated_offers", self.all_generated_offers or offers)
        all_assessments = _tuple_field(
            "all_generated_assessments", self.all_generated_assessments or assessments
        )
        all_metadata = _tuple_field(
            "all_generated_metadata", self.all_generated_metadata or metadata
        )
        object.__setattr__(self, "offers", offers)
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "source_frontiers", frontiers)
        object.__setattr__(self, "source_metadata", metadata)
        object.__setattr__(self, "all_generated_offers", all_offers)
        object.__setattr__(self, "all_generated_assessments", all_assessments)
        object.__setattr__(self, "all_generated_metadata", all_metadata)
        if self.menu_stage not in ("generated", "compacted", "pareto", "displayed"):
            raise SchemaValidationError(
                "menu_stage must be generated, compacted, pareto, or displayed."
            )
        if not isinstance(self.pipeline_diagnostics, PipelineDiagnostics):
            raise SchemaValidationError("pipeline_diagnostics has an invalid type.")
        stage_defaults = {
            "retained_offers": all_offers,
            "retained_assessments": all_assessments,
            "retained_metadata": all_metadata,
            "compacted_offers": all_offers,
            "compacted_assessments": all_assessments,
            "compacted_metadata": all_metadata,
            "pareto_efficient_offers": all_offers,
            "pareto_efficient_assessments": all_assessments,
            "pareto_efficient_metadata": all_metadata,
            "preserved_bau_anchors": tuple(
                offer
                for offer, source in zip(all_offers, all_metadata, strict=True)
                if source.source_kind == "bau"
            ),
            "preserved_bau_anchor_assessments": tuple(
                assessment
                for offer, assessment, source in zip(
                    all_offers, all_assessments, all_metadata, strict=True
                )
                if source.source_kind == "bau"
            ),
            "preserved_bau_anchor_metadata": tuple(
                source for source in all_metadata if source.source_kind == "bau"
            ),
            "pareto_offers": all_offers,
            "pareto_assessments": all_assessments,
            "pareto_metadata": all_metadata,
            "displayed_stage_offers": offers,
            "displayed_stage_assessments": assessments,
            "displayed_stage_metadata": metadata,
            "bau_reference_offers": tuple(
                offer
                for offer, source in zip(all_offers, all_metadata, strict=True)
                if source.source_kind == "bau"
            ),
            "bau_reference_assessments": tuple(
                assessment
                for assessment, source in zip(all_assessments, all_metadata, strict=True)
                if source.source_kind == "bau"
            ),
            "bau_reference_metadata": tuple(
                source for source in all_metadata if source.source_kind == "bau"
            ),
        }
        for name, default in stage_defaults.items():
            if not getattr(self, name):
                object.__setattr__(self, name, default)
        if self.request_diagnostics is None:
            object.__setattr__(self, "request_diagnostics", RequestGenerationDiagnostics())
        elif not isinstance(self.request_diagnostics, RequestGenerationDiagnostics):
            raise SchemaValidationError("request_diagnostics has an invalid type.")
        if self.optimization_diagnostics is None:
            object.__setattr__(self, "optimization_diagnostics", OptimizationDiagnostics())
        elif not isinstance(self.optimization_diagnostics, OptimizationDiagnostics):
            raise SchemaValidationError("optimization_diagnostics has an invalid type.")
        if self.degradation_objective_diagnostics is not None and not isinstance(
            self.degradation_objective_diagnostics, DegradationObjectiveDiagnostics
        ):
            raise SchemaValidationError("invalid degradation objective diagnostics.")
        summaries = tuple(self.target_summaries)
        if any(not isinstance(item, TargetGenerationSummary) for item in summaries):
            raise SchemaValidationError(
                "target_summaries must contain TargetGenerationSummary objects."
            )
        failures = tuple(self.saving_level_failures)
        if any(not isinstance(item, SavingLevelFailure) for item in failures):
            raise SchemaValidationError("saving_level_failures has an invalid item.")
        object.__setattr__(self, "target_summaries", summaries)
        object.__setattr__(self, "saving_level_failures", failures)
        if self.display_cap is not None:
            if isinstance(self.display_cap, bool) or not isinstance(self.display_cap, int):
                raise SchemaValidationError("display_cap must be an integer when supplied.")
            if self.display_cap <= 0:
                raise PhysicalConstraintError("display_cap must be positive.")
        if len(offers) != len(assessments):
            raise SchemaValidationError("offers and assessments must be aligned.")
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
        if self.display_cap is not None and len(self.displayed_offers) > self.display_cap:
            raise PhysicalConstraintError("displayed menu exceeds display_cap.")
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
        # Displayed offers intentionally exclude BAU.  Validate baseline
        # completeness against the immutable rich snapshot instead.
        for source, offer in zip(all_metadata, all_offers, strict=True):
            if source.source_kind == "bau":
                target_bau_count[offer.target_soc] = target_bau_count.get(offer.target_soc, 0) + 1
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
        if len(all_offers) != len(all_assessments) or len(all_offers) != len(all_metadata):
            raise SchemaValidationError("rich generated offer snapshots must be aligned.")
        if any(not isinstance(offer, MenuOffer) for offer in all_offers):
            raise SchemaValidationError("all_generated_offers must contain MenuOffer objects.")
        if any(not isinstance(item, DegradationAssessment) for item in all_assessments):
            raise SchemaValidationError(
                "all_generated_assessments must contain DegradationAssessment objects."
            )
        if any(not isinstance(item, OfferSource) for item in all_metadata):
            raise SchemaValidationError("all_generated_metadata must contain OfferSource objects.")
        if tuple(offer.offer_id for offer in all_offers) != tuple(
            item.candidate_id for item in all_assessments
        ) or tuple(offer.offer_id for offer in all_offers) != tuple(
            item.offer_id for item in all_metadata
        ):
            raise SchemaValidationError("rich generated snapshots must use aligned identities.")
        if self.generated_offer_count not in (0, len(all_offers)):
            raise SchemaValidationError("generated_offer_count is inconsistent.")
        object.__setattr__(self, "generated_offer_count", len(all_offers))
        if self.raw_offer_count < 0 or self.scientific_duplicate_count < 0:
            raise SchemaValidationError("offer counts must be non-negative.")
        stage_fields = (
            ("retained_offers", "retained_assessments", "retained_metadata"),
            ("compacted_offers", "compacted_assessments", "compacted_metadata"),
            (
                "pareto_efficient_offers",
                "pareto_efficient_assessments",
                "pareto_efficient_metadata",
            ),
            (
                "preserved_bau_anchors",
                "preserved_bau_anchor_assessments",
                "preserved_bau_anchor_metadata",
            ),
            ("pareto_offers", "pareto_assessments", "pareto_metadata"),
            (
                "displayed_stage_offers",
                "displayed_stage_assessments",
                "displayed_stage_metadata",
            ),
        )
        for offer_name, assessment_name, metadata_name in stage_fields:
            stage_offers = _tuple_field(offer_name, getattr(self, offer_name))
            stage_assessments = _tuple_field(assessment_name, getattr(self, assessment_name))
            stage_metadata = _tuple_field(metadata_name, getattr(self, metadata_name))
            if not (len(stage_offers) == len(stage_assessments) == len(stage_metadata)):
                raise SchemaValidationError(f"{offer_name} stage fields must be aligned.")
            if any(not isinstance(item, MenuOffer) for item in stage_offers):
                raise SchemaValidationError(f"{offer_name} must contain MenuOffer objects.")
            if any(not isinstance(item, DegradationAssessment) for item in stage_assessments):
                raise SchemaValidationError(f"{assessment_name} contains an invalid item.")
            if any(not isinstance(item, OfferSource) for item in stage_metadata):
                raise SchemaValidationError(f"{metadata_name} contains an invalid item.")
            ids = tuple(item.offer_id for item in stage_offers)
            if len(set(ids)) != len(ids):
                raise SchemaValidationError(f"{offer_name} IDs must be unique.")
            if ids != tuple(item.candidate_id for item in stage_assessments):
                raise SchemaValidationError(f"{offer_name} assessments are misaligned.")
            if ids != tuple(item.offer_id for item in stage_metadata):
                raise SchemaValidationError(f"{offer_name} metadata are misaligned.")
            object.__setattr__(self, offer_name, stage_offers)
            object.__setattr__(self, assessment_name, stage_assessments)
            object.__setattr__(self, metadata_name, stage_metadata)

    @property
    def request_count_total(self) -> int:
        return self.request_diagnostics.request_count_total

    @property
    def request_count_feasible(self) -> int:
        return self.request_diagnostics.request_count_feasible

    @property
    def request_count_positive_saving(self) -> int:
        return self.request_diagnostics.request_count_positive_saving

    @property
    def request_count_no_saving(self) -> int:
        return self.request_diagnostics.request_count_no_saving

    @property
    def request_count_infeasible(self) -> int:
        return self.request_diagnostics.request_count_infeasible

    @property
    def retained_offer_count(self) -> int:
        return self.pipeline_diagnostics.retained_offer_count

    @property
    def compacted_offer_count(self) -> int:
        return self.pipeline_diagnostics.compacted_offer_count

    @property
    def pareto_efficient_offer_count(self) -> int:
        return self.pipeline_diagnostics.pareto_efficient_offer_count

    @property
    def pareto_offer_count(self) -> int:
        return self.pipeline_diagnostics.pareto_offer_count

    @property
    def displayed_offer_count(self) -> int:
        return self.pipeline_diagnostics.displayed_offer_count

    @property
    def nonpositive_removed_count(self) -> int:
        return self.pipeline_diagnostics.nonpositive_removed_count

    @property
    def compaction_removed_count(self) -> int:
        return self.pipeline_diagnostics.compaction_removed_count

    @property
    def pareto_dominated_removed_count(self) -> int:
        return self.pipeline_diagnostics.pareto_dominated_removed_count

    @property
    def display_selection_removed_count(self) -> int:
        return self.pipeline_diagnostics.display_selection_removed_count


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
    """Build the rich, exact-deduplicated Algorithm 1 offer set.

    Reduction utilities from Commit 7 remain available, but are deliberately
    not run here.  An explicit display cap is applied only after every request,
    saving level, validation, and scientific duplicate decision is complete.
    """
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
    fsettings = (
        FrontierSettings(
            saving_step=msettings.saving_step,
            maximum_saving_levels_per_request=msettings.maximum_saving_levels_per_request,
            saving_band_tolerance=msettings.saving_band_tolerance,
            saving_zero_tolerance=msettings.saving_zero_tolerance or msettings.numerical_tolerance,
        )
        if frontier_settings is None
        else frontier_settings
    )
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
    # Every feasible request is retained for Algorithm 1 level generation.
    # The old change-point pruning remains a public utility for later stages.
    requests = tuple(
        sorted(
            (
                candidate
                for candidate in generated_menu.candidates
                if candidate.kind == "minimum_cost"
                and candidate.saving
                > max(
                    asettings.positive_saving_tolerance,
                    msettings.saving_zero_tolerance or msettings.numerical_tolerance,
                )
            ),
            key=lambda candidate: (
                candidate.target_soc,
                candidate.ready_step,
                candidate.candidate_id,
            ),
        )
    )
    frontiers: list[SavingFrontier] = []
    optimization_diagnostics = OptimizationDiagnostics()
    saving_level_failures: list[SavingLevelFailure] = []
    synthetic: list[MenuCandidate] = [bau_by_target[target] for target in sorted(bau_by_target)]
    source_by_candidate_id: dict[str, OfferSource] = {
        candidate.candidate_id: OfferSource(
            offer_id=candidate.candidate_id,
            source_point_id=None,
            source_candidate_id=candidate.candidate_id,
            endpoint_role="bau",
            source_kind="bau",
            provenance_flags=("is_bau",),
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
        except ValueError as exc:
            optimization_diagnostics = optimization_diagnostics.add(
                OptimizationDiagnostics(
                    optimization_attempt_count=1,
                    optimization_solver_failure_count=1,
                )
            )
            raise PhysicalConstraintError(
                "Frontier construction failed for "
                f"candidate={candidate.candidate_id}, target_soc={candidate.target_soc}, "
                f"ready_step={candidate.ready_step}."
            ) from exc
        except PhysicalConstraintError as exc:
            # Algorithm 1 retains the independently validated offers from
            # successful requests; a failed intermediate request is audited
            # by omission rather than making every other request unusable.
            reason: FailureReason = (
                "infeasible_band"
                if any(
                    token in str(exc).lower() for token in ("unattainable", "eligible", "feasible")
                )
                else "solver_failure"
            )
            optimization_diagnostics = optimization_diagnostics.add(
                OptimizationDiagnostics(
                    optimization_attempt_count=1,
                    optimization_infeasible_count=1 if reason == "infeasible_band" else 0,
                    optimization_solver_failure_count=1 if reason == "solver_failure" else 0,
                )
            )
            saving_level_failures.append(
                SavingLevelFailure(
                    ready_step=candidate.ready_step,
                    target_soc=candidate.target_soc,
                    requested_saving=max(candidate.saving, 0.0),
                    reason=reason,
                )
            )
            continue
        except SchemaValidationError:
            optimization_diagnostics = optimization_diagnostics.add(
                OptimizationDiagnostics(
                    optimization_attempt_count=1,
                    optimization_validation_failure_count=1,
                )
            )
            saving_level_failures.append(
                SavingLevelFailure(
                    ready_step=candidate.ready_step,
                    target_soc=candidate.target_soc,
                    requested_saving=max(candidate.saving, 0.0),
                    reason="validation_failure",
                )
            )
            continue
        frontiers.append(frontier)
        optimization_diagnostics = optimization_diagnostics.add(
            frontier.optimization_diagnostics or OptimizationDiagnostics()
        )
        saving_level_failures.extend(frontier.level_failures)
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
                provenance_flags=_flags_for_role(point.endpoint_role),
                saving_provenance=(
                    () if point.requested_saving is None else (point.requested_saving,)
                ),
            )

    scored = score_generated_menu(
        ev=ev,
        session=session,
        signal=signal,
        menu=GeneratedMenu(ev_id=ev.ev_id, candidates=tuple(synthetic)),
        degradation_settings=dsettings,
        menu_settings=msettings,
        normalize_health=False,
    )
    assessment_by_id = {assessment.candidate_id: assessment for assessment in scored.assessments}
    bau_ids = {candidate.candidate_id for candidate in bau_by_target.values()}
    signal_starts = signal.interval_start_minutes
    signal_ends = signal.interval_end_minutes
    scored_offers = tuple(
        replace(
            offer,
            requested_saving=(
                source_by_candidate_id[offer.offer_id].saving_provenance[0]
                if source_by_candidate_id[offer.offer_id].saving_provenance
                else None
            ),
            provenance_flags=source_by_candidate_id[offer.offer_id].provenance_flags,
            ready_boundary_absolute_minute=(
                signal_starts[offer.ready_step]
                if signal_starts is not None and offer.ready_step < len(signal_starts)
                else signal_ends[-1]
                if signal_ends is not None
                else None
            ),
            target_battery_energy_kwh=max(
                session.initial_energy_kwh,
                offer.target_soc * ev.battery_capacity_kwh,
            ),
            scientific_role=source_by_candidate_id[offer.offer_id].endpoint_role,
            saving_band_lower=(
                None
                if source_by_candidate_id[offer.offer_id].source_kind == "bau"
                or not source_by_candidate_id[offer.offer_id].saving_provenance
                else _saving_band_lower(
                    source_by_candidate_id[offer.offer_id].saving_provenance[0],
                    fsettings.effective_saving_band,
                )
            ),
            saving_band_upper=(
                None
                if source_by_candidate_id[offer.offer_id].source_kind == "bau"
                or not source_by_candidate_id[offer.offer_id].saving_provenance
                else source_by_candidate_id[offer.offer_id].saving_provenance[0]
                + fsettings.effective_saving_band
            ),
            saving_band_violation=(
                0.0
                if source_by_candidate_id[offer.offer_id].source_kind == "bau"
                or not source_by_candidate_id[offer.offer_id].saving_provenance
                else max(
                    0.0,
                    _saving_band_lower(
                        source_by_candidate_id[offer.offer_id].saving_provenance[0],
                        fsettings.effective_saving_band,
                    )
                    - offer.advertised_saving,
                    offer.advertised_saving
                    - (
                        source_by_candidate_id[offer.offer_id].saving_provenance[0]
                        + fsettings.effective_saving_band
                    ),
                )
            ),
            battery_metric_model_id=BATTERY_METRIC_MODEL_ID,
            battery_metric_comparison_scope=BATTERY_METRIC_COMPARISON_SCOPE,
        )
        for offer in scored.offers
    )
    positive_or_bau = tuple(
        offer
        for offer in scored_offers
        if offer.offer_id in bau_ids
        or offer.advertised_saving > asettings.positive_saving_tolerance
    )
    deduplicated = _remove_exact_duplicates(positive_or_bau, assessment_by_id, bau_ids)
    # Merge provenance from all exact scientific pathways into the retained
    # representative without changing its physical/scientific values.
    grouped_positive: dict[tuple[object, ...], list[MenuOffer]] = {}
    for offer in positive_or_bau:
        if offer.offer_id not in bau_ids:
            grouped_positive.setdefault(_scientific_offer_key(offer), []).append(offer)
    deduplicated_with_provenance: list[MenuOffer] = []
    for representative in deduplicated:
        duplicates = grouped_positive.get(_scientific_offer_key(representative), ())
        if not duplicates:
            deduplicated_with_provenance.append(representative)
            continue
        source = source_by_candidate_id[representative.offer_id]
        merged_flags = tuple(
            sorted(
                set(source.provenance_flags).union(
                    *(source_by_candidate_id[item.offer_id].provenance_flags for item in duplicates)
                )
            )
        )
        merged_savings = tuple(
            sorted(
                set(source.saving_provenance).union(
                    *(
                        source_by_candidate_id[item.offer_id].saving_provenance
                        for item in duplicates
                    )
                )
            )
        )
        source_by_candidate_id[representative.offer_id] = replace(
            source,
            provenance_flags=merged_flags,
            saving_provenance=merged_savings,
        )
        deduplicated_with_provenance.append(
            replace(
                representative,
                requested_saving=merged_savings[0] if merged_savings else None,
                provenance_flags=merged_flags,
            )
        )
    deduplicated = tuple(deduplicated_with_provenance)
    # Rich generated set: no spacing, compaction, or Pareto reduction.
    rich = tuple(
        sorted(
            deduplicated,
            key=lambda offer: _display_sort_key(
                offer, source_by_candidate_id[offer.offer_id].source_kind
            ),
        )
    )
    generated_source_by_id = dict(source_by_candidate_id)
    zero_tolerance = (
        msettings.saving_zero_tolerance
        if msettings.saving_zero_tolerance is not None
        else msettings.numerical_tolerance
    )
    retained = tuple(
        offer
        for offer in rich
        if offer.offer_id in bau_ids
        or offer.advertised_saving > asettings.positive_saving_tolerance
    )
    derived_source_by_id = dict(generated_source_by_id)
    compacted, compaction_groups, compaction_decisions = _compact_offer_stages(
        retained,
        derived_source_by_id,
        delta_saving_merge=asettings.delta_saving_merge,
        numerical_tolerance=msettings.numerical_tolerance,
        endpoint_tolerance=max(
            fsettings.effective_saving_band,
            fsettings.saving_tolerance,
            fsettings.cost_tolerance,
        ),
    )
    pareto_efficient, preserved_bau_anchors, dominance_pairs = _pareto_offer_stages(
        compacted,
        derived_source_by_id,
        saving_tolerance=asettings.saving_dominance_tolerance,
        target_energy_tolerance=(
            asettings.target_energy_dominance_tolerance_kwh
            if asettings.target_energy_dominance_tolerance_kwh is not None
            else asettings.target_dominance_tolerance
        ),
        battery_stress_tolerance=asettings.battery_stress_dominance_tolerance,
        ready_tolerance_minutes=asettings.ready_tolerance_minutes,
    )
    pareto_by_id = {offer.offer_id: offer for offer in pareto_efficient}
    for anchor in preserved_bau_anchors:
        pareto_by_id[anchor.offer_id] = anchor
    pareto = tuple(
        sorted(
            pareto_by_id.values(),
            key=lambda offer: _display_sort_key(
                offer, derived_source_by_id[offer.offer_id].source_kind
            ),
        )
    )
    stage_offers: dict[MenuStage, tuple[MenuOffer, ...]] = {
        "generated": rich,
        "compacted": compacted,
        "pareto": pareto,
        "displayed": pareto,
    }
    display_diversity = asettings.display_diversity
    if asettings.display_cap is not None:
        display_diversity = replace(
            display_diversity,
            maximum_displayed_offers=asettings.display_cap,
        )
    (
        displayed_stage,
        display_selection_summary,
        target_display_summaries,
        display_selection_decisions,
        distinctness_diagnostics,
    ) = _display_diversity_selection(
        pareto,
        derived_source_by_id,
        display_diversity,
        numerical_tolerance=msettings.numerical_tolerance,
        positive_saving_tolerance=asettings.positive_saving_tolerance,
        saving_tolerance=asettings.saving_dominance_tolerance,
        target_energy_tolerance=(
            asettings.target_energy_dominance_tolerance_kwh
            if asettings.target_energy_dominance_tolerance_kwh is not None
            else asettings.target_dominance_tolerance
        ),
        battery_stress_tolerance=asettings.battery_stress_dominance_tolerance,
        ready_tolerance_minutes=asettings.ready_tolerance_minutes,
    )
    if asettings.menu_stage == "displayed":
        selected = displayed_stage
    else:
        selected = stage_offers[asettings.menu_stage]
    selected = tuple(
        sorted(
            selected,
            key=lambda offer: _display_sort_key(
                offer,
                (
                    generated_source_by_id
                    if asettings.menu_stage == "generated"
                    else derived_source_by_id
                )[offer.offer_id].source_kind,
            ),
        )
    )
    selected_source_by_id = (
        generated_source_by_id if asettings.menu_stage == "generated" else derived_source_by_id
    )
    displayed_assessments = tuple(assessment_by_id[offer.offer_id] for offer in displayed_stage)
    displayed_metadata = tuple(derived_source_by_id[offer.offer_id] for offer in displayed_stage)
    assessments = tuple(assessment_by_id[offer.offer_id] for offer in selected)
    metadata = tuple(selected_source_by_id[offer.offer_id] for offer in selected)
    retained_assessments = tuple(assessment_by_id[offer.offer_id] for offer in retained)
    compacted_assessments = tuple(assessment_by_id[offer.offer_id] for offer in compacted)
    pareto_assessments = tuple(assessment_by_id[offer.offer_id] for offer in pareto)
    efficient_assessments = tuple(assessment_by_id[offer.offer_id] for offer in pareto_efficient)
    anchor_assessments = tuple(assessment_by_id[offer.offer_id] for offer in preserved_bau_anchors)
    pipeline_target_summaries = _build_pipeline_target_summaries(
        generated=rich,
        retained=retained,
        compacted=compacted,
        pareto_efficient=pareto_efficient,
        anchors=preserved_bau_anchors,
        pareto=pareto,
        displayed=displayed_stage,
    )
    pipeline_diagnostics = PipelineDiagnostics(
        generated_offer_count=len(rich),
        retained_offer_count=len(retained),
        compacted_offer_count=len(compacted),
        pareto_efficient_offer_count=len(pareto_efficient),
        pareto_offer_count=len(pareto),
        preserved_bau_anchor_count=len(preserved_bau_anchors),
        displayed_offer_count=len(displayed_stage),
        nonpositive_removed_count=len(rich) - len(retained),
        compaction_removed_count=len(retained) - len(compacted),
        pareto_dominated_removed_count=len(compacted) - len(pareto),
        display_selection_removed_count=display_selection_summary.removed_count,
        compaction_groups=compaction_groups,
        compaction_decisions=compaction_decisions,
        target_summaries=pipeline_target_summaries,
        display_selection_summary=display_selection_summary,
        target_display_summaries=target_display_summaries,
        display_selection_decisions=display_selection_decisions,
        distinctness_diagnostics=distinctness_diagnostics,
        pareto_dominated_offer_ids=tuple(
            sorted(
                {offer.offer_id for offer in compacted}
                - {item.offer_id for item in pareto_efficient}
                - {item.offer_id for item in preserved_bau_anchors}
            )
        ),
        pareto_dominated_bau_anchor_ids=tuple(
            sorted(
                {item.offer_id for item in preserved_bau_anchors}
                - {item.offer_id for item in pareto_efficient}
            )
        ),
        pareto_dominance_pairs=dominance_pairs,
    )
    return AssembledMenu(
        ev_id=ev.ev_id,
        offers=selected,
        assessments=assessments,
        source_frontiers=tuple(frontiers),
        source_metadata=metadata,
        display_cap=asettings.display_cap,
        all_generated_offers=rich,
        all_generated_assessments=tuple(assessment_by_id[offer.offer_id] for offer in rich),
        all_generated_metadata=tuple(generated_source_by_id[offer.offer_id] for offer in rich),
        retained_offers=retained,
        retained_assessments=retained_assessments,
        retained_metadata=tuple(generated_source_by_id[offer.offer_id] for offer in retained),
        compacted_offers=compacted,
        compacted_assessments=compacted_assessments,
        compacted_metadata=tuple(derived_source_by_id[offer.offer_id] for offer in compacted),
        pareto_efficient_offers=pareto_efficient,
        pareto_efficient_assessments=efficient_assessments,
        pareto_efficient_metadata=tuple(
            derived_source_by_id[offer.offer_id] for offer in pareto_efficient
        ),
        preserved_bau_anchors=preserved_bau_anchors,
        preserved_bau_anchor_assessments=anchor_assessments,
        preserved_bau_anchor_metadata=tuple(
            derived_source_by_id[offer.offer_id] for offer in preserved_bau_anchors
        ),
        pareto_offers=pareto,
        pareto_assessments=pareto_assessments,
        pareto_metadata=tuple(derived_source_by_id[offer.offer_id] for offer in pareto),
        displayed_stage_offers=displayed_stage,
        displayed_stage_assessments=displayed_assessments,
        displayed_stage_metadata=displayed_metadata,
        menu_stage=asettings.menu_stage,
        pipeline_diagnostics=pipeline_diagnostics,
        raw_offer_count=len(positive_or_bau),
        scientific_duplicate_count=len(positive_or_bau) - len(rich),
        generated_offer_count=len(rich),
        request_diagnostics=generated_menu.diagnostics,
        optimization_diagnostics=optimization_diagnostics,
        degradation_objective_diagnostics=next(
            (
                frontier.degradation_objective_diagnostics
                for frontier in frontiers
                if frontier.degradation_objective_diagnostics is not None
            ),
            None,
        ),
        target_summaries=_build_target_summaries(
            generated_menu=generated_menu,
            session=session,
            rich=rich,
            frontiers=tuple(frontiers),
            raw_offers=positive_or_bau,
            diagnostics=optimization_diagnostics,
            zero_tolerance=zero_tolerance,
        ),
        saving_level_failures=tuple(saving_level_failures),
    )


def _build_target_summaries(
    *,
    generated_menu: GeneratedMenu,
    session: ChargingSession,
    rich: tuple[MenuOffer, ...],
    frontiers: tuple[SavingFrontier, ...],
    raw_offers: tuple[MenuOffer, ...],
    diagnostics: OptimizationDiagnostics,
    zero_tolerance: float,
) -> tuple[TargetGenerationSummary, ...]:
    """Derive deterministic per-target completeness metadata."""
    del diagnostics  # frontier-local counts provide the target attribution.
    summaries: list[TargetGenerationSummary] = []
    targets = sorted({candidate.target_soc for candidate in generated_menu.candidates})
    for target in targets:
        target_candidates = tuple(
            candidate
            for candidate in generated_menu.candidates
            if candidate.target_soc == target and candidate.kind == "minimum_cost"
        )
        target_frontiers = tuple(
            frontier for frontier in frontiers if frontier.target_soc == target
        )
        target_rich = tuple(offer for offer in rich if offer.target_soc == target)
        target_raw = tuple(offer for offer in raw_offers if offer.target_soc == target)
        bau = next(
            candidate
            for candidate in generated_menu.candidates
            if candidate.target_soc == target and candidate.kind == "immediate_bau"
        )
        roles = tuple(offer.scientific_role or "bau" for offer in target_rich)
        summaries.append(
            TargetGenerationSummary(
                target_soc=target,
                bau_ready=bau.ready_step,
                request_count=session.departure_step - session.arrival_step + 1,
                positive_saving_request_count=sum(
                    candidate.saving > zero_tolerance for candidate in target_candidates
                ),
                selected_saving_level_count=sum(
                    len(frontier.points) for frontier in target_frontiers
                ),
                optimization_success_count=sum(
                    (
                        frontier.optimization_diagnostics or OptimizationDiagnostics()
                    ).optimization_success_count
                    for frontier in target_frontiers
                ),
                generated_offer_count=len(target_rich),
                duplicate_count=max(0, len(target_raw) - len(target_rich)),
                roles_present=roles,
            )
        )
    return tuple(summaries)


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
        key = _scientific_offer_key(offer)
        groups.setdefault(key, []).append(offer)
    retained.extend(min(group, key=lambda item: item.offer_id) for group in groups.values())
    return tuple(sorted(retained, key=lambda item: item.offer_id))


def _scientific_offer_key(offer: MenuOffer) -> tuple[object, ...]:
    return (
        offer.ready_step,
        _quantize(offer.target_soc),
        _quantize(offer.profile.battery_energy_kwh[-1]),
        _quantize(offer.advertised_saving),
        _quantize(offer.raw_battery_stress or 0.0),
        tuple(_quantize(value) for value in offer.profile.grid_energy_kwh),
    )


def _quantize(value: float, tolerance: float = 1e-8) -> int:
    """Canonical scientific quantization (never display rounding)."""
    return round(value / tolerance)


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


_ROLE_PRIORITY = {
    "bau": 0,
    "least_and_maximum": 1,
    "least_degradation": 2,
    "maximum_saving": 3,
    "low_saving": 4,
    "intermediate": 5,
}


def _primary_role_from_flags(flags: Iterable[str], *, is_bau: bool = False) -> str:
    """Return the documented deterministic primary role for merged provenance."""
    if is_bau or "is_bau" in flags:
        return "bau"
    flag_set = set(flags)
    has_max = "is_maximum_saving" in flag_set
    has_least = "is_least_degradation" in flag_set
    if has_max and has_least:
        return "least_and_maximum"
    if has_least:
        return "least_degradation"
    if has_max:
        return "maximum_saving"
    if "is_low_saving" in flag_set:
        return "low_saving"
    return "intermediate"


def _is_bau_offer(offer: MenuOffer, metadata: OfferSource | None = None) -> bool:
    return (
        metadata is not None and metadata.source_kind == "bau"
    ) or "is_bau" in offer.provenance_flags


def _offer_ready_absolute_minute(offer: MenuOffer) -> int:
    return (
        offer.ready_boundary_absolute_minute
        if offer.ready_boundary_absolute_minute is not None
        else offer.ready_step
    )


def _offer_target_energy(offer: MenuOffer) -> float:
    return (
        offer.target_battery_energy_kwh
        if offer.target_battery_energy_kwh is not None
        else offer.target_soc
    )


def _merge_derived_provenance(
    representative: MenuOffer,
    members: tuple[MenuOffer, ...],
    source_by_id: dict[str, OfferSource],
    *,
    preserve_endpoint: bool = False,
) -> tuple[MenuOffer, OfferSource]:
    """Merge provenance on a derived copy without changing scientific values."""
    base_source = source_by_id[representative.offer_id]
    member_sources = tuple(source_by_id[item.offer_id] for item in members)
    merged_flags = set(representative.provenance_flags)
    merged_flags.update(flag for item in members for flag in item.provenance_flags)
    # A maximum endpoint is a scientific fact tied to the endpoint saving; do
    # not transfer it to a lower-saving representative.  Least-degradation and
    # low/intermediate provenance can safely be combined within a value bucket.
    if not preserve_endpoint and "is_maximum_saving" in merged_flags:
        merged_flags.discard("is_maximum_saving")
        merged_flags.discard("is_endpoint")
        merged_flags.discard("is_least_and_maximum")
    role = _primary_role_from_flags(merged_flags, is_bau=_is_bau_offer(representative, base_source))
    if role == "maximum_saving" and "is_endpoint" not in merged_flags:
        merged_flags.add("is_endpoint")
    if role == "least_and_maximum":
        merged_flags.update(("is_endpoint", "is_least_and_maximum"))
    merged_savings = set(base_source.saving_provenance)
    merged_savings.update(value for source in member_sources for value in source.saving_provenance)
    source = replace(
        base_source,
        endpoint_role=role,
        provenance_flags=tuple(sorted(merged_flags)),
        saving_provenance=tuple(sorted(merged_savings)),
    )
    offer = replace(
        representative,
        provenance_flags=source.provenance_flags,
        scientific_role=role,
        requested_saving=source.saving_provenance[0] if source.saving_provenance else None,
    )
    source_by_id[offer.offer_id] = source
    return offer, source


def _compact_offer_stages(
    offers: tuple[MenuOffer, ...],
    source_by_id: dict[str, OfferSource],
    *,
    delta_saving_merge: float,
    numerical_tolerance: float,
    endpoint_tolerance: float,
) -> tuple[
    tuple[MenuOffer, ...], tuple[CompactionGroupSummary, ...], tuple[CompactionDecision, ...]
]:
    """Compact actual savings in fixed absolute-ready/target-energy groups."""
    if delta_saving_merge < 0.0 or not isfinite(delta_saving_merge):
        raise PhysicalConstraintError("delta_saving_merge must be finite and non-negative.")
    # Target energy is the scientific request coordinate; target SOC is kept
    # in the key as an explicit guard so distinct SOC requests can never be
    # compacted together when they happen to share the same energy floor.
    groups: dict[tuple[int, float, float], list[MenuOffer]] = {}
    for offer in offers:
        if _is_bau_offer(offer, source_by_id.get(offer.offer_id)):
            continue
        key = (
            _offer_ready_absolute_minute(offer),
            _offer_target_energy(offer),
            offer.target_soc,
        )
        groups.setdefault(key, []).append(offer)
    selected: dict[str, MenuOffer] = {
        offer.offer_id: offer
        for offer in offers
        if _is_bau_offer(offer, source_by_id.get(offer.offer_id))
    }
    summaries: list[CompactionGroupSummary] = []
    decisions: list[CompactionDecision] = []
    _ = endpoint_tolerance  # compatibility parameter; exact maximum is preserved below
    for (ready, target_energy, _target_soc), members_list in sorted(groups.items()):
        members = tuple(
            sorted(members_list, key=lambda item: (item.advertised_saving, item.offer_id))
        )
        if not members:
            continue
        target_soc = members[0].target_soc
        bucket_members: dict[int, list[MenuOffer]] = {}
        if delta_saving_merge == 0.0:
            # A zero threshold disables near-duplicate compaction while keeping
            # deterministic scientific duplicate handling upstream.
            for item in members:
                bucket_members[len(bucket_members)] = [item]
        else:
            for item in members:
                bucket = int((item.advertised_saving + numerical_tolerance) // delta_saving_merge)
                bucket_members.setdefault(bucket, []).append(item)
        max_saving = max(item.advertised_saving for item in members)
        # Preserve the true maximum achieved saving.  ``endpoint_tolerance``
        # is retained for API compatibility, but a near-maximum intermediate
        # must never replace the exact maximum endpoint.
        endpoint_candidates = tuple(
            item for item in members if max_saving - item.advertised_saving <= numerical_tolerance
        )
        representatives: list[MenuOffer] = []
        removed_ids: list[str] = []
        for bucket, bucket_items_list in sorted(bucket_members.items()):
            bucket_items = tuple(bucket_items_list)
            representative = min(
                bucket_items,
                key=lambda item: (
                    item.raw_battery_stress or 0.0,
                    -item.advertised_saving,
                    item.offer_id,
                ),
            )
            keep_members = bucket_items
            if representative.offer_id not in selected:
                representative, _ = _merge_derived_provenance(
                    representative, keep_members, source_by_id, preserve_endpoint=False
                )
            selected[representative.offer_id] = representative
            representatives.append(representative)
            removed = [
                item
                for item in bucket_items
                if item.offer_id != representative.offer_id
                and item.offer_id not in {endpoint.offer_id for endpoint in endpoint_candidates}
            ]
            removed_ids.extend(item.offer_id for item in removed)
            if removed:
                decisions.append(
                    CompactionDecision(
                        retained_offer_id=representative.offer_id,
                        removed_offer_ids=tuple(item.offer_id for item in removed),
                        saving_interval_lower=bucket * delta_saving_merge
                        if delta_saving_merge
                        else representative.advertised_saving,
                        saving_interval_upper=(bucket + 1) * delta_saving_merge
                        if delta_saving_merge
                        else representative.advertised_saving,
                        reason="same_saving_interval_higher_stress",
                    )
                )
        endpoint = min(
            endpoint_candidates,
            key=lambda item: (item.raw_battery_stress or 0.0, item.offer_id),
        )
        if endpoint.offer_id not in selected:
            endpoint, _ = _merge_derived_provenance(
                endpoint, (endpoint,), source_by_id, preserve_endpoint=True
            )
            selected[endpoint.offer_id] = endpoint
            representatives.append(endpoint)
            decisions.append(
                CompactionDecision(
                    retained_offer_id=endpoint.offer_id,
                    removed_offer_ids=(),
                    saving_interval_lower=endpoint.advertised_saving,
                    saving_interval_upper=endpoint.advertised_saving,
                    reason="same_saving_interval_endpoint_preserved",
                )
            )
        represented_ids = {item.offer_id for item in representatives}
        # If the endpoint was already the bucket representative, ensure its
        # endpoint provenance survives on that existing representative.
        endpoint_rep = selected[endpoint.offer_id]
        endpoint_rep, _ = _merge_derived_provenance(
            endpoint_rep, (endpoint,), source_by_id, preserve_endpoint=True
        )
        selected[endpoint_rep.offer_id] = endpoint_rep
        representative_ids = tuple(sorted(represented_ids | {endpoint_rep.offer_id}))
        summaries.append(
            CompactionGroupSummary(
                ready_absolute_minute=ready,
                target_soc=target_soc,
                input_offer_count=len(members),
                output_offer_count=len(representative_ids),
                saving_min=min(item.advertised_saving for item in members),
                saving_max=max_saving,
                representative_offer_ids=representative_ids,
                removed_offer_ids=tuple(sorted(set(removed_ids))),
            )
        )
    compacted = tuple(
        sorted(
            selected.values(),
            key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind),
        )
    )
    return compacted, tuple(summaries), tuple(decisions)


def _dominance_dimensions(
    a: MenuOffer,
    b: MenuOffer,
    *,
    saving_tolerance: float,
    battery_stress_tolerance: float,
    target_energy_tolerance: float,
    ready_tolerance_minutes: int,
) -> tuple[str, ...]:
    """Return dimensions on which ``a`` is strictly better than ``b``."""
    dimensions: list[str] = []
    if _offer_ready_absolute_minute(a) < _offer_ready_absolute_minute(b) - ready_tolerance_minutes:
        dimensions.append("ready")
    if _offer_target_energy(a) > _offer_target_energy(b) + target_energy_tolerance:
        dimensions.append("target_soc")
    if a.advertised_saving > b.advertised_saving + saving_tolerance:
        dimensions.append("saving")
    if (a.raw_battery_stress or 0.0) < (b.raw_battery_stress or 0.0) - battery_stress_tolerance:
        dimensions.append("battery_stress")
    return tuple(dimensions)


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
    lowest_stress = min(item.raw_battery_stress or 0.0 for item in cluster)
    health_tied = [
        item
        for item in cluster
        if (item.raw_battery_stress or 0.0) - lowest_stress <= health_tie_tolerance
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
    ready_tolerance_minutes: int = 0,
    target_energy_tolerance: float | None = None,
    battery_stress_tolerance: float | None = None,
) -> tuple[MenuOffer, ...]:
    """Compatibility Pareto helper using absolute raw stress direction."""
    target_tolerance = (
        target_tolerance if target_energy_tolerance is None else target_energy_tolerance
    )
    stress_tolerance = (
        health_tolerance if battery_stress_tolerance is None else battery_stress_tolerance
    )
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
                health_tolerance=stress_tolerance,
                target_tolerance=target_tolerance,
                ready_tolerance_minutes=ready_tolerance_minutes,
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
    ready_tolerance_minutes: int = 0,
) -> bool:
    ready_a = _offer_ready_absolute_minute(a)
    ready_b = _offer_ready_absolute_minute(b)
    target_a = _offer_target_energy(a)
    target_b = _offer_target_energy(b)
    stress_a = a.raw_battery_stress or 0.0
    stress_b = b.raw_battery_stress or 0.0
    weak = (
        ready_a <= ready_b + ready_tolerance_minutes
        and target_a >= target_b - target_tolerance
        and a.advertised_saving >= b.advertised_saving - saving_tolerance
        and stress_a <= stress_b + health_tolerance
    )
    strict = bool(
        ready_a < ready_b - ready_tolerance_minutes
        or target_a > target_b + target_tolerance
        or a.advertised_saving > b.advertised_saving + saving_tolerance
        or stress_a < stress_b - health_tolerance
    )
    return weak and strict


def _pareto_offer_stages(
    offers: tuple[MenuOffer, ...],
    source_by_id: dict[str, OfferSource],
    *,
    saving_tolerance: float,
    target_energy_tolerance: float,
    battery_stress_tolerance: float,
    ready_tolerance_minutes: int,
) -> tuple[
    tuple[MenuOffer, ...],
    tuple[MenuOffer, ...],
    tuple[tuple[str, str], ...],
]:
    """Return efficient offers, preserved BAU anchors, and dominance pairs."""
    ordered = tuple(
        sorted(
            offers,
            key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind),
        )
    )
    efficient: list[MenuOffer] = []
    dominated: list[str] = []
    pairs: list[tuple[str, str]] = []
    for offer in ordered:
        dominators = [
            other
            for other in ordered
            if other.offer_id != offer.offer_id
            and _dominates(
                other,
                offer,
                saving_tolerance=saving_tolerance,
                health_tolerance=battery_stress_tolerance,
                target_tolerance=target_energy_tolerance,
                ready_tolerance_minutes=ready_tolerance_minutes,
            )
        ]
        if dominators:
            dominated.append(offer.offer_id)
            pairs.extend((other.offer_id, offer.offer_id) for other in dominators)
            continue
        offer = replace(offer, is_pareto_efficient=True, is_preserved_bau_anchor=False)
        efficient.append(offer)
    anchors: list[MenuOffer] = []
    by_target: dict[float, list[MenuOffer]] = {}
    for offer in ordered:
        if _is_bau_offer(offer, source_by_id[offer.offer_id]):
            by_target.setdefault(offer.target_soc, []).append(offer)
    for target, candidates in sorted(by_target.items()):
        anchor = min(candidates, key=lambda item: item.offer_id)
        anchor = replace(anchor, is_preserved_bau_anchor=True)
        # Keep both flags when the BAU is genuinely efficient; otherwise the
        # anchor flag explicitly distinguishes reference preservation.
        if anchor.offer_id in {item.offer_id for item in efficient}:
            anchor = replace(anchor, is_pareto_efficient=True)
        anchors.append(anchor)
    efficient_ids = {item.offer_id for item in efficient}
    efficient = [
        replace(item, is_pareto_efficient=True) if item.offer_id in efficient_ids else item
        for item in efficient
    ]
    efficient.sort(
        key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind)
    )
    anchors.sort(key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind))
    return tuple(efficient), tuple(anchors), tuple(sorted(set(pairs)))


def _build_pipeline_target_summaries(
    *,
    generated: tuple[MenuOffer, ...],
    retained: tuple[MenuOffer, ...],
    compacted: tuple[MenuOffer, ...],
    pareto_efficient: tuple[MenuOffer, ...],
    anchors: tuple[MenuOffer, ...],
    pareto: tuple[MenuOffer, ...],
    displayed: tuple[MenuOffer, ...],
) -> tuple[PipelineTargetSummary, ...]:
    targets = sorted(
        {
            offer.target_soc
            for stage in (
                generated,
                retained,
                compacted,
                pareto_efficient,
                anchors,
                pareto,
                displayed,
            )
            for offer in stage
        }
    )

    def for_target(stage: tuple[MenuOffer, ...], target: float) -> tuple[MenuOffer, ...]:
        return tuple(offer for offer in stage if offer.target_soc == target)

    def roles(stage: tuple[MenuOffer, ...], target: float) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    offer.scientific_role
                    or ("bau" if offer.advertised_saving == 0.0 else "intermediate")
                    for offer in for_target(stage, target)
                }
            )
        )

    summaries: list[PipelineTargetSummary] = []
    for target in targets:
        stage_roles = {
            "generated": roles(generated, target),
            "retained": roles(retained, target),
            "compacted": roles(compacted, target),
            "pareto_efficient": roles(pareto_efficient, target),
            "preserved_bau_anchor": roles(anchors, target),
            "pareto": roles(pareto, target),
            "displayed": roles(displayed, target),
        }
        summaries.append(
            PipelineTargetSummary(
                target_soc=target,
                generated_offer_count=len(for_target(generated, target)),
                retained_offer_count=len(for_target(retained, target)),
                compacted_offer_count=len(for_target(compacted, target)),
                pareto_efficient_offer_count=len(for_target(pareto_efficient, target)),
                preserved_bau_anchor_count=len(for_target(anchors, target)),
                pareto_offer_count=len(for_target(pareto, target)),
                displayed_offer_count=len(for_target(displayed, target)),
                roles_present=tuple(
                    sorted({role for values in stage_roles.values() for role in values})
                ),
                generated_roles=stage_roles["generated"],
                retained_roles=stage_roles["retained"],
                compacted_roles=stage_roles["compacted"],
                pareto_efficient_roles=stage_roles["pareto_efficient"],
                preserved_bau_anchor_roles=stage_roles["preserved_bau_anchor"],
                pareto_roles=stage_roles["pareto"],
                displayed_roles=stage_roles["displayed"],
            )
        )
    return tuple(summaries)


def _display_target_key(offer: MenuOffer) -> tuple[float, float]:
    """Use unrounded target energy and SOC for diversity grouping."""
    return (_offer_target_energy(offer), offer.target_soc)


def _display_bau_offer(
    offers: tuple[MenuOffer, ...], source_by_id: dict[str, OfferSource]
) -> MenuOffer | None:
    bau = [offer for offer in offers if _is_bau_offer(offer, source_by_id[offer.offer_id])]
    if not bau:
        return None
    return min(
        bau,
        key=lambda offer: (
            _offer_ready_absolute_minute(offer),
            offer.offer_id,
        ),
    )


def _display_stress_difference(a: MenuOffer, b: MenuOffer, stress_floor: float) -> float:
    stress_a = float(a.raw_battery_stress or 0.0)
    stress_b = float(b.raw_battery_stress or 0.0)
    return abs(stress_a - stress_b) / max(abs(stress_a), abs(stress_b), stress_floor)


def _display_meaningful_saving(bau: MenuOffer, parameters: DisplayDiversityParameters) -> float:
    return max(
        parameters.minimum_saving_difference,
        parameters.minimum_saving_fraction_of_bau * abs(bau.same_target_bau_cost),
    )


def _display_profile_differs(a: MenuOffer, b: MenuOffer, tolerance: float) -> bool:
    return any(
        abs(left - right) > tolerance
        for left, right in zip(a.profile.grid_energy_kwh, b.profile.grid_energy_kwh, strict=True)
    )


def _display_close_ready_tradeoff(
    a: MenuOffer,
    b: MenuOffer,
    bau: MenuOffer,
    parameters: DisplayDiversityParameters,
    *,
    numerical_tolerance: float,
    saving_tolerance: float,
    target_energy_tolerance: float,
    battery_stress_tolerance: float,
    ready_tolerance_minutes: int,
) -> bool:
    ready_difference = abs(_offer_ready_absolute_minute(a) - _offer_ready_absolute_minute(b))
    if ready_difference >= parameters.minimum_ready_separation_minutes:
        return False
    saving_difference = abs(a.advertised_saving - b.advertised_saving)
    if saving_difference < _display_meaningful_saving(bau, parameters):
        return False
    stress_difference = _display_stress_difference(a, b, max(numerical_tolerance, 1e-12))
    if stress_difference < parameters.minimum_relative_stress_difference:
        return False
    stress_a = float(a.raw_battery_stress or 0.0)
    stress_b = float(b.raw_battery_stress or 0.0)
    if not (
        (
            a.advertised_saving > b.advertised_saving + saving_tolerance
            and stress_a > stress_b + battery_stress_tolerance
        )
        or (
            b.advertised_saving > a.advertised_saving + saving_tolerance
            and stress_b > stress_a + battery_stress_tolerance
        )
    ):
        return False
    if _dominates(
        a,
        b,
        saving_tolerance=saving_tolerance,
        health_tolerance=battery_stress_tolerance,
        target_tolerance=target_energy_tolerance,
        ready_tolerance_minutes=ready_tolerance_minutes,
    ) or _dominates(
        b,
        a,
        saving_tolerance=saving_tolerance,
        health_tolerance=battery_stress_tolerance,
        target_tolerance=target_energy_tolerance,
        ready_tolerance_minutes=ready_tolerance_minutes,
    ):
        return False
    return _display_profile_differs(a, b, numerical_tolerance)


def _display_pair_allowed(
    candidate: MenuOffer,
    selected: tuple[MenuOffer, ...],
    bau: MenuOffer,
    parameters: DisplayDiversityParameters,
    *,
    numerical_tolerance: float,
    saving_tolerance: float,
    target_energy_tolerance: float,
    battery_stress_tolerance: float,
    ready_tolerance_minutes: int,
) -> tuple[bool, bool]:
    """Return (allowed, used_close-ready-exception)."""
    for current in selected:
        if _display_target_key(current) != _display_target_key(candidate):
            continue
        ready_difference = abs(
            _offer_ready_absolute_minute(candidate) - _offer_ready_absolute_minute(current)
        )
        if ready_difference >= parameters.minimum_ready_separation_minutes:
            continue
        if _display_close_ready_tradeoff(
            candidate,
            current,
            bau,
            parameters,
            numerical_tolerance=numerical_tolerance,
            saving_tolerance=saving_tolerance,
            target_energy_tolerance=target_energy_tolerance,
            battery_stress_tolerance=battery_stress_tolerance,
            ready_tolerance_minutes=ready_tolerance_minutes,
        ):
            continue
        return False, False
    return True, any(
        _display_target_key(current) == _display_target_key(candidate)
        and abs(_offer_ready_absolute_minute(candidate) - _offer_ready_absolute_minute(current))
        < parameters.minimum_ready_separation_minutes
        for current in selected
    )


def _display_coordinates(
    offer: MenuOffer,
    group: tuple[MenuOffer, ...],
    bau: MenuOffer,
    numerical_tolerance: float,
) -> tuple[float, float, float]:
    positive = [item for item in group if item.advertised_saving > numerical_tolerance]
    latest_ready = max(
        [_offer_ready_absolute_minute(item) for item in group] + [_offer_ready_absolute_minute(bau)]
    )
    maximum_saving = max([item.advertised_saving for item in positive] + [0.0])
    stresses = [float(item.raw_battery_stress or 0.0) for item in group]
    minimum_stress = min(stresses + [float(bau.raw_battery_stress or 0.0)])
    maximum_stress = max(stresses + [float(bau.raw_battery_stress or 0.0)])
    delay_denominator = max(
        float(latest_ready - _offer_ready_absolute_minute(bau)), numerical_tolerance
    )
    saving_denominator = max(maximum_saving, numerical_tolerance)
    stress_denominator = max(maximum_stress - minimum_stress, numerical_tolerance)
    return (
        (_offer_ready_absolute_minute(offer) - _offer_ready_absolute_minute(bau))
        / delay_denominator,
        offer.advertised_saving / saving_denominator,
        (float(offer.raw_battery_stress or 0.0) - minimum_stress) / stress_denominator,
    )


def _display_distance(
    a: MenuOffer,
    b: MenuOffer,
    group: tuple[MenuOffer, ...],
    bau: MenuOffer,
    numerical_tolerance: float,
) -> float:
    left = _display_coordinates(a, group, bau, numerical_tolerance)
    right = _display_coordinates(b, group, bau, numerical_tolerance)
    return sqrt(sum((x - y) ** 2 for x, y in zip(left, right, strict=True)))


def _display_nearest(
    offer: MenuOffer,
    selected: tuple[MenuOffer, ...],
) -> MenuOffer | None:
    same_target = [
        item for item in selected if _display_target_key(item) == _display_target_key(offer)
    ]
    if not same_target:
        return None
    return min(
        same_target,
        key=lambda item: (
            abs(_offer_ready_absolute_minute(item) - _offer_ready_absolute_minute(offer)),
            abs(item.advertised_saving - offer.advertised_saving),
            item.offer_id,
        ),
    )


def _display_diversity_selection_legacy(
    offers: tuple[MenuOffer, ...],
    source_by_id: dict[str, OfferSource],
    parameters: DisplayDiversityParameters,
    *,
    numerical_tolerance: float,
    positive_saving_tolerance: float,
    saving_tolerance: float,
    target_energy_tolerance: float,
    battery_stress_tolerance: float,
    ready_tolerance_minutes: int,
) -> tuple[
    tuple[MenuOffer, ...],
    DisplaySelectionSummary,
    tuple[TargetDisplaySelectionSummary, ...],
    tuple[DisplaySelectionDecision, ...],
]:
    """Select a compact, deterministic customer menu from the Pareto stage."""
    ordered = tuple(
        sorted(
            offers,
            key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind),
        )
    )
    groups: dict[tuple[float, float], list[MenuOffer]] = {}
    for offer in ordered:
        groups.setdefault(_display_target_key(offer), []).append(offer)
    if (
        parameters.maximum_displayed_offers is not None
        and parameters.maximum_displayed_offers < len(groups)
    ):
        raise PhysicalConstraintError(
            "maximum_displayed_offers must be at least the number of target groups: "
            f"cap={parameters.maximum_displayed_offers}, targets={len(groups)}."
        )
    selected_info: dict[str, tuple[SelectionReason, float, set[str]]] = {}
    rejected_reasons: dict[str, SelectionReason] = {}
    target_groups: dict[tuple[float, float], tuple[MenuOffer, ...]] = {}

    def add_offer(
        offer: MenuOffer,
        reason: SelectionReason,
        group: tuple[MenuOffer, ...],
        bau: MenuOffer,
        score: float = 0.0,
    ) -> bool:
        if offer.offer_id in selected_info:
            existing_reason, existing_score, existing_flags = selected_info[offer.offer_id]
            if reason == "target_maximum_saving":
                existing_flags.add("maximum_saving")
            elif reason == "target_least_degradation":
                existing_flags.add("least_degradation")
            elif reason in ("intermediate_diversity", "target_positive_saving_coverage"):
                existing_flags.add("intermediate")
            selected_info[offer.offer_id] = (
                existing_reason,
                max(existing_score, score),
                existing_flags,
            )
            return True
        current = tuple(
            item
            for item in ordered
            if item.offer_id in selected_info
            and _display_target_key(item) == _display_target_key(offer)
        )
        allowed, used_exception = _display_pair_allowed(
            offer,
            current,
            bau,
            parameters,
            numerical_tolerance=numerical_tolerance,
            saving_tolerance=saving_tolerance,
            target_energy_tolerance=target_energy_tolerance,
            battery_stress_tolerance=battery_stress_tolerance,
            ready_tolerance_minutes=ready_tolerance_minutes,
        )
        if not allowed:
            rejected_reasons[offer.offer_id] = (
                "insufficient_saving_difference"
                if any(
                    abs(offer.advertised_saving - item.advertised_saving)
                    < _display_meaningful_saving(bau, parameters)
                    for item in current
                )
                else "too_close_in_ready_time"
            )
            return False
        flags: set[str] = set()
        if reason == "bau_anchor":
            flags.add("bau")
        elif reason == "target_maximum_saving":
            flags.add("maximum_saving")
        elif reason == "target_least_degradation":
            flags.add("least_degradation")
        else:
            flags.add("intermediate")
        if used_exception:
            reason = "close_ready_saving_stress_tradeoff"
        selected_info[offer.offer_id] = (reason, score, flags)
        return True

    for target_key, member_list in sorted(groups.items()):
        group = tuple(member_list)
        target_groups[target_key] = group
        bau = _display_bau_offer(group, source_by_id)
        if bau is None:
            raise PhysicalConstraintError("Every target group must contain a BAU anchor.")
        assert bau is not None
        bau_offer: MenuOffer = bau
        positive = tuple(
            item for item in group if item.advertised_saving > positive_saving_tolerance
        )
        add_offer(bau, "bau_anchor", group, bau)
        if not positive:
            continue
        maximum = max(
            positive,
            key=lambda item: (
                item.advertised_saving,
                -_offer_ready_absolute_minute(item),
                -(item.raw_battery_stress or 0.0),
                -item.charging_cost,
                item.offer_id,
            ),
        )
        least = min(
            positive,
            key=lambda item: (
                item.raw_battery_stress or 0.0,
                -item.advertised_saving,
                _offer_ready_absolute_minute(item),
                item.offer_id,
            ),
        )
        earliest = min(
            positive,
            key=lambda item: (_offer_ready_absolute_minute(item), item.offer_id),
        )
        latest = max(
            positive,
            key=lambda item: (_offer_ready_absolute_minute(item), item.offer_id),
        )
        anchor_candidates: tuple[tuple[MenuOffer, SelectionReason], ...] = (
            (maximum, "target_maximum_saving"),
            (least, "target_least_degradation"),
        )
        for candidate, reason in anchor_candidates:
            meaningful = (
                _offer_ready_absolute_minute(candidate) - _offer_ready_absolute_minute(bau)
                >= parameters.minimum_ready_separation_minutes
                or candidate.advertised_saving >= _display_meaningful_saving(bau, parameters)
                or (
                    candidate.advertised_saving > positive_saving_tolerance
                    and (candidate.raw_battery_stress or 0.0) < (bau.raw_battery_stress or 0.0)
                    and _display_stress_difference(candidate, bau, max(numerical_tolerance, 1e-12))
                    >= parameters.minimum_relative_stress_difference
                )
            )
            if (
                meaningful
                and len(
                    [
                        item
                        for item in selected_info
                        if _display_target_key(
                            next(offer for offer in ordered if offer.offer_id == item)
                        )
                        == target_key
                    ]
                )
                < parameters.maximum_offers_per_target
            ):
                add_offer(candidate, reason, group, bau)
        target_selected = [item for item in group if item.offer_id in selected_info]
        if not any(item.advertised_saving > positive_saving_tolerance for item in target_selected):
            for candidate in (maximum, least, earliest, latest):
                if candidate.advertised_saving <= positive_saving_tolerance:
                    continue
                if len(target_selected) >= parameters.maximum_offers_per_target:
                    break
                if add_offer(
                    candidate,
                    "target_positive_saving_coverage",
                    group,
                    bau,
                ):
                    target_selected = [item for item in group if item.offer_id in selected_info]
                    break
        while (
            len([item for item in group if item.offer_id in selected_info])
            < parameters.maximum_offers_per_target
        ):
            selected_group = tuple(item for item in group if item.offer_id in selected_info)
            candidates = [item for item in positive if item.offer_id not in selected_info]
            if not candidates:
                break
            candidate = max(
                candidates,
                key=lambda item: (
                    min(
                        _display_distance(item, current, group, bau_offer, numerical_tolerance)
                        for current in selected_group
                    ),
                    1 if (source_by_id[item.offer_id].endpoint_role == "intermediate") else 0,
                    item.advertised_saving,
                    -(item.raw_battery_stress or 0.0),
                    -_offer_ready_absolute_minute(item),
                    item.offer_id,
                ),
            )
            score = min(
                _display_distance(candidate, current, group, bau_offer, numerical_tolerance)
                for current in selected_group
            )
            if not add_offer(candidate, "intermediate_diversity", group, bau, score):
                rejected_reasons.setdefault(candidate.offer_id, "lower_diversity_contribution")
                positive = tuple(item for item in positive if item.offer_id != candidate.offer_id)

    def selected_offer_items() -> tuple[MenuOffer, ...]:
        return tuple(item for item in ordered if item.offer_id in selected_info)

    if parameters.maximum_displayed_offers is not None:
        while len(selected_info) > parameters.maximum_displayed_offers:
            current_selected = selected_offer_items()
            optional = [
                item
                for item in current_selected
                if not _is_bau_offer(item, source_by_id[item.offer_id])
                and sum(
                    1
                    for other in current_selected
                    if _display_target_key(other) == _display_target_key(item)
                    and other.advertised_saving > positive_saving_tolerance
                )
                > 1
            ]
            if not optional:
                raise PhysicalConstraintError(
                    "display diversity constraints cannot satisfy maximum_displayed_offers "
                    "while preserving BAU and positive target coverage."
                )
            contributions: dict[str, float] = {}
            for item in optional:
                group = target_groups[_display_target_key(item)]
                bau = _display_bau_offer(group, source_by_id)
                assert bau is not None
                peers = [
                    other
                    for other in current_selected
                    if other.offer_id != item.offer_id
                    and _display_target_key(other) == _display_target_key(item)
                ]
                contributions[item.offer_id] = min(
                    (
                        _display_distance(item, peer, group, bau_offer, numerical_tolerance)
                        for peer in peers
                    ),
                    default=0.0,
                )
            remove = min(
                optional,
                key=lambda item: (
                    contributions[item.offer_id],
                    item.advertised_saving,
                    item.raw_battery_stress or 0.0,
                    min(
                        abs(_offer_ready_absolute_minute(item) - _offer_ready_absolute_minute(peer))
                        for peer in current_selected
                        if peer.offer_id != item.offer_id
                        and _display_target_key(peer) == _display_target_key(item)
                    ),
                    item.offer_id,
                ),
            )
            selected_info.pop(remove.offer_id)
            rejected_reasons[remove.offer_id] = "global_limit"

    selected = selected_offer_items()
    selected_by_target: dict[tuple[float, float], list[MenuOffer]] = {}
    for item in selected:
        selected_by_target.setdefault(_display_target_key(item), []).append(item)
    derived_selected: list[MenuOffer] = []
    for target_key, selected_members in sorted(selected_by_target.items()):
        ordered_group = tuple(
            sorted(
                selected_members,
                key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind),
            )
        )
        for rank, item in enumerate(ordered_group, start=1):
            reason, score, categories = selected_info[item.offer_id]
            derived_selected.append(
                replace(
                    item,
                    is_selected_bau_anchor="bau" in categories,
                    is_selected_maximum_saving="maximum_saving" in categories,
                    is_selected_least_degradation="least_degradation" in categories,
                    is_selected_intermediate="intermediate" in categories,
                    selection_reason=reason,
                    selection_rank_within_target=rank,
                    diversity_score=score,
                )
            )
    displayed = tuple(
        sorted(
            derived_selected,
            key=lambda item: _display_sort_key(item, source_by_id[item.offer_id].source_kind),
        )
    )
    selected_ids = {item.offer_id for item in displayed}
    target_summaries: list[TargetDisplaySelectionSummary] = []
    for target_key, group_members in sorted(groups.items()):
        group = tuple(group_members)
        target_selected_members = tuple(
            item for item in displayed if _display_target_key(item) == target_key
        )
        ready_times = tuple(_offer_ready_absolute_minute(item) for item in target_selected_members)
        pairwise = [
            abs(left - right)
            for index, left in enumerate(ready_times)
            for right in ready_times[index + 1 :]
        ]
        target_summaries.append(
            TargetDisplaySelectionSummary(
                target_soc=group[0].target_soc,
                target_energy_kwh=_offer_target_energy(group[0]),
                input_pareto_count=len(group),
                bau_count=sum(
                    1 for item in group if _is_bau_offer(item, source_by_id[item.offer_id])
                ),
                selected_count=len(target_selected_members),
                selected_offer_ids=tuple(item.offer_id for item in target_selected_members),
                selected_ready_times=ready_times,
                roles_present=tuple(
                    item.scientific_role or "intermediate" for item in target_selected_members
                ),
                minimum_pairwise_ready_separation=min(pairwise) if pairwise else None,
            )
        )
    decisions: list[DisplaySelectionDecision] = []
    for offer in ordered:
        nearest = _display_nearest(offer, displayed)
        if offer.offer_id in selected_ids:
            reason, score, _ = selected_info[offer.offer_id]
            decision_reason: SelectionReason = reason
            decision_score = score
        else:
            decision_reason = rejected_reasons.get(offer.offer_id, "lower_diversity_contribution")
            decision_score = None
        decisions.append(
            DisplaySelectionDecision(
                offer_id=offer.offer_id,
                selected=offer.offer_id in selected_ids,
                reason=decision_reason,
                nearest_selected_offer_id=nearest.offer_id if nearest is not None else None,
                ready_difference_minutes=(
                    abs(_offer_ready_absolute_minute(offer) - _offer_ready_absolute_minute(nearest))
                    if nearest is not None
                    else None
                ),
                saving_difference=(
                    abs(offer.advertised_saving - nearest.advertised_saving)
                    if nearest is not None
                    else None
                ),
                relative_stress_difference=(
                    _display_stress_difference(offer, nearest, max(numerical_tolerance, 1e-12))
                    if nearest is not None
                    else None
                ),
                diversity_score=decision_score,
            )
        )
    positive_coverage = sum(
        1
        for group in groups.values()
        if any(
            item.offer_id in selected_ids and item.advertised_saving > positive_saving_tolerance
            for item in group
        )
    )
    summary = DisplaySelectionSummary(
        pareto_input_count=len(ordered),
        target_count=len(groups),
        bau_anchor_count=sum(
            1 for item in displayed if _is_bau_offer(item, source_by_id[item.offer_id])
        ),
        positive_target_coverage_count=positive_coverage,
        selected_count=len(displayed),
        removed_count=len(ordered) - len(displayed),
        minimum_ready_separation_minutes=parameters.minimum_ready_separation_minutes,
        maximum_offers_per_target=parameters.maximum_offers_per_target,
        maximum_displayed_offers=parameters.maximum_displayed_offers,
    )
    return displayed, summary, tuple(target_summaries), tuple(decisions)


def _offer_total_fade(offer: MenuOffer) -> float:
    """Return the raw total fade used by display diagnostics and tie-breaks."""
    if offer.total_capacity_fade is not None:
        return float(offer.total_capacity_fade)
    if offer.raw_battery_stress is not None:
        return float(offer.raw_battery_stress)
    return 0.0


def _distinctness_distance(
    a: MenuOffer,
    b: MenuOffer,
    universe: tuple[MenuOffer, ...],
    parameters: DistinctnessParameters,
) -> float:
    """Distance over ready delay, target SOC, saving, and raw total fade."""
    values = tuple(
        (
            float(_offer_ready_absolute_minute(item)),
            float(item.target_soc),
            float(item.advertised_saving),
            _offer_total_fade(item),
        )
        for item in universe
    )
    left = (
        float(_offer_ready_absolute_minute(a)),
        float(a.target_soc),
        float(a.advertised_saving),
        _offer_total_fade(a),
    )
    right = (
        float(_offer_ready_absolute_minute(b)),
        float(b.target_soc),
        float(b.advertised_saving),
        _offer_total_fade(b),
    )
    weights = (
        parameters.ready_weight,
        parameters.target_weight,
        parameters.saving_weight,
        parameters.fade_weight,
    )
    return sqrt(
        sum(
            weight
            * (
                (left[index] - right[index])
                / (
                    1.0
                    if index == 1
                    else max(
                        max(item[index] for item in values) - min(item[index] for item in values),
                        parameters.epsilon,
                    )
                )
            )
            ** 2
            for index, weight in enumerate(weights)
        )
    )


def _display_distinctness_report(
    displayed: tuple[MenuOffer, ...],
    selected_info: dict[str, tuple[SelectionReason, float | None, int]],
    parameters: DistinctnessParameters,
) -> MenuDistinctnessDiagnostics:
    """Build deterministic, non-scientific-spread diagnostics for displayed offers."""
    if not displayed:
        return MenuDistinctnessDiagnostics(
            feature_definitions=(
                ("ready_delay", "absolute ready minute; min-max scaled"),
                ("target_soc", "SOC fraction; fixed [0,1] scaling"),
                ("saving", "same-target currency saving; min-max scaled"),
                ("total_capacity_fade", "raw capacity-fade fraction; min-max scaled"),
            ),
            pairwise_metrics=(),
            minimum_pairwise_distance=None,
            mean_pairwise_distance=None,
            median_pairwise_distance=None,
            mean_nearest_neighbour_distance=None,
            minimum_nearest_neighbour_distance=None,
            ready_range=None,
            target_range=None,
            saving_range=None,
            fade_range=None,
            offers_per_target=(),
            target_coverage_count=0,
            target_share_entropy=0.0,
            near_duplicate_pair_count=0,
            warning_threshold=parameters.distinctness_warning_threshold,
            option_metrics=(),
            closest_pairs=(),
            most_distinct_pairs=(),
        )

    raw = {
        offer.offer_id: (
            float(_offer_ready_absolute_minute(offer)),
            float(offer.target_soc),
            float(offer.advertised_saving),
            _offer_total_fade(offer),
        )
        for offer in displayed
    }
    mins = tuple(min(values[index] for values in raw.values()) for index in range(4))
    maxs = tuple(max(values[index] for values in raw.values()) for index in range(4))

    def scaled(values: tuple[float, float, float, float]) -> tuple[float, ...]:
        return tuple(
            value
            if index == 1
            else (value - mins[index]) / max(maxs[index] - mins[index], parameters.epsilon)
            for index, value in enumerate(values)
        )

    coordinates = {offer_id: scaled(values) for offer_id, values in raw.items()}
    weights = (
        parameters.ready_weight,
        parameters.target_weight,
        parameters.saving_weight,
        parameters.fade_weight,
    )
    pairs: list[PairwiseDistinctness] = []
    for index, left in enumerate(displayed):
        for right in displayed[index + 1 :]:
            differences = tuple(
                abs(coordinates[left.offer_id][dimension] - coordinates[right.offer_id][dimension])
                for dimension in range(4)
            )
            pairs.append(
                PairwiseDistinctness(
                    offer_a=left.offer_id,
                    offer_b=right.offer_id,
                    ready_distance=differences[0],
                    target_distance=differences[1],
                    saving_distance=differences[2],
                    fade_distance=differences[3],
                    overall_distance=sqrt(
                        sum(
                            weight * difference * difference
                            for weight, difference in zip(weights, differences, strict=True)
                        )
                    ),
                )
            )
    ordered_pairs = tuple(
        sorted(pairs, key=lambda pair: (pair.overall_distance, pair.offer_a, pair.offer_b))
    )
    nearest_by_id: dict[str, PairwiseDistinctness] = {}
    for pair in ordered_pairs:
        nearest_by_id.setdefault(pair.offer_a, pair)
        nearest_by_id.setdefault(pair.offer_b, pair)
    distances = tuple(pair.overall_distance for pair in ordered_pairs)
    nearest_distances = tuple(pair.overall_distance for pair in nearest_by_id.values())
    target_counts: dict[float, int] = {}
    for offer in displayed:
        target_counts[offer.target_soc] = target_counts.get(offer.target_soc, 0) + 1
    total = len(displayed)
    entropy = -sum((count / total) * log(count / total) for count in target_counts.values())

    def distinctness_label(distance: float) -> str:
        if distance < parameters.distinctness_warning_threshold:
            return "near_duplicate"
        if distance < parameters.distinctness_warning_threshold * 2.0:
            return "weakly_distinct"
        if distance < parameters.distinctness_warning_threshold * 4.0:
            return "moderately_distinct"
        return "highly_distinct"

    option_metrics: list[OfferDistinctnessDiagnostics] = []
    for offer in displayed:
        nearest = nearest_by_id.get(offer.offer_id)
        reason, contribution, order = selected_info[offer.offer_id]
        option_metrics.append(
            OfferDistinctnessDiagnostics(
                offer_id=offer.offer_id,
                nearest_offer_id=(
                    nearest.offer_b
                    if nearest is not None and nearest.offer_a == offer.offer_id
                    else nearest.offer_a
                    if nearest is not None
                    else None
                ),
                nearest_offer_distance=nearest.overall_distance if nearest else None,
                nearest_ready_distance=nearest.ready_distance if nearest else None,
                nearest_target_distance=nearest.target_distance if nearest else None,
                nearest_saving_distance=nearest.saving_distance if nearest else None,
                nearest_fade_distance=nearest.fade_distance if nearest else None,
                marginal_diversity_contribution=contribution,
                selection_order=order,
                selection_reason=reason,
                distinctness_label=distinctness_label(nearest.overall_distance)
                if nearest
                else "highly_distinct",
            )
        )
    return MenuDistinctnessDiagnostics(
        feature_definitions=(
            ("ready_delay", "absolute ready minute; min-max scaled"),
            ("target_soc", "SOC fraction; fixed [0,1] scaling"),
            ("saving", "same-target currency saving; min-max scaled"),
            ("total_capacity_fade", "raw capacity-fade fraction; min-max scaled"),
        ),
        pairwise_metrics=ordered_pairs,
        minimum_pairwise_distance=min(distances) if distances else None,
        mean_pairwise_distance=sum(distances) / len(distances) if distances else None,
        median_pairwise_distance=median(distances) if distances else None,
        mean_nearest_neighbour_distance=(
            sum(nearest_distances) / len(nearest_distances) if nearest_distances else None
        ),
        minimum_nearest_neighbour_distance=min(nearest_distances) if nearest_distances else None,
        ready_range=(mins[0], maxs[0]),
        target_range=(mins[1], maxs[1]),
        saving_range=(mins[2], maxs[2]),
        fade_range=(mins[3], maxs[3]),
        offers_per_target=tuple(sorted(target_counts.items())),
        target_coverage_count=len(target_counts),
        target_share_entropy=entropy,
        near_duplicate_pair_count=sum(
            pair.overall_distance < parameters.distinctness_warning_threshold
            for pair in ordered_pairs
        ),
        warning_threshold=parameters.distinctness_warning_threshold,
        option_metrics=tuple(sorted(option_metrics, key=lambda item: item.offer_id)),
        closest_pairs=ordered_pairs[:5],
        most_distinct_pairs=tuple(
            sorted(
                ordered_pairs, key=lambda pair: (-pair.overall_distance, pair.offer_a, pair.offer_b)
            )[:5]
        ),
    )


def _display_diversity_selection(
    offers: tuple[MenuOffer, ...],
    source_by_id: dict[str, OfferSource],
    parameters: DisplayDiversityParameters,
    *,
    numerical_tolerance: float,
    positive_saving_tolerance: float,
    saving_tolerance: float,
    target_energy_tolerance: float,
    battery_stress_tolerance: float,
    ready_tolerance_minutes: int,
) -> tuple[
    tuple[MenuOffer, ...],
    DisplaySelectionSummary,
    tuple[TargetDisplaySelectionSummary, ...],
    tuple[DisplaySelectionDecision, ...],
    MenuDistinctnessDiagnostics,
]:
    """Select only non-BAU customer options using global farthest-point spread."""
    del saving_tolerance, target_energy_tolerance, battery_stress_tolerance, ready_tolerance_minutes
    ordered = tuple(
        sorted(
            (
                offer
                for offer in offers
                if source_by_id[offer.offer_id].source_kind != "bau"
                and offer.advertised_saving > positive_saving_tolerance
            ),
            key=lambda item: _display_sort_key(item, "optimized"),
        )
    )
    groups: dict[tuple[float, float], list[MenuOffer]] = {}
    for offer in ordered:
        groups.setdefault(_display_target_key(offer), []).append(offer)
    cap = parameters.maximum_displayed_offers
    if cap is None:
        cap = len(ordered)
    selected: list[MenuOffer] = []
    selected_info: dict[str, tuple[SelectionReason, float | None, int]] = {}
    rejected: dict[str, SelectionReason] = {}

    def add(offer: MenuOffer, reason: SelectionReason, contribution: float | None) -> None:
        if offer.offer_id in selected_info:
            return
        selected.append(offer)
        selected_info[offer.offer_id] = (reason, contribution, len(selected))

    # Minimum target coverage, then scientifically meaningful maximum/least-
    # degradation anchors where they are not identical.  A user-supplied cap
    # below the number of targets is still honoured: maximum-saving anchors
    # receive deterministic priority and unrepresented targets are visible in
    # the per-target diagnostics rather than silently exceeding the cap.
    anchor_candidates: list[tuple[MenuOffer, SelectionReason]] = []
    least_candidates: list[tuple[MenuOffer, SelectionReason]] = []
    for _target_key, members in sorted(groups.items()):
        maximum = max(
            members,
            key=lambda item: (
                item.advertised_saving,
                -_offer_total_fade(item),
                item.offer_id,
            ),
        )
        anchor_candidates.append((maximum, "target_coverage_anchor"))
        least = min(
            members,
            key=lambda item: (
                _offer_total_fade(item),
                -item.advertised_saving,
                item.offer_id,
            ),
        )
        if least.offer_id != maximum.offer_id:
            least_candidates.append((least, "least_degradation_anchor"))
    for candidate, reason in anchor_candidates:
        if len(selected) >= min(cap, len(ordered)):
            rejected[candidate.offer_id] = "global_limit"
        else:
            add(candidate, reason, None)
    for candidate, reason in least_candidates:
        if len(selected) >= min(cap, len(ordered)):
            rejected[candidate.offer_id] = "global_limit"
        else:
            add(candidate, reason, None)
    while len(selected) < min(cap, len(ordered)):
        candidates = [offer for offer in ordered if offer.offer_id not in selected_info]
        candidates = [
            offer
            for offer in candidates
            if sum(item.target_soc == offer.target_soc for item in selected)
            < parameters.maximum_offers_per_target
        ]
        if not candidates:
            break

        def score(candidate: MenuOffer) -> tuple[float, float, float, float, float, str]:
            distances = [
                _distinctness_distance(candidate, current, tuple(ordered), parameters.distinctness)
                for current in selected
            ]
            minimum = min(distances) if distances else float("inf")
            return (
                minimum,
                candidate.advertised_saving,
                -_offer_total_fade(candidate),
                -_offer_ready_absolute_minute(candidate),
                candidate.target_soc,
                candidate.offer_id,
            )

        chosen = max(candidates, key=score)
        contribution = score(chosen)[0]
        if selected and contribution < parameters.distinctness.minimum_overall_distinctness:
            rejected[chosen.offer_id] = "anchor_suppressed_similarity"
            break
        add(chosen, "farthest_point_diversity", contribution)
    selected_tuple = tuple(sorted(selected, key=lambda item: _display_sort_key(item, "optimized")))
    selected_ids = {item.offer_id for item in selected_tuple}
    target_ranks: dict[float, int] = {}
    rank_by_id: dict[str, int] = {}
    for item in selected_tuple:
        target_ranks[item.target_soc] = target_ranks.get(item.target_soc, 0) + 1
        rank_by_id[item.offer_id] = target_ranks[item.target_soc]
    derived = tuple(
        replace(
            item,
            is_selected_bau_anchor=False,
            is_selected_maximum_saving=selected_info[item.offer_id][0] == "target_coverage_anchor",
            is_selected_least_degradation=selected_info[item.offer_id][0]
            == "least_degradation_anchor",
            is_selected_intermediate=selected_info[item.offer_id][0] == "farthest_point_diversity",
            selection_reason=selected_info[item.offer_id][0],
            selection_rank_within_target=rank_by_id[item.offer_id],
            diversity_score=selected_info[item.offer_id][1],
        )
        for item in selected_tuple
    )
    target_summaries = tuple(
        TargetDisplaySelectionSummary(
            target_soc=members[0].target_soc,
            target_energy_kwh=_offer_target_energy(members[0]),
            input_pareto_count=len(members),
            bau_count=0,
            selected_count=sum(item.target_soc == target_key[1] for item in derived),
            selected_offer_ids=tuple(
                item.offer_id for item in derived if item.target_soc == target_key[1]
            ),
            selected_ready_times=tuple(
                _offer_ready_absolute_minute(item)
                for item in derived
                if item.target_soc == target_key[1]
            ),
            roles_present=tuple(
                sorted(
                    {
                        item.scientific_role or "intermediate"
                        for item in derived
                        if item.target_soc == target_key[1]
                    }
                )
            ),
            minimum_pairwise_ready_separation=None,
        )
        for target_key, members in sorted(groups.items())
    )
    distinctness = _display_distinctness_report(derived, selected_info, parameters.distinctness)
    nearest_by_id = {metric.offer_id: metric for metric in distinctness.option_metrics}
    decisions = tuple(
        DisplaySelectionDecision(
            offer_id=offer.offer_id,
            selected=offer.offer_id in selected_ids,
            reason=selected_info[offer.offer_id][0]
            if offer.offer_id in selected_info
            else rejected.get(offer.offer_id, "lower_diversity_contribution"),
            nearest_selected_offer_id=(
                nearest_by_id[offer.offer_id].nearest_offer_id
                if offer.offer_id in selected_ids and offer.offer_id in nearest_by_id
                else None
            ),
            diversity_score=selected_info[offer.offer_id][1]
            if offer.offer_id in selected_info
            else None,
        )
        for offer in offers
    )
    summary = DisplaySelectionSummary(
        pareto_input_count=len(offers),
        target_count=len(groups),
        bau_anchor_count=0,
        positive_target_coverage_count=sum(
            any(item.target_soc == key[1] for item in derived) for key in groups
        ),
        selected_count=len(derived),
        removed_count=len(offers) - len(derived),
        minimum_ready_separation_minutes=parameters.minimum_ready_separation_minutes,
        maximum_offers_per_target=parameters.maximum_offers_per_target,
        maximum_displayed_offers=parameters.maximum_displayed_offers,
    )
    return derived, summary, target_summaries, decisions, distinctness


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
                item.raw_battery_stress or 0.0,
                item.ready_step,
                -item.target_soc,
                item.offer_id,
            ),
        )
        max_health = min(
            non_bau,
            key=lambda item: (
                item.raw_battery_stress or 0.0,
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
                    item.raw_battery_stress or 0.0,
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
            item.raw_battery_stress or 0.0,
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
    role_order = {
        "bau": 0,
        "least_and_maximum": 1,
        "least_degradation": 2,
        "maximum_saving": 3,
        "low_saving": 4,
        "intermediate": 5,
    }
    role = "bau" if is_bau else (offer.scientific_role or "intermediate")
    return (
        offer.ready_step,
        offer.target_soc,
        offer.advertised_saving,
        role_order.get(role, 99),
        offer.raw_battery_stress or 0.0,
        tuple(_quantize(value) for value in offer.profile.grid_energy_kwh),
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
