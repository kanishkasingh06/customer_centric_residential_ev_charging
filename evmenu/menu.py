"""Deterministic candidate-menu generation for one EV charging session.

Commit 4 combines the validated target, feasibility, and analytical profile
layers from Commits 1--3.  It deliberately stops before degradation scoring,
Pareto filtering, customer-choice modelling, Monte Carlo simulation, and
network analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

from .exceptions import PhysicalConstraintError, SchemaValidationError
from .feasibility import build_target_options, evaluate_request_feasibility
from .profiles import (
    ConstructedProfile,
    build_immediate_charging_profile,
    build_minimum_cost_charging_profile,
)
from .schemas import (
    ChargingProfile,
    ChargingSession,
    EVSpec,
    MenuSettings,
    PlanningSignal,
    TargetOption,
    TargetSource,
)
from .validation import ValidationReport, ValidationTolerances

CandidateKind = Literal["immediate_bau", "minimum_cost"]
_TARGET_SOURCE_ORDER: dict[TargetSource, int] = {
    "minimum_required": 0,
    "standard_80": 1,
    "standard_90": 2,
    "standard_100": 3,
}


@dataclass(frozen=True, slots=True)
class MenuCandidate:
    """One physically validated pre-degradation menu candidate.

    ``same_target_bau_cost`` always refers to immediate charging to the same
    target SOC.  Consequently, ``saving`` is comparable within and across ready
    times for that target.  Battery-degradation attributes are intentionally
    absent until the literature model is added in a later commit.
    """

    candidate_id: str
    ev_id: str
    kind: CandidateKind
    target_soc: float
    target_sources: tuple[TargetSource, ...]
    target_label: str
    ready_step: int
    charging_cost: float
    same_target_bau_cost: float
    saving: float
    required_grid_energy_kwh: float
    profile: ChargingProfile
    validation: ValidationReport

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_id", self.candidate_id),
            ("ev_id", self.ev_id),
            ("target_label", self.target_label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if self.kind not in ("immediate_bau", "minimum_cost"):
            raise SchemaValidationError("kind must be 'immediate_bau' or 'minimum_cost'.")
        if isinstance(self.ready_step, bool) or not isinstance(self.ready_step, int):
            raise SchemaValidationError("ready_step must be an integer.")
        if self.ready_step < 0:
            raise SchemaValidationError("ready_step must be non-negative.")
        for numeric_name, numeric_value in (
            ("target_soc", self.target_soc),
            ("charging_cost", self.charging_cost),
            ("same_target_bau_cost", self.same_target_bau_cost),
            ("saving", self.saving),
            ("required_grid_energy_kwh", self.required_grid_energy_kwh),
        ):
            _finite_real(numeric_name, numeric_value)
        if not 0.0 <= self.target_soc <= 1.0:
            raise PhysicalConstraintError("target_soc must lie in [0, 1].")
        if not isinstance(self.target_sources, tuple):
            raise SchemaValidationError("target_sources must be a tuple.")
        if self.required_grid_energy_kwh < 0.0:
            raise PhysicalConstraintError("required_grid_energy_kwh must be non-negative.")
        if not self.target_sources:
            raise SchemaValidationError("target_sources cannot be empty.")
        if any(
            not isinstance(source, str) or source not in _TARGET_SOURCE_ORDER
            for source in self.target_sources
        ):
            raise SchemaValidationError("target_sources contains an unsupported target source.")
        if len(set(self.target_sources)) != len(self.target_sources):
            raise SchemaValidationError("target_sources cannot contain duplicates.")
        if self.target_sources != tuple(
            sorted(self.target_sources, key=_TARGET_SOURCE_ORDER.__getitem__)
        ):
            raise SchemaValidationError("target_sources must use deterministic source ordering.")
        if not isinstance(self.profile, ChargingProfile):
            raise SchemaValidationError("profile must be a ChargingProfile instance.")
        if not isinstance(self.validation, ValidationReport) or not self.validation.is_valid:
            raise PhysicalConstraintError("MenuCandidate requires a valid validation report.")


@dataclass(frozen=True, slots=True)
class GeneratedMenu:
    """Deterministic pre-degradation menu for one EV and one session."""

    ev_id: str
    candidates: tuple[MenuCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ev_id, str) or not self.ev_id.strip():
            raise SchemaValidationError("ev_id must be a non-empty string.")
        object.__setattr__(self, "ev_id", self.ev_id.strip())
        try:
            candidate_tuple = tuple(self.candidates)
        except TypeError as exc:
            raise SchemaValidationError(
                "candidates must be an iterable of MenuCandidate objects."
            ) from exc
        object.__setattr__(self, "candidates", candidate_tuple)
        if not candidate_tuple:
            raise PhysicalConstraintError("generated menu cannot be empty.")
        if any(not isinstance(candidate, MenuCandidate) for candidate in candidate_tuple):
            raise SchemaValidationError("candidates must contain only MenuCandidate objects.")
        if any(candidate.ev_id != self.ev_id for candidate in candidate_tuple):
            raise SchemaValidationError("all candidates must belong to the generated-menu EV.")
        ids = tuple(candidate.candidate_id for candidate in candidate_tuple)
        if len(set(ids)) != len(ids):
            raise SchemaValidationError("candidate identifiers must be unique.")

    def candidates_for_target(self, target_soc: float) -> tuple[MenuCandidate, ...]:
        """Return candidates matching a target within exact stored precision."""
        target = _finite_real("target_soc", target_soc)
        return tuple(candidate for candidate in self.candidates if candidate.target_soc == target)


@dataclass(frozen=True, slots=True)
class MenuGenerationSettings:
    """Settings local to deterministic candidate generation."""

    include_immediate_bau: bool = True
    include_minimum_cost: bool = True
    deduplicate_identical_profiles: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("include_immediate_bau", self.include_immediate_bau),
            ("include_minimum_cost", self.include_minimum_cost),
            ("deduplicate_identical_profiles", self.deduplicate_identical_profiles),
        ):
            if not isinstance(value, bool):
                raise SchemaValidationError(f"{name} must be a bool.")
        if not self.include_immediate_bau:
            raise SchemaValidationError("include_immediate_bau must remain True.")


def generate_candidate_menu(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    menu_settings: MenuSettings | None = None,
    generation_settings: MenuGenerationSettings | None = None,
    validation_tolerances: ValidationTolerances | None = None,
) -> GeneratedMenu:
    """Generate validated immediate and minimum-cost candidates.

    For each personalized/standard target, immediate charging defines the
    same-target BAU reference.  Minimum-cost candidates are generated for every
    exactly feasible ready boundary.  When two requests produce the same target
    and identical grid-energy trajectory, only the earliest ready promise is
    retained because any later promise is weakly dominated before degradation
    is considered.
    """
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec instance.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession instance.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal instance.")
    settings = MenuSettings() if menu_settings is None else menu_settings
    generation = MenuGenerationSettings() if generation_settings is None else generation_settings
    tolerances = ValidationTolerances() if validation_tolerances is None else validation_tolerances
    if not isinstance(settings, MenuSettings):
        raise SchemaValidationError("menu_settings must be a MenuSettings instance.")
    if not isinstance(generation, MenuGenerationSettings):
        raise SchemaValidationError(
            "generation_settings must be a MenuGenerationSettings instance."
        )
    if not isinstance(tolerances, ValidationTolerances):
        raise SchemaValidationError(
            "validation_tolerances must be a ValidationTolerances instance."
        )

    session.validate_for_ev(ev)
    signal.validate_session_window(session)
    target_options = build_target_options(ev, session, settings)
    candidates: list[MenuCandidate] = []

    for target_index, target in enumerate(target_options):
        bau = build_immediate_charging_profile(
            ev=ev,
            session=session,
            signal=signal,
            target_soc=target.target_soc,
            tolerances=tolerances,
        )
        target_key = _target_key(target_index, target.target_soc)
        candidates.append(
            _candidate_from_constructed(
                ev=ev,
                target_sources=target.sources,
                target_label=target.label,
                target_key=target_key,
                kind="immediate_bau",
                constructed=bau,
                bau_cost=bau.charging_cost,
                saving=0.0,
            )
        )

        if not generation.include_minimum_cost:
            continue

        seen_minimum_cost: set[tuple[object, ...]] = set()

        for ready_step in range(session.arrival_step, session.departure_step + 1):
            feasibility = evaluate_request_feasibility(
                ev,
                session,
                signal,
                target_soc=target.target_soc,
                ready_step=ready_step,
                tolerance=0.0,
            )
            if not feasibility.is_feasible:
                continue
            constructed = build_minimum_cost_charging_profile(
                ev=ev,
                session=session,
                signal=signal,
                target_soc=target.target_soc,
                ready_step=ready_step,
                tolerances=tolerances,
            )
            saving = bau.charging_cost - constructed.charging_cost
            if not isfinite(saving):
                raise SchemaValidationError("candidate saving must be finite.")
            duplicate_key = _minimum_cost_duplicate_key(target, constructed, saving)
            if generation.deduplicate_identical_profiles and duplicate_key in seen_minimum_cost:
                continue
            seen_minimum_cost.add(duplicate_key)
            candidates.append(
                _candidate_from_constructed(
                    ev=ev,
                    target_sources=target.sources,
                    target_label=target.label,
                    target_key=target_key,
                    kind="minimum_cost",
                    constructed=constructed,
                    bau_cost=bau.charging_cost,
                    saving=saving,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            candidate.target_soc,
            candidate.ready_step,
            0 if candidate.kind == "immediate_bau" else 1,
            candidate.charging_cost,
            candidate.candidate_id,
        )
    )
    return GeneratedMenu(ev_id=ev.ev_id, candidates=tuple(candidates))


def _candidate_from_constructed(
    *,
    ev: EVSpec,
    target_sources: tuple[TargetSource, ...],
    target_label: str,
    target_key: str,
    kind: CandidateKind,
    constructed: ConstructedProfile,
    bau_cost: float,
    saving: float,
) -> MenuCandidate:
    suffix = "bau" if kind == "immediate_bau" else f"mc-r{constructed.ready_step}"
    return MenuCandidate(
        candidate_id=f"{ev.ev_id}-{target_key}-{suffix}",
        ev_id=ev.ev_id,
        kind=kind,
        target_soc=constructed.target_soc,
        target_sources=target_sources,
        target_label=target_label,
        ready_step=constructed.ready_step,
        charging_cost=constructed.charging_cost,
        same_target_bau_cost=bau_cost,
        saving=saving,
        required_grid_energy_kwh=constructed.required_grid_energy_kwh,
        profile=constructed.profile,
        validation=constructed.validation,
    )


def _target_key(index: int, target_soc: float) -> str:
    return f"t{index:02d}-{target_soc:.6f}".replace(".", "p")


def _minimum_cost_duplicate_key(
    target: TargetOption,
    constructed: ConstructedProfile,
    saving: float,
) -> tuple[object, ...]:
    """Build the exact Commit 4 duplicate key for one target's MC candidates."""
    return (
        target.target_soc,
        target.sources,
        "minimum_cost",
        constructed.profile.grid_energy_kwh,
        constructed.profile.power_kw,
        constructed.profile.battery_energy_kwh,
        constructed.profile.soc,
        constructed.charging_cost,
        saving,
    )


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)
