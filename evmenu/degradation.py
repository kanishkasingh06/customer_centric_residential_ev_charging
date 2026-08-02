"""Chemistry-aware degradation assessment and within-menu health scoring.

The model follows the project's semi-empirical additive structure:

    session fade = charging-window calendar fade
                 + parked-day calendar fade
                 + cycle fade.

All fades are fractions of usable capacity.  Default coefficients are explicit,
replaceable calibration parameters anchored at 30 degC; they are not hidden
cell-identification claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, floor, isclose, isfinite
from numbers import Real

from .exceptions import PhysicalConstraintError, SchemaValidationError
from .menu import GeneratedMenu, MenuCandidate
from .schemas import (
    ChargingSession,
    Chemistry,
    EVSpec,
    MenuOffer,
    MenuSettings,
    PlanningSignal,
)
from .validation import ValidationTolerances, validate_charging_profile

_GAS_CONSTANT_J_PER_MOL_K = 8.314462618
_HOURS_PER_YEAR = 8760.0
_REFERENCE_AGE_YEARS = 1.0
_ARRHENIUS_MIN_EXPONENT = -745.0
_ARRHENIUS_MAX_EXPONENT = 709.0
_FADE_RELATIVE_TOLERANCE = 1e-12
_FADE_ABSOLUTE_TOLERANCE = 1e-15


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)


@dataclass(frozen=True, slots=True)
class ChemistryDegradationParameters:
    """Replaceable chemistry coefficients for the semi-empirical model."""

    calendar_a0: float
    calendar_a1: float
    calendar_a2: float
    calendar_soc_knee: float
    calendar_time_exponent: float
    activation_energy_j_per_mol: float
    cycle_reference_coefficient: float
    cycle_dod_coefficient: float = 1.0
    cycle_c_rate_coefficient: float = 1.0
    cycle_time_exponent: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("calendar_a0", self.calendar_a0),
            ("calendar_a1", self.calendar_a1),
            ("calendar_a2", self.calendar_a2),
            ("calendar_soc_knee", self.calendar_soc_knee),
            ("calendar_time_exponent", self.calendar_time_exponent),
            ("activation_energy_j_per_mol", self.activation_energy_j_per_mol),
            ("cycle_reference_coefficient", self.cycle_reference_coefficient),
            ("cycle_dod_coefficient", self.cycle_dod_coefficient),
            ("cycle_c_rate_coefficient", self.cycle_c_rate_coefficient),
            ("cycle_time_exponent", self.cycle_time_exponent),
        ):
            _finite(name, value)
        if self.calendar_a0 < 0.0 or self.calendar_a1 < 0.0 or self.calendar_a2 < 0.0:
            raise PhysicalConstraintError("calendar coefficients must be non-negative.")
        if not 0.0 <= self.calendar_soc_knee <= 1.0:
            raise PhysicalConstraintError("calendar_soc_knee must lie in [0, 1].")
        if not 0.0 < self.calendar_time_exponent <= 1.0:
            raise PhysicalConstraintError("calendar_time_exponent must lie in (0, 1].")
        if self.activation_energy_j_per_mol < 0.0:
            raise PhysicalConstraintError("activation_energy_j_per_mol must be non-negative.")
        if self.cycle_reference_coefficient < 0.0:
            raise PhysicalConstraintError("cycle_reference_coefficient must be non-negative.")
        if self.cycle_dod_coefficient < 0.0 or self.cycle_c_rate_coefficient < 0.0:
            raise PhysicalConstraintError("cycle stress coefficients must be non-negative.")
        if not 0.0 < self.cycle_time_exponent <= 1.0:
            raise PhysicalConstraintError("cycle_time_exponent must lie in (0, 1].")


# Calendar g(s) is expressed as fraction/year at 30 degC and age one year.
# Anchors: LFP 1.00% at 50% and 1.24% at 100%; NMC 1.78%, 2.41%,
# and 3.02% at 50%, 80%, and 100% SOC respectively.
DEFAULT_LFP_PARAMETERS = ChemistryDegradationParameters(
    calendar_a0=0.0100,
    calendar_a1=0.0,
    calendar_a2=0.02666666666666667,
    calendar_soc_knee=0.70,
    calendar_time_exponent=0.50,
    activation_energy_j_per_mol=24000.0,
    cycle_reference_coefficient=0.00070,
)
DEFAULT_NMC_PARAMETERS = ChemistryDegradationParameters(
    calendar_a0=0.00730,
    calendar_a1=0.0210,
    calendar_a2=0.04750,
    calendar_soc_knee=0.80,
    calendar_time_exponent=0.75,
    activation_energy_j_per_mol=24000.0,
    cycle_reference_coefficient=0.00090,
)


@dataclass(frozen=True, slots=True)
class DegradationSettings:
    """Scenario inputs for one-session degradation assessment."""

    battery_age_years: float = 3.0
    cumulative_equivalent_full_cycles: float = 300.0
    minimum_reference_fec: float = 1.0
    parked_day_hours: float = 16.0
    reference_temperature_c: float = 30.0
    fallback_temperature_c: float = 30.0
    health_score_resolution: float = 5.0
    degradation_comparison_tolerance: float = 1e-12
    reference_age_years: float = _REFERENCE_AGE_YEARS
    lfp: ChemistryDegradationParameters = DEFAULT_LFP_PARAMETERS
    nmc: ChemistryDegradationParameters = DEFAULT_NMC_PARAMETERS

    def __post_init__(self) -> None:
        for name, value in (
            ("battery_age_years", self.battery_age_years),
            ("cumulative_equivalent_full_cycles", self.cumulative_equivalent_full_cycles),
            ("minimum_reference_fec", self.minimum_reference_fec),
            ("parked_day_hours", self.parked_day_hours),
            ("reference_temperature_c", self.reference_temperature_c),
            ("fallback_temperature_c", self.fallback_temperature_c),
            ("health_score_resolution", self.health_score_resolution),
            ("degradation_comparison_tolerance", self.degradation_comparison_tolerance),
            ("reference_age_years", self.reference_age_years),
        ):
            _finite(name, value)
        if self.battery_age_years <= 0.0:
            raise PhysicalConstraintError("battery_age_years must be positive.")
        if self.cumulative_equivalent_full_cycles < 0.0:
            raise PhysicalConstraintError("cumulative_equivalent_full_cycles must be non-negative.")
        if self.minimum_reference_fec <= 0.0:
            raise PhysicalConstraintError("minimum_reference_fec must be positive.")
        if self.parked_day_hours < 0.0:
            raise PhysicalConstraintError("parked_day_hours must be non-negative.")
        if self.reference_temperature_c <= -273.15 or self.fallback_temperature_c <= -273.15:
            raise PhysicalConstraintError("temperatures must be above absolute zero.")
        if not 0.0 < self.health_score_resolution <= 100.0:
            raise PhysicalConstraintError("health_score_resolution must lie in (0, 100].")
        if self.degradation_comparison_tolerance < 0.0:
            raise PhysicalConstraintError("degradation_comparison_tolerance must be non-negative.")
        if self.reference_age_years <= 0.0:
            raise PhysicalConstraintError("reference_age_years must be positive.")
        if not isinstance(self.lfp, ChemistryDegradationParameters) or not isinstance(
            self.nmc, ChemistryDegradationParameters
        ):
            raise SchemaValidationError(
                "lfp and nmc must be ChemistryDegradationParameters instances."
            )

    def parameters_for(self, chemistry: object) -> ChemistryDegradationParameters:
        if chemistry == "LFP":
            return self.lfp
        if chemistry == "NMC":
            return self.nmc
        raise SchemaValidationError("chemistry must be exactly 'LFP' or 'NMC'.")


@dataclass(frozen=True, slots=True)
class DegradationAssessment:
    """Decomposition of incremental capacity fade for one menu candidate."""

    candidate_id: str
    ev_id: str
    chemistry: Chemistry
    charging_window_calendar_fade: float
    parked_day_calendar_fade: float
    cycle_fade: float
    total_fade: float
    annualized_degradation_pct: float
    parked_soc: float
    peak_c_rate: float

    def __post_init__(self) -> None:
        for name, value in (("candidate_id", self.candidate_id), ("ev_id", self.ev_id)):
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if self.chemistry not in ("LFP", "NMC"):
            raise SchemaValidationError("chemistry must be exactly 'LFP' or 'NMC'.")
        for numeric_name, numeric_value in (
            ("charging_window_calendar_fade", self.charging_window_calendar_fade),
            ("parked_day_calendar_fade", self.parked_day_calendar_fade),
            ("cycle_fade", self.cycle_fade),
            ("total_fade", self.total_fade),
            ("annualized_degradation_pct", self.annualized_degradation_pct),
            ("parked_soc", self.parked_soc),
            ("peak_c_rate", self.peak_c_rate),
        ):
            _finite(numeric_name, numeric_value)
        if (
            min(
                self.charging_window_calendar_fade,
                self.parked_day_calendar_fade,
                self.cycle_fade,
                self.total_fade,
                self.annualized_degradation_pct,
                self.peak_c_rate,
            )
            < 0.0
        ):
            raise PhysicalConstraintError("degradation outputs must be non-negative.")
        if not 0.0 <= self.parked_soc <= 1.0:
            raise PhysicalConstraintError("parked_soc must lie in [0, 1].")
        component_sum = (
            self.charging_window_calendar_fade + self.parked_day_calendar_fade + self.cycle_fade
        )
        if not isclose(
            self.total_fade,
            component_sum,
            rel_tol=_FADE_RELATIVE_TOLERANCE,
            abs_tol=_FADE_ABSOLUTE_TOLERANCE,
        ):
            raise PhysicalConstraintError(
                "total_fade must equal the sum of its degradation components."
            )


@dataclass(frozen=True, slots=True)
class DegradationScoredMenu:
    """Customer-facing offers plus their auditable degradation decomposition."""

    ev_id: str
    offers: tuple[MenuOffer, ...]
    assessments: tuple[DegradationAssessment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ev_id, str) or not self.ev_id.strip():
            raise SchemaValidationError("ev_id must be a non-empty string.")
        object.__setattr__(self, "ev_id", self.ev_id.strip())
        try:
            offers = tuple(self.offers)
            assessments = tuple(self.assessments)
        except TypeError as exc:
            raise SchemaValidationError("offers and assessments must be iterable.") from exc
        object.__setattr__(self, "offers", offers)
        object.__setattr__(self, "assessments", assessments)
        if not self.offers or len(self.offers) != len(self.assessments):
            raise SchemaValidationError(
                "offers and assessments must be nonempty and have equal lengths."
            )
        if any(not isinstance(offer, MenuOffer) for offer in self.offers):
            raise SchemaValidationError("offers must contain only MenuOffer objects.")
        if any(not isinstance(item, DegradationAssessment) for item in self.assessments):
            raise SchemaValidationError(
                "assessments must contain only DegradationAssessment objects."
            )
        if any(offer.ev_id != self.ev_id for offer in self.offers):
            raise SchemaValidationError("all offers must belong to ev_id.")
        offer_ids = tuple(offer.offer_id for offer in self.offers)
        if len(set(offer_ids)) != len(offer_ids):
            raise SchemaValidationError("offer identifiers must be unique.")
        for offer, assessment in zip(self.offers, self.assessments, strict=True):
            if assessment.candidate_id != offer.offer_id:
                raise SchemaValidationError("assessment candidate IDs must match offer IDs.")
            if assessment.ev_id != self.ev_id:
                raise SchemaValidationError("assessments must belong to ev_id.")


def calendar_soc_stress(soc: float, parameters: ChemistryDegradationParameters) -> float:
    """Return convex chemistry-specific calendar stress g(s)."""
    if not isinstance(parameters, ChemistryDegradationParameters):
        raise SchemaValidationError("parameters must be a ChemistryDegradationParameters instance.")
    value = _finite("soc", soc)
    if not 0.0 <= value <= 1.0:
        raise PhysicalConstraintError("soc must lie in [0, 1].")
    hinge = max(value - parameters.calendar_soc_knee, 0.0)
    return (
        parameters.calendar_a0
        + parameters.calendar_a1 * value
        + parameters.calendar_a2 * hinge * hinge
    )


def _relative_age_factor(
    *,
    age_years: float,
    alpha_time: float,
    reference_age_years: float,
) -> float:
    """Return the local time-power-law slope relative to reference age."""
    age = _finite("age_years", age_years)
    alpha = _finite("alpha_time", alpha_time)
    reference_age = _finite("reference_age_years", reference_age_years)
    if age <= 0.0:
        raise PhysicalConstraintError("age_years must be positive.")
    if not 0.0 < alpha <= 1.0:
        raise PhysicalConstraintError("alpha_time must lie in (0, 1].")
    if reference_age <= 0.0:
        raise PhysicalConstraintError("reference_age_years must be positive.")
    numerator = alpha * age ** (alpha - 1.0)
    denominator = alpha * reference_age ** (alpha - 1.0)
    factor = float(numerator / denominator)
    if not isfinite(factor):
        raise PhysicalConstraintError("relative age factor must be finite.")
    return factor


def _validate_candidate_context(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    menu_settings: MenuSettings,
) -> None:
    """Validate a candidate again against the assessment context."""
    session.validate_for_ev(ev)
    signal.validate_session_window(session)
    if candidate.ev_id != ev.ev_id:
        raise SchemaValidationError("candidate does not belong to ev.")
    if not candidate.validation.is_valid:
        raise PhysicalConstraintError("candidate has an invalid validation report.")

    profile = candidate.profile
    interval_count = session.departure_step - session.arrival_step
    if profile.start_step != session.arrival_step:
        raise PhysicalConstraintError("candidate profile starts outside the session.")
    if len(profile.power_kw) != interval_count or len(profile.grid_energy_kwh) != interval_count:
        raise PhysicalConstraintError("candidate profile does not span the session intervals.")
    if (
        len(profile.battery_energy_kwh) != interval_count + 1
        or len(profile.soc) != interval_count + 1
    ):
        raise PhysicalConstraintError("candidate profile state vectors are misaligned.")
    if not session.arrival_step <= candidate.ready_step <= session.departure_step:
        raise PhysicalConstraintError("candidate ready_step lies outside the session.")
    target_soc = _finite("candidate.target_soc", candidate.target_soc)
    if not 0.0 <= target_soc <= 1.0:
        raise PhysicalConstraintError("candidate target_soc lies outside [0, 1].")
    expected_grid_energy = sum(profile.grid_energy_kwh)
    if not isclose(
        candidate.required_grid_energy_kwh,
        expected_grid_energy,
        rel_tol=0.0,
        abs_tol=menu_settings.numerical_tolerance,
    ):
        raise PhysicalConstraintError("candidate required energy does not match its profile.")
    expected_terminal_energy = max(
        session.initial_energy_kwh,
        target_soc * ev.battery_capacity_kwh,
    )
    if candidate.kind == "immediate_bau":
        expected_ready_step = next(
            (
                profile.start_step + index
                for index, energy in enumerate(profile.battery_energy_kwh)
                if energy >= expected_terminal_energy - menu_settings.numerical_tolerance
            ),
            None,
        )
        if expected_ready_step != candidate.ready_step:
            raise PhysicalConstraintError(
                "immediate candidate ready_step does not match profile completion."
            )
    elif not candidate.candidate_id.endswith(f"-mc-r{candidate.ready_step}"):
        raise PhysicalConstraintError(
            "minimum-cost candidate ready_step does not match its identifier."
        )
    if not isclose(
        profile.battery_energy_kwh[-1],
        expected_terminal_energy,
        rel_tol=0.0,
        abs_tol=menu_settings.numerical_tolerance,
    ):
        raise PhysicalConstraintError("candidate terminal energy does not match its target.")
    expected_terminal_soc = expected_terminal_energy / ev.battery_capacity_kwh
    if not isclose(
        profile.soc[-1],
        expected_terminal_soc,
        rel_tol=0.0,
        abs_tol=menu_settings.numerical_tolerance,
    ):
        raise PhysicalConstraintError("candidate terminal SOC does not match its energy.")

    report = validate_charging_profile(
        ev=ev,
        session=session,
        signal=signal,
        target_soc=target_soc,
        ready_step=candidate.ready_step,
        profile=profile,
        tolerances=ValidationTolerances(),
    )
    if not report.is_valid:
        details = "; ".join(f"{issue.code.value}: {issue.message}" for issue in report.issues)
        raise PhysicalConstraintError(f"candidate failed context validation: {details}")


def assess_candidate_degradation(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    menu_settings: MenuSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
) -> DegradationAssessment:
    """Assess one validated candidate using additive calendar and cycle fade."""
    settings = MenuSettings() if menu_settings is None else menu_settings
    model = DegradationSettings() if degradation_settings is None else degradation_settings
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec instance.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession instance.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal instance.")
    if not isinstance(candidate, MenuCandidate):
        raise SchemaValidationError("candidate must be a MenuCandidate instance.")
    if not isinstance(settings, MenuSettings) or not isinstance(model, DegradationSettings):
        raise SchemaValidationError("invalid degradation or menu settings.")
    _validate_candidate_context(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        menu_settings=settings,
    )

    params = model.parameters_for(ev.chemistry)
    profile = candidate.profile
    age_factor = _relative_age_factor(
        age_years=model.battery_age_years,
        alpha_time=params.calendar_time_exponent,
        reference_age_years=model.reference_age_years,
    )
    window_fade = 0.0
    for local_step, soc in enumerate(profile.soc[:-1]):
        global_step = profile.start_step + local_step
        temperature_c = (
            model.fallback_temperature_c
            if signal.battery_temperature_c is None
            else signal.battery_temperature_c[global_step]
        )
        window_fade += (
            calendar_soc_stress(soc, params)
            * _temperature_factor(
                temperature_c,
                model.reference_temperature_c,
                params.activation_energy_j_per_mol,
            )
            * age_factor
            * signal.timestep_hours
            / _HOURS_PER_YEAR
        )

    delivered_energy = max(
        session.initial_energy_kwh,
        candidate.target_soc * ev.battery_capacity_kwh,
    )
    parked_soc = (delivered_energy - session.commute_energy_kwh) / ev.battery_capacity_kwh
    if not 0.0 <= parked_soc <= 1.0:
        raise PhysicalConstraintError("computed parked SOC lies outside [0, 1].")
    parked_temperature = (
        model.fallback_temperature_c
        if signal.battery_temperature_c is None
        else signal.battery_temperature_c[session.departure_step - 1]
    )
    parked_fade = (
        calendar_soc_stress(parked_soc, params)
        * _temperature_factor(
            parked_temperature,
            model.reference_temperature_c,
            params.activation_energy_j_per_mol,
        )
        * age_factor
        * model.parked_day_hours
        / _HOURS_PER_YEAR
    )

    throughput_fraction = (
        ev.charging_efficiency * sum(profile.grid_energy_kwh) / ev.battery_capacity_kwh
    )
    peak_c_rate = (
        ev.charging_efficiency * max(profile.power_kw, default=0.0) / ev.battery_capacity_kwh
    )
    effective_fec = max(
        model.cumulative_equivalent_full_cycles,
        model.minimum_reference_fec,
    )
    cycle_slope = params.cycle_time_exponent * effective_fec ** (params.cycle_time_exponent - 1.0)
    cycle_fade = (
        params.cycle_reference_coefficient
        * (1.0 + params.cycle_dod_coefficient * throughput_fraction)
        * (1.0 + params.cycle_c_rate_coefficient * peak_c_rate)
        * cycle_slope
        * throughput_fraction
    )
    total = window_fade + parked_fade + cycle_fade
    annualized_pct = total * settings.equivalent_sessions_per_year * 100.0
    return DegradationAssessment(
        candidate_id=candidate.candidate_id,
        ev_id=ev.ev_id,
        chemistry=ev.chemistry,
        charging_window_calendar_fade=window_fade,
        parked_day_calendar_fade=parked_fade,
        cycle_fade=cycle_fade,
        total_fade=total,
        annualized_degradation_pct=annualized_pct,
        parked_soc=parked_soc,
        peak_c_rate=peak_c_rate,
    )


def score_generated_menu(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    menu: GeneratedMenu,
    menu_settings: MenuSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
) -> DegradationScoredMenu:
    """Assess all candidates and assign within-menu quantized health scores."""
    if not isinstance(ev, EVSpec):
        raise SchemaValidationError("ev must be an EVSpec instance.")
    if not isinstance(session, ChargingSession):
        raise SchemaValidationError("session must be a ChargingSession instance.")
    if not isinstance(signal, PlanningSignal):
        raise SchemaValidationError("signal must be a PlanningSignal instance.")
    if not isinstance(menu, GeneratedMenu):
        raise SchemaValidationError("menu must be a GeneratedMenu instance.")
    settings = MenuSettings() if menu_settings is None else menu_settings
    model = DegradationSettings() if degradation_settings is None else degradation_settings
    if not isinstance(settings, MenuSettings) or not isinstance(model, DegradationSettings):
        raise SchemaValidationError("invalid degradation or menu settings.")
    if menu.ev_id != ev.ev_id:
        raise SchemaValidationError("menu does not belong to ev.")
    assessments = tuple(
        assess_candidate_degradation(
            ev=ev,
            session=session,
            signal=signal,
            candidate=candidate,
            menu_settings=settings,
            degradation_settings=model,
        )
        for candidate in menu.candidates
    )
    fades = tuple(item.total_fade for item in assessments)
    minimum = min(fades)
    maximum = max(fades)
    offers: list[MenuOffer] = []
    for candidate, assessment in zip(menu.candidates, assessments, strict=True):
        health = _health_score(
            assessment.total_fade,
            minimum,
            maximum,
            model.health_score_resolution,
            model.degradation_comparison_tolerance,
        )
        offers.append(
            MenuOffer(
                offer_id=candidate.candidate_id,
                ev_id=candidate.ev_id,
                target_sources=candidate.target_sources,
                ready_step=candidate.ready_step,
                target_soc=candidate.target_soc,
                charging_cost=candidate.charging_cost,
                same_target_bau_cost=candidate.same_target_bau_cost,
                advertised_saving=candidate.saving,
                incremental_degradation=assessment.total_fade,
                annualized_degradation_pct=assessment.annualized_degradation_pct,
                charging_health_score=health,
                profile=candidate.profile,
            )
        )
    return DegradationScoredMenu(
        ev_id=ev.ev_id,
        offers=tuple(offers),
        assessments=assessments,
    )


def _temperature_factor(
    temperature_c: float,
    reference_temperature_c: float,
    activation_energy_j_per_mol: float,
) -> float:
    temperature_c = _finite("temperature_c", temperature_c)
    reference_temperature_c = _finite("reference_temperature_c", reference_temperature_c)
    activation_energy_j_per_mol = _finite(
        "activation_energy_j_per_mol", activation_energy_j_per_mol
    )
    temperature_k = temperature_c + 273.15
    reference_k = reference_temperature_c + 273.15
    if temperature_k <= 0.0 or reference_k <= 0.0:
        raise PhysicalConstraintError("Kelvin temperatures must be positive.")
    exponent = (
        -activation_energy_j_per_mol
        / _GAS_CONSTANT_J_PER_MOL_K
        * (1.0 / temperature_k - 1.0 / reference_k)
    )
    if not isfinite(exponent):
        raise PhysicalConstraintError("Arrhenius exponent must be finite.")
    if not _ARRHENIUS_MIN_EXPONENT <= exponent <= _ARRHENIUS_MAX_EXPONENT:
        raise PhysicalConstraintError("Arrhenius exponent is outside the safe range.")
    return exp(exponent)


def _health_score(
    fade: float,
    minimum: float,
    maximum: float,
    resolution: float,
    comparison_tolerance: float,
) -> float:
    spread = maximum - minimum
    if spread <= comparison_tolerance:
        return 100.0
    normalized = (fade - minimum) / spread
    raw = 100.0 * (1.0 - normalized)
    quantized = resolution * floor(raw / resolution + 0.5 + 1e-12)
    return min(100.0, max(0.0, quantized))
