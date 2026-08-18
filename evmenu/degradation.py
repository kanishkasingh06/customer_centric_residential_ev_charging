"""Chemistry-specific incremental capacity-fade assessment.

The model follows the finalized two-component physical structure:

    session fade = charging-window calendar fade
                 + parked-day calendar fade
                 + cycle fade.

All fades are fractions of nominal usable capacity.  The parameter values in
this repository are literature-anchor calibrations (the supplied repository
does not contain the fitted coefficient tables from the cited papers), not
manufacturer-specific cell-identification claims.  They are immutable and
carry their source/status provenance so a later audited coefficient table can
replace them without changing the public evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, isfinite
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
BATTERY_METRIC_MODEL_ID = "semi_empirical_total_fade_v1"  # compatibility alias
BATTERY_METRIC_COMPARISON_SCOPE = "same EV model and degradation parameterization"
# These identifiers describe literature-family anchor calibrations.  They are
# deliberately not named as exact reproductions of the cited fitted models.
LFP_DEGRADATION_MODEL_ID = "naumann_lfp_capacity_fade_anchor_v1"
NMC_DEGRADATION_MODEL_ID = "schmalstieg_nmc_capacity_fade_anchor_v1"
DEGRADATION_MODEL_VERSION = "1"
PARAMETER_STATUS = "literature_anchor_calibrated"
DEFAULT_MINIMUM_EFFECTIVE_FEC = 1.0
FEC_REGULARIZATION_SOURCE = "DegradationSettings.minimum_effective_fec"
PARKED_PERIOD_DEFAULT_HOURS = 16.0
PARKED_PERIOD_DEFAULT_SOURCE = "assumed_default"
PARKED_PERIOD_USER_SOURCE = "user_input"
PARKED_PERIOD_DISABLED_SOURCE = "deferred_not_modeled"


@dataclass(frozen=True, slots=True)
class CalendarAgingParameters:
    """Chemistry-specific calendar-aging parameters and provenance.

    ``a0``, ``a1`` and ``a2`` define the convex SOC shape
    ``a0 + a1*s + a2*max(s-soc_knee, 0)**2`` at the reference temperature
    and age.  Coefficients are fractions per year at the reference point.
    """

    a0: float
    a1: float
    a2: float
    soc_knee: float
    activation_energy_j_per_mol: float
    reference_temperature_k: float
    reference_calendar_coefficient: float
    time_exponent: float

    def __post_init__(self) -> None:
        for name, value in (
            ("a0", self.a0),
            ("a1", self.a1),
            ("a2", self.a2),
            ("soc_knee", self.soc_knee),
            ("activation_energy_j_per_mol", self.activation_energy_j_per_mol),
            ("reference_temperature_k", self.reference_temperature_k),
            ("reference_calendar_coefficient", self.reference_calendar_coefficient),
            ("time_exponent", self.time_exponent),
        ):
            _finite(name, value)
        if (
            min(
                self.a0,
                self.a1,
                self.a2,
                self.activation_energy_j_per_mol,
                self.reference_calendar_coefficient,
            )
            < 0.0
        ):
            raise PhysicalConstraintError("calendar parameters must be non-negative.")
        if not 0.0 <= self.soc_knee <= 1.0:
            raise PhysicalConstraintError("soc_knee must lie in [0, 1].")
        if self.reference_temperature_k <= 0.0:
            raise PhysicalConstraintError("reference_temperature_k must be positive.")
        if not 0.0 < self.time_exponent <= 1.0:
            raise PhysicalConstraintError("time_exponent must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class CycleAgingParameters:
    """Chemistry-specific cycle-aging parameters and provenance."""

    reference_cycle_coefficient: float
    dod_coefficient: float
    c_rate_coefficient: float
    throughput_exponent: float
    activation_energy_j_per_mol: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_cycle_coefficient", self.reference_cycle_coefficient),
            ("dod_coefficient", self.dod_coefficient),
            ("c_rate_coefficient", self.c_rate_coefficient),
            ("throughput_exponent", self.throughput_exponent),
        ):
            _finite(name, value)
        if self.activation_energy_j_per_mol is not None:
            _finite("activation_energy_j_per_mol", self.activation_energy_j_per_mol)
            if self.activation_energy_j_per_mol < 0.0:
                raise PhysicalConstraintError("cycle activation energy must be non-negative.")
        if (
            min(self.reference_cycle_coefficient, self.dod_coefficient, self.c_rate_coefficient)
            < 0.0
        ):
            raise PhysicalConstraintError("cycle parameters must be non-negative.")
        if not 0.0 < self.throughput_exponent <= 1.0:
            raise PhysicalConstraintError("throughput_exponent must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class BatteryDegradationParameters:
    """Complete immutable chemistry parameter set with citation metadata."""

    chemistry: Chemistry
    calendar: CalendarAgingParameters
    cycle: CycleAgingParameters
    source_citations: tuple[str, ...]
    parameter_set_id: str
    parameter_set_version: str = "1"
    parameter_status: str = PARAMETER_STATUS
    source_title: str = ""
    source_authors: str = ""
    source_year: int | None = None
    source_equation_or_section: str = ""
    source_implementation: str = ""

    def __post_init__(self) -> None:
        if self.chemistry not in ("LFP", "NMC"):
            raise SchemaValidationError("chemistry must be exactly 'LFP' or 'NMC'.")
        if not isinstance(self.calendar, CalendarAgingParameters):
            raise SchemaValidationError("calendar must be CalendarAgingParameters.")
        if not isinstance(self.cycle, CycleAgingParameters):
            raise SchemaValidationError("cycle must be CycleAgingParameters.")
        citations = tuple(self.source_citations)
        if not citations or any(
            not isinstance(item, str) or not item.strip() for item in citations
        ):
            raise SchemaValidationError("source_citations must contain non-empty strings.")
        object.__setattr__(self, "source_citations", tuple(item.strip() for item in citations))
        for name in ("parameter_set_id", "parameter_set_version", "parameter_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        for name in (
            "source_title",
            "source_authors",
            "source_equation_or_section",
            "source_implementation",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise SchemaValidationError(f"{name} must be a string.")
            object.__setattr__(self, name, value.strip())
        if self.source_year is not None:
            if isinstance(self.source_year, bool) or not isinstance(self.source_year, int):
                raise SchemaValidationError("source_year must be an integer when supplied.")
            if self.source_year < 1900:
                raise PhysicalConstraintError("source_year must be a plausible publication year.")


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

# The public legacy coefficient objects above remain import-compatible.  The
# nested parameter sets are the canonical model representation used by new
# evaluations and diagnostics.  The citations identify the literature family;
# the status deliberately records that the supplied repository does not ship
# the original fitted coefficient table.
DEFAULT_LFP_DEGRADATION_PARAMETERS = BatteryDegradationParameters(
    chemistry="LFP",
    calendar=CalendarAgingParameters(
        a0=DEFAULT_LFP_PARAMETERS.calendar_a0,
        a1=DEFAULT_LFP_PARAMETERS.calendar_a1,
        a2=DEFAULT_LFP_PARAMETERS.calendar_a2,
        soc_knee=DEFAULT_LFP_PARAMETERS.calendar_soc_knee,
        activation_energy_j_per_mol=DEFAULT_LFP_PARAMETERS.activation_energy_j_per_mol,
        reference_temperature_k=303.15,
        reference_calendar_coefficient=1.0,
        time_exponent=0.50,
    ),
    cycle=CycleAgingParameters(
        reference_cycle_coefficient=DEFAULT_LFP_PARAMETERS.cycle_reference_coefficient,
        dod_coefficient=DEFAULT_LFP_PARAMETERS.cycle_dod_coefficient,
        c_rate_coefficient=DEFAULT_LFP_PARAMETERS.cycle_c_rate_coefficient,
        throughput_exponent=DEFAULT_LFP_PARAMETERS.cycle_time_exponent,
    ),
    source_citations=(
        "Naumann et al. (2018), Analysis and modeling of calendar aging of a commercial LFP/graphite cell",
        "Naumann et al. (2018), SimSES cell-aging parameterization (cycle-aging family)",
    ),
    parameter_set_id="naumann_lfp_literature_anchor_v1",
    source_title="Analysis and modeling of calendar aging of a commercial LiFePO4/graphite cell",
    source_authors="Naumann et al.",
    source_year=2018,
    source_equation_or_section="calendar and cycle aging equations; Section 5 methodology mapping",
    source_implementation="literature-anchor calibration; original fitted table unavailable in checkout",
)
DEFAULT_NMC_DEGRADATION_PARAMETERS = BatteryDegradationParameters(
    chemistry="NMC",
    calendar=CalendarAgingParameters(
        a0=DEFAULT_NMC_PARAMETERS.calendar_a0,
        a1=DEFAULT_NMC_PARAMETERS.calendar_a1,
        a2=DEFAULT_NMC_PARAMETERS.calendar_a2,
        soc_knee=DEFAULT_NMC_PARAMETERS.calendar_soc_knee,
        activation_energy_j_per_mol=DEFAULT_NMC_PARAMETERS.activation_energy_j_per_mol,
        reference_temperature_k=303.15,
        reference_calendar_coefficient=1.0,
        time_exponent=0.75,
    ),
    cycle=CycleAgingParameters(
        reference_cycle_coefficient=DEFAULT_NMC_PARAMETERS.cycle_reference_coefficient,
        dod_coefficient=DEFAULT_NMC_PARAMETERS.cycle_dod_coefficient,
        c_rate_coefficient=DEFAULT_NMC_PARAMETERS.cycle_c_rate_coefficient,
        throughput_exponent=DEFAULT_NMC_PARAMETERS.cycle_time_exponent,
    ),
    source_citations=(
        "Schmalstieg et al. (2014), Degradation of lithium ion batteries employing graphite negatives and NMC/LMO positives",
        "Schmalstieg et al. (2014), calendar-plus-cycle semi-empirical parameterization",
    ),
    parameter_set_id="schmalstieg_nmc_literature_anchor_v1",
    source_title="Degradation of lithium ion batteries employing graphite negatives and NMC/LMO positives",
    source_authors="Schmalstieg et al.",
    source_year=2014,
    source_equation_or_section="calendar and cycle aging equations; Section 5 methodology mapping",
    source_implementation="literature-anchor calibration; original fitted table unavailable in checkout",
)


@dataclass(frozen=True, slots=True)
class DegradationSettings:
    """Scenario inputs for one-session degradation assessment."""

    battery_age_years: float = 1.0
    cumulative_equivalent_full_cycles: float = 0.0
    # The local cycle-aging derivative is singular at FEC=0 for the default
    # square-root exponent.  This positive, explicit floor is a numerical
    # regularization parameter, not an invented extra aging contribution.
    minimum_effective_fec: float = DEFAULT_MINIMUM_EFFECTIVE_FEC
    # Compatibility spelling retained for earlier callers.  It is normalized
    # to ``minimum_effective_fec`` during validation and is never an independent
    # second parameter.
    minimum_reference_fec: float | None = None
    # The service default is an explicit 16-hour planning assumption.  ``None``
    # disables the optional parked-period subcomponent rather than pretending
    # that its duration is zero.
    parked_day_hours: float | None = PARKED_PERIOD_DEFAULT_HOURS
    parked_period_source: str = PARKED_PERIOD_DEFAULT_SOURCE
    reference_temperature_c: float = 30.0
    fallback_temperature_c: float = 25.0
    # Removed normalized-health quantization.  Supplying the old field fails
    # loudly so no dormant normalization path can be mistaken for active code.
    health_score_resolution: float | None = None
    degradation_comparison_tolerance: float = 1e-12
    reference_age_years: float = _REFERENCE_AGE_YEARS
    lfp: ChemistryDegradationParameters = DEFAULT_LFP_PARAMETERS
    nmc: ChemistryDegradationParameters = DEFAULT_NMC_PARAMETERS

    def __post_init__(self) -> None:
        values_to_validate: tuple[tuple[str, float], ...] = (
            ("battery_age_years", self.battery_age_years),
            ("cumulative_equivalent_full_cycles", self.cumulative_equivalent_full_cycles),
            ("minimum_effective_fec", self.minimum_effective_fec),
            ("reference_temperature_c", self.reference_temperature_c),
            ("fallback_temperature_c", self.fallback_temperature_c),
            ("degradation_comparison_tolerance", self.degradation_comparison_tolerance),
            ("reference_age_years", self.reference_age_years),
        )
        for field_name, field_value in values_to_validate:
            _finite(field_name, field_value)
        legacy_floor = self.minimum_reference_fec
        if legacy_floor is not None:
            legacy_floor = _finite("minimum_reference_fec", legacy_floor)
            if (
                legacy_floor != DEFAULT_MINIMUM_EFFECTIVE_FEC
                and self.minimum_effective_fec != DEFAULT_MINIMUM_EFFECTIVE_FEC
                and self.minimum_effective_fec != legacy_floor
            ):
                raise SchemaValidationError(
                    "minimum_reference_fec is a compatibility alias and must match "
                    "minimum_effective_fec."
                )
            if legacy_floor != DEFAULT_MINIMUM_EFFECTIVE_FEC:
                object.__setattr__(self, "minimum_effective_fec", legacy_floor)
        object.__setattr__(self, "minimum_reference_fec", self.minimum_effective_fec)
        if self.parked_day_hours is not None:
            _finite("parked_day_hours", self.parked_day_hours)
        if self.battery_age_years <= 0.0:
            raise PhysicalConstraintError("battery_age_years must be positive.")
        if self.cumulative_equivalent_full_cycles < 0.0:
            raise PhysicalConstraintError("cumulative_equivalent_full_cycles must be non-negative.")
        if self.minimum_effective_fec <= 0.0:
            raise PhysicalConstraintError("minimum_effective_fec must be positive.")
        if self.parked_day_hours is not None and self.parked_day_hours < 0.0:
            raise PhysicalConstraintError("parked_day_hours must be non-negative.")
        if self.reference_temperature_c <= -273.15 or self.fallback_temperature_c <= -273.15:
            raise PhysicalConstraintError("temperatures must be above absolute zero.")
        if self.health_score_resolution is not None:
            raise SchemaValidationError(
                "health_score_resolution was removed with normalized health; "
                "compare total_capacity_fade directly."
            )
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
        if not isinstance(self.parked_period_source, str) or not self.parked_period_source.strip():
            raise SchemaValidationError("parked_period_source must be a non-empty string.")
        object.__setattr__(self, "parked_period_source", self.parked_period_source.strip())

    @property
    def effective_fec_floor(self) -> float:
        """Positive FEC floor used by the cycle-aging local derivative."""
        return self.minimum_effective_fec

    @property
    def parked_day_hours_source(self) -> str:
        """Compatibility alias for the parked-period provenance."""
        return self.parked_period_source

    def parameters_for(self, chemistry: object) -> ChemistryDegradationParameters:
        if chemistry == "LFP":
            return self.lfp
        if chemistry == "NMC":
            return self.nmc
        raise SchemaValidationError("chemistry must be exactly 'LFP' or 'NMC'.")

    def parameter_set_for(self, chemistry: object) -> BatteryDegradationParameters:
        """Return the canonical chemistry-specific parameter set."""
        if chemistry == "LFP":
            return DEFAULT_LFP_DEGRADATION_PARAMETERS
        if chemistry == "NMC":
            return DEFAULT_NMC_DEGRADATION_PARAMETERS
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
    # Canonical names and provenance for the finalized model.  The first
    # seven legacy fields above remain readable by Commit 6/7 consumers.
    calendar_fade_connected_window: float = 0.0
    calendar_fade_parked_period: float | None = None
    calendar_capacity_fade: float = 0.0
    cycle_capacity_fade: float = 0.0
    total_capacity_fade: float = 0.0
    capacity_fade_percent: float = 0.0
    battery_temperature_c: float | None = None
    battery_age_years: float | None = None
    accumulated_fec_at_start: float | None = None
    effective_fec_at_start: float | None = None
    minimum_effective_fec: float | None = None
    fec_regularization_applied: bool = False
    fec_regularization_source: str = FEC_REGULARIZATION_SOURCE
    parked_period_hours: float | None = None
    parked_period_source: str = PARKED_PERIOD_DISABLED_SOURCE
    degradation_model_id: str | None = None
    degradation_model_version: str = DEGRADATION_MODEL_VERSION
    parameter_set_id: str | None = None
    parameter_status: str = PARAMETER_STATUS
    battery_chemistry: Chemistry | None = None
    calendar_scope: str = "connected_window_only"
    temperature_source: str = "planning_signal_or_assumed_default"
    battery_age_source: str = "degradation_settings"
    accumulated_fec_source: str = "degradation_settings"

    @property
    def raw_battery_stress(self) -> float:
        """Schedule-intrinsic stress metric (lower is better)."""
        return self.total_fade

    @property
    def effective_fec(self) -> float | None:
        """Compatibility alias for the regularized starting FEC."""
        return self.effective_fec_at_start

    @property
    def battery_metric_model_id(self) -> str:
        return BATTERY_METRIC_MODEL_ID

    @property
    def battery_metric_comparison_scope(self) -> str:
        return BATTERY_METRIC_COMPARISON_SCOPE

    def __post_init__(self) -> None:
        for field_name, field_value in (("candidate_id", self.candidate_id), ("ev_id", self.ev_id)):
            if not isinstance(field_value, str) or not field_value.strip():
                raise SchemaValidationError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, field_value.strip())
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
        connected = (
            self.charging_window_calendar_fade
            if self.calendar_fade_connected_window is None
            else self.calendar_fade_connected_window
        )
        parked = self.parked_day_calendar_fade
        if self.calendar_fade_parked_period is not None:
            parked = self.calendar_fade_parked_period
        cycle = self.cycle_fade if self.cycle_capacity_fade is None else self.cycle_capacity_fade
        calendar = connected + parked
        total = calendar + cycle
        object.__setattr__(self, "calendar_fade_connected_window", float(connected))
        object.__setattr__(self, "calendar_capacity_fade", float(calendar))
        object.__setattr__(self, "cycle_capacity_fade", float(cycle))
        object.__setattr__(self, "total_capacity_fade", float(total))
        object.__setattr__(self, "capacity_fade_percent", float(total * 100.0))
        if self.calendar_fade_parked_period is not None and self.calendar_fade_parked_period < 0.0:
            raise PhysicalConstraintError("calendar_fade_parked_period must be non-negative.")
        for fade_name, fade_value in (
            ("calendar_fade_connected_window", connected),
            ("calendar_capacity_fade", calendar),
            ("cycle_capacity_fade", cycle),
            ("total_capacity_fade", total),
            ("capacity_fade_percent", total * 100.0),
        ):
            _finite(fade_name, fade_value)
            if fade_value < 0.0:
                raise PhysicalConstraintError(f"{fade_name} must be non-negative.")
        if self.battery_temperature_c is not None:
            _finite("battery_temperature_c", self.battery_temperature_c)
        if self.battery_age_years is not None:
            _finite("battery_age_years", self.battery_age_years)
        if self.accumulated_fec_at_start is not None:
            _finite("accumulated_fec_at_start", self.accumulated_fec_at_start)
            if self.accumulated_fec_at_start < 0.0:
                raise PhysicalConstraintError("accumulated_fec_at_start must be non-negative.")
        if self.effective_fec_at_start is not None:
            _finite("effective_fec_at_start", self.effective_fec_at_start)
            if self.effective_fec_at_start < 0.0:
                raise PhysicalConstraintError("effective_fec_at_start must be non-negative.")
        if self.minimum_effective_fec is not None:
            _finite("minimum_effective_fec", self.minimum_effective_fec)
            if self.minimum_effective_fec <= 0.0:
                raise PhysicalConstraintError("minimum_effective_fec must be positive.")
        if not isinstance(self.fec_regularization_applied, bool):
            raise SchemaValidationError("fec_regularization_applied must be bool.")
        if self.parked_period_hours is not None:
            _finite("parked_period_hours", self.parked_period_hours)
            if self.parked_period_hours < 0.0:
                raise PhysicalConstraintError("parked_period_hours must be non-negative.")
        model_id = self.degradation_model_id
        if model_id is None:
            model_id = (
                LFP_DEGRADATION_MODEL_ID if self.chemistry == "LFP" else NMC_DEGRADATION_MODEL_ID
            )
        parameter_id = self.parameter_set_id
        if parameter_id is None:
            parameter_id = (
                "naumann_lfp_literature_anchor_v1"
                if self.chemistry == "LFP"
                else "schmalstieg_nmc_literature_anchor_v1"
            )
        for name, value in (
            ("degradation_model_id", model_id),
            ("degradation_model_version", self.degradation_model_version),
            ("parameter_set_id", parameter_id),
            ("parameter_status", self.parameter_status),
            ("calendar_scope", self.calendar_scope),
            ("temperature_source", self.temperature_source),
            ("battery_age_source", self.battery_age_source),
            ("accumulated_fec_source", self.accumulated_fec_source),
            ("fec_regularization_source", self.fec_regularization_source),
            ("parked_period_source", self.parked_period_source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if self.battery_chemistry is not None and self.battery_chemistry != self.chemistry:
            raise SchemaValidationError("battery_chemistry must match chemistry.")
        object.__setattr__(self, "battery_chemistry", self.chemistry)


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


# Name used by the finalized methodology.  Keep the richer historical class
# name as the implementation so existing callers remain source-compatible.
DegradationEvaluation = DegradationAssessment


@dataclass(frozen=True, slots=True)
class BatteryMetric:
    """Immutable schedule-intrinsic metric with explicit model provenance."""

    raw_stress: float
    capacity_loss_fraction: float | None
    rul_years: float | None
    model_id: str = BATTERY_METRIC_MODEL_ID
    metric_units: str = "incremental capacity-fade fraction"
    comparison_scope: str = BATTERY_METRIC_COMPARISON_SCOPE

    def __post_init__(self) -> None:
        _finite("raw_stress", self.raw_stress)
        if self.raw_stress < 0.0:
            raise PhysicalConstraintError("raw_stress must be non-negative.")
        for name, value in (
            ("capacity_loss_fraction", self.capacity_loss_fraction),
            ("rul_years", self.rul_years),
        ):
            if value is not None:
                _finite(name, value)
                if value < 0.0:
                    raise PhysicalConstraintError(f"{name} must be non-negative.")
        for metadata_name, metadata_value in (
            ("model_id", self.model_id),
            ("metric_units", self.metric_units),
            ("comparison_scope", self.comparison_scope),
        ):
            if not isinstance(metadata_value, str) or not metadata_value.strip():
                raise SchemaValidationError(f"{metadata_name} must be a non-empty string.")
            object.__setattr__(self, metadata_name, metadata_value.strip())


def calendar_soc_stress(
    soc: float,
    parameters: ChemistryDegradationParameters | CalendarAgingParameters,
) -> float:
    """Return convex chemistry-specific calendar stress g(s)."""
    if isinstance(parameters, CalendarAgingParameters):
        a0, a1, a2, knee = (
            parameters.a0,
            parameters.a1,
            parameters.a2,
            parameters.soc_knee,
        )
    elif isinstance(parameters, ChemistryDegradationParameters):
        a0, a1, a2, knee = (
            parameters.calendar_a0,
            parameters.calendar_a1,
            parameters.calendar_a2,
            parameters.calendar_soc_knee,
        )
    else:
        raise SchemaValidationError(
            "parameters must be ChemistryDegradationParameters or CalendarAgingParameters."
        )
    value = _finite("soc", soc)
    if not 0.0 <= value <= 1.0:
        raise PhysicalConstraintError("soc must lie in [0, 1].")
    hinge = max(value - knee, 0.0)
    return a0 + a1 * value + a2 * hinge * hinge


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


def calendar_age_slope(
    battery_age_years: float,
    time_exponent: float,
    *,
    minimum_age_years: float = 1e-6,
) -> float:
    """Return ``z * age**(z-1)`` with a finite positive age floor."""
    age = _finite("battery_age_years", battery_age_years)
    exponent = _finite("time_exponent", time_exponent)
    lower = _finite("minimum_age_years", minimum_age_years)
    if lower <= 0.0 or age < 0.0:
        raise PhysicalConstraintError("battery age and its lower bound must be non-negative.")
    if not 0.0 < exponent <= 1.0:
        raise PhysicalConstraintError("time_exponent must lie in (0, 1].")
    value = exponent * max(age, lower) ** (exponent - 1.0)
    if not isfinite(value):
        raise PhysicalConstraintError("calendar age slope must be finite.")
    return float(value)


def cycle_age_slope(
    accumulated_fec: float,
    *,
    throughput_exponent: float = 0.5,
    minimum_fec: float = DEFAULT_MINIMUM_EFFECTIVE_FEC,
) -> float:
    """Return the local cycle power-law slope at the accumulated FEC.

    ``minimum_fec`` is an explicit positive regularization floor for the
    singular derivative at zero FEC.  It changes only the local slope used by
    the incremental surrogate; it does not add a cycle or calendar-fade term.
    """
    fec = _finite("accumulated_fec", accumulated_fec)
    exponent = _finite("throughput_exponent", throughput_exponent)
    lower = _finite("minimum_fec", minimum_fec)
    if fec < 0.0 or lower <= 0.0:
        raise PhysicalConstraintError("FEC must be non-negative and minimum_fec positive.")
    if not 0.0 < exponent <= 1.0:
        raise PhysicalConstraintError("throughput_exponent must lie in (0, 1].")
    value = exponent * max(fec, lower) ** (exponent - 1.0)
    if not isfinite(value):
        raise PhysicalConstraintError("cycle age slope must be finite.")
    return float(value)


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
            * signal.interval_durations[global_step]
            / _HOURS_PER_YEAR
        )

    delivered_energy = max(
        session.initial_energy_kwh,
        candidate.target_soc * ev.battery_capacity_kwh,
    )
    parked_soc = (delivered_energy - session.commute_energy_kwh) / ev.battery_capacity_kwh
    if not 0.0 <= parked_soc <= 1.0:
        raise PhysicalConstraintError("computed parked SOC lies outside [0, 1].")
    parked_fade: float | None = None
    if model.parked_day_hours is not None:
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
        model.minimum_effective_fec,
    )
    cycle_slope = cycle_age_slope(
        model.cumulative_equivalent_full_cycles,
        throughput_exponent=params.cycle_time_exponent,
        minimum_fec=model.minimum_effective_fec,
    )
    cycle_fade = (
        params.cycle_reference_coefficient
        * (1.0 + params.cycle_dod_coefficient * throughput_fraction)
        * (1.0 + params.cycle_c_rate_coefficient * peak_c_rate)
        * cycle_slope
        * throughput_fraction
    )
    legacy_parked_fade = 0.0 if parked_fade is None else parked_fade
    total = window_fade + legacy_parked_fade + cycle_fade
    parameter_set = model.parameter_set_for(ev.chemistry)
    model_id = LFP_DEGRADATION_MODEL_ID if ev.chemistry == "LFP" else NMC_DEGRADATION_MODEL_ID
    representative_temperature = (
        model.fallback_temperature_c
        if signal.battery_temperature_c is None
        else signal.battery_temperature_c[session.arrival_step]
    )
    annualized_pct = total * settings.equivalent_sessions_per_year * 100.0
    return DegradationAssessment(
        candidate_id=candidate.candidate_id,
        ev_id=ev.ev_id,
        chemistry=ev.chemistry,
        charging_window_calendar_fade=window_fade,
        parked_day_calendar_fade=legacy_parked_fade,
        cycle_fade=cycle_fade,
        total_fade=total,
        annualized_degradation_pct=annualized_pct,
        parked_soc=parked_soc,
        peak_c_rate=peak_c_rate,
        calendar_fade_connected_window=window_fade,
        calendar_fade_parked_period=parked_fade,
        calendar_capacity_fade=window_fade + legacy_parked_fade,
        cycle_capacity_fade=cycle_fade,
        total_capacity_fade=total,
        capacity_fade_percent=total * 100.0,
        battery_temperature_c=representative_temperature,
        battery_age_years=model.battery_age_years,
        accumulated_fec_at_start=model.cumulative_equivalent_full_cycles,
        effective_fec_at_start=effective_fec,
        minimum_effective_fec=model.minimum_effective_fec,
        fec_regularization_applied=(
            model.cumulative_equivalent_full_cycles < model.minimum_effective_fec
        ),
        fec_regularization_source=FEC_REGULARIZATION_SOURCE,
        parked_period_hours=model.parked_day_hours,
        parked_period_source=(
            model.parked_period_source
            if model.parked_day_hours is not None
            else PARKED_PERIOD_DISABLED_SOURCE
        ),
        degradation_model_id=model_id,
        parameter_set_id=parameter_set.parameter_set_id,
        parameter_status=parameter_set.parameter_status,
        battery_chemistry=ev.chemistry,
        calendar_scope=(
            "connected_window_plus_parked_period"
            if parked_fade is not None
            else "connected_window_only"
        ),
    )


def evaluate_battery_metric(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    menu_settings: MenuSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
) -> BatteryMetric:
    """Evaluate the absolute battery-stress proxy for one validated trajectory.

    This public helper intentionally reports no calibrated capacity-loss or RUL
    estimate.  The current semi-empirical model supplies an incremental fade
    proxy only; callers must not interpret it as a manufacturer life forecast.
    """
    assessment = assess_candidate_degradation(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        menu_settings=menu_settings,
        degradation_settings=degradation_settings,
    )
    return BatteryMetric(
        raw_stress=assessment.raw_battery_stress,
        capacity_loss_fraction=None,
        rul_years=None,
        model_id=assessment.degradation_model_id or BATTERY_METRIC_MODEL_ID,
    )


def evaluate_degradation(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    candidate: MenuCandidate,
    menu_settings: MenuSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
) -> DegradationAssessment:
    """Canonical name for an absolute two-component degradation evaluation."""
    return assess_candidate_degradation(
        ev=ev,
        session=session,
        signal=signal,
        candidate=candidate,
        menu_settings=menu_settings,
        degradation_settings=degradation_settings,
    )


def score_generated_menu(
    *,
    ev: EVSpec,
    session: ChargingSession,
    signal: PlanningSignal,
    menu: GeneratedMenu,
    menu_settings: MenuSettings | None = None,
    degradation_settings: DegradationSettings | None = None,
    normalize_health: bool = False,
) -> DegradationScoredMenu:
    """Assess all candidates with schedule-intrinsic degradation metrics.

    ``charging_health_score`` is retained only for compatibility with earlier
    commits.  Rich Algorithm 1 consumers use ``raw_battery_stress`` and never
    compare a menu-relative score.  ``normalize_health`` is a deprecated
    compatibility keyword: ``True`` is rejected and no normalization helper
    remains in the package.
    """
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
    if not isinstance(normalize_health, bool):
        raise SchemaValidationError("normalize_health must be a bool.")
    if normalize_health:
        raise SchemaValidationError(
            "menu-relative health normalization was removed; compare total_capacity_fade directly."
        )
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
    offers: list[MenuOffer] = []
    for candidate, assessment in zip(menu.candidates, assessments, strict=True):
        # Compatibility only: this is a deterministic scaled fade value, not
        # a menu-relative health score and never participates in optimization.
        health = min(100.0, assessment.total_capacity_fade * 100.0)
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
                incremental_degradation=assessment.total_capacity_fade,
                annualized_degradation_pct=assessment.annualized_degradation_pct,
                charging_health_score=health,
                raw_battery_stress=assessment.total_capacity_fade,
                profile=candidate.profile,
                degradation_model_id=assessment.degradation_model_id,
                degradation_model_version=assessment.degradation_model_version,
                parameter_set_id=assessment.parameter_set_id,
                parameter_status=assessment.parameter_status,
                battery_chemistry=assessment.battery_chemistry,
                calendar_capacity_fade=assessment.calendar_capacity_fade,
                cycle_capacity_fade=assessment.cycle_capacity_fade,
                total_capacity_fade=assessment.total_capacity_fade,
                capacity_fade_percent=assessment.capacity_fade_percent,
                calendar_fade_connected_window=assessment.calendar_fade_connected_window,
                calendar_fade_parked_period=assessment.calendar_fade_parked_period,
                battery_temperature_c=assessment.battery_temperature_c,
                battery_age_years=assessment.battery_age_years,
                accumulated_fec_at_start=assessment.accumulated_fec_at_start,
                effective_fec_at_start=assessment.effective_fec_at_start,
                minimum_effective_fec=assessment.minimum_effective_fec,
                fec_regularization_applied=assessment.fec_regularization_applied,
                fec_regularization_source=assessment.fec_regularization_source,
                parked_period_hours=assessment.parked_period_hours,
                parked_period_source=assessment.parked_period_source,
                calendar_scope=assessment.calendar_scope,
                temperature_source=assessment.temperature_source,
                battery_age_source=assessment.battery_age_source,
                accumulated_fec_source=assessment.accumulated_fec_source,
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
