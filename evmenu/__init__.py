"""Residential EV customer-menu package."""

from .exceptions import (
    EVMenuError,
    PhysicalConstraintError,
    SchemaValidationError,
    SignalValidationError,
)
from .feasibility import (
    RequestFeasibility,
    available_grid_energy_kwh,
    build_target_options,
    compute_buffer_energy,
    delivered_target_energy_kwh,
    evaluate_request_feasibility,
    minimum_required_departure_energy_kwh,
    minimum_required_target_soc,
    request_is_feasible,
    required_grid_energy_kwh,
)
from .profiles import (
    ConstructedProfile,
    build_immediate_charging_profile,
    build_minimum_cost_charging_profile,
)
from .schemas import (
    ChargingProfile,
    ChargingSession,
    Chemistry,
    EVSpec,
    MenuOffer,
    MenuSettings,
    PlanningSignal,
    TargetOption,
    TargetSource,
)

__all__ = [
    "ChargingProfile",
    "ChargingSession",
    "Chemistry",
    "ConstructedProfile",
    "EVMenuError",
    "EVSpec",
    "MenuOffer",
    "MenuSettings",
    "PhysicalConstraintError",
    "PlanningSignal",
    "RequestFeasibility",
    "SchemaValidationError",
    "SignalValidationError",
    "TargetOption",
    "TargetSource",
    "ValidationCode",
    "ValidationIssue",
    "ValidationReport",
    "ValidationTolerances",
    "available_grid_energy_kwh",
    "build_immediate_charging_profile",
    "build_minimum_cost_charging_profile",
    "build_target_options",
    "compute_buffer_energy",
    "delivered_target_energy_kwh",
    "evaluate_request_feasibility",
    "minimum_required_departure_energy_kwh",
    "minimum_required_target_soc",
    "request_is_feasible",
    "required_grid_energy_kwh",
    "validate_charging_profile",
]

from .validation import (
    ValidationCode,
    ValidationIssue,
    ValidationReport,
    ValidationTolerances,
    validate_charging_profile,
)
