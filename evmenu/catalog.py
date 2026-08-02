"""Validated EV model catalogue for the high-level menu service.

Catalogue values are explicit research assumptions, not manufacturer claims. Callers may
supply a custom :class:`EVModel` when a vehicle is not listed or verified specifications
are available from another source.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .exceptions import PhysicalConstraintError, SchemaValidationError
from .schemas import Chemistry, EVSpec


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number.")
    return float(value)


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class EVModel:
    """User-facing EV model parameters required by the physical core."""

    model_id: str
    display_name: str
    usable_battery_kwh: float
    minimum_soc: float
    onboard_ac_power_kw: float
    charging_efficiency: float
    chemistry: Chemistry
    consumption_kwh_per_km: float
    assumption_note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text("model_id", self.model_id))
        object.__setattr__(self, "display_name", _text("display_name", self.display_name))
        object.__setattr__(self, "assumption_note", _text("assumption_note", self.assumption_note))
        capacity = _finite("usable_battery_kwh", self.usable_battery_kwh)
        minimum_soc = _finite("minimum_soc", self.minimum_soc)
        charger = _finite("onboard_ac_power_kw", self.onboard_ac_power_kw)
        efficiency = _finite("charging_efficiency", self.charging_efficiency)
        consumption = _finite("consumption_kwh_per_km", self.consumption_kwh_per_km)
        if capacity <= 0.0 or charger <= 0.0 or consumption <= 0.0:
            raise PhysicalConstraintError(
                "usable_battery_kwh, onboard_ac_power_kw, and consumption must be positive."
            )
        if not 0.0 <= minimum_soc < 1.0:
            raise PhysicalConstraintError("minimum_soc must lie in [0, 1).")
        if not 0.0 < efficiency <= 1.0:
            raise PhysicalConstraintError("charging_efficiency must lie in (0, 1].")
        if self.chemistry not in ("LFP", "NMC"):
            raise SchemaValidationError("chemistry must be 'LFP' or 'NMC'.")

    def to_ev_spec(self) -> EVSpec:
        """Convert the catalogue entry into the core immutable EV specification."""
        return EVSpec(
            ev_id=self.model_id,
            battery_capacity_kwh=self.usable_battery_kwh,
            minimum_energy_kwh=self.minimum_soc * self.usable_battery_kwh,
            charger_power_kw=self.onboard_ac_power_kw,
            charging_efficiency=self.charging_efficiency,
            chemistry=self.chemistry,
        )


_GENERIC_NOTE = (
    "Illustrative research assumption; replace with verified usable capacity, AC limit, "
    "efficiency, chemistry, and consumption for deployment."
)

_MODELS: tuple[EVModel, ...] = (
    EVModel(
        model_id="generic_40kwh_lfp",
        display_name="Generic 40 kWh LFP EV",
        usable_battery_kwh=40.0,
        minimum_soc=0.05,
        onboard_ac_power_kw=7.2,
        charging_efficiency=0.90,
        chemistry="LFP",
        consumption_kwh_per_km=0.15,
        assumption_note=_GENERIC_NOTE,
    ),
    EVModel(
        model_id="generic_60kwh_nmc",
        display_name="Generic 60 kWh NMC EV",
        usable_battery_kwh=60.0,
        minimum_soc=0.05,
        onboard_ac_power_kw=7.2,
        charging_efficiency=0.90,
        chemistry="NMC",
        consumption_kwh_per_km=0.17,
        assumption_note=_GENERIC_NOTE,
    ),
)
if not all(model.model_id for model in _MODELS) or len(
    {model.model_id for model in _MODELS}
) != len(_MODELS):
    raise SchemaValidationError("catalogue model IDs must be non-empty and unique.")
_MODEL_BY_ID = {model.model_id: model for model in _MODELS}


def list_ev_models() -> tuple[EVModel, ...]:
    """Return catalogue entries in deterministic model-ID order."""
    return tuple(sorted(_MODELS, key=lambda model: model.model_id))


def get_ev_model(model: str | EVModel) -> EVModel:
    """Resolve a case-sensitive catalogue ID or return a custom model."""
    if isinstance(model, EVModel):
        return model
    model_id = _text("ev_model", model)
    try:
        return _MODEL_BY_ID[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_BY_ID))
        raise SchemaValidationError(
            f"Unknown ev_model {model_id!r}. Available models: {available}."
        ) from exc


def search_ev_models(query: str) -> tuple[EVModel, ...]:
    """Search IDs and display names case-insensitively; empty queries are rejected."""
    term = _text("query", query).casefold()
    return tuple(
        model
        for model in list_ev_models()
        if term in model.model_id.casefold() or term in model.display_name.casefold()
    )
