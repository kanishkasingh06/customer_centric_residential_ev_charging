"""Customer V2G consent envelope and bidirectional converter limits.

Formulation reference: Rev 4.2 Block 1, eq. (35)-(40).

The envelope is declared by the customer at enrolment, BEFORE any menu is
generated.  Every offer the menu displays is feasible within it, so no
displayed option can confront the customer with discharge they did not
authorise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from evmenu.exceptions import PhysicalConstraintError, SchemaValidationError


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SchemaValidationError(f"{name} must be a finite real number; received {value!r}.")
    return float(value)


@dataclass(frozen=True, slots=True)
class V2GEnvelope:
    """Declared consent envelope ``V_i = (T^v2g, beta, zeta, Ebar)``.

    Parameters
    ----------
    discharge_power_kw:
        Bidirectional converter rating on the grid side.  ``0.0`` disables
        export entirely and makes every result identical to V1G.
    discharge_efficiency:
        ``eta^out`` in (0, 1].  Battery-side withdrawal for a grid-side
        export ``E`` is ``E / eta^out``.
    reserve_soc:
        ``beta_i``.  Export may never drive the pack below
        ``max(B^min, beta * B^max)``.
    departure_floor_soc:
        ``zeta_i``.  ``None`` means "protect the arrival energy", i.e.
        ``zeta_i = s_i(0)`` resolved per session, which reproduces the V1G
        implementation's terminal condition exactly.  Setting it to the
        requested target releases the arrival surplus for export.
    session_export_allowance_kwh:
        ``Ebar_i``, grid-side, PER CONNECTED SESSION.  Not per calendar day:
        a residential session crosses midnight, and a per-day allowance
        would grant such a session two allocations.  ``None`` means no
        allowance limit beyond the converter rating and the floors.
    export_window:
        Optional ``(start_minute, end_minute)`` clock window, inclusive of
        start and exclusive of end, within which export is permitted.
        ``None`` permits export throughout the connected session.
    """

    discharge_power_kw: float = 0.0
    discharge_efficiency: float = 0.95
    reserve_soc: float = 0.20
    departure_floor_soc: float | None = None
    session_export_allowance_kwh: float | None = None
    export_window: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        power = _finite("discharge_power_kw", self.discharge_power_kw)
        if power < 0.0:
            raise PhysicalConstraintError("discharge_power_kw must be non-negative.")
        efficiency = _finite("discharge_efficiency", self.discharge_efficiency)
        if not 0.0 < efficiency <= 1.0:
            raise PhysicalConstraintError("discharge_efficiency must lie in (0, 1].")
        reserve = _finite("reserve_soc", self.reserve_soc)
        if not 0.0 <= reserve <= 1.0:
            raise PhysicalConstraintError("reserve_soc must lie in [0, 1].")
        if self.departure_floor_soc is not None:
            floor = _finite("departure_floor_soc", self.departure_floor_soc)
            if not 0.0 <= floor <= 1.0:
                raise PhysicalConstraintError("departure_floor_soc must lie in [0, 1].")
        if self.session_export_allowance_kwh is not None:
            allowance = _finite(
                "session_export_allowance_kwh", self.session_export_allowance_kwh
            )
            if allowance < 0.0:
                raise PhysicalConstraintError(
                    "session_export_allowance_kwh must be non-negative."
                )
        if self.export_window is not None:
            window = self.export_window
            if not isinstance(window, tuple) or len(window) != 2:
                raise SchemaValidationError("export_window must be a (start, end) minute pair.")
            start, end = window
            for label, value in (("start", start), ("end", value_end := end)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise SchemaValidationError(f"export_window {label} must be an integer.")
            del value_end
            if not 0 <= start < end:
                raise PhysicalConstraintError(
                    "export_window must satisfy 0 <= start < end."
                )

    @property
    def enabled(self) -> bool:
        """Return whether this envelope permits any export at all."""
        return self.discharge_power_kw > 0.0 and (
            self.session_export_allowance_kwh is None
            or self.session_export_allowance_kwh > 0.0
        )

    def resolved_departure_floor(self, arrival_soc: float) -> float:
        """Return ``zeta_i``, defaulting to full arrival-energy protection."""
        if self.departure_floor_soc is None:
            return _finite("arrival_soc", arrival_soc)
        return float(self.departure_floor_soc)


V1G_ENVELOPE = V2GEnvelope(discharge_power_kw=0.0)
"""An envelope that permits no export.  Results must match evmenu exactly."""
