"""Bidirectional feasible set and trajectory solves.

Formulation reference: Rev 4.2 Block 2 (eq. 38-41), Block 3 (eq. 43-44),
Block 4 (eq. 56-60), Block 5 (eq. 61-62), Block 7 (Prop 1-3, Remark 1).

Decision vector ``x = [E_in(0..n-1), E_out(0..n-1)]``, both grid-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.optimize import minimize

from evmenu.degradation import DegradationSettings, calendar_soc_stress, cycle_age_slope
from evmenu.degradation import _HOURS_PER_YEAR, _relative_age_factor, _temperature_factor
from evmenu.exceptions import PhysicalConstraintError
from evmenu.schemas import ChargingSession, EVSpec, PlanningSignal

from .envelope import V2GEnvelope
from .fade import assess_trajectory, BidirectionalFade

_FEASIBILITY_TOLERANCE = 1e-6
_SIMULTANEITY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything fixed by one ``(ready_step, target_soc)`` request."""

    ev: EVSpec
    session: ChargingSession
    signal: PlanningSignal
    envelope: V2GEnvelope
    settings: DegradationSettings
    ready_step: int
    target_soc: float
    sell_price: tuple[float, ...]
    discharge_severity: float

    @property
    def n(self) -> int:
        return self.session.departure_step - self.session.arrival_step

    @property
    def terminal_energy_kwh(self) -> float:
        """``B^dagger = max(zeta, z) * B_max``, eq. (39)."""
        arrival_soc = self.session.initial_energy_kwh / self.ev.battery_capacity_kwh
        zeta = self.envelope.resolved_departure_floor(arrival_soc)
        return max(zeta, self.target_soc) * self.ev.battery_capacity_kwh

    @property
    def reserve_energy_kwh(self) -> float:
        """``max(B_min, beta * B_max)``, eq. (36)."""
        return max(
            self.ev.minimum_energy_kwh,
            self.envelope.reserve_soc * self.ev.battery_capacity_kwh,
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A solved bidirectional trajectory with its evaluated attributes."""

    grid_in_kwh: tuple[float, ...]
    grid_out_kwh: tuple[float, ...]
    battery_energy_kwh: tuple[float, ...]
    soc: tuple[float, ...]
    cost: float
    saving: float
    fade: BidirectionalFade
    simultaneous_periods: int
    feasible: bool
    violations: tuple[str, ...] = field(default=())

    @property
    def net_power_kw(self) -> tuple[float, ...]:
        return tuple(self.grid_in_kwh[i] - self.grid_out_kwh[i] for i in range(len(self.grid_in_kwh)))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def interval_durations(ctx: RequestContext) -> list[float]:
    base = ctx.session.arrival_step
    return [ctx.signal.interval_durations[base + i] for i in range(ctx.n)]


def capacities(ctx: RequestContext) -> tuple[list[float], list[float]]:
    """Per-interval grid-side import and export energy caps."""
    durations = interval_durations(ctx)
    cap_in = [ctx.ev.charger_power_kw * d for d in durations]
    window = ctx.envelope.export_window
    # A disabled envelope must collapse the export bounds to zero rather than
    # relying on the allowance constraint to hold them there.  Both describe
    # the same feasible set, but only the former gives the solver the same
    # parameterization as the unidirectional path, and the regression test
    # compares the two bitwise.
    enabled = ctx.envelope.enabled
    ready_local = max(0, min(ctx.n, ctx.ready_step - ctx.session.arrival_step))
    cap_out: list[float] = []
    for index, duration in enumerate(durations):
        allowed = enabled
        if allowed and not ctx.envelope.allow_export_after_ready and index >= ready_local:
            # Once the advertised ready-by time has passed the customer has
            # been told the car is ready; do not sell out of that window.
            allowed = False
        if allowed and window is not None:
            starts = ctx.signal.interval_start_minutes
            if starts is not None:
                minute = starts[ctx.session.arrival_step + index] % 1440
                allowed = window[0] <= minute < window[1]
        cap_out.append(ctx.envelope.discharge_power_kw * duration if allowed else 0.0)
    return cap_in, cap_out


def battery_states(ctx: RequestContext, x: list[float] | tuple[float, ...]) -> list[float]:
    """Battery energy at every state boundary, ``n + 1`` values."""
    n = ctx.n
    eta_in = ctx.ev.charging_efficiency
    eta_out = ctx.envelope.discharge_efficiency
    states = [ctx.session.initial_energy_kwh]
    for i in range(n):
        states.append(states[-1] + eta_in * x[i] - x[n + i] / eta_out)
    return states


def _soc(ctx: RequestContext, states: list[float]) -> list[float]:
    return [s / ctx.ev.battery_capacity_kwh for s in states]


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def _build_constraints(
    ctx: RequestContext,
    *,
    bau_cost: float,
    saving_low: float | None,
    saving_high: float | None,
    fade_cap: float | None,
) -> list[dict]:
    n = ctx.n
    eta_in = ctx.ev.charging_efficiency
    eta_out = ctx.envelope.discharge_efficiency
    b_max = ctx.ev.battery_capacity_kwh
    b_reserve = ctx.reserve_energy_kwh
    b_terminal = ctx.terminal_energy_kwh
    ready_local = max(0, min(n, ctx.ready_step - ctx.session.arrival_step))
    prices = [ctx.signal.price_per_kwh[ctx.session.arrival_step + i] for i in range(n)]
    sell = list(ctx.sell_price)

    def _state_jac(k: int) -> list[float]:
        row = [0.0] * (2 * n)
        for j in range(k):
            row[j] = eta_in
            row[n + j] = -1.0 / eta_out
        return row

    cons: list[dict] = []

    # Terminal equality, eq. (39).
    cons.append(
        {
            "type": "eq",
            "fun": lambda x: battery_states(ctx, x)[n] - b_terminal,
            "jac": lambda x, row=_state_jac(n): row,
        }
    )

    # Reserve floor and capacity ceiling at every boundary, eq. (36).
    for k in range(1, n + 1):
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, k=k: battery_states(ctx, x)[k] - b_reserve,
                "jac": lambda x, row=_state_jac(k): row,
            }
        )
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, k=k: b_max - battery_states(ctx, x)[k],
                "jac": lambda x, row=[-v for v in _state_jac(k)]: row,
            }
        )

    # Availability from the ready-by boundary onward, eq. (38).
    floor = ctx.target_soc * b_max
    for k in range(ready_local, n + 1):
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, k=k: battery_states(ctx, x)[k] - floor,
                "jac": lambda x, row=_state_jac(k): row,
            }
        )

    # Per-session export allowance, eq. (40).
    allowance = ctx.envelope.session_export_allowance_kwh
    if allowance is not None and ctx.envelope.enabled:
        # Vacuous once the export bounds are already zero, and a redundant
        # constraint still perturbs the SLSQP working set.
        row = [0.0] * (2 * n)
        for j in range(n):
            row[n + j] = -1.0
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x: allowance - sum(x[n:]),
                "jac": lambda x, row=row: row,
            }
        )

    # Saving band.  Lower edge is floored at zero by the caller (eq. 56).
    if saving_low is not None or saving_high is not None:
        srow = [0.0] * (2 * n)
        for j in range(n):
            srow[j] = -prices[j]
            srow[n + j] = sell[j]

        def _saving(x: list[float]) -> float:
            return bau_cost - (
                sum(prices[j] * x[j] for j in range(n))
                - sum(sell[j] * x[n + j] for j in range(n))
            )

        if saving_low is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x, lo=saving_low: _saving(x) - lo,
                    "jac": lambda x, row=srow: row,
                }
            )
        if saving_high is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x, hi=saving_high: hi - _saving(x),
                    "jac": lambda x, row=[-v for v in srow]: row,
                }
            )

    # One-sided realized-fade cap, eq. (61).
    if fade_cap is not None:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, cap=fade_cap: cap - _fade_objective(ctx, x)[0],
            }
        )

    return cons


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------


def _cost_objective(ctx: RequestContext, x) -> tuple[float, list[float]]:
    n = ctx.n
    prices = [ctx.signal.price_per_kwh[ctx.session.arrival_step + i] for i in range(n)]
    sell = list(ctx.sell_price)
    value = sum(prices[j] * x[j] for j in range(n)) - sum(sell[j] * x[n + j] for j in range(n))
    jac = [0.0] * (2 * n)
    for j in range(n):
        jac[j] = prices[j]
        jac[n + j] = -sell[j]
    return value, jac


def _fade_objective(
    ctx: RequestContext, x, *, guard_weight: float = 0.0
) -> tuple[float, list[float]]:
    """Solver surrogate, eq. (58)-(59): window calendar fade + cycle fade.

    The parked term is omitted as a constant of the request (Proposition 1).
    """
    n = ctx.n
    ev, settings = ctx.ev, ctx.settings
    params = settings.parameters_for(ev.chemistry)
    eta_in = ev.charging_efficiency
    eta_out = ctx.envelope.discharge_efficiency
    b_max = ev.battery_capacity_kwh
    lam = ctx.discharge_severity
    durations = interval_durations(ctx)

    states = battery_states(ctx, x)
    soc = _soc(ctx, states)
    age_factor = _relative_age_factor(
        age_years=settings.battery_age_years,
        alpha_time=params.calendar_time_exponent,
        reference_age_years=settings.reference_age_years,
    )

    # --- calendar window term and its gradient -----------------------------
    weights: list[float] = []
    cal = 0.0
    for k in range(n):
        global_step = ctx.session.arrival_step + k
        temperature = (
            settings.fallback_temperature_c
            if ctx.signal.battery_temperature_c is None
            else ctx.signal.battery_temperature_c[global_step]
        )
        w = (
            _temperature_factor(
                temperature, settings.reference_temperature_c,
                params.activation_energy_j_per_mol,
            )
            * age_factor
            * durations[k]
            / _HOURS_PER_YEAR
        )
        weights.append(w)
        cal += calendar_soc_stress(min(1.0, max(0.0, soc[k])), params) * w

    jac = [0.0] * (2 * n)
    for j in range(n):
        acc = 0.0
        for k in range(j + 1, n):
            hinge = max(soc[k] - params.calendar_soc_knee, 0.0)
            acc += (params.calendar_a1 + 2.0 * params.calendar_a2 * hinge) * weights[k]
        jac[j] += acc * eta_in / b_max
        jac[n + j] += -acc / (eta_out * b_max)

    # --- cycle term --------------------------------------------------------
    phi = (eta_in * sum(x[:n]) + sum(x[n:]) / eta_out) / b_max
    k_hi = max(range(n + 1), key=lambda k: soc[k])
    k_lo = min(range(n + 1), key=lambda k: soc[k])
    delta = soc[k_hi] - soc[k_lo]

    chat_num, chat_arg = 0.0, None
    for t in range(n):
        charge = eta_in * x[t] / durations[t]
        discharge = lam * x[n + t] / (eta_out * durations[t])
        if charge >= chat_num:
            chat_num, chat_arg = charge, ("in", t)
        if discharge >= chat_num:
            chat_num, chat_arg = discharge, ("out", t)
    chat = chat_num / b_max

    slope = cycle_age_slope(
        settings.cumulative_equivalent_full_cycles,
        throughput_exponent=params.cycle_time_exponent,
        minimum_fec=settings.minimum_effective_fec,
    )
    c0, c_d, c_c = (
        params.cycle_reference_coefficient,
        params.cycle_dod_coefficient,
        params.cycle_c_rate_coefficient,
    )
    depth_f, rate_f = 1.0 + c_d * delta, 1.0 + c_c * chat
    cyc = c0 * depth_f * rate_f * slope * phi

    # d(phi), d(delta), d(chat) -- subgradients at the active indices.
    for j in range(n):
        dphi_u, dphi_v = eta_in / b_max, 1.0 / (eta_out * b_max)
        ddelta_u = (eta_in / b_max) * ((1.0 if j < k_hi else 0.0) - (1.0 if j < k_lo else 0.0))
        ddelta_v = (-1.0 / (eta_out * b_max)) * (
            (1.0 if j < k_hi else 0.0) - (1.0 if j < k_lo else 0.0)
        )
        dchat_u = (
            eta_in / (durations[j] * b_max)
            if chat_arg == ("in", j)
            else 0.0
        )
        dchat_v = (
            lam / (eta_out * durations[j] * b_max)
            if chat_arg == ("out", j)
            else 0.0
        )
        jac[j] += c0 * slope * (
            c_d * ddelta_u * rate_f * phi
            + depth_f * c_c * dchat_u * phi
            + depth_f * rate_f * dphi_u
        )
        jac[n + j] += c0 * slope * (
            c_d * ddelta_v * rate_f * phi
            + depth_f * c_c * dchat_v * phi
            + depth_f * rate_f * dphi_v
        )

    total = cal + cyc

    # --- optional plating guard (w_p = 0 is the V2G default, eq. 60) -------
    if guard_weight > 0.0:
        for t in range(n):
            pu = eta_in * x[t] / (b_max * durations[t])
            pv = lam * x[n + t] / (eta_out * b_max * durations[t])
            total += guard_weight * (pu * pu + pv * pv) * durations[t]
            jac[t] += guard_weight * 2.0 * eta_in * eta_in * x[t] / (b_max * b_max * durations[t])
            jac[n + t] += (
                guard_weight * 2.0 * lam * lam * x[n + t]
                / (eta_out * eta_out * b_max * b_max * durations[t])
            )

    return total, jac


def _net_square_objective(ctx: RequestContext, x) -> tuple[float, list[float]]:
    """Realization objective: squared NET grid power, eq. (61)."""
    n = ctx.n
    durations = interval_durations(ctx)
    value = 0.0
    jac = [0.0] * (2 * n)
    for t in range(n):
        net = (x[t] - x[n + t]) / durations[t]
        value += net * net * durations[t]
        jac[t] += 2.0 * net
        jac[n + t] += -2.0 * net
    return value, jac


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------


def warm_start(ctx: RequestContext) -> list[float]:
    """Unidirectional incumbent meeting the SAME ready-by and terminal guarantees.

    Charges (or, when the departure floor sits below the arrival SOC,
    discharges) at the maximum permitted rate from the start of the session.
    Feasible whenever the corresponding unidirectional service request is
    feasible, which is the property the formulation relies on.
    """
    n = ctx.n
    cap_in, cap_out = capacities(ctx)
    x = [0.0] * (2 * n)
    deficit = ctx.terminal_energy_kwh - ctx.session.initial_energy_kwh
    if deficit >= 0.0:
        remaining = deficit / ctx.ev.charging_efficiency
        for t in range(n):
            if remaining <= 0.0:
                break
            take = min(cap_in[t], remaining)
            x[t] = take
            remaining -= take
    else:
        remaining = -deficit * ctx.envelope.discharge_efficiency
        for t in range(n):
            if remaining <= 0.0:
                break
            take = min(cap_out[t], remaining)
            x[n + t] = take
            remaining -= take
    return x


# ---------------------------------------------------------------------------
# evaluation and solving
# ---------------------------------------------------------------------------


def evaluate(ctx: RequestContext, x, *, bau_cost: float) -> Trajectory:
    """Evaluate a decision vector: states, cost, saving, fade, violations."""
    n = ctx.n
    states = battery_states(ctx, x)
    soc = _soc(ctx, states)
    grid_in = tuple(float(v) for v in x[:n])
    grid_out = tuple(float(v) for v in x[n:])
    cost, _ = _cost_objective(ctx, x)
    fade = assess_trajectory(
        ev=ctx.ev,
        session=ctx.session,
        signal=ctx.signal,
        grid_in_kwh=grid_in,
        grid_out_kwh=grid_out,
        soc=soc,
        terminal_energy_kwh=ctx.terminal_energy_kwh,
        discharge_efficiency=ctx.envelope.discharge_efficiency,
        discharge_severity=ctx.discharge_severity,
        settings=ctx.settings,
    )
    simultaneous = sum(
        1
        for t in range(n)
        if grid_in[t] > _SIMULTANEITY_TOLERANCE and grid_out[t] > _SIMULTANEITY_TOLERANCE
    )
    violations: list[str] = []
    tol = _FEASIBILITY_TOLERANCE
    if abs(states[n] - ctx.terminal_energy_kwh) > tol:
        violations.append(f"terminal energy off by {states[n] - ctx.terminal_energy_kwh:.3e} kWh")
    if min(states) < ctx.reserve_energy_kwh - tol:
        violations.append("reserve floor breached")
    if max(states) > ctx.ev.battery_capacity_kwh + tol:
        violations.append("capacity ceiling breached")
    ready_local = max(0, min(n, ctx.ready_step - ctx.session.arrival_step))
    floor = ctx.target_soc * ctx.ev.battery_capacity_kwh
    if any(states[k] < floor - tol for k in range(ready_local, n + 1)):
        violations.append("availability breached after ready-by")
    allowance = ctx.envelope.session_export_allowance_kwh
    if allowance is not None and sum(grid_out) > allowance + tol:
        violations.append("session export allowance exceeded")
    return Trajectory(
        grid_in_kwh=grid_in,
        grid_out_kwh=grid_out,
        battery_energy_kwh=tuple(states),
        soc=tuple(soc),
        cost=cost,
        saving=bau_cost - cost,
        fade=fade,
        simultaneous_periods=simultaneous,
        feasible=not violations,
        violations=tuple(violations),
    )


def solve(
    ctx: RequestContext,
    *,
    objective: str,
    bau_cost: float,
    saving_low: float | None = None,
    saving_high: float | None = None,
    fade_cap: float | None = None,
    guard_weight: float = 0.0,
    restarts: int = 1,
    maxiter: int = 200,
) -> Trajectory | None:
    """Solve one trajectory problem, retaining the warm-start incumbent.

    ``objective`` is ``"cost"``, ``"fade"`` or ``"net_square"``.  The
    warm-start objective is kept and the returned point is rejected if it
    is worse: the formulation establishes that a no-worse feasible point
    exists, not that a nonlinear solver will return it.
    """
    n = ctx.n
    cap_in, cap_out = capacities(ctx)
    bounds = [(0.0, c) for c in cap_in] + [(0.0, c) for c in cap_out]
    cons = _build_constraints(
        ctx,
        bau_cost=bau_cost,
        saving_low=saving_low,
        saving_high=saving_high,
        fade_cap=fade_cap,
    )

    if objective == "cost":
        fun = lambda x: _cost_objective(ctx, x)[0]
        jac = lambda x: _cost_objective(ctx, x)[1]
    elif objective == "fade":
        fun = lambda x: _fade_objective(ctx, x, guard_weight=guard_weight)[0]
        jac = lambda x: _fade_objective(ctx, x, guard_weight=guard_weight)[1]
    elif objective == "net_square":
        fun = lambda x: _net_square_objective(ctx, x)[0]
        jac = lambda x: _net_square_objective(ctx, x)[1]
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown objective {objective!r}")

    incumbent = warm_start(ctx)
    best_x, best_value = None, None
    candidate_starts = [incumbent]
    if restarts > 1:
        import random

        rng = random.Random(0)
        for _ in range(restarts - 1):
            candidate_starts.append(
                [rng.uniform(0.0, b[1]) if b[1] > 0 else 0.0 for b in bounds]
            )

    # The warm start counts as an incumbent only when it is itself feasible,
    # which it need not be once a saving band is imposed.
    incumbent_trajectory = evaluate(ctx, incumbent, bau_cost=bau_cost)
    incumbent_ok = incumbent_trajectory.feasible and _within_band(
        incumbent_trajectory.saving, saving_low, saving_high
    )
    if incumbent_ok:
        best_x = incumbent
        best_value = fun(incumbent)

    for start in candidate_starts:
        try:
            result = minimize(
                fun,
                start,
                jac=jac,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": maxiter, "ftol": 1e-10},
            )
        except Exception:  # pragma: no cover - solver robustness
            continue
        if not result.success:
            continue
        candidate = [float(v) for v in result.x]
        trajectory = evaluate(ctx, candidate, bau_cost=bau_cost)
        if not trajectory.feasible or not _within_band(
            trajectory.saving, saving_low, saving_high
        ):
            continue
        value = fun(candidate)
        if best_value is None or value < best_value:
            best_x, best_value = candidate, value

    if best_x is None:
        return None
    best = evaluate(ctx, best_x, bau_cost=bau_cost)
    if best.simultaneous_periods == 0:
        return best
    repaired = _repair_complementarity(
        ctx,
        best,
        fun=fun,
        jac=jac,
        bounds=bounds,
        cons=cons,
        bau_cost=bau_cost,
        saving_low=saving_low,
        saving_high=saving_high,
        maxiter=maxiter,
    )
    return repaired if repaired is not None else best


def _repair_complementarity(
    ctx: RequestContext,
    relaxed: Trajectory,
    *,
    fun,
    jac,
    bounds: list[tuple[float, float]],
    cons: list[dict],
    bau_cost: float,
    saving_low: float | None,
    saving_high: float | None,
    maxiter: int,
) -> Trajectory | None:
    """Re-solve with the direction fixed wherever the relaxation overlapped.

    Proposition 3 makes the relaxation exact for the cost-minimising solve,
    but NOT for the band-constrained ones: the exchange that removes
    simultaneous operation raises the saving and can cross the upper band
    edge.  Rather than carrying a binary per interval through every solve,
    detect the overlap, fix ``u_i(t)`` to the dominant direction there, and
    re-solve the resulting continuous problem -- eq. (63) with the binaries
    pinned instead of searched.  Returns ``None`` if that fails, and the
    caller then keeps the relaxed point and reports the overlap.
    """
    n = ctx.n
    eta_in = ctx.ev.charging_efficiency
    eta_out = ctx.envelope.discharge_efficiency

    # Step 1: the Proposition 3 exchange map, applied in closed form.  One
    # pass drives one of the two directions to zero in every overlapping
    # interval, leaves the battery trajectory bit-identical (so the terminal
    # equality, floors and availability all survive) and weakly increases
    # the saving.  The only constraint it can break is the upper band edge.
    start = list(relaxed.grid_in_kwh) + list(relaxed.grid_out_kwh)
    round_trip = eta_in * eta_out
    for t in range(n):
        if start[t] <= _SIMULTANEITY_TOLERANCE or start[n + t] <= _SIMULTANEITY_TOLERANCE:
            continue
        epsilon = min(start[n + t], round_trip * start[t])
        start[n + t] -= epsilon
        start[t] -= epsilon / round_trip
        if start[t] < 0.0:
            start[t] = 0.0
        if start[n + t] < 0.0:
            start[n + t] = 0.0

    exchanged = evaluate(ctx, start, bau_cost=bau_cost)
    if (
        exchanged.simultaneous_periods == 0
        and exchanged.feasible
        and _within_band(exchanged.saving, saving_low, saving_high)
    ):
        return exchanged

    # Step 2: the exchange crossed the upper band edge.  Its direction
    # pattern is still the right one, so pin it and re-solve the resulting
    # continuous problem from that point.
    pinned = list(bounds)
    for t in range(n):
        if start[t] <= _SIMULTANEITY_TOLERANCE and relaxed.grid_out_kwh[t] > 0.0:
            pinned[t] = (0.0, 0.0)
        elif start[n + t] <= _SIMULTANEITY_TOLERANCE and relaxed.grid_in_kwh[t] > 0.0:
            pinned[n + t] = (0.0, 0.0)
    start = [
        min(max(value, pinned[i][0]), pinned[i][1]) for i, value in enumerate(start)
    ]
    try:
        result = minimize(
            fun,
            start,
            jac=jac,
            method="SLSQP",
            bounds=pinned,
            constraints=cons,
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
    except Exception:  # pragma: no cover - solver robustness
        return None
    if not result.success:
        return None
    candidate = evaluate(ctx, [float(v) for v in result.x], bau_cost=bau_cost)
    if not candidate.feasible or not _within_band(candidate.saving, saving_low, saving_high):
        return None
    if candidate.simultaneous_periods:
        return None
    return candidate


def _within_band(saving: float, low: float | None, high: float | None) -> bool:
    tol = 1e-6
    if low is not None and saving < low - tol:
        return False
    if high is not None and saving > high + tol:
        return False
    return True


def require_feasible(ctx: RequestContext) -> None:
    """Raise when the request cannot be served at all within the envelope."""
    x = warm_start(ctx)
    states = battery_states(ctx, x)
    if abs(states[ctx.n] - ctx.terminal_energy_kwh) > 1e-6:
        raise PhysicalConstraintError(
            "request is not serviceable: the session is too short to reach the "
            "terminal energy guarantee at the available rate."
        )
