"""Extract real charging trajectories from evmenu_v2g for visualisation.

Same EV, same request (ready-by 08:00, target 80%) under five different
objectives, so the profiles are directly comparable.
"""

from __future__ import annotations

import json

from evmenu.catalog import get_ev_model
from evmenu.degradation import DegradationSettings
from evmenu.schemas import EVSpec

from evmenu_v2g import V1G_ENVELOPE, V2GEnvelope
from evmenu.pricing import load_weekly_price_profile_csv
from evmenu_v2g.api import build_session_and_signal
from evmenu_v2g.menu import bau_cost_for, saving_levels
from evmenu_v2g.trajectory import RequestContext, evaluate, solve, warm_start

TIMESTEP = 30
MODEL = "generic_60kwh_nmc"
ARRIVAL, DEPARTURE = "19:30", "08:00"
CURRENT_SOC = 0.35
TRIP_KM = 40.0
TARGET = 0.80  # the 80% tier; the service minimum is reported separately

TARIFF_CSV = "data/hourly_tariff.csv"
ARRIVAL_DAY = "Wed"
RESERVE = 0.20

V2G = V2GEnvelope(
    discharge_power_kw=7.2,
    discharge_efficiency=0.93,
    reserve_soc=RESERVE,
    session_export_allowance_kwh=15.0,
    allow_export_after_ready=False,
)


def build():
    model = get_ev_model(MODEL)
    ev = EVSpec(
        ev_id="EV-A",
        battery_capacity_kwh=model.usable_battery_kwh,
        minimum_energy_kwh=model.minimum_soc * model.usable_battery_kwh,
        charger_power_kw=model.onboard_ac_power_kw,
        charging_efficiency=model.charging_efficiency,
        chemistry=model.chemistry,
    )
    weekly = load_weekly_price_profile_csv(TARIFF_CSV, profile_format="hour_of_week")
    session, signal = build_session_and_signal(
        ev=ev,
        arrival_time=ARRIVAL,
        departure_time=DEPARTURE,
        current_soc=CURRENT_SOC,
        next_trip_distance_km=TRIP_KM,
        consumption_kwh_per_km=model.consumption_kwh_per_km,
        buffer_soc=0.10,
        timestep_minutes=TIMESTEP,
        weekly_profile=weekly,
        arrival_day=ARRIVAL_DAY,
        battery_temperature_c=25.0,
    )
    settings = DegradationSettings(
        battery_age_years=4.0,
        cumulative_equivalent_full_cycles=1200.0,
        fallback_temperature_c=25.0,
    )
    return ev, session, signal, settings


def context(ev, session, signal, settings, envelope):
    n = session.departure_step - session.arrival_step
    sell = tuple(
        signal.price_per_kwh[session.arrival_step + i] * 0.80 for i in range(n)
    )
    return RequestContext(
        ev=ev,
        session=session,
        signal=signal,
        envelope=envelope,
        settings=settings,
        ready_step=session.departure_step,
        target_soc=TARGET,
        sell_price=sell,
        discharge_severity=1.0,
    )


def series(ctx, traj, bau):
    durations = [
        ctx.signal.interval_durations[ctx.session.arrival_step + i] for i in range(ctx.n)
    ]
    starts = ctx.signal.interval_start_minutes
    return {
        "minutes": [int(starts[ctx.session.arrival_step + i]) for i in range(ctx.n)],
        "net_kw": [
            round((traj.grid_in_kwh[i] - traj.grid_out_kwh[i]) / durations[i], 4)
            for i in range(ctx.n)
        ],
        "soc": [round(s * 100, 3) for s in traj.soc],
        "saving": round(traj.saving, 2),
        "cost": round(traj.cost, 2),
        "bau_cost": round(bau, 2),
        "fade_pct": round(traj.fade.total * 100, 6),
        "cal_window": round(traj.fade.calendar_window * 100, 6),
        "cal_parked": round(traj.fade.calendar_parked * 100, 6),
        "cycle": round(traj.fade.cycle * 100, 6),
        "import_kwh": round(sum(traj.grid_in_kwh), 2),
        "export_kwh": round(sum(traj.grid_out_kwh), 2),
        "efc": round(traj.fade.equivalent_full_cycles, 4),
        "excursion": round(traj.fade.soc_excursion * 100, 2),
        "peak_c": round(traj.fade.peak_c_rate, 4),
    }


SERVICE_MIN = 0.0


def main() -> None:
    global SERVICE_MIN
    ev, session, signal, settings = build()
    from evmenu_v2g.menu import service_minimum_soc
    SERVICE_MIN = service_minimum_soc(ev=ev, session=session, reserve_soc=RESERVE)
    ctx1 = context(ev, session, signal, settings, V1G_ENVELOPE)
    ctx2 = context(ev, session, signal, settings, V2G)
    bau = bau_cost_for(
        ev=ev, session=session, signal=signal, terminal_energy_kwh=ctx1.terminal_energy_kwh
    )

    profiles = {}

    # 1. BAU: immediate full-rate charging, the same-service reference.
    profiles["bau"] = series(ctx1, evaluate(ctx1, warm_start(ctx1), bau_cost=bau), bau)

    # 2/3. V1G cost-minimising and health endpoint.
    v1g_cost = solve(ctx1, objective="cost", bau_cost=bau)
    profiles["v1g_max_saving"] = series(ctx1, v1g_cost, bau)
    lo = max(0.0, 0.0 - 0.5)
    v1g_health = solve(
        ctx1, objective="fade", bau_cost=bau, saving_low=lo, saving_high=0.5
    )
    profiles["v1g_health"] = series(ctx1, v1g_health, bau)

    # 4/5/6. V2G cost-minimising, mid-level, health endpoint.
    v2g_cost = solve(ctx2, objective="cost", bau_cost=bau)
    profiles["v2g_max_saving"] = series(ctx2, v2g_cost, bau)
    levels = saving_levels(v2g_cost.saving, levels=4)
    mid = levels[2]
    v2g_mid = solve(
        ctx2, objective="fade", bau_cost=bau,
        saving_low=max(0.0, mid - 0.5), saving_high=mid + 0.5,
    )
    profiles["v2g_mid"] = series(ctx2, v2g_mid, bau)
    v2g_health = solve(
        ctx2, objective="fade", bau_cost=bau, saving_low=0.0, saving_high=0.5
    )
    profiles["v2g_health"] = series(ctx2, v2g_health, bau)

    n = ctx1.n
    out = {
        "meta": {
            "tariff": "repo hourly_tariff.csv (hour_of_week), Wed arrival",
            "service_minimum_pct": round(SERVICE_MIN * 100, 1),
            "reserve_soc_pct_decl": RESERVE * 100,
            "ev": "60 kWh NMC, 7.2 kW onboard AC",
            "pack_age_years": 4.0,
            "accumulated_fec": 1200.0,
            "session": f"{ARRIVAL} to {DEPARTURE}",
            "timestep_minutes": TIMESTEP,
            "arrival_soc_pct": CURRENT_SOC * 100,
            "target_soc_pct": TARGET * 100,
            "terminal_kwh": round(ctx1.terminal_energy_kwh, 2),
            "reserve_soc_pct": V2G.reserve_soc * 100,
            "discharge_kw": V2G.discharge_power_kw,
            "allowance_kwh": V2G.session_export_allowance_kwh,
            "sell_ratio": 0.80,
            "bau_cost": round(bau, 2),
        },
        "minutes": [
            int(signal.interval_start_minutes[session.arrival_step + i]) for i in range(n)
        ],
        "price": [
            round(signal.price_per_kwh[session.arrival_step + i], 3) for i in range(n)
        ],
        "profiles": profiles,
    }
    with open("profiles.json", "w") as handle:
        json.dump(out, handle, indent=1)
    for key, value in profiles.items():
        print(
            f"{key:16} saving {value['saving']:8.2f}  fade {value['fade_pct']:.5f}%"
            f"  imp {value['import_kwh']:6.2f}  exp {value['export_kwh']:6.2f}"
            f"  EFC {value['efc']:.3f}  swing {value['excursion']:5.1f}pp"
        )


if __name__ == "__main__":
    main()
