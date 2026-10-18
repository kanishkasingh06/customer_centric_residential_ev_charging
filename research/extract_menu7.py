"""Extract the seven genuinely distinct menu options as plottable schedules.

Selection: Pareto-filtered displayed offers, then greedy farthest-point in
the normalised four-attribute space (ready-by, target SOC, saving, fade)
scored by L-infinity, cut at k=7 where the minimum pairwise separation is
still 0.339 and the next step drops to 0.239.
"""

from __future__ import annotations

import json
from pathlib import Path

from evmenu_v2g import V2GEnvelope, generate_ev_menu_v2g

RESEARCH_DIR = Path(__file__).parent
TARIFF = RESEARCH_DIR / "hourly_tariff.csv"
K = 7

ENV = V2GEnvelope(
    discharge_power_kw=7.2,
    discharge_efficiency=0.93,
    reserve_soc=0.20,
    session_export_allowance_kwh=15.0,
    allow_export_after_ready=False,
)
CASE = {
    "ev_model": "generic_60kwh_nmc",
    "arrival_time": "19:30",
    "departure_time": "08:00",
    "current_soc": 0.35,
    "next_trip_distance_km": 40.0,
    "battery_age_years": 4.0,
    "accumulated_equivalent_full_cycles": 1200.0,
    "levels": 4,
    "maximum_ready_steps": 4,
}


def farthest_point(offers, k):
    S = [o.saving for o in offers]
    F = [o.capacity_fade for o in offers]
    R = [o.ready_minute % 1440 for o in offers]
    T = [o.target_soc for o in offers]

    def n(v, lo, hi):
        return 0.0 if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    pts = [
        (n(o.ready_minute % 1440, min(R), max(R)), n(o.target_soc, min(T), max(T)),
         n(o.saving, min(S), max(S)), n(o.capacity_fade, min(F), max(F)))
        for o in offers
    ]
    d = lambda a, b: max(abs(x - y) for x, y in zip(a, b))
    sel = [0]
    while len(sel) < len(pts):
        sel.append(max((i for i in range(len(pts)) if i not in sel),
                       key=lambda i: min(d(pts[i], pts[j]) for j in sel)))
    chosen = sel[:k]
    sub = chosen
    sep = min(d(pts[a], pts[b]) for ii, a in enumerate(sub) for b in sub[ii + 1:])
    return chosen, sep


def main() -> None:
    menu = generate_ev_menu_v2g(envelope=ENV, tariff_csv=TARIFF, **CASE)
    offers = sorted(menu.displayed, key=lambda o: -o.saving)
    chosen, sep = farthest_point(offers, K)

    n_int = len(offers[0].trajectory.grid_in_kwh)
    departure_step = max(o.ready_step for o in menu.generated)
    arrival_step = departure_step - n_int
    step = menu.timestep_minutes
    arrival_minute = menu.arrival_minute
    minutes = [arrival_minute + i * step for i in range(n_int)]

    # Price over the session, recovered from the offers' own signal via the
    # saving/cost identity is not available here, so re-derive from the CSV.
    from evmenu.pricing import MINUTES_PER_WEEK, load_weekly_price_profile_csv
    weekly = load_weekly_price_profile_csv(TARIFF, profile_format="hour_of_week")
    day = 2  # Wednesday arrival, as in the menu build
    price = [
        round(weekly.price_at((day * 1440 + m) % MINUTES_PER_WEEK), 4) for m in minutes
    ]

    schedules = []
    for rank, idx in enumerate(chosen, 1):
        o = offers[idx]
        t = o.trajectory
        durations = step / 60.0
        schedules.append({
            "rank": rank,
            "ready_minute": int(o.ready_minute % 1440),
            "ready_local": max(0, min(n_int, o.ready_step - arrival_step)),
            "target_pct": round(o.target_soc * 100, 1),
            "saving": round(o.saving, 2),
            "cost": round(o.charging_cost, 2),
            "bau_cost": round(o.bau_cost, 2),
            "fade_pct": round(o.capacity_fade * 100, 6),
            "cycle_pct": round(o.cycle_fade * 100, 6),
            "cal_window_pct": round(o.calendar_window_fade * 100, 6),
            "cal_parked_pct": round(o.calendar_parked_fade * 100, 6),
            "import_kwh": round(o.import_energy_kwh, 2),
            "export_kwh": round(o.export_energy_kwh, 2),
            "efc": round(o.equivalent_full_cycles, 4),
            "excursion_pp": round(o.soc_excursion * 100, 1),
            "net_kw": [round((t.grid_in_kwh[i] - t.grid_out_kwh[i]) / durations, 4)
                       for i in range(n_int)],
            "soc": [round(s * 100, 3) for s in t.soc],
        })

    out = {
        "meta": {
            "ev": "60 kWh NMC, 7.2 kW onboard AC",
            "pack": "4 yr · 1200 EFC",
            "session": "19:30 → 08:00 (Wed → Thu)",
            "timestep_minutes": step,
            "arrival_soc_pct": round(CASE["current_soc"] * 100, 1),
            "service_minimum_pct": min(s["target_pct"] for s in schedules),
            "reserve_pct": ENV.reserve_soc * 100,
            "discharge_kw": ENV.discharge_power_kw,
            "allowance_kwh": ENV.session_export_allowance_kwh,
            "sell_ratio": 0.80,
            "tariff": "repo hourly_tariff.csv (hour_of_week)",
            "funnel": [len(menu.generated), len(menu.compacted), len(menu.displayed), K],
            "separation": round(sep, 3),
        },
        "minutes": minutes,
        "price": price,
        "schedules": schedules,
    }
    with (RESEARCH_DIR / "menu7.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"funnel {out['meta']['funnel']}  min separation {sep:.3f}")
    for s in schedules:
        print(f"  #{s['rank']} ready {s['ready_minute']//60:02}:{s['ready_minute']%60:02}"
              f"  target {s['target_pct']:5.1f}%  saving {s['saving']:7.2f}"
              f"  fade {s['fade_pct']:.5f}%  exp {s['export_kwh']:5.2f}  EFC {s['efc']:.3f}")


if __name__ == "__main__":
    main()
