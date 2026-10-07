"""Generate V2G menus for two EVs and report the runtime the Block 9
tractability projection cares about.

    python demo_v2g.py
"""

from __future__ import annotations

import time

from evmenu_v2g import V1G_ENVELOPE, V2GEnvelope, format_menu, generate_ev_menu_v2g

ENVELOPE = V2GEnvelope(
    discharge_power_kw=7.2,
    discharge_efficiency=0.93,
    reserve_soc=0.25,
    departure_floor_soc=None,       # default: full arrival-energy protection
    session_export_allowance_kwh=15.0,
)

CASES = [
    dict(
        ev_model="generic_60kwh_nmc",
        ev_id="EV-A (NMC, aged pack)",
        arrival_time="19:30",
        departure_time="08:00",
        current_soc=0.35,
        next_trip_distance_km=40.0,
        battery_age_years=4.0,
        accumulated_equivalent_full_cycles=1200.0,
    ),
    dict(
        ev_model="generic_40kwh_lfp",
        ev_id="EV-B (LFP, newer pack)",
        arrival_time="18:00",
        departure_time="07:30",
        current_soc=0.45,
        next_trip_distance_km=25.0,
        battery_age_years=1.5,
        accumulated_equivalent_full_cycles=300.0,
    ),
]


def run() -> None:
    for case in CASES:
        for label, envelope in (("V1G", V1G_ENVELOPE), ("V2G", ENVELOPE)):
            start = time.perf_counter()
            menu = generate_ev_menu_v2g(
                envelope=envelope,
                levels=3,
                maximum_ready_steps=3,
                sell_to_buy_ratio=0.80,
                **case,
            )
            elapsed = time.perf_counter() - start
            print(f"\n{'=' * 72}\n{label}  —  {elapsed:.2f}s")
            print(format_menu(menu))
            if menu.displayed:
                best_saving = max(o.saving for o in menu.displayed)
                least_fade = min(o.capacity_fade for o in menu.displayed)
                print(
                    f"  best saving {best_saving:.2f}   least fade"
                    f" {least_fade * 100:.5f}%   export offers"
                    f" {menu.export_offer_count}/{len(menu.displayed)}"
                )


if __name__ == "__main__":
    run()
