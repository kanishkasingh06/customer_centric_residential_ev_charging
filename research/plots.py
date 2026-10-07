"""Prepare the dashboard payload from the sweep outputs.

Reads results.json + profiles.npz, seed-averages every cell, downsamples the
5-minute raster to the 15-minute nominal grid a DSO would actually meter, trims
the empty afternoon/late-morning tails, and inlines the result into
dashboard.html to produce dashboard_built.html.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEEDS = (11, 22, 33)
PREFS = ["cost_driven", "balanced", "health_driven", "convenience_driven"]
PATS = ["concentrated", "dispersed", "bimodal"]

# Raster starts at 12:00. Show 15:00 -> 09:00; outside that every scenario is
# flat zero and the dead width just shrinks the part anyone reads.
WINDOW = slice(12, 85)


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    profiles = np.load(HERE / "profiles.npz")

    labels_full = [f"{(12*60+i*15)//60%24:02d}:{(12*60+i*15)%60:02d}" for i in range(96)]

    def profile(pattern: str, preference: str, size: int) -> list[float]:
        averaged = np.mean(
            [profiles[f"{pattern}|{preference}|{size}|{s}"] for s in SEEDS], axis=0
        )
        quarter = [float(averaged[i * 3 : (i + 1) * 3].mean()) for i in range(96)]
        return [round(v, 2) for v in quarter[WINDOW]]

    def metric(pattern: str, preference: str, size: int, key: str):
        rows = [
            r for r in results
            if r["pattern"] == pattern and r["preference"] == preference
            and r["fleet_size"] == size
        ]
        if key == "peak_time":
            times = [r[key] for r in rows]
            return max(set(times), key=times.count)
        return round(float(np.mean([r[key] for r in rows])), 3)

    data = {
        "labels": labels_full[WINDOW],
        "byPreference": {p: profile("concentrated", p, 150) for p in PREFS},
        "byPattern": {p: profile(p, "balanced", 150) for p in PATS},
        "scaling": {
            p: {
                "n": [40, 80, 150],
                "peak": [metric("concentrated", p, n, "peak_kw") for n in (40, 80, 150)],
                "cf": [metric("concentrated", p, n, "coincidence_factor") for n in (40, 80, 150)],
            }
            for p in PREFS
        },
        "table": [
            {
                "pattern": pat, "preference": pref,
                "peak": metric(pat, pref, 150, "peak_kw"),
                "peakTime": metric(pat, pref, 150, "peak_time"),
                "energy": metric(pat, pref, 150, "energy_kwh"),
                "cf": metric(pat, pref, 150, "coincidence_factor"),
                "cost": metric(pat, pref, 150, "mean_cost_rs"),
                "wear": metric(pat, pref, 150, "mean_wear_rs"),
                "target": metric(pat, pref, 150, "mean_target_soc"),
                "noCharge": metric(pat, pref, 150, "share_no_charge"),
                "delay": metric(pat, pref, 150, "mean_ready_delay_h"),
            }
            for pat in PATS for pref in PREFS
        ],
    }

    payload = json.dumps(data, separators=(",", ":"))
    (HERE / "viz_data.json").write_text(payload, encoding="utf-8")

    template = (HERE / "dashboard.html").read_text(encoding="utf-8")
    if "__DATA__" not in template:
        raise SystemExit("dashboard.html is missing the __DATA__ placeholder")
    (HERE / "dashboard_built.html").write_text(
        template.replace("__DATA__", payload), encoding="utf-8"
    )
    peak = max(max(v) for v in data["byPreference"].values())
    print(f"window {data['labels'][0]}-{data['labels'][-1]} ({len(data['labels'])} points), "
          f"max {peak:.1f} kW -> dashboard_built.html")


if __name__ == "__main__":
    main()
