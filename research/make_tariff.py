"""Generate the hourly-varying weekly tariff used for the choice-modelling phase.

The shape is a residential duck-curve ToU: cheap overnight, a cheap solar
midday trough, an expensive evening peak, with a flatter weekend. A small
smooth hour-to-hour component is added on purpose -- a purely blocky tariff
makes the minimum-cost allocation degenerate (many exactly-tied intervals),
which suppresses menu diversity for reasons that have nothing to do with
customer preferences.

Values are illustrative research assumptions in Rs/kWh. They are NOT a
regulated retail tariff.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUT = Path(__file__).parent / "hourly_tariff.csv"

# Weekday base shape by hour of day (Rs/kWh), illustrative.
_WEEKDAY = [
    4.20, 4.00, 3.90, 3.90, 4.10, 4.60,   # 00-05 overnight trough
    6.30, 7.60, 8.10, 7.40, 6.20, 5.40,   # 06-11 morning ramp then solar
    5.00, 4.90, 5.10, 5.80, 7.20, 9.10,   # 12-17 midday trough into ramp
    10.60, 10.90, 10.20, 8.80, 7.10, 5.40,  # 18-23 evening peak, decay
]


def weekend_price(hour: float) -> float:
    """Flatter weekend shape: lower peak, shallower midday trough."""
    base = _WEEKDAY[int(hour)]
    mean = sum(_WEEKDAY) / 24.0
    return round(mean + 0.65 * (base - mean), 3)


def build_rows() -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for hour_of_week in range(168):
        day, hour = divmod(hour_of_week, 24)
        if day < 5:  # Mon-Fri
            price = _WEEKDAY[hour]
        else:  # Sat-Sun
            price = weekend_price(hour)
        # Smooth deterministic ripple so adjacent hours are rarely exactly tied.
        ripple = 0.18 * math.sin(2.0 * math.pi * hour_of_week / 17.0)
        rows.append((hour_of_week, round(price + ripple, 4)))
    return rows


def main() -> None:
    rows = build_rows()
    assert len(rows) == 168, "weekly hour-of-week profile must have exactly 168 rows"
    assert all(price > 0 for _, price in rows), "illustrative tariff should stay positive"
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hour_of_week", "price"])
        writer.writerows(rows)
    weekday = [p for h, p in rows if h < 120]
    print(f"wrote {OUT} ({len(rows)} rows)")
    print(f"  weekday min={min(weekday):.2f}  max={max(weekday):.2f}  "
          f"mean={sum(weekday)/len(weekday):.2f} Rs/kWh")
    print(f"  distinct prices: {len({p for _, p in rows})} of 168 (near-unique => few allocation ties)")


if __name__ == "__main__":
    main()
