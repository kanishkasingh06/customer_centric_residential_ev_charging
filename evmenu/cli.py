"""Command-line interface for deterministic single-EV menu generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, NoReturn, TextIO

from .assembly import MenuAssemblySettings
from .catalog import list_ev_models
from .exceptions import EVMenuError
from .pricing import load_price_profile_csv
from .service import GeneratedCustomerMenu, generate_ev_menu


class _CLIArgumentParser(argparse.ArgumentParser):
    """Argument parser that keeps help and errors on the caller's streams."""

    def __init__(
        self,
        *args: Any,
        output_stream: TextIO | None = None,
        error_stream: TextIO | None = None,
        **kwargs: Any,
    ) -> None:
        self._output_stream = output_stream if output_stream is not None else sys.stdout
        self._error_stream = error_stream if error_stream is not None else sys.stderr
        super().__init__(*args, **kwargs)

    def print_help(self, file: Any = None) -> None:
        super().print_help(self._output_stream if file is None else file)

    def print_usage(self, file: Any = None) -> None:
        super().print_usage(self._error_stream if file is None else file)

    def error(self, message: str) -> NoReturn:
        self.print_usage(file=self._error_stream)
        self._print_message(f"{self.prog}: error: {message}\n", self._error_stream)
        self.exit(2)


def _soc_fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SOC must be a number in percent, from 0 to 100.") from exc
    if not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("SOC must lie between 0 and 100 percent.")
    return parsed / 100.0


def _finite_float(name: str, *, minimum: float | None = None) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a number.") from exc
        if not isfinite(parsed):
            raise argparse.ArgumentTypeError(f"{name} must be finite.")
        if minimum is not None and parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} must be at least {minimum}.")
        return parsed

    return parse


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def build_parser(
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = _CLIArgumentParser(
        prog="evmenu",
        description="Generate deterministic residential EV charging menus.",
        allow_abbrev=False,
        output_stream=stdout,
        error_stream=stderr,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_CLIArgumentParser,
    )

    generate = subparsers.add_parser(
        "generate",
        help="generate a customer charging menu",
        allow_abbrev=False,
        output_stream=stdout,
        error_stream=stderr,
    )
    generate.add_argument("--ev-model", required=True, help="case-sensitive catalogue model ID")
    generate.add_argument("--arrival", required=True, help="strict local 24-hour HH:MM")
    generate.add_argument("--departure", required=True, help="strict local 24-hour HH:MM")
    generate.add_argument(
        "--current-soc",
        required=True,
        type=_soc_fraction,
        metavar="PERCENT",
        help="current battery SOC in percent, e.g. 35",
    )
    generate.add_argument(
        "--next-trip-km",
        required=True,
        type=_finite_float("next-trip-km", minimum=0.0),
        help="next-trip distance in kilometres",
    )
    generate.add_argument(
        "--buffer-soc",
        type=_soc_fraction,
        default=0.10,
        metavar="PERCENT",
        help="safety buffer in percent of usable capacity (default: 10)",
    )
    generate.add_argument(
        "--tariff",
        choices=("research_tou", "flat", "custom"),
        default="research_tou",
        help="illustrative research TOU, flat, or machine-readable custom tariff",
    )
    generate.add_argument(
        "--flat-price",
        type=_finite_float("flat-price"),
        default=None,
        help="currency/kWh for --tariff flat (default: 7.0; negative prices supported)",
    )
    generate.add_argument(
        "--temperature-c",
        type=_finite_float("temperature-c", minimum=-273.149999),
        default=30.0,
        help="constant battery temperature in degrees Celsius (default: 30)",
    )
    generate.add_argument(
        "--timestep-minutes",
        type=_positive_int,
        default=15,
        help="nominal wall-clock grid in minutes; must be a positive divisor of 1440",
    )
    generate.add_argument(
        "--display-cap",
        type=_positive_int,
        default=None,
        help="maximum displayed offers, including required BAU references",
    )
    generate.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    generate.add_argument(
        "--include-schedule",
        action="store_true",
        help="include interval grid-power arrays in JSON output",
    )
    generate.add_argument(
        "--include-intervals",
        action="store_true",
        help="include exact interval boundaries, durations, and prices in JSON output",
    )
    generate.add_argument(
        "--price-profile",
        help="CSV price profile path for --tariff custom",
    )
    generate.add_argument(
        "--price-profile-format",
        choices=("weekly", "hour_of_week", "timestamped"),
        help="custom CSV schema (weekly, hour_of_week, or timestamped)",
    )
    generate.add_argument(
        "--arrival-day",
        choices=("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
        help="arrival weekday for recurring weekly custom profiles",
    )
    generate.add_argument(
        "--arrival-date",
        help="arrival date YYYY-MM-DD for timestamped custom profiles",
    )

    subparsers.add_parser(
        "models",
        help="list built-in illustrative EV models",
        allow_abbrev=False,
        output_stream=stdout,
        error_stream=stderr,
    )
    subparsers.add_parser(
        "tariffs",
        help="list supported tariff identifiers",
        allow_abbrev=False,
        output_stream=stdout,
        error_stream=stderr,
    )
    return parser


def _menu_payload(
    menu: GeneratedCustomerMenu,
    *,
    include_schedule: bool,
    include_intervals: bool,
) -> dict[str, object]:
    offers: list[dict[str, object]] = []
    for row in menu.offers:
        row_payload: dict[str, object] = {
            "offer_id": row.offer_id,
            "ready_time": row.ready_time,
            "target_soc_percent": row.target_soc_percent,
            "charging_cost": row.charging_cost,
            "saving": row.saving,
            "health_score": row.health_score,
            "energy_drawn_kwh": row.energy_drawn_kwh,
            "role": row.role,
        }
        if include_schedule:
            row_payload["charging_schedule_kw"] = list(row.charging_schedule_kw)
        offers.append(row_payload)
    payload: dict[str, object] = {
        "ev_model_id": menu.ev_model.model_id,
        "ev_model_name": menu.ev_model.display_name,
        "arrival_time": menu.arrival_time,
        "departure_time": menu.departure_time,
        "timestep_minutes": menu.timestep_minutes,
        "current_soc_percent": menu.current_soc * 100.0,
        "next_trip_distance_km": menu.next_trip_distance_km,
        "tariff_name": menu.tariff_name,
        "tariff_is_illustrative": menu.tariff_is_illustrative,
        "offers": offers,
    }
    if menu.profile_id is not None:
        payload["price_profile_id"] = menu.profile_id
    payload["currency_label"] = menu.currency_label
    if menu.arrival_day is not None:
        payload["arrival_day"] = menu.arrival_day
    if menu.arrival_date is not None:
        payload["arrival_date"] = menu.arrival_date
    if include_intervals:
        payload["intervals"] = [
            {
                "start_time": menu.interval_start_times[index],
                "end_time": menu.interval_end_times[index],
                "start_minute": menu.interval_start_minutes[index],
                "end_minute": menu.interval_end_minutes[index],
                "duration_minutes": menu.interval_duration_minutes[index],
                "price_per_kwh": menu.interval_price_per_kwh[index],
            }
            for index in range(len(menu.interval_start_minutes))
        ]
    return payload


def _render_text(menu: GeneratedCustomerMenu, stream: TextIO) -> None:
    print(f"EV: {menu.ev_model.display_name} ({menu.ev_model.model_id})", file=stream)
    print(
        "Assumptions: built-in EV model values are illustrative research assumptions.",
        file=stream,
    )
    print(
        f"Connected: {menu.arrival_time}-{menu.departure_time} | "
        f"SOC: {menu.current_soc * 100.0:.1f}% | "
        f"Next trip: {menu.next_trip_distance_km:g} km",
        file=stream,
    )
    tariff_note = " (illustrative)" if menu.tariff_is_illustrative else ""
    print(
        f"Tariff: {menu.tariff_name}{tariff_note} | Nominal interval: {menu.timestep_minutes} min | "
        f"Generated intervals: {len(menu.interval_duration_minutes)}",
        file=stream,
    )
    print(file=stream)
    print(
        f"#  Ready  Target   Cost({menu.currency_label})  "
        f"Saving({menu.currency_label})  Health(score)  Role",
        file=stream,
    )
    for index, row in enumerate(menu.offers, start=1):
        print(
            f"{index:<2} {row.ready_time:<6} "
            f"{row.target_soc_percent:>6.1f}% "
            f"{row.charging_cost:>10.2f} "
            f"{row.saving:>10.2f} "
            f"{row.health_score:>7.1f}  {row.role}",
            file=stream,
        )


def _run_generate(
    namespace: argparse.Namespace,
    *,
    stdout: TextIO,
) -> None:
    assembly_settings = (
        MenuAssemblySettings(display_cap=namespace.display_cap)
        if namespace.display_cap is not None
        else None
    )
    custom_profile = None
    if namespace.price_profile is not None:
        custom_profile = load_price_profile_csv(
            namespace.price_profile,
            profile_format=namespace.price_profile_format,
        )
    menu = generate_ev_menu(
        ev_model=namespace.ev_model,
        arrival_time=namespace.arrival,
        departure_time=namespace.departure,
        current_soc=namespace.current_soc,
        next_trip_distance_km=namespace.next_trip_km,
        buffer_soc=namespace.buffer_soc,
        tariff_name=namespace.tariff,
        flat_price_per_kwh=7.0 if namespace.flat_price is None else namespace.flat_price,
        battery_temperature_c=namespace.temperature_c,
        timestep_minutes=namespace.timestep_minutes,
        assembly_settings=assembly_settings,
        custom_price_profile=custom_profile,
        arrival_day=namespace.arrival_day,
        arrival_date=namespace.arrival_date,
    )
    if namespace.format == "json":
        json.dump(
            _menu_payload(
                menu,
                include_schedule=namespace.include_schedule,
                include_intervals=namespace.include_intervals,
            ),
            stdout,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stdout.write("\n")
    else:
        _render_text(menu, stdout)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code without calling ``sys.exit``."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser(stdout=out, stderr=err)
    try:
        namespace = parser.parse_args(argv)
        if namespace.command == "models":
            for model in list_ev_models():
                print(
                    f"{model.model_id}\t{model.display_name}\t"
                    f"capacity={model.usable_battery_kwh:g} kWh\t"
                    f"charger_power={model.onboard_ac_power_kw:g} kW\t"
                    f"chemistry={model.chemistry}\t"
                    f"consumption={model.consumption_kwh_per_km:g} kWh/km\t"
                    f"{model.assumption_note}",
                    file=out,
                )
        elif namespace.command == "tariffs":
            print(
                "research_tou\tIllustrative research assumption; not official/current\t"
                "00:00-06:00=4.0;06:00-17:00=7.0;"
                "17:00-23:00=10.0;23:00-24:00=5.0 currency/kWh",
                file=out,
            )
            print(
                "flat\tCaller-configurable constant price; default 7.0 currency/kWh; "
                "finite negative values supported",
                file=out,
            )
            print(
                "custom\tMachine-readable CSV profile; use --price-profile and an explicit "
                "weekly arrival day or timestamped arrival date",
                file=out,
            )
        else:
            if namespace.flat_price is not None and namespace.tariff != "flat":
                print(
                    "evmenu: error: --flat-price requires --tariff flat",
                    file=err,
                )
                return 2
            if namespace.price_profile is not None and namespace.tariff != "custom":
                print("evmenu: error: --price-profile requires --tariff custom", file=err)
                return 2
            if namespace.tariff == "custom" and namespace.price_profile is None:
                print("evmenu: error: --tariff custom requires --price-profile", file=err)
                return 2
            if namespace.price_profile is not None and namespace.price_profile_format is None:
                print(
                    "evmenu: error: --price-profile-format is required with --price-profile",
                    file=err,
                )
                return 2
            if (
                namespace.tariff == "custom"
                and namespace.price_profile_format in ("weekly", "hour_of_week")
                and namespace.arrival_date is not None
            ):
                print(
                    "evmenu: error: weekly custom profiles reject --arrival-date; use --arrival-day",
                    file=err,
                )
                return 2
            if (
                namespace.tariff == "custom"
                and namespace.price_profile_format in ("weekly", "hour_of_week")
                and namespace.arrival_day is None
            ):
                print("evmenu: error: weekly custom profiles require --arrival-day", file=err)
                return 2
            if (
                namespace.tariff == "custom"
                and namespace.price_profile_format == "timestamped"
                and namespace.arrival_day is not None
            ):
                print(
                    "evmenu: error: timestamped custom profiles reject --arrival-day; use --arrival-date",
                    file=err,
                )
                return 2
            if (
                namespace.tariff == "custom"
                and namespace.price_profile_format == "timestamped"
                and namespace.arrival_date is None
            ):
                print("evmenu: error: timestamped custom profiles require --arrival-date", file=err)
                return 2
            if namespace.tariff != "custom" and (
                namespace.price_profile_format is not None
                or namespace.arrival_day is not None
                or namespace.arrival_date is not None
            ):
                print("evmenu: error: custom profile options require --tariff custom", file=err)
                return 2
            if namespace.include_schedule and namespace.format != "json":
                print(
                    "evmenu: error: --include-schedule requires --format json",
                    file=err,
                )
                return 2
            if namespace.include_intervals and namespace.format != "json":
                print(
                    "evmenu: error: --include-intervals requires --format json",
                    file=err,
                )
                return 2
            _run_generate(namespace, stdout=out)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except EVMenuError as exc:
        print(f"evmenu: error: {exc}", file=err)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through evmenu.__main__
    raise SystemExit(main())
