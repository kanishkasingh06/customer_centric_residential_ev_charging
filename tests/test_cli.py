from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from evmenu import PricePeriod, WeeklyPriceProfile, generate_ev_menu
from evmenu.cli import _render_text, build_parser, main


def _run(*args: str) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = main(args, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_models_lists_catalogue_deterministically() -> None:
    code, stdout, stderr = _run("models")
    assert code == 0
    assert stderr == ""
    lines = stdout.splitlines()
    assert lines == sorted(lines)
    assert any(line.startswith("generic_40kwh_lfp\t") for line in lines)
    assert any(line.startswith("generic_60kwh_nmc\t") for line in lines)
    assert all("capacity=" in line for line in lines)
    assert all("charger_power=" in line for line in lines)
    assert all("consumption=" in line for line in lines)
    assert all("Illustrative research assumption" in line for line in lines)


def test_tariffs_lists_supported_identifiers() -> None:
    code, stdout, stderr = _run("tariffs")
    assert code == 0
    assert stderr == ""
    assert stdout.splitlines() == [
        (
            "research_tou\tIllustrative research assumption; not official/current\t"
            "00:00-06:00=4.0;06:00-17:00=7.0;"
            "17:00-23:00=10.0;23:00-24:00=5.0 currency/kWh"
        ),
        (
            "flat\tCaller-configurable constant price; default 7.0 currency/kWh; "
            "finite negative values supported"
        ),
        (
            "custom\tMachine-readable CSV profile; use --price-profile and an explicit "
            "weekly arrival day or timestamped arrival date"
        ),
    ]


def test_generate_text_smoke_and_determinism() -> None:
    args = (
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
    )
    first = _run(*args)
    second = _run(*args)
    assert first == second
    code, stdout, stderr = first
    assert code == 0
    assert stderr == ""
    assert "Generic 40 kWh LFP EV" in stdout
    assert "Connected: 19:00-07:00" in stdout
    assert "illustrative research assumptions" in stdout
    assert "Cost(currency)" in stdout
    assert "Ready" in stdout
    assert "bau" in stdout


def test_parser_errors_return_two_and_use_injected_stderr() -> None:
    code, stdout, stderr = _run("generate")
    assert code == 2
    assert stdout == ""
    assert "required" in stderr


def test_help_uses_injected_stdout() -> None:
    code, stdout, stderr = _run("--help")
    assert code == 0
    assert "usage: evmenu" in stdout
    assert stderr == ""


def test_importing_module_entry_point_has_no_side_effect() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import evmenu.__main__"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "missing",
    ["--ev-model", "--arrival", "--departure", "--current-soc", "--next-trip-km"],
)
def test_required_generate_arguments_are_rejected(missing: str) -> None:
    args = [
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
    ]
    index = args.index(missing)
    del args[index : index + 2]
    code, stdout, stderr = _run(*args)
    assert code == 2
    assert stdout == ""
    assert missing in stderr


def test_ambiguous_option_abbreviation_is_rejected() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-mo",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
    )
    assert code == 2
    assert stdout == ""
    assert "--ev-model" in stderr


def test_semantically_incompatible_generate_options_are_rejected() -> None:
    common = (
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
    )
    code, stdout, stderr = _run(*common, "--flat-price", "9")
    assert (code, stdout) == (2, "")
    assert "requires --tariff flat" in stderr

    code, stdout, stderr = _run(*common, "--include-schedule")
    assert (code, stdout) == (2, "")
    assert "requires --format json" in stderr


def test_generate_json_is_machine_readable_and_schedule_optional() -> None:
    base = (
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--format",
        "json",
    )
    code, stdout, stderr = _run(*base)
    assert code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["ev_model_id"] == "generic_40kwh_lfp"
    assert payload["timestep_minutes"] == 15
    assert payload["offers"]
    assert "charging_schedule_kw" not in payload["offers"][0]

    code, stdout, stderr = _run(*base, "--include-schedule")
    assert code == 0
    assert stderr == ""
    scheduled = json.loads(stdout)
    assert len(scheduled["offers"][0]["charging_schedule_kw"]) == 48


def test_generate_supports_flat_negative_price_and_30_minute_steps() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--tariff",
        "flat",
        "--flat-price",
        "-1.5",
        "--timestep-minutes",
        "30",
        "--format",
        "json",
        "--include-schedule",
    )
    assert code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["tariff_name"] == "flat"
    assert payload["tariff_is_illustrative"] is False
    assert len(payload["offers"][0]["charging_schedule_kw"]) == 24


def test_domain_error_uses_stderr_and_exit_code_two() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "missing",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
    )
    assert code == 2
    assert stdout == ""
    assert "Unknown ev_model" in stderr


def test_impossible_display_cap_is_reported() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--display-cap",
        "1",
    )
    assert code == 2
    assert stdout == ""
    assert "display_cap" in stderr


@pytest.mark.parametrize("value", ["-1", "101", "nan", "inf"])
def test_soc_argument_validation_exits_via_argparse(value: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            [
                "generate",
                "--ev-model",
                "generic_40kwh_lfp",
                "--arrival",
                "19:00",
                "--departure",
                "07:00",
                "--current-soc",
                value,
                "--next-trip-km",
                "45",
            ]
        )
    assert raised.value.code == 2


def test_buffer_soc_uses_percentage_cli_convention() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--buffer-soc",
        "5",
        "--format",
        "json",
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["offers"]


def test_arbitrary_times_and_interval_metadata_are_json_auditable() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "11:07",
        "--departure",
        "18:52",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--format",
        "json",
        "--include-intervals",
    )
    assert (code, stderr) == (0, "")
    payload = json.loads(stdout)
    assert payload["intervals"][0]["duration_minutes"] == 8
    assert payload["intervals"][-1]["duration_minutes"] == 7
    assert payload["intervals"][0]["start_time"] == "11:07"
    assert payload["intervals"][-1]["end_time"] == "18:52"


def test_custom_hourly_csv_is_accepted_and_options_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "hour_of_week,price\n" + "".join(f"{hour},{hour / 10:g}\n" for hour in range(168)),
        encoding="utf-8",
    )
    common = (
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "11:07",
        "--departure",
        "18:52",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--tariff",
        "custom",
        "--price-profile",
        str(path),
        "--price-profile-format",
        "hour_of_week",
        "--arrival-day",
        "Mon",
        "--format",
        "json",
        "--include-intervals",
    )
    code, stdout, stderr = _run(*common)
    assert (code, stderr) == (0, "")
    payload = json.loads(stdout)
    assert payload["tariff_name"] == "custom"
    assert payload["arrival_day"] == "Mon"
    assert payload["price_profile_id"] == "prices"
    assert payload["intervals"]

    missing_day = tuple(value for value in common if value not in ("--arrival-day", "Mon"))
    code, stdout, stderr = _run(*missing_day)
    assert code == 2 and stdout == ""
    assert "arrival-day" in stderr


def test_interval_metadata_requires_json_output() -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "07:00",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--include-intervals",
    )
    assert (code, stdout) == (2, "")
    assert "requires --format json" in stderr


def test_custom_profile_missing_or_malformed_file_has_no_partial_output(tmp_path: Path) -> None:
    common = (
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "11:07",
        "--departure",
        "18:52",
        "--current-soc",
        "35",
        "--next-trip-km",
        "45",
        "--tariff",
        "custom",
        "--price-profile-format",
        "hour_of_week",
        "--arrival-day",
        "Mon",
        "--format",
        "json",
    )
    code, stdout, stderr = _run(*common, "--price-profile", str(tmp_path / "missing.csv"))
    assert (code, stdout) == (2, "")
    assert "cannot read price profile" in stderr

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("hour_of_week,price\n0,2\n", encoding="utf-8")
    code, stdout, stderr = _run(*common, "--price-profile", str(malformed))
    assert (code, stdout) == (2, "")
    assert "exactly 168" in stderr


def test_cli_rejects_irrelevant_custom_profile_day_and_date_options(tmp_path: Path) -> None:
    weekly_path = tmp_path / "weekly.csv"
    timestamped_path = tmp_path / "timestamped.csv"
    common = (
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "19:15",
        "--current-soc",
        "100",
        "--next-trip-km",
        "0",
        "--tariff",
        "custom",
    )
    code, stdout, stderr = _run(
        *common,
        "--price-profile",
        str(weekly_path),
        "--price-profile-format",
        "hour_of_week",
        "--arrival-day",
        "Mon",
        "--arrival-date",
        "2026-08-03",
    )
    assert code == 2 and stdout == ""
    assert "reject --arrival-date" in stderr

    code, stdout, stderr = _run(
        *common,
        "--price-profile",
        str(timestamped_path),
        "--price-profile-format",
        "timestamped",
        "--arrival-date",
        "2026-08-03",
        "--arrival-day",
        "Mon",
    )
    assert code == 2 and stdout == ""
    assert "reject --arrival-day" in stderr


@pytest.mark.parametrize("option", ["--arrival-day", "--arrival-date"])
def test_cli_rejects_day_and_date_for_built_in_tariffs(option: str) -> None:
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "19:15",
        "--current-soc",
        "100",
        "--next-trip-km",
        "0",
        option,
        "Mon" if option == "--arrival-day" else "2026-08-03",
    )
    assert code == 2 and stdout == ""
    assert "custom profile options" in stderr


def test_cli_malformed_utf8_is_stderr_only_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "bad-utf8.csv"
    path.write_bytes(b"hour_of_week,price\n0,\xff\n")
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "19:15",
        "--current-soc",
        "100",
        "--next-trip-km",
        "0",
        "--tariff",
        "custom",
        "--price-profile",
        str(path),
        "--price-profile-format",
        "hour_of_week",
        "--arrival-day",
        "Mon",
        "--format",
        "json",
    )
    assert code == 2
    assert stdout == ""
    assert str(path) in stderr
    assert "not valid UTF-8" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    ("profile_format", "profile_content", "extra_args"),
    [
        ("hour_of_week", "hour_of_week,price\n0,3.5,unexpected\n", ("--arrival-day", "Mon")),
        (
            "timestamped",
            "start_time,end_time,price\n2026-08-03 00:00,2026-08-03 01:00,3.5,extra\n",
            ("--arrival-date", "2026-08-03"),
        ),
    ],
)
def test_cli_trailing_columns_have_no_partial_output(
    tmp_path: Path, profile_format: str, profile_content: str, extra_args: tuple[str, ...]
) -> None:
    path = tmp_path / f"trailing-{profile_format}.csv"
    path.write_text(profile_content, encoding="utf-8")
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "19:15",
        "--current-soc",
        "100",
        "--next-trip-km",
        "0",
        "--tariff",
        "custom",
        "--price-profile",
        str(path),
        "--price-profile-format",
        profile_format,
        *extra_args,
        "--format",
        "json",
    )
    assert code == 2 and stdout == ""
    assert str(path) in stderr
    assert "unexpected values" in stderr
    assert "Traceback" not in stderr


def test_cli_blank_profile_has_no_partial_output(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text("\n  \n", encoding="utf-8")
    code, stdout, stderr = _run(
        "generate",
        "--ev-model",
        "generic_40kwh_lfp",
        "--arrival",
        "19:00",
        "--departure",
        "19:15",
        "--current-soc",
        "100",
        "--next-trip-km",
        "0",
        "--tariff",
        "custom",
        "--price-profile",
        str(path),
        "--price-profile-format",
        "hour_of_week",
        "--arrival-day",
        "Mon",
        "--format",
        "json",
    )
    assert code == 2 and stdout == ""
    assert str(path) in stderr
    assert "profile" in stderr


@pytest.mark.parametrize("label", ["Rs", "USD", "EUR", "currency"])
def test_text_output_uses_custom_currency_label(label: str) -> None:
    profile = WeeklyPriceProfile(
        "custom",
        (PricePeriod(0, 10080, 1.0),),
        currency_label=label,
    )
    menu = generate_ev_menu(
        ev_model="generic_40kwh_lfp",
        arrival_time="19:00",
        departure_time="19:15",
        current_soc=1.0,
        next_trip_distance_km=0.0,
        buffer_soc=0.0,
        tariff="custom",
        custom_price_profile=profile,
        arrival_day="Mon",
    )
    stream = StringIO()
    _render_text(menu, stream)
    rendered = stream.getvalue()
    assert f"Cost({label})" in rendered
    assert f"Saving({label})" in rendered
