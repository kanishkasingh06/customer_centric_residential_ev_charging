from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO

import pytest

from evmenu.cli import build_parser, main


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
