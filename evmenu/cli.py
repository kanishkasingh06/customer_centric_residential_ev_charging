"""Command-line interface for deterministic single-EV menu generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from math import isfinite
from typing import Any, NoReturn, TextIO

from .assembly import (
    DisplayDiversityParameters,
    MenuAssemblySettings,
    PairwiseDistinctness,
)
from .catalog import list_ev_models
from .exceptions import EVMenuError
from .pricing import load_price_profile_csv
from .service import CustomerMenuRow, GeneratedCustomerMenu, generate_ev_menu


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


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
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
        "--battery-temperature-c",
        "--temperature-c",
        dest="temperature_c",
        type=_finite_float("temperature-c", minimum=-273.149999),
        default=25.0,
        help="constant battery temperature in degrees Celsius (default: 25)",
    )
    generate.add_argument(
        "--battery-age-years",
        type=_finite_float("battery-age-years", minimum=1e-12),
        default=1.0,
        help="battery age used by the local power-law slope (default: 1)",
    )
    generate.add_argument(
        "--accumulated-fec",
        type=_finite_float("accumulated-fec", minimum=0.0),
        default=0.0,
        help="accumulated equivalent full cycles at session start (default: 0)",
    )
    generate.add_argument(
        "--daytime-parked-hours",
        type=_finite_float("daytime-parked-hours", minimum=0.0),
        default=None,
        help="optional post-commute parked duration for calendar aging",
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
        help="legacy alias for --max-displayed-offers",
    )
    generate.add_argument(
        "--max-displayed-offers",
        "--maximum-displayed-offers",
        dest="max_displayed_offers",
        type=_positive_int,
        default=None,
        help="maximum non-BAU customer-facing options (default: 12)",
    )
    generate.add_argument(
        "--max-offers-per-target",
        type=_positive_int,
        default=None,
        help="generous safety cap per target; global diversity selects the mix (default: 12)",
    )
    generate.add_argument(
        "--min-ready-separation-minutes",
        type=_nonnegative_int,
        default=None,
        help="minimum same-target displayed readiness separation (default: 120)",
    )
    generate.add_argument(
        "--min-saving-difference",
        type=_finite_float("min-saving-difference", minimum=0.0),
        default=None,
        help="absolute saving threshold for close-ready exceptions (default: 5)",
    )
    generate.add_argument(
        "--min-saving-fraction-of-bau",
        type=_finite_float("min-saving-fraction-of-bau", minimum=0.0),
        default=None,
        help="BAU-relative saving threshold for close-ready exceptions (default: 0.05)",
    )
    generate.add_argument(
        "--min-relative-stress-difference",
        type=_finite_float("min-relative-stress-difference", minimum=0.0),
        default=None,
        help="relative raw-stress threshold for close-ready exceptions (default: 0.02)",
    )
    generate.add_argument(
        "--menu-stage",
        choices=("generated", "compacted", "pareto", "displayed"),
        default="displayed",
        help="pipeline stage to return (default: diversity-selected displayed stage)",
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
        "--include-diagnostics",
        action="store_true",
        help="include request, optimization, target, and failure diagnostics in JSON output",
    )
    generate.add_argument(
        "--include-pipeline-stages",
        action="store_true",
        help="include generated, retained, compacted, Pareto, and BAU-anchor arrays in JSON",
    )
    generate.add_argument(
        "--exclude-bau-from-display",
        action="store_true",
        help="explicitly request the default behavior: BAU rows are references, not choices",
    )
    generate.add_argument(
        "--include-bau-references",
        action="store_true",
        help="include separate BAU reference rows in JSON output",
    )
    generate.add_argument(
        "--include-distinctness-diagnostics",
        action="store_true",
        help="include feature scaling, pairwise spread, and nearest-neighbour diagnostics",
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
    include_diagnostics: bool = False,
    include_pipeline_stages: bool = False,
    include_bau_references: bool = False,
    include_distinctness_diagnostics: bool = False,
) -> dict[str, object]:
    def row_payload(row: CustomerMenuRow) -> dict[str, object]:
        payload: dict[str, object] = {
            "offer_id": row.offer_id,
            "scientific_id": row.offer_id,
            "ready_time": row.ready_time,
            "ready_absolute_minute": row.ready_boundary_absolute_minute,
            "target_soc_percent": row.target_soc_percent,
            "target_soc": row.target_soc_percent / 100.0,
            "target_battery_energy_kwh": row.target_battery_energy_kwh,
            "target_energy_kwh": row.target_battery_energy_kwh,
            "charging_cost": row.charging_cost,
            "saving": row.saving,
            "requested_saving": row.requested_saving,
            "actual_saving": row.actual_saving,
            "actual_cost": row.actual_cost,
            "battery_stress": row.raw_battery_stress,
            "same_target_bau_cost": row.same_target_bau_cost,
            "raw_battery_stress": row.raw_battery_stress,
            "estimated_capacity_loss": row.estimated_capacity_loss,
            "estimated_rul_years": row.estimated_rul_years,
            "saving_band_lower": row.saving_band_lower,
            "saving_band_upper": row.saving_band_upper,
            "saving_band_violation": row.saving_band_violation,
            "battery_metric_model_id": row.battery_metric_model_id,
            "battery_metric_comparison_scope": row.battery_metric_comparison_scope,
            "degradation_model_id": row.degradation_model_id,
            "degradation_model_version": row.degradation_model_version,
            "parameter_set_id": row.parameter_set_id,
            "parameter_status": row.parameter_status,
            "battery_chemistry": row.battery_chemistry,
            "calendar_capacity_fade": row.calendar_capacity_fade,
            "cycle_capacity_fade": row.cycle_capacity_fade,
            "total_capacity_fade": row.total_capacity_fade,
            "capacity_fade_percent": row.capacity_fade_percent,
            "calendar_fade_connected_window": row.calendar_fade_connected_window,
            "calendar_fade_parked_period": row.calendar_fade_parked_period,
            "battery_temperature_c": row.battery_temperature_c,
            "battery_age_years": row.battery_age_years,
            "accumulated_fec_at_start": row.accumulated_fec_at_start,
            "effective_fec_at_start": row.effective_fec_at_start,
            "minimum_effective_fec": row.minimum_effective_fec,
            "fec_regularization_applied": row.fec_regularization_applied,
            "fec_regularization_source": row.fec_regularization_source,
            "parked_period_hours": row.parked_period_hours,
            "parked_period_source": row.parked_period_source,
            "calendar_scope": row.calendar_scope,
            "temperature_source": row.temperature_source,
            "battery_age_source": row.battery_age_source,
            "accumulated_fec_source": row.accumulated_fec_source,
            "energy_drawn_kwh": row.energy_drawn_kwh,
            "role": row.role,
            "is_bau": row.role == "bau" or "is_bau" in row.provenance_flags,
            "provenance_flags": list(row.provenance_flags),
            "ready_boundary_absolute_minute": row.ready_boundary_absolute_minute,
            "is_pareto_efficient": row.is_pareto_efficient,
            "is_preserved_bau_anchor": row.is_preserved_bau_anchor,
            "is_selected_bau_anchor": row.is_selected_bau_anchor,
            "is_selected_maximum_saving": row.is_selected_maximum_saving,
            "is_selected_least_degradation": row.is_selected_least_degradation,
            "is_selected_intermediate": row.is_selected_intermediate,
            "selection_reason": row.selection_reason,
            "selection_rank_within_target": row.selection_rank_within_target,
            "diversity_score": row.diversity_score,
            "selected_as": row.selection_reason,
        }
        if include_schedule:
            payload["charging_schedule_kw"] = list(row.charging_schedule_kw)
        distinctness = menu.pipeline_diagnostics.distinctness_diagnostics
        if distinctness is not None:
            metric = next(
                (item for item in distinctness.option_metrics if item.offer_id == row.offer_id),
                None,
            )
            if metric is not None:
                payload.update(asdict(metric))
        return payload

    returned_rows = menu.offers
    displayed_rows = menu.displayed_offers
    customer_visible_rows = _display_rows(menu, menu.displayed_offers)
    offers = [row_payload(row) for row in returned_rows]
    pipeline = menu.pipeline_diagnostics
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
        "displayed_offers": [row_payload(row) for row in displayed_rows],
        "menu_stage": menu.menu_stage,
        "raw_offer_count": menu.raw_offer_count,
        "exact_duplicates_removed": menu.scientific_duplicate_count,
        "generated_offer_count": menu.generated_offer_count,
        "offers_returned": len(offers),
        "customer_visible_duplicates_removed": max(
            0, menu.generated_offer_count - len(customer_visible_rows)
        ),
        "pipeline_counts": {
            "generated_offer_count": pipeline.generated_offer_count,
            "retained_offer_count": pipeline.retained_offer_count,
            "compacted_offer_count": pipeline.compacted_offer_count,
            "pareto_efficient_offer_count": pipeline.pareto_efficient_offer_count,
            "pareto_offer_count": pipeline.pareto_offer_count,
            "preserved_bau_anchor_count": pipeline.preserved_bau_anchor_count,
            "diversity_selected_offer_count": pipeline.displayed_offer_count,
            "displayed_offer_count": pipeline.displayed_offer_count,
            "nonpositive_removed_count": pipeline.nonpositive_removed_count,
            "compaction_removed_count": pipeline.compaction_removed_count,
            "pareto_dominated_removed_count": pipeline.pareto_dominated_removed_count,
            "display_selection_removed_count": pipeline.display_selection_removed_count,
            "bau_reference_offer_count": len(menu.bau_references),
        },
    }
    if pipeline.display_selection_summary is not None:
        summary = pipeline.display_selection_summary
        payload["display_selection_summary"] = {
            "pareto_input_count": summary.pareto_input_count,
            "target_count": summary.target_count,
            "bau_anchor_count": summary.bau_anchor_count,
            "positive_target_coverage_count": summary.positive_target_coverage_count,
            "selected_count": summary.selected_count,
            "removed_count": summary.removed_count,
            "minimum_ready_separation_minutes": summary.minimum_ready_separation_minutes,
            "maximum_offers_per_target": summary.maximum_offers_per_target,
            "maximum_displayed_offers": summary.maximum_displayed_offers,
            "displayed_options_are_non_bau": True,
        }
    if include_bau_references or include_pipeline_stages:
        payload["bau_reference_offers"] = [row_payload(row) for row in menu.bau_references]
    if include_distinctness_diagnostics or include_diagnostics:
        distinctness = pipeline.distinctness_diagnostics
        if distinctness is not None:

            def pair_payload(pair: PairwiseDistinctness) -> dict[str, object]:
                return {
                    "offer_a": pair.offer_a,
                    "offer_b": pair.offer_b,
                    "ready_distance": pair.ready_distance,
                    "target_distance": pair.target_distance,
                    "saving_distance": pair.saving_distance,
                    "fade_distance": pair.fade_distance,
                    "overall_distance": pair.overall_distance,
                }

            payload["distinctness_diagnostics"] = {
                "feature_definitions": dict(distinctness.feature_definitions),
                "pairwise_metrics": [pair_payload(pair) for pair in distinctness.pairwise_metrics],
                "menu_metrics": {
                    "minimum_pairwise_distance": distinctness.minimum_pairwise_distance,
                    "mean_pairwise_distance": distinctness.mean_pairwise_distance,
                    "median_pairwise_distance": distinctness.median_pairwise_distance,
                    "mean_nearest_neighbour_distance": distinctness.mean_nearest_neighbour_distance,
                    "minimum_nearest_neighbour_distance": distinctness.minimum_nearest_neighbour_distance,
                    "ready_range": distinctness.ready_range,
                    "target_range": distinctness.target_range,
                    "saving_range": distinctness.saving_range,
                    "fade_range": distinctness.fade_range,
                    "offers_per_target": dict(distinctness.offers_per_target),
                    "target_coverage_count": distinctness.target_coverage_count,
                    "target_share_entropy": distinctness.target_share_entropy,
                    "near_duplicate_pair_count": distinctness.near_duplicate_pair_count,
                    "warning_threshold": distinctness.warning_threshold,
                },
                "closest_pairs": [pair_payload(pair) for pair in distinctness.closest_pairs],
                "most_distinct_pairs": [
                    pair_payload(pair) for pair in distinctness.most_distinct_pairs
                ],
                "option_metrics": [asdict(item) for item in distinctness.option_metrics],
            }
    if include_pipeline_stages:
        payload["generated_offers"] = [row_payload(row) for row in menu.generated_offers]
        payload["retained_offers"] = [row_payload(row) for row in menu.retained_offers]
        payload["compacted_offers"] = [row_payload(row) for row in menu.compacted_offers]
        payload["pareto_efficient_offers"] = [
            row_payload(row) for row in menu.pareto_efficient_offers
        ]
        payload["preserved_bau_anchors"] = [row_payload(row) for row in menu.preserved_bau_anchors]
        payload["pareto_offers"] = [row_payload(row) for row in menu.pareto_offers]
        payload["displayed_offers"] = [row_payload(row) for row in displayed_rows]
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
    if include_diagnostics:
        request = menu.request_diagnostics
        optimization = menu.optimization_diagnostics
        payload["diagnostics"] = {
            "request_count_total": request.request_count_total,
            "request_count_feasible": request.request_count_feasible,
            "request_count_positive_saving": request.request_count_positive_saving,
            "request_count_no_saving": request.request_count_no_saving,
            "request_count_infeasible": request.request_count_infeasible,
            "optimization_attempt_count": optimization.optimization_attempt_count,
            "optimization_success_count": optimization.optimization_success_count,
            "optimization_infeasible_count": optimization.optimization_infeasible_count,
            "optimization_validation_failure_count": optimization.optimization_validation_failure_count,
            "optimization_solver_failure_count": optimization.optimization_solver_failure_count,
            "degradation_objective": (
                {
                    "reporting_model_id": menu.degradation_objective_diagnostics.reporting_model_id,
                    "optimization_model_id": menu.degradation_objective_diagnostics.optimization_model_id,
                    "exact_reporting_model_used_in_objective": menu.degradation_objective_diagnostics.exact_reporting_model_used_in_objective,
                    "objective_convex": menu.degradation_objective_diagnostics.objective_convex,
                    "constant_terms_excluded_from_objective": menu.degradation_objective_diagnostics.constant_terms_excluded_from_objective,
                }
                if menu.degradation_objective_diagnostics is not None
                else None
            ),
            "target_summaries": [
                {
                    "target_soc": summary.target_soc,
                    "bau_ready": summary.bau_ready,
                    "request_count": summary.request_count,
                    "positive_saving_request_count": summary.positive_saving_request_count,
                    "selected_saving_level_count": summary.selected_saving_level_count,
                    "optimization_success_count": summary.optimization_success_count,
                    "generated_offer_count": summary.generated_offer_count,
                    "duplicate_count": summary.duplicate_count,
                    "roles_present": list(summary.roles_present),
                }
                for summary in menu.target_summaries
            ],
            "saving_level_failures": [
                {
                    "ready_step": failure.ready_step,
                    "target_soc": failure.target_soc,
                    "requested_saving": failure.requested_saving,
                    "reason": failure.reason,
                }
                for failure in menu.saving_level_failures
            ],
            "degradation": (
                {
                    "chemistry": menu.ev_model.chemistry,
                    "degradation_model_id": menu.generated_offers[0].degradation_model_id,
                    "degradation_model_version": menu.generated_offers[0].degradation_model_version,
                    "parameter_set_id": menu.generated_offers[0].parameter_set_id,
                    "parameter_status": menu.generated_offers[0].parameter_status,
                    "battery_temperature_c": menu.generated_offers[0].battery_temperature_c,
                    "battery_age_years": menu.generated_offers[0].battery_age_years,
                    "accumulated_fec_at_start": menu.generated_offers[0].accumulated_fec_at_start,
                    "effective_fec_at_start": menu.generated_offers[0].effective_fec_at_start,
                    "minimum_effective_fec": menu.generated_offers[0].minimum_effective_fec,
                    "fec_regularization_applied": menu.generated_offers[
                        0
                    ].fec_regularization_applied,
                    "fec_regularization_source": menu.generated_offers[0].fec_regularization_source,
                    "parked_period_hours": menu.generated_offers[0].parked_period_hours,
                    "parked_period_source": menu.generated_offers[0].parked_period_source,
                    "calendar_scope": menu.generated_offers[0].calendar_scope,
                }
                if menu.generated_offers
                else None
            ),
            "compaction_groups": [
                {
                    "ready_absolute_minute": group.ready_absolute_minute,
                    "target_soc": group.target_soc,
                    "input_offer_count": group.input_offer_count,
                    "output_offer_count": group.output_offer_count,
                    "saving_min": group.saving_min,
                    "saving_max": group.saving_max,
                    "representative_offer_ids": list(group.representative_offer_ids),
                    "removed_offer_ids": list(group.removed_offer_ids),
                }
                for group in menu.pipeline_diagnostics.compaction_groups
            ],
            "compaction_decisions": [
                {
                    "retained_offer_id": decision.retained_offer_id,
                    "removed_offer_ids": list(decision.removed_offer_ids),
                    "saving_interval_lower": decision.saving_interval_lower,
                    "saving_interval_upper": decision.saving_interval_upper,
                    "reason": decision.reason,
                }
                for decision in menu.pipeline_diagnostics.compaction_decisions
            ],
            "pareto_dominated_offer_ids": list(
                menu.pipeline_diagnostics.pareto_dominated_offer_ids
            ),
            "pareto_dominated_bau_anchor_ids": list(
                menu.pipeline_diagnostics.pareto_dominated_bau_anchor_ids
            ),
            "pareto_dominance_pairs": [
                list(pair) for pair in menu.pipeline_diagnostics.pareto_dominance_pairs
            ],
            "pipeline_target_summaries": [
                {
                    "target_soc": summary.target_soc,
                    "generated_offer_count": summary.generated_offer_count,
                    "retained_offer_count": summary.retained_offer_count,
                    "compacted_offer_count": summary.compacted_offer_count,
                    "pareto_efficient_offer_count": summary.pareto_efficient_offer_count,
                    "preserved_bau_anchor_count": summary.preserved_bau_anchor_count,
                    "pareto_offer_count": summary.pareto_offer_count,
                    "displayed_offer_count": summary.displayed_offer_count,
                    "roles_present": list(summary.roles_present),
                    "generated_roles": list(summary.generated_roles),
                    "retained_roles": list(summary.retained_roles),
                    "compacted_roles": list(summary.compacted_roles),
                    "pareto_efficient_roles": list(summary.pareto_efficient_roles),
                    "preserved_bau_anchor_roles": list(summary.preserved_bau_anchor_roles),
                    "pareto_roles": list(summary.pareto_roles),
                    "displayed_roles": list(summary.displayed_roles),
                }
                for summary in menu.pipeline_diagnostics.target_summaries
            ],
            "target_display_summaries": [
                {
                    "target_soc": summary.target_soc,
                    "target_energy_kwh": summary.target_energy_kwh,
                    "input_pareto_count": summary.input_pareto_count,
                    "bau_count": summary.bau_count,
                    "selected_count": summary.selected_count,
                    "selected_offer_ids": list(summary.selected_offer_ids),
                    "selected_ready_times": list(summary.selected_ready_times),
                    "roles_present": list(summary.roles_present),
                    "minimum_pairwise_ready_separation": summary.minimum_pairwise_ready_separation,
                }
                for summary in menu.pipeline_diagnostics.target_display_summaries
            ],
            "display_selection_decisions": [
                {
                    "offer_id": decision.offer_id,
                    "selected": decision.selected,
                    "reason": decision.reason,
                    "nearest_selected_offer_id": decision.nearest_selected_offer_id,
                    "ready_difference_minutes": decision.ready_difference_minutes,
                    "saving_difference": decision.saving_difference,
                    "relative_stress_difference": decision.relative_stress_difference,
                    "diversity_score": decision.diversity_score,
                }
                for decision in menu.pipeline_diagnostics.display_selection_decisions
            ],
        }
    return payload


def _render_text(menu: GeneratedCustomerMenu, stream: TextIO) -> None:
    print(f"EV: {menu.ev_model.display_name} ({menu.ev_model.model_id})", file=stream)
    print(
        "Assumptions: built-in EV model values are illustrative research assumptions.",
        file=stream,
    )
    if menu.generated_offer_count > 100:
        print(
            f"Warning: Generated {menu.generated_offer_count} rich offers. "
            "Use --max-displayed-offers to limit the final customer menu.",
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
    rendered_rows = menu.offers if menu.menu_stage != "displayed" else _display_rows(menu)
    print(
        f"Menu stage: {menu.menu_stage} | Offers returned: {len(rendered_rows)}",
        file=stream,
    )
    print(
        f"Customer options: {len(menu.displayed_offers)} non-BAU | "
        f"BAU references: {len(menu.bau_references)} (not counted)",
        file=stream,
    )
    pipeline = menu.pipeline_diagnostics
    print(
        "Menu pipeline: "
        f"Generated: {pipeline.generated_offer_count} | "
        f"Retained: {pipeline.retained_offer_count} | "
        f"Compacted: {pipeline.compacted_offer_count} | "
        f"Pareto efficient: {pipeline.pareto_efficient_offer_count} | "
        f"BAU anchors preserved: {pipeline.preserved_bau_anchor_count} | "
        f"Pareto stage: {pipeline.pareto_offer_count} | "
        f"Diversity selected: {pipeline.displayed_offer_count} | "
        f"Displayed non-BAU: {pipeline.displayed_offer_count}",
        file=stream,
    )
    print(
        "TotalFade: estimated incremental capacity loss fraction; lower is better. "
        "BatteryStress is a deprecated compatibility alias. BAU anchors may be retained "
        "even when mathematically dominated.",
        file=stream,
    )
    if menu.generated_offers:
        reference = menu.generated_offers[0]
        print(
            "Degradation provenance: "
            f"effective FEC {reference.effective_fec_at_start:g} "
            f"(start {reference.accumulated_fec_at_start:g}, "
            f"floor {reference.minimum_effective_fec:g}, "
            f"regularized={reference.fec_regularization_applied}); "
            f"parked period {reference.parked_period_hours!r} h "
            f"({reference.parked_period_source}).",
            file=stream,
        )
    print(
        f"#  Ready  Target   Cost({menu.currency_label})  Saving({menu.currency_label})  "
        "TotalFade (lower is better)  Role  SelectedAs",
        file=stream,
    )
    for index, row in enumerate(rendered_rows, start=1):
        print(
            f"{index:<2} {row.ready_time:<6} "
            f"{row.target_soc_percent:>6.1f}% "
            f"{row.charging_cost:>10.2f} "
            f"{row.saving:>10.2f} "
            f"{float(row.raw_battery_stress or 0.0):>14.8g}  {row.role:<18} "
            f"{row.selection_reason or '-'}",
            file=stream,
        )


def _display_rows(
    menu: GeneratedCustomerMenu,
    rows: tuple[CustomerMenuRow, ...] | None = None,
) -> tuple[CustomerMenuRow, ...]:
    """Collapse rows indistinguishable after customer-visible rounding."""
    groups: dict[tuple[object, ...], list[CustomerMenuRow]] = {}
    source_rows = menu.offers if rows is None else rows
    for row in source_rows:
        key = (
            row.ready_time,
            round(row.target_soc_percent, 1),
            round(row.charging_cost, 2),
            round(row.saving, 2),
            round(float(row.raw_battery_stress or 0.0), 8),
            row.role,
        )
        groups.setdefault(key, []).append(row)
    representatives = [
        min(
            rows,
            key=lambda row: (
                float(row.raw_battery_stress or 0.0),
                row.charging_cost,
                row.charging_schedule_kw,
                row.offer_id,
            ),
        )
        for rows in groups.values()
    ]
    return tuple(
        sorted(
            representatives,
            key=lambda row: (
                row.ready_boundary_absolute_minute or 0,
                row.target_soc_percent,
                row.actual_saving if row.actual_saving is not None else row.saving,
                row.role,
                row.offer_id,
            ),
        )
    )


def _run_generate(
    namespace: argparse.Namespace,
    *,
    stdout: TextIO,
) -> None:
    if namespace.display_cap is not None and namespace.max_displayed_offers is not None:
        raise EVMenuError("--display-cap and --max-displayed-offers are mutually exclusive")
    diversity = DisplayDiversityParameters()
    diversity_updates = {
        name: value
        for name, value in (
            ("maximum_displayed_offers", namespace.max_displayed_offers),
            ("maximum_offers_per_target", namespace.max_offers_per_target),
            ("minimum_ready_separation_minutes", namespace.min_ready_separation_minutes),
            ("minimum_saving_difference", namespace.min_saving_difference),
            ("minimum_saving_fraction_of_bau", namespace.min_saving_fraction_of_bau),
            ("minimum_relative_stress_difference", namespace.min_relative_stress_difference),
        )
        if value is not None
    }
    diversity = replace(diversity, **diversity_updates)
    assembly_settings = MenuAssemblySettings(
        display_cap=namespace.display_cap,
        display_diversity=diversity,
        menu_stage=namespace.menu_stage,
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
        battery_age_years=namespace.battery_age_years,
        accumulated_equivalent_full_cycles=namespace.accumulated_fec,
        daytime_parked_hours=namespace.daytime_parked_hours,
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
                include_diagnostics=namespace.include_diagnostics,
                include_pipeline_stages=namespace.include_pipeline_stages,
                include_bau_references=namespace.include_bau_references,
                include_distinctness_diagnostics=namespace.include_distinctness_diagnostics,
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
            if namespace.include_pipeline_stages and namespace.format != "json":
                print(
                    "evmenu: error: --include-pipeline-stages requires --format json",
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
