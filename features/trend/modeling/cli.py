from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from features.trend.modeling.registry import (
    build_experiment_registry,
    build_post_adx_ablation_registry,
    build_post_adx_persistence_rewrite_registry,
    build_ridge_alpha_sweep,
    filter_table_for_era,
)
from features.trend.modeling.table import DEFAULT_SESSION_NAMES, build_modeling_table, write_modeling_table_cache
from features.trend.modeling.walkforward import run_walkforward_experiment, summarize_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trend modeling table builder and walk-forward harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_table_parser = subparsers.add_parser("build-table", help="Build and cache the canonical modeling table")
    build_table_parser.add_argument("--input-path", required=True)
    build_table_parser.add_argument("--instrument", required=True)
    build_table_parser.add_argument("--output-path", required=True)
    build_table_parser.add_argument(
        "--session-name",
        dest="session_names",
        action="append",
        choices=DEFAULT_SESSION_NAMES,
        help="Limit the table to one or more specific session windows.",
    )

    run_parser = subparsers.add_parser("run-experiments", help="Run the configured walk-forward experiment group")
    run_parser.add_argument("--table-path", required=True)
    run_parser.add_argument("--session-name", required=True, choices=DEFAULT_SESSION_NAMES)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument(
        "--experiment-group",
        default="representation_sweep",
        choices=("representation_sweep", "ridge_alpha_sweep", "post_adx_ablation", "post_adx_persistence_rewrites"),
    )
    run_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    run_parser.add_argument("--target-column", default="descriptive_target")
    run_parser.add_argument("--holdout-fraction", type=float, default=0.15)
    run_parser.add_argument("--train-months", type=int, default=24)
    run_parser.add_argument("--valid-months", type=int, default=3)
    run_parser.add_argument("--step-months", type=int, default=3)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize saved experiment outputs")
    summarize_parser.add_argument("--experiments-dir", required=True)
    summarize_parser.add_argument("--output-path")

    return parser


def _build_table_command(args: argparse.Namespace) -> int:
    table = build_modeling_table(
        input_path=args.input_path,
        instrument=args.instrument,
        session_names=args.session_names or DEFAULT_SESSION_NAMES,
    )
    write_modeling_table_cache(table, args.output_path)
    print(f"Wrote {len(table)} rows to {args.output_path}")
    return 0


def _select_experiment_specs(args: argparse.Namespace):
    if args.experiment_group == "ridge_alpha_sweep":
        return build_ridge_alpha_sweep(session_name=args.session_name)
    if args.experiment_group == "post_adx_ablation":
        return build_post_adx_ablation_registry(session_name=args.session_name, ridge_alpha=args.ridge_alpha)
    if args.experiment_group == "post_adx_persistence_rewrites":
        return build_post_adx_persistence_rewrite_registry(session_name=args.session_name, ridge_alpha=args.ridge_alpha)
    return build_experiment_registry(session_name=args.session_name, ridge_alpha=args.ridge_alpha)


def _run_experiments_command(args: argparse.Namespace) -> int:
    table = pd.read_parquet(args.table_path)
    specs = _select_experiment_specs(args)

    for spec in specs:
        filtered = filter_table_for_era(table, spec.era_name)
        run_walkforward_experiment(
            table=filtered,
            model_spec=spec,
            output_root=args.output_dir,
            target_column=args.target_column,
            holdout_fraction=args.holdout_fraction,
            train_months=args.train_months,
            valid_months=args.valid_months,
            step_months=args.step_months,
        )

    summary = summarize_experiments(Path(args.output_dir) / args.session_name)
    if summary.empty:
        print("No experiment outputs found.")
    else:
        print(summary.to_string(index=False))
    return 0


def _summarize_command(args: argparse.Namespace) -> int:
    summary = summarize_experiments(args.experiments_dir)
    if summary.empty:
        print("No experiment outputs found.")
        return 0

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".parquet":
            summary.to_parquet(output_path, index=False)
        else:
            summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-table":
        return _build_table_command(args)
    if args.command == "run-experiments":
        return _run_experiments_command(args)
    if args.command == "summarize":
        return _summarize_command(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
