#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from features.trend.modeling import build_modeling_table
from features.trend.modeling.labels import DEFAULT_LABEL_THRESHOLDS, assign_three_scalar_labels
from features.trend.modeling.table import DEFAULT_SESSION_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build historical regime scores and labels")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--session-name",
        dest="session_names",
        action="append",
        choices=DEFAULT_SESSION_NAMES,
        help="Limit the output to one or more session windows.",
    )
    parser.add_argument("--trend-high", type=float, default=DEFAULT_LABEL_THRESHOLDS["trend_high"])
    parser.add_argument("--containment-high", type=float, default=DEFAULT_LABEL_THRESHOLDS["containment_high"])
    parser.add_argument("--chop-high", type=float, default=DEFAULT_LABEL_THRESHOLDS["chop_high"])
    parser.add_argument("--low-cutoff", type=float, default=DEFAULT_LABEL_THRESHOLDS["low_cutoff"])
    parser.add_argument(
        "--containment-chop-max",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS["containment_chop_max"],
    )
    return parser


def _write_output(frame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
        return
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
        return
    raise ValueError("output-path must end with .parquet or .csv")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table = build_modeling_table(
        input_path=args.input_path,
        instrument=args.instrument,
        session_names=args.session_names or DEFAULT_SESSION_NAMES,
    )
    labeled = assign_three_scalar_labels(
        frame=table,
        trend_high=args.trend_high,
        containment_high=args.containment_high,
        chop_high=args.chop_high,
        low_cutoff=args.low_cutoff,
        containment_chop_max=args.containment_chop_max,
    )
    _write_output(labeled, args.output_path)
    print(f"Wrote {len(labeled)} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
