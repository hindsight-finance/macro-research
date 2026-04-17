from __future__ import annotations

from features.trend.modeling.cli import build_parser


def test_build_parser_accepts_modeling_commands():
    parser = build_parser()

    build_args = parser.parse_args(
        [
            "build-table",
            "--input-path",
            "bars.parquet",
            "--instrument",
            "NQ",
            "--output-path",
            "table.parquet",
        ]
    )
    run_args = parser.parse_args(
        [
            "run-experiments",
            "--table-path",
            "table.parquet",
            "--session-name",
            "1pm-3pm",
            "--output-dir",
            "outputs/trend_modeling/experiments",
        ]
    )
    summary_args = parser.parse_args(
        [
            "summarize",
            "--experiments-dir",
            "outputs/trend_modeling/experiments/1pm-3pm",
        ]
    )

    assert build_args.command == "build-table"
    assert run_args.command == "run-experiments"
    assert summary_args.command == "summarize"


def test_build_parser_accepts_post_adx_ablation_group():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-experiments",
            "--table-path",
            "table.parquet",
            "--session-name",
            "1pm-3pm",
            "--output-dir",
            "outputs/trend_modeling/experiments",
            "--experiment-group",
            "post_adx_ablation",
        ]
    )

    assert args.experiment_group == "post_adx_ablation"


def test_build_parser_accepts_post_adx_persistence_rewrite_group():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-experiments",
            "--table-path",
            "table.parquet",
            "--session-name",
            "1pm-3pm",
            "--output-dir",
            "outputs/trend_modeling/experiments",
            "--experiment-group",
            "post_adx_persistence_rewrites",
        ]
    )

    assert args.experiment_group == "post_adx_persistence_rewrites"


def test_build_parser_accepts_containment_v2_group():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-experiments",
            "--table-path",
            "table.parquet",
            "--session-name",
            "1pm-3pm",
            "--output-dir",
            "outputs/trend_modeling/experiments",
            "--experiment-group",
            "containment_v2",
        ]
    )

    assert args.experiment_group == "containment_v2"


def test_build_parser_accepts_containment_research_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "containment-research",
            "--table-path",
            "table.parquet",
            "--output-dir",
            "outputs/trend_modeling/research",
        ]
    )

    assert args.command == "containment-research"
