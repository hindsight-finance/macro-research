from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

GLOBEX_1M_INPUT_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
MACRO_1M_INPUT_PATH = Path("outputs/nq_macro_volume_delta_1m.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_delta_reversal.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_delta_reversal_summary.parquet")

GLOBEX_REQUIRED_COLUMNS = {
    "trade_date_et",
    "session_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}
MACRO_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}

PREDICTORS = [
    "eth_pre_rth",
    "rth_pre_macro",
    "day_pre_macro",
    "macro_pre59",
    "rth_plus_macro_pre59",
    "day_plus_macro_pre59",
]


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_inputs(globex_1m: pl.DataFrame, macro_1m: pl.DataFrame) -> None:
    globex_missing = _missing_columns(globex_1m, GLOBEX_REQUIRED_COLUMNS)
    if globex_missing:
        raise ValueError(f"Missing Globex volume-delta columns: {globex_missing}")
    macro_missing = _missing_columns(macro_1m, MACRO_REQUIRED_COLUMNS)
    if macro_missing:
        raise ValueError(f"Missing macro volume-delta columns: {macro_missing}")


def build_macro_delta_reversal(globex_1m: pl.DataFrame, macro_1m: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_1m)
    return pl.DataFrame()


def summarize_macro_delta_reversal(study: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame()


def load_volume_delta_inputs(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(globex_path), pl.read_parquet(macro_path)


def write_macro_delta_reversal(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    globex_1m, macro_1m = load_volume_delta_inputs(globex_path, macro_path)
    study = build_macro_delta_reversal(globex_1m, macro_1m)
    summary = summarize_macro_delta_reversal(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not GLOBEX_1M_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {GLOBEX_1M_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not MACRO_1M_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_1M_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_delta_reversal()
    print(f"[OK] Wrote macro delta reversal → {output}")
    print(f"[OK] Wrote macro delta reversal summary → {summary_output}")


if __name__ == "__main__":
    main()
