from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CORE5_FEATURE_COLUMNS = ("mss", "adx_quality", "irr", "er", "log_vr")
CORE5_DRA_FEATURE_COLUMNS = ("mss", "adx_quality", "irr", "er", "log_vr", "dra")
ADX_PARTS_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence",
    "adx_crossover",
    "irr",
    "er",
    "log_vr",
)
ADX_PARTS_MINUS_PERSISTENCE_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_crossover",
    "irr",
    "er",
    "log_vr",
)
ADX_PARTS_MINUS_CROSSOVER_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence",
    "irr",
    "er",
    "log_vr",
)
ADX_PARTS_MINUS_LOG_VR_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence",
    "adx_crossover",
    "irr",
    "er",
)
ADX_PARTS_MINUS_IRR_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence",
    "adx_crossover",
    "er",
    "log_vr",
)
ADX_PARTS_MINUS_PERSISTENCE_LOG_VR_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_crossover",
    "irr",
    "er",
)
ADX_PARTS_PERSISTENCE_MARGIN_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence_margin",
    "adx_crossover",
    "irr",
    "er",
    "log_vr",
)
ADX_PARTS_PERSISTENCE_CONTROL_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence_control",
    "adx_crossover",
    "irr",
    "er",
    "log_vr",
)
ADX_PARTS_PERSISTENCE_RECENCY_FEATURE_COLUMNS = (
    "mss",
    "adx_strength",
    "adx_persistence_recency",
    "adx_crossover",
    "irr",
    "er",
    "log_vr",
)

FEATURE_SETS = {
    "core5": CORE5_FEATURE_COLUMNS,
    "core5_dra": CORE5_DRA_FEATURE_COLUMNS,
    "adx_parts": ADX_PARTS_FEATURE_COLUMNS,
    "adx_parts_minus_persistence": ADX_PARTS_MINUS_PERSISTENCE_FEATURE_COLUMNS,
    "adx_parts_minus_crossover": ADX_PARTS_MINUS_CROSSOVER_FEATURE_COLUMNS,
    "adx_parts_minus_log_vr": ADX_PARTS_MINUS_LOG_VR_FEATURE_COLUMNS,
    "adx_parts_minus_irr": ADX_PARTS_MINUS_IRR_FEATURE_COLUMNS,
    "adx_parts_minus_persistence_log_vr": ADX_PARTS_MINUS_PERSISTENCE_LOG_VR_FEATURE_COLUMNS,
    "adx_parts_persistence_margin": ADX_PARTS_PERSISTENCE_MARGIN_FEATURE_COLUMNS,
    "adx_parts_persistence_control": ADX_PARTS_PERSISTENCE_CONTROL_FEATURE_COLUMNS,
    "adx_parts_persistence_recency": ADX_PARTS_PERSISTENCE_RECENCY_FEATURE_COLUMNS,
}

RIDGE_ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
ELASTIC_NET_ALPHAS = [0.03, 0.1, 0.3, 1.0]
ELASTIC_NET_L1_RATIOS = [0.1, 0.25, 0.5]

COVID_PRE_CUTOFF = pd.Timestamp("2020-02-28")
COVID_POST_START = pd.Timestamp("2020-07-01")
COVID_TRANSITION_START = pd.Timestamp("2020-03-01")
COVID_TRANSITION_END = pd.Timestamp("2020-06-30")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_group: str
    session_name: str
    era_name: str
    feature_set_name: str
    feature_columns: tuple[str, ...]
    model_name: str
    alpha: float
    l1_ratio: float | None = None


def _representation_specs(session_name: str, ridge_alpha: float) -> list[ExperimentSpec]:
    return [
        ExperimentSpec("EXP03_full_core5", "representation_sweep", session_name, "full_dev", "core5", CORE5_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP04_full_core5_dra", "representation_sweep", session_name, "full_dev", "core5_dra", CORE5_DRA_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP05_full_adx_parts", "representation_sweep", session_name, "full_dev", "adx_parts", ADX_PARTS_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP06_post_core5", "representation_sweep", session_name, "post_covid", "core5", CORE5_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP07_post_core5_dra", "representation_sweep", session_name, "post_covid", "core5_dra", CORE5_DRA_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP08_post_adx_parts", "representation_sweep", session_name, "post_covid", "adx_parts", ADX_PARTS_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP09_pre_core5", "representation_sweep", session_name, "pre_covid", "core5", CORE5_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP10_pre_core5_dra", "representation_sweep", session_name, "pre_covid", "core5_dra", CORE5_DRA_FEATURE_COLUMNS, "ridge", ridge_alpha),
        ExperimentSpec("EXP11_pre_adx_parts", "representation_sweep", session_name, "pre_covid", "adx_parts", ADX_PARTS_FEATURE_COLUMNS, "ridge", ridge_alpha),
    ]


def build_experiment_registry(session_name: str, ridge_alpha: float = 1.0) -> list[ExperimentSpec]:
    return _representation_specs(session_name=session_name, ridge_alpha=ridge_alpha)


def build_post_adx_ablation_registry(session_name: str, ridge_alpha: float = 1.0) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            "EXP20_post_adx_parts_base",
            "post_adx_ablation",
            session_name,
            "post_covid",
            "adx_parts",
            ADX_PARTS_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP21_post_adx_parts_minus_persistence",
            "post_adx_ablation",
            session_name,
            "post_covid",
            "adx_parts_minus_persistence",
            ADX_PARTS_MINUS_PERSISTENCE_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP22_post_adx_parts_minus_crossover",
            "post_adx_ablation",
            session_name,
            "post_covid",
            "adx_parts_minus_crossover",
            ADX_PARTS_MINUS_CROSSOVER_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP23_post_adx_parts_minus_log_vr",
            "post_adx_ablation",
            session_name,
            "post_covid",
            "adx_parts_minus_log_vr",
            ADX_PARTS_MINUS_LOG_VR_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP24_post_adx_parts_minus_irr",
            "post_adx_ablation",
            session_name,
            "post_covid",
            "adx_parts_minus_irr",
            ADX_PARTS_MINUS_IRR_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP25_post_adx_parts_minus_persistence_log_vr",
            "post_adx_ablation",
            session_name,
            "post_covid",
            "adx_parts_minus_persistence_log_vr",
            ADX_PARTS_MINUS_PERSISTENCE_LOG_VR_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
    ]


def build_post_adx_persistence_rewrite_registry(session_name: str, ridge_alpha: float = 1.0) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            "EXP30_post_adx_persistence_base",
            "post_adx_persistence_rewrites",
            session_name,
            "post_covid",
            "adx_parts",
            ADX_PARTS_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP31_post_adx_persistence_margin",
            "post_adx_persistence_rewrites",
            session_name,
            "post_covid",
            "adx_parts_persistence_margin",
            ADX_PARTS_PERSISTENCE_MARGIN_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP32_post_adx_persistence_control",
            "post_adx_persistence_rewrites",
            session_name,
            "post_covid",
            "adx_parts_persistence_control",
            ADX_PARTS_PERSISTENCE_CONTROL_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
        ExperimentSpec(
            "EXP33_post_adx_persistence_recency",
            "post_adx_persistence_rewrites",
            session_name,
            "post_covid",
            "adx_parts_persistence_recency",
            ADX_PARTS_PERSISTENCE_RECENCY_FEATURE_COLUMNS,
            "ridge",
            ridge_alpha,
        ),
    ]


def build_ridge_alpha_sweep(session_name: str) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for era_prefix, era_name, exp_prefix in (
        ("full", "full_dev", "EXP01"),
        ("post", "post_covid", "EXP02"),
    ):
        for alpha in RIDGE_ALPHAS:
            alpha_label = str(alpha).replace(".", "p")
            specs.append(
                ExperimentSpec(
                    experiment_id=f"{exp_prefix}_{era_prefix}_core5_alpha_{alpha_label}",
                    experiment_group="ridge_alpha_sweep",
                    session_name=session_name,
                    era_name=era_name,
                    feature_set_name="core5",
                    feature_columns=CORE5_FEATURE_COLUMNS,
                    model_name="ridge",
                    alpha=alpha,
                )
            )
    return specs


def filter_table_for_era(table: pd.DataFrame, era_name: str) -> pd.DataFrame:
    trade_dates = pd.to_datetime(table["trade_date"])
    transition_mask = (trade_dates >= COVID_TRANSITION_START) & (trade_dates <= COVID_TRANSITION_END)

    if era_name == "full_dev":
        mask = ~transition_mask
    elif era_name == "pre_covid":
        mask = trade_dates <= COVID_PRE_CUTOFF
    elif era_name == "post_covid":
        mask = trade_dates >= COVID_POST_START
    else:
        raise ValueError(f"Unknown era_name: {era_name}")

    return table.loc[mask].copy()
