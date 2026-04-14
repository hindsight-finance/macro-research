from __future__ import annotations

from datetime import date

import pandas as pd

from features.trend.modeling.registry import (
    build_experiment_registry,
    build_post_adx_ablation_registry,
    build_post_adx_persistence_rewrite_registry,
    filter_table_for_era,
)


def test_build_experiment_registry_contains_core_variants():
    registry = build_experiment_registry(session_name="1pm-3pm")
    ids = {experiment.experiment_id for experiment in registry}

    assert "EXP03_full_core5" in ids
    assert "EXP04_full_core5_dra" in ids
    assert "EXP05_full_adx_parts" in ids
    assert "EXP06_post_core5" in ids
    assert "EXP09_pre_core5" in ids


def test_build_post_adx_ablation_registry_contains_reduced_variants():
    registry = build_post_adx_ablation_registry(session_name="1pm-3pm")
    ids = {experiment.experiment_id for experiment in registry}

    assert "EXP20_post_adx_parts_base" in ids
    assert "EXP21_post_adx_parts_minus_persistence" in ids
    assert "EXP22_post_adx_parts_minus_crossover" in ids
    assert "EXP23_post_adx_parts_minus_log_vr" in ids
    assert "EXP24_post_adx_parts_minus_irr" in ids
    assert "EXP25_post_adx_parts_minus_persistence_log_vr" in ids


def test_build_post_adx_persistence_rewrite_registry_contains_variants():
    registry = build_post_adx_persistence_rewrite_registry(session_name="1pm-3pm")
    ids = {experiment.experiment_id for experiment in registry}

    assert "EXP30_post_adx_persistence_base" in ids
    assert "EXP31_post_adx_persistence_margin" in ids
    assert "EXP32_post_adx_persistence_control" in ids
    assert "EXP33_post_adx_persistence_recency" in ids


def test_filter_table_for_era_excludes_covid_transition_dates():
    table = pd.DataFrame(
        {
            "trade_date": [
                date(2020, 2, 28),
                date(2020, 3, 2),
                date(2020, 6, 30),
                date(2020, 7, 1),
                date(2021, 1, 4),
            ],
            "descriptive_target": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    full = filter_table_for_era(table, "full_dev")
    pre = filter_table_for_era(table, "pre_covid")
    post = filter_table_for_era(table, "post_covid")

    assert set(full["trade_date"]) == {date(2020, 2, 28), date(2020, 7, 1), date(2021, 1, 4)}
    assert set(pre["trade_date"]) == {date(2020, 2, 28)}
    assert set(post["trade_date"]) == {date(2020, 7, 1), date(2021, 1, 4)}
