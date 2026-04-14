from __future__ import annotations

import features.trend.modeling as modeling


def test_modeling_package_exports_core_helpers():
    assert callable(modeling.build_descriptive_target)
    assert callable(modeling.build_modeling_table)
    assert callable(modeling.build_experiment_registry)
    assert callable(modeling.run_walkforward_experiment)
