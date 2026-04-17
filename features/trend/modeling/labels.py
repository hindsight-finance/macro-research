from __future__ import annotations

import pandas as pd


THREE_SCALAR_LABELS = ("trend", "containment", "chop")

DEFAULT_LABEL_THRESHOLDS = {
    "trend_high": 0.70,
    "containment_high": 0.70,
    "chop_high": 0.70,
    "low_cutoff": 0.40,
    "containment_chop_max": 0.55,
}


def assign_three_scalar_labels(
    frame: pd.DataFrame,
    trend_high: float,
    containment_high: float,
    chop_high: float,
    low_cutoff: float,
    containment_chop_max: float | None = None,
    uncertain_label: str = "uncertain",
    drop_uncertain: bool = False,
) -> pd.DataFrame:
    chop_cap = containment_chop_max if containment_chop_max is not None else 0.55
    labels = pd.Series(uncertain_label, index=frame.index, dtype="object")

    labels.loc[
        (frame["trend_score"] >= trend_high)
        & (frame["containment_score"] <= low_cutoff)
        & (frame["chop_score"] <= low_cutoff)
    ] = "trend"
    labels.loc[
        (frame["containment_score"] >= containment_high)
        & (frame["trend_score"] <= low_cutoff)
        & (frame["chop_score"] <= chop_cap)
    ] = "containment"
    labels.loc[
        (frame["chop_score"] >= chop_high)
        & (frame["trend_score"] <= low_cutoff)
        & (frame["containment_score"] <= low_cutoff)
    ] = "chop"

    labeled = frame.copy()
    labeled["label"] = labels.to_numpy()
    if drop_uncertain:
        labeled = labeled.loc[labeled["label"].isin(THREE_SCALAR_LABELS)].copy()
    return labeled
