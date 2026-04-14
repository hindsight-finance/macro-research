from __future__ import annotations

import numpy as np


def build_descriptive_target(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict:
    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)

    if not (open_arr.size and high_arr.size and low_arr.size and close_arr.size):
        raise ValueError("open_, high, low, and close must be non-empty")

    if not (open_arr.shape == high_arr.shape == low_arr.shape == close_arr.shape):
        raise ValueError("open_, high, low, and close must have matching shapes")

    if not (
        np.isfinite(open_arr).all()
        and np.isfinite(high_arr).all()
        and np.isfinite(low_arr).all()
        and np.isfinite(close_arr).all()
    ):
        raise ValueError("open_, high, low, and close must contain only finite values")

    if np.any(close_arr <= 0):
        raise ValueError("close must contain only positive values")

    returns = np.diff(np.log(close_arr))
    n_returns = returns.size

    if n_returns:
        return_std = np.std(returns, ddof=1) if n_returns > 1 else 0.0
        target_strength = 1.0 - np.exp(
            -abs(np.sum(returns)) / (return_std * np.sqrt(n_returns) + 1e-12)
        )
    else:
        target_strength = 0.0

    nonzero_returns = returns[returns != 0]
    if nonzero_returns.size:
        up_share = float(np.mean(nonzero_returns > 0))
        down_share = float(np.mean(nonzero_returns < 0))
        target_consistency = 2.0 * max(up_share, down_share) - 1.0
        sign_changes = np.count_nonzero(np.sign(nonzero_returns[1:]) != np.sign(nonzero_returns[:-1]))
        target_smoothness = 1.0 - (sign_changes / (nonzero_returns.size - 1) if nonzero_returns.size > 1 else 0.0)
    else:
        target_consistency = 0.0
        target_smoothness = 1.0

    path_range = float(high_arr.max() - low_arr.min())
    target_retention = np.clip(
        abs(float(close_arr[-1] - open_arr[0])) / (path_range + 1e-12),
        0.0,
        1.0,
    )

    descriptive_target = (
        0.35 * target_strength
        + 0.25 * target_consistency
        + 0.20 * target_smoothness
        + 0.20 * target_retention
    )

    return {
        "target_strength": float(target_strength),
        "target_consistency": float(target_consistency),
        "target_smoothness": float(target_smoothness),
        "target_retention": float(target_retention),
        "descriptive_target": float(np.clip(descriptive_target, 0.0, 1.0)),
        "target_status": "ok",
    }
