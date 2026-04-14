import numpy as np


def _validate_di_arrays(plus_di: np.ndarray, minus_di: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plus_arr = np.asarray(plus_di, dtype=float)
    minus_arr = np.asarray(minus_di, dtype=float)

    if len(plus_arr) != len(minus_arr):
        raise ValueError("DI arrays must be same length")

    return plus_arr, minus_arr


def _dominance_sign_and_margin(plus_di: np.ndarray, minus_di: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plus_arr, minus_arr = _validate_di_arrays(plus_di, minus_di)
    if len(plus_arr) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    signs = np.zeros(len(plus_arr), dtype=float)
    signs[0] = 1.0 if plus_arr[0] >= minus_arr[0] else -1.0

    for i in range(1, len(plus_arr)):
        if plus_arr[i] > minus_arr[i]:
            signs[i] = 1.0
        elif plus_arr[i] < minus_arr[i]:
            signs[i] = -1.0
        else:
            signs[i] = signs[i - 1]

    margins = np.abs(plus_arr - minus_arr) / (plus_arr + minus_arr + 1e-12)
    margins = np.clip(margins, 0.0, 1.0)
    return signs, margins


def calculate_di_persistence(plus_di: np.ndarray, minus_di: np.ndarray) -> float:
    """
    Default persistence definition.

    Returns:
        Margin-weighted persistence score between 0 and 1.
    """
    return calculate_margin_weighted_persistence(plus_di, minus_di)


def calculate_di_persistence_avg(plus_di: np.ndarray, minus_di: np.ndarray) -> float:
    """
    Calculate average length of consecutive dominance periods.
    More sensitive to multiple trend attempts.
    """
    signs, _ = _dominance_sign_and_margin(plus_di, minus_di)
    if len(signs) < 2:
        return 0.0

    runs = []
    current_run = 1

    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1

    runs.append(current_run)
    avg_run = np.mean(runs)
    return avg_run / len(signs)


def calculate_margin_weighted_persistence(plus_di: np.ndarray, minus_di: np.ndarray) -> float:
    """
    Reward long dominance runs, scaled by how decisively the winning DI leads.
    """
    signs, margins = _dominance_sign_and_margin(plus_di, minus_di)
    n = len(signs)
    if n < 2:
        return 0.0

    score = 0.0
    run_start = 0

    for i in range(1, n + 1):
        if i == n or signs[i] != signs[run_start]:
            run_end = i
            run_length = run_end - run_start
            run_share = run_length / n
            run_margin = float(np.mean(margins[run_start:run_end]))
            score += (run_share ** 2) * run_margin
            run_start = i

    return float(np.clip(score, 0.0, 1.0))


def calculate_time_in_control_persistence(plus_di: np.ndarray, minus_di: np.ndarray) -> float:
    """
    Measure how strongly one side controlled the full window on average.
    """
    signs, margins = _dominance_sign_and_margin(plus_di, minus_di)
    if len(signs) < 2:
        return 0.0

    control = np.abs(np.mean(signs * margins))
    return float(np.clip(control, 0.0, 1.0))


def calculate_recency_weighted_persistence(
    plus_di: np.ndarray,
    minus_di: np.ndarray,
    alpha: float = 0.35,
) -> float:
    """
    Measure how persistently one side controls the most recent bars.
    """
    signs, margins = _dominance_sign_and_margin(plus_di, minus_di)
    if len(signs) < 2:
        return 0.0

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    state = signs * margins
    ema = float(state[0])
    for value in state[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema

    return float(np.clip(abs(ema), 0.0, 1.0))
