import numpy as np


def calculate_di_persistence(plus_di: np.array, minus_di: np.array) -> float:
    """
    Calculate what percentage of bars show consistent DI dominance.
    
    Args:
        plus_di: Array of +DI values
        minus_di: Array of -DI values
        
    Returns:
        Persistence score between 0 and 1
        1.0 = same DI led entire period
        0.0 = perfect alternation
    """
    if len(plus_di) != len(minus_di):
        raise ValueError("DI arrays must be same length")
    
    if len(plus_di) < 2:
        return 0.0
    
    # Determine which DI is dominant for each bar
    # True = +DI dominant, False = -DI dominant
    # Handle equality: maintain previous state (forward-fill)
    dominant_di = np.zeros(len(plus_di), dtype=bool)
    
    # Set initial value: if equal, default to +DI (True)
    dominant_di[0] = plus_di[0] >= minus_di[0]
    
    # Forward-fill equal values with previous state
    for i in range(1, len(plus_di)):
        if plus_di[i] > minus_di[i]:
            dominant_di[i] = True
        elif plus_di[i] < minus_di[i]:
            dominant_di[i] = False
        else:
            # Equal case: maintain previous bar's dominance
            dominant_di[i] = dominant_di[i-1]
    
    # Count consecutive bars where same DI leads
    max_consecutive = 1
    current_consecutive = 1
    
    for i in range(1, len(dominant_di)):
        if dominant_di[i] == dominant_di[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    # Normalize: max_consecutive / total_bars
    persistence_score = max_consecutive / len(dominant_di)
    
    return persistence_score

def calculate_di_persistence_avg(plus_di: np.array, minus_di: np.array) -> float:
    """
    Calculate average length of consecutive dominance periods.
    More sensitive to multiple trend attempts.
    
    Args:
        plus_di: Array of +DI values
        minus_di: Array of -DI values
        
    Returns:
        Persistence score between 0 and 1 based on average run length
    """
    if len(plus_di) != len(minus_di):
        raise ValueError("DI arrays must be same length")
    
    if len(plus_di) < 2:
        return 0.0
    
    # Determine which DI is dominant for each bar
    # Handle equality: maintain previous state (forward-fill)
    dominant_di = np.zeros(len(plus_di), dtype=bool)
    
    # Set initial value: if equal, default to +DI (True)
    dominant_di[0] = plus_di[0] >= minus_di[0]
    
    # Forward-fill equal values with previous state
    for i in range(1, len(plus_di)):
        if plus_di[i] > minus_di[i]:
            dominant_di[i] = True
        elif plus_di[i] < minus_di[i]:
            dominant_di[i] = False
        else:
            # Equal case: maintain previous bar's dominance
            dominant_di[i] = dominant_di[i-1]
    
    runs = []
    current_run = 1
    
    for i in range(1, len(dominant_di)):
        if dominant_di[i] == dominant_di[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    
    runs.append(current_run)  # Add final run
    
    # Average run length divided by total bars
    avg_run = np.mean(runs)
    persistence_score = avg_run / len(dominant_di)
    
    return persistence_score


# Example usage:
# Bars:  [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
# +DI:   [20, 25, 30, 15, 10, 12, 35, 40, 38, 36]
# -DI:   [15, 10, 12, 25, 28, 30, 20, 15, 18, 20]
#
# Dominant: [+, +, +, -, -, -, +, +, +, +]
#
# Max consecutive = 4 bars (+DI at end)
# Persistence score = 4/10 = 0.4