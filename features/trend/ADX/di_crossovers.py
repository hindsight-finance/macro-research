import numpy as np


def count_di_crossovers(plus_di: np.array, minus_di: np.array) -> int:
    """
    Count number of times +DI and -DI cross (switch dominance).
    
    Args:
        plus_di: Array of +DI values
        minus_di: Array of -DI values
        
    Returns:
        Number of crossovers (sign changes)
    """
    if len(plus_di) != len(minus_di):
        raise ValueError("DI arrays must be same length")
    
    if len(plus_di) < 2:
        return 0
    
    # Calculate difference: positive = +DI leads, negative = -DI leads
    di_difference = plus_di - minus_di
    
    # Count sign changes
    crossovers = 0
    for i in range(1, len(di_difference)):
        # Check if sign changed (crossed zero)
        if (di_difference[i-1] > 0 and di_difference[i] < 0) or \
           (di_difference[i-1] < 0 and di_difference[i] > 0):
            crossovers += 1
    
    return crossovers


def calculate_crossover_penalty(
    plus_di: np.array, 
    minus_di: np.array, 
    max_expected: float = None
) -> float:
    """
    Calculate crossover penalty score (0 to 1).
    0 = many crossovers (choppy)
    1 = no crossovers (smooth)
    
    Args:
        plus_di: Array of +DI values
        minus_di: Array of -DI values
        max_expected: Maximum expected crossovers for normalization.
                      If None, defaults to len(array) / 2 (extremely choppy)
        
    Returns:
        Penalty score between 0 and 1
    """
    crossovers = count_di_crossovers(plus_di, minus_di)
    
    # Normalize: fewer crossovers = higher score
    if max_expected is None:
        # Assume max expected crossovers = len(array) / 2 (extremely choppy)
        max_expected = len(plus_di) / 2
    
    if max_expected == 0:
        return 1.0  # No expected crossovers, perfect score
    
    penalty = 1 - (crossovers / max_expected)
    
    # Clamp between 0 and 1
    return max(0.0, min(1.0, penalty))


# Example usage:
# Bars:  [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
# +DI:   [20, 25, 30, 15, 10, 12, 35, 40, 38, 36]
# -DI:   [15, 10, 12, 25, 28, 30, 20, 15, 18, 20]
#
# Diff:  [+5,+15,+18,-10,-18,-18,+15,+25,+20,+16]
# Signs: [+,  +,  +,  -,  -,  -,  +,  +,  +,  +]
#
# Crossovers = 2 (at bars 4 and 7)
# Penalty = 1 - (2 / 5) = 0.6
