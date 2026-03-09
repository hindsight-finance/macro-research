import pandas as pd
import numpy as np
from typing import List, Optional, Union

# Configuration constants
DEFAULT_WINDOW = 10  # bars for rolling average


class DRA:
    """
    Dynamic Range Analysis (DRA) - Tracks overlap between current bars and initial range.
    
    Measures how much subsequent price action overlaps with an initial reference range
    (typically 3pm-3:10pm). Higher overlap indicates consolidation within the range,
    while lower overlap suggests breakout/trending behavior.
    """
    
    def __init__(self, window: int = DEFAULT_WINDOW):
        """
        Initialize DRA calculator.
        
        Args:
            window: Number of bars for rolling average calculation (default: 10)
        """
        self.initial_high: Optional[float] = None
        self.initial_low: Optional[float] = None
        self.initial_range: Optional[float] = None
        self.overlaps: List[float] = []
        self.window: int = window
        
    def set_initial_range(self, bars_3pm_to_310pm: Union[List, pd.DataFrame]) -> None:
        """
        Set the initial reference range from bars between 3pm-3:10pm.
        
        Args:
            bars_3pm_to_310pm: List of bar objects with 'high' and 'low' attributes,
                              or DataFrame with 'high' and 'low' columns
                              
        Raises:
            ValueError: If input is empty or missing required columns
        """
        if isinstance(bars_3pm_to_310pm, pd.DataFrame):
            if bars_3pm_to_310pm.empty:
                raise ValueError("DataFrame is empty. Cannot set initial range.")
            if 'high' not in bars_3pm_to_310pm.columns or 'low' not in bars_3pm_to_310pm.columns:
                raise ValueError("DataFrame must contain 'high' and 'low' columns.")
            self.initial_high = bars_3pm_to_310pm['high'].max()
            self.initial_low = bars_3pm_to_310pm['low'].min()
        else:
            if not bars_3pm_to_310pm:
                raise ValueError("List is empty. Cannot set initial range.")
            self.initial_high = max(b.high for b in bars_3pm_to_310pm)
            self.initial_low = min(b.low for b in bars_3pm_to_310pm)
        
        self.initial_range = self.initial_high - self.initial_low
        
    def update(self, bar: Union[object, pd.Series]) -> float:
        """
        Update DRA with a new bar and return rolling average overlap.
        
        Args:
            bar: Bar object with 'high' and 'low' attributes, or pandas Series
                 with 'high' and 'low' values
                 
        Returns:
            Rolling average overlap ratio (0.0 to 1.0), or 0.0 if initial_range is 0
        """
        if self.initial_range is None:
            raise ValueError("Initial range not set. Call set_initial_range() first.")
        
        if self.initial_range == 0:
            return 0.0  # Avoid division by zero
        
        # Extract high and low from bar object or Series
        if isinstance(bar, pd.Series):
            bar_high = bar['high']
            bar_low = bar['low']
        else:
            bar_high = bar.high
            bar_low = bar.low
        
        # Calculate overlap
        overlap_high = min(bar_high, self.initial_high)
        overlap_low = max(bar_low, self.initial_low)
        overlap = (overlap_high - overlap_low) / self.initial_range
        
        # Cap overlap at 0 (no negative overlaps)
        self.overlaps.append(max(0.0, overlap))
        
        # Calculate rolling average
        recent = self.overlaps[-self.window:]
        return sum(recent) / len(recent)