"""
Low Resistance Liquidity (LRLR) Detection Module

This module detects two types of low resistance liquidity:
1. Trendline LRLR - unbroken trendlines through swing points
2. Equal Highs/Lows LRLR - tick equal or relatively equal levels
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class LRLRType(Enum):
    """Types of LRLR patterns"""
    TRENDLINE = "trendline"
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"


class EqualStrength(Enum):
    """Strength levels for equal highs/lows"""
    SWING_TICK_EQUAL = "swing_tick_equal"  # Strongest - both are swings AND tick equal
    TICK_SWING = "tick_swing"  # Very strong - tick equal candles forming swing peak
    TICK_EQUAL = "tick_equal"  # Strong - non-swing tick equal
    OLDER_HIGHER = "older_higher"  # Moderate - relative equal, older higher


@dataclass
class SwingPoint:
    """Represents a swing high or low"""
    index: int  # Bar index
    price: float
    is_high: bool  # True for swing high, False for swing low
    is_tick_swing: bool = False  # True if this is a tick swing (adjacent tick equal candles)


@dataclass
class LRLRPattern:
    """Represents a detected LRLR pattern"""
    type: LRLRType
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    swing_points: List[SwingPoint]
    strength: Optional[EqualStrength] = None  # Only for equal highs/lows
    slope: Optional[float] = None  # Only for trendlines
    is_buyside: bool = True  # True for resistance, False for support
    
    def calculate_weight(self, current_bar: int, timeframe_minutes: int = 5) -> float:
        """
        Calculate liquidity strength weight for this pattern
        
        Args:
            current_bar: Current bar index for time decay calculation
            timeframe_minutes: Timeframe in minutes (affects tick equal strength)
            
        Returns:
            Weight value (higher = stronger)
        """
        # Base strength by pattern type
        if self.type == LRLRType.TRENDLINE:
            base_strength = 0.9  # Strong structural level
        elif self.type == LRLRType.EQUAL_HIGHS or self.type == LRLRType.EQUAL_LOWS:
            if self.strength == EqualStrength.SWING_TICK_EQUAL:
                base_strength = 1.0  # Strongest - both are swings AND tick equal (separated)
            elif self.strength == EqualStrength.TICK_SWING:
                base_strength = 0.9  # Very strong - adjacent tick equal candles forming peak
            elif self.strength == EqualStrength.TICK_EQUAL:
                base_strength = 0.75  # Strong - non-swing tick equal (separated)
            elif self.strength == EqualStrength.OLDER_HIGHER:
                base_strength = 0.7  # Moderate - relative equal, older higher
            else:
                base_strength = 0.5
        else:
            base_strength = 0.5
        
        # Time decay factor (exponential decay based on recency)
        bars_ago = current_bar - self.end_index
        if bars_ago < 0:
            bars_ago = 0
        
        # Decay rate: ~50% strength after 100 bars, ~25% after 200 bars
        decay_rate = 0.007
        time_factor = np.exp(-decay_rate * bars_ago)
        
        # Timeframe multiplier for tick-based patterns (higher TF = stronger)
        timeframe_multiplier = 1.0
        if self.strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL):
            # 1min = 1.0x, 5min = 1.4x, 15min = 2.0x, 1hr = 3.0x, etc.
            timeframe_multiplier = 1.0 + (np.log(timeframe_minutes) / 3.0)
        
        # Final weight
        weight = base_strength * time_factor * timeframe_multiplier
        
        return weight


class LRLRDetector:
    """Detects Low Resistance Liquidity patterns in price data"""
    
    def __init__(
        self,
        swing_lookback: int = 3,
        min_trendline_touches: int = 3,
        max_trendline_touches: int = 7,  # NEW: Limit for "one clean pivot"
        trendline_tolerance_pct: float = 0.0005,  # 0.05% for intraday
        equal_tolerance_pct: float = 0.001,  # 0.1% for "relatively equal"
        slope_variance_pct: float = 0.20,  # NEW: 15% max variance in slope between segments
        price_deviation_multiplier: float = 3.0  # NEW: How many std devs allowed from trendline
    ):
        """
        Initialize LRLR detector
        
        Args:
            swing_lookback: Bars to look back/forward for swing points
            min_trendline_touches: Minimum swing points for trendline LRLR
            max_trendline_touches: Maximum swing points (for "one clean pivot")
            trendline_tolerance_pct: Allowed deviation from trendline for swing points
            equal_tolerance_pct: Tolerance for "relatively equal" levels
            slope_variance_pct: Maximum allowed variance in slope between consecutive segments
            price_deviation_multiplier: Std dev multiplier for price deviation from trendline
        """
        self.swing_lookback = swing_lookback
        self.min_touches = min_trendline_touches
        self.max_touches = max_trendline_touches
        self.trendline_tolerance = trendline_tolerance_pct
        self.equal_tolerance = equal_tolerance_pct
        self.slope_variance = slope_variance_pct
        self.deviation_multiplier = price_deviation_multiplier
    
    def find_swing_highs(self, highs: np.ndarray) -> List[SwingPoint]:
        """
        Find swing high points in price data, including tick swing highs.
        
        Tick swing high: 4-candle pattern where candles 2 & 3 have equal highs,
        and candles 1 & 4 have lower highs than 2 & 3.
        """
        swings = []
        lookback = self.swing_lookback
        skip_next = False  # Track if we should skip the next bar (part of tick swing)
        
        for i in range(lookback, len(highs) - lookback):
            # Skip if this bar was part of a previously detected tick swing
            if skip_next:
                skip_next = False
                continue
            
            center_high = highs[i]
            
            # Check for regular swing high
            is_swing = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] >= center_high:
                    is_swing = False
                    break
            
            if is_swing:
                swings.append(SwingPoint(i, center_high, True, is_tick_swing=False))
                continue
            
            # Check for tick swing high (4-candle pattern)
            # Pattern: [lower, EQUAL, EQUAL, lower]
            #          [i-1,  i,    i+1,  i+2]
            if (i > 0 and i + 2 < len(highs) and 
                highs[i] == highs[i + 1] and  # Candles 2 & 3 are tick equal
                highs[i - 1] < highs[i] and    # Candle 1 is lower than 2
                highs[i + 2] < highs[i]):      # Candle 4 is lower than 3
                
                tick_swing_high = highs[i]
                is_tick_swing = True
                
                # Additional check: ensure candles 2 & 3 are higher than surrounding lookback
                for j in range(max(0, i - lookback), min(len(highs), i + 3 + lookback)):
                    if j < i - 1 or j > i + 2:  # Exclude the 4-candle pattern itself
                        if highs[j] >= tick_swing_high:
                            is_tick_swing = False
                            break
                
                if is_tick_swing:
                    # Use the first tick equal candle's index as the swing point
                    swings.append(SwingPoint(i, tick_swing_high, True, is_tick_swing=True))
                    skip_next = True  # Skip the next candle (i+1) as it's part of this pattern
        
        return swings
    
    def find_swing_lows(self, lows: np.ndarray) -> List[SwingPoint]:
        """
        Find swing low points in price data, including tick swing lows.
        
        Tick swing low: 4-candle pattern where candles 2 & 3 have equal lows,
        and candles 1 & 4 have higher lows than 2 & 3.
        """
        swings = []
        lookback = self.swing_lookback
        skip_next = False  # Track if we should skip the next bar (part of tick swing)
        
        for i in range(lookback, len(lows) - lookback):
            # Skip if this bar was part of a previously detected tick swing
            if skip_next:
                skip_next = False
                continue
            
            center_low = lows[i]
            
            # Check for regular swing low
            is_swing = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and lows[j] <= center_low:
                    is_swing = False
                    break
            
            if is_swing:
                swings.append(SwingPoint(i, center_low, False, is_tick_swing=False))
                continue
            
            # Check for tick swing low (4-candle pattern)
            # Pattern: [higher, EQUAL, EQUAL, higher]
            #          [i-1,    i,     i+1,   i+2]
            if (i > 0 and i + 2 < len(lows) and 
                lows[i] == lows[i + 1] and  # Candles 2 & 3 are tick equal
                lows[i - 1] > lows[i] and    # Candle 1 is higher than 2
                lows[i + 2] > lows[i]):      # Candle 4 is higher than 3
                
                tick_swing_low = lows[i]
                is_tick_swing = True
                
                # Additional check: ensure candles 2 & 3 are lower than surrounding lookback
                for j in range(max(0, i - lookback), min(len(lows), i + 3 + lookback)):
                    if j < i - 1 or j > i + 2:  # Exclude the 4-candle pattern itself
                        if lows[j] <= tick_swing_low:
                            is_tick_swing = False
                            break
                
                if is_tick_swing:
                    # Use the first tick equal candle's index as the swing point
                    swings.append(SwingPoint(i, tick_swing_low, False, is_tick_swing=True))
                    skip_next = True  # Skip the next candle (i+1) as it's part of this pattern
        
        return swings
    

    def validate_trendline(self, swing_points: List[SwingPoint], 
                          highs: np.ndarray, lows: np.ndarray,
                          closes: np.ndarray, is_buyside: bool) -> bool:
        """
        Validate that trendline remains unbroken
        
        Checks:
        1. Slope direction (negative for buyside, positive for sellside)
        2. Oldest point is highest/lowest
        3. No closes break the trendline
        4. Swing points close to the line
        5. Slope consistency between consecutive segments (Option 4)
        6. Price deviation from trendline within acceptable range (Your idea)
        """
        if len(swing_points) < 2:
            return False
        
        # Calculate trendline using first and last points
        start = swing_points[0]
        end = swing_points[-1]
        
        slope = (end.price - start.price) / (end.index - start.index)
        
        # Check 1: Slope direction
        if is_buyside:
            if slope >= 0:
                return False
            oldest_price = start.price
            for point in swing_points:
                if point.price > oldest_price:
                    return False
        else:
            if slope <= 0:
                return False
            oldest_price = start.price
            for point in swing_points:
                if point.price < oldest_price:
                    return False
        
        # Check 2: Slope consistency between consecutive swing segments (Option 4)
        if len(swing_points) > 2:
            segment_slopes = []
            for i in range(len(swing_points) - 1):
                p1 = swing_points[i]
                p2 = swing_points[i + 1]
                seg_slope = (p2.price - p1.price) / (p2.index - p1.index)
                segment_slopes.append(seg_slope)
            
            # Check variance of segment slopes against main slope
            for seg_slope in segment_slopes:
                if abs(slope) > 0:  # Avoid division by zero
                    slope_diff = abs((seg_slope - slope) / slope)
                    if slope_diff > self.slope_variance:
                        return False
        
        # Check 3: Price deviation from trendline between swings (Your idea)
        # Calculate standard deviation of price distance from trendline
        deviations = []
        price_array = highs if is_buyside else lows
        
        for i in range(start.index, end.index + 1):
            expected_price = start.price + slope * (i - start.index)
            actual_price = price_array[i]
            deviation = abs(actual_price - expected_price)
            deviations.append(deviation)
        
        if len(deviations) > 1:
            std_dev = np.std(deviations)
            mean_dev = np.mean(deviations)
            
            # Check if any price point deviates too far
            for i in range(start.index, end.index + 1):
                expected_price = start.price + slope * (i - start.index)
                actual_price = price_array[i]
                deviation = abs(actual_price - expected_price)
                
                # Reject if deviation exceeds mean + N * std_dev
                if deviation > mean_dev + (self.deviation_multiplier * std_dev):
                    return False
        
        # Check 4: No closes break the line
        for i in range(start.index, end.index + 1):
            expected_price = start.price + slope * (i - start.index)
            
            if is_buyside:
                if closes[i] > expected_price * (1 + self.trendline_tolerance):
                    return False
            else:
                if closes[i] < expected_price * (1 - self.trendline_tolerance):
                    return False
        
        # Check 5: Swing points are reasonably close to the line
        for point in swing_points:
            expected = start.price + slope * (point.index - start.index)
            deviation = abs(point.price - expected) / expected
            
            if deviation > self.trendline_tolerance:
                return False
        
        return True
    
    def find_trendline_lrlr(self, swing_points: List[SwingPoint],
                           highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, is_buyside: bool) -> List[LRLRPattern]:
        """
        Find valid trendline LRLR patterns
        
        Option 5: "One clean pivot" approach
        - Limits each trendline to max_trendline_touches
        - Prevents overly long trendlines
        - Creates discrete, clean trendlines
        """
        patterns = []
        n = len(swing_points)
        
        if n < self.min_touches:
            return patterns
        
        used_indices = set()  # Track which swing points are already used
        
        # Limit search to max_touches for "one clean pivot"
        max_length = min(self.max_touches, n)
        
        # Try different combinations of swing points
        # Start with longest allowed, work backwards
        for length in range(max_length, self.min_touches - 1, -1):
            for start_idx in range(n - length + 1):
                # Skip if any of these points are already used in another trendline
                if any(i in used_indices for i in range(start_idx, start_idx + length)):
                    continue
                
                end_idx = start_idx + length
                points = swing_points[start_idx:end_idx]
                
                # Validate this trendline
                if self.validate_trendline(points, highs, lows, closes, is_buyside):
                    start_point = points[0]
                    end_point = points[-1]
                    
                    slope = (end_point.price - start_point.price) / \
                           (end_point.index - start_point.index)
                    
                    pattern = LRLRPattern(
                        type=LRLRType.TRENDLINE,
                        start_index=start_point.index,
                        end_index=end_point.index,
                        start_price=start_point.price,
                        end_price=end_point.price,
                        swing_points=points,
                        slope=slope,
                        is_buyside=is_buyside
                    )
                    patterns.append(pattern)
                    
                    # Mark these swing points as used (Option 5)
                    for i in range(start_idx, end_idx):
                        used_indices.add(i)
        
        return patterns
    
    def find_tick_swing_patterns(self, swing_points: List[SwingPoint],
                                  is_buyside: bool) -> List[LRLRPattern]:
        """
        Create patterns from detected tick swings.
        
        A tick swing is a 4-candle pattern where the middle 2 candles are tick equal.
        This creates a TICK_SWING pattern from the two tick equal candles.
        """
        patterns = []
        
        for swing in swing_points:
            if swing.is_tick_swing:
                # Create a pattern representing the two tick equal candles
                # The swing point index is at the first tick equal candle (candle 2 in the 4-candle pattern)
                point1 = SwingPoint(swing.index, swing.price, is_buyside, is_tick_swing=False)
                point2 = SwingPoint(swing.index + 1, swing.price, is_buyside, is_tick_swing=False)
                
                pattern_type = LRLRType.EQUAL_HIGHS if is_buyside else LRLRType.EQUAL_LOWS
                
                pattern = LRLRPattern(
                    type=pattern_type,
                    start_index=swing.index,
                    end_index=swing.index + 1,
                    start_price=swing.price,
                    end_price=swing.price,
                    swing_points=[point1, point2],
                    strength=EqualStrength.TICK_SWING,
                    is_buyside=is_buyside
                )
                patterns.append(pattern)
        
        return patterns
    
    def find_equal_highs_lows(self, swing_points: List[SwingPoint],
                             highs: np.ndarray, lows: np.ndarray,
                             is_buyside: bool) -> List[LRLRPattern]:
        """
        Find relatively equal swing highs or lows (OLDER_HIGHER patterns only)
        
        Note: Tick equal patterns are handled by find_all_tick_equal_patterns()
        
        Requirements:
        1. Two swing points must be relatively equal (within tolerance BUT not tick equal)
        2. Older point must be higher (for highs) or lower (for lows) - no swept levels
        3. No price in between can exceed the level (strict validation)
        
        Returns only OLDER_HIGHER strength patterns
        """
        patterns = []
        
        # Look for pairs of relatively equal levels (excluding tick swings, which are handled separately)
        for i in range(len(swing_points) - 1):
            for j in range(i + 1, len(swing_points)):
                point1 = swing_points[i]  # Older point
                point2 = swing_points[j]  # Newer point
                
                # Skip if either point is a tick swing (handled by find_tick_swing_patterns)
                if point1.is_tick_swing or point2.is_tick_swing:
                    continue
                
                price_diff = abs(point1.price - point2.price)
                avg_price = (point1.price + point2.price) / 2
                diff_pct = price_diff / avg_price
                
                # Skip tick equal patterns - they are handled by find_all_tick_equal_patterns
                if price_diff == 0:
                    continue
                
                strength = None
                
                # Check if relatively equal (within tolerance but not tick equal)
                if diff_pct <= self.equal_tolerance:
                    # Relatively equal - only accept if older is higher/lower (never swept)
                    if is_buyside:
                        # Buyside (highs): older must be >= newer (not swept)
                        if point1.price >= point2.price:
                            strength = EqualStrength.OLDER_HIGHER
                        else:
                            continue  # Skip - older is lower means it was swept
                    else:
                        # Sellside (lows): older must be <= newer (not swept)
                        if point1.price <= point2.price:
                            strength = EqualStrength.OLDER_HIGHER
                        else:
                            continue  # Skip - older is higher means it was swept
                else:
                    continue  # Not equal enough
                
                # Strict validation: NO price in between can exceed the level
                max_level = max(point1.price, point2.price)
                min_level = min(point1.price, point2.price)
                
                valid = True
                for k in range(point1.index + 1, point2.index):
                    if is_buyside:
                        # NO high can be higher than the max level (strict)
                        if highs[k] > max_level:
                            valid = False
                            break
                    else:
                        # NO low can be lower than the min level (strict)
                        if lows[k] < min_level:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                pattern_type = LRLRType.EQUAL_HIGHS if is_buyside else LRLRType.EQUAL_LOWS
                
                pattern = LRLRPattern(
                    type=pattern_type,
                    start_index=point1.index,
                    end_index=point2.index,
                    start_price=point1.price,
                    end_price=point2.price,
                    swing_points=[point1, point2],
                    strength=strength,
                    is_buyside=is_buyside
                )
                patterns.append(pattern)
        
        return patterns
    
    def find_all_tick_equal_patterns(self, highs: np.ndarray, lows: np.ndarray,
                                     swing_points: List[SwingPoint],
                                     is_buyside: bool) -> List[LRLRPattern]:
        """
        Find all tick equal patterns, classified by strength.
        
        Checks in order of strength:
        1. SWING_TICK_EQUAL: Both bars are regular swings AND tick equal
        2. TICK_SWING: 4-candle pattern where middle 2 are tick equal (detected as swing)
        3. TICK_EQUAL: Non-swing bars that are tick equal
        
        Only applies to exact price matches. Price must not trade past these levels.
        """
        patterns = []
        price_array = highs if is_buyside else lows
        
        # Build lookup maps for swing classification
        swing_index_map = {}  # Maps index -> SwingPoint
        for swing in swing_points:
            swing_index_map[swing.index] = swing
            # Tick swings also cover the adjacent bar
            if swing.is_tick_swing:
                swing_index_map[swing.index + 1] = swing
        
        # Group ALL bars by exact price
        price_map = {}
        for i in range(len(price_array)):
            price = price_array[i]
            if price not in price_map:
                price_map[price] = []
            price_map[price].append(i)
        
        # Find and classify tick equal pairs
        for price, indices in price_map.items():
            if len(indices) < 2:
                continue
            
            # Check all pairs at this exact price
            for i in range(len(indices) - 1):
                for j in range(i + 1, len(indices)):
                    idx1 = indices[i]
                    idx2 = indices[j]
                    
                    # Validate price didn't trade past this level between the two points
                    valid = True
                    for k in range(idx1 + 1, idx2):
                        if is_buyside:
                            if highs[k] > price:
                                valid = False
                                break
                        else:
                            if lows[k] < price:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # Determine strength based on bar classification
                    swing1 = swing_index_map.get(idx1)
                    swing2 = swing_index_map.get(idx2)
                    
                    # Check 1: Are both REGULAR swings? (not tick swings)
                    if (swing1 and swing2 and 
                        not swing1.is_tick_swing and not swing2.is_tick_swing and
                        swing1.index == idx1 and swing2.index == idx2):
                        strength = EqualStrength.SWING_TICK_EQUAL
                    
                    # Check 2: Is this a TICK_SWING pattern? (already handled by find_tick_swing_patterns)
                    # Skip if either bar is part of a tick swing to avoid duplicates
                    elif swing1 and swing1.is_tick_swing and swing1.index == idx1 and idx2 == idx1 + 1:
                        continue  # This will be handled by find_tick_swing_patterns
                    elif swing2 and swing2.is_tick_swing and swing2.index == idx2 - 1 and idx1 == idx2 - 1:
                        continue  # This will be handled by find_tick_swing_patterns
                    
                    # Check 3: Default to TICK_EQUAL (at least one is not a swing)
                    else:
                        strength = EqualStrength.TICK_EQUAL
                    
                    # Create pattern
                    point1 = SwingPoint(idx1, price, is_buyside)
                    point2 = SwingPoint(idx2, price, is_buyside)
                    
                    pattern_type = LRLRType.EQUAL_HIGHS if is_buyside else LRLRType.EQUAL_LOWS
                    
                    pattern = LRLRPattern(
                        type=pattern_type,
                        start_index=idx1,
                        end_index=idx2,
                        start_price=price,
                        end_price=price,
                        swing_points=[point1, point2],
                        strength=strength,
                        is_buyside=is_buyside
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def detect(self, highs: np.ndarray, lows: np.ndarray, 
              closes: np.ndarray) -> Dict[str, List[LRLRPattern]]:
        """
        Main detection method
        
        Pattern Detection Hierarchy (in order of strength):
        1. SWING_TICK_EQUAL (1.0): Regular swings that are tick equal (find_all_tick_equal_patterns)
        2. TICK_SWING (0.9): 4-candle pattern with middle 2 tick equal (find_tick_swing_patterns)
        3. TRENDLINE (0.9): Unbroken trendlines through swings (find_trendline_lrlr)
        4. TICK_EQUAL (0.75): Non-swing bars that are tick equal (find_all_tick_equal_patterns)
        5. OLDER_HIGHER (0.7): Regular swings relatively equal (find_equal_highs_lows)
        
        The unified find_all_tick_equal_patterns() checks tick equal bars in order:
        - First checks if both are regular swings → SWING_TICK_EQUAL
        - Then checks if it's a tick swing (skips to avoid duplicates)
        - Finally classifies as TICK_EQUAL for non-swing bars
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            
        Returns:
            Dictionary with keys 'buyside' and 'sellside', each containing list of patterns
        """
        # Find swing points (includes both regular swings and tick swings)
        swing_highs = self.find_swing_highs(highs)
        swing_lows = self.find_swing_lows(lows)
        
        # Find buyside LRLR (resistance)
        buyside_trendlines = self.find_trendline_lrlr(
            swing_highs, highs, lows, closes, is_buyside=True
        )
        buyside_tick_swings = self.find_tick_swing_patterns(
            swing_highs, is_buyside=True
        )
        # Unified tick equal detection (checks in order: SWING_TICK_EQUAL -> TICK_SWING -> TICK_EQUAL)
        buyside_tick_equals = self.find_all_tick_equal_patterns(
            highs, lows, swing_highs, is_buyside=True
        )
        # Relatively equal swings (OLDER_HIGHER)
        buyside_relative_equals = self.find_equal_highs_lows(
            swing_highs, highs, lows, is_buyside=True
        )
        
        # Find sellside LRLR (support)
        sellside_trendlines = self.find_trendline_lrlr(
            swing_lows, highs, lows, closes, is_buyside=False
        )
        sellside_tick_swings = self.find_tick_swing_patterns(
            swing_lows, is_buyside=False
        )
        # Unified tick equal detection (checks in order: SWING_TICK_EQUAL -> TICK_SWING -> TICK_EQUAL)
        sellside_tick_equals = self.find_all_tick_equal_patterns(
            highs, lows, swing_lows, is_buyside=False
        )
        # Relatively equal swings (OLDER_HIGHER)
        sellside_relative_equals = self.find_equal_highs_lows(
            swing_lows, highs, lows, is_buyside=False
        )
        
        return {
            'buyside': buyside_trendlines + buyside_tick_swings + buyside_tick_equals + buyside_relative_equals,
            'sellside': sellside_trendlines + sellside_tick_swings + sellside_tick_equals + sellside_relative_equals
        }