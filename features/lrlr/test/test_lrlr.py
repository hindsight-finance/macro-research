"""
Test Low Resistance Liquidity (LRLR) Detection on NQ 1-minute bars.

Tests the LRLRDetector class with:
- Trendline LRLR detection (buyside and sellside)
- Equal highs/lows LRLR detection (swing-based and tick equal)
- Tick equal detection from all bars (not just swings)
- Pattern quality and strength distribution analysis
- Custom parameter testing
- Visualization of each detected pattern (optional, configurable)

The test suite includes:
1. test_lrlr_detection() - Main detection test with visualizations
2. test_tick_equal_detection() - Specific tests for tick equal patterns
3. test_pattern_quality_analysis() - Quality metrics and distribution analysis
4. test_custom_parameters() - Testing with different parameter settings

Run this script from the project root directory (macro/) using:
    python -m features.lrlr.test.test_lrlr [options]

Options:
    --no-viz              Skip all visualizations
    --max-viz N           Limit visualizations to first N patterns (default: 10)
    --viz-all             Create all visualizations (with multiprocessing!)
    --output-dir PATH     Output directory for visualizations
    --jobs N              Number of parallel jobs (-1 for all CPU cores, default: -1)
    --types ...           Pattern types to include (default: all). Choices:
                          trendline, equal, tick_equal, all

Performance:
    - Uses Polars for fast data loading (10-100x faster than pandas)
    - TRUE multiprocessing with joblib (utilizes all CPU cores)
    - Typical speed: 10-50+ charts/second depending on CPU cores
    - Linear scaling: 8 cores ≈ 8x faster than single-threaded

Examples:
    python -m features.lrlr.test.test_lrlr --no-viz                    # Fast mode, no charts
    python -m features.lrlr.test.test_lrlr --max-viz 20                # Create 20 charts (all cores)
    python -m features.lrlr.test.test_lrlr --viz-all --jobs 4          # All charts, 4 cores
    python -m features.lrlr.test.test_lrlr --viz-all                   # All charts, all cores
"""
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from typing import Optional, Tuple, Dict, List, Any
import time
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')  # Suppress matplotlib warnings in parallel

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add lrlr directory to path
lrlr_dir = Path(__file__).parent.parent
if str(lrlr_dir) not in sys.path:
    sys.path.insert(0, str(lrlr_dir))

# Import LRLR module
from lrlr import LRLRDetector, LRLRType, EqualStrength, LRLRPattern


def _calculate_weight_from_dict(pattern_dict: Dict[str, Any], current_bar: int,
                                timeframe_minutes: int = 1) -> float:
    """
    Calculate weight from pattern dictionary (for parallel processing).
    Mirrors LRLRPattern.calculate_weight() logic.
    """
    pattern_type = LRLRType(pattern_dict['type'])
    strength_val = pattern_dict.get('strength')
    strength = EqualStrength(strength_val) if strength_val else None
    is_buyside = pattern_dict['is_buyside']
    end_index = pattern_dict['end_index']

    # Base strength by pattern type
    if pattern_type == LRLRType.TRENDLINE:
        base_strength = 0.9
    elif pattern_type in (LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS):
        if strength == EqualStrength.SWING_TICK_EQUAL:
            base_strength = 1.0
        elif strength == EqualStrength.TICK_SWING:
            base_strength = 0.9
        elif strength == EqualStrength.TICK_EQUAL:
            base_strength = 0.75
        elif strength == EqualStrength.OLDER_HIGHER:
            base_strength = 0.7
        else:
            base_strength = 0.5
    else:
        base_strength = 0.5

    # Time decay
    bars_ago = max(0, current_bar - end_index)
    decay_rate = 0.007
    time_factor = np.exp(-decay_rate * bars_ago)

    # Timeframe multiplier for tick-based patterns
    timeframe_multiplier = 1.0
    if strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL):
        timeframe_multiplier = 1.0 + (np.log(timeframe_minutes) / 3.0)

    return base_strength * time_factor * timeframe_multiplier


def resolve_allowed_types(types_arg: List[str]) -> Dict[str, bool]:
    """
    Convert CLI types arg to flags.
    We support three logical buckets:
      - trendline
      - equal (both highs and lows)
      - tick_equal (strength == TICK_EQUAL)
    """
    flags = {
        "trendline": True,
        "equal": True,
        "tick_equal": True,
    }
    if "all" in types_arg or not types_arg:
        return flags

    # Start with all False, enable selected
    flags = {k: False for k in flags}
    for t in types_arg:
        if t == "trendline":
            flags["trendline"] = True
        elif t == "equal_highs" or t == "equal_lows" or t == "equal":
            flags["equal"] = True
        elif t == "tick_equal":
            flags["tick_equal"] = True
        elif t == "all":
            flags = {k: True for k in flags}
    return flags


def filter_patterns_by_type(patterns: List[LRLRPattern], flags: Dict[str, bool]) -> List[LRLRPattern]:
    """Filter patterns to only allowed logical buckets"""
    keep: List[LRLRPattern] = []
    want_trend = flags.get("trendline", False)
    want_equal = flags.get("equal", False)
    want_tick_equal = flags.get("tick_equal", False)

    for p in patterns:
        # Trendlines
        if want_trend and p.type == LRLRType.TRENDLINE:
            keep.append(p)
            continue

        # Equal highs/lows (normal)
        if want_equal and p.type in (LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS):
            # If tick-equal only is requested without equal, skip non-tick-based here
            strength = getattr(p, "strength", None)
            is_tick_based = strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL)
            if not want_tick_equal or not is_tick_based:
                keep.append(p)
                continue

        # Tick-based patterns treated as their own type (subset of equal highs/lows)
        if want_tick_equal and p.type in (LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS):
            strength = getattr(p, "strength", None)
            if strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL):
                keep.append(p)
                continue

    return keep


def create_single_visualization(
    df_dict: Dict[str, List],
    pattern_dict: Dict[str, Any],
    pattern_idx: int,
    output_dir: Path,
    start_idx: int,
    end_idx: int,
    timeframe_minutes: int = 1
) -> Tuple[str, int]:
    """
    Top-level worker function for parallel visualization.
    Must be at module level for pickling.

    Args:
        df_dict: Dictionary representation of price data
        pattern_dict: Dictionary representation of pattern
        pattern_idx: Pattern index number
        output_dir: Output directory path
        start_idx: Start index for x-axis labeling
        end_idx: End index for x-axis labeling (last bar in visualization window)
        timeframe_minutes: Timeframe in minutes (for weight calculation)

    Returns:
        Tuple of (filename, pattern_idx)
    """
    import polars as pl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Reconstruct dataframe
    plot_data = pl.DataFrame(df_dict)

    # Reconstruct pattern object from dict
    # We'll work with the dict directly to avoid pickling issues
    pattern_type = LRLRType(pattern_dict['type'])
    is_buyside = pattern_dict['is_buyside']

    # Calculate weight based on the last bar in visualization window
    current_bar = end_idx - 1
    weight = _calculate_weight_from_dict(pattern_dict, current_bar, timeframe_minutes)

    # Calculate bars_ago for display (from end of visualization window)
    bars_ago = max(0, current_bar - pattern_dict['end_index'])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot candlesticks
    width = 0.6
    for i in range(plot_data.height):
        open_price = plot_data['open'][i]
        high_price = plot_data['high'][i]
        low_price = plot_data['low'][i]
        close_price = plot_data['close'][i]

        is_bullish = close_price >= open_price
        body_color = '#26a69a' if is_bullish else '#ef5350'
        wick_color = body_color

        # Draw wick
        ax.plot([i, i], [low_price, high_price],
               color=wick_color, linewidth=1.0, alpha=0.8)

        # Draw body
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_height = body_top - body_bottom

        if body_height == 0:
            ax.plot([i - width/2, i + width/2], [open_price, open_price],
                   color=body_color, linewidth=2.0, alpha=0.8)
        else:
            rect = Rectangle((i - width/2, body_bottom), width, body_height,
                           facecolor=body_color, edgecolor=body_color,
                           linewidth=1.0, alpha=0.8)
            ax.add_patch(rect)

    # Set x-axis
    ax.set_xlim(-0.5, plot_data.height - 0.5)
    ax.set_xticks(range(0, plot_data.height, max(1, plot_data.height // 10)))
    ax.set_xticklabels([start_idx + i for i in range(0, plot_data.height, max(1, plot_data.height // 10))])

    # Pattern visualization
    pattern_start_plot = pattern_dict['start_index'] - start_idx
    pattern_end_plot = pattern_dict['end_index'] - start_idx

    # Color based on pattern type and side
    if is_buyside:
        color = 'red' if pattern_type == LRLRType.TRENDLINE else 'orange'
        label_prefix = 'Buyside'
    else:
        color = 'blue' if pattern_type == LRLRType.TRENDLINE else 'cyan'
        label_prefix = 'Sellside'

    # Draw pattern
    if pattern_type == LRLRType.TRENDLINE:
        swing_points = pattern_dict['swing_points']
        swing_x = [sp['index'] - start_idx for sp in swing_points]
        swing_y = [sp['price'] for sp in swing_points]

        ax.plot(swing_x, swing_y, color=color, linewidth=2.5,
                linestyle='--', alpha=0.8, label=f'{label_prefix} Trendline LRLR')
        ax.scatter(swing_x, swing_y, color=color, s=100, zorder=5,
                  marker='o', edgecolors='white', linewidths=1.5)

        if len(swing_x) >= 2 and pattern_dict.get('slope'):
            slope = pattern_dict['slope']
            x_extend = [swing_x[0] - 5, swing_x[-1] + 5]
            y_extend = [
                swing_y[0] + slope * (x_extend[0] - swing_x[0]),
                swing_y[-1] + slope * (x_extend[1] - swing_x[-1])
            ]
            ax.plot(x_extend, y_extend, color=color, linewidth=1.5,
                   linestyle=':', alpha=0.5)

    elif pattern_type in [LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]:
        equal_price = pattern_dict['start_price']

        ax.axhline(y=equal_price, color=color, linewidth=2.5,
                  linestyle='--', alpha=0.8,
                  label=f'{label_prefix} Equal {"Highs" if pattern_type == LRLRType.EQUAL_HIGHS else "Lows"} LRLR')

        for sp in pattern_dict['swing_points']:
            sp_x = sp['index'] - start_idx
            ax.scatter(sp_x, sp['price'], color=color, s=120, zorder=5,
                      marker='s', edgecolors='white', linewidths=1.5)

        if pattern_dict.get('strength'):
            strength_text = {
                'swing_tick_equal': "Swing Tick Equal (Strongest) ✓✓",
                'tick_swing': "Tick Swing - Adjacent (Very Strong) ✓",
                'tick_equal': "Tick Equal - Separated (Strong)",
                'older_higher': "Older Higher - Relative (Moderate)"
            }.get(pattern_dict['strength'], "Unknown")

            bars_between = pattern_dict['end_index'] - pattern_dict['start_index']
            annotation_text = f"{strength_text}\n{bars_between} bars apart"

            mid_x = (pattern_dict['start_index'] + pattern_dict['end_index']) / 2 - start_idx
            y_range = plot_data['high'].max() - plot_data['low'].min()
            y_offset = y_range * (0.05 if is_buyside else -0.05)
            ax.annotate(annotation_text,
                       xy=(mid_x, equal_price),
                       xytext=(mid_x, equal_price + y_offset),
                       fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                                alpha=0.3, edgecolor=color, linewidth=1.5))

    # Highlight pattern region
    ax.axvspan(pattern_start_plot, pattern_end_plot,
              alpha=0.1, color=color, label='Pattern Region')

    # Formatting
    ax.set_xlabel('Bar Index (relative)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # Title
    pattern_type_str = {
        LRLRType.TRENDLINE: "Trendline",
        LRLRType.EQUAL_HIGHS: "Equal Highs",
        LRLRType.EQUAL_LOWS: "Equal Lows"
    }.get(pattern_type, "Unknown")

    side_str = "Buyside (Resistance)" if is_buyside else "Sellside (Support)"

    title = f"LRLR Pattern #{pattern_idx}: {pattern_type_str} - {side_str}\n"
    title += f"Weight: {weight:.3f} | Bars Ago: {bars_ago} | "
    title += f"Bars {pattern_dict['start_index']} to {pattern_dict['end_index']} | "
    title += f"Price: ${pattern_dict['start_price']:.2f} to ${pattern_dict['end_price']:.2f}"

    if pattern_type == LRLRType.TRENDLINE and pattern_dict.get('slope'):
        title += f"\nSlope: {pattern_dict['slope']:.4f}"

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = f"lrlr_pattern_{pattern_idx:03d}_{pattern_type.value}_{'buyside' if is_buyside else 'sellside'}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return filename, pattern_idx


def load_nq_data(data_path: Path) -> pl.DataFrame:
    """
    Load NQ CSV data using Polars for faster performance.

    Args:
        data_path: Path to NQ.csv

    Returns:
        Polars DataFrame with OHLC data
    """
    # Load with Polars (much faster than pandas)
    df = pl.read_csv(data_path)

    # Parse datetime and rename columns
    df = df.with_columns([
        pl.col("DateTime_ET").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    ])

    # Rename columns to lowercase for consistency
    df = df.rename({
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Open": "open"
    })

    return df


def plot_candlesticks(ax, data, width=0.6):
    """
    Plot candlestick chart on the given axes.

    Args:
        ax: Matplotlib axes object
        data: DataFrame with 'open', 'high', 'low', 'close' columns
        width: Width of candlestick body (default 0.6)
    """
    for i, (idx, row) in enumerate(data.iterrows()):
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']

        # Determine if bullish (green) or bearish (red)
        is_bullish = close_price >= open_price

        # Color scheme
        body_color = '#26a69a' if is_bullish else '#ef5350'  # Green for up, red for down
        wick_color = '#26a69a' if is_bullish else '#ef5350'

        # Draw wick (high-low line)
        ax.plot([i, i], [low_price, high_price],
               color=wick_color, linewidth=1.0, alpha=0.8)

        # Draw body (open-close rectangle)
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_height = body_top - body_bottom

        # If body height is 0, draw a horizontal line
        if body_height == 0:
            ax.plot([i - width/2, i + width/2], [open_price, open_price],
                   color=body_color, linewidth=2.0, alpha=0.8)
        else:
            rect = plt.Rectangle((i - width/2, body_bottom), width, body_height,
                               facecolor=body_color, edgecolor=body_color,
                               linewidth=1.0, alpha=0.8)
            ax.add_patch(rect)


def visualize_lrlr_pattern(df: pl.DataFrame, pattern, pattern_idx: int,
                          output_dir: Path = None, timeframe_minutes: int = 1):
    """
    Visualize a single LRLR pattern on a price chart.

    Args:
        df: Polars DataFrame with OHLC data
        pattern: LRLRPattern object to visualize
        pattern_idx: Index of the pattern (for filename)
        output_dir: Optional directory to save the plot
        timeframe_minutes: Timeframe in minutes (for weight calculation)
    """
    # Extract the relevant price data (window to visualize)
    start_idx = max(0, pattern.start_index - 20)  # Add some context before
    end_idx = min(len(df), pattern.end_index + 20)  # Add some context after

    # Calculate weight based on the last bar in visualization window
    current_bar = end_idx - 1
    weight = pattern.calculate_weight(current_bar, timeframe_minutes)

    # Calculate bars_ago for display (from end of visualization window)
    bars_ago = max(0, current_bar - pattern.end_index)

    # Convert to pandas for plotting (only the slice we need)
    plot_data = df[start_idx:end_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot candlesticks
    plot_candlesticks(ax, plot_data)

    # Set x-axis limits and ticks
    ax.set_xlim(-0.5, plot_data.height - 0.5)
    ax.set_xticks(range(0, plot_data.height, max(1, plot_data.height // 10)))
    ax.set_xticklabels([start_idx + i for i in range(0, plot_data.height, max(1, plot_data.height // 10))])

    # Highlight the pattern region
    pattern_start_plot = pattern.start_index - start_idx
    pattern_end_plot = pattern.end_index - start_idx

    # Color based on pattern type and side
    if pattern.is_buyside:
        color = 'red' if pattern.type == LRLRType.TRENDLINE else 'orange'
        label_prefix = 'Buyside'
    else:
        color = 'blue' if pattern.type == LRLRType.TRENDLINE else 'cyan'
        label_prefix = 'Sellside'

    # Draw the pattern
    if pattern.type == LRLRType.TRENDLINE:
        # Draw trendline through swing points
        swing_x = [sp.index - start_idx for sp in pattern.swing_points]
        swing_y = [sp.price for sp in pattern.swing_points]

        # Plot trendline
        ax.plot(swing_x, swing_y, color=color, linewidth=2.5,
                linestyle='--', alpha=0.8, label=f'{label_prefix} Trendline LRLR')

        # Mark swing points
        ax.scatter(swing_x, swing_y, color=color, s=100, zorder=5,
                  marker='o', edgecolors='white', linewidths=1.5)

        # Extend trendline slightly for visibility
        if len(swing_x) >= 2:
            x_extend = [swing_x[0] - 5, swing_x[-1] + 5]
            y_extend = [
                swing_y[0] + pattern.slope * (x_extend[0] - swing_x[0]),
                swing_y[-1] + pattern.slope * (x_extend[1] - swing_x[-1])
            ]
            ax.plot(x_extend, y_extend, color=color, linewidth=1.5,
                   linestyle=':', alpha=0.5)

    elif pattern.type in [LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]:
        # Draw horizontal line at equal level
        equal_price = pattern.start_price  # Both should be approximately equal

        # Draw horizontal line
        ax.axhline(y=equal_price, color=color, linewidth=2.5,
                  linestyle='--', alpha=0.8,
                  label=f'{label_prefix} Equal {"Highs" if pattern.type == LRLRType.EQUAL_HIGHS else "Lows"} LRLR')

        # Mark the two swing points
        for sp in pattern.swing_points:
            sp_x = sp.index - start_idx
            ax.scatter(sp_x, sp.price, color=color, s=120, zorder=5,
                      marker='s', edgecolors='white', linewidths=1.5)

        # Add strength annotation
        strength_text = {
            EqualStrength.SWING_TICK_EQUAL: "Swing Tick Equal (Strongest) ✓✓",
            EqualStrength.TICK_SWING: "Tick Swing - Adjacent (Very Strong) ✓",
            EqualStrength.TICK_EQUAL: "Tick Equal - Separated (Strong)",
            EqualStrength.OLDER_HIGHER: "Older Higher - Relative (Moderate)"
        }.get(pattern.strength, "Unknown")

        # Add text annotation with bar count
        bars_between = pattern.end_index - pattern.start_index
        annotation_text = f"{strength_text}\n{bars_between} bars apart"

        mid_x = (pattern.start_index + pattern.end_index) / 2 - start_idx
        y_offset = (plot_data['high'].max() - plot_data['low'].min()) * (0.05 if pattern.is_buyside else -0.05)
        ax.annotate(annotation_text,
                   xy=(mid_x, equal_price),
                   xytext=(mid_x, equal_price + y_offset),
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5))

    # Highlight the pattern region with a shaded area
    ax.axvspan(pattern_start_plot, pattern_end_plot,
              alpha=0.1, color=color, label='Pattern Region')

    # Formatting
    ax.set_xlabel('Bar Index (relative)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # Create title with pattern details
    pattern_type_str = {
        LRLRType.TRENDLINE: "Trendline",
        LRLRType.EQUAL_HIGHS: "Equal Highs",
        LRLRType.EQUAL_LOWS: "Equal Lows"
    }.get(pattern.type, "Unknown")

    side_str = "Buyside (Resistance)" if pattern.is_buyside else "Sellside (Support)"

    title = f"LRLR Pattern #{pattern_idx}: {pattern_type_str} - {side_str}\n"
    title += f"Weight: {weight:.3f} | Bars Ago: {bars_ago} | "
    title += f"Bars {pattern.start_index} to {pattern.end_index} | "
    title += f"Price: ${pattern.start_price:.2f} to ${pattern.end_price:.2f}"

    if pattern.type == LRLRType.TRENDLINE and pattern.slope:
        title += f"\nSlope: {pattern.slope:.4f}"

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"lrlr_pattern_{pattern_idx:03d}_{pattern.type.value}_{'buyside' if pattern.is_buyside else 'sellside'}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")

    plt.close()


def test_lrlr_detection(max_visualizations: Optional[int] = 10,
                        output_dir: Optional[Path] = None,
                        create_viz: bool = True,
                        n_jobs: int = -1,
                        allowed_flags: Dict[str, bool] = None):
    """
    Test LRLR detection on NQ data.

    Args:
        max_visualizations: Max number of visualizations to create (None = all)
        output_dir: Directory to save visualizations
        create_viz: Whether to create visualizations at all
    """
    print("=" * 80)
    print("LRLR Detection Test")
    print("=" * 80)

    # Load data with Polars (much faster)
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)

    print(f"\nLoaded {len(df)} bars")
    print(f"Date range: {df['DateTime_ET'].min()} to {df['DateTime_ET'].max()}")
    print(f"Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

    # Resolve allowed flags
    if allowed_flags is None:
        allowed_flags = {"trendline": True, "equal": True, "tick_equal": True}
    enabled_labels = []
    if allowed_flags.get("trendline"):
        enabled_labels.append("trendline")
    if allowed_flags.get("equal"):
        enabled_labels.append("equal (highs+lows)")
    if allowed_flags.get("tick_equal"):
        enabled_labels.append("tick_equal")
    print(f"\nIncluded pattern types: {', '.join(enabled_labels)}")

    # Convert to numpy arrays (Polars is very fast at this)
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()

    # Initialize detector with current defaults from lrlr.py
    detector = LRLRDetector(
        swing_lookback=3,
        min_trendline_touches=3,
        trendline_tolerance_pct=0.0005,  # 0.05%
        equal_tolerance_pct=0.0001      # 0.01%
    )

    print(f"\nDetector Configuration:")
    print(f"  - Swing lookback: {detector.swing_lookback} bars")
    print(f"  - Min trendline touches: {detector.min_touches}")
    print(f"  - Trendline tolerance: {detector.trendline_tolerance * 100:.2f}%")
    print(f"  - Equal tolerance: {detector.equal_tolerance * 100:.2f}%")

    # Detect patterns
    print(f"\nDetecting LRLR patterns...")
    results = detector.detect(highs, lows, closes)

    buyside_patterns = filter_patterns_by_type(results['buyside'], allowed_flags)
    sellside_patterns = filter_patterns_by_type(results['sellside'], allowed_flags)

    print(f"\nDetection Results:")
    print(f"  - Buyside patterns: {len(buyside_patterns)}")
    print(f"  - Sellside patterns: {len(sellside_patterns)}")
    print(f"  - Total patterns: {len(buyside_patterns) + len(sellside_patterns)}")

    # Breakdown by type
    buyside_trendlines = [p for p in buyside_patterns if p.type == LRLRType.TRENDLINE]
    buyside_equals = [p for p in buyside_patterns if p.type in [LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]]
    sellside_trendlines = [p for p in sellside_patterns if p.type == LRLRType.TRENDLINE]
    sellside_equals = [p for p in sellside_patterns if p.type in [LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]]

    # Further breakdown by equal strength
    buyside_swing_tick = [p for p in buyside_equals if p.strength == EqualStrength.SWING_TICK_EQUAL]
    buyside_tick_swing = [p for p in buyside_equals if p.strength == EqualStrength.TICK_SWING]
    buyside_tick_equal = [p for p in buyside_equals if p.strength == EqualStrength.TICK_EQUAL]
    buyside_older_higher = [p for p in buyside_equals if p.strength == EqualStrength.OLDER_HIGHER]

    sellside_swing_tick = [p for p in sellside_equals if p.strength == EqualStrength.SWING_TICK_EQUAL]
    sellside_tick_swing = [p for p in sellside_equals if p.strength == EqualStrength.TICK_SWING]
    sellside_tick_equal = [p for p in sellside_equals if p.strength == EqualStrength.TICK_EQUAL]
    sellside_older_higher = [p for p in sellside_equals if p.strength == EqualStrength.OLDER_HIGHER]

    print(f"\nPattern Breakdown:")
    print(f"  Buyside:")
    print(f"    - Trendlines: {len(buyside_trendlines)}")
    print(f"    - Equal highs (total): {len(buyside_equals)}")
    print(f"      • Swing tick equal (separated, strongest): {len(buyside_swing_tick)}")
    print(f"      • Tick swing (adjacent, very strong): {len(buyside_tick_swing)}")
    print(f"      • Tick equal (separated, strong): {len(buyside_tick_equal)}")
    print(f"      • Older higher (relative, moderate): {len(buyside_older_higher)}")
    print(f"  Sellside:")
    print(f"    - Trendlines: {len(sellside_trendlines)}")
    print(f"    - Equal lows (total): {len(sellside_equals)}")
    print(f"      • Swing tick equal (separated, strongest): {len(sellside_swing_tick)}")
    print(f"      • Tick swing (adjacent, very strong): {len(sellside_tick_swing)}")
    print(f"      • Tick equal (separated, strong): {len(sellside_tick_equal)}")
    print(f"      • Older higher (relative, moderate): {len(sellside_older_higher)}")

    # Show details of first few patterns
    print(f"\nSample Patterns (first 5):")
    all_patterns = buyside_patterns + sellside_patterns
    current_bar = len(df) - 1
    timeframe_minutes = 1  # NQ 1-minute data

    for i, pattern in enumerate(all_patterns[:5], 1):
        pattern_type_str = {
            LRLRType.TRENDLINE: "Trendline",
            LRLRType.EQUAL_HIGHS: "Equal Highs",
            LRLRType.EQUAL_LOWS: "Equal Lows"
        }.get(pattern.type, "Unknown")

        side_str = "Buyside" if pattern.is_buyside else "Sellside"
        weight = pattern.calculate_weight(current_bar, timeframe_minutes)
        bars_ago = max(0, current_bar - pattern.end_index)

        print(f"\n  Pattern {i}: {pattern_type_str} - {side_str} | Weight: {weight:.3f} | Bars Ago: {bars_ago}")
        print(f"    Bars: {pattern.start_index} to {pattern.end_index} ({pattern.end_index - pattern.start_index + 1} bars)")
        print(f"    Price: ${pattern.start_price:.2f} to ${pattern.end_price:.2f}")
        print(f"    Swing points: {len(pattern.swing_points)}")

        if pattern.type == LRLRType.TRENDLINE and pattern.slope:
            print(f"    Slope: {pattern.slope:.4f}")
        elif pattern.strength:
            strength_str = {
                EqualStrength.SWING_TICK_EQUAL: "Swing Tick Equal (Strongest)",
                EqualStrength.TICK_SWING: "Tick Swing - Adjacent (Very Strong)",
                EqualStrength.TICK_EQUAL: "Tick Equal - Separated (Strong)",
                EqualStrength.OLDER_HIGHER: "Older Higher - Relative (Moderate)"
            }.get(pattern.strength, "Unknown")
            print(f"    Strength: {strength_str}")

    # Create visualizations
    if create_viz:
        print(f"\n" + "=" * 80)
        print("Creating Visualizations (Parallel)")
        print("=" * 80)

        if output_dir is None:
            output_dir = Path(__file__).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)

        # Determine how many to create
        num_to_create = len(all_patterns)
        if max_visualizations is not None:
            num_to_create = min(max_visualizations, len(all_patterns))

        # Determine number of jobs
        if n_jobs == -1:
            import os
            n_workers = os.cpu_count()
        else:
            n_workers = min(n_jobs, num_to_create)

        print(f"\nCreating {num_to_create} of {len(all_patterns)} visualizations")
        print(f"Using {n_workers} parallel workers (cores)")
        print(f"Saving to: {output_dir}")

        # Prepare data for parallel processing
        viz_jobs = []
        for i, pattern in enumerate(all_patterns[:num_to_create], 1):
            start_idx = max(0, pattern.start_index - 20)
            end_idx = min(len(df), pattern.end_index + 20)

            # Convert slice to dict
            plot_data = df[start_idx:end_idx]
            df_dict = plot_data.to_dict(as_series=False)

            # Convert pattern to dict for serialization
            pattern_dict = {
                'type': pattern.type.value,
                'start_index': pattern.start_index,
                'end_index': pattern.end_index,
                'start_price': pattern.start_price,
                'end_price': pattern.end_price,
                'is_buyside': pattern.is_buyside,
                'swing_points': [{'index': sp.index, 'price': sp.price}
                                for sp in pattern.swing_points],
                'slope': pattern.slope if hasattr(pattern, 'slope') else None,
                'strength': pattern.strength.value if pattern.strength else None
            }

            viz_jobs.append((df_dict, pattern_dict, i, output_dir, start_idx, end_idx))

        # Create visualizations in parallel using joblib
        start_time = time.time()
        timeframe_minutes = 1  # NQ 1-minute data

        print(f"\n🚀 Starting parallel processing...")
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(create_single_visualization)(
                df_dict, pattern_dict, idx, output_dir, start_idx, end_idx, timeframe_minutes
            ) for df_dict, pattern_dict, idx, output_dir, start_idx, end_idx in viz_jobs
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Created {num_to_create} visualizations in {elapsed:.1f}s "
              f"({num_to_create/elapsed:.1f} charts/sec)")
        print(f"   Output directory: {output_dir}")
        print(f"   🚀 Parallel processing with {n_workers} CPU cores")
        print(f"   ⚡ Speed-up: ~{n_workers:.0f}x vs single-threaded")

        if num_to_create < len(all_patterns):
            print(f"\n   ℹ️  Skipped {len(all_patterns) - num_to_create} visualizations")
            print(f"   Use --viz-all to create all {len(all_patterns)} visualizations")
    else:
        print(f"\n⏭️  Skipping visualizations (use --max-viz N to create N visualizations)")

    return results, df


def test_tick_equal_detection(allowed_flags: Dict[str, bool] = None):
    """
    Test tick equal highs/lows detection specifically.
    """
    print("\n" + "=" * 80)
    print("Tick Equal Detection Test")
    print("=" * 80)

    if allowed_flags is None:
        allowed_flags = {"trendline": True, "equal": True, "tick_equal": True}

    if not allowed_flags.get("tick_equal", True):
        print("⏭️  Skipping tick equal detection (tick_equal not requested)")
        return None

    # Load data with Polars
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)

    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()

    # Initialize detector with current defaults
    detector = LRLRDetector(
        swing_lookback=5,
        min_trendline_touches=3,
        trendline_tolerance_pct=0.005,
        equal_tolerance_pct=0.001
    )

    print(f"\nSearching for tick equal patterns in {len(df)} bars...")

    # Detect patterns
    results = detector.detect(highs, lows, closes)

    # Filter according to flags, then select all tick-based patterns
    filtered_buyside = filter_patterns_by_type(results['buyside'], allowed_flags)
    filtered_sellside = filter_patterns_by_type(results['sellside'], allowed_flags)

    tick_based_buyside = [p for p in filtered_buyside
                          if p.strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL)]
    tick_based_sellside = [p for p in filtered_sellside
                           if p.strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL)]

    print(f"\nTick-Based Pattern Results:")
    print(f"  - Buyside (equal highs): {len(tick_based_buyside)}")
    print(f"  - Sellside (equal lows): {len(tick_based_sellside)}")
    print(f"  - Total tick-based: {len(tick_based_buyside) + len(tick_based_sellside)}")

    # Show details of first few tick-based patterns
    print(f"\nSample Tick-Based Patterns (first 3 of each side):")

    print(f"\n  Buyside Tick-Based Highs:")
    for i, pattern in enumerate(tick_based_buyside[:3], 1):
        bars_between = pattern.end_index - pattern.start_index
        strength_name = pattern.strength.value if pattern.strength else "unknown"
        print(f"    {i}. {strength_name} | Price: ${pattern.start_price:.2f} | "
              f"Bars: {pattern.start_index} to {pattern.end_index} ({bars_between} bars apart)")

    print(f"\n  Sellside Tick-Based Lows:")
    for i, pattern in enumerate(tick_based_sellside[:3], 1):
        bars_between = pattern.end_index - pattern.start_index
        strength_name = pattern.strength.value if pattern.strength else "unknown"
        print(f"    {i}. {strength_name} | Price: ${pattern.start_price:.2f} | "
              f"Bars: {pattern.start_index} to {pattern.end_index} ({bars_between} bars apart)")

    # Analyze distribution of bars between tick-based patterns
    if tick_based_buyside or tick_based_sellside:
        all_tick_based = tick_based_buyside + tick_based_sellside
        bars_between = [p.end_index - p.start_index for p in all_tick_based]

        print(f"\nTick-Based Pattern Statistics:")
        print(f"  - Min bars between: {min(bars_between)}")
        print(f"  - Max bars between: {max(bars_between)}")
        print(f"  - Average bars between: {np.mean(bars_between):.1f}")
        print(f"  - Median bars between: {np.median(bars_between):.1f}")

    return results


def test_pattern_quality_analysis():
    """
    Analyze the quality and distribution of detected patterns.
    """
    print("\n" + "=" * 80)
    print("Pattern Quality Analysis")
    print("=" * 80)

    # Load data with Polars
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)

    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()

    # Initialize detector with current defaults
    detector = LRLRDetector(
        swing_lookback=5,
        min_trendline_touches=3,
        trendline_tolerance_pct=0.005,
        equal_tolerance_pct=0.001
    )

    results = detector.detect(highs, lows, closes)
    all_patterns = results['buyside'] + results['sellside']

    print(f"\nAnalyzing {len(all_patterns)} total patterns...")

    # Analyze by strength (for equal patterns)
    equal_patterns = [p for p in all_patterns if p.type in [LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]]

    if equal_patterns:
        print(f"\nEqual Patterns Strength Distribution:")
        swing_tick_equal = [p for p in equal_patterns if p.strength == EqualStrength.SWING_TICK_EQUAL]
        tick_swing = [p for p in equal_patterns if p.strength == EqualStrength.TICK_SWING]
        tick_equal = [p for p in equal_patterns if p.strength == EqualStrength.TICK_EQUAL]
        older_higher = [p for p in equal_patterns if p.strength == EqualStrength.OLDER_HIGHER]

        total_equal = len(equal_patterns)
        print(f"  - Swing Tick Equal: {len(swing_tick_equal)} ({len(swing_tick_equal)/total_equal*100:.1f}%)")
        print(f"  - Tick Swing (adjacent): {len(tick_swing)} ({len(tick_swing)/total_equal*100:.1f}%)")
        print(f"  - Tick Equal: {len(tick_equal)} ({len(tick_equal)/total_equal*100:.1f}%)")
        print(f"  - Older Higher: {len(older_higher)} ({len(older_higher)/total_equal*100:.1f}%)")

    # Analyze trendline patterns
    trendline_patterns = [p for p in all_patterns if p.type == LRLRType.TRENDLINE]

    if trendline_patterns:
        print(f"\nTrendline Patterns Analysis:")
        touches = [len(p.swing_points) for p in trendline_patterns]
        slopes = [abs(p.slope) for p in trendline_patterns if p.slope]

        print(f"  - Average touches: {np.mean(touches):.1f}")
        print(f"  - Max touches: {max(touches)}")
        print(f"  - Average slope (abs): {np.mean(slopes):.6f}" if slopes else "  - No slopes calculated")

    # Analyze pattern duration
    durations = [p.end_index - p.start_index for p in all_patterns]

    print(f"\nPattern Duration Analysis:")
    print(f"  - Min duration: {min(durations)} bars")
    print(f"  - Max duration: {max(durations)} bars")
    print(f"  - Average duration: {np.mean(durations):.1f} bars")
    print(f"  - Median duration: {np.median(durations):.1f} bars")

    # Find longest patterns of each type
    print(f"\nLongest Patterns by Type:")
    for pattern_type in [LRLRType.TRENDLINE, LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]:
        type_patterns = [p for p in all_patterns if p.type == pattern_type]
        if type_patterns:
            longest = max(type_patterns, key=lambda p: p.end_index - p.start_index)
            duration = longest.end_index - longest.start_index
            side = "Buyside" if longest.is_buyside else "Sellside"
            print(f"  - {pattern_type.value}: {duration} bars ({side})")


def test_custom_parameters():
    """
    Test LRLR detection with different parameter settings.
    """
    print("\n" + "=" * 80)
    print("Custom Parameters Test")
    print("=" * 80)

    # Load data with Polars
    data_path = Path(__file__).parent / "NQ.csv"
    df = load_nq_data(data_path)

    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()

    # Test with different parameters
    test_configs = [
        {
            "name": "Strict (fewer, higher quality patterns)",
            "swing_lookback": 7,
            "min_trendline_touches": 3,
            "trendline_tolerance_pct": 0.003,  # 0.3% (tighter than default 0.5%)
            "equal_tolerance_pct": 0.0005     # 0.05% (tighter than default 0.1%)
        },
        {
            "name": "Relaxed (more patterns, lower quality threshold)",
            "swing_lookback": 3,
            "min_trendline_touches": 3,
            "trendline_tolerance_pct": 0.008,  # 0.8% (looser than default)
            "equal_tolerance_pct": 0.002      # 0.2% (looser than default)
        },
    ]

    for config in test_configs:
        print(f"\n{config['name']}:")
        print(f"  Swing lookback: {config['swing_lookback']}")
        print(f"  Min touches: {config['min_trendline_touches']}")
        print(f"  Trendline tolerance: {config['trendline_tolerance_pct'] * 100:.2f}%")
        print(f"  Equal tolerance: {config['equal_tolerance_pct'] * 100:.2f}%")

        detector = LRLRDetector(**{k: v for k, v in config.items() if k != 'name'})
        results = detector.detect(highs, lows, closes)

        # Count by type
        trendlines = len([p for p in results['buyside'] + results['sellside']
                         if p.type == LRLRType.TRENDLINE])
        equals = len([p for p in results['buyside'] + results['sellside']
                     if p.type in [LRLRType.EQUAL_HIGHS, LRLRType.EQUAL_LOWS]])
        tick_based = len([p for p in results['buyside'] + results['sellside']
                         if p.strength in (EqualStrength.SWING_TICK_EQUAL, EqualStrength.TICK_SWING, EqualStrength.TICK_EQUAL)])

        total = len(results['buyside']) + len(results['sellside'])
        print(f"  Total patterns detected: {total}")
        print(f"    - Buyside: {len(results['buyside'])}")
        print(f"    - Sellside: {len(results['sellside'])}")
        print(f"    - Trendlines: {trendlines}")
        print(f"    - Equals (all): {equals}")
        print(f"    - Tick-based: {tick_based}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Test LRLR detection with configurable visualization options"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip all visualizations"
    )
    parser.add_argument(
        "--max-viz",
        type=int,
        default=10,
        help="Maximum number of visualizations to create (default: 10)"
    )
    parser.add_argument(
        "--viz-all",
        action="store_true",
        help="Create all visualizations (overrides --max-viz)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for visualizations (default: ./visualizations)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for visualization (-1 = all CPU cores, default: -1)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["all", "trendline", "equal", "equal_highs", "equal_lows", "tick_equal"],
        default=["all"],
        help="Pattern types: trendline, equal (highs+lows), tick_equal, or all."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Determine visualization settings
    if args.no_viz:
        create_viz = False
        max_viz = None
    elif args.viz_all:
        create_viz = True
        max_viz = None
    else:
        create_viz = True
        max_viz = args.max_viz

    print("=" * 80)
    print("Low Resistance Liquidity (LRLR) Test Suite")
    print("=" * 80)

    if not create_viz:
        print("\n⚡ Running in fast mode (no visualizations)")
    elif max_viz is not None:
        print(f"\n⚡ Limited visualization mode (max {max_viz} charts)")
    else:
        print(f"\n🎨 Full visualization mode (all charts will be created)")

    print()

    # Main test with visualization control
    allowed_flags = resolve_allowed_types(args.types)
    results, df = test_lrlr_detection(
        max_visualizations=max_viz,
        output_dir=args.output_dir,
        create_viz=create_viz,
        n_jobs=args.jobs,
        allowed_flags=allowed_flags
    )

    # Test tick equal detection specifically
    test_tick_equal_detection(allowed_flags)

    # Analyze pattern quality
    test_pattern_quality_analysis()

    # Test with custom parameters
    test_custom_parameters()

    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80)

    if create_viz:
        output_dir = args.output_dir or (Path(__file__).parent / "visualizations")
        print(f"\n✅ Visualizations saved to: {output_dir}")
    else:
        print(f"\n⚡ Completed in fast mode (no visualizations created)")

