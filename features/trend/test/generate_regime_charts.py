"""
Generate 20 visual chart samples showing detected market regimes.

Scans NQ 1-minute PM sessions, runs the StateDetector on each,
selects 20 diverse sessions across regimes, and produces annotated
candlestick charts saved as PNGs.
"""

import sys
from pathlib import Path

# Ensure trend package is importable
TREND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREND_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import mplfinance as mpf
from datetime import datetime

import state_detector as sd

# ============================================================================
# CONFIG
# ============================================================================

PARQUET_PATH = TREND_DIR.parent.parent / 'outputs' / 'nq_1m.parquet'
OUTPUT_DIR = Path(__file__).resolve().parent / 'regime_charts'
N_CHARTS = 20

# Regime color scheme
REGIME_COLORS = {
    'STRONG_TREND':   '#1a9641',  # dark green
    'WEAK_TREND':     '#a6d96a',  # light green
    'CONSOLIDATION':  '#fdae61',  # orange
    'CHOPPY':         '#d7191c',  # red
    'UNCERTAIN':      '#999999',  # gray
}

REGIME_BG_COLORS = {
    'STRONG_TREND':   '#e6f5e1',
    'WEAK_TREND':     '#f0f9e8',
    'CONSOLIDATION':  '#fff3e0',
    'CHOPPY':         '#fde0dc',
    'UNCERTAIN':      '#f0f0f0',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pm_sessions(parquet_path: Path) -> dict:
    """Load parquet and group into PM (1pm-3pm) sessions by date."""
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={'datetime_et': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to PM session (1pm-3pm ET)
    mask = (df['timestamp'].dt.hour >= 13) & (df['timestamp'].dt.hour < 15)
    pm = df[mask].copy()

    # Group by date
    pm['date'] = pm['timestamp'].dt.date
    sessions = {}
    for date, group in pm.groupby('date'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        if len(group) >= 60:  # need at least 60 bars for meaningful analysis
            sessions[date] = group

    print(f"Found {len(sessions)} valid PM sessions")
    return sessions


# ============================================================================
# REGIME SCANNING
# ============================================================================

def scan_regimes(sessions: dict) -> pd.DataFrame:
    """Run StateDetector on all sessions, return results DataFrame."""
    print("Scanning regimes across all sessions...")
    detector = sd.StateDetector(dynamic_weights=True)
    records = []

    for i, (date, session_df) in enumerate(sessions.items()):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(sessions)}...")
        try:
            result = detector.detect(session_df, session='1pm-3pm')
            hurst = None
            dw_info = result.metadata.get('dynamic_weighting', {})
            if dw_info:
                hurst = dw_info.get('hurst_raw')

            records.append({
                'date': date,
                'state': result.state,
                'direction': result.direction,
                'confidence': result.confidence,
                'n_bars': len(session_df),
                'n_signals': len(result.signals),
                'hurst': hurst,
                'price_change_pct': (
                    (session_df['close'].iloc[-1] - session_df['close'].iloc[0])
                    / session_df['close'].iloc[0] * 100
                ),
                'warnings': len(result.warnings),
            })
        except Exception as e:
            print(f"  ERROR on {date}: {e}")
            continue

    scan_df = pd.DataFrame(records)
    print(f"\nRegime distribution:")
    print(scan_df['state'].value_counts().to_string())
    print(f"\nDirection distribution:")
    print(scan_df['direction'].value_counts().to_string())
    return scan_df


def select_diverse_sessions(scan_df: pd.DataFrame, n: int = 20) -> list:
    """
    Select N diverse sessions spanning different regimes, confidences, and dates.
    Prioritizes getting representation from each regime.
    """
    selected_dates = []

    # Target allocation per regime (proportional but ensuring at least 1 per state found)
    states = scan_df['state'].unique()
    per_state = max(2, n // len(states))
    remainder = n

    for state in ['STRONG_TREND', 'WEAK_TREND', 'CONSOLIDATION', 'CHOPPY', 'UNCERTAIN']:
        state_df = scan_df[scan_df['state'] == state].copy()
        if state_df.empty:
            continue

        # Pick samples spread across confidence range and time
        state_df = state_df.sort_values('confidence', ascending=False)
        take = min(per_state, len(state_df), remainder)
        if take <= 0:
            continue

        # Spread evenly through the sorted list
        indices = np.linspace(0, len(state_df) - 1, take, dtype=int)
        picks = state_df.iloc[indices]
        selected_dates.extend(picks['date'].tolist())
        remainder -= take

    # Fill remainder with highest-confidence diverse samples
    if remainder > 0:
        remaining = scan_df[~scan_df['date'].isin(selected_dates)]
        remaining = remaining.sort_values('confidence', ascending=False)
        extra = remaining.head(remainder)
        selected_dates.extend(extra['date'].tolist())

    # Deduplicate and limit
    seen = set()
    unique = []
    for d in selected_dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique[:n]


# ============================================================================
# CHART GENERATION
# ============================================================================

def draw_regime_chart(
    session_df: pd.DataFrame,
    result: sd.StateResult,
    date,
    output_path: Path,
):
    """Draw an annotated candlestick chart for one session."""
    # Prepare data for mplfinance
    ohlc = session_df.copy()
    ohlc = ohlc.set_index('timestamp')
    ohlc = ohlc.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
    })
    ohlc = ohlc[['Open', 'High', 'Low', 'Close', 'Volume']]
    ohlc.index = pd.DatetimeIndex(ohlc.index)

    # Colors
    regime = result.state
    regime_color = REGIME_COLORS.get(regime, '#999999')
    bg_color = REGIME_BG_COLORS.get(regime, '#f0f0f0')

    # Build signal bar data for subplot
    signal_names = list(result.signals.keys())
    signal_vals = [result.signals[k] for k in signal_names]
    weight_vals = [result.weights.get(k, 0) for k in signal_names]

    # Dynamic weighting info
    dw = result.metadata.get('dynamic_weighting', {})
    hurst_raw = dw.get('hurst_raw', None)
    adjustment = dw.get('adjustment', None)

    # --- Create figure with gridspec ---
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[3, 1, 0.05],
        width_ratios=[3, 1],
        hspace=0.35, wspace=0.3,
        left=0.06, right=0.96, top=0.88, bottom=0.06,
    )

    ax_candle = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_candle)
    ax_signals = fig.add_subplot(gs[0, 1])
    ax_weights = fig.add_subplot(gs[1, 1])

    # --- Title ---
    date_str = str(date)
    dir_arrow = {'UP': '\u2191', 'DOWN': '\u2193', 'NEUTRAL': '\u2194'}
    arrow = dir_arrow.get(result.direction, '')
    pct = ((ohlc['Close'].iloc[-1] - ohlc['Close'].iloc[0]) / ohlc['Close'].iloc[0]) * 100

    fig.suptitle(
        f"NQ  {date_str}  PM Session (1:00-3:00 ET)",
        fontsize=14, fontweight='bold', y=0.97,
    )

    # Regime badge
    hurst_str = f"{hurst_raw:.3f}" if hurst_raw is not None else "N/A"
    fig.text(
        0.5, 0.925,
        f"  {regime}  {arrow}  Confidence: {result.confidence:.1%}  |  "
        f"Hurst: {hurst_str}  |  "
        f"Move: {pct:+.2f}%  ",
        ha='center', fontsize=12, fontweight='bold',
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor=regime_color,
            edgecolor='none',
            alpha=0.9,
        ),
    )

    # --- Candlestick chart ---
    up_color = '#26a69a'
    down_color = '#ef5350'

    for idx in range(len(ohlc)):
        row = ohlc.iloc[idx]
        t = mdates.date2num(ohlc.index[idx])
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        color = up_color if c >= o else down_color
        # Wick
        ax_candle.plot([t, t], [l, h], color=color, linewidth=0.6)
        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.01:
            body_height = 0.25
        rect = plt.Rectangle(
            (t - 0.0003, body_bottom), 0.0006, body_height,
            facecolor=color, edgecolor=color, linewidth=0.3,
        )
        ax_candle.add_patch(rect)

    ax_candle.set_xlim(
        mdates.date2num(ohlc.index[0]) - 0.001,
        mdates.date2num(ohlc.index[-1]) + 0.001,
    )
    ax_candle.set_ylim(ohlc['Low'].min() - 2, ohlc['High'].max() + 2)
    ax_candle.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_candle.set_ylabel('Price', fontsize=9)
    ax_candle.set_facecolor(bg_color)
    ax_candle.grid(True, alpha=0.3, linestyle='--')
    ax_candle.tick_params(labelsize=8)
    plt.setp(ax_candle.get_xticklabels(), visible=False)

    # --- Volume ---
    colors_vol = [up_color if ohlc['Close'].iloc[i] >= ohlc['Open'].iloc[i]
                  else down_color for i in range(len(ohlc))]
    ax_vol.bar(
        [mdates.date2num(t) for t in ohlc.index],
        ohlc['Volume'].values,
        width=0.0005, color=colors_vol, alpha=0.7,
    )
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_vol.set_ylabel('Volume', fontsize=9)
    ax_vol.set_facecolor(bg_color)
    ax_vol.grid(True, alpha=0.3, linestyle='--')
    ax_vol.tick_params(labelsize=8)
    for label in ax_vol.get_xticklabels():
        label.set_rotation(30)

    # --- Signals bar chart ---
    bar_colors = []
    for s in signal_names:
        d = sd.INDICATOR_DIRECTION.get(s, 0)
        if d == 1:
            bar_colors.append('#4caf50')
        elif d == -1:
            bar_colors.append('#ff9800')
        else:
            bar_colors.append('#2196f3')

    y_pos = np.arange(len(signal_names))
    bars = ax_signals.barh(y_pos, signal_vals, color=bar_colors, alpha=0.8, height=0.6)
    ax_signals.set_yticks(y_pos)
    ax_signals.set_yticklabels([s.upper() for s in signal_names], fontsize=8)
    ax_signals.set_xlim(0, 1.05)
    ax_signals.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_signals.set_xlabel('Signal (0=consolidating, 1=trending)', fontsize=8)
    ax_signals.set_title('Indicator Signals', fontsize=10, fontweight='bold')
    ax_signals.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax_signals.tick_params(labelsize=8)

    # Value labels on bars
    for bar, val in zip(bars, signal_vals):
        ax_signals.text(
            min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=7, color='#333',
        )

    # Legend for signal categories
    legend_elements = [
        mpatches.Patch(facecolor='#4caf50', alpha=0.8, label='Trend (+1)'),
        mpatches.Patch(facecolor='#ff9800', alpha=0.8, label='Consol (-1)'),
        mpatches.Patch(facecolor='#2196f3', alpha=0.8, label='Neutral (0)'),
    ]
    ax_signals.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # --- Weights bar chart ---
    bars_w = ax_weights.barh(y_pos, weight_vals, color=bar_colors, alpha=0.5, height=0.6)
    # Overlay default weights for comparison
    default_w = [sd.StateDetector.DEFAULT_WEIGHTS.get(s, 0) for s in signal_names]
    ax_weights.barh(
        y_pos, default_w, color='none', edgecolor='black',
        linewidth=1.0, linestyle='--', height=0.6, alpha=0.6,
    )
    ax_weights.set_yticks(y_pos)
    ax_weights.set_yticklabels([s.upper() for s in signal_names], fontsize=8)
    ax_weights.set_xlim(0, max(max(weight_vals), max(default_w)) * 1.3)
    ax_weights.set_xlabel('Weight', fontsize=8)
    title_suffix = ''
    if adjustment is not None:
        title_suffix = f' (adj={adjustment:+.3f})'
    ax_weights.set_title(f'Weights{title_suffix}', fontsize=10, fontweight='bold')
    ax_weights.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax_weights.tick_params(labelsize=8)

    legend_w = [
        mpatches.Patch(facecolor='gray', alpha=0.5, label='Dynamic'),
        mpatches.Patch(facecolor='none', edgecolor='black', linestyle='--', label='Default'),
    ]
    ax_weights.legend(handles=legend_w, fontsize=7, loc='lower right')

    # --- Save ---
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    sessions = load_pm_sessions(PARQUET_PATH)

    # Scan all sessions for regime classification
    scan_df = scan_regimes(sessions)

    # Select 20 diverse sessions
    selected_dates = select_diverse_sessions(scan_df, N_CHARTS)
    print(f"\nSelected {len(selected_dates)} sessions for charting:")

    # Get regime info for selected dates
    selected_info = scan_df[scan_df['date'].isin(selected_dates)]
    print(selected_info[['date', 'state', 'direction', 'confidence', 'hurst']].to_string(index=False))

    # Generate charts
    detector = sd.StateDetector(dynamic_weights=True)
    for i, date in enumerate(sorted(selected_dates)):
        session_df = sessions[date]
        result = detector.detect(session_df, session='1pm-3pm')

        regime_tag = result.state.lower().replace('_', '-')
        filename = f"{i+1:02d}_{date}_{regime_tag}.png"
        output_path = OUTPUT_DIR / filename

        print(f"  [{i+1:2d}/{N_CHARTS}] {date} -> {result.state} ({result.confidence:.1%}) -> {filename}")
        draw_regime_chart(session_df, result, date, output_path)

    print(f"\nDone! Charts saved to: {OUTPUT_DIR}")

    # Summary index
    summary_path = OUTPUT_DIR / 'index.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Regime Detection Chart Samples\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Data: NQ 1-min, PM sessions (1pm-3pm ET)\n")
        f.write(f"Sessions scanned: {len(scan_df)}\n")
        f.write(f"Charts generated: {N_CHARTS}\n\n")
        f.write(f"Regime distribution (full scan):\n")
        f.write(scan_df['state'].value_counts().to_string())
        f.write(f"\n\nSelected samples:\n")
        f.write(f"{'-'*60}\n")
        for i, date in enumerate(sorted(selected_dates)):
            row = scan_df[scan_df['date'] == date].iloc[0]
            h_str = f"{row['hurst']:.3f}" if pd.notna(row['hurst']) else "  N/A"
            f.write(
                f"{i+1:2d}. {date}  {row['state']:16s}  "
                f"dir={row['direction']:7s}  conf={row['confidence']:.3f}  "
                f"hurst={h_str:>6}  "
                f"chg={row['price_change_pct']:+.2f}%\n"
            )
    print(f"Summary index: {summary_path}")


if __name__ == '__main__':
    main()
