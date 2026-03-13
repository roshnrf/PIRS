"""
PIRS V2 - LAYER 3: BEHAVIORAL DRIFT DETECTION
===============================================
Detects whether a user's deviation is GROWING over a rolling window.

The key insight: a single anomalous day could be noise.
A rising SLOPE of deviation over 14 days = genuine behavioral drift.

For each user on each day, computes:
  - drift_score:  rolling mean of deviation_score (smoothed signal)
  - drift_slope:  linear regression slope over last DRIFT_WINDOW days
                  positive slope = escalating behavior
  - drift_accel:  change in slope (is the drift accelerating?)
  - drift_label:  STABLE / LOW / MODERATE / HIGH / CRITICAL

This is the core of the "7-14 day early warning" claim.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()

DRIFT_LABELS = {
    (0.00, 0.05): 'STABLE',
    (0.05, 0.15): 'LOW',
    (0.15, 0.30): 'MODERATE',
    (0.30, 0.60): 'HIGH',
    (0.60, 9999): 'CRITICAL',
}


def label_drift(slope):
    for (lo, hi), label in DRIFT_LABELS.items():
        if lo <= slope < hi:
            return label
    return 'STABLE'


def compute_drift_for_user(user_df: pd.DataFrame) -> pd.DataFrame:
    """Compute drift metrics for a single user's time series."""
    user_df = user_df.sort_values('day').copy()
    n = len(user_df)

    drift_scores = []
    drift_slopes = []
    drift_accels = []

    prev_slope = 0.0

    for i in range(n):
        # Rolling window: last DRIFT_WINDOW days up to and including today
        window = user_df.iloc[max(0, i - cfg.DRIFT_WINDOW + 1): i + 1]

        # Drift score: rolling mean of deviation_score (smoothed)
        d_score = window['deviation_score'].mean()
        drift_scores.append(d_score)

        # Need at least MIN_DAYS_DRIFT points for a meaningful slope
        if len(window) >= cfg.MIN_DAYS_DRIFT:
            x = window['day'].values.astype(float)
            y = window['deviation_score'].values.astype(float)
            # Linear regression slope
            slope, _, _, _, _ = sp_stats.linregress(x, y)
            slope = max(slope, 0.0)   # We only care about upward drift
        else:
            slope = 0.0

        drift_slopes.append(slope)
        drift_accels.append(slope - prev_slope)
        prev_slope = slope

    user_df['drift_score'] = drift_scores
    user_df['drift_slope'] = drift_slopes
    user_df['drift_accel'] = drift_accels
    user_df['drift_label'] = [label_drift(s) for s in drift_slopes]

    return user_df


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drift metrics for all users.

    Args:
        df: dataframe with columns: user, day, deviation_score

    Returns:
        df with added columns: drift_score, drift_slope, drift_accel, drift_label
    """
    print(f"\n[L3] Computing behavioral drift (window={cfg.DRIFT_WINDOW} days)...")

    results = []
    users = df['user'].unique()

    for i, user in enumerate(users):
        user_df = df[df['user'] == user].copy()
        results.append(compute_drift_for_user(user_df))

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1:,}/{len(users):,} users...", end='\r')

    print(f"\n  Drift computed for {len(users):,} users")

    out = pd.concat(results, ignore_index=True)

    # Summary
    label_dist = out.groupby('drift_label').size()
    print(f"  Drift distribution:")
    for label, count in label_dist.items():
        pct = 100 * count / len(out)
        print(f"    {label:10s}: {count:>8,} rows ({pct:.1f}%)")

    return out
