"""
PIRS V2 - LAYER 2: DAILY DEVIATION FROM PERSONAL BASELINE
===========================================================
For each user on each day, compute how far their behavior deviates
from their personal baseline (z-score normalization).

deviation = (today - personal_mean) / (personal_std + epsilon)

A deviation of +3 means "3 standard deviations above your own normal" --
much more meaningful than comparing to other users.

Output: original df + deviation columns {feature}_dev
        + composite deviation_score (mean of absolute z-scores)
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

EPSILON = 1e-6   # Avoid divide-by-zero for features with zero std


def compute_deviations(df: pd.DataFrame,
                        baseline: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
    """
    Merge baseline stats into df and compute z-score deviation per feature.

    Args:
        df:           full features dataframe (user, day, *features)
        baseline:     per-user baseline (from layer_1)
        feature_cols: list of feature names

    Returns:
        df with added columns: {feature}_dev, deviation_score
    """
    print(f"\n[L2] Computing daily deviations from personal baseline...")

    # Merge baseline into main df
    df = df.merge(baseline, on='user', how='left')

    dev_cols = []
    for col in feature_cols:
        mean_col = f'{col}_mean'
        std_col  = f'{col}_std'

        if mean_col not in df.columns:
            continue

        dev_col = f'{col}_dev'
        df[dev_col] = (df[col] - df[mean_col]) / (df[std_col] + EPSILON)

        # Clip extreme outliers at ±10 sigma (keeps scale meaningful)
        df[dev_col] = df[dev_col].clip(-10, 10)
        dev_cols.append(dev_col)

    # Composite deviation score: mean of absolute z-scores across all features
    df['deviation_score'] = df[dev_cols].abs().mean(axis=1)

    # Clean up baseline stat columns from the main df (keep only dev columns)
    stat_cols = [f'{c}_mean' for c in feature_cols] + \
                [f'{c}_std'  for c in feature_cols]
    df = df.drop(columns=[c for c in stat_cols if c in df.columns])

    print(f"  Computed {len(dev_cols)} deviation features")
    print(f"  Deviation score: mean={df['deviation_score'].mean():.3f}, "
          f"max={df['deviation_score'].max():.3f}")

    return df, dev_cols


def run(df: pd.DataFrame, baseline: pd.DataFrame, feature_cols: list):
    return compute_deviations(df, baseline, feature_cols)
