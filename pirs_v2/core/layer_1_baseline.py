"""
PIRS V2 - LAYER 1: PERSONAL BEHAVIORAL BASELINE
================================================
Computes each user's personal "normal" behavior from their first BASELINE_DAYS days.

Key insight: We compare each user to THEMSELVES, not to the population.
A power user who copies 50 files/day is normal for them.
A quiet user who suddenly copies 50 files/day is anomalous.

Output adds columns: {feature}_mean, {feature}_std  (per user)
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()


def compute_personal_baseline(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    For each user, compute mean and std of each feature over the first
    BASELINE_DAYS days. Returns a per-user baseline dataframe.

    Args:
        df:           features dataframe with columns: user, day, *features
        feature_cols: list of feature column names to baseline

    Returns:
        baseline_df: user x (feature_mean, feature_std) dataframe
    """
    print(f"\n[L1] Computing personal baselines (first {cfg.BASELINE_DAYS} days per user)...")

    baseline_period = df[df['day'] <= cfg.BASELINE_DAYS].copy()

    # For users with < 5 days in baseline, use all available days
    user_day_counts = baseline_period.groupby('user')['day'].count()
    sparse_users = user_day_counts[user_day_counts < 5].index
    if len(sparse_users) > 0:
        extra = df[df['user'].isin(sparse_users)]
        baseline_period = pd.concat([baseline_period, extra]).drop_duplicates()

    agg = {}
    for col in feature_cols:
        agg[f'{col}_mean'] = (col, 'mean')
        agg[f'{col}_std']  = (col, 'std')

    baseline = baseline_period.groupby('user').agg(**agg).reset_index()

    # Replace NaN std (only 1 observation) with 0
    std_cols = [f'{c}_std' for c in feature_cols]
    baseline[std_cols] = baseline[std_cols].fillna(0)

    print(f"  Baselines computed for {len(baseline):,} users")
    print(f"  Baseline features: {len(feature_cols)} x 2 (mean + std) = {len(feature_cols)*2} columns")

    return baseline


def get_feature_cols(df: pd.DataFrame, exclude=None) -> list:
    """Return numeric feature columns, excluding metadata columns."""
    if exclude is None:
        exclude = ['user', 'day', 'date', 'insider', 'redteam',
                   'O', 'C', 'E', 'A', 'N']
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def run(df: pd.DataFrame, feature_cols: list = None):
    """
    Main entry point.

    Returns:
        baseline_df: per-user baseline statistics
        feature_cols: list of feature columns used
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(df)

    baseline = compute_personal_baseline(df, feature_cols)
    return baseline, feature_cols
