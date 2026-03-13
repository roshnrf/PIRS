"""
PIRS V2 - LAYER 6: PERSONALITY CONTEXT
========================================
Maps each user to one of 5 behavioral personality dimensions using
their deviation patterns. Uses OCEAN psychometric scores if available.

Dimensions:
  COMPLIANT   -- policy-adherent, regular hours, structured behavior
  SOCIAL      -- high email/communication volume, external contacts
  CAREFULL    -- organized file handling, work-hour focused
  RISK_TAKER  -- after-hours USB, external emails, risky sites
  AUTONOMOUS  -- broad HTTP usage, cloud activity, independent behavior

The personality dimension is used by Layer 7 to SELECT the right intervention.
Same risk score, different personality = different intervention.

Note: This layer is CERT-specific (requires behavioral features).
      LANL pipeline skips this layer.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig, CERTConfig

cfg  = ModelConfig()
ccfg = CERTConfig()

# Semantic feature groups for each dimension
# Uses deviation features (feature_dev) if available, else raw features
DIMENSION_FEATURES = {
    'COMPLIANT':  ['n_logon', 'work_hour_ratio', 'n_email_sent',
                   'n_file_ops', 'is_weekday'],
    'SOCIAL':     ['n_email_sent', 'n_email_recv', 'n_email_external',
                   'n_social_media', 'n_email_with_att'],
    'CAREFULL':   ['work_hour_ratio', 'n_unique_pcs', 'n_file_doc',
                   'n_file_ops', 'n_logon'],
    'RISK_TAKER': ['n_afterhour_usb', 'n_file_to_usb', 'n_email_external',
                   'n_job_sites', 'after_hours_ratio', 'n_hack_sites',
                   'n_email_bcc_ext'],
    'AUTONOMOUS': ['n_http', 'n_cloud_upload', 'n_file_ops',
                   'n_hack_sites', 'n_afterhour_http'],
}


def compute_dimension_scores(df: pd.DataFrame, use_deviations: bool = True) -> pd.DataFrame:
    """
    Compute per-user personality dimension scores.
    Prefers deviation features (_dev suffix) for better signal.
    """
    print(f"\n[L6] Computing personality dimensions...")

    user_scores = []

    for user, udf in df.groupby('user'):
        # Use last 30 days for personality (most recent behavior is most relevant)
        recent = udf.nlargest(30, 'day')
        row = {'user': user}

        for dim, features in DIMENSION_FEATURES.items():
            available = []
            for f in features:
                # Prefer deviation feature
                dev_col = f'{f}_dev'
                if use_deviations and dev_col in recent.columns:
                    available.append(dev_col)
                elif f in recent.columns:
                    available.append(f)

            if not available:
                row[dim] = 0.0
                continue

            vals = recent[available].values.astype(float)
            vals = np.nan_to_num(vals, nan=0.0)
            raw = vals.mean()
            row[dim] = max(raw, 0.0)   # Personality scores are non-negative

        user_scores.append(row)

    scores_df = pd.DataFrame(user_scores)

    # Normalize each dimension to [0, 1] across all users
    for dim in cfg.PERSONALITY_DIMS:
        if dim in scores_df.columns:
            col = scores_df[dim]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                scores_df[dim] = (col - col_min) / (col_max - col_min)
            else:
                scores_df[dim] = 0.5   # All same = neutral

    # Primary dimension
    scores_df['primary_dim'] = scores_df[cfg.PERSONALITY_DIMS].idxmax(axis=1)

    print(f"  Personality profiles computed for {len(scores_df):,} users")
    print(f"  Primary dimension distribution:")
    for dim, cnt in scores_df['primary_dim'].value_counts().items():
        pct = 100 * cnt / len(scores_df)
        print(f"    {dim:12s}: {cnt:>5,} ({pct:.1f}%)")

    return scores_df


def add_ocean_context(scores_df: pd.DataFrame,
                       df: pd.DataFrame) -> pd.DataFrame:
    """
    If OCEAN scores are available in df, merge them into scores_df.
    Validates: SOCIAL should correlate with E (Extraversion),
               CAREFULL should correlate with C (Conscientiousness).
    """
    ocean_cols = ['user', 'O', 'C', 'E', 'A', 'N']
    if not all(c in df.columns for c in ocean_cols):
        return scores_df

    ocean = df[ocean_cols].drop_duplicates('user')
    scores_df = scores_df.merge(ocean, on='user', how='left')

    # Quick correlation check
    if 'E' in scores_df.columns and 'SOCIAL' in scores_df.columns:
        corr_se = scores_df['SOCIAL'].corr(scores_df['E'] / 50.0)
        corr_cc = scores_df['CAREFULL'].corr(scores_df['C'] / 50.0)
        print(f"  OCEAN validation - SOCIAL vs E: {corr_se:.3f}, "
              f"CAREFULL vs C: {corr_cc:.3f}")

    return scores_df


def run(df: pd.DataFrame) -> tuple:
    """
    Returns:
        scores_df: per-user personality profile
        df:        original df with primary_dim merged in
    """
    scores_df = compute_dimension_scores(df, use_deviations=True)
    scores_df = add_ocean_context(scores_df, df)

    # Merge primary_dim back into main df
    dim_map = scores_df.set_index('user')['primary_dim'].to_dict()
    df['primary_dim'] = df['user'].map(dim_map)

    return scores_df, df
