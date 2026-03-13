"""
PIRS V2 - LAYER 5: 7/14-DAY BREACH PREDICTION
===============================================
This is the core research contribution: predicting BEFORE it happens.

Approach:
  - Combine anomaly_score + drift_slope + drift_accel into a composite risk
  - For each user-day, project their trajectory forward 7 and 14 days
  - Predict: will this user's risk exceed RISK_HIGH in the next 7/14 days?

Validation (for CERT):
  - For each of the 5 insider users, check prediction score at:
      day (first_malicious_day - 14), - 10, - 7, - 3
  - Did we flag them before they acted?

Output columns:
  risk_score          -- composite risk today (0-10)
  projected_risk_7d   -- projected risk in 7 days
  projected_risk_14d  -- projected risk in 14 days
  will_breach_7d      -- bool: predicted breach within 7 days
  will_breach_14d     -- bool: predicted breach within 14 days
  days_to_breach      -- estimated days until risk exceeds threshold
  alert_level         -- NORMAL / WATCH / ELEVATED / HIGH / CRITICAL
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()

ALERT_LEVELS = [
    (0.0,  2.0,  'NORMAL'),
    (2.0,  4.0,  'WATCH'),
    (4.0,  6.0,  'ELEVATED'),
    (6.0,  8.0,  'HIGH'),
    (8.0,  10.1, 'CRITICAL'),
]


def get_alert_level(score):
    for lo, hi, label in ALERT_LEVELS:
        if lo <= score < hi:
            return label
    return 'NORMAL'


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite risk = weighted combination of anomaly score and drift signals.
    Drift slope upweights users whose behavior is escalating.
    """
    # Normalize drift_slope to 0-10 scale
    slope_max = df['drift_slope'].quantile(0.99)
    slope_norm = (df['drift_slope'].clip(0, slope_max) / (slope_max + 1e-6)) * 10

    # Drift acceleration bonus (escalating drift is worse)
    accel_max  = df['drift_accel'].quantile(0.99)
    accel_norm = (df['drift_accel'].clip(0, accel_max) / (accel_max + 1e-6)) * 10

    # Composite risk score (weights tuned to emphasize temporal escalation)
    df['risk_score'] = (
        0.50 * df['anomaly_score'] +
        0.35 * slope_norm +
        0.15 * accel_norm
    ).clip(0, 10)

    return df


def project_trajectory(user_df: pd.DataFrame, window: int) -> np.ndarray:
    """
    For each row, project risk_score forward `window` days using
    the current drift slope.

    projected = current_risk + slope * window
    """
    projected = (user_df['risk_score'] +
                 user_df['drift_slope'] * window).clip(0, 10)
    return projected.values


def estimate_days_to_breach(risk, slope, threshold=None):
    """
    Given current risk and slope, estimate days until threshold is crossed.
    Returns NaN if slope <= 0 (not escalating) or already above threshold.
    """
    if threshold is None:
        threshold = cfg.RISK_HIGH
    if risk >= threshold:
        return 0
    if slope <= 0:
        return np.nan
    return (threshold - risk) / slope


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk scores and 7/14-day predictions for all users.

    Args:
        df: dataframe with anomaly_score, drift_slope, drift_accel

    Returns:
        df with prediction columns added
    """
    print(f"\n[L5] Computing risk scores and 7/14-day predictions...")

    # Composite risk score
    df = compute_risk_score(df)

    results = []
    users = df['user'].unique()

    for user in users:
        udf = df[df['user'] == user].sort_values('day').copy()

        # Project forward
        udf['projected_risk_7d']  = project_trajectory(udf, 7)
        udf['projected_risk_14d'] = project_trajectory(udf, 14)

        # Breach predictions
        udf['will_breach_7d']  = udf['projected_risk_7d']  >= cfg.RISK_HIGH
        udf['will_breach_14d'] = udf['projected_risk_14d'] >= cfg.RISK_HIGH

        # Days to breach estimate
        udf['days_to_breach'] = udf.apply(
            lambda r: estimate_days_to_breach(r['risk_score'], r['drift_slope']),
            axis=1
        )

        # Alert level
        udf['alert_level'] = udf['risk_score'].apply(get_alert_level)

        results.append(udf)

    df = pd.concat(results, ignore_index=True)

    # Summary
    breach_7  = df['will_breach_7d'].sum()
    breach_14 = df['will_breach_14d'].sum()
    high_risk = (df['risk_score'] >= cfg.RISK_HIGH).sum()

    print(f"  Risk score: mean={df['risk_score'].mean():.3f}, "
          f"max={df['risk_score'].max():.3f}")
    print(f"  Currently high-risk (>={cfg.RISK_HIGH}): {high_risk:,} user-days")
    print(f"  Predicted breach in  7 days: {breach_7:,} user-days")
    print(f"  Predicted breach in 14 days: {breach_14:,} user-days")
    print(f"  Alert level distribution:")
    for lvl, cnt in df['alert_level'].value_counts().items():
        print(f"    {lvl:10s}: {cnt:>8,}")

    return df


# ---------------------------------------------------------------------------
# PRE-MALICIOUS WINDOW EVALUATION (CERT-specific validation)
# ---------------------------------------------------------------------------

def evaluate_early_warning(df: pd.DataFrame,
                             insider_malicious_days: dict,
                             windows: list = None) -> pd.DataFrame:
    """
    For each insider user, check what their risk_score and predictions looked
    like at [first_malicious_day - W] for each W in windows.

    This answers: "Did PIRS flag them W days BEFORE they acted?"

    Returns a summary dataframe.
    """
    if windows is None:
        windows = cfg.PREDICT_WINDOWS + [3, 10]   # Check at 3, 7, 10, 14 days before

    print(f"\n[VALIDATION] Pre-malicious window evaluation...")
    print(f"  Checking at {windows} days before first malicious day")

    rows = []
    for user, bad_days in insider_malicious_days.items():
        first_malicious = min(bad_days)
        user_df = df[df['user'] == user].sort_values('day')

        if user_df.empty:
            print(f"  [WARN] {user}: no data found")
            continue

        row = {'user': user, 'first_malicious_day': first_malicious}

        for W in windows:
            target_day = first_malicious - W
            if target_day < 1:
                row[f'risk_{W}d_before'] = np.nan
                row[f'alert_{W}d_before'] = 'N/A'
                row[f'breach_pred_{W}d'] = False
                continue

            # Find closest available day at or before target_day
            candidates = user_df[user_df['day'] <= target_day]
            if candidates.empty:
                row[f'risk_{W}d_before'] = np.nan
                row[f'alert_{W}d_before'] = 'N/A'
                row[f'breach_pred_{W}d'] = False
                continue

            closest = candidates.iloc[-1]
            row[f'risk_{W}d_before']  = closest['risk_score']
            row[f'alert_{W}d_before'] = closest['alert_level']

            # Was a breach predicted at this lookahead?
            if W <= 7:
                row[f'breach_pred_{W}d'] = bool(closest.get('will_breach_7d', False))
            else:
                row[f'breach_pred_{W}d'] = bool(closest.get('will_breach_14d', False))

        # Peak risk on actual malicious days
        malicious_df = user_df[user_df['day'].isin(bad_days)]
        row['peak_risk_malicious'] = malicious_df['risk_score'].max() if not malicious_df.empty else np.nan
        row['n_malicious_days'] = len(bad_days)

        rows.append(row)

    result = pd.DataFrame(rows)

    print(f"\n  Results:")
    print(result.to_string(index=False))

    return result
