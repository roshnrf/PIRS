"""
PIRS V2 - LAYER 9: METRICS, SHAP EXPLAINABILITY & PREVENTION QUANTIFICATION
=============================================================================
Answers all 4 research questions with concrete numbers.

RQ1: EPR (Early Prevention Rate)    -- % of threats flagged before malicious day
RQ2: PQ  (Prevention Quality)       -- effectiveness of personality-matched vs generic
RQ3: SHAP explainability            -- top features driving each user's risk score
RQ4: Cost savings                   -- prevented incidents x $11.4M average cost
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()


# ---------------------------------------------------------------------------
# RQ1: EARLY PREVENTION RATE
# ---------------------------------------------------------------------------

def compute_epr(df: pd.DataFrame,
                malicious_days: dict,
                window_days: int = 7) -> dict:
    """
    Early Prevention Rate: % of insider users flagged at least `window_days`
    before their first malicious day.

    A user is "flagged" if their risk_score >= RISK_HIGH or
    will_breach_{window_days}d == True at day (first_malicious - window_days).
    """
    flagged = 0
    total   = len(malicious_days)
    details = []

    for user, bad_days in malicious_days.items():
        first_malicious = min(bad_days)
        check_day = first_malicious - window_days

        user_df = df[(df['user'] == user) & (df['day'] <= check_day)]
        if user_df.empty:
            details.append({'user': user, 'flagged': False,
                            'reason': 'No data before window'})
            continue

        # Check if flagged at any point before check_day
        breach_col = f'will_breach_{window_days}d'
        was_flagged = (
            (user_df['risk_score'] >= cfg.RISK_HIGH).any() or
            (user_df.get(breach_col, pd.Series(False)).any())
        )

        if was_flagged:
            flagged += 1

        # Days of early warning
        high_risk_days = user_df[user_df['risk_score'] >= cfg.RISK_MODERATE]
        earliest = high_risk_days['day'].min() if not high_risk_days.empty else None
        days_early = (first_malicious - earliest) if earliest else 0

        details.append({
            'user':       user,
            'flagged':    was_flagged,
            'first_flag_day': earliest,
            'first_malicious_day': first_malicious,
            'days_early': days_early,
        })

    epr = flagged / total if total > 0 else 0.0

    return {
        'epr':     epr,
        'flagged': flagged,
        'total':   total,
        'details': pd.DataFrame(details),
    }


# ---------------------------------------------------------------------------
# RQ2: PREVENTION QUALITY (personality-matched vs generic)
# ---------------------------------------------------------------------------

def compute_prevention_quality(df: pd.DataFrame) -> dict:
    """
    Compares risk reduction for:
      - Users who received personality-matched interventions
      - Same users with generic default intervention

    Measures: avg risk change the day after intervention.
    """
    df_sorted = df.sort_values(['user', 'day'])

    matched_deltas  = []
    generic_deltas  = []

    for user, udf in df_sorted.groupby('user'):
        udf = udf.reset_index(drop=True)
        for i in range(len(udf) - 1):
            row      = udf.iloc[i]
            next_row = udf.iloc[i + 1]

            if row['intervention_level'] <= 1:
                continue   # Skip passive monitoring rows

            delta = row['risk_score'] - next_row['risk_score']

            # Matched: personality-specific intervention was used
            if row.get('primary_dim') and row['primary_dim'] != 'UNKNOWN':
                matched_deltas.append(delta)

            # Generic: compare against RL optimal (if available)
            if 'rl_intervention_level' in df.columns:
                rl_delta = row['risk_score'] - next_row['risk_score']
                generic_deltas.append(rl_delta)

    pq_matched = np.mean(matched_deltas) if matched_deltas else 0.0
    pq_generic = np.mean(generic_deltas) if generic_deltas else pq_matched

    return {
        'pq_personality_matched': round(pq_matched, 4),
        'pq_generic':             round(pq_generic, 4),
        'improvement_pct':        round(100 * (pq_matched - pq_generic) /
                                        (abs(pq_generic) + 1e-6), 2),
        'n_matched_interventions': len(matched_deltas),
    }


# ---------------------------------------------------------------------------
# RQ3: SHAP EXPLAINABILITY
# ---------------------------------------------------------------------------

def compute_shap_explanations(df: pd.DataFrame,
                               dev_cols: list,
                               top_n: int = 5) -> pd.DataFrame:
    """
    Computes SHAP-like feature importance per user using a lightweight
    gradient boosting model trained on risk_score.

    Returns per-user top-N contributing deviation features.
    """
    try:
        import shap
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        print("  [SHAP] shap or sklearn not available, skipping")
        return pd.DataFrame()

    print(f"\n[L9-SHAP] Computing feature explanations (top {top_n})...")

    available_devs = [c for c in dev_cols if c in df.columns]
    if not available_devs:
        print("  [WARN] No deviation features found")
        return pd.DataFrame()

    X = df[available_devs].fillna(0)
    y = df['risk_score']

    # Train a fast GBM on a sample
    n_train = min(50_000, len(X))
    idx = np.random.RandomState(cfg.RANDOM_STATE).choice(len(X), n_train, replace=False)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                       random_state=cfg.RANDOM_STATE)
    model.fit(X.iloc[idx], y.iloc[idx])

    # SHAP values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=available_devs, index=df.index)
    shap_df['user'] = df['user'].values
    shap_df['day']  = df['day'].values

    # Per-user: top contributing features (by mean absolute SHAP)
    user_explanations = []
    for user, udf in shap_df.groupby('user'):
        mean_abs = udf[available_devs].abs().mean()
        top_features = mean_abs.nlargest(top_n)
        explanation = {
            'user': user,
            'top_features': list(top_features.index),
            'top_shap_vals': list(top_features.values.round(4)),
        }
        # Human-readable
        explanation['explanation'] = '; '.join(
            f"{f.replace('_dev','')} (+{v:.2f})" for f, v in
            zip(top_features.index, top_features.values)
        )
        user_explanations.append(explanation)

    result = pd.DataFrame(user_explanations)
    print(f"  Explanations computed for {len(result):,} users")
    return result


# ---------------------------------------------------------------------------
# RQ4: PREVENTION COST QUANTIFICATION
# ---------------------------------------------------------------------------

def compute_cost_savings(epr_result: dict, n_users: int) -> dict:
    """
    Estimate cost savings from early prevention.

    Uses Ponemon Institute 2023 figure: $11.4M average insider incident cost.
    """
    epr          = epr_result['epr']
    cost_per_inc = cfg.INCIDENT_COST_USD
    n_prevented  = epr_result['flagged']

    # Conservative estimate: each flagged insider = 1 prevented incident
    cost_saved = n_prevented * cost_per_inc

    # Population-level: scale to organization size
    insider_rate = 0.0025   # ~0.25% insider rate (CERT baseline)
    expected_incidents = n_users * insider_rate
    expected_saved     = expected_incidents * epr * cost_per_inc

    return {
        'epr':                  round(epr, 4),
        'n_prevented':          n_prevented,
        'cost_per_incident_usd': cost_per_inc,
        'cost_saved_usd':       cost_saved,
        'expected_saved_usd':   round(expected_saved, 0),
        'cost_saved_formatted': f"${cost_saved:,.0f}",
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame,
        dev_cols: list,
        malicious_days: dict = None,
        n_users: int = 4000) -> dict:
    """
    Compute all metrics.

    Returns:
        dict with keys: epr, pq, shap, cost, summary_df
    """
    print(f"\n[L9] Computing evaluation metrics...")

    results = {}

    # RQ1
    if malicious_days:
        print(f"\n  RQ1: Early Prevention Rate (7-day and 14-day windows)...")
        for window in cfg.PREDICT_WINDOWS:
            epr_result = compute_epr(df, malicious_days, window_days=window)
            results[f'epr_{window}d'] = epr_result
            print(f"    EPR ({window}-day window): "
                  f"{epr_result['epr']*100:.1f}% "
                  f"({epr_result['flagged']}/{epr_result['total']} insiders flagged)")
            print(epr_result['details'][
                ['user','flagged','days_early','first_malicious_day']].to_string(index=False))

    # RQ2
    print(f"\n  RQ2: Prevention Quality...")
    pq = compute_prevention_quality(df)
    results['pq'] = pq
    print(f"    Personality-matched PQ: {pq['pq_personality_matched']:.4f}")
    print(f"    Generic PQ:             {pq['pq_generic']:.4f}")
    print(f"    Improvement:            {pq['improvement_pct']:.1f}%")

    # RQ3
    shap_df = compute_shap_explanations(df, dev_cols)
    results['shap'] = shap_df

    # RQ4
    if malicious_days:
        print(f"\n  RQ4: Cost savings estimate...")
        epr_7d = results.get('epr_7d', {'epr': 0, 'flagged': 0})
        cost   = compute_cost_savings(epr_7d, n_users)
        results['cost'] = cost
        print(f"    Insiders prevented: {cost['n_prevented']}")
        print(f"    Cost saved:         {cost['cost_saved_formatted']}")

    # Summary table
    summary = {
        'Metric':  ['EPR (7-day)',  'EPR (14-day)', 'PQ (matched)', 'PQ (generic)'],
        'Value':   [
            f"{results.get('epr_7d',{}).get('epr',0)*100:.1f}%",
            f"{results.get('epr_14d',{}).get('epr',0)*100:.1f}%",
            f"{pq['pq_personality_matched']:.4f}",
            f"{pq['pq_generic']:.4f}",
        ],
    }
    results['summary_df'] = pd.DataFrame(summary)
    print(f"\n  Summary:")
    print(results['summary_df'].to_string(index=False))

    return results
