"""
PIRS V2 - LANL VALIDATION
==========================
Validates PIRS core detection engine against LANL red team ground truth.

Metrics:
  - ROC-AUC on red team day detection
  - Early warning: were red team users flagged before their attack day?
  - Comparison: red team drift trajectories vs normal users

Usage:
    python validation/lanl_validator.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import LANLConfig, ModelConfig

cfg  = LANLConfig()
mcfg = ModelConfig()

RESULTS_FILE = os.path.join(cfg.OUTPUT_DIR, 'lanl_complete.csv')
PLOTS_DIR    = os.path.join(cfg.OUTPUT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"[ERROR] {RESULTS_FILE} not found. Run pipeline_lanl.py first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    print(f"Loaded {len(df):,} rows, {df['user'].nunique():,} users")
    return df


def load_redteam_dict() -> dict:
    from extractors.lanl_extractor import load_redteam
    _, rt_df = load_redteam()
    mal = {}
    for _, row in rt_df.iterrows():
        mal.setdefault(row['user'], []).append(int(row['day']))
    return mal


def validate_early_warning(df: pd.DataFrame, malicious: dict) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  LANL EARLY WARNING VALIDATION")
    print("="*60)

    check_windows = [3, 7]    # LANL only has 58 days, so 14d often unavailable
    rows = []

    # Sample top 20 red team users for readable output
    sample_users = sorted(malicious.keys(),
                           key=lambda u: min(malicious[u]))[:20]

    for user in sample_users:
        bad_days  = malicious[user]
        first_mal = min(bad_days)
        user_df   = df[df['user'] == user].sort_values('day')
        row = {'user': user, 'first_attack_day': first_mal,
               'n_attack_days': len(bad_days)}

        for W in check_windows:
            target = first_mal - W
            past   = user_df[user_df['day'] <= target]

            if past.empty:
                row[f'risk_{W}d_before'] = np.nan
                row[f'flagged_{W}d']     = False
                continue

            latest  = past.iloc[-1]
            risk    = latest['risk_score']
            flagged = risk >= mcfg.RISK_MODERATE
            row[f'risk_{W}d_before'] = round(risk, 3)
            row[f'flagged_{W}d']     = flagged

        rows.append(row)

    result = pd.DataFrame(rows)

    for W in check_windows:
        col = f'flagged_{W}d'
        if col in result:
            n_flagged = result[col].sum()
            pct = 100 * n_flagged / len(result)
            print(f"  Flagged at {W}d before: {n_flagged}/{len(result)} "
                  f"({pct:.0f}%)")

    out = os.path.join(cfg.OUTPUT_DIR, 'lanl_validation_early_warning.csv')
    result.to_csv(out, index=False)
    print(f"  Saved: {os.path.basename(out)}")
    return result


def compute_roc_auc(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("  LANL ROC-AUC")
    print("="*60)

    y_true = df['redteam'] if 'redteam' in df.columns else df.get('insider', 0)
    y_pred = df['risk_score']

    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"  [WARN] {e}")
        auc = 0.0

    print(f"  Red team rows: {y_true.sum():,} / {len(y_true):,}")
    print(f"  ROC-AUC: {auc:.4f}")

    return {'auc': auc}


def plot_comparison(df: pd.DataFrame, malicious: dict):
    """Plot average risk trajectory: red team users vs. normal users."""
    print("\n[PLOTS] Generating LANL comparison plot...")

    rt_users  = list(malicious.keys())
    non_rt    = [u for u in df['user'].unique() if u not in rt_users]
    sample_nt = np.random.RandomState(42).choice(non_rt,
                                                   min(100, len(non_rt)),
                                                   replace=False)

    # Average daily risk
    rt_df  = df[df['user'].isin(rt_users)].groupby('day')['risk_score'].mean()
    nt_df  = df[df['user'].isin(sample_nt)].groupby('day')['risk_score'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rt_df.index,  rt_df.values,  color='#e74c3c', linewidth=2,
            label='Red team users (avg)')
    ax.plot(nt_df.index,  nt_df.values,  color='steelblue', linewidth=1.5,
            alpha=0.7, label='Normal users (avg, n=100)')
    ax.axhline(mcfg.RISK_HIGH, color='#c0392b', linestyle='--',
               label=f'Alert threshold ({mcfg.RISK_HIGH})')

    ax.set_title('PIRS V2 -- LANL: Red Team vs Normal User Risk Trajectories',
                 fontweight='bold')
    ax.set_xlabel('Day')
    ax.set_ylabel('Average Risk Score')
    ax.legend()
    ax.set_ylim(0, 10.5)

    out = os.path.join(PLOTS_DIR, 'lanl_comparison.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def run():
    df       = load_results()
    malicious = load_redteam_dict()

    ew       = validate_early_warning(df, malicious)
    auc_res  = compute_roc_auc(df)
    plot_comparison(df, malicious)

    summary = pd.DataFrame({
        'Metric': ['ROC-AUC',
                   'Flagged at 7d (%)',
                   'Total red team users'],
        'Value':  [
            f"{auc_res['auc']:.4f}",
            f"{100*ew['flagged_7d'].mean():.0f}%" if 'flagged_7d' in ew else 'N/A',
            str(len(malicious)),
        ]
    })

    out = os.path.join(cfg.OUTPUT_DIR, 'lanl_validation_summary.csv')
    summary.to_csv(out, index=False)

    print("\n" + "="*60)
    print("  LANL VALIDATION COMPLETE")
    print("="*60)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run()
