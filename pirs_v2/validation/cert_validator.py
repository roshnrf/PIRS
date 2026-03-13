"""
PIRS V2 - CERT VALIDATION
==========================
Validates the CERT pipeline output against the 5 real insider users.

Answers:
  - At 7 days before first malicious act: was user flagged?
  - At 14 days before: was user flagged?
  - What was the risk trajectory (daily) for each insider?
  - ROC-AUC on malicious-day vs all-day classification
  - Compare insider risk trajectories vs random non-insider sample

Usage:
    python validation/cert_validator.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CERTConfig, ModelConfig

cfg  = CERTConfig()
mcfg = ModelConfig()

OUTPUT_DIR  = cfg.OUTPUT_DIR
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'cert_complete.csv')
PLOTS_DIR    = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

SCENARIO_NAMES = {
    'ACM2278': 'Sc1: Wikileaks Upload',
    'CMP2946': 'Sc2: USB Theft',
    'PLJ1771': 'Sc3: Keylogger Sabotage',
    'CDE1846': 'Sc4: Email Exfiltration',
    'MBG3183': 'Sc5: Dropbox Upload',
}


def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_FILE):
        print(f"[ERROR] Results file not found: {RESULTS_FILE}")
        print("  Run pipeline_cert.py first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    print(f"Loaded {len(df):,} rows, {df['user'].nunique():,} users")
    return df


def validate_early_warning(df: pd.DataFrame) -> pd.DataFrame:
    """Check each insider at 3, 7, 10, 14 days before first malicious day."""
    print("\n" + "="*60)
    print("  EARLY WARNING VALIDATION")
    print("="*60)

    check_windows = [3, 7, 10, 14]
    rows = []

    for user, bad_days in cfg.INSIDER_MALICIOUS_DAYS.items():
        first_mal  = min(bad_days)
        scenario   = SCENARIO_NAMES.get(user, user)
        user_df    = df[df['user'] == user].sort_values('day')

        print(f"\n  {user} ({scenario})")
        print(f"  First malicious day: {first_mal} | "
              f"Total malicious days: {len(bad_days)}")

        row = {'user': user, 'scenario': scenario,
               'first_malicious_day': first_mal}

        for W in check_windows:
            target = first_mal - W
            past   = user_df[user_df['day'] <= target]

            if past.empty:
                row[f'risk_{W}d_before']  = np.nan
                row[f'flagged_{W}d']      = False
                row[f'alert_{W}d_before'] = 'N/A'
                continue

            latest = past.iloc[-1]
            risk   = latest['risk_score']
            alert  = latest.get('alert_level', 'N/A')
            breach = latest.get(f'will_breach_{min(W,14)}d', False)

            flagged = risk >= mcfg.RISK_MODERATE or bool(breach)

            row[f'risk_{W}d_before']  = round(risk, 3)
            row[f'alert_{W}d_before'] = alert
            row[f'flagged_{W}d']      = flagged

            status = 'FLAGGED' if flagged else 'missed'
            print(f"    {W:2d}d before (day {target:3d}): "
                  f"risk={risk:.2f}  alert={alert:9s}  [{status}]")

        # Peak on actual malicious days
        mal_df = user_df[user_df['day'].isin(bad_days)]
        row['peak_risk_malicious'] = round(mal_df['risk_score'].max(), 3) \
                                     if not mal_df.empty else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)

    # Summary
    print(f"\n  Summary (7-day window):")
    flagged_7d = result['flagged_7d'].sum()
    print(f"  Flagged at 7d: {flagged_7d}/{len(result)}")
    flagged_14d = result['flagged_14d'].sum()
    print(f"  Flagged at 14d: {flagged_14d}/{len(result)}")

    out = os.path.join(OUTPUT_DIR, 'cert_validation_early_warning.csv')
    result.to_csv(out, index=False)
    print(f"  Saved: {os.path.basename(out)}")

    return result


def compute_roc_auc(df: pd.DataFrame) -> dict:
    """Compute ROC-AUC for malicious day detection."""
    print("\n" + "="*60)
    print("  ROC-AUC EVALUATION")
    print("="*60)

    # Label: 1 if malicious day, 0 otherwise
    df['is_malicious'] = 0
    for user, days in cfg.INSIDER_MALICIOUS_DAYS.items():
        mask = (df['user'] == user) & (df['day'].isin(days))
        df.loc[mask, 'is_malicious'] = 1

    y_true = df['is_malicious']
    y_pred = df['risk_score']

    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        auc = 0.0
        print(f"  [WARN] AUC error: {e}")

    # Threshold-based
    threshold = mcfg.RISK_HIGH
    y_binary  = (y_pred >= threshold).astype(int)
    n_tp = ((y_binary == 1) & (y_true == 1)).sum()
    n_fp = ((y_binary == 1) & (y_true == 0)).sum()
    n_fn = ((y_binary == 0) & (y_true == 1)).sum()
    precision = n_tp / (n_tp + n_fp + 1e-6)
    recall    = n_tp / (n_tp + n_fn + 1e-6)

    print(f"  Malicious day rows: {y_true.sum()} / {len(y_true):,}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Threshold: {threshold}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"  True Positives:  {n_tp}")
    print(f"  False Positives: {n_fp}")
    print(f"  False Negatives: {n_fn}")

    return {'auc': auc, 'precision': precision, 'recall': recall,
            'tp': n_tp, 'fp': n_fp, 'fn': n_fn}


def plot_insider_trajectories(df: pd.DataFrame):
    """Plot risk score timeline for each insider vs. a random non-insider sample."""
    print("\n[PLOTS] Generating insider trajectory plots...")

    # Random non-insider sample (3 users for comparison)
    non_insiders = [u for u in df['user'].unique() if u not in cfg.INSIDER_USERS]
    sample_users = np.random.RandomState(42).choice(non_insiders, 3, replace=False)

    fig, axes = plt.subplots(len(cfg.INSIDER_USERS), 1,
                             figsize=(14, 4 * len(cfg.INSIDER_USERS)),
                             sharex=False)

    for ax, user in zip(axes, cfg.INSIDER_USERS):
        udf = df[df['user'] == user].sort_values('day')
        bad = cfg.INSIDER_MALICIOUS_DAYS[user]

        ax.plot(udf['day'], udf['risk_score'],
                color='#e74c3c', linewidth=2, label=f'{user} (insider)')
        ax.axhline(mcfg.RISK_HIGH, color='#c0392b', linestyle='--',
                   alpha=0.6, label=f'Alert threshold ({mcfg.RISK_HIGH})')
        ax.axhline(mcfg.RISK_MODERATE, color='#e67e22', linestyle=':',
                   alpha=0.6, label=f'Watch threshold ({mcfg.RISK_MODERATE})')

        # Shade malicious period
        ax.axvspan(min(bad), max(bad), alpha=0.15, color='red',
                   label='Malicious period')

        # Mark 7 and 14 days before
        for W, color in [(7, '#9b59b6'), (14, '#3498db')]:
            ax.axvline(min(bad) - W, color=color, linestyle='--',
                       alpha=0.7, label=f'{W}d before')

        # Sample non-insiders in grey
        for comp in sample_users:
            cdf = df[df['user'] == comp].sort_values('day')
            ax.plot(cdf['day'], cdf['risk_score'],
                    color='grey', linewidth=0.5, alpha=0.4)

        scenario = SCENARIO_NAMES.get(user, user)
        ax.set_title(f'{scenario}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Risk Score')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(0, 10.5)

    plt.suptitle('PIRS V2 -- Insider Risk Trajectories\n'
                 '(Grey = random non-insiders, Red = insider, '
                 'Shaded = malicious period)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, 'insider_trajectories.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def run():
    df = load_results()

    ew_results = validate_early_warning(df)
    auc_results = compute_roc_auc(df)
    plot_insider_trajectories(df)

    # Save combined summary
    summary = {
        'Metric':  ['ROC-AUC',
                    'Flagged at 7d (of 5)',
                    'Flagged at 14d (of 5)',
                    'Precision @ threshold 6.0',
                    'Recall @ threshold 6.0'],
        'Value': [
            f"{auc_results['auc']:.4f}",
            f"{ew_results['flagged_7d'].sum()}/5",
            f"{ew_results['flagged_14d'].sum()}/5",
            f"{auc_results['precision']:.4f}",
            f"{auc_results['recall']:.4f}",
        ]
    }
    summary_df = pd.DataFrame(summary)
    out = os.path.join(OUTPUT_DIR, 'cert_validation_summary.csv')
    summary_df.to_csv(out, index=False)

    print("\n" + "="*60)
    print("  VALIDATION COMPLETE")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\n  Plots saved in: {PLOTS_DIR}")


if __name__ == '__main__':
    run()
