"""
PIRS BACKEND - LAYER VALIDATION: INSIDER DETECTION EVALUATION
=============================================================
Validates the PIRS anomaly detection system against the 5 known
CERT r6.2 insider users. Provides:

  1. User-level ROC-AUC (5 positives vs ~3,995 negatives)
  2. Detection rate at top-1%, top-5%, top-10% risk thresholds
  3. Per-insider risk ranking and scenario analysis
  4. Early warning analysis: did drift detect BEFORE malicious days?
  5. Escalation effectiveness per insider scenario

This module addresses the thesis limitation: "no ground truth labels".
Ground truth IS present in the dataset -- 35 labeled rows for 5 users.

Can be run standalone: python layer_validation.py
Or imported:           from layer_validation import run_validation

Author: Roshan A Rauof
Defense: March 12, 2026
"""

import os
import sys
import time
import numpy as np
import pandas as pd

try:
    from config import PIRSConfig
    from feature_engineering import (
        INSIDER_USERS, INSIDER_SCENARIOS,
        INSIDER_MALICIOUS_DAYS, SCENARIO_NAMES
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig
    from feature_engineering import (
        INSIDER_USERS, INSIDER_SCENARIOS,
        INSIDER_MALICIOUS_DAYS, SCENARIO_NAMES
    )

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN]  sklearn not available - ROC-AUC will be computed manually")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pipeline_output():
    """
    Load the most complete pipeline output available.
    Priority: pirs_complete.csv > layer_8_full_results.csv > layer_4_drift.csv
              > layer_1_3_baseline.csv
    """
    print("\n[DIR] Loading pipeline output for validation...")

    candidates = [
        (PIRSConfig.OUTPUT_FILES['master'],   'pirs_complete.csv'),
        ('layer_8_full_results.csv',          'Layer 8 full results'),
        (PIRSConfig.OUTPUT_FILES['qlearning'],'Layer 7 Q-learning'),
        (PIRSConfig.OUTPUT_FILES['drift'],    'Layer 4 drift'),
        (PIRSConfig.OUTPUT_FILES['baseline'], 'Layer 1-3 baseline'),
    ]

    for filename, label in candidates:
        path = os.path.join(PIRSConfig.OUTPUT_DIR, filename)
        if os.path.exists(path):
            print(f"   Loading {label}: {path}")
            df = pd.read_csv(path)
            print(f"[OK] Loaded {len(df):,} rows, {df['user'].nunique():,} unique users")
            return df, label

    raise FileNotFoundError(
        "No pipeline output found. Run master_pipeline.py first.\n"
        f"Expected files in: {PIRSConfig.OUTPUT_DIR}"
    )


def load_insider_labels():
    """
    Load the insider labels from data_features_semantic.csv or
    data_processed.csv to get day-level ground truth.
    """
    semantic_path   = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_features_semantic.csv')
    processed_path  = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_processed.csv')

    for path in [semantic_path, processed_path]:
        if os.path.exists(path):
            cols = ['user', 'day', 'insider']
            df = pd.read_csv(path, usecols=cols)
            insider_rows = df[df['insider'] > 0]
            print(f"\n[INFO] Ground truth loaded: {len(insider_rows)} labeled "
                  f"insider rows from {os.path.basename(path)}")
            return df

    # Fallback: construct labels from known constants
    print("[WARN]  Label file not found -- constructing ground truth from constants")
    rows = []
    for user, days in INSIDER_MALICIOUS_DAYS.items():
        for day in days:
            rows.append({'user': user, 'day': day,
                         'insider': INSIDER_SCENARIOS[user]})
    return pd.DataFrame(rows)


# ============================================================================
# MANUAL ROC-AUC (no sklearn dependency)
# ============================================================================

def manual_roc_auc(y_true, y_score):
    """Compute ROC-AUC without sklearn."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count pairs where positive > negative
    auc = sum(
        (1.0 if p > n else 0.5 if p == n else 0.0)
        for p in pos_scores for n in neg_scores
    ) / (n_pos * n_neg)
    return auc


def compute_roc_auc(y_true, y_score):
    if SKLEARN_AVAILABLE:
        try:
            return roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return manual_roc_auc(np.array(y_true), np.array(y_score))


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def user_level_analysis(df):
    """
    Aggregate risk scores to user level and evaluate insider detection.
    Returns user_risk dataframe and ROC-AUC score.
    """
    print("\n" + "="*70)
    print("1. USER-LEVEL RISK SCORE ANALYSIS")
    print("="*70)

    # Aggregate per user
    agg = {'risk_score': ['max', 'mean', 'std', 'count']}
    if 'drift_score' in df.columns:
        agg['drift_score'] = ['max', 'mean']
    if 'intervention_level' in df.columns:
        agg['intervention_level'] = 'max'

    user_risk = df.groupby('user').agg(agg).reset_index()

    # Flatten multi-index columns
    user_risk.columns = [
        '_'.join(filter(None, col)).strip('_')
        for col in user_risk.columns
    ]
    # Rename for clarity
    rename = {
        'risk_score_max':   'max_risk',
        'risk_score_mean':  'mean_risk',
        'risk_score_std':   'std_risk',
        'risk_score_count': 'obs_days',
    }
    if 'drift_score_max' in user_risk.columns:
        rename['drift_score_max']  = 'max_drift'
        rename['drift_score_mean'] = 'mean_drift'
    user_risk = user_risk.rename(columns=rename)

    # Insider ground truth
    user_risk['is_insider'] = user_risk['user'].isin(INSIDER_USERS).astype(int)
    n_total   = len(user_risk)
    n_insider = user_risk['is_insider'].sum()

    print(f"\n   Total users:    {n_total:,}")
    print(f"   Known insiders: {n_insider}  (Users {INSIDER_USERS})")

    # Summary statistics
    ins  = user_risk[user_risk['is_insider'] == 1]
    norm = user_risk[user_risk['is_insider'] == 0]

    print(f"\n   +---------------------------------------------+")
    print(f"   |           Max Risk Score Comparison          |")
    print(f"   +---------------------------------------------+")
    print(f"   | Insider users:  u = {ins['max_risk'].mean():.3f}  "
          f"max = {ins['max_risk'].max():.3f}  |")
    print(f"   | Normal  users:  u = {norm['max_risk'].mean():.3f}  "
          f"max = {norm['max_risk'].max():.3f}  |")
    print(f"   +---------------------------------------------+")

    # ROC-AUC
    auc_max  = compute_roc_auc(user_risk['is_insider'].values,
                                user_risk['max_risk'].values)
    auc_mean = compute_roc_auc(user_risk['is_insider'].values,
                                user_risk['mean_risk'].values)

    print(f"\n   ROC-AUC (max risk  score): {auc_max:.4f}")
    print(f"   ROC-AUC (mean risk score): {auc_mean:.4f}")
    print(f"   (0.5 = random chance, 1.0 = perfect detection)")

    if auc_max > 0.70:
        print(f"   [OK] GOOD: System performs significantly above random baseline")
    elif auc_max > 0.55:
        print(f"   [WARN]  MODERATE: System shows some discrimination ability")
    else:
        print(f"   [WARN]  LOW: System needs improvement for insider detection")

    return user_risk, auc_max


def per_insider_analysis(user_risk):
    """Detailed per-insider risk ranking."""
    print("\n" + "="*70)
    print("2. PER-INSIDER DETAILED BREAKDOWN")
    print("="*70)

    n_total = len(user_risk)
    ranked  = user_risk.sort_values('max_risk', ascending=False).reset_index(drop=True)
    ranked['rank'] = ranked.index + 1

    print(f"\n   {'User':<6} {'Scen':<5} {'Scenario Type':<44} "
          f"{'MaxRisk':<9} {'Rank':<12} {'Top%':<8} {'Status'}")
    print(f"   " + "-"*105)

    summary = {'detected_top1': 0, 'detected_top5': 0, 'detected_top10': 0}

    for user in INSIDER_USERS:
        scenario    = INSIDER_SCENARIOS[user]
        scen_name   = SCENARIO_NAMES[scenario][:44]
        user_data   = ranked[ranked['user'] == user]

        if len(user_data) == 0:
            print(f"   {user:<6} {scenario:<5} {scen_name:<44} NOT IN RESULTS")
            continue

        max_risk = user_data['max_risk'].iloc[0]
        rank     = user_data['rank'].iloc[0]
        top_pct  = 100 * rank / n_total

        if top_pct <= 1.0:
            status = "[OK] TOP 1%"
            summary['detected_top1'] += 1
            summary['detected_top5'] += 1
            summary['detected_top10'] += 1
        elif top_pct <= 5.0:
            status = "[WARN] TOP 5%"
            summary['detected_top5'] += 1
            summary['detected_top10'] += 1
        elif top_pct <= 10.0:
            status = "[HIGH] TOP 10%"
            summary['detected_top10'] += 1
        else:
            status = "[ERROR] MISSED"

        print(f"   {user:<6} {scenario:<5} {scen_name:<44} "
              f"{max_risk:<9.3f} {rank}/{n_total:<8} {top_pct:<8.1f}% {status}")

    print(f"\n   Detection Summary:")
    print(f"     In top  1% (>=99th pct): {summary['detected_top1']}/5 insiders")
    print(f"     In top  5% (>=95th pct): {summary['detected_top5']}/5 insiders")
    print(f"     In top 10% (>=90th pct): {summary['detected_top10']}/5 insiders")

    return summary


def threshold_analysis(user_risk):
    """Detection performance at different alert thresholds."""
    print("\n" + "="*70)
    print("3. THRESHOLD-BASED DETECTION PERFORMANCE")
    print("="*70)

    n_total   = len(user_risk)
    n_insider = len(INSIDER_USERS)

    print(f"\n   {'Threshold':<16} {'Flagged':<10} {'Insiders':<12} "
          f"{'Recall':<10} {'Precision':<12} {'F1':<8}")
    print(f"   " + "-"*70)

    percentiles = [90, 95, 99, 99.5, 99.9]

    results = []
    for pct in percentiles:
        threshold = np.percentile(user_risk['max_risk'], pct)
        flagged   = user_risk[user_risk['max_risk'] >= threshold]
        ins_found = flagged['is_insider'].sum()
        recall    = ins_found / n_insider if n_insider > 0 else 0
        precision = ins_found / len(flagged) if len(flagged) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        label = f"Top {100-pct:.1f}% (>={threshold:.2f})"
        print(f"   {label:<16} {len(flagged):<10} {ins_found}/{n_insider:<9} "
              f"{recall:<10.2f} {precision:<12.2f} {f1:.2f}")

        results.append({
            'threshold': threshold,
            'flagged':   len(flagged),
            'insiders':  ins_found,
            'recall':    recall,
            'precision': precision,
            'f1':        f1
        })

    return results


def early_warning_analysis(df, user_risk):
    """
    Check whether the drift model predicted escalation BEFORE
    the insiders' labeled malicious days.
    """
    if 'drift_score' not in df.columns:
        print("\n[WARN]  Skipping early warning analysis (no drift_score column)")
        return {}

    print("\n" + "="*70)
    print("4. EARLY WARNING ANALYSIS")
    print("   Did drift detection fire BEFORE labeled malicious days?")
    print("="*70)

    early_warning_results = {}

    for user in INSIDER_USERS:
        scenario         = INSIDER_SCENARIOS[user]
        malicious_days   = INSIDER_MALICIOUS_DAYS[user]
        first_malicious  = min(malicious_days)

        user_df     = df[df['user'] == user].sort_values('day')
        pre_malicious = user_df[user_df['day'] < first_malicious]

        print(f"\n   User {user} -- Scenario {scenario}: {SCENARIO_NAMES[scenario]}")
        print(f"     Total observation days: {len(user_df)}")
        print(f"     First labeled malicious day: Day {first_malicious}")
        print(f"     Pre-malicious days available: {len(pre_malicious)}")

        if len(pre_malicious) == 0:
            print(f"     [WARN]  No pre-malicious data available")
            continue

        max_pre_risk  = pre_malicious['risk_score'].max()
        max_pre_drift = pre_malicious['drift_score'].max()

        print(f"     Max risk score (before malicious): {max_pre_risk:.3f}")
        print(f"     Max drift score (before malicious): {max_pre_drift:.3f}")

        # Check for breach warnings
        if 'will_breach' in pre_malicious.columns:
            breach_warnings = pre_malicious[pre_malicious['will_breach'] == True]
            print(f"     Breach warnings issued: {len(breach_warnings)}")

            if len(breach_warnings) > 0:
                earliest = breach_warnings['day'].min()
                advance  = first_malicious - earliest
                print(f"     [OK] Earliest warning: Day {earliest} "
                      f"({advance} days BEFORE first malicious day)")
                early_warning_results[user] = {
                    'advance_days': advance,
                    'earliest_warning_day': earliest
                }
            else:
                print(f"     [ERROR] No advance breach warnings issued")
                early_warning_results[user] = {'advance_days': 0}
        else:
            early_warning_results[user] = {
                'max_pre_risk': float(max_pre_risk),
                'max_pre_drift': float(max_pre_drift)
            }

        # On-malicious-day risk
        on_malicious = user_df[user_df['day'].isin(malicious_days)]
        if len(on_malicious) > 0:
            max_mal_risk = on_malicious['risk_score'].max()
            print(f"     Max risk score (on malicious days): {max_mal_risk:.3f}")
            if max_mal_risk > max_pre_risk:
                lift = max_mal_risk - max_pre_risk
                print(f"     [OK] Risk ELEVATED on malicious days (+{lift:.3f})")
            else:
                print(f"     [WARN]  Risk not elevated on malicious days")

    return early_warning_results


def composite_risk_feature_analysis(df):
    """
    Show whether the new composite risk features (from feature_engineering.py)
    are elevated for insider users vs normal users.
    """
    composite_features = [
        'exfiltration_score', 'policy_violation_score',
        'timing_anomaly_score', 'insider_risk_composite',
        'files_to_usb', 'hack_site_visits', 'job_search_visits',
        'cloud_storage_visits', 'after_hours_usb', 'external_email_count'
    ]

    available = [f for f in composite_features if f in df.columns]
    if not available:
        print("\n[WARN]  Composite features not in pipeline output "
              "(run with semantic features enabled)")
        return

    print("\n" + "="*70)
    print("5. SEMANTIC FEATURE ELEVATION (Insider vs Normal)")
    print("="*70)

    insider_mask = df['user'].isin(INSIDER_USERS)

    print(f"\n   {'Feature':<35} {'Insider u':<12} {'Normal u':<12} {'Ratio':<8} {'Signal'}")
    print(f"   " + "-"*80)

    for feat in available:
        ins_mean  = df.loc[insider_mask, feat].mean()
        norm_mean = df.loc[~insider_mask, feat].mean()
        ratio     = ins_mean / (norm_mean + 1e-9)
        signal    = "*** STRONG ***" if ratio > 5 else "** MODERATE **" if ratio > 2 else ""
        print(f"   {feat:<35} {ins_mean:<12.4f} {norm_mean:<12.4f} "
              f"{ratio:<8.1f}x {signal}")


def save_validation_report(user_risk, auc, summary, threshold_results,
                            early_warnings, output_dir):
    """Save a text validation report."""
    report_path = os.path.join(output_dir, 'validation_report.txt')

    with open(report_path, 'w') as f:
        f.write("PIRS INSIDER VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Defense Date: March 12, 2026\n")
        f.write(f"Ground Truth: 5 CERT r6.2 insider users\n\n")

        f.write("ROC-AUC RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"User-level ROC-AUC (max risk): {auc:.4f}\n")
        f.write(f"  0.5 = random baseline, 1.0 = perfect\n\n")

        f.write("DETECTION SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  In top  1%: {summary['detected_top1']}/5 insiders\n")
        f.write(f"  In top  5%: {summary['detected_top5']}/5 insiders\n")
        f.write(f"  In top 10%: {summary['detected_top10']}/5 insiders\n\n")

        f.write("PER-INSIDER RESULTS\n")
        f.write("-" * 40 + "\n")
        n_total = len(user_risk)
        ranked  = user_risk.sort_values('max_risk', ascending=False).reset_index(drop=True)
        ranked['rank'] = ranked.index + 1
        for user in INSIDER_USERS:
            u = ranked[ranked['user'] == user]
            if len(u) > 0:
                r = u.iloc[0]
                top_pct = 100 * r['rank'] / n_total
                f.write(f"  User {user} (Scenario {INSIDER_SCENARIOS[user]}): "
                        f"max_risk={r['max_risk']:.3f}, "
                        f"rank={r['rank']}/{n_total} ({top_pct:.1f}%)\n")

        f.write("\nEARLY WARNING RESULTS\n")
        f.write("-" * 40 + "\n")
        for user, result in early_warnings.items():
            adv = result.get('advance_days', 0)
            f.write(f"  User {user}: {adv} days advance warning\n")

    print(f"\n[SAVE] Validation report saved: {report_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_validation():
    """Main validation function."""
    print("\n" + "="*70)
    print("LAYER VALIDATION: INSIDER THREAT DETECTION EVALUATION")
    print("="*70)
    print(f"\nGround truth: {len(INSIDER_USERS)} insider users {INSIDER_USERS}")

    start_time = time.time()

    # Load data
    df, source_label = load_pipeline_output()

    # 1. User-level ROC-AUC analysis
    user_risk, auc = user_level_analysis(df)

    # 2. Per-insider breakdown
    summary = per_insider_analysis(user_risk)

    # 3. Threshold analysis
    threshold_results = threshold_analysis(user_risk)

    # 4. Early warning analysis
    early_warnings = early_warning_analysis(df, user_risk)

    # 5. Composite feature analysis
    composite_risk_feature_analysis(df)

    # Save report
    save_validation_report(user_risk, auc, summary, threshold_results,
                            early_warnings, PIRSConfig.OUTPUT_DIR)

    elapsed = time.time() - start_time

    print(f"\n" + "="*70)
    print(f"[OK] VALIDATION COMPLETE")
    print(f"   ROC-AUC:     {auc:.4f}")
    print(f"   Top-1%:  {summary['detected_top1']}/5 insiders detected")
    print(f"   Top-5%:  {summary['detected_top5']}/5 insiders detected")
    print(f"   Top-10%: {summary['detected_top10']}/5 insiders detected")
    print(f"   Time: {elapsed:.1f}s")
    print("="*70 + "\n")

    return user_risk, auc, summary


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        user_risk, auc, summary = run_validation()
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
