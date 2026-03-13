"""
lanl_pipeline.py
PIRS Layers 1-4 adapted for LANL dataset.
Validates drift detection and anomaly scoring against redteam.txt labels.

Pipeline:
  Layer 1: Personal behavioral baseline per user
  Layer 2: Drift detection (7-day rolling window)
  Layer 3: Isolation Forest anomaly scoring
  Layer 4: Risk scoring + validation vs red team users
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
import os

warnings.filterwarnings("ignore")

FEATURES_PATH = r"C:\Users\rosha\Documents\PIRS\lanl_data\lanl_features.csv"
REDTEAM_PATH = r"C:\Users\rosha\Documents\PIRS\lanl_data\redteam_parsed.csv"
OUTPUT_DIR = r"C:\Users\rosha\Documents\PIRS\lanl_data\lanl_outputs"

FEATURE_COLS = [
    "n_auth", "n_logon", "n_logoff", "n_failed",
    "n_unique_dst", "n_unique_src",
    "n_workhour", "n_afterhour",
    "n_ntlm", "n_kerberos",
    "n_network", "n_service", "n_lateral",
    "fail_rate", "afterhour_rate", "lateral_rate", "ntlm_rate", "dst_diversity",
]

DRIFT_WINDOW = 7    # days for rolling baseline
MIN_HISTORY = 3     # minimum days of history to score a user


def load_data():
    print("Loading LANL features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  {len(df):,} user-day records | {df['user'].nunique():,} users | {df['day'].nunique()} days")

    print("Loading redteam labels...")
    rt = pd.read_csv(REDTEAM_PATH)
    redteam_users = set(rt["user"].unique())
    redteam_events = set(zip(rt["user"], rt["day"]))
    print(f"  {len(rt):,} events | {len(redteam_users)} unique red team users")

    return df, redteam_users, redteam_events


# ── Layer 1: Personal Baseline ──────────────────────────────────────────────

def compute_personal_baseline(df):
    """Compute rolling mean and std baseline per user."""
    print("\nLayer 1: Computing personal behavioral baselines...")

    df = df.sort_values(["user", "day"]).copy()
    baseline_records = []

    for user, udf in df.groupby("user"):
        if len(udf) < MIN_HISTORY:
            continue
        udf = udf.sort_values("day").reset_index(drop=True)
        for col in FEATURE_COLS:
            if col not in udf.columns:
                continue
            udf[f"baseline_{col}"] = (
                udf[col].expanding(min_periods=1).mean().shift(1)
            )
            udf[f"std_{col}"] = (
                udf[col].expanding(min_periods=1).std().shift(1).fillna(0)
            )
        baseline_records.append(udf)

    result = pd.concat(baseline_records, ignore_index=True)
    print(f"  Baseline computed for {result['user'].nunique():,} users")
    return result


# ── Layer 2: Drift Detection ─────────────────────────────────────────────────

def compute_drift_score(df):
    """
    Drift = average z-score deviation from personal baseline
    over a rolling 7-day window.
    """
    print("\nLayer 2: Computing behavioral drift scores...")

    df = df.copy()
    drift_scores = []

    for user, udf in df.groupby("user"):
        udf = udf.sort_values("day").reset_index(drop=True)
        z_cols = []
        for col in FEATURE_COLS:
            bc = f"baseline_{col}"
            sc = f"std_{col}"
            if bc not in udf.columns:
                continue
            z = (udf[col] - udf[bc]) / (udf[sc] + 0.01)
            z = z.clip(-5, 5).fillna(0)
            z_cols.append(z)

        if not z_cols:
            continue

        udf["drift_score"] = pd.concat(z_cols, axis=1).abs().mean(axis=1)

        # Rolling max drift (7-day window)
        udf["drift_rolling"] = (
            udf["drift_score"].rolling(window=DRIFT_WINDOW, min_periods=1).max()
        )

        drift_scores.append(udf)

    result = pd.concat(drift_scores, ignore_index=True)
    print(f"  Drift scores computed | mean={result['drift_score'].mean():.3f}")
    return result


# ── Layer 3: Isolation Forest Anomaly Scoring ────────────────────────────────

def compute_anomaly_score(df):
    """
    Train Isolation Forest on normal user-days (days 1-14 = training).
    Score all user-days.
    """
    print("\nLayer 3: Isolation Forest anomaly scoring...")

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train on first half of days
    max_day = df["day"].max()
    train_mask = df["day"] <= (max_day // 2)
    X_train = X_scaled[train_mask]

    print(f"  Training on {train_mask.sum():,} records (days 1-{max_day//2})")
    print(f"  Scoring all {len(df):,} records...")

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)

    # Score: flip sign so higher = more anomalous
    df = df.copy()
    df["anomaly_score"] = -iso.score_samples(X_scaled)
    print(f"  Anomaly scores | mean={df['anomaly_score'].mean():.3f}")
    return df


# ── Layer 4: Risk Scoring + Validation ──────────────────────────────────────

def compute_risk_score(df):
    """Combine drift + anomaly into final risk score (0-10)."""
    print("\nLayer 4: Computing composite risk scores...")

    df = df.copy()

    # Normalize to 0-1
    def norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    drift_norm = norm(df.get("drift_rolling", df.get("drift_score", pd.Series(0, index=df.index))))
    anomaly_norm = norm(df["anomaly_score"])

    # Weighted combination: drift 50%, anomaly 50%
    df["risk_score"] = (0.5 * drift_norm + 0.5 * anomaly_norm) * 10.0

    print(f"  Risk score | mean={df['risk_score'].mean():.2f} | max={df['risk_score'].max():.2f}")
    return df


def validate(df, redteam_users, redteam_events):
    """Compute ROC-AUC at user-day level and user level."""
    print("\nValidation: ROC-AUC vs Red Team Labels")
    print("=" * 50)

    df = df.copy()

    # Label 1: this user-day is a red team event
    df["label_event"] = df.apply(
        lambda r: 1 if (r["user"], int(r["day"])) in redteam_events else 0,
        axis=1
    )

    # Label 2: this user is ANY red team user
    df["label_user"] = df["user"].apply(lambda u: 1 if u in redteam_users else 0)

    n_pos = df["label_event"].sum()
    n_neg = (df["label_event"] == 0).sum()
    print(f"  User-day labels: {n_pos} red team events, {n_neg} normal events")

    results = {}

    # ROC-AUC at event level
    if n_pos > 0:
        auc_event = roc_auc_score(df["label_event"], df["risk_score"])
        results["roc_auc_event"] = auc_event
        print(f"  ROC-AUC (event level):  {auc_event:.4f}")

    # ROC-AUC at user level (max risk per user)
    user_risk = df.groupby("user")["risk_score"].max().reset_index()
    user_risk["label"] = user_risk["user"].apply(
        lambda u: 1 if u in redteam_users else 0
    )
    n_rt = user_risk["label"].sum()
    n_norm = (user_risk["label"] == 0).sum()
    print(f"\n  User-level labels: {n_rt} red team users, {n_norm} normal users")

    if n_rt > 0:
        auc_user = roc_auc_score(user_risk["label"], user_risk["risk_score"])
        results["roc_auc_user"] = auc_user
        print(f"  ROC-AUC (user level):   {auc_user:.4f}")

    # Top-N detection
    for pct in [5, 10, 20]:
        n_top = max(1, int(len(user_risk) * pct / 100))
        top_users = set(
            user_risk.nlargest(n_top, "risk_score")["user"].tolist()
        )
        detected = len(top_users & redteam_users)
        print(f"  Top {pct:2d}% ({n_top:4d} users): {detected}/{n_rt} red team users detected "
              f"({100*detected/max(n_rt,1):.1f}%)")
        results[f"top_{pct}pct_detection"] = detected / max(n_rt, 1)

    # Precision-Recall AUC
    if n_pos > 0:
        prec, rec, _ = precision_recall_curve(df["label_event"], df["risk_score"])
        pr_auc = auc(rec, prec)
        results["pr_auc"] = pr_auc
        print(f"\n  PR-AUC (event level):   {pr_auc:.4f}")

    return results, user_risk


def save_outputs(df, user_risk, results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "lanl_scored.csv"), index=False)
    user_risk.to_csv(os.path.join(OUTPUT_DIR, "lanl_user_risk.csv"), index=False)

    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(OUTPUT_DIR, "lanl_results.csv"), index=False)

    print(f"\nOutputs saved to {OUTPUT_DIR}")


def run():
    print("=" * 60)
    print("PIRS LANL Validation Pipeline (Layers 1-4)")
    print("=" * 60)

    df, redteam_users, redteam_events = load_data()

    df = compute_personal_baseline(df)
    df = compute_drift_score(df)
    df = compute_anomaly_score(df)
    df = compute_risk_score(df)

    results, user_risk = validate(df, redteam_users, redteam_events)

    save_outputs(df, user_risk, results)

    print("\n" + "=" * 60)
    print("LANL Pipeline Complete")
    print("=" * 60)
    print("\nKey Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    run()
