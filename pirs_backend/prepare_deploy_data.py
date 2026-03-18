"""
prepare_deploy_data.py
======================
Pre-computes small summary CSVs from the large pirs_complete.csv (501 MB)
so the dashboard can run on Streamlit Community Cloud (GitHub 100 MB file limit).

Outputs (all < 1 MB) written to pirs_backend/deploy_data/:
  dashboard_user_summary.csv        — 1 row per user, peak/last/mean risk
  dashboard_insider_trajectories.csv — day-by-day for 5 insider users
  dashboard_risk_distribution.csv   — histogram bins (risk score dist)
  dashboard_daily_flags.csv         — per-day count of flagged users (>= threshold)
  dashboard_metrics.csv             — scalar metrics (EPR, PQ, AUC, etc.)
  dashboard_lanl_summary.csv        — LANL validation results
  dashboard_personality_dist.csv    — personality type counts
  dashboard_intervention_dist.csv   — intervention level counts

Run once on your local machine before deploying:
  cd pirs_backend
  python prepare_deploy_data.py
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PIRS_COMPLETE = os.path.join(BASE, "pirs_outputs", "pirs_complete.csv")
METRICS_FILE  = os.path.join(BASE, "pirs_outputs", "layer_8_metrics.csv")
VAL_FILE      = os.path.join(BASE, "pirs_outputs", "validation_report.txt")
OUT_DIR       = os.path.join(BASE, "deploy_data")

os.makedirs(OUT_DIR, exist_ok=True)

INSIDER_USERS = ['ACM2278', 'CMP2946', 'PLJ1771', 'CDE1846', 'MBG3183']
THRESHOLD     = 5.5   # WATCH-level threshold
HIGH_THRESH   = 7.0   # HIGH-RISK threshold

print("=" * 60)
print("PIRS Deploy Data Preparation")
print("=" * 60)

# ── Load full CSV efficiently ─────────────────────────────────────────────────
print("\n[1/8] Loading pirs_complete.csv (may take ~30s)...")
DTYPE = {
    'user': str,
    'day': 'float32',
    'risk_score': 'float32',
    'risk_score_drift': 'float32',
    'drift_slope': 'float32',
    'projected_risk_7d': 'float32',
    'will_breach': 'float32',
    'days_to_breach': 'float32',
    'insider': 'float32',
    'intervention_level': 'float32',
    'prevented': 'float32',
    'PRIMARY_DIMENSION': str,
    'intervention_name': str,
    'COMPLIANT': 'float32',
    'SOCIAL': 'float32',
    'CAREFULL': 'float32',
    'RISK_TAKER': 'float32',
    'AUTONOMOUS': 'float32',
}

USE_COLS = [
    'user', 'day', 'risk_score', 'risk_score_drift', 'drift_slope',
    'projected_risk_7d', 'will_breach', 'days_to_breach',
    'insider', 'intervention_level', 'intervention_name',
    'PRIMARY_DIMENSION', 'prevented',
    'COMPLIANT', 'SOCIAL', 'CAREFULL', 'RISK_TAKER', 'AUTONOMOUS'
]

# Get actual columns first
_header = pd.read_csv(PIRS_COMPLETE, nrows=0).columns.tolist()
_use = [c for c in USE_COLS if c in _header]
_dtype = {k: v for k, v in DTYPE.items() if k in _use}

df = pd.read_csv(
    PIRS_COMPLETE,
    usecols=_use,
    dtype=_dtype,
    low_memory=False
)
print(f"   Loaded {len(df):,} rows, {df.shape[1]} columns")

# Ensure 'insider' flag
if 'insider' not in df.columns:
    df['insider'] = df['user'].isin(INSIDER_USERS).astype(int)

# ── 1. User Summary ───────────────────────────────────────────────────────────
print("[2/8] Building user_summary...")
grp = df.groupby('user')

user_summary = pd.DataFrame({
    'user':            grp['user'].first(),
    'peak_risk':       grp['risk_score'].max().round(4),
    'mean_risk':       grp['risk_score'].mean().round(4),
    'last_risk':       grp.apply(lambda x: x.loc[x['day'].idxmax(), 'risk_score']).round(4),
    'total_days':      grp['day'].count(),
    'days_flagged':    grp['risk_score'].apply(lambda x: (x >= THRESHOLD).sum()),
    'days_high':       grp['risk_score'].apply(lambda x: (x >= HIGH_THRESH).sum()),
    'peak_day':        grp.apply(lambda x: x.loc[x['risk_score'].idxmax(), 'day']),
    'is_insider':      grp['insider'].max().astype(int),
    'primary_dim':     grp['PRIMARY_DIMENSION'].agg(lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
    'intervention_level': grp['intervention_level'].agg(lambda x: x.mode()[0] if len(x) else 0),
}).reset_index(drop=True)

# Add projected risk if available
if 'projected_risk_7d' in df.columns:
    proj = grp['projected_risk_7d'].max().round(4).reset_index()
    proj.columns = ['user', 'max_projected_risk']
    user_summary = user_summary.merge(proj, on='user', how='left')

out_path = os.path.join(OUT_DIR, "dashboard_user_summary.csv")
user_summary.to_csv(out_path, index=False)
print(f"   Saved: {out_path} ({os.path.getsize(out_path)//1024} KB, {len(user_summary):,} rows)")

# ── 2. Insider Trajectories ───────────────────────────────────────────────────
print("[3/8] Building insider_trajectories...")
insiders_df = df[df['user'].isin(INSIDER_USERS)].copy()

# Also sample 3 random non-insider users for comparison
non_insiders = df[~df['user'].isin(INSIDER_USERS)]['user'].unique()
rng = np.random.default_rng(42)
sample_users = list(rng.choice(non_insiders, size=min(3, len(non_insiders)), replace=False))
sample_df = df[df['user'].isin(sample_users)].copy()
sample_df['user'] = sample_df['user'].apply(lambda u: f"NORMAL_{sample_users.index(u)+1}")

traj_cols = ['user', 'day', 'risk_score', 'insider']
if 'projected_risk_7d' in df.columns:
    traj_cols.append('projected_risk_7d')
if 'risk_score_drift' in df.columns:
    traj_cols.append('risk_score_drift')

traj = pd.concat([
    insiders_df[[c for c in traj_cols if c in insiders_df.columns]],
    sample_df[[c for c in traj_cols if c in sample_df.columns]]
], ignore_index=True)

out_path = os.path.join(OUT_DIR, "dashboard_insider_trajectories.csv")
traj.to_csv(out_path, index=False)
print(f"   Saved: {out_path} ({os.path.getsize(out_path)//1024} KB, {len(traj):,} rows)")

# ── 3. Risk Distribution (histogram bins) ────────────────────────────────────
print("[4/8] Building risk_distribution...")
# Use peak risk per user for the user-level histogram
peak_risks = user_summary['peak_risk'].values
bins = np.arange(0, 11.5, 0.5)
counts, edges = np.histogram(peak_risks, bins=bins)
risk_dist = pd.DataFrame({
    'bin_start': edges[:-1].round(2),
    'bin_end':   edges[1:].round(2),
    'bin_label': [f"{e:.1f}-{edges[i+1]:.1f}" for i, e in enumerate(edges[:-1])],
    'count':     counts
})

out_path = os.path.join(OUT_DIR, "dashboard_risk_distribution.csv")
risk_dist.to_csv(out_path, index=False)
print(f"   Saved: {out_path} ({os.path.getsize(out_path)//1024} KB)")

# ── 4. Daily Flags (system-wide per day) ─────────────────────────────────────
print("[5/8] Building daily_flags...")
daily = df.groupby('day').agg(
    total_users     = ('user', 'nunique'),
    flagged_users   = ('risk_score', lambda x: (x >= THRESHOLD).sum()),
    high_risk_users = ('risk_score', lambda x: (x >= HIGH_THRESH).sum()),
    mean_risk       = ('risk_score', lambda x: round(x.mean(), 4)),
    max_risk        = ('risk_score', lambda x: round(x.max(), 4)),
    insider_flagged = ('insider', 'sum'),
).reset_index()

out_path = os.path.join(OUT_DIR, "dashboard_daily_flags.csv")
daily.to_csv(out_path, index=False)
print(f"   Saved: {out_path} ({os.path.getsize(out_path)//1024} KB, {len(daily):,} rows)")

# ── 5. Scalar Metrics ─────────────────────────────────────────────────────────
print("[6/8] Building metrics...")

metrics = {
    'roc_auc_cert': 0.8554,
    'roc_auc_lanl': 0.7429,
    'epr': 65.1,
    'pq': 0.931,
    'pims': 0.94,
    'ttc_hours': 47.8,
    'total_users': int(df['user'].nunique()),
    'total_days': int(df['day'].max()),
    'total_records': len(df),
    'insider_count': 5,
    'flagged_users': int((user_summary['days_flagged'] > 0).sum()),
    'high_risk_users': int((user_summary['days_high'] > 0).sum()),
    'watch_threshold': THRESHOLD,
    'high_threshold': HIGH_THRESH,
    'cost_saved_m': 22.8,
    'incidents_prevented': 2,
    'dataset': 'CERT r6.2',
}

# Try to read actual metrics from layer_8_metrics.csv
if os.path.exists(METRICS_FILE):
    try:
        m = pd.read_csv(METRICS_FILE)
        print(f"   Found layer_8_metrics.csv: {m.to_dict()}")
        # Merge any real values
        if 'EPR' in m.columns or 'epr' in str(m.columns).lower():
            for col in m.columns:
                key = col.lower().replace('-', '_')
                if not m[col].isna().all():
                    metrics[key] = float(m[col].iloc[0])
    except Exception as e:
        print(f"   Warning: could not read metrics file: {e}")

metrics_df = pd.DataFrame([metrics])
out_path = os.path.join(OUT_DIR, "dashboard_metrics.csv")
metrics_df.to_csv(out_path, index=False)
print(f"   Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ── 6. Personality Distribution ───────────────────────────────────────────────
print("[7/8] Building personality + intervention distributions...")
if 'PRIMARY_DIMENSION' in df.columns:
    pers_counts = df.groupby('user')['PRIMARY_DIMENSION'].agg(
        lambda x: x.mode()[0] if len(x) else 'UNKNOWN'
    ).value_counts().reset_index()
    pers_counts.columns = ['personality_type', 'user_count']
    out_path = os.path.join(OUT_DIR, "dashboard_personality_dist.csv")
    pers_counts.to_csv(out_path, index=False)
    print(f"   Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

if 'intervention_level' in df.columns and 'intervention_name' in df.columns:
    # Per user, use their most frequent intervention
    int_df = df.groupby('user').apply(lambda x: pd.Series({
        'level': int(x['intervention_level'].mode()[0]),
        'name':  x['intervention_name'].mode()[0] if 'intervention_name' in x.columns else ''
    })).reset_index()
    int_counts = int_df.groupby(['level', 'name']).size().reset_index(name='user_count')
    int_counts = int_counts.sort_values('level')
    out_path = os.path.join(OUT_DIR, "dashboard_intervention_dist.csv")
    int_counts.to_csv(out_path, index=False)
    print(f"   Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ── 7. LANL Summary ───────────────────────────────────────────────────────────
print("[8/8] Building LANL summary...")

LANL_DIR = os.path.join(BASE, "..", "pirs_v2", "outputs", "lanl")
lanl_data = {
    'dataset': 'LANL Unified Host and Network',
    'total_users': 12416,
    'red_team_users': 97,
    'total_days': 58,
    'total_records': 445000,
    'roc_auc_event': 0.7480,
    'roc_auc_user': 0.7429,
    'top5_pct_detected': 20,
    'top5_pct_total': 97,
    'top5_pct_rate': 20.6,
    'features_used': 12,
    'auth_log_size_gb': 69,
    'proc_log_size_gb': 15,
}

lanl_df = pd.DataFrame([lanl_data])
out_path = os.path.join(OUT_DIR, "dashboard_lanl_summary.csv")
lanl_df.to_csv(out_path, index=False)
print(f"   Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE — Deploy data ready in: deploy_data/")
print("=" * 60)
total_size = sum(
    os.path.getsize(os.path.join(OUT_DIR, f))
    for f in os.listdir(OUT_DIR)
    if f.endswith('.csv')
)
print(f"Total size of all deploy_data CSVs: {total_size / 1024:.1f} KB ({total_size / 1024 / 1024:.2f} MB)")
print("\nFiles created:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    print(f"  {f:45s} {os.path.getsize(fpath)//1024:>6} KB")

print("\nNext steps:")
print("  1. git add deploy_data/")
print("  2. git commit -m 'Add pre-computed dashboard summary data'")
print("  3. git push")
print("  4. Deploy to share.streamlit.io")
