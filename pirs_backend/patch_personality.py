"""
Patch pirs_complete.csv with personality scores computed from data_extracted.csv.
Fixes the user ID mismatch (data_processed.csv uses integer IDs, data_extracted.csv uses string IDs).
Run: python patch_personality.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import PIRSConfig

OUTPUT_DIR = PIRSConfig.OUTPUT_DIR

SEMANTIC_GROUPS = {
    'COMPLIANT':  ['n_logon', 'work_hour_ratio', 'n_email_sent', 'n_file_ops', 'is_weekday'],
    'SOCIAL':     ['n_email_sent', 'n_email_recv', 'n_email_external', 'n_social_media', 'n_email_with_att'],
    'CAREFULL':   ['work_hour_ratio', 'n_unique_pcs', 'n_file_doc', 'n_file_ops', 'n_logon'],
    'RISK_TAKER': ['n_afterhour_usb', 'n_file_to_usb', 'n_email_external', 'n_job_sites',
                   'after_hours_ratio', 'n_hack_sites', 'n_email_bcc_ext'],
    'AUTONOMOUS': ['n_http', 'n_cloud_upload', 'n_file_ops', 'n_hack_sites', 'n_afterhour_http'],
}

def compute_personality(data_path):
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows, {df['user'].nunique():,} users")

    # Add week column
    if 'week' not in df.columns and 'day' in df.columns:
        df['week'] = df['day'] // 7

    # Compute each dimension score per row
    for dim, features in SEMANTIC_GROUPS.items():
        available = [f for f in features if f in df.columns]
        if not available:
            print(f"  [WARN] {dim}: no features found, defaulting to 0")
            df[dim] = 0.0
            continue
        vals = df[available].values.astype(float)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        raw = vals.mean(axis=1)
        vmin, vmax = raw.min(), raw.max()
        if vmax > vmin:
            df[dim] = (raw - vmin) / (vmax - vmin)
        else:
            df[dim] = 0.0
        print(f"  {dim} ({len(available)} feats): mean={df[dim].mean():.3f} std={df[dim].std():.3f}")

    # Aggregate by user: use their most recent week
    latest_week = df.groupby('user')['week'].transform('max')
    df_latest = df[df['week'] == latest_week].copy()

    # Average personality dims per user
    agg_dict = {dim: 'mean' for dim in SEMANTIC_GROUPS}
    user_personality = df_latest.groupby('user').agg(agg_dict).reset_index()

    # Assign primary dimension
    dim_matrix = user_personality[list(SEMANTIC_GROUPS.keys())].values
    primary_idx = dim_matrix.argmax(axis=1)
    user_personality['PRIMARY_DIMENSION'] = [list(SEMANTIC_GROUPS.keys())[i] for i in primary_idx]

    print(f"\n  User personality computed for {len(user_personality):,} users")
    print(f"  Primary dimension distribution:")
    print(user_personality['PRIMARY_DIMENSION'].value_counts().to_string())

    return user_personality


def patch_pirs_complete(user_personality):
    master_path = os.path.join(OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['master'])
    print(f"\nLoading {master_path}...")
    master = pd.read_csv(master_path)
    print(f"  {len(master):,} rows")

    # Drop existing personality columns (they are all NaN)
    drop_cols = ['PRIMARY_DIMENSION'] + list(SEMANTIC_GROUPS.keys())
    master = master.drop(columns=[c for c in drop_cols if c in master.columns])

    # Merge personality scores
    user_personality['user'] = user_personality['user'].astype(str)
    master['user'] = master['user'].astype(str)
    master = master.merge(user_personality[['user', 'PRIMARY_DIMENSION'] + list(SEMANTIC_GROUPS.keys())],
                          on='user', how='left')

    nan_count = master[list(SEMANTIC_GROUPS.keys())].isna().any(axis=1).sum()
    print(f"  Rows with NaN personality after patch: {nan_count}")
    print(f"  Sample COMPLIANT values: {master['COMPLIANT'].head(5).values}")

    master.to_csv(master_path, index=False)
    print(f"\n[OK] Patched {master_path}")
    print(f"  Total rows: {len(master):,}, columns: {len(master.columns)}")


if __name__ == '__main__':
    data_path = os.path.join(OUTPUT_DIR, 'data_extracted.csv')
    user_personality = compute_personality(data_path)
    patch_pirs_complete(user_personality)
    print("\nDone. Restart the dashboard to see updated personality radar charts.")
