"""
PIRS V2 - LANL EXTRACTOR
=========================
Extracts per-user-per-day behavioral features from LANL auth.txt and proc.txt.

auth.txt fields: time, src_user@domain, dst_user@domain, src_computer,
                 dst_computer, auth_type, logon_type, auth_orientation, success/fail

proc.txt fields: time, user@domain, computer, process_name, start/end

Output: lanl_features.csv
  Columns: user, day, + behavioral features + redteam label
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import LANLConfig

cfg = LANLConfig()

# Known attack tool process names (partial match)
SUSPICIOUS_PROCS = [
    'mimikatz', 'psexec', 'wce', 'fgdump', 'pwdump',
    'meterpreter', 'cobalt', 'empire', 'metasploit',
    'netcat', 'nmap', 'scanner', 'exploit', 'payload'
]


def seconds_to_day(time_series):
    """Convert LANL timestamps (seconds from epoch) to day numbers (1-indexed)."""
    secs = pd.to_numeric(time_series, errors='coerce')
    return ((secs // 86400) + 1).astype('Int64')


def seconds_to_hour(time_series):
    secs = pd.to_numeric(time_series, errors='coerce')
    return (secs % 86400) // 3600


def is_after_hours_lanl(hour_series):
    return (hour_series < cfg.WORK_START_SEC // 3600) | \
           (hour_series >= cfg.WORK_END_SEC   // 3600)


def clean_user(user_series):
    """Strip domain suffix: USER@DOMAIN -> USER"""
    return user_series.astype(str).str.split('@').str[0].str.upper()


# ---------------------------------------------------------------------------
# LOAD REDTEAM GROUND TRUTH
# ---------------------------------------------------------------------------

def load_redteam():
    """
    redteam.txt fields: time, user@domain, src_computer, dst_computer
    Returns: set of (user, day) pairs that are red team activity
    """
    print("[REDTEAM] Loading ground truth...")
    df = pd.read_csv(cfg.REDTEAM_FILE, header=None,
                     names=['time', 'user', 'src_computer', 'dst_computer'])
    df['user'] = clean_user(df['user'])
    df['day']  = seconds_to_day(df['time'])
    redteam_set = set(zip(df['user'], df['day']))
    print(f"  {len(df):,} red team events | {df['user'].nunique()} users | "
          f"{df['day'].nunique()} days")
    return redteam_set, df


# ---------------------------------------------------------------------------
# EXTRACT AUTH FEATURES
# ---------------------------------------------------------------------------

def extract_auth_features():
    """
    Extract per-user-per-day features from auth.txt.
    Reads in chunks to handle the ~1.6B row file.
    """
    print("[AUTH] Extracting authentication features (large file, be patient)...")

    auth_cols = ['time', 'src_user', 'dst_user', 'src_computer', 'dst_computer',
                 'auth_type', 'logon_type', 'auth_orientation', 'success']

    chunks = []
    chunksize = 500_000
    total = 0

    for chunk in pd.read_csv(cfg.AUTH_FILE, header=None, names=auth_cols,
                              low_memory=False, chunksize=chunksize):

        chunk['src_user'] = clean_user(chunk['src_user'])
        chunk['day']      = seconds_to_day(chunk['time'])
        chunk['hour']     = seconds_to_hour(chunk['time'])
        chunk['is_after'] = is_after_hours_lanl(chunk['hour'])
        chunk['is_fail']  = chunk['success'].astype(str).str.upper() == 'FAIL'

        # Logon type flags
        chunk['is_remote']      = chunk['logon_type'].astype(str).str.contains(
                                   'RemoteInteractive|Network', na=False)
        chunk['is_interactive'] = chunk['logon_type'].astype(str).str.contains(
                                   'Interactive', na=False)
        chunk['is_service']     = chunk['logon_type'].astype(str).str.contains(
                                   'Service|Batch', na=False)

        # Same-machine vs lateral
        chunk['is_lateral'] = chunk['src_computer'] != chunk['dst_computer']

        g = chunk.groupby(['src_user', 'day'])
        agg = g.agg(
            n_auth_events        = ('time',           'count'),
            n_unique_dst_comp    = ('dst_computer',   'nunique'),
            n_unique_src_comp    = ('src_computer',   'nunique'),
            n_failed_logons      = ('is_fail',        'sum'),
            n_afterhour_logons   = ('is_after',       'sum'),
            n_remote_logons      = ('is_remote',      'sum'),
            n_interactive_logons = ('is_interactive', 'sum'),
            n_service_logons     = ('is_service',     'sum'),
            n_lateral_moves      = ('is_lateral',     'sum'),
        ).reset_index().rename(columns={'src_user': 'user'})

        chunks.append(agg)
        total += len(chunk)
        if total % 5_000_000 == 0:
            print(f"  Processed {total:,} auth rows...", end='\r')

    print(f"\n  Total auth rows processed: {total:,}")
    df = pd.concat(chunks).groupby(['user', 'day']).sum().reset_index()

    # Derived features
    df['failed_logon_ratio'] = np.where(
        df['n_auth_events'] > 0,
        df['n_failed_logons'] / df['n_auth_events'],
        0.0
    )
    df['after_hours_ratio'] = np.where(
        df['n_auth_events'] > 0,
        df['n_afterhour_logons'] / df['n_auth_events'],
        0.0
    )
    df['lateral_ratio'] = np.where(
        df['n_auth_events'] > 0,
        df['n_lateral_moves'] / df['n_auth_events'],
        0.0
    )

    print(f"  {len(df):,} user-day records from auth")
    return df


# ---------------------------------------------------------------------------
# EXTRACT PROCESS FEATURES
# ---------------------------------------------------------------------------

def extract_proc_features():
    """
    Extract per-user-per-day features from proc.txt.
    """
    print("[PROC] Extracting process features...")

    proc_cols = ['time', 'user', 'computer', 'process_name', 'event_type']
    chunks = []
    chunksize = 500_000
    total = 0

    for chunk in pd.read_csv(cfg.PROC_FILE, header=None, names=proc_cols,
                              low_memory=False, chunksize=chunksize):

        chunk['user'] = clean_user(chunk['user'])
        chunk['day']  = seconds_to_day(chunk['time'])

        proc_lower = chunk['process_name'].fillna('').str.lower()
        chunk['is_suspicious'] = proc_lower.apply(
            lambda p: any(k in p for k in SUSPICIOUS_PROCS)
        )

        starts = chunk[chunk['event_type'].astype(str).str.contains('start', case=False, na=False)]
        g = starts.groupby(['user', 'day'])
        agg = g.agg(
            n_process_starts    = ('time',          'count'),
            n_unique_processes  = ('process_name',  'nunique'),
            n_suspicious_procs  = ('is_suspicious', 'sum'),
        ).reset_index()

        chunks.append(agg)
        total += len(chunk)
        if total % 5_000_000 == 0:
            print(f"  Processed {total:,} proc rows...", end='\r')

    print(f"\n  Total proc rows processed: {total:,}")
    df = pd.concat(chunks).groupby(['user', 'day']).sum().reset_index()
    print(f"  {len(df):,} user-day records from proc")
    return df


# ---------------------------------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------------------------------

def run_lanl_extraction():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n[START] LANL Feature Extraction")

    # Ground truth
    redteam_set, _ = load_redteam()

    # Extract features
    auth = extract_auth_features()

    if os.path.exists(cfg.PROC_FILE):
        proc = extract_proc_features()
        df   = auth.merge(proc, on=['user', 'day'], how='left')
    else:
        print("[WARN] proc.txt not found - skipping process features")
        df = auth.copy()

    # Fill missing
    feat_cols = [c for c in df.columns if c not in ['user', 'day']]
    df[feat_cols] = df[feat_cols].fillna(0)

    # Add redteam label
    df['redteam'] = df.apply(
        lambda r: 1 if (r['user'], r['day']) in redteam_set else 0, axis=1
    )

    # Save
    out_path = cfg.FEATURES_FILE
    df.to_csv(out_path, index=False)

    print(f"\n[OK] Saved: {out_path}")
    print(f"  Rows:       {len(df):,}")
    print(f"  Users:      {df['user'].nunique():,}")
    print(f"  Features:   {len(feat_cols):,}")
    print(f"  Red team rows: {df['redteam'].sum():,}")

    return df


if __name__ == '__main__':
    run_lanl_extraction()
