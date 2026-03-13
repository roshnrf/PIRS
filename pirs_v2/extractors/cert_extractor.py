"""
PIRS V2 - CERT EXTRACTOR
=========================
Extracts per-user-per-day behavioral features from raw CERT r6.2 event files.
HTTP is ENABLED - critical for scenarios 1 (wikileaks), 2 (job sites), 5 (Dropbox).

Output: cert_features.csv
  Columns: user, day, date, + 50 behavioral features + insider label
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CERTConfig

cfg = CERTConfig()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def parse_date_to_day(date_series, origin_date):
    """Convert date strings to integer day numbers (day 1 = first day in dataset)."""
    dates = pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce')
    return ((dates - origin_date).dt.days + 1).astype('Int64')


def get_hour(date_series):
    return pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce').dt.hour


def is_after_hours(hour_series):
    return (hour_series < cfg.WORK_START) | (hour_series >= cfg.WORK_END)


def is_weekday_series(date_series):
    return pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce').dt.dayofweek < 5


# ---------------------------------------------------------------------------
# LOAD LDAP (user metadata + psychometric)
# ---------------------------------------------------------------------------

def load_ldap():
    """Load all LDAP files and return a user metadata dataframe."""
    print("[LDAP] Loading user metadata...")
    frames = []
    for fname in sorted(os.listdir(cfg.LDAP_DIR)):
        if fname.endswith('.csv'):
            fpath = os.path.join(cfg.LDAP_DIR, fname)
            try:
                df = pd.read_csv(fpath)
                frames.append(df)
            except Exception:
                pass
    if not frames:
        print("  [WARN] No LDAP files found.")
        return pd.DataFrame()
    ldap = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['user_id'])
    ldap = ldap.rename(columns={'user_id': 'user'})
    ldap['user'] = ldap['user'].astype(str)
    print(f"  Loaded {len(ldap):,} users from LDAP")
    return ldap


def load_psychometric():
    """Load OCEAN personality scores."""
    print("[OCEAN] Loading psychometric scores...")
    if not os.path.exists(cfg.PSYCHO_FILE):
        print("  [WARN] psychometric.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(cfg.PSYCHO_FILE)
    # Normalize user ID column
    id_col = 'user_id' if 'user_id' in df.columns else 'employee_name'
    df = df.rename(columns={id_col: 'user'})
    df['user'] = df['user'].astype(str)
    print(f"  Loaded {len(df):,} psychometric records")
    return df[['user', 'O', 'C', 'E', 'A', 'N']]


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION PER FILE TYPE
# ---------------------------------------------------------------------------

def extract_logon_features(origin_date):
    print("[LOGON] Extracting logon features...")
    df = pd.read_csv(cfg.LOGON_FILE, usecols=['date', 'user', 'pc', 'activity'],
                     low_memory=False)
    df['user'] = df['user'].astype(str)
    df['hour']     = get_hour(df['date'])
    df['day']      = parse_date_to_day(df['date'], origin_date)
    df['date_str'] = pd.to_datetime(df['date'], infer_datetime_format=True,
                                    errors='coerce').dt.date.astype(str)
    df['is_after'] = is_after_hours(df['hour'])
    df['is_wd']    = is_weekday_series(df['date'])

    logon  = df[df['activity'] == 'Logon']
    logoff = df[df['activity'] == 'Logoff']

    g = logon.groupby(['user', 'day'])
    feats = pd.DataFrame({
        'n_logon':           g.size(),
        'n_afterhour_logon': g['is_after'].sum(),
        'n_unique_pcs':      g['pc'].nunique(),
        'is_weekday':        g['is_wd'].first().astype(int),
        'date':              g['date_str'].first(),
    }).reset_index()

    g2 = logoff.groupby(['user', 'day'])
    feats['n_logoff'] = g2.size().reindex(
        pd.MultiIndex.from_frame(feats[['user', 'day']])).values

    feats['work_hour_ratio'] = np.where(
        feats['n_logon'] > 0,
        1 - feats['n_afterhour_logon'] / feats['n_logon'],
        1.0
    )
    print(f"  {len(feats):,} user-day records from logon")
    return feats


def extract_usb_features(origin_date):
    print("[USB] Extracting USB features...")
    df = pd.read_csv(cfg.DEVICE_FILE, usecols=['date', 'user', 'activity'],
                     low_memory=False)
    df['user']     = df['user'].astype(str)
    df['hour']     = get_hour(df['date'])
    df['day']      = parse_date_to_day(df['date'], origin_date)
    df['is_after'] = is_after_hours(df['hour'])

    conn = df[df['activity'] == 'Connect']
    g = conn.groupby(['user', 'day'])
    feats = pd.DataFrame({
        'n_usb_connect':   g.size(),
        'n_afterhour_usb': g['is_after'].sum(),
    }).reset_index()
    print(f"  {len(feats):,} user-day records from USB")
    return feats


def extract_file_features(origin_date):
    print("[FILE] Extracting file features...")
    df = pd.read_csv(cfg.FILE_FILE,
                     usecols=['date', 'user', 'filename', 'activity',
                               'to_removable_media', 'from_removable_media'],
                     low_memory=False)
    df['user']     = df['user'].astype(str)
    df['hour']     = get_hour(df['date'])
    df['day']      = parse_date_to_day(df['date'], origin_date)
    df['is_after'] = is_after_hours(df['hour'])

    # File type from extension
    df['ext'] = df['filename'].str.lower().str.extract(r'\.(\w+)$')[0].fillna('other')
    df['is_doc'] = df['ext'].isin(['doc','docx','pdf','xls','xlsx','ppt','pptx'])
    df['is_exe'] = df['ext'].isin(['exe','dll','bat','sh','py','ps1'])

    df['to_usb']   = df['to_removable_media'].astype(str).str.strip() == 'TRUE'
    df['from_usb'] = df['from_removable_media'].astype(str).str.strip() == 'TRUE'

    g = df.groupby(['user', 'day'])
    feats = pd.DataFrame({
        'n_file_ops':        g.size(),
        'n_file_to_usb':     g['to_usb'].sum(),
        'n_file_from_usb':   g['from_usb'].sum(),
        'n_file_doc':        g['is_doc'].sum(),
        'n_file_exe':        g['is_exe'].sum(),
        'n_afterhour_file':  g['is_after'].sum(),
    }).reset_index()
    print(f"  {len(feats):,} user-day records from file")
    return feats


def extract_email_features(origin_date):
    print("[EMAIL] Extracting email features...")
    cols = ['date', 'user', 'to', 'cc', 'bcc', 'from', 'activity', 'attachments']
    df = pd.read_csv(cfg.EMAIL_FILE, usecols=cols, low_memory=False)
    df['user'] = df['user'].astype(str)
    df['day']  = parse_date_to_day(df['date'], origin_date)

    # Internal domain check (CERT uses dtaa.com)
    def count_external(addr_series):
        return addr_series.fillna('').apply(
            lambda x: sum(1 for a in str(x).split(';') if a.strip() and 'dtaa.com' not in a)
        )

    sent = df[df['activity'] == 'Send']
    g = sent.groupby(['user', 'day'])
    feats = pd.DataFrame({
        'n_email_sent':    g.size(),
        'n_email_with_att': (sent['attachments'].fillna('').astype(str) != '').groupby(
                             [sent['user'], sent['day']]).sum(),
    }).reset_index()

    feats['n_email_external'] = sent.groupby(['user', 'day']).apply(
        lambda x: count_external(x['to']).sum() + count_external(x['cc']).sum()
    ).reset_index(drop=True)

    feats['n_email_bcc_ext'] = sent.groupby(['user', 'day']).apply(
        lambda x: count_external(x['bcc']).sum()
    ).reset_index(drop=True)

    recv = df[df['activity'] == 'View']
    g2 = recv.groupby(['user', 'day'])
    recv_feats = g2.size().reset_index(name='n_email_recv')

    feats = feats.merge(recv_feats, on=['user', 'day'], how='outer')
    feats = feats.fillna(0)
    print(f"  {len(feats):,} user-day records from email")
    return feats


def extract_http_features(origin_date):
    print("[HTTP] Extracting HTTP features (this may take a few minutes)...")

    # Risky URL keyword categories
    JOB_KEYWORDS   = ['job', 'career', 'hire', 'recruit', 'employ', 'monster',
                      'indeed', 'linkedin', 'glassdoor', 'salary']
    HACK_KEYWORDS  = ['hack', 'exploit', 'rootkit', 'keylog', 'malware', 'trojan',
                      'crack', 'bypass', 'vulnerability', 'zero-day']
    CLOUD_KEYWORDS = ['dropbox', 'drive', 'onedrive', 'wikileaks', 'pastebin',
                      'cloud', 'upload', 'transfer', 'wetransfer', 'sendspace']
    SOCIAL_KEYWORDS = ['facebook', 'twitter', 'instagram', 'reddit', 'social',
                       'forum', 'blog', 'chat']

    chunks = []
    chunksize = 200_000
    total = 0

    for chunk in pd.read_csv(cfg.HTTP_FILE,
                              usecols=['date', 'user', 'url', 'activity'],
                              low_memory=False, chunksize=chunksize):
        chunk['user']     = chunk['user'].astype(str)
        chunk['hour']     = get_hour(chunk['date'])
        chunk['day']      = parse_date_to_day(chunk['date'], origin_date)
        chunk['is_after'] = is_after_hours(chunk['hour'])
        url = chunk['url'].fillna('').str.lower()

        chunk['is_job']    = url.apply(lambda u: any(k in u for k in JOB_KEYWORDS))
        chunk['is_hack']   = url.apply(lambda u: any(k in u for k in HACK_KEYWORDS))
        chunk['is_cloud']  = url.apply(lambda u: any(k in u for k in CLOUD_KEYWORDS))
        chunk['is_social'] = url.apply(lambda u: any(k in u for k in SOCIAL_KEYWORDS))
        chunk['is_upload'] = chunk['activity'].str.lower().str.contains('upload', na=False)

        chunks.append(chunk.groupby(['user', 'day']).agg(
            n_http           = ('url', 'count'),
            n_job_sites      = ('is_job', 'sum'),
            n_hack_sites     = ('is_hack', 'sum'),
            n_cloud_upload   = ('is_cloud', 'sum'),
            n_social_media   = ('is_social', 'sum'),
            n_afterhour_http = ('is_after', 'sum'),
            n_http_upload    = ('is_upload', 'sum'),
        ).reset_index())

        total += len(chunk)
        print(f"  Processed {total:,} HTTP rows...", end='\r')

    print()
    df = pd.concat(chunks).groupby(['user', 'day']).sum().reset_index()
    print(f"  {len(df):,} user-day records from HTTP")
    return df


# ---------------------------------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------------------------------

def run_cert_extraction():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Determine origin date from logon file
    print("\n[START] CERT Feature Extraction (HTTP enabled)")
    sample = pd.read_csv(cfg.LOGON_FILE, usecols=['date'], nrows=10000)
    origin = pd.to_datetime(sample['date'], infer_datetime_format=True,
                             errors='coerce').min().normalize()
    print(f"  Dataset origin date: {origin.date()}")

    # Extract all feature sets
    logon = extract_logon_features(origin)
    usb   = extract_usb_features(origin)
    file  = extract_file_features(origin)
    email = extract_email_features(origin)
    http  = extract_http_features(origin)

    # Merge all on user + day
    print("\n[MERGE] Combining all feature sets...")
    df = logon.copy()
    for other in [usb, file, email, http]:
        df = df.merge(other, on=['user', 'day'], how='left')

    # Fill missing with 0 (user had no activity in that category that day)
    feature_cols = [c for c in df.columns if c not in ['user', 'day', 'date']]
    df[feature_cols] = df[feature_cols].fillna(0)

    # Derived ratio features
    df['after_hours_ratio'] = np.where(
        df['n_logon'] > 0,
        df['n_afterhour_logon'] / df['n_logon'],
        0.0
    )

    # Add OCEAN psychometric scores
    psycho = load_psychometric()
    if not psycho.empty:
        df = df.merge(psycho, on='user', how='left')

    # Add insider ground truth label
    df['insider'] = 0
    for user, bad_days in cfg.INSIDER_MALICIOUS_DAYS.items():
        mask = (df['user'] == user) & (df['day'].isin(bad_days))
        df.loc[mask, 'insider'] = 1

    # Save
    out_path = cfg.FEATURES_FILE
    df.to_csv(out_path, index=False)

    print(f"\n[OK] Saved: {out_path}")
    print(f"  Rows:     {len(df):,}")
    print(f"  Users:    {df['user'].nunique():,}")
    print(f"  Features: {len([c for c in df.columns if c not in ['user','day','date','insider']]):,}")
    print(f"  Insider rows: {df['insider'].sum():,}")

    return df


if __name__ == '__main__':
    run_cert_extraction()
