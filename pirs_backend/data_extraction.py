"""
PIRS BACKEND - RAW CERT r6.2 FEATURE EXTRACTION
================================================
Builds a clean user-day feature matrix directly from the raw CERT r6.2
event files, replacing the pre-aggregated dayr6.2.csv.

Input files (dataset/):
  logon.csv          (3.5 M rows)  - logon/logoff events
  device.csv         (1.5 M rows)  - USB connect/disconnect events
  file.csv           (2.0 M rows)  - file read/write/copy events
  email.csv         (11.0 M rows)  - email send/view events
  http.csv          (~85 GB)       - web browsing events  <- chunked
  psychometric.csv   (175 KB)      - OCEAN personality scores
  LDAP/             (monthly CSVs) - employee role / department
  answers/insiders.csv             - GROUND TRUTH: 5 insider users

Output:
  pirs_outputs/data_extracted.csv        <- replaces data_processed.csv
  pirs_outputs/behavioral_features.npy   <- 42 semantic feature names
  pirs_outputs/semantic_groups.npy       <- personality dimension mapping

Features (42 total):
  LOGON    (4): n_logon, n_afterhour_logon, n_unique_pcs, n_logoff
  DEVICE   (3): n_usb_connect, n_afterhour_usb, usb_mean_files
  FILE     (8): n_file_ops, n_file_to_usb, n_file_from_usb,
                n_afterhour_file, n_file_doc, n_file_exe, n_file_zip,
                n_file_to_usb_doc
  EMAIL    (7): n_email_sent, n_email_external, n_email_bcc_ext,
                n_email_with_att, n_afterhour_email, email_mean_size,
                n_email_recv
  HTTP     (7): n_http, n_http_upload, n_job_sites, n_cloud_upload,
                n_hack_sites, n_social_media, n_afterhour_http
  COMPOSITE(4): exfiltration_score, policy_violation_score,
                timing_anomaly_score, insider_risk_composite
  RATIO    (3): after_hours_ratio, work_hour_ratio, external_email_ratio
  OCEAN    (5): O, C, E, A, N  (merged from psychometric.csv)
  META     (1): is_weekday

Ground truth (r6.2 scenarios):
  ACM2278  Scenario 1  Cloud upload / wikileaks
  CMP2946  Scenario 2  Job search + USB data theft
  PLJ1771  Scenario 3  Sysadmin disgruntled + keylogger
  CDE1846  Scenario 4  IP theft via email (home address)
  MBG3183  Scenario 5  Dropbox upload

Usage:
    python data_extraction.py
    (Estimated runtime: 25-60 min depending on disk speed)
    Set SKIP_HTTP = True for a fast test run (~5 min, no web features)

Author: Roshan A Rauof
Defense: March 12, 2026
"""

import os
import sys
import gc
import time
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR   = 'dataset/r6.2'
OUTPUT_DIR    = PIRSConfig.OUTPUT_DIR
CHUNK_SIZE    = 200_000       # rows per chunk for large files
WORK_HOUR_START = 7           # 07:00 = start of work day
WORK_HOUR_END   = 18          # 18:00 = end of work day
COMPANY_DOMAIN  = 'dtaa.com'  # internal email domain

# Set True to skip http.csv (fast test run, ~5 min)
SKIP_HTTP = False

# Reference date: earliest date in dataset
REF_DATE = datetime(2010, 1, 1)

# ============================================================================
# GROUND TRUTH: r6.2 INSIDER USERS
# ============================================================================

R62_INSIDERS = {
    'ACM2278': {'scenario': 1, 'start': '08/18/2010', 'end': '08/24/2010',
                'desc': 'Cloud upload / wikileaks exfiltration'},
    'CMP2946': {'scenario': 2, 'start': '02/07/2011', 'end': '03/04/2011',
                'desc': 'Job search + USB data theft'},
    'PLJ1771': {'scenario': 3, 'start': '08/12/2010', 'end': '08/12/2010',
                'desc': 'Disgruntled sysadmin + keylogger mass email'},
    'CDE1846': {'scenario': 4, 'start': '02/21/2011', 'end': '04/25/2011',
                'desc': 'IP theft via email to home address'},
    'MBG3183': {'scenario': 5, 'start': '10/12/2010', 'end': '10/12/2010',
                'desc': 'Dropbox document upload'},
}

# Insider user set for quick lookup
INSIDER_USER_SET = set(R62_INSIDERS.keys())

# Semantic personality groups (same logic as feature_engineering.py)
SEMANTIC_GROUPS = {
    'COMPLIANT': [
        'n_logon', 'work_hour_ratio', 'n_email_sent', 'n_file_ops', 'is_weekday'
    ],
    'SOCIAL': [
        'n_email_sent', 'n_email_recv', 'n_email_external',
        'n_social_media', 'n_email_with_att'
    ],
    'CAREFULL': [
        'work_hour_ratio', 'n_unique_pcs', 'n_file_doc', 'n_file_ops', 'n_logon'
    ],
    'RISK_TAKER': [
        'n_afterhour_usb', 'n_file_to_usb', 'n_email_external',
        'n_job_sites', 'after_hours_ratio', 'n_hack_sites', 'n_email_bcc_ext'
    ],
    'AUTONOMOUS': [
        'n_http', 'n_cloud_upload', 'n_file_ops', 'n_hack_sites', 'n_afterhour_http'
    ],
}

# ============================================================================
# HELPERS
# ============================================================================

def parse_date(date_str):
    """Parse CERT date format MM/DD/YYYY HH:MM:SS -> datetime."""
    try:
        return datetime.strptime(date_str.strip(), '%m/%d/%Y %H:%M:%S')
    except Exception:
        return None


def to_day_number(dt):
    """Convert datetime to integer day number relative to REF_DATE."""
    return (dt.date() - REF_DATE.date()).days


def is_after_hours(dt):
    """Return True if timestamp is outside work hours or on a weekend."""
    if dt is None:
        return False
    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return True
    return dt.hour < WORK_HOUR_START or dt.hour >= WORK_HOUR_END


def is_external_email(address):
    """Return True if email address is external (not @company_domain)."""
    if not isinstance(address, str):
        return False
    return COMPANY_DOMAIN not in address.lower()


def classify_url(url):
    """
    Classify a URL into a category relevant to insider threat detection.
    Returns one of: 'job', 'cloud', 'hack', 'social', 'other'
    """
    if not isinstance(url, str):
        return 'other'
    u = url.lower()

    JOB_KEYWORDS    = ('job', 'career', 'employ', 'recruit', 'monster.com',
                        'indeed.com', 'careerbuilder', 'glassdoor', 'linkedin',
                        'hotjobs', 'simplyhired', 'dice.com', 'hired.com')
    CLOUD_KEYWORDS  = ('dropbox', 'wikileaks', 'gdrive', 'box.com', 'mega.',
                        'mediafire', 'upload', '4shared', 'rapidshare',
                        'sendspace', 'fileserve', 'hotfile', 'bitly')
    HACK_KEYWORDS   = ('hack', 'exploit', 'malware', 'keylog', 'crack',
                        'inject', 'rootkit', 'trojan', 'backdoor', 'payload',
                        'shellcode', 'metasploit', 'kali', 'nmap', 'sqlmap')
    SOCIAL_KEYWORDS = ('facebook', 'twitter', 'reddit', 'youtube', 'instagram',
                        'myspace', 'tumblr', 'pinterest', 'snapchat',
                        'tiktok', 'social', 'digg', 'slashdot')

    for kw in JOB_KEYWORDS:
        if kw in u:
            return 'job'
    for kw in CLOUD_KEYWORDS:
        if kw in u:
            return 'cloud'
    for kw in HACK_KEYWORDS:
        if kw in u:
            return 'hack'
    for kw in SOCIAL_KEYWORDS:
        if kw in u:
            return 'social'
    return 'other'


def file_extension_category(filename):
    """Classify filename into doc/exe/zip/other."""
    if not isinstance(filename, str):
        return 'other'
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in ('doc', 'docx', 'pdf', 'ppt', 'pptx', 'xls', 'xlsx', 'odt', 'txt', 'rtf'):
        return 'doc'
    if ext in ('exe', 'bat', 'sh', 'py', 'vbs', 'ps1', 'cmd', 'msi', 'dll'):
        return 'exe'
    if ext in ('zip', '7z', 'rar', 'tar', 'gz', 'bz2', 'xz', 'cab'):
        return 'zip'
    return 'other'


def empty_day_record(user, day, date_str, dt):
    """Return a zeroed feature record for a user-day."""
    return {
        'user': user, 'day': day, 'date': date_str,
        'week': day // 7,
        'is_weekday': int(dt.weekday() < 5) if dt else 1,
        # LOGON
        'n_logon': 0, 'n_logoff': 0,
        'n_afterhour_logon': 0, 'n_unique_pcs': 0,
        # DEVICE
        'n_usb_connect': 0, 'n_afterhour_usb': 0, 'usb_mean_files': 0.0,
        # FILE
        'n_file_ops': 0, 'n_file_to_usb': 0, 'n_file_from_usb': 0,
        'n_afterhour_file': 0, 'n_file_doc': 0, 'n_file_exe': 0,
        'n_file_zip': 0, 'n_file_to_usb_doc': 0,
        # EMAIL
        'n_email_sent': 0, 'n_email_recv': 0, 'n_email_external': 0,
        'n_email_bcc_ext': 0, 'n_email_with_att': 0,
        'n_afterhour_email': 0, 'email_mean_size': 0.0,
        # HTTP
        'n_http': 0, 'n_http_upload': 0, 'n_job_sites': 0,
        'n_cloud_upload': 0, 'n_hack_sites': 0, 'n_social_media': 0,
        'n_afterhour_http': 0,
    }


# ============================================================================
# STEP 1: LOGON
# ============================================================================

def process_logon():
    """Aggregate logon.csv -> per-user-day logon features."""
    path = os.path.join(DATASET_DIR, 'logon.csv')
    print(f"\n[DIR] Processing logon.csv ...")

    records = defaultdict(lambda: defaultdict(lambda: {
        'n_logon': 0, 'n_logoff': 0, 'n_afterhour_logon': 0, 'pcs': set()
    }))

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                              usecols=['date', 'user', 'pc', 'activity']):
        for _, row in chunk.iterrows():
            dt = parse_date(row['date'])
            if dt is None:
                continue
            day  = to_day_number(dt)
            user = str(row['user'])
            r    = records[user][day]

            act = str(row['activity']).strip().lower()
            if 'logon' in act:
                r['n_logon'] += 1
                if is_after_hours(dt):
                    r['n_afterhour_logon'] += 1
            elif 'logoff' in act:
                r['n_logoff'] += 1

            r['pcs'].add(str(row['pc']))

    # Flatten
    rows = []
    for user, days in records.items():
        for day, r in days.items():
            rows.append({
                'user': user, 'day': day,
                'n_logon':          r['n_logon'],
                'n_logoff':         r['n_logoff'],
                'n_afterhour_logon': r['n_afterhour_logon'],
                'n_unique_pcs':     len(r['pcs']),
            })

    df = pd.DataFrame(rows)
    print(f"   [OK] {len(df):,} user-day records from {df['user'].nunique():,} users")
    return df


# ============================================================================
# STEP 2: DEVICE (USB)
# ============================================================================

def process_device():
    """Aggregate device.csv -> per-user-day USB features."""
    path = os.path.join(DATASET_DIR, 'device.csv')
    print(f"\n[DIR] Processing device.csv ...")

    records = defaultdict(lambda: defaultdict(lambda: {
        'n_usb_connect': 0, 'n_afterhour_usb': 0, 'file_counts': []
    }))

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                              usecols=['date', 'user', 'file_tree', 'activity']):
        for _, row in chunk.iterrows():
            act = str(row['activity']).strip().lower()
            if 'connect' not in act:
                continue
            dt = parse_date(row['date'])
            if dt is None:
                continue
            day  = to_day_number(dt)
            user = str(row['user'])
            r    = records[user][day]

            r['n_usb_connect'] += 1
            if is_after_hours(dt):
                r['n_afterhour_usb'] += 1

            # Count files in file_tree (semicolon-separated paths)
            tree = str(row['file_tree'])
            if tree and tree != 'nan':
                n_files = len([x for x in tree.split(';') if x.strip()])
                r['file_counts'].append(n_files)

    rows = []
    for user, days in records.items():
        for day, r in days.items():
            rows.append({
                'user': user, 'day': day,
                'n_usb_connect':   r['n_usb_connect'],
                'n_afterhour_usb': r['n_afterhour_usb'],
                'usb_mean_files':  float(np.mean(r['file_counts']))
                                   if r['file_counts'] else 0.0,
            })

    df = pd.DataFrame(rows)
    print(f"   [OK] {len(df):,} user-day records")
    return df


# ============================================================================
# STEP 3: FILE OPERATIONS
# ============================================================================

def process_file():
    """Aggregate file.csv -> per-user-day file operation features."""
    path = os.path.join(DATASET_DIR, 'file.csv')
    print(f"\n[DIR] Processing file.csv ...")

    records = defaultdict(lambda: defaultdict(lambda: {
        'n_file_ops': 0, 'n_to_usb': 0, 'n_from_usb': 0,
        'n_afterhour': 0, 'n_doc': 0, 'n_exe': 0, 'n_zip': 0,
        'n_doc_to_usb': 0,
    }))

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                              usecols=['date', 'user', 'filename',
                                       'to_removable_media',
                                       'from_removable_media']):
        for _, row in chunk.iterrows():
            dt = parse_date(row['date'])
            if dt is None:
                continue
            day  = to_day_number(dt)
            user = str(row['user'])
            r    = records[user][day]

            r['n_file_ops'] += 1

            to_usb   = str(row['to_removable_media']).lower()   == 'true'
            from_usb = str(row['from_removable_media']).lower() == 'true'

            if to_usb:
                r['n_to_usb'] += 1
            if from_usb:
                r['n_from_usb'] += 1
            if is_after_hours(dt):
                r['n_afterhour'] += 1

            cat = file_extension_category(str(row['filename']))
            if cat == 'doc':
                r['n_doc'] += 1
                if to_usb:
                    r['n_doc_to_usb'] += 1
            elif cat == 'exe':
                r['n_exe'] += 1
            elif cat == 'zip':
                r['n_zip'] += 1

    rows = []
    for user, days in records.items():
        for day, r in days.items():
            rows.append({
                'user': user, 'day': day,
                'n_file_ops':      r['n_file_ops'],
                'n_file_to_usb':   r['n_to_usb'],
                'n_file_from_usb': r['n_from_usb'],
                'n_afterhour_file':r['n_afterhour'],
                'n_file_doc':      r['n_doc'],
                'n_file_exe':      r['n_exe'],
                'n_file_zip':      r['n_zip'],
                'n_file_to_usb_doc': r['n_doc_to_usb'],
            })

    df = pd.DataFrame(rows)
    print(f"   [OK] {len(df):,} user-day records")
    return df


# ============================================================================
# STEP 4: EMAIL
# ============================================================================

def process_email():
    """Aggregate email.csv -> per-user-day email features."""
    path = os.path.join(DATASET_DIR, 'email.csv')
    print(f"\n[DIR] Processing email.csv  (large file -- please wait) ...")

    records = defaultdict(lambda: defaultdict(lambda: {
        'n_sent': 0, 'n_recv': 0, 'n_external': 0,
        'n_bcc_ext': 0, 'n_with_att': 0, 'n_afterhour': 0,
        'sizes': [],
    }))

    chunk_count = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                              usecols=['date', 'user', 'to', 'bcc', 'from',
                                       'activity', 'size', 'attachments'],
                              on_bad_lines='skip'):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"   ... chunk {chunk_count} ({chunk_count * CHUNK_SIZE:,} rows)")

        for _, row in chunk.iterrows():
            dt = parse_date(row['date'])
            if dt is None:
                continue
            day  = to_day_number(dt)
            user = str(row['user'])
            r    = records[user][day]

            act = str(row['activity']).strip().lower()

            if act == 'send':
                r['n_sent'] += 1

                # External recipient
                to_addr = str(row.get('to', ''))
                if is_external_email(to_addr):
                    r['n_external'] += 1

                # External BCC
                bcc_addr = str(row.get('bcc', ''))
                if bcc_addr and bcc_addr != 'nan' and is_external_email(bcc_addr):
                    r['n_bcc_ext'] += 1

                # Attachments
                att = str(row.get('attachments', ''))
                if att and att != 'nan' and att.lower() != 'none':
                    r['n_with_att'] += 1

                # Email size
                try:
                    r['sizes'].append(float(row['size']))
                except (ValueError, TypeError):
                    pass

                if is_after_hours(dt):
                    r['n_afterhour'] += 1

            elif act in ('view', 'reply', 'reply all', 'forward'):
                r['n_recv'] += 1

    rows = []
    for user, days in records.items():
        for day, r in days.items():
            rows.append({
                'user': user, 'day': day,
                'n_email_sent':      r['n_sent'],
                'n_email_recv':      r['n_recv'],
                'n_email_external':  r['n_external'],
                'n_email_bcc_ext':   r['n_bcc_ext'],
                'n_email_with_att':  r['n_with_att'],
                'n_afterhour_email': r['n_afterhour'],
                'email_mean_size':   float(np.mean(r['sizes']))
                                     if r['sizes'] else 0.0,
            })

    df = pd.DataFrame(rows)
    print(f"   [OK] {len(df):,} user-day records from {df['user'].nunique():,} users")
    return df


# ============================================================================
# STEP 5: HTTP (large -- chunked, url+activity columns only)
# ============================================================================

def process_http():
    """Aggregate http.csv -> per-user-day web activity features."""
    path = os.path.join(DATASET_DIR, 'http.csv')

    if SKIP_HTTP:
        print(f"\n[WARN]  SKIP_HTTP=True -- HTTP features will be 0")
        print(f"   Set SKIP_HTTP=False in data_extraction.py for full features")
        return pd.DataFrame()

    print(f"\n[DIR] Processing http.csv  (very large ~85GB -- this will take a while) ...")
    print(f"   Reading columns: user, date, url, activity only")

    records = defaultdict(lambda: defaultdict(lambda: {
        'n_http': 0, 'n_upload': 0, 'n_job': 0,
        'n_cloud': 0, 'n_hack': 0, 'n_social': 0, 'n_afterhour': 0,
    }))

    chunk_count = 0
    total_rows  = 0
    t_start     = time.time()

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                              usecols=['date', 'user', 'url', 'activity'],
                              on_bad_lines='skip'):
        chunk_count += 1
        total_rows  += len(chunk)

        if chunk_count % 50 == 0:
            elapsed = time.time() - t_start
            rate    = total_rows / elapsed
            print(f"   ... {total_rows:,} rows ({rate/1e6:.1f}M rows/sec, "
                  f"{elapsed/60:.1f} min elapsed)")

        for _, row in chunk.iterrows():
            dt = parse_date(row['date'])
            if dt is None:
                continue
            day  = to_day_number(dt)
            user = str(row['user'])
            r    = records[user][day]

            r['n_http'] += 1
            act = str(row['activity']).strip().lower()
            if 'upload' in act:
                r['n_upload'] += 1

            cat = classify_url(str(row['url']))
            if cat == 'job':
                r['n_job'] += 1
            elif cat == 'cloud':
                r['n_cloud'] += 1
            elif cat == 'hack':
                r['n_hack'] += 1
            elif cat == 'social':
                r['n_social'] += 1

            if is_after_hours(dt):
                r['n_afterhour'] += 1

    elapsed = time.time() - t_start
    print(f"   Processed {total_rows:,} rows in {elapsed/60:.1f} min")

    rows = []
    for user, days in records.items():
        for day, r in days.items():
            rows.append({
                'user': user, 'day': day,
                'n_http':          r['n_http'],
                'n_http_upload':   r['n_upload'],
                'n_job_sites':     r['n_job'],
                'n_cloud_upload':  r['n_cloud'],
                'n_hack_sites':    r['n_hack'],
                'n_social_media':  r['n_social'],
                'n_afterhour_http':r['n_afterhour'],
            })

    df = pd.DataFrame(rows)
    print(f"   [OK] {len(df):,} user-day records from {df['user'].nunique():,} users")
    return df


# ============================================================================
# STEP 6: PSYCHOMETRIC (OCEAN scores)
# ============================================================================

def load_psychometric():
    """Load OCEAN personality scores per user."""
    path = os.path.join(DATASET_DIR, 'psychometric.csv')
    df   = pd.read_csv(path)
    df   = df.rename(columns={'user_id': 'user'})
    df   = df[['user', 'O', 'C', 'E', 'A', 'N']]
    print(f"\n[OK] Psychometric: {len(df):,} users with OCEAN scores")
    return df


# ============================================================================
# STEP 7: LDAP (role / department)
# ============================================================================

def load_ldap():
    """Load employee metadata (most recent record per user)."""
    ldap_dir = os.path.join(DATASET_DIR, 'LDAP')
    files    = sorted(glob.glob(os.path.join(ldap_dir, '*.csv')))

    if not files:
        print("[WARN]  No LDAP files found")
        return pd.DataFrame()

    # Load all, keep most recent record per user
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=['user_id', 'role', 'department',
                                      'business_unit'],
                         on_bad_lines='skip')
        df = df.rename(columns={'user_id': 'user'})
        frames.append(df)

    ldap_all = pd.concat(frames, ignore_index=True)
    ldap_latest = ldap_all.drop_duplicates(subset='user', keep='last')
    print(f"\n[OK] LDAP: {len(ldap_latest):,} unique users")
    return ldap_latest


# ============================================================================
# STEP 8: INSIDER GROUND TRUTH LABELS
# ============================================================================

def build_insider_labels():
    """
    Build a (user, day) -> scenario mapping from answers/insiders.csv
    using the r6.2 rows.
    """
    print(f"\n[LABEL]  Building insider ground truth labels ...")

    # Parse the start/end date ranges for each insider
    insider_days = {}   # user -> set of day numbers
    scenarios    = {}   # user -> scenario number

    for user, info in R62_INSIDERS.items():
        try:
            start = datetime.strptime(info['start'], '%m/%d/%Y')
            end   = datetime.strptime(info['end'],   '%m/%d/%Y')
        except ValueError:
            # Try alternate format
            try:
                start = datetime.strptime(info['start'], '%m/%d/%Y %H:%M:%S')
                end   = datetime.strptime(info['end'],   '%m/%d/%Y %H:%M:%S')
            except Exception:
                print(f"   [WARN]  Could not parse dates for {user}")
                continue

        # All days in [start, end] are labeled
        days = set()
        cur  = start
        while cur <= end:
            days.add(to_day_number(cur))
            cur += timedelta(days=1)

        insider_days[user] = days
        scenarios[user]    = info['scenario']

        print(f"   {user}  Scenario {info['scenario']}: "
              f"{start.date()} - {end.date()}  "
              f"({len(days)} labeled days)  [{info['desc']}]")

    return insider_days, scenarios


# ============================================================================
# STEP 9: MERGE & COMPUTE COMPOSITE FEATURES
# ============================================================================

def merge_all(df_logon, df_device, df_file, df_email, df_http,
              df_psych, df_ldap, insider_days, scenarios):
    """Merge all per-user-day DataFrames and add composite features."""
    print(f"\n[LINK] Merging all feature tables ...")

    # Start with the union of all user-day combos from logon
    base = df_logon.copy()

    # Outer-merge each event table (fill missing with 0)
    def merge_ft(base_df, feature_df, cols):
        if feature_df is None or len(feature_df) == 0:
            for c in cols:
                base_df[c] = 0
            return base_df
        merged = base_df.merge(feature_df[['user', 'day'] + cols],
                                on=['user', 'day'], how='left')
        for c in cols:
            merged[c] = merged[c].fillna(0)
        return merged

    base = merge_ft(base, df_device,
                    ['n_usb_connect', 'n_afterhour_usb', 'usb_mean_files'])
    base = merge_ft(base, df_file,
                    ['n_file_ops', 'n_file_to_usb', 'n_file_from_usb',
                     'n_afterhour_file', 'n_file_doc', 'n_file_exe',
                     'n_file_zip', 'n_file_to_usb_doc'])
    base = merge_ft(base, df_email,
                    ['n_email_sent', 'n_email_recv', 'n_email_external',
                     'n_email_bcc_ext', 'n_email_with_att',
                     'n_afterhour_email', 'email_mean_size'])

    http_cols = ['n_http', 'n_http_upload', 'n_job_sites',
                 'n_cloud_upload', 'n_hack_sites', 'n_social_media',
                 'n_afterhour_http']
    base = merge_ft(base, df_http, http_cols)

    # Add date string and week number
    def day_to_date(d):
        return (REF_DATE + timedelta(days=int(d))).strftime('%Y-%m-%d')

    base['date']       = base['day'].apply(day_to_date)
    base['week']       = base['day'] // 7
    base['is_weekday'] = base['day'].apply(
        lambda d: int((REF_DATE + timedelta(days=int(d))).weekday() < 5)
    )

    # Merge OCEAN
    if len(df_psych) > 0:
        base = base.merge(df_psych, on='user', how='left')
        for col in ['O', 'C', 'E', 'A', 'N']:
            base[col] = base[col].fillna(base[col].median())
    else:
        for col in ['O', 'C', 'E', 'A', 'N']:
            base[col] = 25  # neutral default

    # Merge LDAP role
    if len(df_ldap) > 0:
        base = base.merge(df_ldap, on='user', how='left')
        base['role']          = base['role'].fillna('Unknown')
        base['department']    = base['department'].fillna('Unknown')
        base['business_unit'] = base['business_unit'].fillna(0)
    else:
        base['role'] = 'Unknown'
        base['department'] = 'Unknown'

    # ---- Composite risk features ----
    total_activity = (
        base['n_logon'] + base['n_file_ops'] +
        base['n_email_sent'] + base['n_http'] + 1
    )
    workhour_activity = (
        base['n_logon'] -  base['n_afterhour_logon'] +
        base['n_file_ops'] - base['n_afterhour_file'] +
        base['n_email_sent'] - base['n_afterhour_email'] + 1
    ).clip(lower=0)

    base['after_hours_ratio']    = (
        (base['n_afterhour_logon'] + base['n_afterhour_file'] +
         base['n_afterhour_email'] + base['n_afterhour_usb']) /
        total_activity
    ).clip(0, 1)

    base['work_hour_ratio']      = (
        workhour_activity / total_activity
    ).clip(0, 1)

    base['external_email_ratio'] = (
        base['n_email_external'] / (base['n_email_sent'] + 1)
    ).clip(0, 1)

    # Evidence-weighted composite scores (same weights as feature_engineering.py)
    base['exfiltration_score'] = (
        base['n_file_to_usb']    * 3.0 +
        base['n_afterhour_usb']  * 2.0 +
        base['n_email_external'] * 1.5 +
        base['n_cloud_upload']   * 2.0 +
        base['n_http_upload']    * 2.5
    )
    base['policy_violation_score'] = (
        base['n_hack_sites']     * 4.0 +
        base['n_job_sites']      * 2.0 +
        base['n_afterhour_http'] * 0.5 +
        base['n_email_bcc_ext']  * 2.5
    )
    base['timing_anomaly_score'] = (
        base['n_afterhour_logon'] * 1.0 +
        base['n_afterhour_usb']   * 3.0 +
        base['n_afterhour_file']  * 1.5 +
        base['n_afterhour_email'] * 1.0
    )
    base['insider_risk_composite'] = (
        base['exfiltration_score'] +
        base['policy_violation_score'] +
        base['timing_anomaly_score']
    )

    # ---- Insider ground truth labels ----
    print(f"\n[LABEL]  Labeling insider rows ...")
    base['insider'] = 0
    labeled = 0
    for user, days in insider_days.items():
        scenario = scenarios[user]
        mask = (base['user'] == user) & (base['day'].isin(days))
        base.loc[mask, 'insider'] = scenario
        labeled += mask.sum()
        print(f"   {user}  Scenario {scenario}: {mask.sum()} rows labeled")

    print(f"   Total insider-labeled rows: {labeled}")

    return base


# ============================================================================
# DEFINE & SAVE OUTPUT
# ============================================================================

BEHAVIORAL_FEATURES = [
    # LOGON
    'n_logon', 'n_logoff', 'n_afterhour_logon', 'n_unique_pcs',
    # DEVICE
    'n_usb_connect', 'n_afterhour_usb', 'usb_mean_files',
    # FILE
    'n_file_ops', 'n_file_to_usb', 'n_file_from_usb',
    'n_afterhour_file', 'n_file_doc', 'n_file_exe',
    'n_file_zip', 'n_file_to_usb_doc',
    # EMAIL
    'n_email_sent', 'n_email_recv', 'n_email_external',
    'n_email_bcc_ext', 'n_email_with_att',
    'n_afterhour_email', 'email_mean_size',
    # HTTP
    'n_http', 'n_http_upload', 'n_job_sites',
    'n_cloud_upload', 'n_hack_sites', 'n_social_media',
    'n_afterhour_http',
    # COMPOSITE
    'exfiltration_score', 'policy_violation_score',
    'timing_anomaly_score', 'insider_risk_composite',
    # RATIOS
    'after_hours_ratio', 'work_hour_ratio', 'external_email_ratio',
    # OCEAN
    'O', 'C', 'E', 'A', 'N',
    # ACTIVITY FLAG
    'is_weekday',
]


def save_outputs(df):
    """Save extracted features and update feature lists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(OUTPUT_DIR, 'data_extracted.csv')
    print(f"\n[SAVE] Saving data_extracted.csv ...")
    df.to_csv(out_path, index=False)
    mb = os.path.getsize(out_path) / 1e6
    print(f"   [OK] {out_path}  ({mb:.0f} MB)")

    # Update behavioral_features.npy to point to new 42 features
    feat_path = os.path.join(OUTPUT_DIR, 'behavioral_features.npy')
    np.save(feat_path, BEHAVIORAL_FEATURES, allow_pickle=True)
    print(f"   [OK] behavioral_features.npy  ({len(BEHAVIORAL_FEATURES)} features)")

    # Save semantic groups
    grp_path = os.path.join(OUTPUT_DIR, 'semantic_groups.npy')
    np.save(grp_path, SEMANTIC_GROUPS, allow_pickle=True)
    print(f"   [OK] semantic_groups.npy")

    # Print insider comparison
    insider_mask = df['user'].isin(INSIDER_USER_SET)
    print(f"\n[CHART] Key feature comparison  (insider vs normal):")
    print(f"   {'Feature':<35} {'Insider u':>10} {'Normal u':>10} {'Ratio':>7}")
    print(f"   " + "-" * 65)
    for feat in ['n_file_to_usb', 'n_afterhour_usb', 'n_email_external',
                 'n_cloud_upload', 'n_hack_sites', 'n_job_sites',
                 'exfiltration_score', 'insider_risk_composite']:
        if feat not in df.columns:
            continue
        ins_m  = df.loc[insider_mask, feat].mean()
        nor_m  = df.loc[~insider_mask, feat].mean()
        ratio  = ins_m / (nor_m + 1e-9)
        flag   = " *** HIGH ***" if ratio > 5 else ""
        print(f"   {feat:<35} {ins_m:>10.3f} {nor_m:>10.3f} {ratio:>7.1f}x{flag}")

    return out_path


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_extraction_summary(df, elapsed):
    total_rows = len(df)
    n_users    = df['user'].nunique()
    n_days     = df['day'].nunique()
    n_insider  = (df['insider'] > 0).sum()
    n_ins_users= df[df['insider'] > 0]['user'].nunique()

    print(f"\n" + "="*70)
    print(f"[OK]  DATA EXTRACTION COMPLETE")
    print(f"    Total time:      {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"    Rows:            {total_rows:,}  (user-day records)")
    print(f"    Users:           {n_users:,}")
    print(f"    Days:            {n_days}")
    print(f"    Features:        {len(BEHAVIORAL_FEATURES)}")
    print(f"    Insider rows:    {n_insider}  ({n_ins_users} insider users)")
    print(f"\n    Insider users in dataset:")
    for user in INSIDER_USER_SET:
        u_rows = df[df['user'] == user]
        if len(u_rows) > 0:
            ins_rows = (u_rows['insider'] > 0).sum()
            scen     = R62_INSIDERS[user]['scenario']
            print(f"      {user}  Scenario {scen}: {len(u_rows)} total days, "
                  f"{ins_rows} labeled insider days")
    print(f"\n    Output: pirs_outputs/data_extracted.csv")
    print(f"\n    NEXT STEPS:")
    print(f"    1. Update config.py: EXTRACTED_FEATURES_FILE = 'data_extracted.csv'")
    print(f"    2. Run: python master_pipeline.py")
    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def run_extraction():
    """Full feature extraction pipeline from raw CERT r6.2 files."""
    print("\n" + "="*70)
    print("PIRS: RAW CERT r6.2 FEATURE EXTRACTION")
    print("="*70)
    print(f"\nDataset directory: {os.path.abspath(DATASET_DIR)}")
    print(f"Output directory:  {os.path.abspath(OUTPUT_DIR)}")
    print(f"SKIP_HTTP:         {SKIP_HTTP}")
    print(f"Insider users:     {list(R62_INSIDERS.keys())}")

    t0 = time.time()

    # Build ground truth labels first
    insider_days, scenarios = build_insider_labels()

    # Process each event source
    df_logon  = process_logon()
    df_device = process_device()
    df_file   = process_file()
    df_email  = process_email()
    df_http   = process_http()
    df_psych  = load_psychometric()
    df_ldap   = load_ldap()

    # Merge everything
    df_final = merge_all(df_logon, df_device, df_file, df_email, df_http,
                          df_psych, df_ldap, insider_days, scenarios)

    gc.collect()

    # Save
    save_outputs(df_final)

    elapsed = time.time() - t0
    print_extraction_summary(df_final, elapsed)

    return df_final


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        df = run_extraction()
    except KeyboardInterrupt:
        print("\n[WARN]  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
