"""
PIRS BACKEND - FEATURE ENGINEERING MODULE
==========================================
Creates 40 semantically meaningful insider-threat features from the
873 raw PCA features in dayr6.2.csv.

Why: The original approach split 873 features into 5 arbitrary equal groups.
     These 40 features are grounded in insider threat research and map
     directly to the 5 CERT r6.2 insider scenarios.

Insider scenarios addressed:
  Scenario 1 (User 2840): Cloud upload          -> cloud_storage_visits, leak_site_visits
  Scenario 2 (User 2330): Malicious download    -> hack_site_visits, after_hours_http
  Scenario 3 (User 1282): Espionage             -> external_email_count, files_to_usb
  Scenario 4 (User 654):  IP theft              -> files_to_usb, usb_activity, docs_to_usb
  Scenario 5 (User 1494): Sabotage              -> hack_site_visits, after_hours_anomaly

Output: pirs_outputs/data_features_semantic.csv
        pirs_outputs/behavioral_features.npy  (updated to use 40 features)
        pirs_outputs/semantic_groups.npy      (feature groups for Layer 5)

Usage:
    python feature_engineering.py
    (Run AFTER data_loading.py has created data_processed.csv)

Author: Roshan A Rauof
Defense: March 12, 2026
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd

try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

# ============================================================================
# GROUND TRUTH: 5 CERT r6.2 INSIDER USERS
# ============================================================================

INSIDER_USERS = ['ACM2278', 'CMP2946', 'PLJ1771', 'CDE1846', 'MBG3183']

INSIDER_SCENARIOS = {
    'ACM2278': 1,   # Cloud Upload (data exfiltration)
    'CMP2946': 2,   # Malicious Download
    'PLJ1771': 3,   # Espionage (recruited spy)
    'CDE1846': 4,   # IP Theft (before resignation)
    'MBG3183': 5    # Sabotage (disgruntled)
}

INSIDER_MALICIOUS_DAYS = {
    'ACM2278': list(range(229, 236)),   # days 229-235 (6 days)
    'CMP2946': list(range(402, 428)),   # days 402-427 (20 days)
    'PLJ1771': [223],                   # 1 day
    'CDE1846': list(range(416, 480)),   # days 416-479 (45 days)
    'MBG3183': [284]                    # 1 day
}

SCENARIO_NAMES = {
    1: "Cloud Upload (personal storage exfiltration)",
    2: "Malicious Download (weaponized content)",
    3: "Espionage (recruited spy stealing secrets)",
    4: "IP Theft (exfiltration before resignation)",
    5: "Sabotage (disgruntled employee)"
}

# ============================================================================
# SEMANTIC PERSONALITY GROUPS (for Layer 5)
# Maps each personality dimension to semantically relevant features
# ============================================================================

SEMANTIC_GROUPS = {
    'COMPLIANT': [
        # Policy-following, rule-abiding behavior
        'total_logons',
        'work_hour_ratio',
        'total_activity',
        'total_emails',
        'file_depth',
    ],
    'SOCIAL': [
        # Communication-oriented, interpersonal activity
        'total_emails',
        'external_email_count',
        'social_media_visits',
        'doc_attachments',
        'email_attachment_size',
    ],
    'CAREFULL': [
        # Detail-oriented, careful data handling
        'work_hour_ratio',
        'file_depth',
        'total_file_ops',
        'usb_file_tree_size',
        'total_activity',
    ],
    'RISK_TAKER': [
        # Boundary-pushing, risky behaviors (key insider indicators)
        'after_hours_usb',
        'files_to_usb',
        'external_email_count',
        'job_search_visits',
        'after_hours_ratio',
        'hack_site_visits',
        'external_bcc_count',
    ],
    'AUTONOMOUS': [
        # Self-directed, independent, broad access
        'total_http',
        'cloud_storage_visits',
        'total_file_ops',
        'leak_site_visits',
        'after_hours_http',
    ]
}

# ============================================================================
# FEATURE CREATION
# ============================================================================

def safe_col(df, col, default=0.0):
    """Safely get a column value; returns series of default if col missing."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index, dtype='float64')


def create_derived_features(df):
    """
    Creates 40 semantic insider-threat features from the raw behavioral columns.

    Feature groups:
      GROUP 1: After-hours timing signals     (6 features)
      GROUP 2: Data exfiltration via USB      (5 features)
      GROUP 3: Email exfiltration signals     (8 features)
      GROUP 4: Web policy violations          (8 features)
      GROUP 5: General activity patterns      (5 features)
      GROUP 6: Composite risk scores          (4 features)

    Returns: (df_augmented, list_of_40_feature_names)
    """
    derived = {}

    # ------------------------------------------------------------------
    # GROUP 1: AFTER-HOURS TIMING SIGNALS
    # Insider research: unusual after-hours activity is a primary indicator
    # ------------------------------------------------------------------
    derived['after_hours_ratio'] = (
        safe_col(df, 'n_afterhourallact') / (safe_col(df, 'n_allact') + 1)
    )
    derived['after_hours_logon'] = safe_col(df, 'n_afterhourlogon')
    derived['after_hours_usb']   = safe_col(df, 'n_afterhourusb')
    derived['after_hours_email'] = safe_col(df, 'n_afterhouremail')
    derived['after_hours_file']  = safe_col(df, 'n_afterhourfile')
    derived['after_hours_http']  = safe_col(df, 'n_afterhourhttp')

    # ------------------------------------------------------------------
    # GROUP 2: DATA EXFILTRATION VIA USB/FILE
    # Scenarios 3, 4: copying documents to USB drives
    # ------------------------------------------------------------------
    derived['usb_activity'] = safe_col(df, 'n_usb')
    derived['usb_file_tree_size'] = safe_col(df, 'usb_mean_file_tree_len')
    derived['files_to_usb'] = (
        safe_col(df, 'file_n-to_usb1') +
        safe_col(df, 'workhourfile_n-to_usb1') +
        safe_col(df, 'afterhourfile_n-to_usb1')
    )
    derived['docs_to_usb'] = (
        safe_col(df, 'file_docf_n-to_usb1') +
        safe_col(df, 'afterhourfile_docf_n-to_usb1')
    )
    derived['compressed_to_usb'] = (
        safe_col(df, 'file_compf_n-to_usb1') +
        safe_col(df, 'afterhourfile_compf_n-to_usb1')
    )

    # ------------------------------------------------------------------
    # GROUP 3: EMAIL EXFILTRATION SIGNALS
    # Scenarios 3, 4: emailing confidential data externally
    # ------------------------------------------------------------------
    derived['total_emails']          = safe_col(df, 'n_email')
    derived['external_email_count']  = (
        safe_col(df, 'email_n-Xemail1') +
        safe_col(df, 'email_send_mail_n-Xemail1')
    )
    derived['external_bcc_count']    = (
        safe_col(df, 'email_n-exbccmail1') +
        safe_col(df, 'email_send_mail_n-exbccmail1')
    )
    derived['sent_external_ratio']   = (
        safe_col(df, 'email_mean_n_exdes') /
        (safe_col(df, 'email_mean_n_des') + 1)
    )
    derived['email_attachment_size'] = safe_col(df, 'email_mean_n_atts')
    derived['doc_attachments']       = (
        safe_col(df, 'email_mean_e_att_doc') +
        safe_col(df, 'email_send_mail_mean_e_att_doc')
    )
    derived['compressed_attachments'] = (
        safe_col(df, 'email_mean_e_att_comp') +
        safe_col(df, 'email_send_mail_mean_e_att_comp')
    )
    derived['after_hours_email_sent'] = safe_col(df, 'afterhouremail_n_send_mail')

    # ------------------------------------------------------------------
    # GROUP 4: WEB POLICY VIOLATIONS
    # Scenario 1: cloud upload; Scenario 2: hacking sites; Scenarios 3/4: job search
    # ------------------------------------------------------------------
    derived['total_http'] = safe_col(df, 'n_http')
    derived['job_search_visits'] = (
        safe_col(df, 'http_n_jobf') +
        safe_col(df, 'workhourhttp_n_jobf') +
        safe_col(df, 'afterhourhttp_n_jobf')
    )
    derived['cloud_storage_visits'] = (
        safe_col(df, 'http_n_cloudf') +
        safe_col(df, 'workhourhttp_n_cloudf') +
        safe_col(df, 'afterhourhttp_n_cloudf')
    )
    derived['leak_site_visits'] = (
        safe_col(df, 'http_n_leakf') +
        safe_col(df, 'workhourhttp_n_leakf') +
        safe_col(df, 'afterhourhttp_n_leakf')
    )
    derived['hack_site_visits'] = (
        safe_col(df, 'http_n_hackf') +
        safe_col(df, 'workhourhttp_n_hackf') +
        safe_col(df, 'afterhourhttp_n_hackf')
    )
    derived['social_media_visits'] = (
        safe_col(df, 'http_n_socnetf') +
        safe_col(df, 'workhourhttp_n_socnetf')
    )
    derived['after_hours_cloud'] = safe_col(df, 'afterhourhttp_n_cloudf')
    derived['after_hours_hack']  = safe_col(df, 'afterhourhttp_n_hackf')

    # ------------------------------------------------------------------
    # GROUP 5: GENERAL ACTIVITY PATTERNS
    # Baseline behavioral fingerprint (normal vs anomalous overall)
    # ------------------------------------------------------------------
    derived['total_activity']   = safe_col(df, 'n_allact')
    derived['work_hour_ratio']  = (
        safe_col(df, 'n_workhourallact') / (safe_col(df, 'n_allact') + 1)
    )
    derived['total_logons']     = safe_col(df, 'n_logon')
    derived['total_file_ops']   = safe_col(df, 'n_file')
    derived['file_depth']       = safe_col(df, 'file_mean_file_depth')

    # ------------------------------------------------------------------
    # GROUP 6: COMPOSITE RISK SCORES
    # Evidence-weighted combinations specific to insider threat patterns
    # ------------------------------------------------------------------
    derived['exfiltration_score'] = (
        safe_col(df, 'file_n-to_usb1') * 3.0 +
        safe_col(df, 'n_afterhourusb') * 2.0 +
        safe_col(df, 'email_n-Xemail1') * 1.5 +
        safe_col(df, 'http_n_leakf') * 2.5 +
        safe_col(df, 'http_n_cloudf') * 2.0
    )
    derived['policy_violation_score'] = (
        safe_col(df, 'http_n_hackf') * 4.0 +
        safe_col(df, 'http_n_jobf') * 2.0 +
        safe_col(df, 'afterhourhttp_n_hackf') * 4.0 +
        safe_col(df, 'email_n-exbccmail1') * 2.5
    )
    derived['timing_anomaly_score'] = (
        safe_col(df, 'n_afterhourlogon') * 1.0 +
        safe_col(df, 'n_afterhourusb') * 3.0 +
        safe_col(df, 'n_afterhourfile') * 1.5 +
        safe_col(df, 'n_afterhouremail') * 1.0
    )
    derived['insider_risk_composite'] = (
        derived['exfiltration_score'] +
        derived['policy_violation_score'] +
        derived['timing_anomaly_score']
    )

    # ------------------------------------------------------------------
    # Add to dataframe and clean
    # ------------------------------------------------------------------
    feature_names = list(derived.keys())
    for name, values in derived.items():
        df[name] = values

    # Clean: replace inf/nan with 0
    df[feature_names] = (
        df[feature_names]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .clip(lower=0)   # All features are non-negative by definition
    )

    return df, feature_names


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_feature_engineering():
    """
    Reads data_processed.csv in chunks, creates 40 semantic features,
    and saves data_features_semantic.csv + updates behavioral_features.npy.
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING: SEMANTIC INSIDER-THREAT FEATURES")
    print("="*70)
    print(f"\nCreating 40 semantic features from 873 raw behavioral features")
    print(f"Ground truth: {len(INSIDER_USERS)} insider users {INSIDER_USERS}")

    start_time = time.time()

    processed_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_processed.csv')
    output_path    = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_features_semantic.csv')

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"data_processed.csv not found. Run data_loading.py first.\n"
            f"Expected: {processed_path}"
        )

    # Columns needed: metadata + all behavioral columns (for derived features)
    keep_cols = ['user', 'day', 'week', 'datetime', 'insider',
                 'O', 'C', 'E', 'A', 'N']

    # Detect which feature source columns we need
    # (read header only to check availability)
    header_df = pd.read_csv(processed_path, nrows=0)
    all_cols   = list(header_df.columns)

    # We need all cols present in processed data (will use safe_col for missing)
    print(f"\n[INFO] Source dataset: {len(all_cols)} columns")

    # Process in chunks
    chunks_out = []
    total_rows = 0
    chunk_num  = 0
    chunk_size = PIRSConfig.CHUNK_SIZE  # 50,000 rows

    print(f"\n[PROC]  Processing in chunks of {chunk_size:,} rows...")

    for chunk in pd.read_csv(processed_path, chunksize=chunk_size):
        chunk_num  += 1
        total_rows += len(chunk)

        # Ensure insider col exists
        if 'insider' not in chunk.columns:
            chunk['insider'] = 0

        # Create derived features
        chunk, feature_names = create_derived_features(chunk)

        # Keep only what we need for downstream layers
        output_cols = keep_cols + feature_names
        output_cols = [c for c in output_cols if c in chunk.columns]
        chunks_out.append(chunk[output_cols].copy())

        if chunk_num % 5 == 0:
            elapsed = time.time() - start_time
            rate    = total_rows / elapsed
            print(f"   Processed {total_rows:,} rows ({rate:,.0f} rows/sec)")

        del chunk
        gc.collect()

    print(f"\n[LINK] Concatenating {len(chunks_out)} chunks...")
    df_out = pd.concat(chunks_out, ignore_index=True)
    del chunks_out
    gc.collect()

    # Print feature statistics and insider comparison
    print(f"\n[INFO] Feature statistics (insider vs normal users):")
    insider_mask = df_out['user'].isin(INSIDER_USERS)
    for feat in feature_names:
        ins_mean = df_out.loc[insider_mask, feat].mean()
        norm_mean = df_out.loc[~insider_mask, feat].mean()
        ratio = ins_mean / (norm_mean + 1e-9)
        flag = " *** HIGH SIGNAL ***" if ratio > 3 else ""
        print(f"   {feat:<35}: insider={ins_mean:.3f}  normal={norm_mean:.3f}  "
              f"ratio={ratio:.1f}x{flag}")

    # Save output
    print(f"\n[SAVE] Saving semantic feature file...")
    df_out.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[OK] Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"   Rows: {len(df_out):,}")
    print(f"   Columns: {len(df_out.columns)} "
          f"(metadata + {len(feature_names)} semantic features)")

    # Update behavioral_features.npy to use the 40 semantic features
    features_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'behavioral_features.npy')
    np.save(features_path, feature_names, allow_pickle=True)
    print(f"[OK] Updated behavioral_features.npy -> {len(feature_names)} semantic features")

    # Save semantic groups for Layer 5
    groups_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'semantic_groups.npy')
    np.save(groups_path, SEMANTIC_GROUPS, allow_pickle=True)
    print(f"[OK] Saved semantic_groups.npy")

    # Insider label stats
    print(f"\n[SEARCH] Insider Label Verification:")
    insider_rows = df_out[df_out['insider'] > 0]
    print(f"   Total insider-labeled rows: {len(insider_rows)}")
    if len(insider_rows) > 0:
        print(f"   Insider users found: {sorted(insider_rows['user'].unique())}")
        for user in INSIDER_USERS:
            u_rows = insider_rows[insider_rows['user'] == user]
            if len(u_rows) > 0:
                scenario = INSIDER_SCENARIOS[user]
                print(f"   User {user} (Scenario {scenario} - "
                      f"{SCENARIO_NAMES[scenario][:40]}...): "
                      f"{len(u_rows)} labeled days, "
                      f"days {u_rows['day'].min()}-{u_rows['day'].max()}")

    elapsed = time.time() - start_time
    print(f"\n" + "="*70)
    print(f"[OK] FEATURE ENGINEERING COMPLETE")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Features: {len(feature_names)} semantic features")
    print(f"   Output: data_features_semantic.csv ({size_mb:.0f} MB)")
    print(f"\n   NEXT: Run master_pipeline.py to train models on semantic features")
    print("="*70 + "\n")

    return df_out, feature_names


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        df, features = run_feature_engineering()
        print(f"\n[OK] Feature engineering complete. {len(features)} features created.")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
