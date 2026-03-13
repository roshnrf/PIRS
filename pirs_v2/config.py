"""
PIRS V2 - CONFIGURATION
========================
Predictive Insider Risk & Stabilization System
Pre-incident detection framework: detects behavioral drift 7-14 days
before a threat event occurs.

Supports: CERT r6.2 (insider threat) + LANL (red team attacks)

Authors: Roshan A Rauof, Reem Fariha
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CERTConfig:
    """CERT r6.2 dataset configuration"""

    # --- Paths ---
    DATASET_DIR = os.path.join(BASE_DIR, '..', 'pirs_backend', 'dataset')
    OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs', 'cert')

    # Raw CERT event files
    LOGON_FILE  = os.path.join(DATASET_DIR, 'logon.csv')
    DEVICE_FILE = os.path.join(DATASET_DIR, 'device.csv')
    FILE_FILE   = os.path.join(DATASET_DIR, 'file.csv')
    EMAIL_FILE  = os.path.join(DATASET_DIR, 'email.csv')
    HTTP_FILE   = os.path.join(DATASET_DIR, 'http.csv')
    LDAP_DIR    = os.path.join(DATASET_DIR, 'LDAP')
    PSYCHO_FILE = os.path.join(DATASET_DIR, 'psychometric.csv')

    # Extracted features file (will be created by cert_extractor.py)
    FEATURES_FILE = os.path.join(OUTPUT_DIR, 'cert_features.csv')

    # --- Ground truth ---
    INSIDER_USERS = ['ACM2278', 'CMP2946', 'PLJ1771', 'CDE1846', 'MBG3183']
    INSIDER_SCENARIOS = {
        'ACM2278': 1,   # Wikileaks upload
        'CMP2946': 2,   # Job search + USB theft
        'PLJ1771': 3,   # Keylogger + sabotage
        'CDE1846': 4,   # Email exfiltration (3-month arc)
        'MBG3183': 5,   # Dropbox upload after layoffs
    }
    INSIDER_MALICIOUS_DAYS = {
        'ACM2278': list(range(229, 236)),
        'CMP2946': list(range(402, 428)),
        'PLJ1771': [223],
        'CDE1846': list(range(416, 480)),
        'MBG3183': [284],
    }

    # --- Feature groups (for personality mapping) ---
    FEATURE_GROUPS = {
        'logon':   ['n_logon', 'n_logoff', 'n_afterhour_logon', 'n_unique_pcs', 'work_hour_ratio'],
        'usb':     ['n_usb_connect', 'n_afterhour_usb', 'usb_mean_files'],
        'file':    ['n_file_ops', 'n_file_to_usb', 'n_file_from_usb', 'n_file_doc',
                    'n_file_exe', 'n_afterhour_file'],
        'email':   ['n_email_sent', 'n_email_recv', 'n_email_external', 'n_email_bcc_ext',
                    'n_email_with_att'],
        'http':    ['n_http', 'n_job_sites', 'n_hack_sites', 'n_cloud_upload',
                    'n_social_media', 'n_afterhour_http'],
        'derived': ['after_hours_ratio', 'is_weekday', 'n_unique_pcs'],
    }

    # --- Dataset properties ---
    N_USERS   = 4000
    N_DAYS    = 516
    WORK_START = 8   # 8 AM
    WORK_END   = 18  # 6 PM


class LANLConfig:
    """LANL dataset configuration"""

    # --- Paths (update to where you save LANL files) ---
    DATASET_DIR = os.path.join(BASE_DIR, '..', 'lanl_data')
    OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs', 'lanl')

    AUTH_FILE    = os.path.join(DATASET_DIR, 'auth.txt')
    PROC_FILE    = os.path.join(DATASET_DIR, 'proc.txt')
    REDTEAM_FILE = os.path.join(DATASET_DIR, 'redteam.txt')

    # Extracted features file
    FEATURES_FILE = os.path.join(OUTPUT_DIR, 'lanl_features.csv')

    # --- Dataset properties ---
    TOTAL_SECONDS = 58 * 24 * 3600   # 58 days in seconds
    N_USERS_APPROX = 12000
    WORK_START_SEC = 8  * 3600       # 8 AM in seconds from midnight
    WORK_END_SEC   = 18 * 3600       # 6 PM in seconds from midnight


class ModelConfig:
    """Shared ML model configuration (same for CERT and LANL)"""

    # --- Baseline ---
    BASELINE_DAYS = 60          # Days to establish personal normal behavior

    # --- Prediction windows ---
    PREDICT_WINDOWS = [7, 14]   # Evaluate at both 7-day and 14-day horizons

    # --- Drift detection ---
    DRIFT_WINDOW   = 14         # Rolling window for drift calculation (days)
    MIN_DAYS_DRIFT = 5          # Minimum days needed before computing drift

    # --- Anomaly detection ensemble ---
    ISOFOREST_WEIGHT  = 0.50
    LSTM_WEIGHT       = 0.35
    OCSVM_WEIGHT      = 0.15

    ISOFOREST_CONTAMINATION = 0.005
    ISOFOREST_N_ESTIMATORS  = 100
    LSTM_LATENT_DIM  = 32
    LSTM_EPOCHS      = 10
    LSTM_BATCH_SIZE  = 64
    LSTM_TRAIN_SAMPLE = 50000
    OCSVM_NU     = 0.005
    OCSVM_KERNEL = 'rbf'
    OCSVM_GAMMA  = 'scale'

    # --- Risk thresholds ---
    RISK_HIGH     = 6.0
    RISK_MODERATE = 4.0

    # --- Personality dimensions (CERT only) ---
    PERSONALITY_DIMS = ['COMPLIANT', 'SOCIAL', 'CAREFULL', 'RISK_TAKER', 'AUTONOMOUS']

    # --- Intervention levels ---
    INTERVENTION_LEVELS = {
        1: 'Standard Monitoring',
        2: 'Passive Friction',
        3: 'Warning Banner',
        4: 'Behavioral Training',
        5: 'Security Acknowledgment',
        6: 'Manager Intervention',
        7: 'Account Lock',
    }

    # --- Q-learning ---
    QL_ALPHA   = 0.1    # Learning rate
    QL_GAMMA   = 0.9    # Discount factor
    QL_EPSILON = 0.1    # Exploration rate

    # --- Prevention cost model ---
    INCIDENT_COST_USD = 11_400_000   # Ponemon 2023 average insider incident cost

    RANDOM_STATE = 42
