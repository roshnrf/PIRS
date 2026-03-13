"""
PIRS BACKEND - CONFIGURATION & SETUP
=====================================
Predictive Intervention and Risk Stabilization System
Configuration file with all system settings

Author: Roshan A Rauof, Reem Fariha
Defense Date: March 12, 2026
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PIRSConfig:
    """Central configuration for PIRS system"""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    DATA_PATH = 'ExtractedData/dayr6.2.csv'
    OUTPUT_DIR = 'pirs_outputs'
    
    # ========================================================================
    # PROCESSING SETTINGS
    # ========================================================================
    CHUNK_SIZE = 50000  # Rows per chunk for memory efficiency
    RANDOM_STATE = 42
    
    # ========================================================================
    # LAYER 1-3: BASELINE DETECTION
    # ========================================================================
    # Model weights (must sum to 1.0)
    ISOLATION_FOREST_WEIGHT = 0.50
    LSTM_AUTOENCODER_WEIGHT = 0.35
    OCSVM_WEIGHT = 0.15
    
    # Isolation Forest parameters
    ISOLATION_N_ESTIMATORS = 100
    ISOLATION_CONTAMINATION = 0.005  # 0.5% expected anomalies
    
    # LSTM Autoencoder parameters
    LSTM_LATENT_DIM = 32  # Compression dimension (was 16 -- increased for better discrimination)
    LSTM_EPOCHS = 10       # Training epochs (was 2 -- increased for convergence)
    LSTM_BATCH_SIZE = 64
    LSTM_TRAIN_SAMPLE = 50000  # Sample size for training (was 10000)
    
    # One-Class SVM parameters
    OCSVM_NU = 0.005  # Outlier fraction
    OCSVM_KERNEL = 'rbf'
    OCSVM_GAMMA = 'auto'
    
    # Risk thresholds
    RISK_THRESHOLD_HIGH = 6.0  # Alert threshold
    RISK_THRESHOLD_MODERATE = 4.0
    
    # ========================================================================
    # LAYER 4: DRIFT DETECTION
    # ========================================================================
    DRIFT_WINDOW = 7  # Days to look back for trajectory
    FORECAST_HORIZON = 7  # Days to predict forward
    DRIFT_THRESHOLD_LOW = 0.15  # Minimal drift
    DRIFT_THRESHOLD_MODERATE = 0.25  # Moderate drift
    DRIFT_THRESHOLD_HIGH = 0.45  # High drift
    DRIFT_THRESHOLD_CRITICAL = 0.80  # Critical drift
    
    # Minimum days of data required for drift calculation
    MIN_DAYS_FOR_DRIFT = 3
    
    # ========================================================================
    # LAYER 5: PERSONALITY PROFILING
    # ========================================================================
    PERSONALITY_DIMS = ['COMPLIANT', 'SOCIAL', 'CAREFULL', 'RISK_TAKER', 'AUTONOMOUS']
    
    # OCEAN personality traits
    OCEAN_TRAITS = ['O', 'C', 'E', 'A', 'N']
    OCEAN_RANGE = (0, 50)  # Expected range for OCEAN scores
    
    # Aggregation window for personality calculation
    PERSONALITY_WINDOW = 7  # Days to aggregate for personality
    
    # ========================================================================
    # LAYER 6: INTERVENTION ENGINE
    # ========================================================================
    INTERVENTION_LEVELS = {
        1: "Level 1: Standard Monitoring",
        2: "Level 2: Passive Friction",
        3: "Level 3: Warning Banner",
        4: "Level 4: Behavioral Training",
        5: "Level 5: Security Acknowledgment",
        6: "Level 6: Manager Intervention",
        7: "Level 7: Account Lock"
    }
    
    # Intervention thresholds (drift_score based)
    INTERVENTION_THRESHOLDS = {
        'critical': (0.80, 7),  # drift > 0.80 -> Level 7
        'high_risk_taker': (0.60, 6),  # drift > 0.60 AND RISK-TAKER -> Level 6
        'high': (0.45, 5),  # drift > 0.45 -> Level 5
        'moderate_careful': (0.35, 4),  # drift > 0.35 AND CAREFULL -> Level 4
        'moderate': (0.25, 3),  # drift > 0.25 -> Level 3
        'low': (0.15, 2),  # drift > 0.15 -> Level 2
        'baseline': (0.0, 1)  # default -> Level 1
    }
    
    # ========================================================================
    # LAYER 7: Q-LEARNING
    # ========================================================================
    Q_LEARNING_ALPHA = 0.1  # Learning rate
    Q_LEARNING_GAMMA = 0.6  # Discount factor
    Q_LEARNING_EPSILON = 0.2  # Exploration rate (e-greedy)
    Q_LEARNING_EPISODES = 100
    
    # Number of intervention actions (7 levels)
    Q_NUM_ACTIONS = 7
    
    # ========================================================================
    # LAYER 8: PREVENTION METRICS
    # ========================================================================
    # Literature-based prevention effectiveness rates
    # Format: {personality: {intervention_level: success_probability}}
    PREVENTION_EFFECTIVENESS = {
        'COMPLIANT': {
            'L1': 0.50, 'L2': 0.75, 'L3': 0.80, 'L4': 0.70, 
            'L5': 0.85, 'L6': 0.78, 'L7': 0.90
        },
        'SOCIAL': {
            'L1': 0.50, 'L2': 0.65, 'L3': 0.70, 'L4': 0.68, 
            'L5': 0.80, 'L6': 0.85, 'L7': 0.90
        },
        'CAREFULL': {
            'L1': 0.50, 'L2': 0.78, 'L3': 0.82, 'L4': 0.88, 
            'L5': 0.85, 'L6': 0.80, 'L7': 0.90
        },
        'RISK_TAKER': {
            'L1': 0.30, 'L2': 0.50, 'L3': 0.55, 'L4': 0.60, 
            'L5': 0.65, 'L6': 0.75, 'L7': 0.85
        },
        'AUTONOMOUS': {
            'L1': 0.40, 'L2': 0.60, 'L3': 0.65, 'L4': 0.62, 
            'L5': 0.70, 'L6': 0.78, 'L7': 0.88
        }
    }
    
    # Target metric ranges
    METRIC_TARGETS = {
        'EPR': (40, 55),  # Escalation Prevention Rate (%)
        'PQ': (0.50, 0.70),  # Preventability Quotient
        'PIMS': (1.15, 1.30),  # Personality-Intervention Match Score
        'TTC': (24, 48)  # Time-to-Correction (hours)
    }
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    # Use semantic insider-threat features (40) instead of raw PCA features (873)
    # Set True after running feature_engineering.py
    USE_SEMANTIC_FEATURES = True

    # Input file for anomaly detection layers -- priority order:
    #   1. data_extracted.csv   <- from data_extraction.py (raw CERT files, best)
    #   2. data_features_semantic.csv <- from feature_engineering.py (derived from dayr6.2)
    #   3. data_processed.csv   <- original dayr6.2-based processing (fallback)
    EXTRACTED_FEATURES_FILE = 'data_extracted.csv'
    SEMANTIC_FEATURES_FILE  = 'data_features_semantic.csv'

    # ========================================================================
    # GROUND TRUTH: CERT r6.2 INSIDER USERS
    # ========================================================================
    # 5 confirmed insider users with labeled malicious days (string IDs from data_extracted.csv)
    INSIDER_USERS = ['ACM2278', 'CMP2946', 'PLJ1771', 'CDE1846', 'MBG3183']
    INSIDER_SCENARIOS = {
        'ACM2278': 1,  # Cloud upload / data exfiltration
        'CMP2946': 2,  # Malicious download / job search
        'PLJ1771': 3,  # Espionage (recruited spy)
        'CDE1846': 4,  # IP Theft before resignation
        'MBG3183': 5,  # Sabotage (disgruntled)
    }

    # ========================================================================
    # EXPLAINABILITY
    # ========================================================================
    SHAP_SAMPLE_SIZE = 100  # Background samples for SHAP
    TOP_FEATURES_DISPLAY = 10  # Top features to show in explanations
    
    # ========================================================================
    # OUTPUT FILES
    # ========================================================================
    OUTPUT_FILES = {
        'baseline': 'layer_1_3_baseline.csv',
        'drift': 'layer_4_drift.csv',
        'personality': 'layer_5_personality.csv',
        'interventions': 'layer_6_interventions.csv',
        'qlearning': 'layer_7_qlearning.csv',
        'metrics': 'layer_8_metrics.csv',
        'explainability': 'layer_9_explainability.csv',
        'master': 'pirs_complete.csv',
        'models': 'trained_models.pkl'
    }

# ========================================================================
# SETUP FUNCTIONS
# ========================================================================

def setup_environment():
    """Initialize environment and create output directory"""
    # Create output directory
    os.makedirs(PIRSConfig.OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output directory created: {PIRSConfig.OUTPUT_DIR}")
    
    # Check TensorFlow GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] GPU Detected: {len(gpus)} device(s)")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
            print("[OK] GPU memory growth enabled")
        else:
            print("[WARN]  No GPU detected - using CPU (will be slower)")
    except ImportError:
        print("[WARN]  TensorFlow not found - LSTM layer will fail")
    
    return True

def validate_data_path():
    """Check if data file exists"""
    if not os.path.exists(PIRSConfig.DATA_PATH):
        print(f"[ERROR] ERROR: Data file not found at {PIRSConfig.DATA_PATH}")
        print(f"   Please ensure dayr6.2.csv is in the correct location")
        return False
    
    print(f"[OK] Data file found: {PIRSConfig.DATA_PATH}")
    return True

def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("PIRS BACKEND CONFIGURATION")
    print("="*70)
    print(f"Data Path: {PIRSConfig.DATA_PATH}")
    print(f"Output Directory: {PIRSConfig.OUTPUT_DIR}")
    print(f"Random State: {PIRSConfig.RANDOM_STATE}")
    print(f"\nLayer 1-3 Weights: IF={PIRSConfig.ISOLATION_FOREST_WEIGHT}, "
          f"LSTM={PIRSConfig.LSTM_AUTOENCODER_WEIGHT}, "
          f"SVM={PIRSConfig.OCSVM_WEIGHT}")
    print(f"Layer 4: Drift Window={PIRSConfig.DRIFT_WINDOW} days, "
          f"Forecast={PIRSConfig.FORECAST_HORIZON} days")
    print(f"Layer 7: Q-Learning Episodes={PIRSConfig.Q_LEARNING_EPISODES}")
    print("="*70 + "\n")

# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIRS CONFIGURATION MODULE")
    print("="*70)
    
    # Setup
    setup_environment()
    validate_data_path()
    print_config_summary()
    
    print("[OK] Configuration loaded successfully")
    print("   Import this module in other scripts:")
    print("   from config import PIRSConfig")