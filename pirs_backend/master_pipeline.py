"""
PIRS BACKEND - MASTER PIPELINE
===============================
Run all 8 layers sequentially and generate complete output

Usage: python 09_master_pipeline.py
"""

import os
import sys
import time
from datetime import datetime

# Import configuration
from config import PIRSConfig, setup_environment, print_config_summary

# Import all layer modules
# data_extraction.py replaces data_loading + feature_engineering when
# raw CERT files are available in dataset/
try:
    from data_extraction import run_extraction as run_data_extraction
    _HAS_EXTRACTION = True
except ImportError:
    _HAS_EXTRACTION = False

from data_loading import load_data
from feature_engineering import run_feature_engineering
from layer_1_3_baseline import run_baseline_detection
from layer_4_drift import run_drift_detection
from layer_5_personality import run_personality_profiling
from layer_6_interventions import run_intervention_engine
from layer_7_qlearning import run_qlearning
from layer_8_metrics import run_prevention_metrics
from layer_validation import run_validation

def print_header():
    """Print pipeline header"""
    print("\n" + "="*70)
    print("="*70)
    print("           PIRS BACKEND - MASTER PIPELINE")
    print("  Predictive Intervention and Risk Stabilization System")
    print("="*70)
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")

def print_progress(layer_num, layer_name, status="Running"):
    """Print layer progress"""
    status_emoji = "[WAIT]" if status == "Running" else "[OK]" if status == "Complete" else "[SKIP]" if status == "Skipped" else "[ERROR]"
    print(f"\n{status_emoji} Layer {layer_num}: {layer_name}")
    if status == "Running":
        print("   " + "-" * 60)

def merge_all_outputs():
    """Merge all layer outputs into master file"""
    print("\n[LINK] Merging all outputs into master file...")
    
    import pandas as pd
    
    # Load all layer outputs
    baseline = pd.read_csv(os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['baseline']))
    drift = pd.read_csv(os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['drift']))
    personality = pd.read_csv(os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['personality']))
    interventions = pd.read_csv(os.path.join(PIRSConfig.OUTPUT_DIR, 'layer_8_full_results.csv'))

    # Align user column types (new pipeline uses string user IDs)
    for df in [baseline, drift, personality, interventions]:
        if 'user' in df.columns:
            df['user'] = df['user'].astype(str)
    
    # Merge on user and day (handle 'date' vs 'datetime' column name)
    print("   Merging baseline + drift...")
    merge_keys = ['user', 'day']
    if 'datetime' in baseline.columns and 'datetime' in drift.columns:
        merge_keys.append('datetime')
    elif 'date' in baseline.columns and 'date' in drift.columns:
        merge_keys.append('date')
    master = baseline.merge(drift, on=merge_keys, how='left', suffixes=('', '_drift'))
    
    print("   Merging personality...")
    # Get most recent personality per user
    personality_latest = personality.groupby('user').tail(1)[
        ['user', 'PRIMARY_DIMENSION'] + PIRSConfig.PERSONALITY_DIMS
    ]
    master = master.merge(personality_latest, on='user', how='left')
    
    print("   Merging interventions and outcomes...")
    interventions_subset = interventions[
        ['user', 'day', 'intervention_level', 'intervention_name', 
         'learned_action', 'q_value', 'prevented', 'ttc_hours']
    ]
    master = master.merge(interventions_subset, on=['user', 'day'], how='left')
    
    # Save
    master_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['master'])
    master.to_csv(master_file, index=False)
    
    print(f"\n[OK] Master file created: {master_file}")
    print(f"   Total rows: {len(master):,}")
    print(f"   Total columns: {len(master.columns)}")
    print(f"   File size: {os.path.getsize(master_file) / 1e6:.1f} MB")
    
    return master

def print_summary(total_time, metrics):
    """Print final summary"""
    print("\n" + "="*70)
    print("="*70)
    print("           PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print("="*70)
    
    print(f"\n[TIME]  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n[INFO] FINAL PREVENTION METRICS:")
    print("   " + "-" * 60)
    if metrics:
        print(f"   EPR  (Escalation Prevention Rate):  {metrics['EPR']:.1f}%")
        print(f"   PQ   (Preventability Quotient):     {metrics['PQ']:.2f}")
        print(f"   PIMS (Personality Match Score):     {metrics['PIMS']:.2f}")
        print(f"   IES  (Intervention Efficiency):     {metrics['IES']:.2f}")
        print(f"   TTC  (Time-to-Correction):          {metrics['TTC']:.1f}h")
    
    print("\n[DIR] OUTPUT FILES:")
    print("   " + "-" * 60)
    for key, filename in PIRSConfig.OUTPUT_FILES.items():
        filepath = os.path.join(PIRSConfig.OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1e6
            print(f"   [OK] {filename} ({size_mb:.1f} MB)")
    
    # Check for additional files
    additional_files = ['data_processed.csv', 'layer_8_full_results.csv', 'baseline_models.pkl']
    for filename in additional_files:
        filepath = os.path.join(PIRSConfig.OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1e6
            print(f"   [OK] {filename} ({size_mb:.1f} MB)")
    
    print("\n[TARGET] NEXT STEPS:")
    print("   " + "-" * 60)
    print("   1. Review outputs in pirs_outputs/ directory")
    print("   2. Check pirs_complete.csv for complete dataset")
    print("   3. Review prevention metrics above")
    print("   4. Ready for dashboard development!")
    
    print("\n" + "="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def run_master_pipeline():
    """Execute complete PIRS pipeline"""
    pipeline_start = time.time()
    
    print_header()
    
    # Setup
    print("[SETUP] Setting up environment...")
    setup_environment()
    print_config_summary()
    
    metrics = None
    
    try:
        # Data Extraction / Loading
        # If raw CERT files exist -> use data_extraction.py (best quality)
        # Else fall back to dayr6.2.csv pipeline
        extracted_path = os.path.join(PIRSConfig.OUTPUT_DIR,
                                       PIRSConfig.EXTRACTED_FEATURES_FILE)
        if not os.path.exists(extracted_path):
            if _HAS_EXTRACTION and os.path.exists(
                    os.path.join('dataset', 'logon.csv')):
                print_progress("0", "Raw CERT Feature Extraction", "Running")
                run_data_extraction()
                print_progress("0", "Raw CERT Feature Extraction", "Complete")
            else:
                print_progress("0-1", "Data Loading (dayr6.2.csv)", "Running")
                df_data, behavioral_cols, metadata = load_data()
                print_progress("0-1", "Data Loading (dayr6.2.csv)", "Complete")
                print_progress("FE", "Semantic Feature Engineering", "Running")
                run_feature_engineering()
                print_progress("FE", "Semantic Feature Engineering", "Complete")
        else:
            print(f"\n[OK] Using existing extracted features: {extracted_path}")

        def _layer_done(key):
            """Return True if output file for this layer already exists."""
            path = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES[key])
            if os.path.exists(path):
                print(f"[OK] Skipping layer (output exists): {path}")
                return True
            return False

        # Layer 1-3: Baseline Detection
        if not _layer_done('baseline'):
            print_progress("1-3", "Baseline Risk Detection (Ensemble ML)", "Running")
            df_baseline = run_baseline_detection()
            print_progress("1-3", "Baseline Risk Detection (Ensemble ML)", "Complete")
        else:
            print_progress("1-3", "Baseline Risk Detection (Ensemble ML)", "Skipped")

        # Layer 4: Drift Detection
        if not _layer_done('drift'):
            print_progress("4", "Prospective Drift Detection", "Running")
            df_drift = run_drift_detection()
            print_progress("4", "Prospective Drift Detection", "Complete")
        else:
            print_progress("4", "Prospective Drift Detection", "Skipped")

        # Layer 5: Personality Profiling
        if not _layer_done('personality'):
            print_progress("5", "Dynamic Personality Profiling", "Running")
            df_personality = run_personality_profiling()
            print_progress("5", "Dynamic Personality Profiling", "Complete")
        else:
            print_progress("5", "Dynamic Personality Profiling", "Skipped")

        # Layer 6: Intervention Engine
        if not _layer_done('interventions'):
            print_progress("6", "Graduated Intervention Matching", "Running")
            df_interventions = run_intervention_engine()
            print_progress("6", "Graduated Intervention Matching", "Complete")
        else:
            print_progress("6", "Graduated Intervention Matching", "Skipped")

        # Layer 7: Q-Learning
        if not _layer_done('qlearning'):
            print_progress("7", "Q-Learning Optimization", "Running")
            df_qlearning, q_table = run_qlearning()
            print_progress("7", "Q-Learning Optimization", "Complete")
        else:
            print_progress("7", "Q-Learning Optimization", "Skipped")

        # Layer 8: Prevention Metrics
        if not _layer_done('metrics'):
            print_progress("8", "Prevention Metrics Framework", "Running")
            metrics, df_final = run_prevention_metrics()
            print_progress("8", "Prevention Metrics Framework", "Complete")
        else:
            print_progress("8", "Prevention Metrics Framework", "Skipped")
        
        # Merge all outputs
        master_df = merge_all_outputs()

        # Validation: Insider Detection Evaluation
        print_progress("V", "Insider Detection Validation (ROC-AUC)", "Running")
        try:
            user_risk, auc, val_summary = run_validation()
            print_progress("V", "Insider Detection Validation (ROC-AUC)", "Complete")
        except Exception as val_err:
            print(f"[WARN]  Validation step failed: {val_err}")
            auc = None
            val_summary = {}

        # Print summary
        total_time = time.time() - pipeline_start
        print_summary(total_time, metrics)

        if auc is not None:
            print(f"\n[TARGET] INSIDER DETECTION (ROC-AUC): {auc:.4f}")
            print(f"   Top-1%:  {val_summary.get('detected_top1', '?')}/5 insiders")
            print(f"   Top-5%:  {val_summary.get('detected_top5', '?')}/5 insiders")

        return master_df, metrics
        
    except Exception as e:
        print(f"\n[ERROR] PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("\n[START] Starting PIRS Master Pipeline...")
    
    master_df, metrics = run_master_pipeline()
    
    print("\n[OK] Pipeline execution successful!")
    print("\n[PKG] All outputs saved to: pirs_outputs/")
    print("\nYou can now:")
    print("  - Analyze results in pirs_outputs/pirs_complete.csv")
    print("  - Review prevention metrics printed above")
    print("  - Proceed to dashboard development")
    print("  - Use outputs for Chapter 4 (Results)")
    
    sys.exit(0)
