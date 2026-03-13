"""
PIRS BACKEND - LAYER 6: GRADUATED INTERVENTION ENGINE
======================================================
Match interventions to user risk/drift/personality

Can be run standalone: python 05_layer_6_interventions.py
Or imported: from layer_6_interventions import run_intervention_engine
"""

import os
import sys
import time
import pandas as pd
import numpy as np

try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

def load_drift_and_personality():
    """Load drift scores and personality profiles"""
    print("\n[DIR] Loading drift and personality data...")
    
    drift_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['drift'])
    personality_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['personality'])
    
    if not os.path.exists(drift_file):
        raise FileNotFoundError("Drift results not found. Run 03_layer_4_drift.py first")
    
    if not os.path.exists(personality_file):
        raise FileNotFoundError("Personality profiles not found. Run 04_layer_5_personality.py first")
    
    df_drift = pd.read_csv(drift_file)
    df_personality = pd.read_csv(personality_file)
    
    print(f"[OK] Drift: {len(df_drift):,} rows")
    print(f"[OK] Personality: {len(df_personality):,} profiles")
    
    return df_drift, df_personality

def merge_data(df_drift, df_personality):
    """Merge drift and personality on user"""
    print("\n[LINK] Merging data...")
    
    # Get most recent personality for each user
    df_personality_latest = df_personality.groupby('user').tail(1)
    
    # Align user column types
    df_drift['user'] = df_drift['user'].astype(str)
    df_personality_latest = df_personality_latest.copy()
    df_personality_latest['user'] = df_personality_latest['user'].astype(str)

    # Merge
    df_merged = df_drift.merge(
        df_personality_latest[['user', 'PRIMARY_DIMENSION'] + PIRSConfig.PERSONALITY_DIMS],
        on='user',
        how='left'
    )
    
    # Fill missing personalities with most common
    if df_merged['PRIMARY_DIMENSION'].isna().any():
        most_common = df_merged['PRIMARY_DIMENSION'].mode()[0] if len(df_merged['PRIMARY_DIMENSION'].mode()) > 0 else 'COMPLIANT'
        df_merged['PRIMARY_DIMENSION'].fillna(most_common, inplace=True)
    
    print(f"[OK] Merged: {len(df_merged):,} observations")
    
    return df_merged

def select_intervention(row):
    """Select intervention level based on drift and personality"""
    drift = row['drift_score']
    archetype = row['PRIMARY_DIMENSION']
    
    # Critical escalation
    if drift > 0.80:
        return 7  # Account Lock
    
    # High drift + risk-taker
    if drift > 0.60 and archetype == 'RISK_TAKER':
        return 6  # Manager Intervention
    
    # High drift
    if drift > 0.45:
        return 5  # Security Acknowledgment
    
    # Moderate drift + careful (unusual)
    if drift > 0.35 and archetype == 'CAREFULL':
        return 4  # Training
    
    # Moderate drift
    if drift > 0.25:
        return 3  # Warning
    
    # Low drift
    if drift > 0.15:
        return 2  # Passive Friction
    
    # Baseline
    return 1  # Standard Monitoring

def apply_intervention_engine(df_merged):
    """Apply intervention selection logic"""
    print("\n[TARGET] Applying intervention engine...")
    
    df_merged['intervention_level'] = df_merged.apply(select_intervention, axis=1)
    df_merged['intervention_name'] = df_merged['intervention_level'].map(PIRSConfig.INTERVENTION_LEVELS)
    
    # Statistics
    print("\n[INFO] Intervention Distribution:")
    for level in sorted(df_merged['intervention_level'].unique()):
        count = (df_merged['intervention_level'] == level).sum()
        pct = 100 * count / len(df_merged)
        print(f"   Level {level}: {count:,} ({pct:.1f}%)")
    
    # Top intervention priorities
    print("\n[ALERT] Top 10 Intervention Priorities:")
    top_interventions = df_merged.nlargest(10, 'drift_score')[
        ['user', 'day', 'drift_score', 'PRIMARY_DIMENSION', 'intervention_level', 'intervention_name']
    ]
    for _, row in top_interventions.iterrows():
        print(f"   User {row['user']}: Drift={row['drift_score']:.3f}, "
              f"{row['PRIMARY_DIMENSION']} -> {row['intervention_name']}")
    
    return df_merged

def save_interventions(df_interventions):
    """Save intervention decisions"""
    print("\n[SAVE] Saving interventions...")
    
    output_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['interventions'])
    df_interventions.to_csv(output_file, index=False)
    
    print(f"[OK] Interventions saved: {output_file}")
    
    return output_file

def run_intervention_engine():
    """Main function"""
    print("\n" + "="*70)
    print("LAYER 6: GRADUATED INTERVENTION ENGINE")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    df_drift, df_personality = load_drift_and_personality()
    
    # Merge
    df_merged = merge_data(df_drift, df_personality)
    
    # Apply intervention logic
    df_interventions = apply_intervention_engine(df_merged)
    
    # Save
    save_interventions(df_interventions)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] INTERVENTION ENGINE COMPLETE")
    print(f"   Total time: {elapsed:.1f} seconds")
    print("="*70 + "\n")
    
    return df_interventions

if __name__ == "__main__":
    try:
        df = run_intervention_engine()
        print("\n[OK] Intervention engine module executed successfully")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)