"""
PIRS BACKEND - DATA LOADING MODULE
===================================
Load and validate dayr6.2.csv dataset
Extract behavioral features and metadata

Can be run standalone: python 01_data_loading.py
Or imported: from data_loading import load_data
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from datetime import datetime

# Import configuration
try:
    from config import PIRSConfig, setup_environment, validate_data_path
except ImportError:
    # If running standalone, add parent dir to path
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig, setup_environment, validate_data_path

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def inspect_data_sample(n_rows=1000):
    """Load and inspect a sample of the data"""
    print(f"\n[INFO] Loading {n_rows} rows for inspection...")
    
    df_sample = pd.read_csv(PIRSConfig.DATA_PATH, nrows=n_rows)
    
    print(f"[OK] Sample loaded: {len(df_sample)} rows x {len(df_sample.columns)} columns")
    
    # Validate required columns
    required_cols = ['user', 'day', 'week', 'starttime', 'O', 'C', 'E', 'A', 'N', 'insider']
    missing_cols = [col for col in required_cols if col not in df_sample.columns]
    
    if missing_cols:
        raise ValueError(f"[ERROR] Missing required columns: {missing_cols}")
    
    print(f"[OK] All required columns present")
    
    # Validate OCEAN scores
    ocean_cols = ['O', 'C', 'E', 'A', 'N']
    print(f"\n[CHART] OCEAN Score Validation:")
    for col in ocean_cols:
        col_min = df_sample[col].min()
        col_max = df_sample[col].max()
        col_mean = df_sample[col].mean()
        print(f"  {col}: min={col_min:.0f}, max={col_max:.0f}, mean={col_mean:.1f}")
        
        # Validate range
        if col_min < 0 or col_max > 50:
            print(f"  [WARN]  Warning: {col} values outside expected range [0, 50]")
    
    # Check insider labels
    print(f"\n[SEARCH] Insider Label Check:")
    insider_count = (df_sample['insider'] > 0).sum()
    print(f"  Insiders in sample: {insider_count}")
    if insider_count > 0:
        scenarios = df_sample[df_sample['insider'] > 0]['insider'].unique()
        print(f"  Insider scenarios present: {sorted(scenarios)}")
    else:
        print(f"  [WARN]  No insider labels found in sample")
        print(f"     System will use synthetic high-risk users for evaluation")
    
    return df_sample

def identify_behavioral_features(df):
    """Identify behavioral features (exclude metadata)"""
    
    # Metadata columns to exclude
    metadata_cols = [
        'starttime', 'endtime', 'user', 'day', 'week', 'insider', 
        'project', 'role', 'b_unit', 'f_unit', 'dept', 'team', 
        'ITAdmin', 'O', 'C', 'E', 'A', 'N', 'isweekday', 'isweekend'
    ]
    
    # Get all numeric columns that are not metadata
    behavioral_cols = []
    for col in df.columns:
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col]):
            behavioral_cols.append(col)
    
    print(f"\n[TARGET] Behavioral Features Identified:")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Behavioral features: {len(behavioral_cols)}")
    
    # Show sample feature names
    if len(behavioral_cols) > 0:
        print(f"\n  Sample features:")
        for i, col in enumerate(behavioral_cols[:10]):
            print(f"    {i+1}. {col}")
        if len(behavioral_cols) > 10:
            print(f"    ... and {len(behavioral_cols) - 10} more")
    
    return behavioral_cols, metadata_cols

def load_full_dataset():
    """Load complete dataset in chunks"""
    print(f"\n[PKG] Loading full dataset: {PIRSConfig.DATA_PATH}")
    print(f"   Chunk size: {PIRSConfig.CHUNK_SIZE:,} rows")
    
    start_time = time.time()
    
    chunks = []
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(PIRSConfig.DATA_PATH, 
                              chunksize=PIRSConfig.CHUNK_SIZE,
                              low_memory=False):
        chunks.append(chunk)
        total_rows += len(chunk)
        chunk_num += 1
        
        if chunk_num % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_rows / elapsed if elapsed > 0 else 0
            print(f"  Loaded {total_rows:,} rows ({rate:,.0f} rows/sec)")
    
    print(f"\n[LINK] Concatenating {len(chunks)} chunks...")
    df_full = pd.concat(chunks, ignore_index=True)
    
    del chunks
    gc.collect()
    
    load_time = time.time() - start_time
    
    print(f"\n[OK] Dataset loaded successfully:")
    print(f"  Total rows: {len(df_full):,}")
    print(f"  Total columns: {len(df_full.columns)}")
    print(f"  Unique users: {df_full['user'].nunique():,}")
    print(f"  Date range: Day {df_full['day'].min()} to Day {df_full['day'].max()}")
    print(f"  Total days: {df_full['day'].nunique()}")
    print(f"  Load time: {load_time:.1f} seconds")
    print(f"  Memory usage: {df_full.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    return df_full, load_time

def process_timestamps(df):
    """Convert timestamps to datetime and sort"""
    print(f"\n[DATE] Processing timestamps...")
    
    # Convert Unix timestamp to datetime
    df['datetime'] = pd.to_datetime(df['starttime'], unit='s')
    
    # Sort by user and time
    df = df.sort_values(['user', 'datetime']).reset_index(drop=True)
    
    print(f"[OK] Timestamps processed:")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    return df

def save_metadata(df, behavioral_cols, metadata_cols, load_time):
    """Save dataset metadata for later use"""
    print(f"\n[SAVE] Saving metadata...")
    
    metadata = {
        'total_rows': len(df),
        'total_users': df['user'].nunique(),
        'total_days': df['day'].nunique(),
        'total_weeks': df['week'].nunique(),
        'date_min': str(df['datetime'].min()),
        'date_max': str(df['datetime'].max()),
        'behavioral_features': len(behavioral_cols),
        'feature_names': behavioral_cols,
        'ocean_cols': ['O', 'C', 'E', 'A', 'N'],
        'metadata_cols': metadata_cols,
        'load_time': load_time,
        'created_at': str(datetime.now()),
        'insider_count': int((df['insider'] > 0).sum()),
        'has_insider_labels': bool((df['insider'] > 0).any())
    }
    
    # Save as numpy file (allows Python objects)
    metadata_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'metadata.npy')
    np.save(metadata_path, metadata, allow_pickle=True)
    
    print(f"[OK] Metadata saved to: {metadata_path}")
    
    # Also save as readable text
    metadata_txt_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'metadata.txt')
    with open(metadata_txt_path, 'w') as f:
        f.write("PIRS DATASET METADATA\n")
        f.write("=" * 70 + "\n\n")
        for key, value in metadata.items():
            if key != 'feature_names':  # Skip long list
                f.write(f"{key}: {value}\n")
        f.write(f"\nBehavioral features: {len(behavioral_cols)} total\n")
        f.write("(See metadata.npy for full feature list)\n")
    
    print(f"[OK] Readable metadata saved to: {metadata_txt_path}")
    
    return metadata

def save_processed_data(df, behavioral_cols):
    """Save processed dataset"""
    print(f"\n[SAVE] Saving processed dataset...")
    
    # Save full dataset
    output_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"[OK] Full dataset saved to: {output_path}")
    
    # Save behavioral features separately (for faster loading later)
    features_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'behavioral_features.npy')
    np.save(features_path, behavioral_cols, allow_pickle=True)
    print(f"[OK] Feature list saved to: {features_path}")
    
    return output_path

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def load_data():
    """
    Main function to load and process data
    Returns: (dataframe, behavioral_cols, metadata)
    """
    print("\n" + "="*70)
    print("PIRS DATA LOADING MODULE")
    print("="*70)
    
    start_time = time.time()
    
    # Setup environment
    setup_environment()
    
    # Validate data path
    if not validate_data_path():
        raise FileNotFoundError(f"Data file not found: {PIRSConfig.DATA_PATH}")
    
    # Step 1: Inspect sample
    df_sample = inspect_data_sample()
    
    # Step 2: Identify features
    behavioral_cols, metadata_cols = identify_behavioral_features(df_sample)
    
    # Step 3: Load full dataset
    df_full, load_time = load_full_dataset()
    
    # Step 4: Process timestamps
    df_full = process_timestamps(df_full)
    
    # Step 5: Save metadata
    metadata = save_metadata(df_full, behavioral_cols, metadata_cols, load_time)
    
    # Step 6: Save processed data
    save_processed_data(df_full, behavioral_cols)
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] DATA LOADING COMPLETE")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Rows: {len(df_full):,}")
    print(f"   Users: {df_full['user'].nunique():,}")
    print(f"   Features: {len(behavioral_cols)}")
    print("="*70 + "\n")
    
    return df_full, behavioral_cols, metadata

# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        df, features, meta = load_data()
        print("\n[OK] Data loading module executed successfully")
        print(f"   Outputs saved to: {PIRSConfig.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)