"""
PIRS BACKEND - LAYER 5: DYNAMIC PERSONALITY PROFILING
======================================================
Profile users across 5 behavioral dimensions

"""

import os
import sys
import time
import numpy as np
import pandas as pd

try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

def load_processed_data():
    """Load processed data. Highest priority: data_extracted.csv (string user IDs)."""
    print("\n[DIR] Loading processed data for personality profiling...")

    extracted_path = os.path.join(PIRSConfig.OUTPUT_DIR,
                                   PIRSConfig.EXTRACTED_FEATURES_FILE)   # data_extracted.csv
    semantic_path  = os.path.join(PIRSConfig.OUTPUT_DIR,
                                   PIRSConfig.SEMANTIC_FEATURES_FILE)    # data_features_semantic.csv
    raw_path       = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_processed.csv')
    features_file  = os.path.join(PIRSConfig.OUTPUT_DIR, 'behavioral_features.npy')

    # Priority: extracted (string IDs) > semantic > raw (both integer IDs)
    if os.path.exists(extracted_path):
        data_file = extracted_path
        print(f"   Using data_extracted.csv (string user IDs)")
    elif PIRSConfig.USE_SEMANTIC_FEATURES and os.path.exists(semantic_path):
        data_file = semantic_path
        print(f"   Using semantic features file")
    elif os.path.exists(raw_path):
        data_file = raw_path
        print(f"   Using raw processed data")
    else:
        raise FileNotFoundError("No processed data found. Run data_extraction.py first.")

    df = pd.read_csv(data_file)

    # Ensure user column is string to match downstream layers
    df['user'] = df['user'].astype(str)

    # Ensure 'week' column exists (needed for aggregation)
    if 'week' not in df.columns and 'day' in df.columns:
        df['week'] = df['day'] // 7

    # Determine behavioral columns
    meta_cols = {'user', 'day', 'week', 'datetime', 'date', 'insider',
                 'O', 'C', 'E', 'A', 'N'}
    if os.path.exists(features_file):
        behavioral_cols = np.load(features_file, allow_pickle=True).tolist()
        # Keep only columns that exist in this file
        behavioral_cols = [c for c in behavioral_cols if c in df.columns]

    if not os.path.exists(features_file) or len(behavioral_cols) == 0:
        # Derive from actual file: all numeric columns except metadata
        behavioral_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in meta_cols
        ]
        print(f"   [INFO] behavioral_features.npy not found -- "
              f"derived {len(behavioral_cols)} columns from file")

    print(f"[OK] Loaded {len(df):,} rows, {len(behavioral_cols)} behavioral features")
    return df, behavioral_cols

def calculate_personality_dimensions(df, behavioral_cols):
    """
    Calculate 5 personality dimensions.

    PRIMARY METHOD (if OCEAN scores present in dataset):
      Uses ground-truth Big Five psychometric scores (O, C, E, A, N) from
      the CERT r6.2 dataset, mapped to PIRS dimensions via validated
      organizational psychology research:

        COMPLIANT   <- high C + high A  (organized, rule-following, cooperative)
        SOCIAL      <- high E + high A  (outgoing, communicative, agreeable)
        CAREFULL    <- high C + low N   (disciplined, emotionally stable)
        RISK_TAKER  <- high N + low C + low A  (impulsive, antagonistic)
        AUTONOMOUS  <- high O + low E   (independent, curious, introverted)

    FALLBACK (if no OCEAN scores):
      Uses semantic feature groups from behavioral logs.
    """
    print(f"\n[ML] Calculating personality dimensions...")

    # ── PRIMARY: use real OCEAN scores if available ───────────────────────────
    ocean_cols = ['O', 'C', 'E', 'A', 'N']
    has_ocean = all(c in df.columns for c in ocean_cols)

    if has_ocean:
        print(f"   [OK] Ground-truth OCEAN scores found — using psychometric mapping")

        # Scores range 10–50 in CERT r6.2; normalize to 0–1
        O = (df['O'].values.astype(float) - 10) / 40.0  # Openness
        C = (df['C'].values.astype(float) - 10) / 40.0  # Conscientiousness
        E = (df['E'].values.astype(float) - 10) / 40.0  # Extraversion
        A = (df['A'].values.astype(float) - 10) / 40.0  # Agreeableness
        N = (df['N'].values.astype(float) - 10) / 40.0  # Neuroticism

        dimensions = {
            'COMPLIANT':  (C + A) / 2,
            'SOCIAL':     (E + A) / 2,
            'CAREFULL':   (C + (1.0 - N)) / 2,
            'RISK_TAKER': (N + (1.0 - C) + (1.0 - A)) / 3,
            'AUTONOMOUS': (O + (1.0 - E)) / 2,
        }

        for dim, scores in dimensions.items():
            print(f"   {dim}: mean={scores.mean():.3f}, std={scores.std():.3f}")

        return dimensions

    # ── FALLBACK: behavioral inference ───────────────────────────────────────
    print(f"   [WARN] No OCEAN columns found — falling back to behavioral inference")
    print(f"   Using {len(behavioral_cols)} behavioral features")

    groups_path = os.path.join(PIRSConfig.OUTPUT_DIR, 'semantic_groups.npy')
    semantic_groups = None

    if os.path.exists(groups_path):
        semantic_groups = np.load(groups_path, allow_pickle=True).item()
        print(f"   [OK] Using semantic feature groups (research-grounded)")
    else:
        print(f"   [WARN] Semantic groups not found -- using equal-split fallback")

    dimensions = {}

    if semantic_groups is not None:
        for dim in PIRSConfig.PERSONALITY_DIMS:
            group_features = semantic_groups.get(dim, [])
            available = [f for f in group_features if f in df.columns]

            if not available:
                available = [c for c in behavioral_cols if c in df.columns]
                print(f"   [WARN] {dim}: no semantic features found, using all features")

            group_vals = df[available].values.astype(float)
            group_vals = np.nan_to_num(group_vals, nan=0.0, posinf=0.0, neginf=0.0)
            dim_scores = group_vals.mean(axis=1)

            dim_min, dim_max = dim_scores.min(), dim_scores.max()
            if dim_max > dim_min:
                dim_scores = (dim_scores - dim_min) / (dim_max - dim_min)
            else:
                dim_scores = np.zeros_like(dim_scores)

            dimensions[dim] = dim_scores
            print(f"   {dim} ({len(available)} features): "
                  f"u={dim_scores.mean():.3f}, sigma={dim_scores.std():.3f}")
    else:
        X = df[[c for c in behavioral_cols if c in df.columns]].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n_features = X.shape[1]
        group_size = n_features // 5

        for i, dim in enumerate(PIRSConfig.PERSONALITY_DIMS):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < 4 else n_features
            dim_scores = X[:, start_idx:end_idx].mean(axis=1)
            dim_min, dim_max = dim_scores.min(), dim_scores.max()
            if dim_max > dim_min:
                dim_scores = (dim_scores - dim_min) / (dim_max - dim_min)
            else:
                dim_scores = np.zeros_like(dim_scores)
            dimensions[dim] = dim_scores
            print(f"   {dim}: u={dim_scores.mean():.3f}, sigma={dim_scores.std():.3f}")

    return dimensions

def assign_primary_dimension(dimensions):
    """Assign primary personality archetype"""
    print(f"\n[TARGET] Assigning primary personality dimensions...")
    
    # Stack dimensions
    dim_matrix = np.column_stack([dimensions[dim] for dim in PIRSConfig.PERSONALITY_DIMS])
    
    # Get index of max dimension for each row
    primary_idx = dim_matrix.argmax(axis=1)
    primary_dim = [PIRSConfig.PERSONALITY_DIMS[idx] for idx in primary_idx]
    
    # Count distribution
    unique, counts = np.unique(primary_dim, return_counts=True)
    total = len(primary_dim)
    
    print(f"   Distribution:")
    for dim, count in zip(unique, counts):
        pct = 100 * count / total
        print(f"     {dim}: {count:,} ({pct:.1f}%)")
    
    return primary_dim

def validate_with_ocean(df, dimensions):
    """Validate personality dimensions against OCEAN scores (if available)."""
    print(f"\n[VERIFY] Validating personality dimensions against OCEAN scores...")

    ocean_cols = ['O', 'C', 'E', 'A', 'N']
    ocean_available = all(c in df.columns for c in ocean_cols)

    if not ocean_available:
        print(f"   [INFO] OCEAN columns not present -- using behavioral consistency fallback")
        dim_matrix = np.column_stack(
            [dimensions[d] for d in PIRSConfig.PERSONALITY_DIMS]
        )
        consistency = dim_matrix.std(axis=1)
        consistency = (consistency - consistency.min()) / (consistency.max() - consistency.min() + 1e-9)
        print(f"   Mean behavioral consistency: {consistency.mean():.3f}")
        return consistency

    # Normalize OCEAN scores to 0-1
    O = (df['O'].values.astype(float) - 10) / 40.0
    C = (df['C'].values.astype(float) - 10) / 40.0
    E = (df['E'].values.astype(float) - 10) / 40.0
    A = (df['A'].values.astype(float) - 10) / 40.0
    N = (df['N'].values.astype(float) - 10) / 40.0

    # Correlate each PIRS dimension with its OCEAN ground-truth driver
    expected_corr = {
        'COMPLIANT':  ('C+A', (C + A) / 2),
        'SOCIAL':     ('E+A', (E + A) / 2),
        'CAREFULL':   ('C-N', (C + (1 - N)) / 2),
        'RISK_TAKER': ('N-C-A', (N + (1 - C) + (1 - A)) / 3),
        'AUTONOMOUS': ('O-E', (O + (1 - E)) / 2),
    }

    corr_sum = 0.0
    for dim, (label, ocean_proxy) in expected_corr.items():
        if dim in dimensions:
            corr = np.corrcoef(dimensions[dim], ocean_proxy)[0, 1]
            corr_sum += abs(corr)
            print(f"   Corr {dim} vs OCEAN({label}): {corr:.3f}")

    # Consistency = how well each user's dimensions align with their OCEAN scores
    consistency = (
        dimensions['SOCIAL']     * E +
        dimensions['CAREFULL']   * C +
        dimensions['COMPLIANT']  * A +
        dimensions['AUTONOMOUS'] * O +
        dimensions['RISK_TAKER'] * N
    ) / 5.0

    print(f"   Mean OCEAN consistency score: {consistency.mean():.3f}")
    print(f"   Mean |correlation| across dims: {corr_sum / len(expected_corr):.3f}")

    return consistency

def aggregate_by_user_week(df, dimensions, primary_dim, consistency):
    """Aggregate personality profiles by user-week"""
    print(f"\n[INFO] Aggregating by user-week...")
    
    # Add dimensions to dataframe
    for dim_name, dim_scores in dimensions.items():
        df[dim_name] = dim_scores
    
    df['PRIMARY_DIMENSION'] = primary_dim
    df['Psych_Consistency'] = consistency
    
    # Aggregate by user and week
    agg_dict = {dim: 'mean' for dim in PIRSConfig.PERSONALITY_DIMS}
    agg_dict.update({
        'PRIMARY_DIMENSION': lambda x: x.mode()[0] if len(x) > 0 else x.iloc[0],
        'Psych_Consistency': 'mean',
        'O': 'first', 'C': 'first', 'E': 'first', 'A': 'first', 'N': 'first'
    })
    
    df_profile = df.groupby(['user', 'week']).agg(agg_dict).reset_index()
    
    print(f"[OK] Created {len(df_profile):,} user-week profiles")
    print(f"   {df_profile['user'].nunique():,} unique users")
    print(f"   {df_profile['week'].nunique():,} unique weeks")
    
    return df_profile

def save_personality_profiles(df_profile):
    """Save personality profiles"""
    print(f"\n[SAVE] Saving personality profiles...")
    
    output_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['personality'])
    df_profile.to_csv(output_file, index=False)
    
    print(f"[OK] Personality profiles saved: {output_file}")
    
    return output_file

def run_personality_profiling():
    """Main function for personality profiling"""
    print("\n" + "="*70)
    print("LAYER 5: DYNAMIC PERSONALITY PROFILING")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    df, behavioral_cols = load_processed_data()
    
    # Calculate dimensions
    dimensions = calculate_personality_dimensions(df, behavioral_cols)
    
    # Assign primary dimension
    primary_dim = assign_primary_dimension(dimensions)
    
    # Validate with OCEAN
    consistency = validate_with_ocean(df, dimensions)
    
    # Aggregate by user-week
    df_profile = aggregate_by_user_week(df, dimensions, primary_dim, consistency)
    
    # Save
    save_personality_profiles(df_profile)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] PERSONALITY PROFILING COMPLETE")
    print(f"   Total time: {elapsed:.1f} seconds")
    print("="*70 + "\n")
    
    return df_profile

if __name__ == "__main__":
    try:
        df = run_personality_profiling()
        print("\n[OK] Personality profiling module executed successfully")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)