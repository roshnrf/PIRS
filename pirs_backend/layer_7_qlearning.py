"""
PIRS BACKEND - LAYER 7: Q-LEARNING OPTIMIZATION
================================================
Learn optimal intervention strategies

Can be run standalone: python 06_layer_7_qlearning.py
Or imported: from layer_7_qlearning import run_qlearning
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

def load_interventions():
    """Load intervention decisions"""
    print("\n[DIR] Loading intervention decisions...")
    
    file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['interventions'])
    
    if not os.path.exists(file):
        raise FileNotFoundError("Intervention results not found. Run 05_layer_6_interventions.py first")
    
    df = pd.read_csv(file)
    
    print(f"[OK] Loaded {len(df):,} intervention decisions")
    
    return df

def initialize_qtable(df):
    """Initialize Q-table"""
    n_users = df['user'].nunique()
    n_actions = PIRSConfig.Q_NUM_ACTIONS
    
    print(f"\n[RAND] Initializing Q-table: {n_users} users x {n_actions} actions")
    
    q_table = np.zeros((n_users, n_actions))
    user_to_idx = {user: idx for idx, user in enumerate(df['user'].unique())}
    
    return q_table, user_to_idx

def compute_reward(drift, action):
    """Compute reward for action given drift"""
    # High drift needs high intervention
    if drift > 0.45 and action >= 4:
        return 10
    # Moderate drift needs moderate intervention
    elif drift > 0.25 and action >= 2 and action <= 4:
        return 10
    # Low drift should have low intervention
    elif drift < 0.15 and action <= 2:
        return 5
    # Mismatch
    else:
        return -5

def train_qlearning(df, q_table, user_to_idx):
    """Train Q-learning model"""
    print(f"\n[ML] Training Q-Learning ({PIRSConfig.Q_LEARNING_EPISODES} episodes)...")
    
    alpha = PIRSConfig.Q_LEARNING_ALPHA
    gamma = PIRSConfig.Q_LEARNING_GAMMA
    epsilon = PIRSConfig.Q_LEARNING_EPSILON
    
    # Sample subset for training (for speed)
    df_sample = df.sample(min(50000, len(df)), random_state=PIRSConfig.RANDOM_STATE)
    
    for episode in range(PIRSConfig.Q_LEARNING_EPISODES):
        for _, row in df_sample.iterrows():
            user_idx = user_to_idx[row['user']]
            drift = row['drift_score']
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(0, PIRSConfig.Q_NUM_ACTIONS)
            else:
                action = np.argmax(q_table[user_idx])
            
            # Compute reward
            reward = compute_reward(drift, action)
            
            # Q-update
            old_value = q_table[user_idx, action]
            next_max = np.max(q_table[user_idx])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[user_idx, action] = new_value
        
        if (episode + 1) % 20 == 0:
            avg_q = q_table.mean()
            print(f"   Episode {episode+1}: avg Q-value = {avg_q:.2f}")
    
    final_avg = q_table.mean()
    print(f"\n[OK] Training complete. Final avg Q-value: {final_avg:.2f}")
    
    return q_table

def extract_policy(df, q_table, user_to_idx):
    """Extract learned policy"""
    print("\n[INFO] Extracting learned policy...")
    
    df['learned_action'] = df['user'].map(
        lambda u: np.argmax(q_table[user_to_idx[u]]) + 1 if u in user_to_idx else 1
    )
    df['q_value'] = df['user'].map(
        lambda u: q_table[user_to_idx[u]].max() if u in user_to_idx else 0
    )
    
    # Compare with rule-based
    agreement = (df['learned_action'] == df['intervention_level']).mean()
    print(f"   Agreement with rule-based: {100*agreement:.1f}%")
    
    # Show learned policy distribution
    print(f"\n   Learned Policy Distribution:")
    for level in sorted(df['learned_action'].unique()):
        count = (df['learned_action'] == level).sum()
        pct = 100 * count / len(df)
        print(f"     Level {level}: {count:,} ({pct:.1f}%)")
    
    return df

def save_qlearning_results(df):
    """Save Q-learning results"""
    print("\n[SAVE] Saving Q-learning results...")
    
    output_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['qlearning'])
    df.to_csv(output_file, index=False)
    
    print(f"[OK] Q-learning results saved: {output_file}")
    
    return output_file

def run_qlearning():
    """Main function"""
    print("\n" + "="*70)
    print("LAYER 7: Q-LEARNING OPTIMIZATION")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    df = load_interventions()
    
    # Initialize Q-table
    q_table, user_to_idx = initialize_qtable(df)
    
    # Train Q-learning
    q_table = train_qlearning(df, q_table, user_to_idx)
    
    # Extract policy
    df = extract_policy(df, q_table, user_to_idx)
    
    # Save results
    save_qlearning_results(df)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] Q-LEARNING COMPLETE")
    print(f"   Total time: {elapsed:.1f} seconds")
    print("="*70 + "\n")
    
    return df, q_table

if __name__ == "__main__":
    try:
        df, q_table = run_qlearning()
        print("\n[OK] Q-learning module executed successfully")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)