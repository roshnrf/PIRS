"""
PIRS V2 - LAYER 8: Q-LEARNING INTERVENTION OPTIMIZATION
=========================================================
A Q-learning agent learns which intervention is most effective
for each (risk_level, personality_dim) state combination.

State:  (risk_bucket, personality_dim, drift_label)
Action: intervention_level (1-7)
Reward: +10 if risk decreased after intervention
        -5  if risk increased after intervention
         0  if no change

Over many episodes the agent learns: for a RISK_TAKER user at HIGH
drift, which intervention level most reliably reduces risk?

This is the novelty vs. static rule-based systems.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()

N_ACTIONS   = 7    # Intervention levels 1-7
RISK_BUCKETS = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
DRIFT_STATES = ['STABLE', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL']
PERSONALITY_STATES = cfg.PERSONALITY_DIMS + ['UNKNOWN']

# Build state space
ALL_STATES = [
    (r, p, d)
    for r in RISK_BUCKETS
    for p in PERSONALITY_STATES
    for d in DRIFT_STATES
]
STATE_INDEX = {s: i for i, s in enumerate(ALL_STATES)}
N_STATES = len(ALL_STATES)


def risk_to_bucket(risk_score: float) -> str:
    if risk_score < 4.0:
        return 'LOW'
    elif risk_score < 6.0:
        return 'MODERATE'
    elif risk_score < 8.0:
        return 'HIGH'
    else:
        return 'CRITICAL'


def get_state_index(risk_score, primary_dim, drift_label):
    rb = risk_to_bucket(risk_score)
    pd_ = primary_dim if primary_dim in PERSONALITY_STATES else 'UNKNOWN'
    dl = drift_label if drift_label in DRIFT_STATES else 'STABLE'
    state = (rb, pd_, dl)
    return STATE_INDEX.get(state, 0)


class PIRSQAgent:
    """Q-learning agent for intervention policy optimization."""

    def __init__(self):
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha   = cfg.QL_ALPHA
        self.gamma   = cfg.QL_GAMMA
        self.epsilon = cfg.QL_EPSILON

    def choose_action(self, state_idx: int, exploit: bool = False) -> int:
        """Epsilon-greedy action selection. Returns 0-indexed action."""
        if not exploit and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, state_idx: int, action: int,
               reward: float, next_state_idx: int):
        best_next = np.max(self.Q[next_state_idx])
        self.Q[state_idx, action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state_idx, action]
        )

    def get_optimal_action(self, state_idx: int) -> int:
        """Return best action (1-indexed intervention level)."""
        return int(np.argmax(self.Q[state_idx])) + 1


def compute_reward(risk_before: float, risk_after: float) -> float:
    """Reward based on risk change after intervention."""
    delta = risk_before - risk_after
    if delta > 1.0:
        return 10.0    # Strong reduction
    elif delta > 0.3:
        return 5.0     # Moderate reduction
    elif delta >= 0:
        return 1.0     # Slight improvement
    else:
        return -5.0    # Risk increased -- bad intervention


def train_agent(df: pd.DataFrame, n_episodes: int = 3) -> PIRSQAgent:
    """
    Train Q-learning agent using historical data.
    Each episode shuffles the data and simulates intervention outcomes.
    """
    print(f"\n[L8] Training Q-learning agent ({n_episodes} episodes)...")

    agent = PIRSQAgent()

    # Sort by user + day to get sequential transitions
    df_sorted = df.sort_values(['user', 'day']).reset_index(drop=True)

    for episode in range(n_episodes):
        total_reward = 0
        n_updates = 0

        for user, udf in df_sorted.groupby('user'):
            udf = udf.reset_index(drop=True)
            if len(udf) < 2:
                continue

            for i in range(len(udf) - 1):
                row      = udf.iloc[i]
                next_row = udf.iloc[i + 1]

                state_idx = get_state_index(
                    row['risk_score'],
                    row.get('primary_dim', 'UNKNOWN'),
                    row.get('drift_label', 'STABLE')
                )
                next_state_idx = get_state_index(
                    next_row['risk_score'],
                    next_row.get('primary_dim', 'UNKNOWN'),
                    next_row.get('drift_label', 'STABLE')
                )

                # Use recorded intervention level as action (0-indexed)
                action = int(row.get('intervention_level', 1)) - 1
                action = max(0, min(action, N_ACTIONS - 1))

                reward = compute_reward(row['risk_score'], next_row['risk_score'])
                agent.update(state_idx, action, reward, next_state_idx)

                total_reward += reward
                n_updates += 1

        avg_reward = total_reward / max(n_updates, 1)
        print(f"  Episode {episode+1}/{n_episodes}: "
              f"avg_reward={avg_reward:.3f}, updates={n_updates:,}")

    return agent


def apply_optimal_policy(df: pd.DataFrame, agent: PIRSQAgent) -> pd.DataFrame:
    """
    For each user-day, record the Q-agent's optimal intervention.
    Adds column: rl_intervention_level
    """
    print(f"  Applying learned policy...")

    rl_levels = []
    for _, row in df.iterrows():
        state_idx = get_state_index(
            row['risk_score'],
            row.get('primary_dim', 'UNKNOWN'),
            row.get('drift_label', 'STABLE')
        )
        rl_level = agent.get_optimal_action(state_idx)
        rl_levels.append(rl_level)

    df['rl_intervention_level'] = rl_levels
    df['rl_intervention_name']  = df['rl_intervention_level'].map(
        cfg.INTERVENTION_LEVELS
    )
    return df


def run(df: pd.DataFrame) -> tuple:
    """
    Train Q-agent and apply optimal policy.

    Returns:
        df with rl_intervention_level and rl_intervention_name columns
        agent: trained PIRSQAgent (for inspection/saving)
    """
    agent = train_agent(df, n_episodes=3)
    df    = apply_optimal_policy(df, agent)
    return df, agent
