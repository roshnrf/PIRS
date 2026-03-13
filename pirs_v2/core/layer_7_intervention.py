"""
PIRS V2 - LAYER 7: PERSONALITY-MATCHED INTERVENTION
=====================================================
Selects the appropriate intervention level based on:
  1. risk_score  -- how severe is the risk today?
  2. alert_level -- NORMAL / WATCH / ELEVATED / HIGH / CRITICAL
  3. primary_dim -- the user's personality type
  4. drift_label -- is behavior escalating?

Key novelty: same risk score, different personality = different intervention.
A RISK_TAKER needs friction-based deterrence.
A COMPLIANT user needs awareness and acknowledgment.
An AUTONOMOUS user needs targeted access restriction.

Intervention Levels:
  1: Standard Monitoring      -- passive, no user action
  2: Passive Friction         -- subtle slowdowns, extra confirmations
  3: Warning Banner           -- visible alert on screen
  4: Behavioral Training      -- mandatory security awareness module
  5: Security Acknowledgment  -- must sign policy acknowledgment
  6: Manager Intervention     -- HR/manager notified
  7: Account Lock             -- access suspended pending review
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()

# ---------------------------------------------------------------------------
# INTERVENTION SELECTION RULES
# ---------------------------------------------------------------------------
# Format: (risk_lo, risk_hi, personality_dim) -> intervention_level
# More specific rules (personality-specific) take priority.
# ---------------------------------------------------------------------------

PERSONALITY_RULES = {
    # RISK_TAKER: escalate faster, use friction-based deterrence
    'RISK_TAKER': [
        (0.0, 2.0, 1),
        (2.0, 3.5, 2),   # Friction earlier than others
        (3.5, 5.0, 4),   # Skip banner, go straight to training
        (5.0, 7.0, 5),
        (7.0, 9.0, 6),
        (9.0, 11., 7),
    ],
    # COMPLIANT: slower escalation, acknowledgment-focused
    'COMPLIANT': [
        (0.0, 2.0, 1),
        (2.0, 4.5, 2),
        (4.5, 6.0, 3),
        (6.0, 7.5, 5),   # Acknowledgment fits compliance personality
        (7.5, 9.0, 6),
        (9.0, 11., 7),
    ],
    # SOCIAL: communication-focused interventions
    'SOCIAL': [
        (0.0, 2.0, 1),
        (2.0, 4.0, 3),   # Warning banner (visible, social awareness)
        (4.0, 6.0, 4),
        (6.0, 7.5, 5),
        (7.5, 9.0, 6),
        (9.0, 11., 7),
    ],
    # CAREFULL: awareness and structured training
    'CAREFULL': [
        (0.0, 2.0, 1),
        (2.0, 4.0, 2),
        (4.0, 6.0, 4),   # Structured training fits careful personality
        (6.0, 7.5, 5),
        (7.5, 9.0, 6),
        (9.0, 11., 7),
    ],
    # AUTONOMOUS: access restriction emphasis
    'AUTONOMOUS': [
        (0.0, 2.0, 1),
        (2.0, 3.5, 2),
        (3.5, 5.5, 3),
        (5.5, 7.0, 5),
        (7.0, 8.5, 6),
        (8.5, 11., 7),
    ],
}

# Default (no personality info or LANL)
DEFAULT_RULES = [
    (0.0, 2.0, 1),
    (2.0, 4.0, 2),
    (4.0, 5.5, 3),
    (5.5, 7.0, 4),
    (7.0, 8.0, 5),
    (8.0, 9.0, 6),
    (9.0, 11., 7),
]

# Drift escalation modifier: if drift is HIGH/CRITICAL, bump intervention by 1
DRIFT_ESCALATE = {'HIGH': 1, 'CRITICAL': 2}


def select_intervention(risk_score: float,
                         primary_dim: str = None,
                         drift_label: str = 'STABLE') -> tuple:
    """
    Select intervention level for a single user-day.

    Returns:
        (level: int, name: str, rationale: str)
    """
    rules = PERSONALITY_RULES.get(primary_dim, DEFAULT_RULES)

    level = 1
    for lo, hi, lvl in rules:
        if lo <= risk_score < hi:
            level = lvl
            break

    # Drift escalation bump
    bump = DRIFT_ESCALATE.get(drift_label, 0)
    level = min(level + bump, 7)

    name = cfg.INTERVENTION_LEVELS[level]

    rationale = (f"Risk={risk_score:.2f}, Personality={primary_dim or 'Unknown'}, "
                 f"Drift={drift_label}")

    return level, name, rationale


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply personality-matched intervention selection to all rows.

    Args:
        df: dataframe with risk_score, primary_dim (optional), drift_label

    Returns:
        df with added columns:
          intervention_level, intervention_name, intervention_rationale
    """
    print(f"\n[L7] Selecting personality-matched interventions...")

    has_personality = 'primary_dim' in df.columns
    has_drift_label = 'drift_label' in df.columns

    levels, names, rationales = [], [], []

    for _, row in df.iterrows():
        pdim   = row.get('primary_dim', None) if has_personality else None
        dlabel = row.get('drift_label', 'STABLE') if has_drift_label else 'STABLE'
        lvl, name, rat = select_intervention(row['risk_score'], pdim, dlabel)
        levels.append(lvl)
        names.append(name)
        rationales.append(rat)

    df['intervention_level']     = levels
    df['intervention_name']      = names
    df['intervention_rationale'] = rationales

    # Distribution
    print(f"  Intervention distribution:")
    dist = df.groupby(['intervention_level', 'intervention_name']).size()
    for (lvl, name), cnt in dist.items():
        pct = 100 * cnt / len(df)
        print(f"    L{lvl} {name:30s}: {cnt:>8,} ({pct:.1f}%)")

    return df
