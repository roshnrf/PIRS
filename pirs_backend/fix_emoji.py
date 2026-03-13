"""
Fix UnicodeEncodeError on Windows (cp1252) by replacing emoji with ASCII tags.
Run ONCE before running the pipeline:
    venv\Scripts\python.exe fix_emoji.py
"""

import os

EMOJI_MAP = {
    '\u2705': '[OK]',           # [OK]
    '\u274c': '[ERROR]',        # [ERROR]
    '\u26a0\ufe0f': '[WARN]',   # [WARN]️
    '\u26a0': '[WARN]',         # [WARN] (no variation selector)
    '\U0001f4c2': '[DIR]',      # [DIR]
    '\U0001f4c1': '[DIR]',      # [FOL]
    '\U0001f517': '[LINK]',     # [LNK]
    '\U0001f4be': '[SAVE]',     # [SAVE]
    '\U0001f4ca': '[INFO]',     # [CHART]
    '\U0001f4cb': '[INFO]',     # [LST]
    '\U0001f680': '[START]',    # [RKT]
    '\U0001f3af': '[TARGET]',   # [TARGET]
    '\u23f3': '[WAIT]',         # [WAIT]
    '\u23f1\ufe0f': '[TIME]',   # [TMR]️
    '\u23f1': '[TIME]',         # [TMR] (no variation selector)
    '\U0001f527': '[SETUP]',    # [FIX]
    '\U0001f4e6': '[PKG]',      # [PKG]
    '\U0001f7e1': '[WARN]',     # [YEL]
    '\U0001f7e0': '[HIGH]',     # 🟠
    '\U0001f9e0': '[ML]',       # [BRAIN]
    '\u2699\ufe0f': '[PROC]',   # ⚙️
    '\u2699': '[PROC]',         # ⚙ (no variation selector)
    '\U0001f50d': '[SEARCH]',   # [SEARCH]
    '\U0001f332': '[TREE]',     # [TRE]
    '\U0001f500': '[MERGE]',    # 🔀
    '\U0001f6a8': '[ALERT]',    # [ALERT]
    '\U0001f52c': '[VERIFY]',   # [LAB]
    '\U0001f4c8': '[CHART]',    # [UP]
    '\U0001f3f7\ufe0f': '[LABEL]',  # [TAG]️
    '\U0001f3f7': '[LABEL]',    # [TAG] (no variation selector)
    '\u2764\ufe0f': '[HEART]',  # ❤️
    '\u2764': '[HEART]',        # ❤ (no variation selector)
    '\U0001f4d0': '[NOTE]',     # [TRI]
    '\U0001f4cc': '[PIN]',      # [PIN]
}

TARGET_FILES = [
    'master_pipeline.py',
    'layer_1_3_baseline.py',
    'layer_validation.py',
    'feature_engineering.py',
    'layer_5_personality.py',
    'layer_4_drift.py',
    'layer_6_interventions.py',
    'layer_7_qlearning.py',
    'layer_8_metrics.py',
    'data_loading.py',
    'explainability.py',
    'config.py',
]

total_fixed = 0
for filename in TARGET_FILES:
    if not os.path.exists(filename):
        print(f'[SKIP] {filename} not found')
        continue

    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    original = content
    for emoji, replacement in EMOJI_MAP.items():
        content = content.replace(emoji, replacement)

    if content != original:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'[OK] Fixed: {filename}')
        total_fixed += 1
    else:
        print(f'[SKIP] No emoji found in: {filename}')

print(f'\n[DONE] {total_fixed} files fixed. Ready to run pipeline.')
