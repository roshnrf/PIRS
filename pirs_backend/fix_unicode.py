"""
Fix all non-ASCII characters in Python files that would fail on Windows cp1252.
Run once: python fix_unicode.py
"""
import os
import glob

REPLACEMENTS = {
    # Greek letters
    '\u03bc': 'u',          # μ → u (mean)
    '\u03b1': 'alpha',      # α
    '\u03b2': 'beta',       # β
    '\u03c3': 'sigma',      # σ
    # Math symbols
    '\u00b1': '+/-',        # ±
    '\u00b2': '^2',         # ²
    '\u00b3': '^3',         # ³
    '\u2248': '~',          # ≈
    '\u2260': '!=',         # ≠
    '\u2264': '<=',         # ≤
    '\u2265': '>=',         # ≥
    '\u00d7': 'x',          # ×
    # Arrows
    '\u2192': '->',         # →
    '\u2190': '<-',         # ←
    '\u21d2': '=>',         # ⇒
    # Box drawing
    '\u2500': '-',          # ─
    '\u2502': '|',          # │
    '\u250c': '+',          # ┌
    '\u2510': '+',          # ┐
    '\u2514': '+',          # └
    '\u2518': '+',          # ┘
    '\u251c': '+',          # ├
    '\u2524': '+',          # ┤
    '\u252c': '+',          # ┬
    '\u2534': '+',          # ┴
    '\u253c': '+',          # ┼
    '\u2550': '=',          # ═
    '\u2551': '||',         # ║
    '\u2554': '+',          # ╔
    '\u2557': '+',          # ╗
    '\u255a': '+',          # ╚
    '\u255d': '+',          # ╝
    # Emoji (in case any remain)
    '\u2705': '[OK]',       # ✅
    '\u274c': '[ERROR]',    # ❌
    '\u26a0': '[WARN]',     # ⚠
    '\u2714': '[OK]',       # ✔
    '\u2718': '[FAIL]',     # ✘
    '\U0001f4be': '[SAVE]', # 💾
    '\U0001f4c2': '[DIR]',  # 📂
    '\U0001f4ca': '[CHART]',# 📊
    '\U0001f4c8': '[UP]',   # 📈
    '\U0001f527': '[FIX]',  # 🔧
    '\U0001f9ea': '[TEST]', # 🧪
    '\U0001f3af': '[TARGET]',# 🎯
    '\U0001f6a8': '[ALERT]',# 🚨
    '\U0001f4dd': '[NOTE]', # 📝
    '\U0001f916': '[ML]',   # 🤖
    '\U0001f3f7': '[TAG]',  # 🏷
    '\U0001f4a1': '[IDEA]', # 💡
    '\U0001f552': '[TIME]', # 🕒
    '\u23f3': '[WAIT]',     # ⏳
    '\u2728': '[STAR]',     # ✨
    '\U0001f525': '[HOT]',  # 🔥
    '\U0001f4af': '[100]',  # 💯
    '\U0001f9e0': '[BRAIN]',# 🧠
    '\U0001f50d': '[SEARCH]',# 🔍
    '\U0001f4e6': '[PKG]',  # 📦
    '\U0001f310': '[NET]',  # 🌐
    '\U0001f512': '[LOCK]', # 🔒
    '\u2139': '[INFO]',     # ℹ
    '\u25b6': '>',          # ▶
    '\u25bc': 'v',          # ▼
    '\u25b2': '^',          # ▲
    '\u2022': '*',          # •
    '\u2023': '*',          # ‣
    '\u2026': '...',        # …
    '\u2013': '-',          # –
    '\u2014': '--',         # —
    '\u201c': '"',          # "
    '\u201d': '"',          # "
    '\u2018': "'",          # '
    '\u2019': "'",          # '
    # More Greek
    '\u03b5': 'e',          # ε
    '\u03b8': 'theta',      # θ
    '\u03bb': 'lambda',     # λ
    '\u03c0': 'pi',         # π
    '\u03c1': 'rho',        # ρ
    '\u03c7': 'chi',        # χ
    # More emoji
    '\U0001f4c5': '[DATE]', # 📅
    '\U0001f3b2': '[RAND]', # 🎲
    '\U0001f7e1': '[YEL]',  # 🟡
    '\U0001f4d0': '[TRI]',  # 📐
    '\U0001f52c': '[LAB]',  # 🔬
    '\U0001f4cc': '[PIN]',  # 📌
    '\U0001f4c1': '[FOL]',  # 📁
    '\U0001f517': '[LNK]',  # 🔗
    '\U0001f4cb': '[LST]',  # 📋
    '\u23f1': '[TMR]',      # ⏱
    '\U0001f332': '[TRE]',  # 🌲
    '\U0001f680': '[RKT]',  # 🚀
    '\U0001f319': '[MON]',  # 🌙
    '\U0001f4e7': '[EML]',  # 📧
    '\U0001f510': '[ULK]',  # 🔐
    '\U0001f534': '[RED]',  # 🔴
    '\U0001f389': '[PTY]',  # 🎉
    '\U0001f465': '[USR]',  # 👥
}

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"  [SKIP] Cannot read: {filepath}")
        return 0

    original = content
    count = 0
    for char, replacement in REPLACEMENTS.items():
        if char in content:
            occurrences = content.count(char)
            content = content.replace(char, replacement)
            count += occurrences

    # Also check for any remaining non-ASCII (catch-all)
    remaining = [c for c in content if ord(c) > 127]
    if remaining:
        unique_remaining = set(remaining)
        print(f"  [WARN] {filepath}: {len(unique_remaining)} unmapped non-ASCII chars: "
              f"{[hex(ord(c)) for c in list(unique_remaining)[:10]]}")

    if count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] Fixed {count} chars in {os.path.basename(filepath)}")

    return count

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    py_files = glob.glob(os.path.join(base, '*.py'))

    total_files = 0
    total_chars = 0

    print(f"Scanning {len(py_files)} Python files...")
    for f in sorted(py_files):
        if os.path.basename(f) == 'fix_unicode.py':
            continue
        n = fix_file(f)
        if n > 0:
            total_files += 1
            total_chars += n

    print(f"\nDone. Fixed {total_chars} characters across {total_files} files.")
