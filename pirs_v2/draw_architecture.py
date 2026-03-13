"""
PIRS V2 Architecture Diagram Generator
Saves: outputs/cert/plots/pirs_v2_architecture.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(__file__), 'outputs', 'cert', 'plots', 'pirs_v2_architecture.png')

# ── Palette ─────────────────────────────────────────────────────────────────
BG        = '#0d1117'
CARD_BG   = '#161b22'
BORDER    = '#30363d'
TEXT_MAIN = '#e6edf3'
TEXT_SUB  = '#8b949e'
ARROW     = '#58a6ff'

C_INPUT   = '#1f4e79'   # blue   — raw data
C_EXTRACT = '#1a3a1a'   # green  — feature extraction
C_DRIFT   = '#3a2a00'   # amber  — drift analysis
C_ANOMALY = '#3a001a'   # rose   — anomaly detection
C_RISK    = '#2a003a'   # purple — risk scoring
C_PERS    = '#003a38'   # teal   — personality
C_INTERV  = '#1a2a3a'   # slate  — intervention
C_RL      = '#1a0a00'   # brown  — reinforcement learning
C_EVAL    = '#2a3a00'   # olive  — evaluation
C_OUTPUT  = '#0a1a0a'   # dark   — outputs

ACCENT = {
    C_INPUT:   '#58a6ff',
    C_EXTRACT: '#3fb950',
    C_DRIFT:   '#d29922',
    C_ANOMALY: '#f85149',
    C_RISK:    '#bc8cff',
    C_PERS:    '#39d353',
    C_INTERV:  '#79c0ff',
    C_RL:      '#ffa657',
    C_EVAL:    '#a8d563',
    C_OUTPUT:  '#8b949e',
}

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 24)
ax.set_ylim(0, 14)
ax.axis('off')

# ── Helper: draw a box ───────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, accent, title, lines=(), tag=''):
    rx = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.08',
                         facecolor=color,
                         edgecolor=accent,
                         linewidth=1.5,
                         zorder=3)
    ax.add_patch(rx)
    # accent bar on top
    bar = FancyBboxPatch((x, y+h-0.12), w, 0.12,
                          boxstyle='round,pad=0',
                          facecolor=accent, alpha=0.7, zorder=4)
    ax.add_patch(bar)
    # tag badge
    if tag:
        ax.text(x+0.18, y+h-0.24, tag,
                color='#0d1117', fontsize=6.5, fontweight='bold',
                va='top', ha='left', zorder=5,
                bbox=dict(facecolor=accent, edgecolor='none',
                          boxstyle='round,pad=0.15', alpha=0.9))
    # title
    ax.text(x + w/2, y + h - 0.22, title,
            color=TEXT_MAIN, fontsize=8.5, fontweight='bold',
            ha='center', va='top', zorder=5)
    # body lines
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.52 - i*0.25, line,
                color=TEXT_SUB, fontsize=6.8,
                ha='center', va='top', zorder=5)

# ── Helper: arrow ────────────────────────────────────────────────────────────
def arrow(ax, x1, y1, x2, y2, label='', vertical=False):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW,
                                lw=1.6, connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = (-0.35, 0) if vertical else (0, 0.18)
        ax.text(mx+offset[0], my+offset[1], label,
                color=ARROW, fontsize=6, ha='center', va='bottom', zorder=6)

# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(12, 13.6, 'PIRS V2 — Pre-Incident Detection Architecture',
        color=TEXT_MAIN, fontsize=16, fontweight='bold', ha='center', va='top')
ax.text(12, 13.15, 'CERT r6.2  ·  4,000 users  ·  516 days  ·  9-layer drift-based pipeline',
        color=TEXT_SUB, fontsize=9.5, ha='center', va='top')

# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — Input data sources  (y=10.8)
# ════════════════════════════════════════════════════════════════════════════
Y1 = 10.8
src_data = [
    ('logon.csv\n(~1.4M rows)', 0.3),
    ('device.csv\n(USB events)', 3.1),
    ('file.csv\n(308K rows)', 5.9),
    ('email.csv\n(1.38M rows)', 8.7),
    ('http.csv\n(117M rows)', 11.5),
    ('LDAP /\n(user metadata)', 14.3),
    ('psychometric.csv\n(OCEAN scores)', 17.1),
]
for label, xpos in src_data:
    box(ax, xpos, Y1, 2.55, 1.35, C_INPUT, ACCENT[C_INPUT],
        label.split('\n')[0], lines=(label.split('\n')[1],),
        tag='INPUT')

# bracket / group label
ax.annotate('', xy=(19.95, Y1+0.67), xytext=(0.3, Y1+0.67),
            arrowprops=dict(arrowstyle='-', color=ACCENT[C_INPUT], lw=1.2))
ax.text(20.2, Y1+0.67, 'Raw CERT r6.2\nDataset',
        color=ACCENT[C_INPUT], fontsize=7.5, va='center', fontweight='bold')

# ════════════════════════════════════════════════════════════════════════════
# Arrow → L0
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, 10.0, Y1, 10.0, Y1-0.5, label='')

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — L0 Feature Extraction  (y=8.5)
# ════════════════════════════════════════════════════════════════════════════
Y2 = 8.5
box(ax, 2.0, Y2, 20.0, 1.65, C_EXTRACT, ACCENT[C_EXTRACT],
    'L0 — Feature Extraction  (cert_extractor.py)',
    lines=('Per-user · per-day behavioral features  |  32 features across 5 activity categories',
           'Logon · USB · File · Email · HTTP   →   1,393,129 user-day records',
           'cert_features.csv'),
    tag='LAYER 0')

arrow(ax, 12.0, Y2, 12.0, Y2-0.5)

# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — L1 Baseline & L2 Drift  (y=6.1)
# ════════════════════════════════════════════════════════════════════════════
Y3 = 6.1
box(ax, 0.3, Y3, 9.8, 1.75, C_DRIFT, ACCENT[C_DRIFT],
    'L1 — Personal Behavioral Baseline',
    lines=('60-day rolling window per user',
           'Z-score deviations from personal norm',
           '27 deviation features per user-day'),
    tag='LAYER 1')

box(ax, 11.9, Y3, 9.8, 1.75, C_DRIFT, ACCENT[C_DRIFT],
    'L2 — Drift Detection  (Linear Regression)',
    lines=('14-day rolling window slope per feature',
           'drift_score = Σ weighted slope deviations',
           'drift_accel = Δ slope (acceleration)'),
    tag='LAYER 2')

ax.annotate('', xy=(11.9, Y3+0.88), xytext=(10.1, Y3+0.88),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.5))
ax.text(11.0, Y3+1.05, 'deviations', color=ARROW, fontsize=6, ha='center')

arrow(ax, 16.8, Y3, 16.8, Y3-0.5)

# ════════════════════════════════════════════════════════════════════════════
# ROW 4 — L4 Anomaly Ensemble  (y=3.9)
# ════════════════════════════════════════════════════════════════════════════
Y4 = 3.9
# Three sub-boxes inside the anomaly band
box(ax, 0.3, Y4, 5.8, 1.85, C_ANOMALY, ACCENT[C_ANOMALY],
    'Isolation Forest',
    lines=('weight = 0.50',
           'contamination = 0.5%',
           '100 estimators'),
    tag='ISOFOREST')

box(ax, 7.1, Y4, 5.8, 1.85, C_ANOMALY, ACCENT[C_ANOMALY],
    'LSTM Autoencoder',
    lines=('weight = 0.35',
           '50K seqs · 7 steps · 32 dims',
           '10 epochs'),
    tag='LSTM')

box(ax, 13.9, Y4, 5.8, 1.85, C_ANOMALY, ACCENT[C_ANOMALY],
    'One-Class SVM',
    lines=('weight = 0.15',
           'kernel = RBF · nu = 0.005',
           'Subsampled for speed'),
    tag='OC-SVM')

# Combine arrows into ensemble label
ax.text(12.0, Y4-0.3, '⊕  Weighted Ensemble Score',
        color=ACCENT[C_ANOMALY], fontsize=8, ha='center', fontweight='bold')
for cx in [3.2, 10.0, 16.8]:
    ax.annotate('', xy=(12.0, Y4-0.15), xytext=(cx, Y4),
                arrowprops=dict(arrowstyle='->', color=ACCENT[C_ANOMALY],
                                lw=1.2, connectionstyle='arc3,rad=0.0'))

# Layer 4 banner above sub-boxes
ax.text(12.0, Y4+1.95, 'L4 — Ensemble Anomaly Detection',
        color=ACCENT[C_ANOMALY], fontsize=9.5, fontweight='bold', ha='center', zorder=5)
# outline rect
rect4 = FancyBboxPatch((0.1, Y4-0.05), 19.8, 2.05,
                        boxstyle='round,pad=0.1', facecolor='none',
                        edgecolor=ACCENT[C_ANOMALY], linewidth=1.2,
                        linestyle='--', zorder=2)
ax.add_patch(rect4)

arrow(ax, 12.0, Y4-0.4, 12.0, Y4-0.85)

# ════════════════════════════════════════════════════════════════════════════
# ROW 5 — L5 Risk Score  (y=2.0)
# ════════════════════════════════════════════════════════════════════════════
Y5 = 2.05
box(ax, 0.3, Y5, 23.4, 1.6, C_RISK, ACCENT[C_RISK],
    'L5 — Risk Scoring & 7/14-Day Breach Prediction',
    lines=('risk_score = 0.50 × anomaly  +  0.35 × slope_norm  +  0.15 × accel_norm   |   Scale 0 – 10',
           'Alert levels: NORMAL · WATCH · ELEVATED · HIGH · CRITICAL   |   Breach trajectory at 7d and 14d',
           'cert_after_prediction.csv'),
    tag='LAYER 5')

arrow(ax, 12.0, Y5, 12.0, Y5-0.5)

# ════════════════════════════════════════════════════════════════════════════
# ROW 6 — L6, L7, L8  (y=0.05)
# ════════════════════════════════════════════════════════════════════════════
Y6 = 0.1
box(ax, 0.3, Y6, 6.5, 1.7, C_PERS, ACCENT[C_PERS],
    'L6 — Personality Profiling',
    lines=('OCEAN + behavioral features',
           'COMPLIANT · SOCIAL · CAREFULL',
           'RISK_TAKER · AUTONOMOUS'),
    tag='LAYER 6')

box(ax, 8.75, Y6, 6.5, 1.7, C_INTERV, ACCENT[C_INTERV],
    'L7 — Intervention Selection',
    lines=('Personality-matched L1–L7',
           'L1 Standard Monitoring',
           '↕  L7 Account Lock'),
    tag='LAYER 7')

box(ax, 17.2, Y6, 6.55, 1.7, C_RL, ACCENT[C_RL],
    'L8 — Q-Learning Policy',
    lines=('Reinforcement learning agent',
           'α=0.1  γ=0.9  ε=0.1',
           'Optimises intervention rewards'),
    tag='LAYER 8')

# Connect L5 → L6 L7 L8
for cx in [3.55, 12.0, 20.47]:
    ax.annotate('', xy=(cx, Y6+1.7), xytext=(12.0, Y5),
                arrowprops=dict(arrowstyle='->', color=ARROW,
                                lw=1.2, connectionstyle='arc3,rad=0.0'))

# ════════════════════════════════════════════════════════════════════════════
# RIGHT SIDE — Outputs / Validation panel
# ════════════════════════════════════════════════════════════════════════════
# Side panel (outputs + validation metrics)
bx, bw = 20.3, 3.5

box(ax, bx, 6.1, bw, 2.1, C_EVAL, ACCENT[C_EVAL],
    'L9 — Evaluation Metrics',
    lines=('RQ1  EPR 7d = 40%  (2/5 insiders)',
           'RQ2  PQ = 0.931',
           'RQ4  Cost saved = $22.8M',
           'ROC-AUC = 0.8554'),
    tag='LAYER 9')

box(ax, bx, 3.9, bw, 2.0, C_OUTPUT, ACCENT[C_OUTPUT],
    'Pipeline Outputs',
    lines=('cert_features.csv · cert_after_drift.csv',
           'cert_after_anomaly.csv',
           'cert_after_prediction.csv',
           'cert_complete.csv · cert_metrics.csv'),
    tag='OUTPUT')

box(ax, bx, 2.05, bw, 1.65, C_OUTPUT, ACCENT[C_OUTPUT],
    'Validation Artefacts',
    lines=('cert_validation_summary.csv',
           'cert_validation_early_warning.csv',
           'insider_trajectories.png'),
    tag='VALIDATE')

# Arrows from main pipeline to side panel
arrow(ax, 23.0, 8.5+0.83, bx, 7.15, label='metrics')
arrow(ax, 23.0, 3.9+0.83, bx, 4.9, label='files')
arrow(ax, 23.0, 2.05+0.83, bx, 2.88, label='plots')

# ════════════════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT,   ACCENT[C_INPUT],   'Raw Input Data'),
    (C_EXTRACT, ACCENT[C_EXTRACT], 'Feature Extraction'),
    (C_DRIFT,   ACCENT[C_DRIFT],   'Baseline & Drift'),
    (C_ANOMALY, ACCENT[C_ANOMALY], 'Anomaly Detection'),
    (C_RISK,    ACCENT[C_RISK],    'Risk Scoring'),
    (C_PERS,    ACCENT[C_PERS],    'Personality'),
    (C_INTERV,  ACCENT[C_INTERV],  'Intervention'),
    (C_RL,      ACCENT[C_RL],      'Q-Learning'),
    (C_EVAL,    ACCENT[C_EVAL],    'Evaluation'),
]
lx, ly = 0.25, -0.08
for i, (bg, ac, lbl) in enumerate(legend_items):
    patch = mpatches.Patch(facecolor=bg, edgecolor=ac, linewidth=1.2, label=lbl)
    ax.add_patch(FancyBboxPatch((lx + i*2.63, ly), 0.35, 0.28,
                                 boxstyle='round,pad=0.04',
                                 facecolor=bg, edgecolor=ac, linewidth=1.2, zorder=3))
    ax.text(lx + i*2.63 + 0.48, ly+0.14, lbl,
            color=TEXT_SUB, fontsize=6.5, va='center')

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=160, facecolor=BG, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {OUT}")
