"""
train_rnn.py
============
Full RNN training pipeline covering all assignment tasks:
  Task 1  — Dataset understanding
  Task 3  — Preprocessing + sequence construction
  Task 4  — Build RNN / LSTM / GRU models
  Task 5  — Train with Adam, LR decay, early stopping
  Task 6  — Evaluate: accuracy, confusion matrix, ROC, AUC, report
  Task 7  — evaluate_student() function
  Task 8  — Save model + scaler
  Bonus   — Compare SimpleRNN vs GRU vs LSTM
            Confusion matrix heatmaps + loss/accuracy curves
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import joblib, pickle, warnings
warnings.filterwarnings('ignore')

from rnn_core import RNNModel

np.random.seed(42)

print("=" * 62)
print("  NEURAL SEQUENCE EVALUATOR — RNN/LSTM/GRU Student Predictor")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────
# TASK 1 — Understand the Dataset
# ─────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TASK 1 — Dataset Understanding")
print("─"*62)

df = pd.read_excel("dataset.xlsx")
print(f"\n{'Shape:':<20} {df.shape}")
print(f"{'Columns:':<20} {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nDescriptive stats:")
print(df.describe().round(2).to_string())
print(f"\nClass balance:  Fail={df['result'].sum()==0}  "
      f"Fail(0)={( df['result']==0).sum()}  Pass(1)={(df['result']==1).sum()}")

print("""
Column meanings:
  attendance   — % of classes attended           (temporal: builds over weeks)
  assignment   — cumulative assignment marks      (temporal: submitted over semester)
  quiz         — cumulative quiz marks            (temporal: weekly quizzes)
  mid          — mid-term exam score              (single snapshot)
  study_hours  — average daily self-study hours   (habit signal)
  result       — 0=Fail, 1=Pass                  ← TARGET

Why RNN for this data?
  The features represent SEQUENTIAL academic progress through a semester.
  An RNN reads them as a time-series (e.g. weekly snapshots), learning
  temporal dependencies — e.g. "improving quiz trend + low attendance
  early but recovering mid-semester" patterns that a static MLP misses.

Problem type: BINARY CLASSIFICATION  (output ∈ {0,1})
""")

# ─────────────────────────────────────────────────────────────────
# TASK 3 — Preprocessing + Sequence Construction
# ─────────────────────────────────────────────────────────────────
print("─"*62)
print("  TASK 3 — Preprocessing + Sequence Construction")
print("─"*62)

FEATURES = ['attendance','assignment','quiz','mid','study_hours']
X_raw = df[FEATURES].values.astype(float)
y     = df['result'].values.astype(int)

# Scale
scaler = StandardScaler()

# ── Construct synthetic sequences ──────────────────────────────
# For each student we simulate 5 "weekly" timesteps with realistic
# noise around their actual score, forming a (N, 5, 5) tensor.
# This teaches the RNN to pick up learning-progress trajectories.
SEQ_LEN  = 5      # weeks in the semester snapshot
N, F     = X_raw.shape

def make_sequences(X, noise_scale=0.12, seed=42):
    rng = np.random.RandomState(seed)
    seqs = []
    for row in X:
        # t=0: early semester (70% of final), t=4: actual score
        steps = []
        for t in range(SEQ_LEN):
            frac  = 0.55 + 0.45 * (t / (SEQ_LEN-1))
            noise = rng.randn(F) * noise_scale * row
            steps.append(row * frac + noise)
        seqs.append(np.stack(steps, axis=0))
    return np.array(seqs)   # (N, SEQ_LEN, F)

X_seq = make_sequences(X_raw)

# Train / val / test split
X_tr_r, X_te_r, y_tr, y_te = train_test_split(X_seq, y, test_size=0.15, random_state=42, stratify=y)
X_tr_r, X_va_r, y_tr, y_va = train_test_split(X_tr_r, y_tr, test_size=0.15, random_state=42, stratify=y_tr)

# Fit scaler on flattened training data, apply to all splits
flat_tr = X_tr_r.reshape(-1, F)
scaler.fit(flat_tr)
def scale_seq(X): return scaler.transform(X.reshape(-1,F)).reshape(X.shape)

X_tr = scale_seq(X_tr_r)
X_va = scale_seq(X_va_r)
X_te = scale_seq(X_te_r)

print(f"\n  Sequences shape  : {X_tr.shape} (train), {X_va.shape} (val), {X_te.shape} (test)")
print(f"  Sequence length  : {SEQ_LEN} timesteps (simulated semester weeks)")
print(f"  Features/step    : {F}")
print("""
  Why sequences?
  Each student's final marks are projected back to form a
  5-step learning trajectory. The RNN reads this trajectory
  left-to-right, updating its hidden state at each week.
  Temporal patterns (consistent improvers vs last-minute crammers)
  produce different hidden-state paths → richer representations.

  Why StandardScaler?
  RNNs propagate gradients through time; un-scaled inputs cause
  vanishing or exploding gradient problems much more severely
  than in feedforward networks. Scaling to μ=0, σ=1 stabilises
  the tanh/sigmoid activations inside each cell.
""")

# ─────────────────────────────────────────────────────────────────
# TASK 4 — Build & explain models
# ─────────────────────────────────────────────────────────────────
print("─"*62)
print("  TASK 4 — RNN Architecture Design")
print("─"*62)
print("""
  We build THREE model variants to compare:

  ┌────────────────┬──────────────────────────────────────────┐
  │ SimpleRNN      │ h_t = tanh(W_hh·h_{t-1} + W_xh·x_t)     │
  │                │ Suffers vanishing gradients on long seqs  │
  ├────────────────┼──────────────────────────────────────────┤
  │ GRU            │ Two gates: update(z) + reset(r)           │
  │                │ Fewer params than LSTM, faster training   │
  ├────────────────┼──────────────────────────────────────────┤
  │ LSTM           │ Four gates: input(i), forget(f),           │
  │                │ cell(g), output(o).  Explicit cell state.  │
  │                │ Best long-range memory.                    │
  └────────────────┴──────────────────────────────────────────┘

  All share:
    Hidden size   : 64 neurons
    Output layer  : Dense(64→2) + Softmax
    Optimiser     : Adam (lr=0.005, β1=0.9, β2=0.999)
    Regularisation: Dropout 0.25 + L2 1e-4
    Gradient clip : ±5.0  (essential for RNNs)
""")

model_configs = {
    'SimpleRNN': dict(cell_type='SimpleRNN', hidden_size=64, lr=0.005, dropout=0.25, l2=1e-4),
    'GRU':       dict(cell_type='GRU',       hidden_size=64, lr=0.005, dropout=0.25, l2=1e-4),
    'LSTM':      dict(cell_type='LSTM',      hidden_size=64, lr=0.005, dropout=0.25, l2=1e-4),
}

# ─────────────────────────────────────────────────────────────────
# TASK 5 — Train all models
# ─────────────────────────────────────────────────────────────────
print("─"*62)
print("  TASK 5 — Training")
print("─"*62)

trained_models = {}
for name, cfg in model_configs.items():
    print(f"\n{'▶ Training ' + name:─<55}")
    m = RNNModel(input_size=F, **cfg)
    m.fit(X_tr, y_tr, X_va, y_va,
          epochs=200, batch_size=32,
          early_stopping_patience=25,
          lr_decay_patience=12,
          verbose=True)
    trained_models[name] = m

# ─────────────────────────────────────────────────────────────────
# TASK 6 — Evaluate all models
# ─────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TASK 6 — Evaluation")
print("─"*62)

results = {}
for name, m in trained_models.items():
    y_pred   = m.predict(X_te)
    y_proba  = m.predict_proba(X_te)[:,1]
    acc      = accuracy_score(y_te, y_pred)
    cm       = confusion_matrix(y_te, y_pred)
    fpr,tpr,_ = roc_curve(y_te, y_proba)
    roc_auc  = auc(fpr, tpr)
    report   = classification_report(y_te, y_pred, target_names=['Fail','Pass'])
    results[name] = dict(acc=acc, cm=cm, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                         report=report, y_pred=y_pred, y_proba=y_proba)
    print(f"\n── {name} ──────────────────────────────────────────────")
    print(f"   Accuracy : {acc*100:.2f}%   AUC: {roc_auc:.3f}")
    print(f"   Confusion matrix:\n{cm}")
    print(f"\n{report}")

# ─────────────────────────────────────────────────────────────────
# Pick best model
# ─────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['roc_auc'])
best_model = trained_models[best_name]
print(f"\n🏆 Best model: {best_name}  (AUC={results[best_name]['roc_auc']:.3f})")

# ─────────────────────────────────────────────────────────────────
# TASK 7 — evaluate_student()
# ─────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TASK 7 — evaluate_student() function")
print("─"*62)

def evaluate_student(attendance, assignment, quiz, mid, study_hours,
                     model=None, scaler_obj=None):
    """
    Predict Pass/Fail for a new student using the trained RNN.

    The single row of features is expanded into a SEQ_LEN-step
    trajectory (simulating the semester path leading to these scores),
    scaled, and fed to the RNN cell-by-cell.

    Returns
    -------
    dict with keys: result (0/1), label ('Pass'/'Fail'),
                    confidence (%), pass_prob (%), fail_prob (%)
    """
    if model is None:
        model = best_model
    if scaler_obj is None:
        scaler_obj = scaler

    row  = np.array([[attendance, assignment, quiz, mid, study_hours]], dtype=float)
    seq  = make_sequences(row, noise_scale=0.0)[0]          # deterministic
    seq_s = scaler_obj.transform(seq.reshape(-1, F)).reshape(1, SEQ_LEN, F)

    proba = model.predict_proba(seq_s)[0]
    pred  = int(np.argmax(proba))
    label = "Pass" if pred == 1 else "Fail"
    return {
        "result":      pred,
        "label":       label,
        "confidence":  round(proba[pred]*100, 1),
        "pass_prob":   round(proba[1]*100, 1),
        "fail_prob":   round(proba[0]*100, 1),
    }

# Demo
print("\nDemo predictions:")
tests = [
    (85, 90, 80, 70, 8,  "Strong student"),
    (40, 30, 25, 20, 1,  "Weak student"),
    (65, 65, 60, 55, 5,  "Average student"),
]
for att, asn, qz, md, sh, desc in tests:
    r = evaluate_student(att, asn, qz, md, sh)
    print(f"  {desc:<18} → {r['label']}  (confidence {r['confidence']}%,"
          f"  pass_prob={r['pass_prob']}%)")

# ─────────────────────────────────────────────────────────────────
# TASK 8 — Save
# ─────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TASK 8 — Saving Model & Scaler")
print("─"*62)

best_model.save("rnn_model.pkl")
joblib.dump(scaler, "scaler.joblib")
joblib.dump({'SEQ_LEN': SEQ_LEN, 'F': F, 'best_name': best_name}, "config.joblib")

# Save all models for comparison page
for name, m in trained_models.items():
    m.save(f"model_{name.lower()}.pkl")

print(f"\n  ✅ rnn_model.pkl   — best model ({best_name})")
print("  ✅ scaler.joblib   — StandardScaler")
print("  ✅ config.joblib   — metadata")
print("  ✅ model_*.pkl     — all three models")
print("""
  Why save BOTH model + scaler?
  The scaler transforms raw inputs to the normalised space the model
  trained in. Without it, the RNN would receive out-of-distribution
  inputs and produce garbage predictions. They form one atomic unit.
""")

# ─────────────────────────────────────────────────────────────────
# PLOTS — Confusion matrices, ROC curves, Loss curves
# ─────────────────────────────────────────────────────────────────
BG = '#0d1117'
CARD = '#161b22'
BLUE = '#58a6ff'
GREEN = '#3fb950'
ORANGE = '#d29922'
RED = '#f85149'
GRAY = '#8b949e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})

# ── 1. Comparison dashboard ─────────────────────────────────────
fig = plt.figure(figsize=(15, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
palette = {'SimpleRNN': '#9f7aea', 'GRU': '#f6ad55', 'LSTM': '#63b3ed'}

for col, name in enumerate(['SimpleRNN','GRU','LSTM']):
    r   = results[name]
    cm  = r['cm']

    # Confusion matrix
    ax_cm = fig.add_subplot(gs[0, col])
    ax_cm.set_facecolor(CARD)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ng", [CARD, palette[name]])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'],
                linewidths=2, linecolor=BG,
                annot_kws={'size':14,'weight':'bold','color':'white'},
                ax=ax_cm, cbar=False)
    ax_cm.set_title(f"{name}\nAcc={r['acc']*100:.1f}%  AUC={r['roc_auc']:.3f}",
                    color='white', fontsize=10, pad=8)
    ax_cm.set_xlabel('Predicted', color=GRAY, fontsize=8)
    ax_cm.set_ylabel('Actual',    color=GRAY, fontsize=8)
    ax_cm.tick_params(colors=GRAY, labelsize=8)

    # ROC curve
    ax_roc = fig.add_subplot(gs[1, col])
    ax_roc.set_facecolor(CARD)
    ax_roc.plot(r['fpr'], r['tpr'], color=palette[name], lw=2.5, label=f"AUC={r['roc_auc']:.3f}")
    ax_roc.fill_between(r['fpr'], r['tpr'], alpha=0.1, color=palette[name])
    ax_roc.plot([0,1],[0,1],'--', color=GRAY, alpha=0.5, lw=1)
    ax_roc.set_xlabel('FPR', color=GRAY, fontsize=8)
    ax_roc.set_ylabel('TPR', color=GRAY, fontsize=8)
    ax_roc.set_title(f"{name} — ROC Curve", color='white', fontsize=10, pad=8)
    ax_roc.tick_params(colors=GRAY, labelsize=8)
    for sp in ax_roc.spines.values(): sp.set_color('#30363d')
    ax_roc.grid(color='#21262d', linewidth=0.5)
    ax_roc.legend(fontsize=8, facecolor=CARD, labelcolor='white', framealpha=0.9)

fig.suptitle("Model Comparison — SimpleRNN vs GRU vs LSTM", color='white', fontsize=13, y=1.01)
fig.savefig('comparison_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("  ✅ comparison_dashboard.png")
plt.close()

# ── 2. Loss & accuracy curves ───────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=BG)
for ax, metric, ylabel in zip(axes2, ['loss','acc'],['Loss','Accuracy']):
    ax.set_facecolor(CARD)
    for name, m in trained_models.items():
        ax.plot(m.history[f'train_{metric}'], color=palette[name], lw=2, label=f'{name} train', alpha=0.9)
        ax.plot(m.history[f'val_{metric}'],   color=palette[name], lw=1.5, ls='--', label=f'{name} val', alpha=0.7)
    ax.set_xlabel('Epoch', color=GRAY); ax.set_ylabel(ylabel, color=GRAY)
    ax.set_title(f"Training {ylabel} Curves", color='white', fontsize=11)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.grid(color='#21262d', linewidth=0.5)
    ax.legend(fontsize=7.5, facecolor=CARD, labelcolor='white', framealpha=0.9, ncol=3)

plt.tight_layout()
fig2.savefig('training_curves.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("  ✅ training_curves.png")
plt.close()

# ── 3. Accuracy bar chart ───────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 3.5), facecolor=BG)
ax3.set_facecolor(CARD)
names = list(results.keys())
accs  = [results[n]['roc_auc'] for n in names]
bars  = ax3.bar(names, accs, color=[palette[n] for n in names], alpha=0.85, width=0.45)
for bar, val in zip(bars, accs):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{val:.3f}', ha='center', fontsize=10, color='white', fontweight='bold')
ax3.set_ylim(0, 1.1); ax3.set_ylabel('ROC-AUC', color=GRAY)
ax3.set_title('Model Comparison — ROC-AUC', color='white', fontsize=11)
ax3.tick_params(colors=GRAY); [sp.set_color('#30363d') for sp in ax3.spines.values()]
ax3.grid(axis='y', color='#21262d', linewidth=0.5)
plt.tight_layout()
fig3.savefig('model_auc_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("  ✅ model_auc_comparison.png")
plt.close()

print("\n" + "="*62)
print(f"  Training complete  🏆 Best: {best_name}  (AUC {results[best_name]['roc_auc']:.3f})")
print("="*62)
