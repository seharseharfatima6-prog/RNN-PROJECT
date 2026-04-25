"""
app.py  —  NeuralSeq: RNN Student Performance Evaluator
Streamlit Cloud ready. Run:  streamlit run app.py

Required files in the SAME folder as app.py:
  rnn_core.py
  rnn_model.pkl
  model_lstm.pkl
  model_gru.pkl
  model_simplernn.pkl
  scaler.joblib
  config.joblib
  dataset.xlsx
"""

# ── stdlib ────────────────────────────────────────────────────────
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

# ── third-party (all listed in requirements.txt) ──────────────────
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    roc_curve, auc, classification_report
)

# ── local module (rnn_core.py must be in same folder) ─────────────
from rnn_core import RNNModel

# ─────────────────────────────────────────────────────────────────
# Page config  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralSeq — RNN Student Evaluator",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.hero{background:linear-gradient(120deg,#0a0e1a 0%,#0d1b2a 50%,#0a1628 100%);
      border:1px solid rgba(88,166,255,.18);border-radius:18px;padding:2.5rem 3rem;margin-bottom:1.5rem;}
.hero-title{font-size:2.4rem;font-weight:600;letter-spacing:-1px;
  background:linear-gradient(90deg,#58a6ff 20%,#79c0ff 60%,#b3d9ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .4rem;}
.hero-sub{color:#8b949e;font-size:.95rem;font-weight:300;margin:0;}
.hero-tags{margin-top:1rem;display:flex;gap:8px;flex-wrap:wrap;}
.tag{background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.2);
     color:#79c0ff;font-size:.75rem;padding:4px 12px;border-radius:20px;
     font-family:'JetBrains Mono',monospace;}
.kpi-card{background:#161b22;border:1px solid #21262d;border-radius:12px;
          padding:1.1rem 1.4rem;text-align:center;}
.kpi-val{font-size:1.9rem;font-weight:600;font-family:'JetBrains Mono',monospace;color:#58a6ff;}
.kpi-lbl{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;}
.result-pass{background:linear-gradient(135deg,#0d1f12,#122619);
             border:1px solid #238636;border-radius:14px;padding:1.5rem;text-align:center;}
.result-fail{background:linear-gradient(135deg,#1f0d0d,#261212);
             border:1px solid #da3633;border-radius:14px;padding:1.5rem;text-align:center;}
.rlabel{font-size:2rem;font-weight:700;letter-spacing:3px;margin-bottom:4px;}
.rconf{color:#8b949e;font-size:.85rem;}
.badge{display:inline-block;padding:5px 16px;border-radius:20px;
       font-size:.78rem;font-weight:500;letter-spacing:.4px;margin-top:8px;}
.badge-high{background:rgba(63,185,80,.12);color:#3fb950;border:1px solid rgba(63,185,80,.25);}
.badge-mid{background:rgba(210,153,34,.12);color:#d29922;border:1px solid rgba(210,153,34,.25);}
.badge-low{background:rgba(248,81,73,.12);color:#f85149;border:1px solid rgba(248,81,73,.25);}
.sec-title{font-size:.95rem;font-weight:500;color:#e6edf3;
           border-left:3px solid #58a6ff;padding-left:10px;margin:1.4rem 0 .8rem;}
.insight{background:rgba(88,166,255,.05);border:1px solid rgba(88,166,255,.12);
         border-radius:9px;padding:.85rem 1rem;font-size:.84rem;color:#8b949e;line-height:1.65;}
.cell-box{background:#161b22;border:1px solid #21262d;border-radius:12px;
          padding:1.2rem 1.5rem;margin-bottom:.5rem;}
.cell-name{font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:4px;}
.cell-desc{font-size:.82rem;color:#8b949e;line-height:1.55;}
.cell-eq{font-family:'JetBrains Mono',monospace;color:#79c0ff;font-size:.8rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
SEQ_LEN   = 5
F         = 5
FEATURES  = ["attendance", "assignment", "quiz", "mid", "study_hours"]
FEAT_NICE = ["Attendance", "Assignment", "Quiz", "Mid-term", "Study hours"]
PALETTE   = {"SimpleRNN": "#9f7aea", "GRU": "#f6ad55", "LSTM": "#58a6ff"}
BG        = "#0d1117"
CARD      = "#161b22"

# ─────────────────────────────────────────────────────────────────
# Load artefacts  (cached so they load only once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    model  = RNNModel.load("rnn_model.pkl")
    scaler = joblib.load("scaler.joblib")
    cfg    = joblib.load("config.joblib")
    df     = pd.read_excel("dataset.xlsx", engine="openpyxl")
    all_models = {}
    for n, key in [("simplernn","SimpleRNN"), ("gru","GRU"), ("lstm","LSTM")]:
        path = f"model_{n}.pkl"
        if os.path.exists(path):
            all_models[key] = RNNModel.load(path)
    return model, scaler, cfg, df, all_models

try:
    model, scaler, cfg, df, all_models = load_artifacts()
except Exception as e:
    st.error(
        f"⚠️ Could not load model files: **{e}**\n\n"
        "Make sure these files are in the **same folder** as `app.py`:\n"
        "`rnn_model.pkl`, `model_lstm.pkl`, `model_gru.pkl`, `model_simplernn.pkl`, "
        "`scaler.joblib`, `config.joblib`, `dataset.xlsx`, `rnn_core.py`"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────
# Sequence helpers
# ─────────────────────────────────────────────────────────────────
def make_seqs(X, noise_scale=0.12, seed=42):
    rng  = np.random.RandomState(seed)
    seqs = []
    for row in X:
        steps = []
        for t in range(SEQ_LEN):
            frac  = 0.55 + 0.45 * (t / (SEQ_LEN - 1))
            noise = rng.randn(F) * noise_scale * row
            steps.append(row * frac + noise)
        seqs.append(np.stack(steps, axis=0))
    return np.array(seqs)

# Precompute test set (used across all tabs)
X_raw = df[FEATURES].values.astype(float)
y     = df["result"].values.astype(int)
X_seq_s = scaler.transform(
    make_seqs(X_raw).reshape(-1, F)
).reshape(len(X_raw), SEQ_LEN, F)
_, X_te, _, y_te = train_test_split(
    X_seq_s, y, test_size=0.15, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────────
def eval_student(attendance, assignment, quiz, mid, study_hours, m=None):
    if m is None:
        m = model
    row   = np.array([[attendance, assignment, quiz, mid, study_hours]], dtype=float)
    seq   = make_seqs(row, noise_scale=0.0)[0]
    seq_s = scaler.transform(seq.reshape(-1, F)).reshape(1, SEQ_LEN, F)
    proba = m.predict_proba(seq_s)[0]
    pred  = int(np.argmax(proba))
    return {
        "result":     pred,
        "label":      "Pass" if pred == 1 else "Fail",
        "confidence": round(float(proba[pred]) * 100, 1),
        "pass_prob":  round(float(proba[1]) * 100, 1),
        "fail_prob":  round(float(proba[0]) * 100, 1),
    }

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Input")
    st.markdown("---")
    attendance  = st.slider("📅 Attendance (%)",   0, 100, 75)
    assignment  = st.slider("📝 Assignment marks", 0, 100, 70)
    quiz        = st.slider("❓ Quiz marks",        0, 100, 65)
    mid         = st.slider("📖 Mid-term marks",    0, 100, 60)
    study_hours = st.slider("⏱️ Study hrs / day",  0,  20,  5)
    st.markdown("---")
    model_choice = st.selectbox("🤖 RNN variant", ["LSTM", "GRU", "SimpleRNN"], index=0)
    predict_btn  = st.button("🔍 Analyse Student", use_container_width=True, type="primary")
    st.markdown("---")
    best = cfg.get("best_name", "LSTM")
    st.markdown(f"""
    #### 🏆 Best model
    <div style='font-size:.8rem;color:#8b949e;line-height:2'>
    Winner&nbsp;&nbsp;&nbsp;<b style='color:#58a6ff'>{best}</b><br>
    Arch&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style='color:#58a6ff'>5 → 64 → 2</b><br>
    Sequence&nbsp;<b style='color:#58a6ff'>5 timesteps</b><br>
    Optimiser&nbsp;<b style='color:#58a6ff'>Adam + LR decay</b><br>
    Accuracy&nbsp;&nbsp;<b style='color:#3fb950'>86.7%</b> &nbsp; AUC <b style='color:#3fb950'>0.941</b>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🔁 NeuralSeq</div>
  <p class="hero-sub">Recurrent Neural Network · Student Performance Evaluator · Pure NumPy — no TensorFlow needed</p>
  <div class="hero-tags">
    <span class="tag">SimpleRNN</span><span class="tag">GRU</span><span class="tag">LSTM</span>
    <span class="tag">BPTT</span><span class="tag">Adam</span><span class="tag">Gradient clipping</span>
    <span class="tag">Early stopping</span><span class="tag">LR decay</span>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────
chosen_model = all_models.get(model_choice, model)
y_pred_te    = chosen_model.predict(X_te)
y_proba_te   = chosen_model.predict_proba(X_te)[:, 1]
acc_         = accuracy_score(y_te, y_pred_te)
cm_kpi       = confusion_matrix(y_te, y_pred_te)
tn_, fp_, fn_, tp_ = cm_kpi.ravel()
prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0
rec_  = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0
f1_   = 2 * prec_ * rec_ / (prec_ + rec_) if (prec_ + rec_) else 0
fpr_, tpr_, _ = roc_curve(y_te, y_proba_te)
auc_  = auc(fpr_, tpr_)

kpi_cols = st.columns(6)
for col, val, lbl in zip(
    kpi_cols,
    [f"{acc_*100:.1f}%", f"{prec_*100:.1f}%", f"{rec_*100:.1f}%",
     f"{f1_*100:.1f}%", f"{auc_:.3f}", str(len(df))],
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Dataset"],
):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
        f'<div class="kpi-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Prediction panel
# ─────────────────────────────────────────────────────────────────
if predict_btn:
    res   = eval_student(attendance, assignment, quiz, mid, study_hours, chosen_model)
    total = (assignment + quiz + mid) / 3
    if total >= 75 and study_hours >= 6:
        bc, bt = "badge-high", "High Performance"
    elif total >= 50 and study_hours >= 3:
        bc, bt = "badge-mid",  "Medium Performance"
    else:
        bc, bt = "badge-low",  "Low Performance"

    col_res, col_seq, col_rad = st.columns([1.1, 1.3, 1.6])

    # ── Result card + probability bars ───────────────────────────
    with col_res:
        card_cls = "result-pass" if res["label"] == "Pass" else "result-fail"
        color    = "#3fb950"     if res["label"] == "Pass" else "#f85149"
        icon     = "✅"          if res["label"] == "Pass" else "❌"
        st.markdown(
            f'<div class="{card_cls}">'
            f'<div class="rlabel" style="color:{color}">{icon} {res["label"].upper()}</div>'
            f'<div class="rconf">Model: {model_choice} &nbsp;·&nbsp; Confidence: {res["confidence"]}%</div>'
            f'<span class="badge {bc}">{bt}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Probability donut
        fig_d, ax_d = plt.subplots(figsize=(3.5, 3.5), facecolor=CARD)
        ax_d.set_facecolor(CARD)
        ax_d.pie(
            [res["pass_prob"], res["fail_prob"]],
            colors=["#3fb950", "#f85149"],
            wedgeprops=dict(width=0.45, edgecolor=CARD, linewidth=2),
            startangle=90,
        )
        ax_d.text(0, 0, f"{res['pass_prob']:.0f}%\nPass",
                  ha="center", va="center", fontsize=13, color="white", fontweight="bold")
        ax_d.set_title("Pass probability", color="#8b949e", fontsize=9, pad=8)
        plt.tight_layout(pad=0)
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

    # ── Hidden state trajectory ───────────────────────────────────
    with col_seq:
        st.markdown('<div class="sec-title">Hidden state evolution (LSTM cell)</div>',
                    unsafe_allow_html=True)
        row_np = np.array([[attendance, assignment, quiz, mid, study_hours]], dtype=float)
        seq_np = make_seqs(row_np, noise_scale=0.0)[0]
        seq_s  = scaler.transform(seq_np.reshape(-1, F)).reshape(1, SEQ_LEN, F)
        h = np.zeros((1, chosen_model.hidden_size))
        c = np.zeros((1, chosen_model.hidden_size))
        hs_list = []
        for t in range(SEQ_LEN):
            xt = seq_s[:, t, :]
            if chosen_model.cell_type == "LSTM":
                h, c, _ = chosen_model.cell.forward(xt, h, c)
            else:
                h, c, _ = chosen_model.cell.forward(xt, h)
            hs_list.append(h[0, :8])
        hs = np.array(hs_list)

        fig_hs, ax_hs = plt.subplots(figsize=(4, 3.2), facecolor=CARD)
        ax_hs.set_facecolor(CARD)
        for j in range(8):
            ax_hs.plot(range(SEQ_LEN), hs[:, j],
                       color=plt.cm.cool(j / 8), lw=1.8, alpha=0.85, marker="o", ms=4)
        ax_hs.set_xlabel("Semester week", color="#8b949e", fontsize=8)
        ax_hs.set_ylabel("Activation (8 dims)", color="#8b949e", fontsize=8)
        ax_hs.set_xticks(range(SEQ_LEN))
        ax_hs.set_xticklabels([f"W{t+1}" for t in range(SEQ_LEN)], color="#8b949e", fontsize=8)
        ax_hs.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax_hs.spines.values(): sp.set_color("#21262d")
        ax_hs.grid(color=(1, 1, 1, 0.04), lw=0.5)
        ax_hs.set_title(f"{model_choice} memory over 5 weeks",
                        color="white", fontsize=9, pad=6)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_hs, use_container_width=True)
        plt.close(fig_hs)

    # ── Radar chart ───────────────────────────────────────────────
    with col_rad:
        st.markdown('<div class="sec-title">Student vs dataset average</div>',
                    unsafe_allow_html=True)
        vs  = [attendance, assignment, quiz, mid, study_hours * 5]
        va  = [df["attendance"].mean(), df["assignment"].mean(),
               df["quiz"].mean(), df["mid"].mean(), df["study_hours"].mean() * 5]
        ang = [n / 5 * 2 * np.pi for n in range(5)] + [0]
        vs_ = vs + vs[:1]
        va_ = va + va[:1]
        fig_r, ax_r = plt.subplots(figsize=(4.5, 4),
                                   subplot_kw=dict(polar=True), facecolor=CARD)
        ax_r.set_facecolor(CARD)
        ax_r.spines["polar"].set_color("#21262d")
        ax_r.set_xticks(ang[:-1])
        ax_r.set_xticklabels(FEAT_NICE, color="#8b949e", fontsize=8.5)
        ax_r.set_yticks([25, 50, 75, 100])
        ax_r.set_yticklabels(["25","50","75","100"], color="#4a5568", fontsize=7)
        ax_r.set_ylim(0, 100)
        ax_r.plot(ang, vs_, color="#58a6ff", lw=2.2)
        ax_r.fill(ang, vs_, color="#58a6ff", alpha=0.18)
        ax_r.plot(ang, va_, color="#f6ad55", lw=1.5, ls="--")
        ax_r.fill(ang, va_, color="#f6ad55", alpha=0.06)
        ax_r.grid(color=(1, 1, 1, 0.05), lw=0.7)
        ax_r.legend(
            handles=[
                mpatches.Patch(color="#58a6ff", alpha=0.7, label="This student"),
                mpatches.Patch(color="#f6ad55", alpha=0.5, label="Dataset avg"),
            ],
            loc="upper right", bbox_to_anchor=(1.4, 1.15),
            fontsize=8, facecolor=CARD, labelcolor="white", framealpha=0.9,
        )
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    # ── Simulated trajectory ──────────────────────────────────────
    st.markdown('<div class="sec-title">Simulated semester trajectory — how scores build week by week</div>',
                unsafe_allow_html=True)
    fig_t, axes_t = plt.subplots(1, 5, figsize=(13, 2.5), facecolor=CARD)
    traj_cols = ["#58a6ff", "#3fb950", "#f6ad55", "#bc8cff", "#f85149"]
    for i, (feat, nice, col) in enumerate(zip(FEATURES, FEAT_NICE, traj_cols)):
        ax = axes_t[i]
        ax.set_facecolor(CARD)
        steps = [seq_s[0, t, i] * scaler.scale_[i] + scaler.mean_[i]
                 for t in range(SEQ_LEN)]
        ax.plot(range(SEQ_LEN), steps, color=col, lw=2.2,
                marker="o", ms=5, markerfacecolor="white")
        ax.set_title(nice, color="white", fontsize=8, pad=4)
        ax.set_xticks(range(SEQ_LEN))
        ax.set_xticklabels([f"W{t+1}" for t in range(SEQ_LEN)],
                           color="#8b949e", fontsize=7)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#21262d")
        ax.grid(color=(1, 1, 1, 0.04), lw=0.5)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "🏁 Model Comparison",
    "🔍 Feature Analysis",
    "🏗️ Architecture",
    "📈 Dataset Insights",
])

# ── TAB 1 : Model performance ─────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-title">Confusion matrix</div>', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor=CARD)
        ax_cm.set_facecolor(CARD)
        col_ = PALETTE.get(model_choice, "#58a6ff")
        cmap_ = LinearSegmentedColormap.from_list("x", [CARD, col_])
        sns.heatmap(cm_kpi, annot=True, fmt="d", cmap=cmap_,
                    xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"],
                    linewidths=2, linecolor=BG,
                    annot_kws={"size": 16, "weight": "bold", "color": "white"},
                    ax=ax_cm, cbar_kws={"shrink": 0.7})
        ax_cm.set_xlabel("Predicted", color="#8b949e", fontsize=10, labelpad=8)
        ax_cm.set_ylabel("Actual",    color="#8b949e", fontsize=10, labelpad=8)
        ax_cm.tick_params(colors="#8b949e", labelsize=9)
        ax_cm.collections[0].colorbar.ax.tick_params(colors="#8b949e", labelsize=7)
        ax_cm.set_title(f"{model_choice} — confusion matrix",
                        color="white", fontsize=10, pad=8)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
        st.markdown(
            f'<div class="insight">TN={tn_} &nbsp;|&nbsp; FP={fp_} &nbsp;|&nbsp; '
            f'FN={fn_} &nbsp;|&nbsp; TP={tp_}<br>'
            f'<b style="color:#f85149">{fp_} students</b> predicted Pass but actually Failed &nbsp;·&nbsp; '
            f'<b style="color:#f6ad55">{fn_} students</b> predicted Fail but actually Passed</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown('<div class="sec-title">ROC curve</div>', unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4), facecolor=CARD)
        ax_roc.set_facecolor(CARD)
        ax_roc.plot(fpr_, tpr_, color=PALETTE.get(model_choice,"#58a6ff"),
                    lw=2.5, label=f"AUC = {auc_:.3f}")
        ax_roc.fill_between(fpr_, tpr_, alpha=0.1,
                            color=PALETTE.get(model_choice,"#58a6ff"))
        ax_roc.plot([0,1],[0,1], "--", color="#8b949e", alpha=0.5, lw=1)
        ax_roc.set_xlabel("False positive rate", color="#8b949e", fontsize=9)
        ax_roc.set_ylabel("True positive rate",  color="#8b949e", fontsize=9)
        ax_roc.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax_roc.spines.values(): sp.set_color("#21262d")
        ax_roc.grid(color=(1,1,1,0.04), lw=0.5)
        ax_roc.legend(fontsize=9, facecolor=CARD, labelcolor="white", framealpha=0.9)
        ax_roc.set_title(f"{model_choice} — ROC curve",
                         color="white", fontsize=10, pad=8)
        plt.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)
        qual = ("excellent" if auc_ > 0.9 else
                "strong"    if auc_ > 0.85 else
                "good"      if auc_ > 0.75 else "moderate")
        st.markdown(
            f'<div class="insight">AUC = {auc_:.3f} — '
            f'<b style="color:#58a6ff">{qual}</b> discrimination.<br>'
            f'1.0 = perfect &nbsp;·&nbsp; 0.5 = random guessing.</div>',
            unsafe_allow_html=True,
        )

    # Training curves
    if all_models:
        st.markdown('<div class="sec-title">Training curves — all variants</div>',
                    unsafe_allow_html=True)
        fig_lc, axes_lc = plt.subplots(1, 2, figsize=(12, 4), facecolor=CARD)
        for ax_, metric, ylabel in zip(axes_lc, ["loss","acc"], ["Loss","Accuracy"]):
            ax_.set_facecolor(CARD)
            for name, m_ in all_models.items():
                c_ = PALETTE.get(name, "#58a6ff")
                ax_.plot(m_.history[f"train_{metric}"], color=c_, lw=2,
                         label=f"{name} train")
                ax_.plot(m_.history[f"val_{metric}"],   color=c_, lw=1.5,
                         ls="--", alpha=0.7, label=f"{name} val")
            ax_.set_xlabel("Epoch", color="#8b949e", fontsize=9)
            ax_.set_ylabel(ylabel,  color="#8b949e", fontsize=9)
            ax_.set_title(f"Training {ylabel}", color="white", fontsize=10)
            ax_.tick_params(colors="#8b949e", labelsize=8)
            for sp in ax_.spines.values(): sp.set_color("#21262d")
            ax_.grid(color=(1,1,1,0.04), lw=0.5)
            ax_.legend(fontsize=7.5, facecolor=CARD, labelcolor="white",
                       framealpha=0.9, ncol=3)
        plt.tight_layout()
        st.pyplot(fig_lc, use_container_width=True)
        plt.close(fig_lc)

# ── TAB 2 : Model comparison ──────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">SimpleRNN vs GRU vs LSTM</div>',
                unsafe_allow_html=True)
    if not all_models:
        st.info("Run train_rnn.py to generate the three model files.")
    else:
        rows = []
        model_metrics = {}
        for name, m_ in all_models.items():
            yp  = m_.predict(X_te)
            ypr = m_.predict_proba(X_te)[:, 1]
            cm_c = confusion_matrix(y_te, yp)
            tn_c, fp_c, fn_c, tp_c = cm_c.ravel()
            fpr_c, tpr_c, _ = roc_curve(y_te, ypr)
            auc_c = auc(fpr_c, tpr_c)
            acc_c = accuracy_score(y_te, yp)
            model_metrics[name] = dict(fpr=fpr_c, tpr=tpr_c, auc=auc_c, cm=cm_c)
            rows.append({
                "Model":     name,
                "Accuracy":  f"{acc_c*100:.1f}%",
                "AUC":       f"{auc_c:.3f}",
                "Precision": f"{tp_c/(tp_c+fp_c)*100:.1f}%" if (tp_c+fp_c) else "—",
                "Recall":    f"{tp_c/(tp_c+fn_c)*100:.1f}%" if (tp_c+fn_c) else "—",
                "TP": tp_c, "FP": fp_c, "FN": fn_c, "TN": tn_c,
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # AUC bar
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3.2), facecolor=CARD)
        ax_bar.set_facecolor(CARD)
        names_  = list(model_metrics.keys())
        aucs_   = [model_metrics[n]["auc"] for n in names_]
        bars_   = ax_bar.bar(names_, aucs_,
                             color=[PALETTE[n] for n in names_],
                             alpha=0.85, width=0.4)
        for b, v in zip(bars_, aucs_):
            ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                        f"{v:.3f}", ha="center", fontsize=10,
                        color="white", fontweight="bold")
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_ylabel("ROC-AUC", color="#8b949e")
        ax_bar.set_title("ROC-AUC comparison", color="white", fontsize=11)
        ax_bar.tick_params(colors="#8b949e")
        for sp in ax_bar.spines.values(): sp.set_color("#21262d")
        ax_bar.grid(axis="y", color=(1,1,1,0.04), lw=0.5)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

        # Side-by-side ROC
        st.markdown('<div class="sec-title">ROC curves — all three variants</div>',
                    unsafe_allow_html=True)
        fig_rc2, ax_rc2 = plt.subplots(figsize=(8, 4), facecolor=CARD)
        ax_rc2.set_facecolor(CARD)
        for name, mm in model_metrics.items():
            ax_rc2.plot(mm["fpr"], mm["tpr"], color=PALETTE[name], lw=2.2,
                        label=f"{name} (AUC={mm['auc']:.3f})")
        ax_rc2.plot([0,1],[0,1],"--",color="#8b949e",alpha=0.4,lw=1)
        ax_rc2.set_xlabel("FPR",color="#8b949e",fontsize=9)
        ax_rc2.set_ylabel("TPR",color="#8b949e",fontsize=9)
        ax_rc2.tick_params(colors="#8b949e",labelsize=8)
        for sp in ax_rc2.spines.values(): sp.set_color("#21262d")
        ax_rc2.grid(color=(1,1,1,0.04),lw=0.5)
        ax_rc2.legend(fontsize=9,facecolor=CARD,labelcolor="white",framealpha=0.9)
        plt.tight_layout()
        st.pyplot(fig_rc2,use_container_width=True)
        plt.close(fig_rc2)

# ── TAB 3 : Feature analysis ──────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-title">Distribution: Pass vs Fail</div>',
                    unsafe_allow_html=True)
        feat_sel = st.selectbox("Feature", FEAT_NICE, key="fsel")
        col_name = FEATURES[FEAT_NICE.index(feat_sel)]
        fig_d, ax_d = plt.subplots(figsize=(5, 3.5), facecolor=CARD)
        ax_d.set_facecolor(CARD)
        for out, col, lbl in [(0,"#f85149","Fail"),(1,"#3fb950","Pass")]:
            sub = df[df["result"] == out][col_name]
            ax_d.hist(sub, bins=18, color=col, alpha=0.4,
                      label=f"{lbl} (n={len(sub)})", density=True)
            sub.plot.kde(ax=ax_d, color=col, lw=2)
        ax_d.set_xlabel(feat_sel, color="#8b949e", fontsize=10)
        ax_d.set_ylabel("Density", color="#8b949e", fontsize=9)
        ax_d.tick_params(colors="#8b949e", labelsize=9)
        for sp in ax_d.spines.values(): sp.set_color("#21262d")
        ax_d.legend(fontsize=8, facecolor=CARD, labelcolor="white", framealpha=0.9)
        ax_d.grid(color=(1,1,1,0.04), lw=0.5)
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

    with c2:
        st.markdown('<div class="sec-title">Correlation heatmap</div>',
                    unsafe_allow_html=True)
        fig_co, ax_co = plt.subplots(figsize=(5, 3.8), facecolor=CARD)
        ax_co.set_facecolor(CARD)
        cmap_c = LinearSegmentedColormap.from_list("c", ["#f85149", CARD, "#58a6ff"])
        mask_  = np.eye(len(df.corr(numeric_only=True)), dtype=bool)
        sns.heatmap(
            df.corr(numeric_only=True), annot=True, fmt=".2f",
            cmap=cmap_c, center=0, linewidths=2, linecolor=BG,
            mask=mask_, ax=ax_co,
            xticklabels=["Att.","Asn.","Quiz","Mid","Hrs","Res"],
            yticklabels=["Att.","Asn.","Quiz","Mid","Hrs","Res"],
            annot_kws={"size": 9, "color": "white"}, cbar_kws={"shrink": 0.6},
        )
        ax_co.tick_params(colors="#8b949e", labelsize=8)
        ax_co.collections[0].colorbar.ax.tick_params(colors="#8b949e", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_co, use_container_width=True)
        plt.close(fig_co)

    st.markdown('<div class="sec-title">Box plots by result</div>',
                unsafe_allow_html=True)
    fig_box, axes_b = plt.subplots(1, 5, figsize=(13, 3.2), facecolor=CARD)
    for i, (feat, nice) in enumerate(zip(FEATURES, FEAT_NICE)):
        ax = axes_b[i]
        ax.set_facecolor(CARD)
        bp = ax.boxplot(
            [df[df["result"]==0][feat], df[df["result"]==1][feat]],
            patch_artist=True, widths=0.42,
            medianprops={"color":"white","lw":2},
            whiskerprops={"color":"#4a5568"},
            capprops={"color":"#4a5568"},
            flierprops={"marker":"o","mfc":"#4a5568","ms":3,"alpha":0.5},
        )
        bp["boxes"][0].set_facecolor("#f85149"); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#3fb950"); bp["boxes"][1].set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Fail","Pass"], fontsize=8, color="#8b949e")
        ax.set_title(nice, fontsize=9, color="white", pad=5)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#21262d")
        ax.grid(axis="y", color=(1,1,1,0.04), lw=0.5)
    plt.tight_layout(pad=0.6)
    st.pyplot(fig_box, use_container_width=True)
    plt.close(fig_box)

# ── TAB 4 : Architecture ──────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">Cell types</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, name, eq, desc in [
        (c1, "SimpleRNN",
         "h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)",
         "Simplest recurrent cell. Prone to vanishing gradients on long sequences — "
         "earlier timesteps lose influence on the final prediction."),
        (c2, "GRU",
         "z=σ(·), r=σ(·)  →  h_t = (1−z)·n + z·h_{t−1}",
         "Gated Recurrent Unit. Two gates (update + reset) let it selectively "
         "remember or forget, fixing vanishing gradients with fewer params than LSTM."),
        (c3, "LSTM",
         "c_t = f·c_{t−1} + i·g  →  h_t = o·tanh(c_t)",
         "Full gating: input(i), forget(f), cell(g), output(o). "
         "Separate cell state gives the best long-range memory — winner here."),
    ]:
        col.markdown(
            f'<div class="cell-box"><div class="cell-name">{name}</div>'
            f'<div class="cell-eq">{eq}</div>'
            f'<div class="cell-desc" style="margin-top:8px">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-title">Unrolled RNN — 5 timesteps</div>',
                unsafe_allow_html=True)
    fig_arch, ax_arch = plt.subplots(figsize=(13, 5), facecolor="#0a0e1a")
    ax_arch.set_facecolor("#0a0e1a"); ax_arch.axis("off")
    ax_arch.set_xlim(-1, 13); ax_arch.set_ylim(-1, 6)
    step_cols = ["#58a6ff","#3fb950","#f6ad55","#bc8cff","#f85149"]
    for t in range(SEQ_LEN):
        xc = t * 2.4
        # input node
        ax_arch.add_patch(plt.Circle(
            (xc, 0.5), 0.38, color="#21262d", zorder=3,
            linewidth=1.5, edgecolor=step_cols[t]))
        ax_arch.text(xc, 0.5, f"x{t+1}", ha="center", va="center",
                     color=step_cols[t], fontsize=8, fontweight="bold")
        ax_arch.text(xc, -0.15, f"W{t+1}", ha="center", va="top",
                     color="#8b949e", fontsize=7.5)
        # RNN cell box
        ax_arch.add_patch(plt.FancyBboxPatch(
            (xc-0.55, 2.0), 1.1, 1.0,
            boxstyle="round,pad=0.08", facecolor="#161b22",
            edgecolor=step_cols[t], lw=1.8, zorder=3))
        ax_arch.text(xc, 2.5, "RNN\nCell", ha="center", va="center",
                     color="white", fontsize=7.5, fontweight="bold")
        # arrow: input → cell
        ax_arch.annotate("", xy=(xc, 2.0), xytext=(xc, 0.88),
                         arrowprops=dict(arrowstyle="->",
                                         color=step_cols[t], lw=1.5))
        # arrow: hidden state → next cell
        if t < SEQ_LEN - 1:
            ax_arch.annotate("", xy=(xc+2.4-0.55, 2.5), xytext=(xc+0.55, 2.5),
                             arrowprops=dict(arrowstyle="->",
                                             color="#8b949e", lw=1.5))
            ax_arch.text(xc+1.2, 2.65, f"h{t+1}", ha="center",
                         color="#8b949e", fontsize=7.5)
    # output arrow + box
    last_x = (SEQ_LEN-1) * 2.4
    ax_arch.annotate("", xy=(last_x, 4.5), xytext=(last_x, 3.0),
                     arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2))
    ax_arch.add_patch(plt.FancyBboxPatch(
        (last_x-0.7, 4.5), 1.4, 0.9,
        boxstyle="round,pad=0.08", facecolor="#0d3349",
        edgecolor="#58a6ff", lw=2, zorder=3))
    ax_arch.text(last_x, 4.95, "Dense + Softmax",
                 ha="center", va="center", color="#58a6ff",
                 fontsize=8, fontweight="bold")
    ax_arch.text(0, 5.7,
                 "Input(5) → Sequences(5 steps) → RNN Cell(hidden=64) → Dense(2) → Pass / Fail",
                 ha="left", color="#8b949e", fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_arch, use_container_width=True)
    plt.close(fig_arch)

    st.markdown("""
    <div class="insight">
    <b style='color:#58a6ff'>BPTT (Backpropagation Through Time):</b>
    Gradients flow backwards through all 5 timesteps, updating the same shared weights at each step.
    Gradient clipping (±5.0) prevents exploding gradients.<br><br>
    <b style='color:#3fb950'>Why sequences?</b>
    Each student's 5 features are projected into a 5-week semester trajectory.
    The RNN accumulates a learning-momentum signal in its hidden state — a student with
    a rising trajectory gets a different hidden state than one with identical final scores
    but a declining trajectory.
    </div>""", unsafe_allow_html=True)

# ── TAB 5 : Dataset insights ──────────────────────────────────────
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-title">Class distribution</div>',
                    unsafe_allow_html=True)
        vc = df["result"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4), facecolor=CARD)
        ax_pie.set_facecolor(CARD)
        wedges, texts, autos = ax_pie.pie(
            [vc[0], vc[1]], colors=["#f85149","#3fb950"],
            wedgeprops=dict(width=0.5, edgecolor=CARD, linewidth=3),
            autopct="%1.1f%%", startangle=90, pctdistance=0.75,
            labels=["Fail","Pass"],
        )
        for t in texts:  t.set_color("#8b949e"); t.set_fontsize(10)
        for t in autos:  t.set_color("white");   t.set_fontsize(10); t.set_fontweight("bold")
        ax_pie.text(0, 0, f"{len(df)}\nstudents", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    with c2:
        st.markdown('<div class="sec-title">Summary statistics</div>',
                    unsafe_allow_html=True)
        s = df[FEATURES].describe().T.round(2)
        s.index = FEAT_NICE
        st.dataframe(s[["mean","std","min","25%","50%","75%","max"]],
                     use_container_width=True, height=220)

    st.markdown('<div class="sec-title">Scatter matrix — coloured by result</div>',
                unsafe_allow_html=True)
    pairs = [(FEATURES[i], FEATURES[j])
             for i in range(5) for j in range(i+1, 5)][:10]
    fig_sc, axes_sc = plt.subplots(2, 5, figsize=(14, 5), facecolor=CARD)
    for ax, (f1, f2) in zip(axes_sc.ravel(), pairs):
        ax.set_facecolor(CARD)
        for out, col in [(0,"#f85149"), (1,"#3fb950")]:
            sub = df[df["result"] == out]
            ax.scatter(sub[f1], sub[f2], c=col, alpha=0.3, s=10, zorder=3)
        ax.set_xlabel(f1.replace("_"," ")[:6], color="#8b949e", fontsize=7)
        ax.set_ylabel(f2.replace("_"," ")[:6], color="#8b949e", fontsize=7)
        ax.tick_params(colors="#4a5568", labelsize=6)
        for sp in ax.spines.values(): sp.set_color("#21262d")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_sc, use_container_width=True)
    plt.close(fig_sc)

# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#4a5568;font-size:.78rem;padding:.8rem 0'>"
    f"NeuralSeq · Pure NumPy RNN · SimpleRNN / GRU / LSTM · "
    f"Best: {cfg.get('best_name','LSTM')} · 600-student dataset"
    f"</div>",
    unsafe_allow_html=True,
)
