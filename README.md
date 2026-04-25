# 🔁 NeuralSeq — RNN Student Performance Evaluator

> Pure-NumPy implementation of SimpleRNN, GRU, and LSTM from scratch.  
> Trained on 600-student academic dataset. Best model: **LSTM — 86.7% accuracy, AUC 0.941**

---

## 📁 Project Structure

```
student_rnn_project/
│
├── dataset.xlsx          ← 600-student dataset (attendance, assignment, quiz, mid, study_hours, result)
│
├── rnn_core.py           ← Full RNN engine (pure NumPy, no TensorFlow)
│                            • SimpleRNNCell — vanilla recurrent cell
│                            • LSTMCell      — 4-gate Long Short-Term Memory
│                            • GRUCell       — 2-gate Gated Recurrent Unit
│                            • RNNModel      — Adam, dropout, L2, BPTT, early stopping, LR decay
│
├── train_rnn.py          ← Full training pipeline (Tasks 1–8)
│                            Trains all 3 variants, evaluates, saves models + plots
│
├── compare_models.py     ← Bonus: Logistic Regression vs ANN vs RNN comparison
│
├── predict.py            ← CLI tool: enter marks, get prediction
│
├── app.py                ← Streamlit UI (5 tabs, hidden state visualisation)
│
├── REPORT.md             ← Final explanations (Tasks 11)
├── README.md             ← This file
│
├── rnn_model.pkl         ← Best trained model (LSTM)
├── model_lstm.pkl        ← LSTM model
├── model_gru.pkl         ← GRU model
├── model_simplernn.pkl   ← SimpleRNN model
├── scaler.joblib         ← StandardScaler (must be loaded with model)
├── config.joblib         ← Metadata (SEQ_LEN, F, best_name)
│
├── requirements.txt      ← pip dependencies
│
├── comparison_dashboard.png   ← CM + ROC for all 3 RNN variants
├── training_curves.png        ← Loss/accuracy curves
├── model_auc_comparison.png   ← AUC bar chart
├── full_comparison.png        ← All 5 models compared
└── radar_comparison.png       ← Radar chart across all metrics
```

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (runs in ~2 minutes)
python train_rnn.py

# 4. (Optional) Run model comparison
python compare_models.py

# 5. Launch the web UI
streamlit run app.py

# 6. Or use CLI predictor
python predict.py
```

---

## 📊 Results Summary

| Model               | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|----------|-----------|--------|-------|-------|
| Logistic Regression | 84.4%    | 87.0%     | 83.3%  | 85.1% | 0.938 |
| ANN (MLP)           | 83.3%    | 88.4%     | 79.2%  | 83.5% | 0.928 |
| SimpleRNN           | 81.1%    | 87.8%     | 75.0%  | 80.9% | 0.936 |
| GRU                 | 83.3%    | 85.1%     | 83.3%  | 84.2% | 0.934 |
| **LSTM** ⭐         | **86.7%**| **86.0%** |**89.6%**|**87.8%**|**0.941**|

---

## 🧠 How It Works

### Input shape
Each student row is expanded into a **5-step semester trajectory**:

```
Week 1 → 55% of final marks  (early semester)
Week 2 → 66% of final marks
Week 3 → 77% of final marks
Week 4 → 88% of final marks
Week 5 → actual marks        (end of semester)
```

This turns a `(600, 5)` table into a `(600, 5, 5)` tensor the RNN reads sequentially.

### RNN forward pass
```
for t in 1..5:
    h_t = cell(x_t, h_{t-1})   # update hidden state
output = Dense(h_5)             # classify from final hidden state
```

### LSTM gates
```
i = σ(W_xi·x + W_hi·h + b_i)   # input gate  — what to write
f = σ(W_xf·x + W_hf·h + b_f)   # forget gate — what to erase
g = tanh(W_xg·x + W_hg·h + b_g) # cell gate  — candidate values
o = σ(W_xo·x + W_ho·h + b_o)   # output gate — what to expose
c_t = f·c_{t-1} + i·g           # new cell state
h_t = o·tanh(c_t)               # new hidden state
```

---

## 🖥️ Streamlit App Tabs

| Tab | Content |
|-----|---------|
| **Prediction panel** | Sliders → Pass/Fail, donut chart, hidden state trajectory, radar chart |
| **Model Performance** | Confusion matrix, ROC curve, training curves, per-class metrics |
| **Model Comparison** | SimpleRNN vs GRU vs LSTM side-by-side table + AUC bar |
| **Feature Analysis** | Permutation importance, distribution plots, correlation heatmap |
| **Architecture** | RNN cell diagrams, unrolled visualisation, BPTT explanation |
| **Dataset Insights** | Class distribution, statistics, box plots, scatter matrix |

---

## 📦 Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.30
openpyxl>=3.1
joblib>=1.3
```

> **No TensorFlow or PyTorch required.** The entire RNN engine is implemented in pure NumPy inside `rnn_core.py`.
