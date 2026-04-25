"""
predict.py — CLI interface for RNN Student Evaluator
Run: python predict.py
"""

import numpy as np
import joblib
from rnn_core import RNNModel

SEQ_LEN = 5
F       = 5

model  = RNNModel.load("rnn_model.pkl")
scaler = joblib.load("scaler.joblib")
cfg    = joblib.load("config.joblib")

def make_sequences(X, noise_scale=0.0):
    seqs = []
    for row in X:
        steps = [row * (0.55 + 0.45*(t/(SEQ_LEN-1))) for t in range(SEQ_LEN)]
        seqs.append(np.stack(steps, axis=0))
    return np.array(seqs)

def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    row  = np.array([[attendance, assignment, quiz, mid, study_hours]], dtype=float)
    seq  = make_sequences(row)[0]
    seq_s = scaler.transform(seq.reshape(-1, F)).reshape(1, SEQ_LEN, F)
    proba = model.predict_proba(seq_s)[0]
    pred  = int(np.argmax(proba))
    label = "Pass" if pred == 1 else "Fail"
    total = (assignment + quiz + mid) / 3
    if total >= 75 and study_hours >= 6: level = "High Performance"
    elif total >= 50 and study_hours >= 3: level = "Medium Performance"
    else: level = "Low Performance"
    return {"result": pred, "label": label,
            "confidence": round(proba[pred]*100,1),
            "pass_prob": round(proba[1]*100,1),
            "fail_prob": round(proba[0]*100,1),
            "performance_level": level}

if __name__ == "__main__":
    print(f"\n🧠 RNN Student Evaluator  [{cfg['best_name']}]\n")
    try:
        attendance  = float(input("  Attendance    (0–100): "))
        assignment  = float(input("  Assignment marks     : "))
        quiz        = float(input("  Quiz marks           : "))
        mid         = float(input("  Mid-term marks       : "))
        study_hours = float(input("  Daily study hours    : "))
    except ValueError:
        print("❌ Please enter numeric values."); exit(1)

    r = evaluate_student(attendance, assignment, quiz, mid, study_hours)
    sym = "✅" if r['label']=="Pass" else "❌"
    print(f"\n  {sym}  Result     : {r['label']}")
    print(f"      Confidence : {r['confidence']}%")
    print(f"      Pass prob  : {r['pass_prob']}%  |  Fail prob: {r['fail_prob']}%")
    print(f"      Level      : {r['performance_level']}")
