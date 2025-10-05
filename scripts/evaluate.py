# scripts/evaluate.py
import os, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

INP = "data/features.csv"
OUT = "artifacts/summary.md"

os.makedirs("artifacts", exist_ok=True)

# 1) Load features
df = pd.read_csv(INP)

# ---------------- Regression data split ----------------
y_reg = df["rating_number"]
X_reg = df.drop(columns=[c for c in ["rating_number", "rating_text"] if c in df.columns]).select_dtypes(include=[np.number])
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# ---------------- Classification data split ----------------
def simplify_rating(text):
    if text in ["Poor", "Average"]: return 0
    if text in ["Good", "Very Good", "Excellent"]: return 1
    return None

df["rating_binary"] = df["rating_text"].apply(simplify_rating)
df = df.dropna(subset=["rating_binary"])
X_cls = df.drop(columns=["rating_number", "rating_text", "rating_binary"]).select_dtypes(include=[np.number])
y_cls = df["rating_binary"].astype(int)
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# helpers
def cls_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return acc, prec, rec, f1

lines = []
lines.append("# Results Summary\n\n")

# 2) Evaluate regression models if present
lines.append("## Regression\n")
lin_path = Path("models/linreg.pkl")
sgd_path = Path("models/gdreg.pkl")

if lin_path.exists():
    lin = pickle.load(open(lin_path, "rb"))
    yhat = lin.predict(Xte_r)
    lines.append(f"- **Linear Regression** — MSE: {mean_squared_error(yte_r, yhat):.6f}, R²: {r2_score(yte_r, yhat):.3f}\n")
else:
    lines.append("- **Linear Regression** — model not found (models/linreg.pkl)\n")

if sgd_path.exists():
    sgd = pickle.load(open(sgd_path, "rb"))
    yhat = sgd.predict(Xte_r)
    lines.append(f"- **SGD Regression** — MSE: {mean_squared_error(yte_r, yhat):.6f}, R²: {r2_score(yte_r, yhat):.3f}\n")
else:
    lines.append("- **SGD Regression** — model not found (models/gdreg.pkl)\n")

lines.append("\n")

# 3) Evaluate classification models if present
lines.append("## Classification\n")

def eval_cls_model(name, path):
    if not Path(path).exists():
        lines.append(f"### {name}\n- model not found ({path})\n\n")
        return
    clf = pickle.load(open(path, "rb"))
    ypred = clf.predict(Xte_c)
    acc, prec, rec, f1 = cls_metrics(yte_c, ypred)
    cm = confusion_matrix(yte_c, ypred)
    lines.append(f"### {name}\n")
    lines.append(f"- Accuracy: {acc:.4f}\n- Precision: {prec:.4f}\n- Recall: {rec:.4f}\n- F1: {f1:.4f}\n")
    lines.append(f"- Confusion Matrix:\n  - TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n\n")

eval_cls_model("Logistic Regression", "models/logreg.pkl")
eval_cls_model("SVM (Linear)",        "models/svm.pkl")
eval_cls_model("Random Forest",        "models/rf.pkl")
# eval_cls_model("Neural Net (MLP)",     "models/nn.pkl")  # if you saved it

# 4) Write summary
with open(OUT, "w") as f:
    f.writelines(lines)

print(f"[evaluate] wrote -> {OUT}")
print("".join(lines))
