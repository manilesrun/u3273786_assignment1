# scripts/classification.py
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

INP = "data/features.csv"
os.makedirs("models", exist_ok=True)

# 1) Load features
df_no_outliers = pd.read_csv(INP)

# 2) Simplify rating_text into binary classes
def simplify_rating(text):
    if text in ["Poor", "Average"]:
        return 0   # Class 1
    elif text in ["Good", "Very Good", "Excellent"]:
        return 1   # Class 2
    else:
        return None

df_no_outliers["rating_binary"] = df_no_outliers["rating_text"].apply(simplify_rating)

# 3) Features / target
X_class = df_no_outliers.drop(columns=["rating_number", "rating_text", "rating_binary"])
y_class = df_no_outliers["rating_binary"].astype(int)

# 4) Train/test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 5) Logistic Regression (no scaling, to match notebook)
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_cls, y_train_cls)
y_pred_log = log_reg.predict(X_test_cls)

print("Logistic Regression (Classification) Results")
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_log))
print("Accuracy:", accuracy_score(y_test_cls, y_pred_log))

# 6) SVM (linear kernel, no scaling to match notebook)
svm_clf = SVC(kernel="linear", probability=True, random_state=42)
svm_clf.fit(X_train_cls, y_train_cls)
y_pred_svm = svm_clf.predict(X_test_cls)

print("SVM (Classification) Results")
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_svm))
print("Accuracy:", accuracy_score(y_test_cls, y_pred_svm))

# 7) Random Forest
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train_cls, y_train_cls)
y_pred_rf = rf_clf.predict(X_test_cls)

print("Random Forest (Classification) Results")
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_rf))
print("Accuracy:", accuracy_score(y_test_cls, y_pred_rf))

# 8) Save models (same pattern as regression)
pickle.dump(log_reg, open("models/logreg.pkl","wb"))
pickle.dump(svm_clf, open("models/svm.pkl","wb"))
pickle.dump(rf_clf, open("models/rf.pkl","wb"))
