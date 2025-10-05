# scripts/classification.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")

    # Simplify rating_text into binary
    def simplify_rating(text):
        if text in ["Poor", "Average"]:
            return 0
        elif text in ["Good", "Very Good", "Excellent"]:
            return 1
        return None

    df["rating_binary"] = df["rating_text"].apply(simplify_rating)

    X = df.drop(columns=["rating_number", "rating_text", "rating_binary"], errors="ignore")
    y = df["rating_binary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("✅ Classification done")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, "models/classifier.pkl")
    print("📦 Saved classifier model → models/classifier.pkl")
