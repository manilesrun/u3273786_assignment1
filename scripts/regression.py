# scripts/regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load processed + engineered features
df = pd.read_csv("data/features.csv")

# Drop non-numeric or target columns
drop_cols = ["rating_text", "address", "link", "phone", "title", "cuisine"]
X = df.drop(columns=drop_cols + ["rating_number"], errors="ignore")
y = df["rating_number"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Regression complete")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save predictions
pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv("data/regression_results.csv", index=False)
