import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

INP = "data/features.csv"

os.makedirs("models", exist_ok=True)

# 1) Load
df = pd.read_csv(INP)

# 2) Split features/target
y = df["rating_number"]
X = df.drop(columns=[c for c in ["rating_number", "rating_text"] if c in df.columns])
X = X.select_dtypes(include=[np.number]).copy()

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Model A — Linear Regression (OLS)
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin  = r2_score(y_test, y_pred_lin)

# 5) Model B — SGD (gradient descent) with scaling
sgd_reg = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, random_state=42)
).fit(X_train, y_train)
y_pred_sgd = sgd_reg.predict(X_test)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd  = r2_score(y_test, y_pred_sgd)

# 6) Save models only
with open("models/linreg.pkl","wb") as f: pickle.dump(lin_reg, f)
with open("models/gdreg.pkl","wb") as f:   pickle.dump(sgd_reg, f)

# 7) Print metrics
print(f"[regression] OLS -> MSE: {mse_lin:.6f}  R2: {r2_lin:.3f}")
print(f"[regression] SGD -> MSE: {mse_sgd:.6f}  R2: {r2_sgd:.3f}")
