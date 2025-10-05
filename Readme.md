## Setup

**Requirements**

- Python 3.9+
- (Optional) Java + PySpark for Spark models

**Install**

```bash
pip install -r requirements.txt

```

## Reproduce (DVC)

**Fetch tracked data/models and run the full pipeline**

```bash
dvc pull
dvc repro

```

## Expected Results (baseline)

### Regression (scikit-learn)

- **Linear Regression** — MSE ≈ **0.088**, R² ≈ **0.435**
- **SGD Regression** — MSE ≈ **0.088**, R² ≈ **0.435**

### Classification

- **Logistic Regression** — Acc **0.84**
- **SVM (Linear)** — Acc **0.847**
- **Random Forest** — Acc **0.856**_(best overall balance)_
- **Neural Network** — Acc **0.855**
