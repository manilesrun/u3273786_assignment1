1. Setup
   Prerequisites

   - Python 3.9+
   - Java + PySpark to run the Spark models

   # Install dependencies

   pip install -r requirements.txt

2. How to Run

Reproduce with DVC

This runs the full pipeline defined in dvc.yaml:

dvc init

dvc repro

3. What Results to Expect

Processed & feature data

data/processed.csv (cleaned)

data/features.csv (model-ready features: e.g., cuisine*diversity, cost_bin/encoded, subzone_freq, multi-label type*\* dummies)

Models & metrics

Regression models: models/linreg.pkl, models/gdreg.pkl

Classification models: models/logreg.pkl, models/svm.pkl, models/rf.pkl, models/nn.pkl

Regression metrics: artifacts/regression_metrics.json (MSE, R²)

Classification metrics: artifacts/classification_metrics.json (accuracy, precision, recall, F1)

Summary: artifacts/summary.md

Typical outcomes (example)

Regression (sklearn): MSE ≈ 0.0887, R² ≈ 0.436

Classification: accuracy ≈ 0.85–0.86

Random Forest: best overall balance (e.g., Acc ≈ 0.857)

Logistic Regression: highest Class-1 recall (e.g., 0.83) if you care about catching positives

PySpark (optional):

RF Accuracy ≈ 0.866 (very close to sklearn)

Linear Regression MSE ≈ 0.0976 (slightly worse than sklearn; tune/scaling can narrow the gap)

Visualisations (Part A)

Distributions (cost, ratings, types)

Top suburbs by restaurant count

Cost vs votes relationship (weak positive trend, high variance)

Geospatial: cuisine density per suburb from sydney.geojson

Interactive: one key insight re-plotted with Plotly/Bokeh and a short explanation
