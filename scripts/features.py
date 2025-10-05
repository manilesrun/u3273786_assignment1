
import os, ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

INP  = "data/processed.csv"
OUT  = "data/features.csv"

os.makedirs("data", exist_ok=True)
df = pd.read_csv(INP)

# ---------- 1) cuisine_diversity ----------
def count_cuisines(value):
    if pd.isna(value):
        return 0
    return len(str(value).split(","))

if "cuisine" in df.columns:
    df["cuisine_diversity"] = df["cuisine"].apply(count_cuisines)

# ---------- 2) cost_bin (ordinal) + encode ----------
# Low < Medium < High  ->  0 < 1 < 2
if "cost" in df.columns:
    df["cost_bin"] = pd.qcut(df["cost"], q=3, labels=["Low", "Medium", "High"])
    ord_map = {"Low": 0, "Medium": 1, "High": 2}
    # Use map on string form to be robust to 'category'
    df["cost_bin_encoded"] = df["cost_bin"].astype(str).map(ord_map).astype("Int64")

# ---------- 3) drop unneeded text columns ----------
drop_cols = ["address","link","phone","title","cuisine","color","cuisine_color"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ---------- 4) subzone frequency encode ----------
if "subzone" in df.columns:
    freq = df["subzone"].value_counts().to_dict()
    df["subzone_freq"] = df["subzone"].map(freq).fillna(0).astype(int)
    df = df.drop(columns=["subzone"])

# ---------- 5) cast groupon to int ----------
if "groupon" in df.columns:
    # if bool/NaN mix appears, make it safe
    df["groupon"] = df["groupon"].fillna(False).astype(int)

# ---------- 6) Multi-label one-hot for 'type' ----------
def to_list(x):
    """Turn 'type' into a list. Handles: NaN, 'Cafe', and \"['Cafe','Bakery']\"."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            return [s]
    return [s]

if "type" in df.columns:
    df["type_list"] = df["type"].apply(to_list)
    mlb = MultiLabelBinarizer()
    type_ind = pd.DataFrame(
        mlb.fit_transform(df["type_list"]),
        columns=[f"type_{c}" for c in mlb.classes_],
        index=df.index
    )
    df = pd.concat([df.drop(columns=["type","type_list"]), type_ind], axis=1)

# ---------- 7) clean up & save ----------
# keep rating_text (label for later), everything else should be numeric or that one object col
# remove the helper cost_bin (already encoded it)
if "cost_bin" in df.columns:
    df = df.drop(columns=["cost_bin"])

# replace inf, drop any remaining NaNs (safe for pipeline use)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

df.to_csv(OUT, index=False)
print(f"[features] saved -> {OUT}, rows={len(df)}, cols={df.shape[1]}")
