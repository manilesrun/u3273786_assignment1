import os
import pandas as pd

RAW = "data/zomato_df_final_data.csv"   # put your raw CSV here
OUT = "data/processed.csv"

os.makedirs("data", exist_ok=True)

df = pd.read_csv(RAW)

# Drop rows with missing targets (can't impute truth)
df = df.dropna(subset=["rating_number", "rating_text"])

# Drop rows with missing coordinates (can't meaningfully impute location)
df = df.dropna(subset=["lng", "lat"])

# Drop redundant column
if "cost_2" in df.columns:
    df = df.drop(columns=["cost_2"])

# Impute cost; fill type
if "cost" in df.columns:
    df["cost"] = df["cost"].fillna(df["cost"].median())
if "type" in df.columns:
    df["type"] = df["type"].fillna("Unknown")

df.to_csv(OUT, index=False)
print(f"Saved -> {OUT}, rows={len(df)}")
