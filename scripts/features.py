import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/processed.csv")

    # Example feature engineering
    df["cuisine_diversity"] = df["cuisine"].apply(lambda x: len(str(x).split(",")) if pd.notnull(x) else 0)
    df["cost_bin"] = pd.qcut(df["cost"], q=3, labels=["Low", "Medium", "High"])

    df.to_csv("data/features.csv", index=False)
    print("✅ Features created → data/features.csv")