import pandas as pd

def preprocess(input_path="data/zomato_df_final_data.csv", output_path="data/processed.csv"):
    df = pd.read_csv(input_path)

    # Drop missing targets
    df = df.dropna(subset=["rating_number", "rating_text"])

    # Drop missing lat/lng
    df = df.dropna(subset=["lng", "lat"])

    # Median imputation for numeric
    df[["cost", "cost_2"]] = df[["cost", "cost_2"]].fillna(df[["cost", "cost_2"]].median())

    # Fill categorical
    df["type"] = df["type"].fillna("Unknown")

    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    preprocess()
