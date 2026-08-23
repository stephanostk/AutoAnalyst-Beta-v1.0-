"""
Step 1: Get a messy demo dataset and understand its schema.
"""

import pandas as pd

def load_and_describe(csv_path: str):
    df = pd.read_csv(csv_path)

    print("=== df.info() ===")
    df.info()

    print("\n=== df.head() ===")
    print(df.head())

    print("\n=== Null counts ===")
    print(df.isnull().sum())

    return df


def schema_summary(df: pd.DataFrame) -> str:
    lines = ["Columns and dtypes:"]
    for col, dtype in df.dtypes.items():
        null_count = df[col].isnull().sum()
        lines.append(f"  - {col} ({dtype}), {null_count} nulls")

    lines.append("\nSample rows:")
    lines.append(df.head(3).to_string(index=False))

    return "\n".join(lines)


if __name__ == "__main__":
    df = load_and_describe("sample_sales.csv")
    print("\n=== Schema summary for the LLM ===")
    print(schema_summary(df))