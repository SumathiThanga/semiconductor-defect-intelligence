from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca_transformed.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca_group_summary.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    summary = (
        df.groupby("process_anomaly")[
            ["PC1", "PC2", "PC3", "PC4", "PC5"]
        ]
        .agg(["mean", "std"])
    )

    print("=== PCA Group Separation ===")
    print(summary)

    summary.to_csv(OUTPUT_PATH)

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()