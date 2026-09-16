from pathlib import Path
import pandas as pd

from pca_analysis import (
    run_pca,
    get_explained_variance,
    get_pca_loadings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly_results.csv"
)

VARIANCE_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca_variance.csv"
)

LOADINGS_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca_loadings.csv"
)

PCA_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca_transformed.csv"
)


def main():

    print("Project root:", PROJECT_ROOT)
    print("Dataset:", DATA_PATH)
    print("Dataset exists:", DATA_PATH.exists())

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)

    # Run PCA
    pca, transformed_data, scaler = run_pca(df)

    # Explained variance
    variance_df = get_explained_variance(pca)

    # Loadings
    loadings_df = get_pca_loadings(pca)

    # Add useful identifiers
    pca_result = pd.concat(
        [
            df[
                [
                    "wafer_id",
                    "lot_id",
                    "process_anomaly",
                    "anomaly_prediction",
                    "anomaly_score",
                    "defect_count",
                    "yield_percent",
                ]
            ].reset_index(drop=True),
            transformed_data.reset_index(drop=True),
        ],
        axis=1,
    )

    # Save results
    variance_df.to_csv(
        VARIANCE_OUTPUT,
        index=False,
    )

    loadings_df.to_csv(
        LOADINGS_OUTPUT,
    )

    pca_result.to_csv(
        PCA_OUTPUT,
        index=False,
    )

    print("\n=== Explained Variance ===")
    print(variance_df)

    print("\n=== PCA Loadings ===")
    print(loadings_df)

    print("\nSaved:")
    print(VARIANCE_OUTPUT)
    print(LOADINGS_OUTPUT)
    print(PCA_OUTPUT)


if __name__ == "__main__":
    main()