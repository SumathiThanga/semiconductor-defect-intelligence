import pandas as pd
from pathlib import Path
from anomaly_detection import detect_anomalies


# Project root:
# semiconductor-defect-intelligence/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "wafer_manufacturing.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly_results.csv"
)




def main():
    df = pd.read_csv(DATA_PATH)

    result = detect_anomalies(df)

    print("Dataset shape:", result.shape)

    print("\nInjected anomalies:")
    print(result["process_anomaly"].sum())

    print("\nML detected anomalies:")
    print(result["anomaly_prediction"].sum())

    print("\nDetection counts:")
    print(
        pd.crosstab(
            result["process_anomaly"],
            result["anomaly_prediction"],
            rownames=["Actual"],
            colnames=["ML Prediction"],
        )
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()