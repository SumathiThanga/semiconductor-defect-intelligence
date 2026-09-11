import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"


# ---------------------------------------------------------
# Expected configuration
# ---------------------------------------------------------

EXPECTED_LOTS = 100
EXPECTED_WAFERS = 2500
EXPECTED_ANOMALIES = 125


REQUIRED_COLUMNS = {
    "lots.csv": [
        "lot_id",
        "product_id",
        "equipment_id",
        "chamber_id",
    ],
    "wafers.csv": [
        "wafer_id",
        "lot_id",
        "wafer_number",
    ],
    "process_parameters.csv": [
        "wafer_id",
        "temperature",
        "pressure",
        "rf_power",
        "gas_flow",
        "etch_time",
        "process_anomaly",
    ],
    "defects.csv": [
        "wafer_id",
        "defect_type",
        "defect_count",
    ],
    "yield.csv": [
        "wafer_id",
        "yield_percent",
    ],
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def check_file_exists(filename):
    """Check whether a required dataset file exists."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"[FAIL] Missing file: {filename}")
        return False

    print(f"[PASS] File exists: {filename}")
    return True


def load_data(filename):
    """Load a CSV file into a pandas DataFrame."""

    file_path = DATA_DIR / filename

    return pd.read_csv(file_path)


def check_required_columns(df, filename):
    """Check that all expected columns are present."""

    expected = REQUIRED_COLUMNS[filename]

    missing_columns = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing_columns:
        print(
            f"[FAIL] {filename} missing columns: "
            f"{missing_columns}"
        )
        return False

    print(f"[PASS] Required columns: {filename}")
    return True


def check_row_count(df, filename, expected_count):
    """Check expected number of rows."""

    actual_count = len(df)

    if actual_count != expected_count:
        print(
            f"[FAIL] {filename}: "
            f"expected {expected_count}, "
            f"found {actual_count}"
        )
        return False

    print(
        f"[PASS] Row count {filename}: "
        f"{actual_count}"
    )

    return True


def check_missing_values(df, filename):
    """Check for missing values."""

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if not missing.empty:
        print(
            f"[FAIL] Missing values in {filename}:"
        )
        print(missing)
        return False

    print(f"[PASS] No missing values: {filename}")
    return True


def check_duplicate_ids(df, column, filename):
    """Check whether an ID column contains duplicates."""

    duplicates = df[column].duplicated().sum()

    if duplicates > 0:
        print(
            f"[FAIL] {filename}: "
            f"{duplicates} duplicate {column} values"
        )
        return False

    print(
        f"[PASS] Unique {column}: {filename}"
    )

    return True


# ---------------------------------------------------------
# Relationship checks
# ---------------------------------------------------------

def check_wafer_lot_relationship(
    wafers_df,
    lots_df,
):
    """Every wafer must belong to a valid lot."""

    invalid = ~wafers_df["lot_id"].isin(
        lots_df["lot_id"]
    )

    invalid_count = invalid.sum()

    if invalid_count > 0:
        print(
            "[FAIL] Wafer → Lot relationship: "
            f"{invalid_count} invalid wafers"
        )
        return False

    print(
        "[PASS] Wafer → Lot relationship"
    )

    return True


def check_wafer_references(
    wafers_df,
    other_df,
    other_name,
):
    """Check that all wafer IDs exist in the wafer table."""

    invalid = ~other_df["wafer_id"].isin(
        wafers_df["wafer_id"]
    )

    invalid_count = invalid.sum()

    if invalid_count > 0:
        print(
            f"[FAIL] Wafer → {other_name}: "
            f"{invalid_count} invalid references"
        )
        return False

    print(
        f"[PASS] Wafer → {other_name} relationship"
    )

    return True


# ---------------------------------------------------------
# Value validation
# ---------------------------------------------------------

def check_process_values(process_df):
    """Check basic process parameter ranges."""

    checks_passed = True

    if (process_df["temperature"] <= 0).any():
        print(
            "[FAIL] Temperature contains "
            "invalid values"
        )
        checks_passed = False

    if (process_df["pressure"] <= 0).any():
        print(
            "[FAIL] Pressure contains "
            "invalid values"
        )
        checks_passed = False

    if (process_df["rf_power"] <= 0).any():
        print(
            "[FAIL] RF power contains "
            "invalid values"
        )
        checks_passed = False

    if (process_df["gas_flow"] <= 0).any():
        print(
            "[FAIL] Gas flow contains "
            "invalid values"
        )
        checks_passed = False

    if (process_df["etch_time"] <= 0).any():
        print(
            "[FAIL] Etch time contains "
            "invalid values"
        )
        checks_passed = False

    if checks_passed:
        print(
            "[PASS] Process parameter ranges"
        )

    return checks_passed


def check_defect_values(defects_df):
    """Check defect counts."""

    if (defects_df["defect_count"] < 0).any():
        print(
            "[FAIL] Negative defect count found"
        )
        return False

    print("[PASS] Defect counts")
    return True


def check_yield_values(yield_df):
    """Check that yield is between 0 and 100."""

    invalid = (
        (yield_df["yield_percent"] < 0)
        | (yield_df["yield_percent"] > 100)
    )

    if invalid.any():
        print(
            "[FAIL] Yield outside 0-100 range"
        )
        return False

    print("[PASS] Yield range")
    return True


def check_anomaly_count(process_df):
    """Check expected number of injected anomalies."""

    actual = process_df["process_anomaly"].sum()

    if actual != EXPECTED_ANOMALIES:
        print(
            "[FAIL] Anomaly count: "
            f"expected {EXPECTED_ANOMALIES}, "
            f"found {actual}"
        )
        return False

    print(
        f"[PASS] Anomaly count: {actual}"
    )

    return True


# ---------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("SEMICONDUCTOR DATA VALIDATION")
    print("=" * 60)

    all_passed = True

    # -----------------------------------------------------
    # 1. Check files
    # -----------------------------------------------------

    filenames = [
        "lots.csv",
        "wafers.csv",
        "process_parameters.csv",
        "defects.csv",
        "yield.csv",
    ]

    for filename in filenames:

        if not check_file_exists(filename):
            all_passed = False

    if not all_passed:
        print("\nValidation stopped because files are missing.")
        return

    # -----------------------------------------------------
    # 2. Load data
    # -----------------------------------------------------

    lots_df = load_data("lots.csv")
    wafers_df = load_data("wafers.csv")
    process_df = load_data("process_parameters.csv")
    defects_df = load_data("defects.csv")
    yield_df = load_data("yield.csv")

    print("\nData loaded successfully.")

    # -----------------------------------------------------
    # 3. Column validation
    # -----------------------------------------------------

    for filename, df in [
        ("lots.csv", lots_df),
        ("wafers.csv", wafers_df),
        ("process_parameters.csv", process_df),
        ("defects.csv", defects_df),
        ("yield.csv", yield_df),
    ]:

        if not check_required_columns(df, filename):
            all_passed = False

    # -----------------------------------------------------
    # 4. Row count validation
    # -----------------------------------------------------

    if not check_row_count(
        lots_df,
        "lots.csv",
        EXPECTED_LOTS,
    ):
        all_passed = False

    if not check_row_count(
        wafers_df,
        "wafers.csv",
        EXPECTED_WAFERS,
    ):
        all_passed = False

    if not check_row_count(
        process_df,
        "process_parameters.csv",
        EXPECTED_WAFERS,
    ):
        all_passed = False

    if not check_row_count(
        defects_df,
        "defects.csv",
        EXPECTED_WAFERS,
    ):
        all_passed = False

    if not check_row_count(
        yield_df,
        "yield.csv",
        EXPECTED_WAFERS,
    ):
        all_passed = False

    # -----------------------------------------------------
    # 5. Missing value validation
    # -----------------------------------------------------

    for filename, df in [
        ("lots.csv", lots_df),
        ("wafers.csv", wafers_df),
        ("process_parameters.csv", process_df),
        ("defects.csv", defects_df),
        ("yield.csv", yield_df),
    ]:

        if not check_missing_values(df, filename):
            all_passed = False

    # -----------------------------------------------------
    # 6. Duplicate ID validation
    # -----------------------------------------------------

    if not check_duplicate_ids(
        lots_df,
        "lot_id",
        "lots.csv",
    ):
        all_passed = False

    if not check_duplicate_ids(
        wafers_df,
        "wafer_id",
        "wafers.csv",
    ):
        all_passed = False

    # -----------------------------------------------------
    # 7. Relationship validation
    # -----------------------------------------------------

    if not check_wafer_lot_relationship(
        wafers_df,
        lots_df,
    ):
        all_passed = False

    if not check_wafer_references(
        wafers_df,
        process_df,
        "Process",
    ):
        all_passed = False

    if not check_wafer_references(
        wafers_df,
        defects_df,
        "Defects",
    ):
        all_passed = False

    if not check_wafer_references(
        wafers_df,
        yield_df,
        "Yield",
    ):
        all_passed = False

    # -----------------------------------------------------
    # 8. Value validation
    # -----------------------------------------------------

    if not check_process_values(process_df):
        all_passed = False

    if not check_defect_values(defects_df):
        all_passed = False

    if not check_yield_values(yield_df):
        all_passed = False

    # -----------------------------------------------------
    # 9. Anomaly validation
    # -----------------------------------------------------

    if not check_anomaly_count(process_df):
        all_passed = False

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    if all_passed:
        print("VALIDATION PASSED")
        print("All synthetic semiconductor data checks passed.")
    else:
        print("VALIDATION FAILED")
        print("Please review the errors above.")

    print("=" * 60)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()