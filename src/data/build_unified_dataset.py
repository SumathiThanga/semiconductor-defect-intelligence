import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"


# ---------------------------------------------------------
# Input files
# ---------------------------------------------------------

LOTS_FILE = DATA_DIR / "lots.csv"
WAFERS_FILE = DATA_DIR / "wafers.csv"
PROCESS_FILE = DATA_DIR / "process_parameters.csv"
DEFECTS_FILE = DATA_DIR / "defects.csv"
YIELD_FILE = DATA_DIR / "yield.csv"


OUTPUT_FILE = DATA_DIR / "wafer_manufacturing.csv"


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading datasets...")

lots_df = pd.read_csv(LOTS_FILE)
wafers_df = pd.read_csv(WAFERS_FILE)
process_df = pd.read_csv(PROCESS_FILE)
defects_df = pd.read_csv(DEFECTS_FILE)
yield_df = pd.read_csv(YIELD_FILE)


print("All datasets loaded successfully.")


# ---------------------------------------------------------
# Display input sizes
# ---------------------------------------------------------

print("\nInput dataset sizes:")

print(f"Lots                : {len(lots_df)}")
print(f"Wafers              : {len(wafers_df)}")
print(f"Process parameters  : {len(process_df)}")
print(f"Defects             : {len(defects_df)}")
print(f"Yield               : {len(yield_df)}")


# ---------------------------------------------------------
# Step 1: Join wafer information with lot information
# ---------------------------------------------------------

print("\nJoining wafer → lot information...")

unified_df = wafers_df.merge(
    lots_df,
    on="lot_id",
    how="left",
    validate="many_to_one",
)


# ---------------------------------------------------------
# Step 2: Add process parameters
# ---------------------------------------------------------

print("Adding process parameters...")

unified_df = unified_df.merge(
    process_df,
    on="wafer_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Step 3: Add defect information
# ---------------------------------------------------------

print("Adding defect information...")

unified_df = unified_df.merge(
    defects_df,
    on="wafer_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Step 4: Add yield information
# ---------------------------------------------------------

print("Adding yield information...")

unified_df = unified_df.merge(
    yield_df,
    on="wafer_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Validate final row count
# ---------------------------------------------------------

print("\nValidating unified dataset...")

if len(unified_df) != len(wafers_df):

    raise ValueError(
        "Row count changed during merge! "
        f"Expected {len(wafers_df)}, "
        f"found {len(unified_df)}."
    )


print(
    f"[PASS] Row count preserved: "
    f"{len(unified_df)} wafers"
)


# ---------------------------------------------------------
# Validate wafer uniqueness
# ---------------------------------------------------------

duplicate_wafers = unified_df["wafer_id"].duplicated().sum()

if duplicate_wafers > 0:

    raise ValueError(
        f"Found {duplicate_wafers} duplicate wafer IDs."
    )


print("[PASS] wafer_id remains unique")


# ---------------------------------------------------------
# Validate missing values
# ---------------------------------------------------------

missing_values = unified_df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]

if not missing_values.empty:

    print("\n[WARNING] Missing values detected:")

    print(missing_values)

else:

    print("[PASS] No missing values")


# ---------------------------------------------------------
# Display final columns
# ---------------------------------------------------------

print("\nUnified dataset columns:")

for column in unified_df.columns:

    print(f"  - {column}")


# ---------------------------------------------------------
# Save unified dataset
# ---------------------------------------------------------

unified_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nUnified dataset created successfully!")

print(f"Output file:")
print(OUTPUT_FILE)


# ---------------------------------------------------------
# Display sample
# ---------------------------------------------------------

print("\nSample records:")

print(
    unified_df.head().to_string(index=False)
)