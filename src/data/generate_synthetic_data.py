import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_LOTS = 100
WAFERS_PER_LOT = 25

RANDOM_SEED = 42


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Random generator
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------
# Manufacturing configuration
# ---------------------------------------------------------

PRODUCTS = ["P01", "P02", "P03"]

EQUIPMENT = [
    "ETCH_01",
    "ETCH_02",
    "ETCH_03",
]

CHAMBERS = [
    "C01",
    "C02",
    "C03",
]


DEFECT_TYPES = [
    "Scratch",
    "Edge-Ring",
    "Center",
    "Random",
    "Particle",
]


# ---------------------------------------------------------
# 1. Generate LOT data
# ---------------------------------------------------------

lots = []

for lot_number in range(1, NUM_LOTS + 1):

    lot_id = f"L{lot_number:04d}"

    lots.append(
        {
            "lot_id": lot_id,
            "product_id": rng.choice(PRODUCTS),
            "equipment_id": rng.choice(EQUIPMENT),
            "chamber_id": rng.choice(CHAMBERS),
        }
    )


lots_df = pd.DataFrame(lots)


# ---------------------------------------------------------
# 2. Generate WAFER data
# ---------------------------------------------------------

wafers = []

for _, lot in lots_df.iterrows():

    for wafer_number in range(1, WAFERS_PER_LOT + 1):

        wafer_id = (
            f"{lot['lot_id']}_W{wafer_number:02d}"
        )

        wafers.append(
            {
                "wafer_id": wafer_id,
                "lot_id": lot["lot_id"],
                "wafer_number": wafer_number,
            }
        )


wafers_df = pd.DataFrame(wafers)


# ---------------------------------------------------------
# 3. Generate PROCESS PARAMETERS
# ---------------------------------------------------------

process_data = []

for _, wafer in wafers_df.iterrows():

    # Normal manufacturing operating conditions
    temperature = rng.normal(300, 5)

    pressure = rng.normal(1.20, 0.08)

    rf_power = rng.normal(850, 40)

    gas_flow = rng.normal(50, 3)

    etch_time = rng.normal(60, 4)

    process_data.append(
        {
            "wafer_id": wafer["wafer_id"],
            "temperature": temperature,
            "pressure": pressure,
            "rf_power": rf_power,
            "gas_flow": gas_flow,
            "etch_time": etch_time,
        }
    )


process_df = pd.DataFrame(process_data)


# ---------------------------------------------------------
# 4. Introduce manufacturing anomalies
# ---------------------------------------------------------

# Approximately 5% of wafers will contain abnormal
# process conditions.

num_anomalies = int(len(process_df) * 0.05)

anomaly_indices = rng.choice(
    process_df.index,
    size=num_anomalies,
    replace=False,
)


process_df.loc[
    anomaly_indices,
    "temperature"
] += rng.normal(15, 3, num_anomalies)


process_df.loc[
    anomaly_indices,
    "pressure"
] += rng.normal(0.25, 0.05, num_anomalies)


process_df.loc[
    anomaly_indices,
    "rf_power"
] += rng.normal(100, 20, num_anomalies)


# Mark the injected anomalies.
process_df["process_anomaly"] = 0

process_df.loc[
    anomaly_indices,
    "process_anomaly"
] = 1


# ---------------------------------------------------------
# 5. Generate DEFECT data
# ---------------------------------------------------------

defects = []

for _, row in process_df.iterrows():

    # Base defect probability
    defect_score = 0.05

    # Process conditions influence defects
    if row["temperature"] > 310:
        defect_score += 0.20

    if row["pressure"] > 1.40:
        defect_score += 0.20

    if row["rf_power"] > 900:
        defect_score += 0.20

    # Generate defect count
    defect_count = rng.poisson(
        lam=max(defect_score * 20, 0.1)
    )

    # Select defect type
    defect_type = rng.choice(DEFECT_TYPES)

    defects.append(
        {
            "wafer_id": row["wafer_id"],
            "defect_type": defect_type,
            "defect_count": defect_count,
        }
    )


defects_df = pd.DataFrame(defects)


# ---------------------------------------------------------
# 6. Generate YIELD
# ---------------------------------------------------------

yield_data = []

for _, defect_row in defects_df.iterrows():

    defect_count = defect_row["defect_count"]

    # Start with high yield
    wafer_yield = 99.0

    # Defects reduce yield
    wafer_yield -= defect_count * 0.8

    # Manufacturing noise
    wafer_yield += rng.normal(0, 1)

    # Keep yield within realistic bounds
    wafer_yield = np.clip(
        wafer_yield,
        50,
        100,
    )

    yield_data.append(
        {
            "wafer_id": defect_row["wafer_id"],
            "yield_percent": wafer_yield,
        }
    )


yield_df = pd.DataFrame(yield_data)


# ---------------------------------------------------------
# 7. Save datasets
# ---------------------------------------------------------

lots_path = OUTPUT_DIR / "lots.csv"
wafers_path = OUTPUT_DIR / "wafers.csv"
process_path = OUTPUT_DIR / "process_parameters.csv"
defects_path = OUTPUT_DIR / "defects.csv"
yield_path = OUTPUT_DIR / "yield.csv"


lots_df.to_csv(lots_path, index=False)
wafers_df.to_csv(wafers_path, index=False)
process_df.to_csv(process_path, index=False)
defects_df.to_csv(defects_path, index=False)
yield_df.to_csv(yield_path, index=False)


# ---------------------------------------------------------
# 8. Display summary
# ---------------------------------------------------------

print("\nSynthetic semiconductor dataset created successfully!")

print(f"\nLots   : {len(lots_df)}")
print(f"Wafers : {len(wafers_df)}")
print(f"Anomalous wafers : {process_df['process_anomaly'].sum()}")

print("\nGenerated files:")

print(lots_path)
print(wafers_path)
print(process_path)
print(defects_path)
print(yield_path)