import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


PROCESS_FEATURES = [
    "temperature",
    "pressure",
    "rf_power",
    "gas_flow",
    "etch_time",
]


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect anomalous wafers using Isolation Forest.

    Parameters
    ----------
    df : pandas.DataFrame
        Unified wafer manufacturing dataset.

    contamination : float
        Expected proportion of anomalies.

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with:
        - anomaly_prediction
        - anomaly_score
    """

    result = df.copy()

    X = result[PROCESS_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result["anomaly_prediction"] = (predictions == -1).astype(int)

    # Higher value = more anomalous
    result["anomaly_score"] = -scores

    return result