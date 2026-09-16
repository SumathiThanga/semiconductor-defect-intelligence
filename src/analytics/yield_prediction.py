from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

PROCESS_FEATURES = [
    "temperature",
    "pressure",
    "rf_power",
    "gas_flow",
    "etch_time",
]

TARGET = "yield_percent"


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly_results.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_yield_model.joblib"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "yield_model_features.joblib"
)

def main():

    # -----------------------------
    # Load data
    # -----------------------------

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    X = df[PROCESS_FEATURES]
    y = df[TARGET]

    # -----------------------------
    # Train / test split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # -----------------------------
    # Random Forest
    # -----------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    #save the model..analysis report shows that RF is better than Gradient Boosting for this dataset, so we will save the RF model for future use
    joblib.dump(model, MODEL_PATH)

    print("\nRandom Forest model saved to:")
    print(MODEL_PATH)

    #save the features used for training the model
    joblib.dump(PROCESS_FEATURES, FEATURES_PATH)

    # -----------------------------
    # Prediction
    # -----------------------------

    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(
        y_test,
        y_pred,
    ) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n=== Random Forest Results ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # -----------------------------
    # Feature importance
    # -----------------------------

    feature_importance = pd.DataFrame(
        {
            "feature": PROCESS_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\n=== Feature Importance ===")
    print(feature_importance)

    # -----------------------------
    # Gradient Boosting
    # -----------------------------

    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    gb_model.fit(X_train, y_train)

    gb_pred = gb_model.predict(X_test)

    gb_mae = mean_absolute_error(y_test, gb_pred)

    gb_rmse = mean_squared_error(
        y_test,
        gb_pred,
    ) ** 0.5

    gb_r2 = r2_score(
        y_test,
        gb_pred,
    )

    print("\n=== Gradient Boosting Results ===")
    print(f"MAE  : {gb_mae:.4f}")
    print(f"RMSE : {gb_rmse:.4f}")
    print(f"R²   : {gb_r2:.4f}")

    # -----------------------------
    # Gradient Boosting importance
    # -----------------------------

    gb_importance = pd.DataFrame(
        {
            "feature": PROCESS_FEATURES,
            "importance": gb_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\n=== Gradient Boosting Feature Importance ===")
    print(gb_importance)

    y_pred = model.predict(X_test)
    # -----------------------------
    # Actual vs Predicted Yield
    # -----------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6,
    )

    plt.xlabel("Actual Yield (%)")
    plt.ylabel("Predicted Yield (%)")
    plt.title("Random Forest: Actual vs Predicted Yield")

    # Perfect prediction reference line
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()