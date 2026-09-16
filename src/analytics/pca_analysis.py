import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PROCESS_FEATURES = [
    "temperature",
    "pressure",
    "rf_power",
    "gas_flow",
    "etch_time",
]


def run_pca(
    df: pd.DataFrame,
    n_components=None,
):
    """
    Perform PCA on semiconductor process parameters.

    Returns:
        pca: fitted PCA model
        transformed_data: PCA-transformed dataframe
        scaler: fitted StandardScaler
    """

    X = df[PROCESS_FEATURES]

    # Standardize because process parameters
    # have different scales and variances.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create readable PCA dataframe
    pca_columns = [
        f"PC{i + 1}"
        for i in range(X_pca.shape[1])
    ]

    transformed_data = pd.DataFrame(
        X_pca,
        columns=pca_columns,
        index=df.index,
    )

    return pca, transformed_data, scaler


def get_explained_variance(pca):
    """
    Return explained variance information.
    """

    result = pd.DataFrame(
        {
            "principal_component": [
                f"PC{i + 1}"
                for i in range(len(pca.explained_variance_ratio_))
            ],
            "explained_variance_ratio": (
                pca.explained_variance_ratio_
            ),
        }
    )

    result["cumulative_variance"] = (
        result["explained_variance_ratio"].cumsum()
    )

    return result


def get_pca_loadings(pca):
    """
    Return PCA feature loadings.
    """

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PROCESS_FEATURES,
        columns=[
            f"PC{i + 1}"
            for i in range(pca.n_components_)
        ],
    )

    return loadings