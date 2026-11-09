from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(
    "data/processed"
)

BASELINE_PATH = Path(
    "outputs/baseline_validation_metrics.csv"
)

OUTPUT_PATH = Path(
    "outputs/ridge_validation_metrics.csv"
)

HORIZONS = (
    1.0,
    5.0,
    10.0,
)

ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
)


def horizon_suffix(
    horizon: float,
) -> str:
    return (
        str(float(horizon))
        .replace(".", "p")
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae_bps": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse_bps": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "median_ae_bps": (
            median_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }


def main() -> None:
    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No predictive datasets found."
        )

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    ridge_rows = []

    for path in paths:
        df = pd.read_parquet(
            path
        )

        security = str(
            df["security"].iloc[0]
        )

        quantity = int(
            df[
                "calibrated_quantity"
            ].iloc[0]
        )

        print(
            f"\nProcessing {security}"
        )

        train = df.loc[
            df["split"] == "train"
        ]

        validation = df.loc[
            df["split"]
            == "validation"
        ]

        for horizon in HORIZONS:
            suffix = horizon_suffix(
                horizon
            )

            for side in (
                "buy",
                "sell",
            ):
                target_col = (
                    f"wait_{side}_cost_bps_"
                    f"{suffix}s"
                )

                current_cost_col = (
                    f"{side}_cost_bps_"
                    f"{quantity}"
                )

                feature_columns = [
                    current_cost_col,
                    "spread_bps",
                    "top_imbalance",
                    "depth_10_total",
                ]

                required_columns = (
                    feature_columns
                    + [target_col]
                )

                missing = [
                    column
                    for column
                    in required_columns
                    if column
                    not in df.columns
                ]

                if missing:
                    raise ValueError(
                        f"{security}: missing "
                        f"columns {missing}"
                    )

                train_model = (
                    train[
                        required_columns
                    ]
                    .dropna()
                )

                validation_model = (
                    validation[
                        required_columns
                    ]
                    .dropna()
                )

                X_train = (
                    train_model[
                        feature_columns
                    ]
                )

                y_train = (
                    train_model[
                        target_col
                    ]
                )

                X_validation = (
                    validation_model[
                        feature_columns
                    ]
                )

                y_validation = (
                    validation_model[
                        target_col
                    ]
                )

                best_result = None

                for alpha in ALPHAS:
                    model = Pipeline(
                        [
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                            (
                                "ridge",
                                Ridge(
                                    alpha=alpha
                                ),
                            ),
                        ]
                    )

                    model.fit(
                        X_train,
                        y_train,
                    )

                    prediction = (
                        model.predict(
                            X_validation
                        )
                    )

                    metrics = (
                        regression_metrics(
                            y_validation
                            .to_numpy(),
                            prediction,
                        )
                    )

                    candidate = {
                        "security": security,
                        "quantity": quantity,
                        "horizon_seconds": (
                            horizon
                        ),
                        "side": side,
                        "alpha": alpha,
                        "train_rows": len(
                            train_model
                        ),
                        "validation_rows": len(
                            validation_model
                        ),
                        **metrics,
                    }

                    if (
                        best_result is None
                        or candidate[
                            "mae_bps"
                        ]
                        < best_result[
                            "mae_bps"
                        ]
                    ):
                        best_result = (
                            candidate
                        )

                persistence_row = (
                    baseline.loc[
                        (
                            baseline[
                                "security"
                            ]
                            == security
                        )
                        & (
                            baseline[
                                "horizon_seconds"
                            ]
                            == horizon
                        )
                        & (
                            baseline[
                                "side"
                            ]
                            == side
                        )
                        & (
                            baseline[
                                "model"
                            ]
                            == "persistence"
                        )
                    ]
                )

                if len(
                    persistence_row
                ) != 1:
                    raise ValueError(
                        "Expected exactly one "
                        "persistence baseline row."
                    )

                persistence_mae = float(
                    persistence_row[
                        "mae_bps"
                    ].iloc[0]
                )

                persistence_r2 = float(
                    persistence_row[
                        "r2"
                    ].iloc[0]
                )

                best_result[
                    "persistence_mae_bps"
                ] = persistence_mae

                best_result[
                    "mae_improvement_vs_"
                    "persistence_pct"
                ] = (
                    (
                        persistence_mae
                        - best_result[
                            "mae_bps"
                        ]
                    )
                    / persistence_mae
                    * 100.0
                )

                best_result[
                    "persistence_r2"
                ] = persistence_r2

                best_result[
                    "r2_improvement_vs_"
                    "persistence"
                ] = (
                    best_result["r2"]
                    - persistence_r2
                )

                ridge_rows.append(
                    best_result
                )

    results = pd.DataFrame(
        ridge_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "security",
        "horizon_seconds",
        "side",
        "alpha",
        "mae_bps",
        "persistence_mae_bps",
        "mae_improvement_vs_"
        "persistence_pct",
        "rmse_bps",
        "r2",
        "persistence_r2",
        "r2_improvement_vs_"
        "persistence",
    ]

    print(
        "\nRidge validation results"
    )

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nMean Ridge performance"
    )

    print(
        results[
            [
                "mae_bps",
                "rmse_bps",
                "median_ae_bps",
                "r2",
                "mae_improvement_vs_"
                "persistence_pct",
                "r2_improvement_vs_"
                "persistence",
            ]
        ]
        .mean()
        .to_string()
    )

    print(
        "\nGroups beating persistence "
        "on MAE:"
    )

    wins = (
        results[
            "mae_improvement_vs_"
            "persistence_pct"
        ]
        > 0
    )

    print(
        f"{wins.sum()} / "
        f"{len(results)}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()