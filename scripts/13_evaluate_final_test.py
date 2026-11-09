from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from execution_ml.final_model_spec import (
    FINAL_FEATURES,
    FINAL_FEATURE_SET_NAME,
    HORIZONS,
    build_final_model,
)


DATA_ROOT = Path(
    "data/processed"
)

DETAIL_OUTPUT_PATH = Path(
    "outputs/final_test_metrics.csv"
)

SECURITY_OUTPUT_PATH = Path(
    "outputs/final_test_security_summary.csv"
)

HORIZON_OUTPUT_PATH = Path(
    "outputs/final_test_horizon_summary.csv"
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
            "No predictive Parquet files found."
        )

    rows = []

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

        missing_features = [
            feature
            for feature in FINAL_FEATURES
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"{security}: missing final "
                f"features {missing_features}"
            )

        # Final development sample:
        # after model specification has been frozen,
        # train and validation may be combined.
        development = df.loc[
            df["split"].isin(
                [
                    "train",
                    "validation",
                ]
            )
        ]

        test = df.loc[
            df["split"] == "test"
        ]

        if development.empty:
            raise ValueError(
                f"{security}: empty "
                "development sample."
            )

        if test.empty:
            raise ValueError(
                f"{security}: empty test sample."
            )

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

                required_columns = (
                    FINAL_FEATURES
                    + [
                        current_cost_col,
                        target_col,
                    ]
                )

                missing_columns = [
                    column
                    for column in required_columns
                    if column not in df.columns
                ]

                if missing_columns:
                    raise ValueError(
                        f"{security}: missing "
                        f"columns {missing_columns}"
                    )

                # Use the same complete-case test
                # observations for persistence
                # and the final model.
                development_common = (
                    development[
                        required_columns
                    ]
                    .dropna()
                    .copy()
                )

                test_common = (
                    test[
                        required_columns
                    ]
                    .dropna()
                    .copy()
                )

                if development_common.empty:
                    raise ValueError(
                        f"{security} "
                        f"{horizon}s "
                        f"{side}: no usable "
                        "development rows."
                    )

                if test_common.empty:
                    raise ValueError(
                        f"{security} "
                        f"{horizon}s "
                        f"{side}: no usable "
                        "test rows."
                    )

                X_development = (
                    development_common[
                        FINAL_FEATURES
                    ]
                )

                residual_target = (
                    development_common[
                        target_col
                    ]
                    - development_common[
                        current_cost_col
                    ]
                )

                X_test = (
                    test_common[
                        FINAL_FEATURES
                    ]
                )

                y_test = (
                    test_common[
                        target_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                current_test_cost = (
                    test_common[
                        current_cost_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                # F0 persistence benchmark.
                persistence_prediction = (
                    current_test_cost
                )

                persistence_metrics = (
                    regression_metrics(
                        y_test,
                        persistence_prediction,
                    )
                )

                # Frozen F5 residual HGB.
                model = build_final_model()

                model.fit(
                    X_development,
                    residual_target,
                )

                residual_prediction = (
                    model.predict(
                        X_test
                    )
                )

                final_prediction = (
                    current_test_cost
                    + residual_prediction
                )

                final_metrics = (
                    regression_metrics(
                        y_test,
                        final_prediction,
                    )
                )

                persistence_mae = (
                    persistence_metrics[
                        "mae_bps"
                    ]
                )

                final_mae = (
                    final_metrics[
                        "mae_bps"
                    ]
                )

                improvement = (
                    (
                        persistence_mae
                        - final_mae
                    )
                    / persistence_mae
                    * 100.0
                )

                base_info = {
                    "security": security,
                    "quantity": quantity,
                    "horizon_seconds": horizon,
                    "side": side,
                    "development_rows": len(
                        development_common
                    ),
                    "test_rows": len(
                        test_common
                    ),
                    "test_coverage": (
                        len(test_common)
                        / len(test)
                    ),
                }

                rows.append(
                    {
                        **base_info,
                        "model": (
                            "persistence"
                        ),
                        "feature_set": (
                            "F0_persistence"
                        ),
                        **persistence_metrics,
                        "mae_improvement_vs_"
                        "persistence_pct": 0.0,
                    }
                )

                rows.append(
                    {
                        **base_info,
                        "model": (
                            "residual_hgb"
                        ),
                        "feature_set": (
                            FINAL_FEATURE_SET_NAME
                        ),
                        **final_metrics,
                        "mae_improvement_vs_"
                        "persistence_pct": (
                            improvement
                        ),
                    }
                )

    results = pd.DataFrame(
        rows
    )

    DETAIL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    final_only = (
        results.loc[
            results["model"]
            == "residual_hgb"
        ]
        .copy()
    )

    persistence_only = (
        results.loc[
            results["model"]
            == "persistence"
        ]
        .copy()
    )

    print(
        "\nFinal untouched test results"
    )

    display_columns = [
        "security",
        "horizon_seconds",
        "side",
        "model",
        "mae_bps",
        "rmse_bps",
        "median_ae_bps",
        "r2",
        "mae_improvement_vs_"
        "persistence_pct",
        "test_rows",
        "test_coverage",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    comparison = pd.DataFrame(
        {
            "model": [
                "persistence",
                "final_residual_hgb",
            ],
            "mean_mae_bps": [
                persistence_only[
                    "mae_bps"
                ].mean(),
                final_only[
                    "mae_bps"
                ].mean(),
            ],
            "mean_rmse_bps": [
                persistence_only[
                    "rmse_bps"
                ].mean(),
                final_only[
                    "rmse_bps"
                ].mean(),
            ],
            "mean_median_ae_bps": [
                persistence_only[
                    "median_ae_bps"
                ].mean(),
                final_only[
                    "median_ae_bps"
                ].mean(),
            ],
            "mean_r2": [
                persistence_only[
                    "r2"
                ].mean(),
                final_only[
                    "r2"
                ].mean(),
            ],
        }
    )

    print(
        "\nOverall test comparison"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    improvements = (
        final_only[
            "mae_improvement_vs_"
            "persistence_pct"
        ]
    )

    wins = (
        improvements > 0
    )

    print(
        "\nFinal model vs persistence"
    )

    print(
        "Groups beating persistence: "
        f"{wins.sum()} / {len(wins)}"
    )

    print(
        "Mean MAE improvement (%): "
        f"{improvements.mean():.6f}"
    )

    print(
        "Median MAE improvement (%): "
        f"{improvements.median():.6f}"
    )

    print(
        "Worst group improvement (%): "
        f"{improvements.min():.6f}"
    )

    print(
        "Best group improvement (%): "
        f"{improvements.max():.6f}"
    )

    security_summary = (
          final_only.groupby(
               "security",
               as_index=False,
          )
          .agg(
               mean_mae_bps=(
                    "mae_bps",
                    "mean",
               ),
          mean_improvement_vs_persistence_pct=(
               "mae_improvement_vs_persistence_pct",
               "mean",
               ),
          wins_vs_persistence=(
               "mae_improvement_vs_persistence_pct",
               lambda x: int(
                    (x > 0).sum()
                    ),
               ),
          worst_group_improvement_pct=(
               "mae_improvement_vs_persistence_pct",
               "min",
               ),
          )
     )

    security_summary.to_csv(
        SECURITY_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSecurity test summary"
    )

    print(
        security_summary.to_string(
            index=False
        )
    )

    horizon_summary = (
          final_only.groupby(
               "horizon_seconds",
               as_index=False,
          )
          .agg(
               mean_mae_bps=(
                "mae_bps",
               "mean",
               ),
          mean_improvement_vs_persistence_pct=(
                 "mae_improvement_vs_persistence_pct",
               "mean",
               ),
          wins_vs_persistence=(
                 "mae_improvement_vs_persistence_pct",
               lambda x: int(
                     (x > 0).sum()
                    ),
               ),
          worst_group_improvement_pct=(
                 "mae_improvement_vs_persistence_pct",
               "min",
               ),
          )
     )

    horizon_summary.to_csv(
        HORIZON_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nHorizon test summary"
    )

    print(
        horizon_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        DETAIL_OUTPUT_PATH
    )

    print(
        SECURITY_OUTPUT_PATH
    )

    print(
        HORIZON_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()