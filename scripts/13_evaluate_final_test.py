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

OVERALL_OUTPUT_PATH = Path(
    "outputs/final_test_overall_summary.csv"
)

SECURITY_OUTPUT_PATH = Path(
    "outputs/final_test_security_summary.csv"
)

HORIZON_OUTPUT_PATH = Path(
    "outputs/final_test_horizon_summary.csv"
)

COVERAGE_OUTPUT_PATH = Path(
    "outputs/final_test_coverage_summary.csv"
)


FORBIDDEN_FEATURE_PREFIXES = (
    "wait_",
    "future_",
    "requested_time_",
    "state_staleness_",
    "mid_move_",
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
    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate an empty target array."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    if not np.isfinite(
        y_true
    ).all():
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.isfinite(
        y_pred
    ).all():
        raise ValueError(
            "y_pred contains non-finite values."
        )

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


def validate_frozen_specification(
) -> None:
    if not FINAL_FEATURES:
        raise ValueError(
            "FINAL_FEATURES is empty."
        )

    if len(
        FINAL_FEATURES
    ) != len(
        set(
            FINAL_FEATURES
        )
    ):
        raise ValueError(
            "FINAL_FEATURES contains duplicates."
        )

    suspicious_features = [
        feature
        for feature in FINAL_FEATURES
        if feature.startswith(
            FORBIDDEN_FEATURE_PREFIXES
        )
    ]

    if suspicious_features:
        raise ValueError(
            "Potential leakage features found "
            "in FINAL_FEATURES: "
            f"{suspicious_features}"
        )

    if not HORIZONS:
        raise ValueError(
            "HORIZONS is empty."
        )

    if len(
        HORIZONS
    ) != len(
        set(
            HORIZONS
        )
    ):
        raise ValueError(
            "HORIZONS contains duplicates."
        )

    if any(
        float(
            horizon
        ) <= 0
        for horizon in HORIZONS
    ):
        raise ValueError(
            "All horizons must be positive."
        )


def validate_dataset(
    df: pd.DataFrame,
    path: Path,
) -> tuple[str, int]:
    required_metadata = [
        "security",
        "calibrated_quantity",
        "split",
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column
        not in df.columns
    ]

    if missing_metadata:
        raise ValueError(
            f"{path.name}: missing metadata "
            f"columns {missing_metadata}"
        )

    if df.empty:
        raise ValueError(
            f"{path.name}: dataset is empty."
        )

    securities = (
        df["security"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(
        securities
    ) != 1:
        raise ValueError(
            f"{path.name}: expected exactly "
            f"one security, found "
            f"{list(securities)}"
        )

    quantities = (
        df[
            "calibrated_quantity"
        ]
        .dropna()
        .unique()
    )

    if len(
        quantities
    ) != 1:
        raise ValueError(
            f"{path.name}: expected exactly "
            "one calibrated quantity."
        )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    observed_splits = set(
        df["split"]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_splits = (
        required_splits
        - observed_splits
    )

    if missing_splits:
        raise ValueError(
            f"{path.name}: missing splits "
            f"{sorted(missing_splits)}"
        )

    return (
        str(
            securities[0]
        ),
        int(
            quantities[0]
        ),
    )


def assert_no_infinite_values(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    values = (
        frame[
            columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    if np.isinf(
        values
    ).any():
        raise ValueError(
            f"{label}: infinite values found."
        )


def main() -> None:
    validate_frozen_specification()

    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No predictive Parquet files found."
        )

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for path in paths:
        df = pd.read_parquet(
            path
        )

        security, quantity = (
            validate_dataset(
                df,
                path,
            )
        )

        print(
            f"\nProcessing {security}"
        )

        missing_features = [
            feature
            for feature in FINAL_FEATURES
            if feature
            not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"{security}: missing final "
                f"features {missing_features}"
            )

        # Final specification is already frozen.
        #
        # Refit the model using train + validation.
        #
        # Rows labelled "purged" remain excluded.
        development = df.loc[
            df[
                "split"
            ].isin(
                [
                    "train",
                    "validation",
                ]
            )
        ]

        # Test is never used in model fitting.
        test = df.loc[
            df[
                "split"
            ]
            == "test"
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
                float(
                    horizon
                )
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
                    list(
                        FINAL_FEATURES
                    )
                    + [
                        current_cost_col,
                        target_col,
                    ]
                )

                missing_columns = [
                    column
                    for column
                    in required_columns
                    if column
                    not in df.columns
                ]

                if missing_columns:
                    raise ValueError(
                        f"{security}: missing "
                        f"columns "
                        f"{missing_columns}"
                    )

                # Use exactly the same complete-case
                # observations for persistence and HGB.
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

                assert_no_infinite_values(
                    development_common,
                    required_columns,
                    (
                        f"{security} "
                        f"{horizon}s "
                        f"{side} "
                        "development"
                    ),
                )

                assert_no_infinite_values(
                    test_common,
                    required_columns,
                    (
                        f"{security} "
                        f"{horizon}s "
                        f"{side} test"
                    ),
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

                # F0 persistence:
                #
                # Future cost = current cost.
                persistence_prediction = (
                    current_test_cost
                )

                persistence_metrics = (
                    regression_metrics(
                        y_test,
                        persistence_prediction,
                    )
                )

                # Frozen F5 residual HGB:
                #
                # Future cost
                # =
                # current cost
                # +
                # predicted residual.
                model = (
                    build_final_model()
                )

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

                persistence_mae = float(
                    persistence_metrics[
                        "mae_bps"
                    ]
                )

                final_mae = float(
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
                    "horizon_seconds": float(
                        horizon
                    ),
                    "side": side,
                    "development_rows": len(
                        development_common
                    ),
                    "development_coverage": (
                        len(
                            development_common
                        )
                        / len(
                            development
                        )
                    ),
                    "test_rows": len(
                        test_common
                    ),
                    "test_coverage": (
                        len(
                            test_common
                        )
                        / len(
                            test
                        )
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

    if results.empty:
        raise ValueError(
            "Final test evaluation produced "
            "no result rows."
        )

    key_columns = [
        "security",
        "horizon_seconds",
        "side",
        "model",
    ]

    if results.duplicated(
        key_columns
    ).any():
        raise ValueError(
            "Duplicate final-test result "
            "groups were produced."
        )

    expected_groups = (
        len(
            paths
        )
        * len(
            HORIZONS
        )
        * 2
    )

    persistence_only = (
        results.loc[
            results[
                "model"
            ]
            == "persistence"
        ]
        .copy()
    )

    final_only = (
        results.loc[
            results[
                "model"
            ]
            == "residual_hgb"
        ]
        .copy()
    )

    if len(
        persistence_only
    ) != expected_groups:
        raise ValueError(
            "Unexpected number of persistence "
            "result groups."
        )

    if len(
        final_only
    ) != expected_groups:
        raise ValueError(
            "Unexpected number of final-model "
            "result groups."
        )

    DETAIL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nFinal test results "
        "(frozen specification)"
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

    # These are equal-weight averages across
    # security × horizon × side groups.
    persistence_mean_mae = float(
        persistence_only[
            "mae_bps"
        ].mean()
    )

    final_mean_mae = float(
        final_only[
            "mae_bps"
        ].mean()
    )

    aggregate_mae_improvement = (
        (
            persistence_mean_mae
            - final_mean_mae
        )
        / persistence_mean_mae
        * 100.0
    )

    comparison = pd.DataFrame(
        {
            "model": [
                "persistence",
                "final_residual_hgb",
            ],
            "group_weighted_mean_mae_bps": [
                persistence_mean_mae,
                final_mean_mae,
            ],
            "group_weighted_mean_rmse_bps": [
                persistence_only[
                    "rmse_bps"
                ].mean(),
                final_only[
                    "rmse_bps"
                ].mean(),
            ],
            "group_weighted_mean_"
            "median_ae_bps": [
                persistence_only[
                    "median_ae_bps"
                ].mean(),
                final_only[
                    "median_ae_bps"
                ].mean(),
            ],
            "group_weighted_mean_r2": [
                persistence_only[
                    "r2"
                ].mean(),
                final_only[
                    "r2"
                ].mean(),
            ],
        }
    )

    comparison.to_csv(
        OVERALL_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nOverall test comparison "
        "(equal weight per "
        "security/horizon/side group)"
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
        f"{int(wins.sum())} / "
        f"{len(wins)}"
    )

    print(
        "Mean group-level MAE "
        "improvement (%): "
        f"{improvements.mean():.6f}"
    )

    print(
        "Median group-level MAE "
        "improvement (%): "
        f"{improvements.median():.6f}"
    )

    print(
        "Aggregate group-weighted MAE "
        "improvement (%): "
        f"{aggregate_mae_improvement:.6f}"
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
            median_improvement_vs_persistence_pct=(
                "mae_improvement_vs_persistence_pct",
                "median",
            ),
            wins_vs_persistence=(
                "mae_improvement_vs_persistence_pct",
                lambda x: int(
                    (
                        x > 0
                    ).sum()
                ),
            ),
            worst_group_improvement_pct=(
                "mae_improvement_vs_persistence_pct",
                "min",
            ),
            mean_test_coverage=(
                "test_coverage",
                "mean",
            ),
            min_test_coverage=(
                "test_coverage",
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
            median_improvement_vs_persistence_pct=(
                "mae_improvement_vs_persistence_pct",
                "median",
            ),
            wins_vs_persistence=(
                "mae_improvement_vs_persistence_pct",
                lambda x: int(
                    (
                        x > 0
                    ).sum()
                ),
            ),
            worst_group_improvement_pct=(
                "mae_improvement_vs_persistence_pct",
                "min",
            ),
            mean_test_coverage=(
                "test_coverage",
                "mean",
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

    coverage_summary = (
        final_only.groupby(
            "security",
            as_index=False,
        )
        .agg(
            mean_development_coverage=(
                "development_coverage",
                "mean",
            ),
            min_development_coverage=(
                "development_coverage",
                "min",
            ),
            mean_test_coverage=(
                "test_coverage",
                "mean",
            ),
            min_test_coverage=(
                "test_coverage",
                "min",
            ),
        )
    )

    coverage_summary.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nCoverage summary"
    )

    print(
        coverage_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    for output_path in (
        DETAIL_OUTPUT_PATH,
        OVERALL_OUTPUT_PATH,
        SECURITY_OUTPUT_PATH,
        HORIZON_OUTPUT_PATH,
        COVERAGE_OUTPUT_PATH,
    ):
        print(
            output_path
        )


if __name__ == "__main__":
    main()