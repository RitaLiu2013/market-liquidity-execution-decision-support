from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


DATA_ROOT = Path(
    "data/processed"
)

DETAIL_OUTPUT_PATH = Path(
    "outputs/feature_ablation_validation_metrics.csv"
)

SUMMARY_OUTPUT_PATH = Path(
    "outputs/feature_ablation_summary.csv"
)

SECURITY_OUTPUT_PATH = Path(
    "outputs/feature_ablation_security_summary.csv"
)

HORIZONS = (
    1.0,
    5.0,
    10.0,
)


FEATURE_SETS = {
    "F1_spread": [
        "spread_bps",
    ],
    "F2_spread_depth": [
        "spread_bps",
        "depth_10_total",
    ],
    "F3_basic_book": [
        "spread_bps",
        "depth_10_total",
        "top_imbalance",
    ],
    "F4_richer_book": [
        "spread_bps",
        "depth_10_total",
        "top_imbalance",
        "microprice_minus_mid_bps",
        "depth_5_total",
        "depth_imbalance_5",
        "depth_imbalance_10",
    ],
    "F5_flow": [
        "spread_bps",
        "depth_10_total",
        "top_imbalance",
        "microprice_minus_mid_bps",
        "depth_5_total",
        "depth_imbalance_5",
        "depth_imbalance_10",
        "execution_volume_100",
        "trade_imbalance_100",
        "signed_book_flow",
        "book_pressure_100",
    ],
    "F6_time": [
        "spread_bps",
        "depth_10_total",
        "top_imbalance",
        "microprice_minus_mid_bps",
        "depth_5_total",
        "depth_imbalance_5",
        "depth_imbalance_10",
        "execution_volume_100",
        "trade_imbalance_100",
        "signed_book_flow",
        "book_pressure_100",
        "time_sin",
        "time_cos",
    ],
}


FEATURE_SET_ORDER = [
    "F0_persistence",
    *FEATURE_SETS.keys(),
]


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
        "mae_bps": (
            mean_absolute_error(
                y_true,
                y_pred,
            )
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


def build_hgb() -> (
    HistGradientBoostingRegressor
):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )


def main() -> None:
    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No predictive Parquet files "
            "found."
        )

    all_features = (
        FEATURE_SETS["F6_time"]
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
            for feature in all_features
            if feature
            not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"{security}: missing "
                f"features "
                f"{missing_features}"
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

                required_columns = (
                    all_features
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

                # Important:
                # Use the complete F6 information
                # set to define common rows.
                #
                # Every feature set is therefore
                # evaluated on exactly the same
                # train and validation observations.
                train_common = (
                    train[
                        required_columns
                    ]
                    .dropna()
                    .copy()
                )

                validation_common = (
                    validation[
                        required_columns
                    ]
                    .dropna()
                    .copy()
                )

                if train_common.empty:
                    raise ValueError(
                        f"{security} "
                        f"{horizon}s "
                        f"{side}: no common "
                        "training rows."
                    )

                if validation_common.empty:
                    raise ValueError(
                        f"{security} "
                        f"{horizon}s "
                        f"{side}: no common "
                        "validation rows."
                    )

                y_validation = (
                    validation_common[
                        target_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                current_validation = (
                    validation_common[
                        current_cost_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                persistence_metrics = (
                    regression_metrics(
                        y_validation,
                        current_validation,
                    )
                )

                persistence_mae = (
                    persistence_metrics[
                        "mae_bps"
                    ]
                )

                common_coverage = (
                    len(validation_common)
                    / len(validation)
                    if len(validation) > 0
                    else np.nan
                )

                rows.append(
                    {
                        "security": security,
                        "quantity": quantity,
                        "horizon_seconds": (
                            horizon
                        ),
                        "side": side,
                        "feature_set": (
                            "F0_persistence"
                        ),
                        "model": (
                            "persistence"
                        ),
                        "n_features": 0,
                        "features": (
                            "current_cost_anchor"
                        ),
                        "train_rows": len(
                            train_common
                        ),
                        "validation_rows": len(
                            validation_common
                        ),
                        "validation_coverage": (
                            common_coverage
                        ),
                        **persistence_metrics,
                        "mae_improvement_vs_"
                        "persistence_pct": 0.0,
                    }
                )

                train_residual = (
                    train_common[
                        target_col
                    ]
                    - train_common[
                        current_cost_col
                    ]
                )

                for (
                    feature_set,
                    feature_columns,
                ) in FEATURE_SETS.items():
                    X_train = (
                        train_common[
                            feature_columns
                        ]
                    )

                    X_validation = (
                        validation_common[
                            feature_columns
                        ]
                    )

                    model = build_hgb()

                    model.fit(
                        X_train,
                        train_residual,
                    )

                    residual_prediction = (
                        model.predict(
                            X_validation
                        )
                    )

                    prediction = (
                        current_validation
                        + residual_prediction
                    )

                    metrics = (
                        regression_metrics(
                            y_validation,
                            prediction,
                        )
                    )

                    improvement = (
                        (
                            persistence_mae
                            - metrics[
                                "mae_bps"
                            ]
                        )
                        / persistence_mae
                        * 100.0
                    )

                    rows.append(
                        {
                            "security": (
                                security
                            ),
                            "quantity": quantity,
                            "horizon_seconds": (
                                horizon
                            ),
                            "side": side,
                            "feature_set": (
                                feature_set
                            ),
                            "model": (
                                "residual_hgb"
                            ),
                            "n_features": len(
                                feature_columns
                            ),
                            "features": (
                                "|".join(
                                    feature_columns
                                )
                            ),
                            "train_rows": len(
                                train_common
                            ),
                            "validation_rows": len(
                                validation_common
                            ),
                            "validation_coverage": (
                                common_coverage
                            ),
                            **metrics,
                            "mae_improvement_vs_"
                            "persistence_pct": (
                                improvement
                            ),
                        }
                    )

    results = pd.DataFrame(
        rows
    )

    order_map = {
        feature_set: index
        for index, feature_set
        in enumerate(
            FEATURE_SET_ORDER
        )
    }

    results[
        "feature_set_order"
    ] = (
        results[
            "feature_set"
        ]
        .map(
            order_map
        )
    )

    group_columns = [
        "security",
        "horizon_seconds",
        "side",
    ]

    results = (
        results.sort_values(
            group_columns
            + [
                "feature_set_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "previous_mae_bps"
    ] = (
        results.groupby(
            group_columns
        )[
            "mae_bps"
        ]
        .shift(1)
    )

    results[
        "incremental_mae_improvement_"
        "vs_previous_pct"
    ] = (
        (
            results[
                "previous_mae_bps"
            ]
            - results[
                "mae_bps"
            ]
        )
        / results[
            "previous_mae_bps"
        ]
        * 100.0
    )

    DETAIL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    summary_rows = []

    for feature_set in FEATURE_SET_ORDER:
        subset = results.loc[
            results[
                "feature_set"
            ]
            == feature_set
        ]

        improvement = (
            subset[
                "mae_improvement_vs_"
                "persistence_pct"
            ]
        )

        incremental = (
            subset[
                "incremental_mae_"
                "improvement_vs_previous_pct"
            ]
            .dropna()
        )

        summary_rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "groups": len(
                    subset
                ),
                "mean_mae_bps": (
                    subset[
                        "mae_bps"
                    ].mean()
                ),
                "median_mae_bps": (
                    subset[
                        "mae_bps"
                    ].median()
                ),
                "mean_rmse_bps": (
                    subset[
                        "rmse_bps"
                    ].mean()
                ),
                "mean_median_ae_bps": (
                    subset[
                        "median_ae_bps"
                    ].mean()
                ),
                "mean_r2": (
                    subset[
                        "r2"
                    ].mean()
                ),
                "mean_improvement_vs_"
                "persistence_pct": (
                    improvement.mean()
                ),
                "median_improvement_vs_"
                "persistence_pct": (
                    improvement.median()
                ),
                "wins_vs_persistence": int(
                    (
                        improvement > 0
                    ).sum()
                ),
                "worst_group_"
                "improvement_pct": (
                    improvement.min()
                ),
                "best_group_"
                "improvement_pct": (
                    improvement.max()
                ),
                "mean_incremental_"
                "improvement_vs_"
                "previous_pct": (
                    incremental.mean()
                    if not incremental.empty
                    else np.nan
                ),
                "incremental_wins_"
                "vs_previous": (
                    int(
                        (
                            incremental > 0
                        ).sum()
                    )
                    if not incremental.empty
                    else 0
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    security_rows = []

    for (
        feature_set,
        security,
    ), subset in results.groupby(
        [
            "feature_set",
            "security",
        ],
        sort=False,
    ):
        improvement = (
            subset[
                "mae_improvement_vs_"
                "persistence_pct"
            ]
        )

        security_rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "security": security,
                "mean_mae_bps": (
                    subset[
                        "mae_bps"
                    ].mean()
                ),
                "mean_improvement_vs_"
                "persistence_pct": (
                    improvement.mean()
                ),
                "wins_vs_persistence": int(
                    (
                        improvement > 0
                    ).sum()
                ),
                "groups": len(
                    subset
                ),
                "worst_group_"
                "improvement_pct": (
                    improvement.min()
                ),
            }
        )

    security_summary = (
        pd.DataFrame(
            security_rows
        )
    )

    security_summary.to_csv(
        SECURITY_OUTPUT_PATH,
        index=False,
    )

    best_indices = (
        results.groupby(
            group_columns
        )[
            "mae_bps"
        ]
        .idxmin()
    )

    best_by_group = (
        results.loc[
            best_indices,
            group_columns
            + [
                "feature_set",
                "mae_bps",
                "mae_improvement_vs_"
                "persistence_pct",
            ]
        ]
        .sort_values(
            group_columns
        )
    )

    best_counts = (
        best_by_group[
            "feature_set"
        ]
        .value_counts()
        .reindex(
            FEATURE_SET_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "feature_set"
        )
        .reset_index(
            name="best_group_count"
        )
    )

    print(
        "\nFeature ablation summary"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nBest feature-set counts"
    )

    print(
        best_counts.to_string(
            index=False
        )
    )

    print(
        "\nSecurity stability summary"
    )

    print(
        security_summary[
            [
                "feature_set",
                "security",
                "mean_improvement_vs_"
                "persistence_pct",
                "wins_vs_persistence",
                "groups",
                "worst_group_"
                "improvement_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nBest feature set by "
        "security / horizon / side"
    )

    print(
        best_by_group.to_string(
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
        SUMMARY_OUTPUT_PATH
    )

    print(
        SECURITY_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()