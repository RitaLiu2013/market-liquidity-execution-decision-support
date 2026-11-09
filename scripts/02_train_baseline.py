from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from execution_ml.targets import add_wait_cost_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/execution_features.csv",
    )

    parser.add_argument(
        "--quantity",
        type=float,
        default=1000,
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--buffer-bps",
        type=float,
        default=0.05,
    )

    return parser.parse_args()


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": mean_squared_error(
            y_true,
            y_pred,
        ) ** 0.5,
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }


def decision_metrics(
    current_cost: np.ndarray,
    wait_cost_actual: np.ndarray,
    wait_cost_predicted: np.ndarray,
    buffer_bps: float,
) -> dict[str, float]:
    wait = (
        wait_cost_predicted
        < current_cost - buffer_bps
    )

    realised_policy_cost = np.where(
        wait,
        wait_cost_actual,
        current_cost,
    )

    always_now_cost = current_cost

    oracle_cost = np.minimum(
        current_cost,
        wait_cost_actual,
    )

    savings = (
        always_now_cost
        - realised_policy_cost
    )

    oracle_savings = (
        always_now_cost
        - oracle_cost
    )

    return {
        "wait_rate": float(wait.mean()),
        "always_now_cost_bps": float(
            always_now_cost.mean()
        ),
        "policy_cost_bps": float(
            realised_policy_cost.mean()
        ),
        "policy_savings_bps": float(
            savings.mean()
        ),
        "positive_savings_rate": float(
            (savings > 0).mean()
        ),
        "oracle_savings_bps": float(
            oracle_savings.mean()
        ),
    }


def main() -> None:
    args = parse_args()

    quantity = int(args.quantity)

    df = pd.read_csv(args.input)

    df = add_wait_cost_targets(
        df,
        quantity=args.quantity,
        horizon_seconds=args.horizon,
    )

    current_cost_col = (
        f"buy_cost_bps_{quantity}"
    )

    target_col = (
        f"wait_buy_cost_bps_{quantity}"
    )

    features = [
        "spread_bps",
        "top_imbalance",
        "depth_imbalance_5",
        "depth_imbalance_10",
        "depth_5_total",
        "depth_10_total",
        "microprice_minus_mid_bps",
        "ret_1_event",
        "ret_10_event",
        "rv_50_events_bps",
        "rv_200_events_bps",
        "execution_volume_100",
        "trade_imbalance_100",
        "book_pressure_100",
        "time_sin",
        "time_cos",
        current_cost_col,
    ]

    required = (
        features
        + [
            target_col,
            "time",
        ]
    )

    model_df = (
        df[required]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values("time")
        .reset_index(drop=True)
    )

    n = len(model_df)

    if n < 500:
        raise ValueError(
            "Too few usable observations for a meaningful prototype."
        )

    train_boundary_idx = int(n * 0.60)
    valid_boundary_idx = int(n * 0.80)

    train_boundary_time = model_df.loc[
        train_boundary_idx,
        "time",
    ]

    valid_boundary_time = model_df.loc[
        valid_boundary_idx,
        "time",
    ]

    # Purge the prediction horizon from the end of
    # train and validation so labels do not cross
    # into the next data split.
    train = model_df[
        model_df["time"]
        < train_boundary_time - args.horizon
    ]

    valid = model_df[
        (model_df["time"] >= train_boundary_time)
        & (
            model_df["time"]
            < valid_boundary_time - args.horizon
        )
    ]

    test = model_df[
        model_df["time"]
        >= valid_boundary_time
    ]

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(valid):,} | "
        f"Test: {len(test):,}"
    )

    X_train = train[features]
    y_train = train[target_col]

    X_valid = valid[features]
    y_valid = valid[target_col]

    X_test = test[features]
    y_test = test[target_col]

    models = {
        "ridge": Pipeline(
            steps=[
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(alpha=1.0),
                ),
            ]
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            )
        ),
    }

    validation_results = {}

    # Persistence baseline:
    # future waiting cost ≈ current execution cost.
    persistence_pred = valid[
        current_cost_col
    ].to_numpy()

    validation_results["persistence"] = (
        regression_metrics(
            y_valid.to_numpy(),
            persistence_pred,
        )
    )

    fitted_models = {}

    for name, model in models.items():
        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_valid
        )

        fitted_models[name] = model

        validation_results[name] = (
            regression_metrics(
                y_valid.to_numpy(),
                prediction,
            )
        )

    print("\nValidation metrics")

    for name, metrics in validation_results.items():
        print(
            f"{name:25s} "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"R2={metrics['r2']:.4f}"
        )

    candidate_names = list(models.keys())

    best_name = min(
        candidate_names,
        key=lambda name: (
            validation_results[name]["mae"]
        ),
    )

    print(
        f"\nSelected model: {best_name}"
    )

    best_model = models[best_name]

    train_valid = pd.concat(
        [train, valid],
        axis=0,
    ).sort_values("time")

    best_model.fit(
        train_valid[features],
        train_valid[target_col],
    )

    test_pred = best_model.predict(
        X_test
    )

    test_metrics = regression_metrics(
        y_test.to_numpy(),
        test_pred,
    )

    persistence_test = test[
        current_cost_col
    ].to_numpy()

    persistence_metrics = regression_metrics(
        y_test.to_numpy(),
        persistence_test,
    )

    print("\nTest regression performance")

    print(
        "Persistence              "
        f"MAE={persistence_metrics['mae']:.4f} "
        f"RMSE={persistence_metrics['rmse']:.4f} "
        f"R2={persistence_metrics['r2']:.4f}"
    )

    print(
        f"{best_name:25s} "
        f"MAE={test_metrics['mae']:.4f} "
        f"RMSE={test_metrics['rmse']:.4f} "
        f"R2={test_metrics['r2']:.4f}"
    )

    policy = decision_metrics(
        current_cost=test[
            current_cost_col
        ].to_numpy(),
        wait_cost_actual=y_test.to_numpy(),
        wait_cost_predicted=test_pred,
        buffer_bps=args.buffer_bps,
    )

    print("\nDecision performance")

    for key, value in policy.items():
        print(
            f"{key:25s}: {value:.6f}"
        )


if __name__ == "__main__":
    main()