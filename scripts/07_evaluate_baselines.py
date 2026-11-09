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


DATA_ROOT = Path(
    "data/processed"
)

OUTPUT_PATH = Path(
    "outputs/baseline_validation_metrics.csv"
)

HORIZONS = (
    1.0,
    5.0,
    10.0,
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
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    medae = median_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae_bps": mae,
        "rmse_bps": rmse,
        "median_ae_bps": medae,
        "r2": r2,
    }


def main() -> None:
    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No predictive Parquet files found. "
            "Run "
            "scripts/06_build_predictive_dataset.py "
            "first."
        )

    rows = []

    for path in paths:
        df = pd.read_parquet(
            path
        )

        required_columns = {
            "security",
            "calibrated_quantity",
            "split",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{path.name}: missing columns "
                f"{sorted(missing)}"
            )

        securities = (
            df["security"]
            .dropna()
            .unique()
        )

        quantities = (
            df["calibrated_quantity"]
            .dropna()
            .unique()
        )

        if len(securities) != 1:
            raise ValueError(
                f"{path.name}: expected one "
                "security."
            )

        if len(quantities) != 1:
            raise ValueError(
                f"{path.name}: expected one "
                "calibrated quantity."
            )

        security = str(
            securities[0]
        )

        quantity = int(
            quantities[0]
        )

        train = df.loc[
            df["split"]
            == "train"
        ]

        validation = df.loc[
            df["split"]
            == "validation"
        ]

        print(
            f"\nProcessing {security}"
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

                required_model_columns = {
                    target_col,
                    current_cost_col,
                }

                missing_model_columns = (
                    required_model_columns
                    - set(df.columns)
                )

                if missing_model_columns:
                    raise ValueError(
                        f"{security}: missing "
                        f"model columns "
                        f"{sorted(missing_model_columns)}"
                    )

                train_target = (
                    train[target_col]
                    .dropna()
                )

                if train_target.empty:
                    raise ValueError(
                        f"{security} "
                        f"{horizon}s "
                        f"{side}: no valid "
                        "training target."
                    )

                train_mean = float(
                    train_target.mean()
                )

                evaluation = (
                    validation[
                        [
                            target_col,
                            current_cost_col,
                        ]
                    ]
                    .dropna()
                )

                if evaluation.empty:
                    raise ValueError(
                        f"{security} "
                        f"{horizon}s "
                        f"{side}: no valid "
                        "validation rows."
                    )

                y_true = (
                    evaluation[
                        target_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                persistence_pred = (
                    evaluation[
                        current_cost_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                mean_pred = np.full(
                    len(evaluation),
                    train_mean,
                    dtype=float,
                )

                coverage = (
                    len(evaluation)
                    / len(validation)
                    if len(validation) > 0
                    else np.nan
                )

                base_info = {
                    "security": security,
                    "quantity": quantity,
                    "horizon_seconds": horizon,
                    "side": side,
                    "split": "validation",
                    "evaluation_rows": (
                        len(evaluation)
                    ),
                    "evaluation_coverage": (
                        coverage
                    ),
                    "target_mean_bps": float(
                        np.mean(
                            y_true
                        )
                    ),
                    "target_std_bps": float(
                        np.std(
                            y_true
                        )
                    ),
                }

                mean_metrics = (
                    regression_metrics(
                        y_true,
                        mean_pred,
                    )
                )

                rows.append(
                    {
                        **base_info,
                        "model": "train_mean",
                        **mean_metrics,
                    }
                )

                persistence_metrics = (
                    regression_metrics(
                        y_true,
                        persistence_pred,
                    )
                )

                rows.append(
                    {
                        **base_info,
                        "model": "persistence",
                        **persistence_metrics,
                    }
                )

    results = pd.DataFrame(
        rows
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
        "model",
        "evaluation_rows",
        "evaluation_coverage",
        "mae_bps",
        "rmse_bps",
        "median_ae_bps",
        "r2",
    ]

    print(
        "\nValidation baseline results"
    )

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    summary = (
        results.groupby(
            "model",
            as_index=False,
        )[
            [
                "mae_bps",
                "rmse_bps",
                "median_ae_bps",
                "r2",
            ]
        ]
        .mean()
    )

    print(
        "\nMean performance across "
        "security / horizon / side groups"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()