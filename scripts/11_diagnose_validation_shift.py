from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DATA_ROOT = Path(
    "data/processed"
)

OUTPUT_PATH = Path(
    "outputs/train_validation_shift.csv"
)

FEATURES = [
    "spread_bps",
    "top_imbalance",
    "depth_10_total",
]

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


def safe_iqr(
    series: pd.Series,
) -> float:
    clean = series.dropna()

    if clean.empty:
        return np.nan

    return float(
        clean.quantile(0.75)
        - clean.quantile(0.25)
    )


def summarise_feature_shift(
    security: str,
    feature: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, float | str]:
    train_values = (
        train[feature]
        .dropna()
        .astype(float)
    )

    validation_values = (
        validation[feature]
        .dropna()
        .astype(float)
    )

    train_median = float(
        train_values.median()
    )

    validation_median = float(
        validation_values.median()
    )

    train_iqr = safe_iqr(
        train_values
    )

    if (
        np.isfinite(train_iqr)
        and train_iqr > 0
    ):
        median_shift_iqr = (
            (
                validation_median
                - train_median
            )
            / train_iqr
        )
    else:
        median_shift_iqr = np.nan

    train_min = float(
        train_values.min()
    )

    train_max = float(
        train_values.max()
    )

    outside_train_range = (
        (
            validation_values
            < train_min
        )
        | (
            validation_values
            > train_max
        )
    ).mean()

    ks_statistic = float(
        ks_2samp(
            train_values,
            validation_values,
        ).statistic
    )

    return {
        "security": security,
        "diagnostic_type": "feature",
        "name": feature,
        "side": "",
        "horizon_seconds": np.nan,
        "train_median": train_median,
        "validation_median": (
            validation_median
        ),
        "median_shift_in_train_iqr": (
            median_shift_iqr
        ),
        "train_iqr": train_iqr,
        "validation_iqr": safe_iqr(
            validation_values
        ),
        "validation_outside_"
        "train_range_rate": float(
            outside_train_range
        ),
        "ks_statistic": (
            ks_statistic
        ),
    }


def summarise_residual_shift(
    security: str,
    side: str,
    horizon: float,
    quantity: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, float | str]:
    suffix = horizon_suffix(
        horizon
    )

    target_col = (
        f"wait_{side}_cost_bps_"
        f"{suffix}s"
    )

    current_cost_col = (
        f"{side}_cost_bps_"
        f"{quantity}"
    )

    train_residual = (
        train[target_col]
        - train[current_cost_col]
    ).dropna()

    validation_residual = (
        validation[target_col]
        - validation[current_cost_col]
    ).dropna()

    train_median = float(
        train_residual.median()
    )

    validation_median = float(
        validation_residual.median()
    )

    train_iqr = safe_iqr(
        train_residual
    )

    if (
        np.isfinite(train_iqr)
        and train_iqr > 0
    ):
        median_shift_iqr = (
            (
                validation_median
                - train_median
            )
            / train_iqr
        )
    else:
        median_shift_iqr = np.nan

    ks_statistic = float(
        ks_2samp(
            train_residual,
            validation_residual,
        ).statistic
    )

    return {
        "security": security,
        "diagnostic_type": (
            "residual_target"
        ),
        "name": "residual_cost",
        "side": side,
        "horizon_seconds": (
            horizon
        ),
        "train_median": train_median,
        "validation_median": (
            validation_median
        ),
        "median_shift_in_train_iqr": (
            median_shift_iqr
        ),
        "train_iqr": train_iqr,
        "validation_iqr": safe_iqr(
            validation_residual
        ),
        "validation_outside_"
        "train_range_rate": np.nan,
        "ks_statistic": (
            ks_statistic
        ),
    }


def main() -> None:
    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
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
            for feature in FEATURES
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

        for feature in FEATURES:
            rows.append(
                summarise_feature_shift(
                    security=security,
                    feature=feature,
                    train=train,
                    validation=validation,
                )
            )

        for horizon in HORIZONS:
            for side in (
                "buy",
                "sell",
            ):
                rows.append(
                    summarise_residual_shift(
                        security=security,
                        side=side,
                        horizon=horizon,
                        quantity=quantity,
                        train=train,
                        validation=validation,
                    )
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

    feature_shift = (
        results.loc[
            results[
                "diagnostic_type"
            ]
            == "feature"
        ]
        .sort_values(
            [
                "ks_statistic",
            ],
            ascending=False,
        )
    )

    residual_shift = (
        results.loc[
            results[
                "diagnostic_type"
            ]
            == "residual_target"
        ]
        .sort_values(
            [
                "ks_statistic",
            ],
            ascending=False,
        )
    )

    print(
        "\nFeature distribution shift"
    )

    print(
        feature_shift[
            [
                "security",
                "name",
                "train_median",
                "validation_median",
                "median_shift_in_train_iqr",
                "validation_outside_"
                "train_range_rate",
                "ks_statistic",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nResidual-target shift"
    )

    print(
        residual_shift[
            [
                "security",
                "horizon_seconds",
                "side",
                "train_median",
                "validation_median",
                "median_shift_in_train_iqr",
                "ks_statistic",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()