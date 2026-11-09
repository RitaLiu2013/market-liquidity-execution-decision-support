from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execution_ml.execution_cost import (
    add_execution_metrics,
)
from execution_ml.features import (
    add_microstructure_features,
)
from execution_ml.lobster import load_lobster
from execution_ml.targets import (
    add_future_execution_targets,
)


DATA_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")

SPECIFICATION_PATH = Path(
    "outputs/model_specification.csv"
)

SUMMARY_PATH = Path(
    "outputs/predictive_dataset_summary.csv"
)

LEVELS = 10

HORIZONS = (
    1.0,
    5.0,
    10.0,
)

MAX_HORIZON = max(HORIZONS)


def resolve_file(
    ticker: str,
    kind: str,
) -> Path:
    direct = (
        DATA_ROOT
        / f"{ticker}_{kind}.csv"
    )

    if direct.exists():
        return direct

    matches = list(
        DATA_ROOT.rglob(
            f"{ticker}_*_{kind}_10.csv"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {kind} file found "
            f"for {ticker}"
        )

    raise RuntimeError(
        f"Multiple {kind} files found "
        f"for {ticker}: {matches}"
    )


def assign_split(
    time: pd.Series,
    train_end: float,
    validation_end: float,
) -> pd.Series:
    values = time.to_numpy(
        dtype=float
    )

    split = np.full(
        len(values),
        "purged",
        dtype=object,
    )

    train_mask = (
        values
        < train_end - MAX_HORIZON
    )

    validation_mask = (
        (values >= train_end)
        & (
            values
            < validation_end
            - MAX_HORIZON
        )
    )

    test_mask = (
        values >= validation_end
    )

    split[train_mask] = "train"

    split[
        validation_mask
    ] = "validation"

    split[test_mask] = "test"

    return pd.Series(
        split,
        index=time.index,
        dtype="string",
    )


def orderbook_columns(
    levels: int,
) -> list[str]:
    columns = []

    for level in range(
        1,
        levels + 1,
    ):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )

    return columns


def horizon_suffix(
    horizon: float,
) -> str:
    return (
        str(float(horizon))
        .replace(".", "p")
    )


def build_security_dataset(
    ticker: str,
    quantity: int,
    train_end: float,
    validation_end: float,
) -> pd.DataFrame:
    df = load_lobster(
        message_path=resolve_file(
            ticker,
            "message",
        ),
        orderbook_path=resolve_file(
            ticker,
            "orderbook",
        ),
        levels=LEVELS,
    )

    if not df[
        "time"
    ].is_monotonic_increasing:
        raise ValueError(
            f"{ticker}: time is not "
            "monotonically increasing."
        )

    df = add_microstructure_features(
        df,
        levels=LEVELS,
    )

    df = add_execution_metrics(
        df,
        quantity=quantity,
        levels=LEVELS,
    )

    required_execution_columns = {
        f"buy_vwap_{quantity}",
        f"sell_vwap_{quantity}",
        f"buy_cost_bps_{quantity}",
        f"sell_cost_bps_{quantity}",
    }

    missing_execution_columns = (
        required_execution_columns
        - set(df.columns)
    )

    if missing_execution_columns:
        raise ValueError(
            f"{ticker}: missing execution "
            f"columns: "
            f"{sorted(missing_execution_columns)}"
        )

    df = add_future_execution_targets(
        df,
        quantity=quantity,
        horizons=HORIZONS,
    )

    df["security"] = ticker

    df["calibrated_quantity"] = (
        quantity
    )

    df["order_notional"] = (
        quantity
        * df["mid"]
    )

    df["split"] = assign_split(
        time=df["time"],
        train_end=train_end,
        validation_end=validation_end,
    )

    # The raw Level-1...Level-10 book columns
    # have already been converted into engineered
    # microstructure and execution features.
    #
    # Removing them materially reduces Parquet size
    # and also discourages accidental use of raw,
    # security-scale-dependent prices as ML features.
    raw_book_columns = [
        column
        for column in orderbook_columns(
            LEVELS
        )
        if column in df.columns
    ]

    df = df.drop(
        columns=raw_book_columns
    )

    return df


def main() -> None:
    specification = pd.read_csv(
        SPECIFICATION_PATH
    )

    required_spec_columns = {
        "security",
        "calibrated_quantity",
        "train_end_time",
        "validation_end_time",
    }

    missing_spec_columns = (
        required_spec_columns
        - set(specification.columns)
    )

    if missing_spec_columns:
        raise ValueError(
            "Missing model specification "
            "columns: "
            f"{sorted(missing_spec_columns)}"
        )

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for row in specification.itertuples(
        index=False
    ):
        ticker = str(
            row.security
        )

        quantity = int(
            row.calibrated_quantity
        )

        train_end = float(
            row.train_end_time
        )

        validation_end = float(
            row.validation_end_time
        )

        print(
            f"\nProcessing {ticker}"
        )

        print(
            f"  quantity="
            f"{quantity:,}"
        )

        df = build_security_dataset(
            ticker=ticker,
            quantity=quantity,
            train_end=train_end,
            validation_end=validation_end,
        )

        output_path = (
            PROCESSED_ROOT
            / f"{ticker}_predictive.parquet"
        )

        df.to_parquet(
            output_path,
            index=False,
            compression="zstd",
        )

        split_counts = (
            df["split"]
            .value_counts()
        )

        summary_row = {
            "security": ticker,
            "quantity": quantity,
            "rows": len(df),
            "columns": len(
                df.columns
            ),
            "train_rows": int(
                split_counts.get(
                    "train",
                    0,
                )
            ),
            "validation_rows": int(
                split_counts.get(
                    "validation",
                    0,
                )
            ),
            "test_rows": int(
                split_counts.get(
                    "test",
                    0,
                )
            ),
            "purged_rows": int(
                split_counts.get(
                    "purged",
                    0,
                )
            ),
        }

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

                summary_row[
                    f"{side}_target_"
                    f"{suffix}s_availability"
                ] = (
                    df[target_col]
                    .notna()
                    .mean()
                )

        summary_rows.append(
            summary_row
        )

        print(
            f"  rows="
            f"{len(df):,}"
        )

        print(
            f"  columns="
            f"{len(df.columns):,}"
        )

        print(
            "  splits="
            f"{split_counts.to_dict()}"
        )

        print(
            f"  saved="
            f"{output_path}"
        )

        del df

    summary = pd.DataFrame(
        summary_rows
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "\nPredictive dataset summary"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()