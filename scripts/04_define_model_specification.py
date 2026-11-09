from __future__ import annotations

from pathlib import Path

import pandas as pd

from execution_ml.lobster import load_lobster


DATA_ROOT = Path("data/raw")

OUTPUT_PATH = Path(
    "outputs/model_specification.csv"
)

SECURITIES = [
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
]

LEVELS = 10

PRIMARY_DEPTH_FRACTION = 0.25

TRAIN_FRACTION = 0.60
VALID_FRACTION = 0.20


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
            f"No {kind} file found for {ticker}"
        )

    raise RuntimeError(
        f"Multiple {kind} files found "
        f"for {ticker}: {matches}"
    )


def visible_depth(
    df: pd.DataFrame,
    side: str,
) -> pd.Series:
    columns = [
        f"{side}_size_{level}"
        for level in range(
            1,
            LEVELS + 1,
        )
    ]

    return df[
        columns
    ].sum(axis=1)


def chronological_boundaries(
    time: pd.Series,
) -> tuple[float, float]:
    start_time = time.iloc[0]
    end_time = time.iloc[-1]

    duration = (
        end_time - start_time
    )

    train_end = (
        start_time
        + TRAIN_FRACTION
        * duration
    )

    valid_end = (
        start_time
        + (
            TRAIN_FRACTION
            + VALID_FRACTION
        )
        * duration
    )

    return (
        train_end,
        valid_end,
    )


def main() -> None:
    rows = []

    for ticker in SECURITIES:
        print(
            f"\nProcessing {ticker}"
        )

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

        df["mid"] = (
            df["ask_price_1"]
            + df["bid_price_1"]
        ) / 2.0

        (
            train_end,
            valid_end,
        ) = chronological_boundaries(
            df["time"]
        )

        train = df.loc[
            df["time"] < train_end
        ].copy()

        train_ask_depth = (
            visible_depth(
                train,
                "ask",
            )
        )

        train_bid_depth = (
            visible_depth(
                train,
                "bid",
            )
        )

        median_ask_depth = (
            train_ask_depth.median()
        )

        median_bid_depth = (
            train_bid_depth.median()
        )

        reference_depth = min(
            median_ask_depth,
            median_bid_depth,
        )

        calibrated_quantity = max(
            1,
            int(
                round(
                    PRIMARY_DEPTH_FRACTION
                    * reference_depth
                )
            ),
        )

        median_mid = (
            train["mid"].median()
        )

        approximate_notional = (
            calibrated_quantity
            * median_mid
        )

        rows.append(
            {
                "security": ticker,
                "depth_fraction": (
                    PRIMARY_DEPTH_FRACTION
                ),
                "train_end_time": (
                    train_end
                ),
                "validation_end_time": (
                    valid_end
                ),
                "training_observations": (
                    len(train)
                ),
                "median_train_ask_depth": (
                    median_ask_depth
                ),
                "median_train_bid_depth": (
                    median_bid_depth
                ),
                "reference_depth": (
                    reference_depth
                ),
                "calibrated_quantity": (
                    calibrated_quantity
                ),
                "median_train_mid": (
                    median_mid
                ),
                "approximate_order_notional": (
                    approximate_notional
                ),
            }
        )

        print(
            f"  train observations: "
            f"{len(train):,}"
        )

        print(
            f"  reference depth: "
            f"{reference_depth:,.0f}"
        )

        print(
            f"  25% quantity: "
            f"{calibrated_quantity:,}"
        )

        print(
            f"  approximate notional: "
            f"{approximate_notional:,.0f}"
        )

    specification = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    specification.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nModel specification"
    )

    print(
        specification.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()