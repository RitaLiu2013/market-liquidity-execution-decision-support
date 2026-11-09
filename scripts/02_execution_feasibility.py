from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execution_ml.execution_cost import (
    add_execution_metrics,
)
from execution_ml.lobster import load_lobster


DATA_ROOT = Path("data/raw")

OUTPUT_PATH = Path(
    "outputs/execution_feasibility_summary.csv"
)

SECURITIES = [
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
]

ORDER_SIZES = [
    100,
    250,
    500,
    1000,
]

LEVELS = 10


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

    nested = (
        DATA_ROOT
        / ticker
        / f"{kind}.csv"
    )

    if nested.exists():
        return nested

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
            f"for {ticker}."
        )

    raise RuntimeError(
        f"Multiple {kind} files found "
        f"for {ticker}: {matches}"
    )


def book_columns(
    levels: int,
) -> list[str]:
    columns = ["mid"]

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


def summarise_side(
    df: pd.DataFrame,
    ticker: str,
    quantity: int,
    side: str,
) -> dict[str, float | int | str]:
    cost_col = (
        f"{side}_cost_bps_{quantity}"
    )

    full_fill_col = (
        f"{side}_full_fill_{quantity}"
    )

    fill_ratio_col = (
        f"{side}_fill_ratio_{quantity}"
    )

    levels_col = (
        f"{side}_levels_consumed_{quantity}"
    )

    relative_size_col = (
        f"{side}_relative_order_size_{quantity}"
    )

    full_mask = df[
        full_fill_col
    ].astype(bool)

    cost = df.loc[
        full_mask,
        cost_col,
    ].dropna()

    levels = df.loc[
        full_mask,
        levels_col,
    ].dropna()

    relative_size = df[
        relative_size_col
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return {
        "security": ticker,
        "quantity": quantity,
        "side": side,
        "observations": len(df),

        "full_fill_rate": (
            full_mask.mean()
        ),

        "mean_fill_ratio": (
            df[fill_ratio_col].mean()
        ),

        "mean_cost_bps": (
            cost.mean()
        ),

        "median_cost_bps": (
            cost.median()
        ),

        "p90_cost_bps": (
            cost.quantile(0.90)
        ),

        "p95_cost_bps": (
            cost.quantile(0.95)
        ),

        "p99_cost_bps": (
            cost.quantile(0.99)
        ),

        "mean_levels_consumed": (
            levels.mean()
        ),

        "p95_levels_consumed": (
            levels.quantile(0.95)
        ),

        "median_relative_order_size": (
            relative_size.median()
        ),

        "p95_relative_order_size": (
            relative_size.quantile(0.95)
        ),
    }


def main() -> None:
    results = []

    for ticker in SECURITIES:
        print(
            f"\nProcessing {ticker}"
        )

        message_path = resolve_file(
            ticker,
            "message",
        )

        orderbook_path = resolve_file(
            ticker,
            "orderbook",
        )

        print(
            f"Message: {message_path}"
        )

        print(
            f"Orderbook: {orderbook_path}"
        )

        raw = load_lobster(
            message_path=message_path,
            orderbook_path=orderbook_path,
            levels=LEVELS,
        )

        raw["mid"] = (
            raw["ask_price_1"]
            + raw["bid_price_1"]
        ) / 2.0

        book = raw[
            book_columns(
                LEVELS
            )
        ].copy()

        del raw

        for quantity in ORDER_SIZES:
            print(
                f"  quantity={quantity}"
            )

            evaluated = (
                add_execution_metrics(
                    book,
                    quantity=quantity,
                    levels=LEVELS,
                )
            )

            results.append(
                summarise_side(
                    evaluated,
                    ticker=ticker,
                    quantity=quantity,
                    side="buy",
                )
            )

            results.append(
                summarise_side(
                    evaluated,
                    ticker=ticker,
                    quantity=quantity,
                    side="sell",
                )
            )

            del evaluated

    summary = pd.DataFrame(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "security",
        "quantity",
        "side",
        "full_fill_rate",
        "median_cost_bps",
        "p95_cost_bps",
        "mean_levels_consumed",
        "median_relative_order_size",
    ]

    print(
        "\nExecution feasibility summary"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()