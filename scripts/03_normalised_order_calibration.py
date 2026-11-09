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
    "outputs/normalised_order_calibration.csv"
)

SECURITIES = [
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
]

DEPTH_FRACTIONS = [
    0.10,
    0.25,
    0.50,
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
        f"Multiple {kind} files found for {ticker}: "
        f"{matches}"
    )


def calculate_visible_depth(
    df: pd.DataFrame,
    side: str,
) -> pd.Series:
    size_cols = [
        f"{side}_size_{level}"
        for level in range(1, LEVELS + 1)
    ]

    return df[
        size_cols
    ].sum(axis=1)


def summarise(
    evaluated: pd.DataFrame,
    ticker: str,
    quantity: int,
    depth_fraction: float,
    side: str,
    reference_depth: float,
) -> dict[str, float | int | str]:

    full_fill_col = (
        f"{side}_full_fill_{quantity}"
    )

    cost_col = (
        f"{side}_cost_bps_{quantity}"
    )

    levels_col = (
        f"{side}_levels_consumed_{quantity}"
    )

    relative_col = (
        f"{side}_relative_order_size_{quantity}"
    )

    full_mask = evaluated[
        full_fill_col
    ].astype(bool)

    cost = evaluated.loc[
        full_mask,
        cost_col,
    ].dropna()

    levels = evaluated.loc[
        full_mask,
        levels_col,
    ].dropna()

    relative = evaluated[
        relative_col
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return {
        "security": ticker,
        "depth_fraction": depth_fraction,
        "calibrated_quantity": quantity,
        "side": side,
        "reference_depth": reference_depth,
        "full_fill_rate": full_mask.mean(),
        "median_cost_bps": cost.median(),
        "mean_cost_bps": cost.mean(),
        "p95_cost_bps": cost.quantile(0.95),
        "p99_cost_bps": cost.quantile(0.99),
        "mean_levels_consumed": levels.mean(),
        "median_relative_order_size": (
            relative.median()
        ),
        "p95_relative_order_size": (
            relative.quantile(0.95)
        ),
    }


def main() -> None:
    results = []

    for ticker in SECURITIES:
        print(f"\nProcessing {ticker}")

        raw = load_lobster(
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

        raw["mid"] = (
            raw["ask_price_1"]
            + raw["bid_price_1"]
        ) / 2.0

        ask_depth = calculate_visible_depth(
            raw,
            "ask",
        )

        bid_depth = calculate_visible_depth(
            raw,
            "bid",
        )

        median_ask_depth = (
            ask_depth.median()
        )

        median_bid_depth = (
            bid_depth.median()
        )

        # Use the more conservative side so that the
        # same quantity is used for BUY and SELL.
        reference_depth = min(
            median_ask_depth,
            median_bid_depth,
        )

        print(
            f"  Median ask depth: "
            f"{median_ask_depth:.0f}"
        )

        print(
            f"  Median bid depth: "
            f"{median_bid_depth:.0f}"
        )

        print(
            f"  Reference depth: "
            f"{reference_depth:.0f}"
        )

        for fraction in DEPTH_FRACTIONS:

            quantity = max(
                1,
                int(
                    round(
                        fraction
                        * reference_depth
                    )
                ),
            )

            print(
                f"  fraction={fraction:.0%}, "
                f"quantity={quantity}"
            )

            evaluated = add_execution_metrics(
                raw,
                quantity=quantity,
                levels=LEVELS,
            )

            for side in [
                "buy",
                "sell",
            ]:
                results.append(
                    summarise(
                        evaluated=evaluated,
                        ticker=ticker,
                        quantity=quantity,
                        depth_fraction=fraction,
                        side=side,
                        reference_depth=reference_depth,
                    )
                )

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

    display_cols = [
        "security",
        "depth_fraction",
        "calibrated_quantity",
        "side",
        "full_fill_rate",
        "median_cost_bps",
        "p95_cost_bps",
        "mean_levels_consumed",
        "median_relative_order_size",
    ]

    print(
        "\nNormalised order calibration"
    )

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()