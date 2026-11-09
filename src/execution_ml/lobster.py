from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MESSAGE_COLUMNS = [
    "time",
    "event_type",
    "order_id",
    "size",
    "event_price",
    "direction",
]


def _orderbook_columns(levels: int) -> list[str]:
    columns: list[str] = []

    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )

    return columns


def _scale_lobster_price(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")

    # LOBSTER represents prices as USD * 10,000.
    # Dummy empty-book prices are extremely large positive/negative values.
    valid = (raw > 0) & (raw < 1_000_000_000)

    return pd.Series(
        np.where(valid, raw / 10_000.0, np.nan),
        index=series.index,
        dtype=float,
    )


def load_lobster(
    message_path: str | Path,
    orderbook_path: str | Path,
    levels: int = 10,
) -> pd.DataFrame:
    message_path = Path(message_path)
    orderbook_path = Path(orderbook_path)

    messages = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
    )

    orderbook_columns = _orderbook_columns(levels)

    orderbook = pd.read_csv(
        orderbook_path,
        header=None,
        names=orderbook_columns,
    )

    if len(messages) != len(orderbook):
        raise ValueError(
            "Message and order-book files must contain the same number of rows."
        )

    messages["time"] = pd.to_numeric(messages["time"], errors="coerce")
    messages["event_type"] = pd.to_numeric(
        messages["event_type"], errors="coerce"
    ).astype("Int64")
    messages["size"] = pd.to_numeric(messages["size"], errors="coerce")
    messages["direction"] = pd.to_numeric(
        messages["direction"], errors="coerce"
    )

    messages["event_price"] = _scale_lobster_price(
        messages["event_price"]
    )

    for level in range(1, levels + 1):
        for side in ("ask", "bid"):
            price_col = f"{side}_price_{level}"
            size_col = f"{side}_size_{level}"

            orderbook[price_col] = _scale_lobster_price(
                orderbook[price_col]
            )

            orderbook[size_col] = pd.to_numeric(
                orderbook[size_col],
                errors="coerce",
            )

    df = pd.concat(
        [
            messages.reset_index(drop=True),
            orderbook.reset_index(drop=True),
        ],
        axis=1,
    )

    df = df.sort_values("time").reset_index(drop=True)

    return df