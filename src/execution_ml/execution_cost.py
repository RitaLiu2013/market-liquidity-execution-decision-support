from __future__ import annotations

import numpy as np
import pandas as pd


def _book_arrays(
    df: pd.DataFrame,
    side: str,
    levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    price_cols = [
        f"{side}_price_{level}"
        for level in range(1, levels + 1)
    ]

    size_cols = [
        f"{side}_size_{level}"
        for level in range(1, levels + 1)
    ]

    prices = df[price_cols].to_numpy(dtype=float)
    sizes = df[size_cols].to_numpy(dtype=float)

    return prices, sizes


def _simulate_book_sweep(
    prices: np.ndarray,
    sizes: np.ndarray,
    quantity: float,
) -> dict[str, np.ndarray]:
    if quantity <= 0:
        raise ValueError(
            "quantity must be strictly positive."
        )

    valid = (
        np.isfinite(prices)
        & np.isfinite(sizes)
        & (prices > 0)
        & (sizes > 0)
    )

    safe_prices = np.where(
        valid,
        prices,
        0.0,
    )

    safe_sizes = np.where(
        valid,
        sizes,
        0.0,
    )

    cumulative_size = np.cumsum(
        safe_sizes,
        axis=1,
    )

    cumulative_notional = np.cumsum(
        safe_prices * safe_sizes,
        axis=1,
    )

    visible_depth = cumulative_size[:, -1]

    full_fill = (
        visible_depth >= quantity
    )

    fill_ratio = np.minimum(
        visible_depth / quantity,
        1.0,
    )

    relative_order_size = np.divide(
        quantity,
        visible_depth,
        out=np.full(
            len(visible_depth),
            np.nan,
            dtype=float,
        ),
        where=visible_depth > 0,
    )

    reaches_quantity = (
        cumulative_size >= quantity
    )

    terminal_level = np.argmax(
        reaches_quantity,
        axis=1,
    )

    row_idx = np.arange(
        len(prices)
    )

    previous_level = np.maximum(
        terminal_level - 1,
        0,
    )

    previous_size = np.where(
        terminal_level > 0,
        cumulative_size[
            row_idx,
            previous_level,
        ],
        0.0,
    )

    previous_notional = np.where(
        terminal_level > 0,
        cumulative_notional[
            row_idx,
            previous_level,
        ],
        0.0,
    )

    terminal_price = safe_prices[
        row_idx,
        terminal_level,
    ]

    remaining_quantity = (
        quantity - previous_size
    )

    full_notional = (
        previous_notional
        + remaining_quantity
        * terminal_price
    )

    vwap = np.where(
        full_fill,
        full_notional / quantity,
        np.nan,
    )

    levels_consumed = np.where(
        full_fill,
        terminal_level + 1,
        np.nan,
    )

    return {
        "vwap": vwap,
        "full_fill": full_fill,
        "fill_ratio": fill_ratio,
        "levels_consumed": levels_consumed,
        "visible_depth": visible_depth,
        "relative_order_size": relative_order_size,
    }


def add_execution_metrics(
    df: pd.DataFrame,
    quantity: float = 1000,
    levels: int = 10,
) -> pd.DataFrame:
    out = df.copy()

    ask_prices, ask_sizes = _book_arrays(
        out,
        side="ask",
        levels=levels,
    )

    bid_prices, bid_sizes = _book_arrays(
        out,
        side="bid",
        levels=levels,
    )

    buy = _simulate_book_sweep(
        prices=ask_prices,
        sizes=ask_sizes,
        quantity=quantity,
    )

    sell = _simulate_book_sweep(
        prices=bid_prices,
        sizes=bid_sizes,
        quantity=quantity,
    )

    suffix = int(quantity)

    out[f"buy_vwap_{suffix}"] = (
        buy["vwap"]
    )

    out[f"sell_vwap_{suffix}"] = (
        sell["vwap"]
    )

    out[f"buy_full_fill_{suffix}"] = (
        buy["full_fill"]
    )

    out[f"sell_full_fill_{suffix}"] = (
        sell["full_fill"]
    )

    out[f"buy_fill_ratio_{suffix}"] = (
        buy["fill_ratio"]
    )

    out[f"sell_fill_ratio_{suffix}"] = (
        sell["fill_ratio"]
    )

    out[
        f"buy_levels_consumed_{suffix}"
    ] = buy["levels_consumed"]

    out[
        f"sell_levels_consumed_{suffix}"
    ] = sell["levels_consumed"]

    out[
        f"ask_visible_depth_{levels}"
    ] = buy["visible_depth"]

    out[
        f"bid_visible_depth_{levels}"
    ] = sell["visible_depth"]

    out[
        f"buy_relative_order_size_{suffix}"
    ] = buy["relative_order_size"]

    out[
        f"sell_relative_order_size_{suffix}"
    ] = sell["relative_order_size"]

    mid = out["mid"]

    out[f"buy_cost_bps_{suffix}"] = (
        (
            out[f"buy_vwap_{suffix}"]
            - mid
        )
        / mid
        * 10_000
    )

    out[f"sell_cost_bps_{suffix}"] = (
        (
            mid
            - out[f"sell_vwap_{suffix}"]
        )
        / mid
        * 10_000
    )

    return out


def add_execution_costs(
    df: pd.DataFrame,
    quantity: float = 1000,
    levels: int = 10,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper used by Part 1.
    """
    return add_execution_metrics(
        df=df,
        quantity=quantity,
        levels=levels,
    )