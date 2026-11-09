from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_microstructure_features(
    df: pd.DataFrame,
    levels: int = 10,
) -> pd.DataFrame:
    out = df.copy()

    ask = out["ask_price_1"]
    bid = out["bid_price_1"]

    ask_size = out["ask_size_1"]
    bid_size = out["bid_size_1"]

    out["mid"] = (ask + bid) / 2.0
    out["spread"] = ask - bid

    out["spread_bps"] = (
        _safe_divide(out["spread"], out["mid"]) * 10_000
    )

    top_depth = bid_size + ask_size

    out["top_imbalance"] = _safe_divide(
        bid_size - ask_size,
        top_depth,
    )

    out["microprice"] = _safe_divide(
        ask * bid_size + bid * ask_size,
        top_depth,
    )

    out["microprice_minus_mid_bps"] = (
        _safe_divide(
            out["microprice"] - out["mid"],
            out["mid"],
        )
        * 10_000
    )

    for depth_level in (5, 10):
        if depth_level > levels:
            continue

        bid_cols = [
            f"bid_size_{level}"
            for level in range(1, depth_level + 1)
        ]
        ask_cols = [
            f"ask_size_{level}"
            for level in range(1, depth_level + 1)
        ]

        bid_depth = out[bid_cols].sum(axis=1)
        ask_depth = out[ask_cols].sum(axis=1)

        out[f"bid_depth_{depth_level}"] = bid_depth
        out[f"ask_depth_{depth_level}"] = ask_depth

        out[f"depth_{depth_level}_total"] = (
            bid_depth + ask_depth
        )

        out[f"depth_imbalance_{depth_level}"] = _safe_divide(
            bid_depth - ask_depth,
            bid_depth + ask_depth,
        )

    log_mid = np.log(out["mid"])

    out["ret_1_event"] = log_mid.diff()
    out["ret_10_event"] = log_mid.diff(10)

    out["rv_50_events_bps"] = (
        out["ret_1_event"]
        .rolling(50, min_periods=20)
        .std()
        * 10_000
    )

    out["rv_200_events_bps"] = (
        out["ret_1_event"]
        .rolling(200, min_periods=50)
        .std()
        * 10_000
    )

    is_execution = out["event_type"].isin([4, 5])

    # For an execution, LOBSTER direction refers to the resting
    # limit order. The aggressive trade direction is therefore opposite.
    aggressor_sign = np.where(
        is_execution,
        -out["direction"].fillna(0),
        0,
    )

    execution_volume = np.where(
        is_execution,
        out["size"].fillna(0),
        0,
    )

    signed_execution_volume = (
        aggressor_sign * execution_volume
    )

    out["execution_volume_100"] = (
        pd.Series(execution_volume, index=out.index)
        .rolling(100, min_periods=20)
        .sum()
    )

    signed_exec_roll = (
        pd.Series(
            signed_execution_volume,
            index=out.index,
        )
        .rolling(100, min_periods=20)
        .sum()
    )

    out["trade_imbalance_100"] = _safe_divide(
        signed_exec_roll,
        out["execution_volume_100"],
    )

    liquidity_sign = out["event_type"].map(
        {
            1: 1.0,   # submission adds liquidity
            2: -1.0,  # partial cancellation removes liquidity
            3: -1.0,  # deletion removes liquidity
            4: -1.0,  # visible execution removes liquidity
        }
    ).fillna(0.0)

    out["signed_book_flow"] = (
        out["direction"].fillna(0)
        * out["size"].fillna(0)
        * liquidity_sign
    )

    out["book_pressure_100"] = (
        out["signed_book_flow"]
        .rolling(100, min_periods=20)
        .sum()
    )

    seconds_in_day = 24 * 60 * 60
    angle = 2 * np.pi * out["time"] / seconds_in_day

    out["time_sin"] = np.sin(angle)
    out["time_cos"] = np.cos(angle)

    return out