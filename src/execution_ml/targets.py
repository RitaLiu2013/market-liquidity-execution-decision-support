from __future__ import annotations

import numpy as np
import pandas as pd


def add_wait_cost_targets(
    df: pd.DataFrame,
    quantity: float = 1000,
    horizon_seconds: float = 5.0,
    tolerance_seconds: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()

    suffix = int(quantity)

    times = out["time"].to_numpy(dtype=float)
    mid = out["mid"].to_numpy(dtype=float)

    buy_vwap = out[
        f"buy_vwap_{suffix}"
    ].to_numpy(dtype=float)

    sell_vwap = out[
        f"sell_vwap_{suffix}"
    ].to_numpy(dtype=float)

    desired_times = times + horizon_seconds

    future_idx = np.searchsorted(
        times,
        desired_times,
        side="left",
    )

    valid = future_idx < len(out)

    safe_idx = np.minimum(
        future_idx,
        len(out) - 1,
    )

    future_times = times[safe_idx]

    valid &= (
        future_times
        <= desired_times + tolerance_seconds
    )

    future_mid = np.full(len(out), np.nan)
    future_buy_vwap = np.full(len(out), np.nan)
    future_sell_vwap = np.full(len(out), np.nan)

    future_mid[valid] = mid[safe_idx[valid]]

    future_buy_vwap[valid] = (
        buy_vwap[safe_idx[valid]]
    )

    future_sell_vwap[valid] = (
        sell_vwap[safe_idx[valid]]
    )

    out["future_time"] = np.where(
        valid,
        future_times,
        np.nan,
    )

    out["actual_horizon_seconds"] = (
        out["future_time"] - out["time"]
    )

    out["future_mid"] = future_mid

    out[f"future_buy_vwap_{suffix}"] = (
        future_buy_vwap
    )

    out[f"future_sell_vwap_{suffix}"] = (
        future_sell_vwap
    )

    # Cost of waiting for a BUY order,
    # measured relative to the current decision-time mid.
    out[f"wait_buy_cost_bps_{suffix}"] = (
        (
            out[f"future_buy_vwap_{suffix}"]
            - out["mid"]
        )
        / out["mid"]
        * 10_000
    )

    # Cost of waiting for a SELL order.
    out[f"wait_sell_cost_bps_{suffix}"] = (
        (
            out["mid"]
            - out[f"future_sell_vwap_{suffix}"]
        )
        / out["mid"]
        * 10_000
    )

    # Decomposition for interpretation.
    out["future_mid_move_buy_bps"] = (
        (out["future_mid"] - out["mid"])
        / out["mid"]
        * 10_000
    )

    out[
        f"future_buy_liquidity_cost_bps_{suffix}"
    ] = (
        (
            out[f"future_buy_vwap_{suffix}"]
            - out["future_mid"]
        )
        / out["future_mid"]
        * 10_000
    )

    return out