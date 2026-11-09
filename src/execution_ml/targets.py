from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_execution_targets(
    df: pd.DataFrame,
    quantity: int,
    horizons: tuple[float, ...] = (1.0, 5.0, 10.0),
) -> pd.DataFrame:
    out = df.copy()

    time = out["time"].to_numpy(dtype=float)
    mid = out["mid"].to_numpy(dtype=float)

    buy_vwap = out[
        f"buy_vwap_{quantity}"
    ].to_numpy(dtype=float)

    sell_vwap = out[
        f"sell_vwap_{quantity}"
    ].to_numpy(dtype=float)

    n = len(out)

    for horizon in horizons:
        target_time = time + horizon

        future_idx = np.searchsorted(
            time,
            target_time,
            side="left",
        )

        valid = future_idx < n

        matched_time = np.full(
            n,
            np.nan,
            dtype=float,
        )

        future_mid = np.full(
            n,
            np.nan,
            dtype=float,
        )

        future_buy_vwap = np.full(
            n,
            np.nan,
            dtype=float,
        )

        future_sell_vwap = np.full(
            n,
            np.nan,
            dtype=float,
        )

        matched_time[valid] = (
            time[future_idx[valid]]
        )

        future_mid[valid] = (
            mid[future_idx[valid]]
        )

        future_buy_vwap[valid] = (
            buy_vwap[future_idx[valid]]
        )

        future_sell_vwap[valid] = (
            sell_vwap[future_idx[valid]]
        )

        suffix = (
            str(horizon)
            .replace(".", "p")
        )

        out[
            f"future_time_{suffix}s"
        ] = matched_time

        out[
            f"actual_horizon_{suffix}s"
        ] = (
            matched_time - time
        )

        out[
            f"future_mid_{suffix}s"
        ] = future_mid

        out[
            f"wait_buy_cost_bps_{suffix}s"
        ] = (
            (
                future_buy_vwap
                - mid
            )
            / mid
            * 10_000
        )

        out[
            f"wait_sell_cost_bps_{suffix}s"
        ] = (
            (
                mid
                - future_sell_vwap
            )
            / mid
            * 10_000
        )

        out[
            f"mid_move_bps_{suffix}s"
        ] = (
            (
                future_mid
                - mid
            )
            / mid
            * 10_000
        )

        out[
            f"future_buy_liquidity_cost_bps_{suffix}s"
        ] = (
            (
                future_buy_vwap
                - future_mid
            )
            / future_mid
            * 10_000
        )

        out[
            f"future_sell_liquidity_cost_bps_{suffix}s"
        ] = (
            (
                future_mid
                - future_sell_vwap
            )
            / future_mid
            * 10_000
        )

    return out