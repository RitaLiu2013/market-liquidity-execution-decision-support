import numpy as np
import pandas as pd

from execution_ml.targets import (
    add_future_execution_targets,
)


def test_future_target_uses_latest_state_at_or_before_horizon():
    df = pd.DataFrame(
        {
            "time": [
                0.0,
                0.8,
                1.1,
                2.2,
            ],
            "mid": [
                100.0,
                100.5,
                101.0,
                102.0,
            ],
            "buy_vwap_100": [
                100.1,
                100.6,
                101.2,
                102.2,
            ],
            "sell_vwap_100": [
                99.9,
                100.4,
                100.8,
                101.8,
            ],
        }
    )

    result = (
        add_future_execution_targets(
            df,
            quantity=100,
            horizons=(1.0,),
        )
    )

    # At t=0, requested future time is 1.0.
    # The most recent state at or before 1.0
    # is the snapshot at 0.8.
    assert np.isclose(
        result.loc[
            0,
            "future_state_time_1p0s",
        ],
        0.8,
    )

    assert np.isclose(
        result.loc[
            0,
            "state_staleness_1p0s",
        ],
        0.2,
    )


def test_wait_buy_cost_uses_current_mid():
    df = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "mid": [
                100.0,
                101.0,
            ],
            "buy_vwap_100": [
                100.1,
                101.2,
            ],
            "sell_vwap_100": [
                99.9,
                100.8,
            ],
        }
    )

    result = (
        add_future_execution_targets(
            df,
            quantity=100,
            horizons=(1.0,),
        )
    )

    expected = (
        (101.2 - 100.0)
        / 100.0
        * 10_000
    )

    assert np.isclose(
        result.loc[
            0,
            "wait_buy_cost_bps_1p0s",
        ],
        expected,
    )


def test_target_beyond_end_of_day_is_missing():
    df = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "mid": [
                100.0,
                101.0,
            ],
            "buy_vwap_100": [
                100.1,
                101.2,
            ],
            "sell_vwap_100": [
                99.9,
                100.8,
            ],
        }
    )

    result = (
        add_future_execution_targets(
            df,
            quantity=100,
            horizons=(1.0,),
        )
    )

    assert np.isnan(
        result.loc[
            1,
            "wait_buy_cost_bps_1p0s",
        ]
    )

    assert np.isnan(
        result.loc[
            1,
            "future_state_time_1p0s",
        ]
    )