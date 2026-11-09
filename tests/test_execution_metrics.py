import numpy as np
import pandas as pd

from execution_ml.execution_cost import (
    add_execution_metrics,
)


def make_book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mid": [100.00],
            "ask_price_1": [100.01],
            "ask_size_1": [100],
            "ask_price_2": [100.02],
            "ask_size_2": [200],
            "bid_price_1": [99.99],
            "bid_size_1": [100],
            "bid_price_2": [99.98],
            "bid_size_2": [200],
        }
    )


def test_full_fill_uses_two_levels():
    df = make_book()

    result = add_execution_metrics(
        df,
        quantity=150,
        levels=2,
    )

    expected_vwap = (
        100 * 100.01
        + 50 * 100.02
    ) / 150

    assert result.loc[
        0,
        "buy_full_fill_150",
    ]

    assert np.isclose(
        result.loc[
            0,
            "buy_fill_ratio_150",
        ],
        1.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "buy_levels_consumed_150",
        ],
        2,
    )

    assert np.isclose(
        result.loc[
            0,
            "buy_vwap_150",
        ],
        expected_vwap,
    )

    assert np.isclose(
        result.loc[
            0,
            "buy_relative_order_size_150",
        ],
        150 / 300,
    )


def test_insufficient_depth_is_not_full_fill():
    df = make_book()

    result = add_execution_metrics(
        df,
        quantity=500,
        levels=2,
    )

    assert not result.loc[
        0,
        "buy_full_fill_500",
    ]

    assert np.isclose(
        result.loc[
            0,
            "buy_fill_ratio_500",
        ],
        300 / 500,
    )

    assert np.isnan(
        result.loc[
            0,
            "buy_vwap_500",
        ]
    )

    assert np.isnan(
        result.loc[
            0,
            "buy_cost_bps_500",
        ]
    )

    assert np.isclose(
        result.loc[
            0,
            "buy_relative_order_size_500",
        ],
        500 / 300,
    )