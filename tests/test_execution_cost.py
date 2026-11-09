import numpy as np
import pandas as pd

from execution_ml.execution_cost import add_execution_costs


def test_buy_order_walks_the_book():
    df = pd.DataFrame(
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

    result = add_execution_costs(
        df,
        quantity=150,
        levels=2,
    )

    expected_vwap = (
        100 * 100.01
        + 50 * 100.02
    ) / 150

    assert np.isclose(
        result.loc[0, "buy_vwap_150"],
        expected_vwap,
    )


def test_insufficient_depth_returns_nan():
    df = pd.DataFrame(
        {
            "mid": [100.00],
            "ask_price_1": [100.01],
            "ask_size_1": [100],
            "bid_price_1": [99.99],
            "bid_size_1": [100],
        }
    )

    result = add_execution_costs(
        df,
        quantity=500,
        levels=1,
    )

    assert np.isnan(
        result.loc[0, "buy_vwap_500"]
    )