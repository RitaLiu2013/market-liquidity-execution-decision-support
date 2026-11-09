import numpy as np
import pandas as pd

from execution_ml.features import add_microstructure_features


def test_spread_bps():
    df = pd.DataFrame(
        {
            "ask_price_1": [100.01],
            "ask_size_1": [100],
            "bid_price_1": [99.99],
            "bid_size_1": [100],
            "event_type": [1],
            "direction": [1],
            "size": [100],
            "time": [34_200.0],
        }
    )

    result = add_microstructure_features(
        df,
        levels=1,
    )

    assert np.isclose(
        result.loc[0, "mid"],
        100.0,
    )

    assert np.isclose(
        result.loc[0, "spread_bps"],
        2.0,
    )