from __future__ import annotations

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)


FINAL_FEATURE_SET_NAME = "F5_flow"

FINAL_FEATURES = [
    "spread_bps",
    "depth_10_total",
    "top_imbalance",
    "microprice_minus_mid_bps",
    "depth_5_total",
    "depth_imbalance_5",
    "depth_imbalance_10",
    "execution_volume_100",
    "trade_imbalance_100",
    "signed_book_flow",
    "book_pressure_100",
]

HORIZONS = (
    1.0,
    5.0,
    10.0,
)

MODEL_PARAMETERS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


def build_final_model(
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        **MODEL_PARAMETERS
    )