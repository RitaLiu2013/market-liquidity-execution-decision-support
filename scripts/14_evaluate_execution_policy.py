from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execution_ml.final_model_spec import (
    FINAL_FEATURES,
    HORIZONS,
    build_final_model,
)


DATA_ROOT = Path(
    "data/processed"
)

BUFFER_SELECTION_OUTPUT = Path(
    "outputs/policy_buffer_validation.csv"
)

TEST_GROUP_OUTPUT = Path(
    "outputs/final_policy_test_group_metrics.csv"
)

TEST_SECURITY_OUTPUT = Path(
    "outputs/final_policy_test_security_summary.csv"
)

TEST_ACTION_OUTPUT = Path(
    "outputs/final_policy_test_action_frequencies.csv"
)

TEST_DECISIONS_OUTPUT = Path(
    "outputs/final_policy_test_decisions.parquet"
)


# The grid is defined before looking at policy-test outcomes.
#
# Units are basis points of predicted improvement required
# before we allow the policy to wait.
BUFFER_GRID_BPS = (
    0.00,
    0.05,
    0.10,
    0.25,
    0.50,
)


ACTION_NAMES = (
    "execute_now",
    "wait_1s",
    "wait_5s",
    "wait_10s",
)


def horizon_suffix(
    horizon: float,
) -> str:
    return (
        str(float(horizon))
        .replace(".", "p")
    )


def target_column(
    side: str,
    horizon: float,
) -> str:
    suffix = horizon_suffix(
        horizon
    )

    return (
        f"wait_{side}_cost_bps_"
        f"{suffix}s"
    )


def current_cost_column(
    side: str,
    quantity: int,
) -> str:
    return (
        f"{side}_cost_bps_"
        f"{quantity}"
    )


def validate_configuration() -> None:
    expected_horizons = (
        1.0,
        5.0,
        10.0,
    )

    observed_horizons = tuple(
        float(
            horizon
        )
        for horizon in HORIZONS
    )

    if observed_horizons != expected_horizons:
        raise ValueError(
            "Step 14 currently assumes exactly "
            "1s, 5s, and 10s horizons. "
            f"Found {observed_horizons}."
        )

    if not FINAL_FEATURES:
        raise ValueError(
            "FINAL_FEATURES is empty."
        )

    if len(
        FINAL_FEATURES
    ) != len(
        set(
            FINAL_FEATURES
        )
    ):
        raise ValueError(
            "FINAL_FEATURES contains duplicates."
        )

    if not BUFFER_GRID_BPS:
        raise ValueError(
            "BUFFER_GRID_BPS is empty."
        )

    if any(
        buffer < 0
        for buffer in BUFFER_GRID_BPS
    ):
        raise ValueError(
            "Decision buffers must be "
            "non-negative."
        )

    if tuple(
        sorted(
            set(
                BUFFER_GRID_BPS
            )
        )
    ) != BUFFER_GRID_BPS:
        raise ValueError(
            "BUFFER_GRID_BPS must be sorted "
            "and contain unique values."
        )


def validate_dataset(
    df: pd.DataFrame,
    path: Path,
) -> tuple[str, int]:
    required = [
        "security",
        "calibrated_quantity",
        "split",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name}: missing columns "
            f"{missing}"
        )

    if df.empty:
        raise ValueError(
            f"{path.name}: dataset is empty."
        )

    securities = (
        df["security"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(
        securities
    ) != 1:
        raise ValueError(
            f"{path.name}: expected one "
            "security."
        )

    quantities = (
        df[
            "calibrated_quantity"
        ]
        .dropna()
        .unique()
    )

    if len(
        quantities
    ) != 1:
        raise ValueError(
            f"{path.name}: expected one "
            "calibrated quantity."
        )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    observed_splits = set(
        df[
            "split"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_splits = (
        required_splits
        - observed_splits
    )

    if missing_splits:
        raise ValueError(
            f"{path.name}: missing splits "
            f"{sorted(missing_splits)}"
        )

    return (
        str(
            securities[0]
        ),
        int(
            quantities[0]
        ),
    )


def assert_finite_frame(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    values = (
        frame[
            columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{label}: non-finite values found."
        )


def policy_required_columns(
    side: str,
    quantity: int,
) -> list[str]:
    columns = list(
        FINAL_FEATURES
    )

    columns.append(
        current_cost_column(
            side,
            quantity,
        )
    )

    for horizon in HORIZONS:
        columns.append(
            target_column(
                side,
                float(
                    horizon
                ),
            )
        )

    return columns


def build_policy_frame(
    df: pd.DataFrame,
    split_name: str,
    side: str,
    quantity: int,
) -> pd.DataFrame:
    required_columns = (
        policy_required_columns(
            side,
            quantity,
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing policy columns: "
            f"{missing_columns}"
        )

    split = df.loc[
        df["split"]
        == split_name
    ]

    if split.empty:
        raise ValueError(
            f"Empty split: {split_name}"
        )

    policy_frame = (
        split[
            required_columns
        ]
        .dropna()
        .copy()
    )

    if policy_frame.empty:
        raise ValueError(
            f"No complete policy rows for "
            f"{split_name}, {side}."
        )

    assert_finite_frame(
        policy_frame,
        required_columns,
        (
            f"{split_name} "
            f"{side} policy frame"
        ),
    )

    return policy_frame


def fit_horizon_models_and_predict(
    training_data: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    side: str,
    quantity: int,
) -> dict[float, np.ndarray]:
    predictions: dict[
        float,
        np.ndarray,
    ] = {}

    current_col = (
        current_cost_column(
            side,
            quantity,
        )
    )

    for horizon in HORIZONS:
        horizon = float(
            horizon
        )

        target_col = (
            target_column(
                side,
                horizon,
            )
        )

        training_columns = (
            list(
                FINAL_FEATURES
            )
            + [
                current_col,
                target_col,
            ]
        )

        missing_columns = [
            column
            for column in training_columns
            if column not in training_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing training columns "
                f"{missing_columns}"
            )

        train_common = (
            training_data[
                training_columns
            ]
            .dropna()
            .copy()
        )

        if train_common.empty:
            raise ValueError(
                f"No training rows for "
                f"{side} {horizon}s."
            )

        assert_finite_frame(
            train_common,
            training_columns,
            (
                f"training "
                f"{side} "
                f"{horizon}s"
            ),
        )

        X_train = (
            train_common[
                FINAL_FEATURES
            ]
        )

        residual_target = (
            train_common[
                target_col
            ]
            - train_common[
                current_col
            ]
        )

        X_evaluation = (
            evaluation_frame[
                FINAL_FEATURES
            ]
        )

        current_cost = (
            evaluation_frame[
                current_col
            ]
            .to_numpy(
                dtype=float
            )
        )

        model = (
            build_final_model()
        )

        model.fit(
            X_train,
            residual_target,
        )

        predicted_residual = (
            model.predict(
                X_evaluation
            )
        )

        predicted_future_cost = (
            current_cost
            + predicted_residual
        )

        if not np.isfinite(
            predicted_future_cost
        ).all():
            raise ValueError(
                f"Non-finite predictions for "
                f"{side} {horizon}s."
            )

        predictions[
            horizon
        ] = (
            predicted_future_cost
        )

    return predictions


def build_decision_table(
    policy_frame: pd.DataFrame,
    predictions: dict[
        float,
        np.ndarray,
    ],
    side: str,
    quantity: int,
    buffer_bps: float,
) -> pd.DataFrame:
    current_col = (
        current_cost_column(
            side,
            quantity,
        )
    )

    current_cost = (
        policy_frame[
            current_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted_wait = np.column_stack(
        [
            predictions[
                float(
                    horizon
                )
            ]
            for horizon in HORIZONS
        ]
    )

    realized_wait = np.column_stack(
        [
            policy_frame[
                target_column(
                    side,
                    float(
                        horizon
                    ),
                )
            ]
            .to_numpy(
                dtype=float
            )
            for horizon in HORIZONS
        ]
    )

    if predicted_wait.shape != (
        len(
            policy_frame
        ),
        len(
            HORIZONS
        ),
    ):
        raise ValueError(
            "Unexpected predicted-wait shape."
        )

    if realized_wait.shape != (
        len(
            policy_frame
        ),
        len(
            HORIZONS
        ),
    ):
        raise ValueError(
            "Unexpected realized-wait shape."
        )

    # Find the best WAIT horizon according
    # to the model.
    best_wait_position = (
        np.argmin(
            predicted_wait,
            axis=1,
        )
    )

    row_index = np.arange(
        len(
            policy_frame
        )
    )

    best_predicted_wait = (
        predicted_wait[
            row_index,
            best_wait_position,
        ]
    )

    # Wait only if the predicted advantage
    # exceeds the required buffer.
    choose_wait = (
        best_predicted_wait
        < (
            current_cost
            - buffer_bps
        )
    )

    # Action index:
    # 0 = now
    # 1 = wait 1s
    # 2 = wait 5s
    # 3 = wait 10s
    chosen_action_index = np.where(
        choose_wait,
        best_wait_position + 1,
        0,
    )

    all_realized_costs = (
        np.column_stack(
            [
                current_cost,
                realized_wait,
            ]
        )
    )

    all_predicted_costs = (
        np.column_stack(
            [
                current_cost,
                predicted_wait,
            ]
        )
    )

    realized_policy_cost = (
        all_realized_costs[
            row_index,
            chosen_action_index,
        ]
    )

    predicted_policy_cost = (
        all_predicted_costs[
            row_index,
            chosen_action_index,
        ]
    )

    oracle_action_index = (
        np.argmin(
            all_realized_costs,
            axis=1,
        )
    )

    oracle_cost = (
        all_realized_costs[
            row_index,
            oracle_action_index,
        ]
    )

    savings_vs_now = (
        current_cost
        - realized_policy_cost
    )

    oracle_savings_vs_now = (
        current_cost
        - oracle_cost
    )

    oracle_gap = (
        realized_policy_cost
        - oracle_cost
    )

    decision_table = pd.DataFrame(
        {
            "current_cost_bps": (
                current_cost
            ),
            "predicted_wait_1s_bps": (
                predicted_wait[
                    :,
                    0,
                ]
            ),
            "predicted_wait_5s_bps": (
                predicted_wait[
                    :,
                    1,
                ]
            ),
            "predicted_wait_10s_bps": (
                predicted_wait[
                    :,
                    2,
                ]
            ),
            "realized_wait_1s_bps": (
                realized_wait[
                    :,
                    0,
                ]
            ),
            "realized_wait_5s_bps": (
                realized_wait[
                    :,
                    1,
                ]
            ),
            "realized_wait_10s_bps": (
                realized_wait[
                    :,
                    2,
                ]
            ),
            "chosen_action_index": (
                chosen_action_index
            ),
            "oracle_action_index": (
                oracle_action_index
            ),
            "predicted_policy_cost_bps": (
                predicted_policy_cost
            ),
            "realized_policy_cost_bps": (
                realized_policy_cost
            ),
            "oracle_cost_bps": (
                oracle_cost
            ),
            "savings_vs_now_bps": (
                savings_vs_now
            ),
            "oracle_savings_vs_now_bps": (
                oracle_savings_vs_now
            ),
            "oracle_gap_bps": (
                oracle_gap
            ),
        },
        index=policy_frame.index,
    )

    decision_table[
        "chosen_action"
    ] = [
        ACTION_NAMES[
            int(
                action
            )
        ]
        for action
        in chosen_action_index
    ]

    decision_table[
        "oracle_action"
    ] = [
        ACTION_NAMES[
            int(
                action
            )
        ]
        for action
        in oracle_action_index
    ]

    return decision_table


def summarise_policy(
    decisions: pd.DataFrame,
) -> dict[str, float | int]:
    wait_mask = (
        decisions[
            "chosen_action"
        ]
        != "execute_now"
    )

    waits = (
        decisions.loc[
            wait_mask
        ]
    )

    if len(
        waits
    ) > 0:
        wait_hit_rate = float(
            (
                waits[
                    "savings_vs_now_bps"
                ]
                > 0
            ).mean()
        )

        mean_wait_savings = float(
            waits[
                "savings_vs_now_bps"
            ].mean()
        )
    else:
        wait_hit_rate = np.nan
        mean_wait_savings = np.nan

    policy_cost_over_now = (
        decisions[
            "realized_policy_cost_bps"
        ]
        - decisions[
            "current_cost_bps"
        ]
    )

    return {
        "rows": len(
            decisions
        ),
        "mean_policy_cost_bps": float(
            decisions[
                "realized_policy_cost_bps"
            ].mean()
        ),
        "mean_execute_now_cost_bps": float(
            decisions[
                "current_cost_bps"
            ].mean()
        ),
        "mean_savings_vs_now_bps": float(
            decisions[
                "savings_vs_now_bps"
            ].mean()
        ),
        "median_savings_vs_now_bps": float(
            decisions[
                "savings_vs_now_bps"
            ].median()
        ),
        "positive_savings_rate": float(
            (
                decisions[
                    "savings_vs_now_bps"
                ]
                > 0
            ).mean()
        ),
        "negative_savings_rate": float(
            (
                decisions[
                    "savings_vs_now_bps"
                ]
                < 0
            ).mean()
        ),
        "wait_rate": float(
            wait_mask.mean()
        ),
        "wait_hit_rate": (
            wait_hit_rate
        ),
        "mean_wait_savings_bps": (
            mean_wait_savings
        ),
        "oracle_action_match_rate": float(
            (
                decisions[
                    "chosen_action"
                ]
                == decisions[
                    "oracle_action"
                ]
            ).mean()
        ),
        "mean_oracle_savings_bps": float(
            decisions[
                "oracle_savings_vs_now_bps"
            ].mean()
        ),
        "mean_oracle_gap_bps": float(
            decisions[
                "oracle_gap_bps"
            ].mean()
        ),
        "median_oracle_gap_bps": float(
            decisions[
                "oracle_gap_bps"
            ].median()
        ),
        "p95_cost_over_now_bps": float(
            policy_cost_over_now.quantile(
                0.95
            )
        ),
        "p99_cost_over_now_bps": float(
            policy_cost_over_now.quantile(
                0.99
            )
        ),
    }


def select_buffer(
    validation_group_metrics: pd.DataFrame,
) -> tuple[
    float,
    pd.DataFrame,
]:
    buffer_summary = (
        validation_group_metrics.groupby(
            "buffer_bps",
            as_index=False,
        )
        .agg(
            mean_group_savings_bps=(
                "mean_savings_vs_now_bps",
                "mean",
            ),
            median_group_savings_bps=(
                "mean_savings_vs_now_bps",
                "median",
            ),
            worst_group_savings_bps=(
                "mean_savings_vs_now_bps",
                "min",
            ),
            positive_savings_groups=(
                "mean_savings_vs_now_bps",
                lambda x: int(
                    (
                        x > 0
                    ).sum()
                ),
            ),
            mean_wait_rate=(
                "wait_rate",
                "mean",
            ),
            mean_wait_hit_rate=(
                "wait_hit_rate",
                "mean",
            ),
            mean_oracle_gap_bps=(
                "mean_oracle_gap_bps",
                "mean",
            ),
        )
    )

    # Primary objective:
    # maximize mean economic savings
    # across equally weighted security-side groups.
    #
    # Tie-breakers:
    # 1. better worst-group savings;
    # 2. lower wait rate;
    # 3. larger buffer.
    ranked = (
        buffer_summary.sort_values(
            [
                "mean_group_savings_bps",
                "worst_group_savings_bps",
                "mean_wait_rate",
                "buffer_bps",
            ],
            ascending=[
                False,
                False,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    selected_buffer = float(
        ranked.loc[
            0,
            "buffer_bps",
        ]
    )

    buffer_summary[
        "selected"
    ] = (
        buffer_summary[
            "buffer_bps"
        ]
        == selected_buffer
    )

    buffer_summary = (
        buffer_summary.sort_values(
            "buffer_bps"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        selected_buffer,
        buffer_summary,
    )


def main() -> None:
    validate_configuration()

    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No predictive Parquet files found."
        )

    validation_group_rows = []

    cached_data: dict[
        str,
        tuple[
            pd.DataFrame,
            int,
        ],
    ] = {}

    print(
        "\nPhase 1: validation-only "
        "decision-buffer selection"
    )

    for path in paths:
        df = pd.read_parquet(
            path
        )

        security, quantity = (
            validate_dataset(
                df,
                path,
            )
        )

        cached_data[
            security
        ] = (
            df,
            quantity,
        )

        print(
            f"\nValidation: {security}"
        )

        train = df.loc[
            df["split"]
            == "train"
        ]

        if train.empty:
            raise ValueError(
                f"{security}: empty train split."
            )

        for side in (
            "buy",
            "sell",
        ):
            validation_frame = (
                build_policy_frame(
                    df,
                    "validation",
                    side,
                    quantity,
                )
            )

            validation_predictions = (
                fit_horizon_models_and_predict(
                    training_data=train,
                    evaluation_frame=(
                        validation_frame
                    ),
                    side=side,
                    quantity=quantity,
                )
            )

            validation_split_rows = int(
                (
                    df[
                        "split"
                    ]
                    == "validation"
                ).sum()
            )

            validation_coverage = (
                len(
                    validation_frame
                )
                / validation_split_rows
            )

            for buffer_bps in (
                BUFFER_GRID_BPS
            ):
                decisions = (
                    build_decision_table(
                        policy_frame=(
                            validation_frame
                        ),
                        predictions=(
                            validation_predictions
                        ),
                        side=side,
                        quantity=quantity,
                        buffer_bps=float(
                            buffer_bps
                        ),
                    )
                )

                metrics = (
                    summarise_policy(
                        decisions
                    )
                )

                validation_group_rows.append(
                    {
                        "security": (
                            security
                        ),
                        "side": side,
                        "buffer_bps": float(
                            buffer_bps
                        ),
                        "coverage": (
                            validation_coverage
                        ),
                        **metrics,
                    }
                )

    validation_group_metrics = (
        pd.DataFrame(
            validation_group_rows
        )
    )

    selected_buffer, buffer_summary = (
        select_buffer(
            validation_group_metrics
        )
    )

    BUFFER_SELECTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    buffer_summary.to_csv(
        BUFFER_SELECTION_OUTPUT,
        index=False,
    )

    print(
        "\nValidation buffer selection"
    )

    print(
        buffer_summary.to_string(
            index=False
        )
    )

    print(
        "\nSelected decision buffer: "
        f"{selected_buffer:.4f} bps"
    )

    print(
        "\nPhase 2: frozen-policy "
        "test evaluation"
    )

    test_group_rows = []
    decision_frames = []

    for security in sorted(
        cached_data
    ):
        df, quantity = (
            cached_data[
                security
            ]
        )

        print(
            f"\nTest: {security}"
        )

        development = df.loc[
            df[
                "split"
            ].isin(
                [
                    "train",
                    "validation",
                ]
            )
        ]

        if development.empty:
            raise ValueError(
                f"{security}: empty "
                "development sample."
            )

        for side in (
            "buy",
            "sell",
        ):
            test_frame = (
                build_policy_frame(
                    df,
                    "test",
                    side,
                    quantity,
                )
            )

            test_predictions = (
                fit_horizon_models_and_predict(
                    training_data=development,
                    evaluation_frame=(
                        test_frame
                    ),
                    side=side,
                    quantity=quantity,
                )
            )

            decisions = (
                build_decision_table(
                    policy_frame=(
                        test_frame
                    ),
                    predictions=(
                        test_predictions
                    ),
                    side=side,
                    quantity=quantity,
                    buffer_bps=(
                        selected_buffer
                    ),
                )
            )

            test_split_rows = int(
                (
                    df[
                        "split"
                    ]
                    == "test"
                ).sum()
            )

            coverage = (
                len(
                    test_frame
                )
                / test_split_rows
            )

            metrics = (
                summarise_policy(
                    decisions
                )
            )

            test_group_rows.append(
                {
                    "security": security,
                    "side": side,
                    "quantity": quantity,
                    "selected_buffer_bps": (
                        selected_buffer
                    ),
                    "coverage": coverage,
                    **metrics,
                }
            )

            decisions = (
                decisions.copy()
            )

            decisions[
                "security"
            ] = security

            decisions[
                "side"
            ] = side

            decisions[
                "quantity"
            ] = quantity

            decisions[
                "selected_buffer_bps"
            ] = (
                selected_buffer
            )

            decisions[
                "source_index"
            ] = (
                decisions.index
            )

            if "time" in df.columns:
                decisions[
                    "time"
                ] = (
                    df.loc[
                        decisions.index,
                        "time",
                    ]
                    .to_numpy()
                )

            decision_frames.append(
                decisions.reset_index(
                    drop=True
                )
            )

    test_group_metrics = (
        pd.DataFrame(
            test_group_rows
        )
        .sort_values(
            [
                "security",
                "side",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    all_decisions = pd.concat(
        decision_frames,
        ignore_index=True,
    )

    test_group_metrics.to_csv(
        TEST_GROUP_OUTPUT,
        index=False,
    )

    all_decisions.to_parquet(
        TEST_DECISIONS_OUTPUT,
        index=False,
        compression="zstd",
    )

    security_summary = (
        test_group_metrics.groupby(
            "security",
            as_index=False,
        )
        .agg(
            mean_savings_vs_now_bps=(
                "mean_savings_vs_now_bps",
                "mean",
            ),
            median_group_savings_bps=(
                "mean_savings_vs_now_bps",
                "median",
            ),
            mean_wait_rate=(
                "wait_rate",
                "mean",
            ),
            mean_wait_hit_rate=(
                "wait_hit_rate",
                "mean",
            ),
            mean_oracle_gap_bps=(
                "mean_oracle_gap_bps",
                "mean",
            ),
            mean_coverage=(
                "coverage",
                "mean",
            ),
        )
    )

    security_summary.to_csv(
        TEST_SECURITY_OUTPUT,
        index=False,
    )

    action_frequency = (
        all_decisions.groupby(
            [
                "security",
                "side",
                "chosen_action",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "rows",
            }
        )
    )

    group_totals = (
        action_frequency.groupby(
            [
                "security",
                "side",
            ]
        )[
            "rows"
        ]
        .transform(
            "sum"
        )
    )

    action_frequency[
        "action_rate"
    ] = (
        action_frequency[
            "rows"
        ]
        / group_totals
    )

    action_frequency.to_csv(
        TEST_ACTION_OUTPUT,
        index=False,
    )

    overall_group_weighted = {
        "selected_buffer_bps": (
            selected_buffer
        ),
        "security_side_groups": len(
            test_group_metrics
        ),
        "mean_group_savings_vs_now_bps": float(
            test_group_metrics[
                "mean_savings_vs_now_bps"
            ].mean()
        ),
        "median_group_savings_vs_now_bps": float(
            test_group_metrics[
                "mean_savings_vs_now_bps"
            ].median()
        ),
        "positive_savings_groups": int(
            (
                test_group_metrics[
                    "mean_savings_vs_now_bps"
                ]
                > 0
            ).sum()
        ),
        "mean_group_wait_rate": float(
            test_group_metrics[
                "wait_rate"
            ].mean()
        ),
        "mean_group_wait_hit_rate": float(
            test_group_metrics[
                "wait_hit_rate"
            ].mean()
        ),
        "mean_group_oracle_gap_bps": float(
            test_group_metrics[
                "mean_oracle_gap_bps"
            ].mean()
        ),
        "mean_group_coverage": float(
            test_group_metrics[
                "coverage"
            ].mean()
        ),
        "worst_group_savings_bps": float(
            test_group_metrics[
                "mean_savings_vs_now_bps"
            ].min()
        ),
        "best_group_savings_bps": float(
            test_group_metrics[
                "mean_savings_vs_now_bps"
            ].max()
        ),
    }

    print(
        "\nFinal policy test metrics "
        "by security / side"
    )

    print(
        test_group_metrics.to_string(
            index=False
        )
    )

    print(
        "\nSecurity policy summary"
    )

    print(
        security_summary.to_string(
            index=False
        )
    )

    print(
        "\nAction frequencies"
    )

    print(
        action_frequency.to_string(
            index=False
        )
    )

    print(
        "\nOverall policy summary "
        "(equal weight per "
        "security-side group)"
    )

    for key, value in (
        overall_group_weighted.items()
    ):
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key}: "
                f"{value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        "\nInterpretation:"
    )

    print(
        "Positive mean_savings_vs_now_bps "
        "means the policy reduced realized "
        "execution cost versus always "
        "executing immediately."
    )

    print(
        "mean_oracle_gap_bps measures how "
        "much execution cost remains versus "
        "a perfect foresight policy."
    )

    print(
        "wait_hit_rate is measured only "
        "among decisions where the policy "
        "actually waited."
    )

    print(
        "\nSaved:"
    )

    for path in (
        BUFFER_SELECTION_OUTPUT,
        TEST_GROUP_OUTPUT,
        TEST_SECURITY_OUTPUT,
        TEST_ACTION_OUTPUT,
        TEST_DECISIONS_OUTPUT,
    ):
        print(
            path
        )


if __name__ == "__main__":
    main()