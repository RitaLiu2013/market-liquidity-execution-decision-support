from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "outputs/final_policy_test_decisions.parquet"
)

CALIBRATION_OUTPUT_PATH = Path(
    "outputs/policy_advantage_calibration.csv"
)

SECURITY_CALIBRATION_OUTPUT_PATH = Path(
    "outputs/policy_advantage_calibration_by_security.csv"
)

HORIZON_OUTPUT_PATH = Path(
    "outputs/policy_horizon_diagnostics.csv"
)

ACTION_OUTPUT_PATH = Path(
    "outputs/policy_action_diagnostics.csv"
)


PREDICTED_WAIT_COLUMNS = {
    "wait_1s": (
        "predicted_wait_1s_bps"
    ),
    "wait_5s": (
        "predicted_wait_5s_bps"
    ),
    "wait_10s": (
        "predicted_wait_10s_bps"
    ),
}


REALIZED_WAIT_COLUMNS = {
    "wait_1s": (
        "realized_wait_1s_bps"
    ),
    "wait_5s": (
        "realized_wait_5s_bps"
    ),
    "wait_10s": (
        "realized_wait_10s_bps"
    ),
}


CALIBRATION_BINS = [
    -np.inf,
    0.00,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    np.inf,
]


CALIBRATION_LABELS = [
    "<=0",
    "0-0.05",
    "0.05-0.10",
    "0.10-0.25",
    "0.25-0.50",
    "0.50-1.00",
    ">1.00",
]


def validate_input(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "security",
        "side",
        "current_cost_bps",
        "chosen_action",
        "realized_policy_cost_bps",
    }

    required_columns.update(
        PREDICTED_WAIT_COLUMNS.values()
    )

    required_columns.update(
        REALIZED_WAIT_COLUMNS.values()
    )

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Decision file is empty."
        )


def add_predicted_opportunity_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    actions = list(
        PREDICTED_WAIT_COLUMNS.keys()
    )

    predicted_matrix = np.column_stack(
        [
            result[
                PREDICTED_WAIT_COLUMNS[
                    action
                ]
            ]
            .to_numpy(
                dtype=float
            )
            for action in actions
        ]
    )

    realized_matrix = np.column_stack(
        [
            result[
                REALIZED_WAIT_COLUMNS[
                    action
                ]
            ]
            .to_numpy(
                dtype=float
            )
            for action in actions
        ]
    )

    current = (
        result[
            "current_cost_bps"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        predicted_matrix
    ).all():
        raise ValueError(
            "Predicted wait costs contain "
            "non-finite values."
        )

    if not np.isfinite(
        realized_matrix
    ).all():
        raise ValueError(
            "Realized wait costs contain "
            "non-finite values."
        )

    best_position = np.argmin(
        predicted_matrix,
        axis=1,
    )

    row_index = np.arange(
        len(
            result
        )
    )

    best_predicted_wait_cost = (
        predicted_matrix[
            row_index,
            best_position,
        ]
    )

    realized_cost_for_predicted_horizon = (
        realized_matrix[
            row_index,
            best_position,
        ]
    )

    predicted_advantage = (
        current
        - best_predicted_wait_cost
    )

    realized_savings_if_wait = (
        current
        - realized_cost_for_predicted_horizon
    )

    result[
        "predicted_best_wait_action"
    ] = [
        actions[
            int(
                position
            )
        ]
        for position
        in best_position
    ]

    result[
        "predicted_best_wait_cost_bps"
    ] = (
        best_predicted_wait_cost
    )

    result[
        "predicted_advantage_bps"
    ] = (
        predicted_advantage
    )

    result[
        "realized_cost_for_predicted_"
        "horizon_bps"
    ] = (
        realized_cost_for_predicted_horizon
    )

    result[
        "realized_savings_if_predicted_"
        "wait_bps"
    ] = (
        realized_savings_if_wait
    )

    result[
        "advantage_bin"
    ] = pd.cut(
        result[
            "predicted_advantage_bps"
        ],
        bins=CALIBRATION_BINS,
        labels=CALIBRATION_LABELS,
        right=True,
        include_lowest=True,
    )

    return result


def calibration_summary(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    summary = (
        df.groupby(
            group_columns,
            observed=True,
            as_index=False,
        )
        .agg(
            rows=(
                "predicted_advantage_bps",
                "size",
            ),
            mean_predicted_advantage_bps=(
                "predicted_advantage_bps",
                "mean",
            ),
            median_predicted_advantage_bps=(
                "predicted_advantage_bps",
                "median",
            ),
            mean_realized_savings_if_wait_bps=(
                "realized_savings_if_predicted_wait_bps",
                "mean",
            ),
            median_realized_savings_if_wait_bps=(
                "realized_savings_if_predicted_wait_bps",
                "median",
            ),
            realized_wait_hit_rate=(
                "realized_savings_if_predicted_wait_bps",
                lambda x: float(
                    (
                        x > 0
                    ).mean()
                ),
            ),
            actual_policy_wait_rate=(
                "chosen_action",
                lambda x: float(
                    (
                        x != "execute_now"
                    ).mean()
                ),
            ),
        )
    )

    summary[
        "calibration_gap_bps"
    ] = (
        summary[
            "mean_predicted_advantage_bps"
        ]
        - summary[
            "mean_realized_savings_if_wait_bps"
        ]
    )

    return summary


def build_horizon_diagnostics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for action in (
        "wait_1s",
        "wait_5s",
        "wait_10s",
    ):
        predicted_col = (
            PREDICTED_WAIT_COLUMNS[
                action
            ]
        )

        realized_col = (
            REALIZED_WAIT_COLUMNS[
                action
            ]
        )

        action_rows = (
            df.loc[
                df[
                    "chosen_action"
                ]
                == action
            ]
            .copy()
        )

        if action_rows.empty:
            continue

        action_rows[
            "predicted_advantage_for_action_bps"
        ] = (
            action_rows[
                "current_cost_bps"
            ]
            - action_rows[
                predicted_col
            ]
        )

        action_rows[
            "realized_savings_for_action_bps"
        ] = (
            action_rows[
                "current_cost_bps"
            ]
            - action_rows[
                realized_col
            ]
        )

        action_rows[
            "cost_over_now_bps"
        ] = (
            -action_rows[
                "realized_savings_for_action_bps"
            ]
        )

        for (
            security,
            side,
        ), group in (
            action_rows.groupby(
                [
                    "security",
                    "side",
                ],
                sort=True,
            )
        ):
            rows.append(
                {
                    "security": (
                        security
                    ),
                    "side": side,
                    "chosen_action": action,
                    "rows": len(
                        group
                    ),
                    "mean_predicted_advantage_bps": (
                        group[
                            "predicted_advantage_for_action_bps"
                        ].mean()
                    ),
                    "mean_realized_savings_bps": (
                        group[
                            "realized_savings_for_action_bps"
                        ].mean()
                    ),
                    "median_realized_savings_bps": (
                        group[
                            "realized_savings_for_action_bps"
                        ].median()
                    ),
                    "hit_rate": float(
                        (
                            group[
                                "realized_savings_for_action_bps"
                            ]
                            > 0
                        ).mean()
                    ),
                    "p95_cost_over_now_bps": (
                        group[
                            "cost_over_now_bps"
                        ].quantile(
                            0.95
                        )
                    ),
                    "p99_cost_over_now_bps": (
                        group[
                            "cost_over_now_bps"
                        ].quantile(
                            0.99
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "security",
                "side",
                "chosen_action",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: "
            f"{INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    validate_input(
        df
    )

    diagnostics = (
        add_predicted_opportunity_columns(
            df
        )
    )

    CALIBRATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_calibration = (
        calibration_summary(
            diagnostics,
            [
                "advantage_bin",
            ],
        )
    )

    overall_calibration.to_csv(
        CALIBRATION_OUTPUT_PATH,
        index=False,
    )

    security_calibration = (
        calibration_summary(
            diagnostics,
            [
                "security",
                "advantage_bin",
            ],
        )
    )

    security_calibration.to_csv(
        SECURITY_CALIBRATION_OUTPUT_PATH,
        index=False,
    )

    horizon_diagnostics = (
        build_horizon_diagnostics(
            diagnostics
        )
    )

    horizon_diagnostics.to_csv(
        HORIZON_OUTPUT_PATH,
        index=False,
    )

    action_diagnostics = (
        diagnostics.groupby(
            "chosen_action",
            as_index=False,
        )
        .agg(
            rows=(
                "chosen_action",
                "size",
            ),
            mean_policy_savings_bps=(
                "savings_vs_now_bps",
                "mean",
            ),
            median_policy_savings_bps=(
                "savings_vs_now_bps",
                "median",
            ),
            positive_savings_rate=(
                "savings_vs_now_bps",
                lambda x: float(
                    (
                        x > 0
                    ).mean()
                ),
            ),
            negative_savings_rate=(
                "savings_vs_now_bps",
                lambda x: float(
                    (
                        x < 0
                    ).mean()
                ),
            ),
        )
    )

    action_diagnostics[
        "action_rate"
    ] = (
        action_diagnostics[
            "rows"
        ]
        / len(
            diagnostics
        )
    )

    action_diagnostics.to_csv(
        ACTION_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nPredicted-advantage "
        "calibration"
    )

    print(
        overall_calibration.to_string(
            index=False
        )
    )

    print(
        "\nChosen-horizon diagnostics"
    )

    print(
        horizon_diagnostics.to_string(
            index=False
        )
    )

    print(
        "\nOverall chosen-action "
        "diagnostics"
    )

    print(
        action_diagnostics.to_string(
            index=False
        )
    )

    print(
        "\nInterpretation:"
    )

    print(
        "A useful policy should generally "
        "show larger realized savings and "
        "higher hit rates as predicted "
        "advantage increases."
    )

    print(
        "Large positive calibration_gap_bps "
        "means predicted economic advantage "
        "is larger than realized advantage."
    )

    print(
        "Horizon diagnostics show whether "
        "1s, 5s, or 10s waiting decisions "
        "actually create value when selected."
    )

    print(
        "\nSaved:"
    )

    for output_path in (
        CALIBRATION_OUTPUT_PATH,
        SECURITY_CALIBRATION_OUTPUT_PATH,
        HORIZON_OUTPUT_PATH,
        ACTION_OUTPUT_PATH,
    ):
        print(
            output_path
        )


if __name__ == "__main__":
    main()