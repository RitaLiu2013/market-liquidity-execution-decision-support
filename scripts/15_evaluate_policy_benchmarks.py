from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "outputs/final_policy_test_decisions.parquet"
)

GROUP_OUTPUT_PATH = Path(
    "outputs/policy_benchmark_group_metrics.csv"
)

OVERALL_OUTPUT_PATH = Path(
    "outputs/policy_benchmark_overall_summary.csv"
)

SECURITY_OUTPUT_PATH = Path(
    "outputs/policy_benchmark_security_summary.csv"
)

ML_COMPARISON_OUTPUT_PATH = Path(
    "outputs/policy_benchmark_ml_comparison.csv"
)


POLICY_COST_COLUMNS = {
    "always_execute_now": (
        "current_cost_bps"
    ),
    "always_wait_1s": (
        "realized_wait_1s_bps"
    ),
    "always_wait_5s": (
        "realized_wait_5s_bps"
    ),
    "always_wait_10s": (
        "realized_wait_10s_bps"
    ),
    "frozen_ml_policy": (
        "realized_policy_cost_bps"
    ),
    "oracle": (
        "oracle_cost_bps"
    ),
}


STATIC_POLICIES = (
    "always_execute_now",
    "always_wait_1s",
    "always_wait_5s",
    "always_wait_10s",
)


def validate_input(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "security",
        "side",
        "current_cost_bps",
        "realized_wait_1s_bps",
        "realized_wait_5s_bps",
        "realized_wait_10s_bps",
        "realized_policy_cost_bps",
        "oracle_cost_bps",
    }

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

    numeric_columns = [
        POLICY_COST_COLUMNS[
            policy
        ]
        for policy
        in POLICY_COST_COLUMNS
    ]

    values = (
        df[
            numeric_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Policy-cost columns contain "
            "non-finite values."
        )


def summarise_policy(
    group: pd.DataFrame,
    policy_name: str,
    cost_column: str,
) -> dict[str, object]:
    current = (
        group[
            "current_cost_bps"
        ]
        .to_numpy(
            dtype=float
        )
    )

    policy_cost = (
        group[
            cost_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    savings = (
        current
        - policy_cost
    )

    cost_over_now = (
        policy_cost
        - current
    )

    return {
        "policy": policy_name,
        "rows": len(
            group
        ),
        "mean_cost_bps": float(
            np.mean(
                policy_cost
            )
        ),
        "mean_savings_vs_now_bps": float(
            np.mean(
                savings
            )
        ),
        "median_savings_vs_now_bps": float(
            np.median(
                savings
            )
        ),
        "positive_savings_rate": float(
            np.mean(
                savings > 0
            )
        ),
        "negative_savings_rate": float(
            np.mean(
                savings < 0
            )
        ),
        "p95_cost_over_now_bps": float(
            np.quantile(
                cost_over_now,
                0.95,
            )
        ),
        "p99_cost_over_now_bps": float(
            np.quantile(
                cost_over_now,
                0.99,
            )
        ),
    }


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

    group_rows = []

    for (
        security,
        side,
    ), group in df.groupby(
        [
            "security",
            "side",
        ],
        sort=True,
    ):
        for (
            policy_name,
            cost_column,
        ) in (
            POLICY_COST_COLUMNS.items()
        ):
            metrics = (
                summarise_policy(
                    group,
                    policy_name,
                    cost_column,
                )
            )

            group_rows.append(
                {
                    "security": (
                        security
                    ),
                    "side": side,
                    **metrics,
                }
            )

    group_metrics = (
        pd.DataFrame(
            group_rows
        )
        .sort_values(
            [
                "security",
                "side",
                "policy",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    GROUP_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_metrics.to_csv(
        GROUP_OUTPUT_PATH,
        index=False,
    )

    overall_summary = (
        group_metrics.groupby(
            "policy",
            as_index=False,
        )
        .agg(
            group_weighted_mean_cost_bps=(
                "mean_cost_bps",
                "mean",
            ),
            group_weighted_mean_savings_vs_now_bps=(
                "mean_savings_vs_now_bps",
                "mean",
            ),
            median_group_savings_vs_now_bps=(
                "mean_savings_vs_now_bps",
                "median",
            ),
            positive_savings_groups=(
                "mean_savings_vs_now_bps",
                lambda x: int(
                    (
                        x > 0
                    ).sum()
                ),
            ),
            worst_group_savings_bps=(
                "mean_savings_vs_now_bps",
                "min",
            ),
            best_group_savings_bps=(
                "mean_savings_vs_now_bps",
                "max",
            ),
            mean_p95_cost_over_now_bps=(
                "p95_cost_over_now_bps",
                "mean",
            ),
            mean_p99_cost_over_now_bps=(
                "p99_cost_over_now_bps",
                "mean",
            ),
        )
    )

    # Add observation-weighted metrics as a
    # secondary robustness view.
    observation_rows = []

    for (
        policy_name,
        cost_column,
    ) in POLICY_COST_COLUMNS.items():
        current = (
            df[
                "current_cost_bps"
            ]
            .to_numpy(
                dtype=float
            )
        )

        policy_cost = (
            df[
                cost_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        savings = (
            current
            - policy_cost
        )

        observation_rows.append(
            {
                "policy": (
                    policy_name
                ),
                "observation_weighted_"
                "mean_cost_bps": float(
                    np.mean(
                        policy_cost
                    )
                ),
                "observation_weighted_"
                "mean_savings_vs_now_bps": (
                    float(
                        np.mean(
                            savings
                        )
                    )
                ),
            }
        )

    observation_summary = (
        pd.DataFrame(
            observation_rows
        )
    )

    overall_summary = (
        overall_summary.merge(
            observation_summary,
            on="policy",
            how="left",
            validate="one_to_one",
        )
    )

    overall_summary = (
        overall_summary.sort_values(
            "group_weighted_mean_cost_bps"
        )
        .reset_index(
            drop=True
        )
    )

    overall_summary.to_csv(
        OVERALL_OUTPUT_PATH,
        index=False,
    )

    security_summary = (
        group_metrics.groupby(
            [
                "security",
                "policy",
            ],
            as_index=False,
        )
        .agg(
            mean_cost_bps=(
                "mean_cost_bps",
                "mean",
            ),
            mean_savings_vs_now_bps=(
                "mean_savings_vs_now_bps",
                "mean",
            ),
            worst_side_savings_bps=(
                "mean_savings_vs_now_bps",
                "min",
            ),
        )
        .sort_values(
            [
                "security",
                "mean_cost_bps",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    security_summary.to_csv(
        SECURITY_OUTPUT_PATH,
        index=False,
    )

    # Directly compare frozen ML to every
    # fixed static policy.
    ml_rows = (
        group_metrics.loc[
            group_metrics[
                "policy"
            ]
            == "frozen_ml_policy",
            [
                "security",
                "side",
                "mean_cost_bps",
            ],
        ]
        .rename(
            columns={
                "mean_cost_bps":
                    "ml_mean_cost_bps",
            }
        )
    )

    comparison_rows = []

    for static_policy in (
        STATIC_POLICIES
    ):
        static_rows = (
            group_metrics.loc[
                group_metrics[
                    "policy"
                ]
                == static_policy,
                [
                    "security",
                    "side",
                    "mean_cost_bps",
                ],
            ]
            .rename(
                columns={
                    "mean_cost_bps":
                        "benchmark_mean_cost_bps",
                }
            )
        )

        comparison = (
            ml_rows.merge(
                static_rows,
                on=[
                    "security",
                    "side",
                ],
                how="inner",
                validate="one_to_one",
            )
        )

        comparison[
            "benchmark_policy"
        ] = static_policy

        # Positive means ML has lower
        # realized execution cost.
        comparison[
            "ml_advantage_bps"
        ] = (
            comparison[
                "benchmark_mean_cost_bps"
            ]
            - comparison[
                "ml_mean_cost_bps"
            ]
        )

        comparison_rows.append(
            comparison
        )

    ml_comparison = pd.concat(
        comparison_rows,
        ignore_index=True,
    )

    ml_comparison.to_csv(
        ML_COMPARISON_OUTPUT_PATH,
        index=False,
    )

    ml_comparison_summary = (
        ml_comparison.groupby(
            "benchmark_policy",
            as_index=False,
        )
        .agg(
            mean_ml_advantage_bps=(
                "ml_advantage_bps",
                "mean",
            ),
            median_ml_advantage_bps=(
                "ml_advantage_bps",
                "median",
            ),
            groups_ml_beats_benchmark=(
                "ml_advantage_bps",
                lambda x: int(
                    (
                        x > 0
                    ).sum()
                ),
            ),
            worst_ml_advantage_bps=(
                "ml_advantage_bps",
                "min",
            ),
            best_ml_advantage_bps=(
                "ml_advantage_bps",
                "max",
            ),
        )
    )

    print(
        "\nStatic policy benchmark "
        "overall summary"
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    print(
        "\nML policy versus each "
        "static benchmark"
    )

    print(
        ml_comparison_summary.to_string(
            index=False
        )
    )

    print(
        "\nSecurity-level benchmark "
        "summary"
    )

    print(
        security_summary.to_string(
            index=False
        )
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "The best fixed policy observed "
        "on test is descriptive only. "
        "Do not use test results to "
        "replace or retune the frozen "
        "policy."
    )

    print(
        "\nSaved:"
    )

    for output_path in (
        GROUP_OUTPUT_PATH,
        OVERALL_OUTPUT_PATH,
        SECURITY_OUTPUT_PATH,
        ML_COMPARISON_OUTPUT_PATH,
    ):
        print(
            output_path
        )


if __name__ == "__main__":
    main()