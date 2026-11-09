from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execution_ml.final_model_spec import (
    FINAL_FEATURES,
    HORIZONS,
)


DATA_ROOT = Path("data/processed")

DECISION_INPUT_PATH = Path(
    "outputs/final_policy_test_decisions.parquet"
)

GROUP_OUTPUT_PATH = Path(
    "outputs/deployable_fallback_group_metrics.csv"
)

SECURITY_OUTPUT_PATH = Path(
    "outputs/deployable_fallback_security_summary.csv"
)

OVERALL_OUTPUT_PATH = Path(
    "outputs/deployable_fallback_overall_summary.csv"
)

DECISION_OUTPUT_PATH = Path(
    "outputs/deployable_fallback_decisions.parquet"
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
    return (
        f"wait_{side}_cost_bps_"
        f"{horizon_suffix(horizon)}s"
    )


def current_cost_column(
    side: str,
    quantity: int,
) -> str:
    return (
        f"{side}_cost_bps_{quantity}"
    )


def finite_mask(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    non_missing = (
        frame[
            columns
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result = pd.Series(
        False,
        index=frame.index,
    )

    if not non_missing.any():
        return result

    values = (
        frame.loc[
            non_missing,
            columns,
        ]
        .to_numpy(
            dtype=float
        )
    )

    finite = np.isfinite(
        values
    ).all(
        axis=1
    )

    result.loc[
        non_missing
    ] = finite

    return result


def validate_decisions(
    decisions: pd.DataFrame,
) -> None:
    required = {
        "security",
        "side",
        "source_index",
        "realized_policy_cost_bps",
        "chosen_action",
    }

    missing = (
        required
        - set(
            decisions.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing decision columns: "
            f"{sorted(missing)}"
        )

    key_columns = [
        "security",
        "side",
        "source_index",
    ]

    if decisions.duplicated(
        key_columns
    ).any():
        raise ValueError(
            "Duplicate decision keys found."
        )


def validate_dataset(
    df: pd.DataFrame,
    path: Path,
) -> tuple[str, int]:
    required = {
        "security",
        "calibrated_quantity",
        "split",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path.name}: missing "
            f"{sorted(missing)}"
        )

    securities = (
        df[
            "security"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    quantities = (
        df[
            "calibrated_quantity"
        ]
        .dropna()
        .unique()
    )

    if len(
        securities
    ) != 1:
        raise ValueError(
            f"{path.name}: expected one "
            "security."
        )

    if len(
        quantities
    ) != 1:
        raise ValueError(
            f"{path.name}: expected one "
            "calibrated quantity."
        )

    return (
        str(
            securities[0]
        ),
        int(
            quantities[0]
        ),
    )


def main() -> None:
    if not DECISION_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Step 14 decisions: "
            f"{DECISION_INPUT_PATH}"
        )

    decisions = pd.read_parquet(
        DECISION_INPUT_PATH
    )

    validate_decisions(
        decisions
    )

    paths = sorted(
        DATA_ROOT.glob(
            "*_predictive.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No predictive datasets found."
        )

    group_rows = []
    deployable_frames = []

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

        print(
            f"\nProcessing {security}"
        )

        missing_features = [
            column
            for column
            in FINAL_FEATURES
            if column
            not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"{security}: missing final "
                f"features {missing_features}"
            )

        test = (
            df.loc[
                df[
                    "split"
                ]
                == "test"
            ]
            .copy()
        )

        if test.empty:
            raise ValueError(
                f"{security}: empty test split."
            )

        for side in (
            "buy",
            "sell",
        ):
            current_col = (
                current_cost_column(
                    side,
                    quantity,
                )
            )

            target_cols = [
                target_column(
                    side,
                    float(
                        horizon
                    ),
                )
                for horizon
                in HORIZONS
            ]

            outcome_columns = [
                current_col,
                *target_cols,
            ]

            missing_outcomes = [
                column
                for column
                in outcome_columns
                if column
                not in test.columns
            ]

            if missing_outcomes:
                raise ValueError(
                    f"{security} {side}: "
                    f"missing outcome columns "
                    f"{missing_outcomes}"
                )

            # Offline evaluation universe:
            # every action must have a known
            # realized outcome.
            outcome_available = finite_mask(
                test,
                outcome_columns,
            )

            universe = (
                test.loc[
                    outcome_available
                ]
                .copy()
            )

            if universe.empty:
                raise ValueError(
                    f"{security} {side}: "
                    "empty outcome-evaluable "
                    "test universe."
                )

            universe[
                "source_index"
            ] = universe.index

            feature_available = finite_mask(
                universe,
                list(
                    FINAL_FEATURES
                ),
            )

            universe[
                "feature_available"
            ] = (
                feature_available
            )

            universe[
                "security"
            ] = security

            universe[
                "side"
            ] = side

            universe[
                "quantity"
            ] = quantity

            group_decisions = (
                decisions.loc[
                    (
                        decisions[
                            "security"
                        ]
                        == security
                    )
                    & (
                        decisions[
                            "side"
                        ]
                        == side
                    ),
                    [
                        "security",
                        "side",
                        "source_index",
                        "realized_policy_cost_bps",
                        "chosen_action",
                    ],
                ]
                .copy()
            )

            universe = (
                universe.merge(
                    group_decisions,
                    on=[
                        "security",
                        "side",
                        "source_index",
                    ],
                    how="left",
                    validate="one_to_one",
                )
            )

            covered_without_policy = (
                universe[
                    "feature_available"
                ]
                & universe[
                    "realized_policy_cost_bps"
                ].isna()
            )

            if (
                covered_without_policy.any()
            ):
                raise ValueError(
                    f"{security} {side}: "
                    "F5-available rows are "
                    "missing Step 14 policy "
                    "decisions."
                )

            decision_on_uncovered = (
                ~universe[
                    "feature_available"
                ]
                & universe[
                    "realized_policy_cost_bps"
                ].notna()
            )

            if (
                decision_on_uncovered.any()
            ):
                raise ValueError(
                    f"{security} {side}: "
                    "unexpected ML decisions "
                    "exist on F5-unavailable "
                    "rows."
                )

            current_cost = (
                universe[
                    current_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            realized_wait = (
                universe[
                    target_cols
                ]
                .to_numpy(
                    dtype=float
                )
            )

            all_realized = (
                np.column_stack(
                    [
                        current_cost,
                        realized_wait,
                    ]
                )
            )

            oracle_cost = (
                np.min(
                    all_realized,
                    axis=1,
                )
            )

            feature_mask = (
                universe[
                    "feature_available"
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            ml_policy_cost = (
                universe[
                    "realized_policy_cost_bps"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            deployable_cost = (
                current_cost.copy()
            )

            deployable_cost[
                feature_mask
            ] = (
                ml_policy_cost[
                    feature_mask
                ]
            )

            deployable_action = np.full(
                len(
                    universe
                ),
                "fallback_execute_now",
                dtype=object,
            )

            deployable_action[
                feature_mask
            ] = (
                universe.loc[
                    universe[
                        "feature_available"
                    ],
                    "chosen_action",
                ]
                .to_numpy()
            )

            savings_vs_now = (
                current_cost
                - deployable_cost
            )

            oracle_savings = (
                current_cost
                - oracle_cost
            )

            oracle_gap = (
                deployable_cost
                - oracle_cost
            )

            covered_savings = (
                savings_vs_now[
                    feature_mask
                ]
            )

            cost_over_now = (
                deployable_cost
                - current_cost
            )

            total_test_rows = len(
                test
            )

            universe_rows = len(
                universe
            )

            covered_rows = int(
                feature_mask.sum()
            )

            fallback_rows = (
                universe_rows
                - covered_rows
            )

            outcome_coverage = (
                universe_rows
                / total_test_rows
            )

            ml_coverage_within_universe = (
                covered_rows
                / universe_rows
            )

            conditional_ml_savings = (
                float(
                    covered_savings.mean()
                )
                if covered_rows > 0
                else np.nan
            )

            deployable_mean_savings = float(
                savings_vs_now.mean()
            )

            group_rows.append(
                {
                    "security": security,
                    "side": side,
                    "quantity": quantity,
                    "total_test_rows": (
                        total_test_rows
                    ),
                    "outcome_evaluable_rows": (
                        universe_rows
                    ),
                    "ml_covered_rows": (
                        covered_rows
                    ),
                    "fallback_rows": (
                        fallback_rows
                    ),
                    "outcome_coverage": (
                        outcome_coverage
                    ),
                    "ml_coverage_within_"
                    "evaluable_universe": (
                        ml_coverage_within_universe
                    ),
                    "conditional_ml_"
                    "savings_bps": (
                        conditional_ml_savings
                    ),
                    "deployable_mean_"
                    "savings_bps": (
                        deployable_mean_savings
                    ),
                    "mean_oracle_"
                    "savings_bps": float(
                        oracle_savings.mean()
                    ),
                    "mean_oracle_gap_bps": (
                        float(
                            oracle_gap.mean()
                        )
                    ),
                    "positive_savings_rate": (
                        float(
                            (
                                savings_vs_now
                                > 0
                            ).mean()
                        )
                    ),
                    "negative_savings_rate": (
                        float(
                            (
                                savings_vs_now
                                < 0
                            ).mean()
                        )
                    ),
                    "p95_cost_over_now_bps": (
                        float(
                            np.quantile(
                                cost_over_now,
                                0.95,
                            )
                        )
                    ),
                    "p99_cost_over_now_bps": (
                        float(
                            np.quantile(
                                cost_over_now,
                                0.99,
                            )
                        )
                    ),
                }
            )

            deployable = pd.DataFrame(
                {
                    "security": (
                        security
                    ),
                    "side": side,
                    "quantity": quantity,
                    "source_index": (
                        universe[
                            "source_index"
                        ].to_numpy()
                    ),
                    "feature_available": (
                        feature_mask
                    ),
                    "deployable_action": (
                        deployable_action
                    ),
                    "current_cost_bps": (
                        current_cost
                    ),
                    "deployable_policy_"
                    "cost_bps": (
                        deployable_cost
                    ),
                    "savings_vs_now_bps": (
                        savings_vs_now
                    ),
                    "oracle_cost_bps": (
                        oracle_cost
                    ),
                    "oracle_savings_"
                    "vs_now_bps": (
                        oracle_savings
                    ),
                    "oracle_gap_bps": (
                        oracle_gap
                    ),
                }
            )

            deployable_frames.append(
                deployable
            )

    group_metrics = (
        pd.DataFrame(
            group_rows
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

    all_deployable = pd.concat(
        deployable_frames,
        ignore_index=True,
    )

    GROUP_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_metrics.to_csv(
        GROUP_OUTPUT_PATH,
        index=False,
    )

    all_deployable.to_parquet(
        DECISION_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    security_summary = (
        group_metrics.groupby(
            "security",
            as_index=False,
        )
        .agg(
            mean_ml_coverage=(
                "ml_coverage_within_evaluable_universe",
                "mean",
            ),
            mean_conditional_ml_savings_bps=(
                "conditional_ml_savings_bps",
                "mean",
            ),
            mean_deployable_savings_bps=(
                "deployable_mean_savings_bps",
                "mean",
            ),
            mean_oracle_gap_bps=(
                "mean_oracle_gap_bps",
                "mean",
            ),
        )
    )

    security_summary.to_csv(
        SECURITY_OUTPUT_PATH,
        index=False,
    )

    group_weighted = {
        "security_side_groups": len(
            group_metrics
        ),
        "mean_outcome_coverage": float(
            group_metrics[
                "outcome_coverage"
            ].mean()
        ),
        "mean_ml_coverage_within_"
        "evaluable_universe": float(
            group_metrics[
                "ml_coverage_within_evaluable_universe"
            ].mean()
        ),
        "mean_conditional_ml_"
        "savings_bps": float(
            group_metrics[
                "conditional_ml_savings_bps"
            ].mean()
        ),
        "mean_deployable_"
        "savings_bps": float(
            group_metrics[
                "deployable_mean_savings_bps"
            ].mean()
        ),
        "positive_deployable_"
        "savings_groups": int(
            (
                group_metrics[
                    "deployable_mean_savings_bps"
                ]
                > 0
            ).sum()
        ),
        "worst_group_deployable_"
        "savings_bps": float(
            group_metrics[
                "deployable_mean_savings_bps"
            ].min()
        ),
        "best_group_deployable_"
        "savings_bps": float(
            group_metrics[
                "deployable_mean_savings_bps"
            ].max()
        ),
        "mean_oracle_gap_bps": float(
            group_metrics[
                "mean_oracle_gap_bps"
            ].mean()
        ),
    }

    observation_weighted = {
        "observation_weighted_"
        "ml_coverage": float(
            all_deployable[
                "feature_available"
            ].mean()
        ),
        "observation_weighted_"
        "deployable_savings_bps": float(
            all_deployable[
                "savings_vs_now_bps"
            ].mean()
        ),
        "observation_weighted_"
        "oracle_gap_bps": float(
            all_deployable[
                "oracle_gap_bps"
            ].mean()
        ),
    }

    overall_summary = pd.DataFrame(
        [
            {
                **group_weighted,
                **observation_weighted,
            }
        ]
    )

    overall_summary.to_csv(
        OVERALL_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nDeployable fallback "
        "metrics by security / side"
    )

    print(
        group_metrics.to_string(
            index=False
        )
    )

    print(
        "\nSecurity summary"
    )

    print(
        security_summary.to_string(
            index=False
        )
    )

    print(
        "\nOverall deployable "
        "fallback summary"
    )

    for key, value in (
        overall_summary.iloc[
            0
        ].items()
    ):
        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
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
        "conditional_ml_savings_bps "
        "measures the frozen ML policy "
        "only where F5 is available."
    )

    print(
        "deployable_mean_savings_bps "
        "includes fallback execute-now "
        "decisions where F5 is unavailable."
    )

    print(
        "This is still an offline "
        "outcome-evaluable universe: "
        "future action outcomes must be "
        "observable for evaluation."
    )

    print(
        "\nSaved:"
    )

    for output_path in (
        GROUP_OUTPUT_PATH,
        SECURITY_OUTPUT_PATH,
        OVERALL_OUTPUT_PATH,
        DECISION_OUTPUT_PATH,
    ):
        print(
            output_path
        )


if __name__ == "__main__":
    main()