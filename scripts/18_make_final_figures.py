from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path(
    "outputs/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_figure(
    filename: str,
) -> None:
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_feature_ablation() -> None:
    df = pd.read_csv(
        "outputs/"
        "feature_ablation_summary.csv"
    )

    x = (
        df[
            "feature_set"
        ]
        .astype(str)
    )

    y = (
        df[
            "mean_improvement_"
            "vs_persistence_pct"
        ]
    )

    plt.figure(
        figsize=(
            10,
            5,
        )
    )

    plt.bar(
        x,
        y,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Mean MAE improvement "
        "vs persistence (%)"
    )

    plt.xlabel(
        "Feature set"
    )

    plt.title(
        "Validation feature ablation"
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    save_figure(
        "01_feature_ablation.png"
    )


def plot_final_test_security() -> None:
    df = pd.read_csv(
        "outputs/"
        "final_test_security_summary.csv"
    )

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.bar(
        df[
            "security"
        ],
        df[
            "mean_improvement_"
            "vs_persistence_pct"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Mean MAE improvement "
        "vs persistence (%)"
    )

    plt.xlabel(
        "Security"
    )

    plt.title(
        "Frozen-model out-of-sample "
        "forecast performance"
    )

    save_figure(
        "02_final_test_by_security.png"
    )


def plot_policy_benchmarks() -> None:
    df = pd.read_csv(
        "outputs/"
        "policy_benchmark_overall_summary.csv"
    )

    order = [
        "always_execute_now",
        "always_wait_1s",
        "always_wait_5s",
        "always_wait_10s",
        "frozen_ml_policy",
    ]

    df = (
        df.loc[
            df[
                "policy"
            ].isin(
                order
            )
        ]
        .set_index(
            "policy"
        )
        .loc[
            order
        ]
        .reset_index()
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        df[
            "policy"
        ],
        df[
            "group_weighted_mean_"
            "savings_vs_now_bps"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Mean savings vs "
        "execute now (bps)"
    )

    plt.xlabel(
        "Policy"
    )

    plt.title(
        "Dynamic ML policy vs "
        "fixed timing rules"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    save_figure(
        "03_policy_benchmarks.png"
    )


def plot_policy_security() -> None:
    df = pd.read_csv(
        "outputs/"
        "final_policy_test_security_summary.csv"
    )

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.bar(
        df[
            "security"
        ],
        df[
            "mean_savings_"
            "vs_now_bps"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Mean realized savings "
        "(bps)"
    )

    plt.xlabel(
        "Security"
    )

    plt.title(
        "Frozen execution policy "
        "by security"
    )

    save_figure(
        "04_policy_savings_by_security.png"
    )


def plot_calibration() -> None:
    df = pd.read_csv(
        "outputs/"
        "policy_advantage_calibration.csv"
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.plot(
        df[
            "advantage_bin"
        ],
        df[
            "mean_realized_"
            "savings_if_wait_bps"
        ],
        marker="o",
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Mean realized savings "
        "if waiting (bps)"
    )

    plt.xlabel(
        "Predicted advantage bin"
    )

    plt.title(
        "Predicted economic advantage "
        "vs realized savings"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    save_figure(
        "05_policy_calibration.png"
    )


def plot_deployable_coverage() -> None:
    df = pd.read_csv(
        "outputs/"
        "deployable_fallback_security_summary.csv"
    )

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.bar(
        df[
            "security"
        ],
        df[
            "mean_deployable_"
            "savings_bps"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Coverage-adjusted "
        "savings (bps)"
    )

    plt.xlabel(
        "Security"
    )

    plt.title(
        "Deployable ML policy "
        "with execute-now fallback"
    )

    save_figure(
        "06_deployable_savings.png"
    )


def main() -> None:
    plot_feature_ablation()

    plot_final_test_security()

    plot_policy_benchmarks()

    plot_policy_security()

    plot_calibration()

    plot_deployable_coverage()

    print(
        "\nSaved final figures to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()