from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execution_ml.execution_cost import add_execution_metrics
from execution_ml.lobster import load_lobster
from execution_ml.targets import add_future_execution_targets


DATA_ROOT = Path("data/raw")

SPECIFICATION_PATH = Path(
    "outputs/model_specification.csv"
)

TARGET_OUTPUT_PATH = Path(
    "outputs/predictive_target_validation.csv"
)

SPLIT_OUTPUT_PATH = Path(
    "outputs/predictive_split_summary.csv"
)

LEVELS = 10

HORIZONS = (
    1.0,
    5.0,
    10.0,
)

MAX_HORIZON = max(HORIZONS)


def resolve_file(
    ticker: str,
    kind: str,
) -> Path:
    direct = (
        DATA_ROOT
        / f"{ticker}_{kind}.csv"
    )

    if direct.exists():
        return direct

    matches = list(
        DATA_ROOT.rglob(
            f"{ticker}_*_{kind}_10.csv"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {kind} file found for {ticker}"
        )

    raise RuntimeError(
        f"Multiple {kind} files found "
        f"for {ticker}: {matches}"
    )


def orderbook_columns(
    levels: int,
) -> list[str]:
    columns = ["time"]

    for level in range(
        1,
        levels + 1,
    ):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )

    return columns


def assign_split(
    time: pd.Series,
    train_end: float,
    validation_end: float,
) -> pd.Series:
    """
    Assign chronological train/validation/test splits.

    A purge equal to the maximum target horizon is
    applied before the train/validation and
    validation/test boundaries so that future labels
    cannot cross into the next evaluation period.
    """

    values = time.to_numpy(
        dtype=float
    )

    split = np.full(
        len(values),
        "purged",
        dtype=object,
    )

    train_mask = (
        values
        < train_end - MAX_HORIZON
    )

    validation_mask = (
        (values >= train_end)
        & (
            values
            < validation_end - MAX_HORIZON
        )
    )

    test_mask = (
        values >= validation_end
    )

    split[train_mask] = "train"
    split[validation_mask] = "validation"
    split[test_mask] = "test"

    return pd.Series(
        split,
        index=time.index,
        dtype="string",
    )


def horizon_suffix(
    horizon: float,
) -> str:
    return (
        str(float(horizon))
        .replace(".", "p")
    )


def summarise_target(
    df: pd.DataFrame,
    ticker: str,
    quantity: int,
    horizon: float,
    side: str,
    split_name: str,
) -> dict[str, float | int | str]:
    suffix = horizon_suffix(
        horizon
    )

    staleness_col = (
        f"state_staleness_{suffix}s"
    )

    target_col = (
        f"wait_{side}_cost_bps_"
        f"{suffix}s"
    )

    observations = len(df)

    state_staleness = (
        df[staleness_col]
        .dropna()
    )

    target_available = (
        df[target_col]
        .notna()
    )

    state_count = len(
        state_staleness
    )

    target_count = int(
        target_available.sum()
    )

    staleness_ms = (
        state_staleness
        * 1000.0
    )

    if observations > 0:
        state_available_rate = (
            state_count
            / observations
        )

        target_availability = (
            target_count
            / observations
        )
    else:
        state_available_rate = np.nan
        target_availability = np.nan

    if state_count > 0:
        target_given_state = (
            target_count
            / state_count
        )

        median_staleness_ms = (
            staleness_ms.median()
        )

        p95_staleness_ms = (
            staleness_ms.quantile(
                0.95
            )
        )

        p99_staleness_ms = (
            staleness_ms.quantile(
                0.99
            )
        )
    else:
        target_given_state = np.nan
        median_staleness_ms = np.nan
        p95_staleness_ms = np.nan
        p99_staleness_ms = np.nan

    return {
        "security": ticker,
        "quantity": quantity,
        "horizon_seconds": horizon,
        "side": side,
        "split": split_name,
        "observations": observations,
        "state_available_rate": (
            state_available_rate
        ),
        "target_availability": (
            target_availability
        ),
        "target_availability_given_state": (
            target_given_state
        ),
        "median_state_staleness_ms": (
            median_staleness_ms
        ),
        "p95_state_staleness_ms": (
            p95_staleness_ms
        ),
        "p99_state_staleness_ms": (
            p99_staleness_ms
        ),
    }


def build_target_frame(
    ticker: str,
    quantity: int,
    train_end: float,
    validation_end: float,
) -> pd.DataFrame:
    raw = load_lobster(
        message_path=resolve_file(
            ticker,
            "message",
        ),
        orderbook_path=resolve_file(
            ticker,
            "orderbook",
        ),
        levels=LEVELS,
    )

    if not raw[
        "time"
    ].is_monotonic_increasing:
        raise ValueError(
            f"{ticker}: time is not "
            "monotonically increasing."
        )

    book = raw[
        orderbook_columns(
            LEVELS
        )
    ].copy()

    del raw

    book["mid"] = (
        book["ask_price_1"]
        + book["bid_price_1"]
    ) / 2.0

    evaluated = add_execution_metrics(
        book,
        quantity=quantity,
        levels=LEVELS,
    )

    del book

    targeted = add_future_execution_targets(
        evaluated,
        quantity=quantity,
        horizons=HORIZONS,
    )

    targeted["split"] = assign_split(
        time=targeted["time"],
        train_end=train_end,
        validation_end=validation_end,
    )

    return targeted


def main() -> None:
    specification = pd.read_csv(
        SPECIFICATION_PATH
    )

    required_spec_columns = {
        "security",
        "calibrated_quantity",
        "train_end_time",
        "validation_end_time",
    }

    missing_spec_columns = (
        required_spec_columns
        - set(specification.columns)
    )

    if missing_spec_columns:
        raise ValueError(
            "Missing model specification columns: "
            f"{sorted(missing_spec_columns)}"
        )

    validation_rows = []
    split_rows = []

    for row in specification.itertuples(
        index=False
    ):
        ticker = row.security

        quantity = int(
            row.calibrated_quantity
        )

        train_end = float(
            row.train_end_time
        )

        validation_end = float(
            row.validation_end_time
        )

        print(
            f"\nProcessing {ticker}"
        )

        print(
            f"  quantity={quantity:,}"
        )

        targeted = build_target_frame(
            ticker=ticker,
            quantity=quantity,
            train_end=train_end,
            validation_end=validation_end,
        )

        split_counts = (
            targeted["split"]
            .value_counts()
        )

        split_rows.append(
            {
                "security": ticker,
                "quantity": quantity,
                "total_observations": len(
                    targeted
                ),
                "train_rows": int(
                    split_counts.get(
                        "train",
                        0,
                    )
                ),
                "validation_rows": int(
                    split_counts.get(
                        "validation",
                        0,
                    )
                ),
                "test_rows": int(
                    split_counts.get(
                        "test",
                        0,
                    )
                ),
                "purged_rows": int(
                    split_counts.get(
                        "purged",
                        0,
                    )
                ),
                "train_end_time": train_end,
                "validation_end_time": (
                    validation_end
                ),
            }
        )

        analysis_frame = targeted.loc[
            targeted["split"] != "purged"
        ]

        for horizon in HORIZONS:
            for side in (
                "buy",
                "sell",
            ):
                validation_rows.append(
                    summarise_target(
                        df=analysis_frame,
                        ticker=ticker,
                        quantity=quantity,
                        horizon=horizon,
                        side=side,
                        split_name="overall",
                    )
                )

                for split_name in (
                    "train",
                    "validation",
                    "test",
                ):
                    subset = targeted.loc[
                        targeted["split"]
                        == split_name
                    ]

                    validation_rows.append(
                        summarise_target(
                            df=subset,
                            ticker=ticker,
                            quantity=quantity,
                            horizon=horizon,
                            side=side,
                            split_name=split_name,
                        )
                    )

        del targeted

    validation = pd.DataFrame(
        validation_rows
    )

    split_summary = pd.DataFrame(
        split_rows
    )

    TARGET_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation.to_csv(
        TARGET_OUTPUT_PATH,
        index=False,
    )

    split_summary.to_csv(
        SPLIT_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSplit summary"
    )

    print(
        split_summary.to_string(
            index=False
        )
    )

    overall = (
        validation.loc[
            validation["split"]
            == "overall"
        ]
        .copy()
    )

    display_columns = [
        "security",
        "horizon_seconds",
        "side",
        "state_available_rate",
        "target_availability",
        "target_availability_given_state",
        "median_state_staleness_ms",
        "p95_state_staleness_ms",
    ]

    print(
        "\nOverall target validation"
    )

    print(
        overall[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        TARGET_OUTPUT_PATH
    )

    print(
        SPLIT_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()