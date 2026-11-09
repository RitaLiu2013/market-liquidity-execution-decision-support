from __future__ import annotations

import argparse
from pathlib import Path

from execution_ml.execution_cost import add_execution_costs
from execution_ml.features import add_microstructure_features
from execution_ml.lobster import load_lobster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--message",
        required=True,
        help="Path to LOBSTER message CSV.",
    )

    parser.add_argument(
        "--orderbook",
        required=True,
        help="Path to LOBSTER order-book CSV.",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--quantity",
        type=float,
        default=1000,
    )

    parser.add_argument(
        "--output",
        default="data/processed/execution_features.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_lobster(
        message_path=args.message,
        orderbook_path=args.orderbook,
        levels=args.levels,
    )

    df = add_microstructure_features(
        df,
        levels=args.levels,
    )

    df = add_execution_costs(
        df,
        quantity=args.quantity,
        levels=args.levels,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
          output_path,
          index=False,
          compression="zstd",)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Saved to: {output_path}")

    suffix = int(args.quantity)

    print(
        df[
            [
                "time",
                "mid",
                "spread_bps",
                "top_imbalance",
                f"buy_cost_bps_{suffix}",
                f"sell_cost_bps_{suffix}",
            ]
        ]
        .dropna()
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()