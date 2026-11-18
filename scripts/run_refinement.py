"""Tune LightGBM on CPU and improved BiLSTM configurations on MPS."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from sleep_staging.refinement import compare_families, run_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/refinement.yaml")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("var/research/sc-w30-v2/campaign.json")
    )
    parser.add_argument(
        "--neural-reference",
        type=Path,
        default=Path("var/research/sc-w30-lstm-mps-v1/campaign.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("var/research/sc-w30-refinement-v1")
    )
    parser.add_argument("--family", choices=["lgbm", "bilstm"])
    parser.add_argument(
        "--mode", choices=["benchmark", "run", "compare"], default="run"
    )
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare_families(
            args.output_root.resolve(), args.reference, args.neural_reference
        )
        print(
            json.dumps(
                {"winner": result["winner"], "leaderboard": result["leaderboard"]},
                indent=2,
            )
        )
    else:
        if args.family is None:
            parser.error("--family required for benchmark/run")
        result = run_family(
            args.config, args.reference, args.output_root, args.family, args.mode
        )
        print(
            json.dumps(
                {k: result[k] for k in ["family", "identity", "mlflow_run_id"]},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
