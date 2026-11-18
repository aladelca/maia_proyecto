"""Run the planned feature-sequence LSTM study on MPS without CPU fallback."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from sleep_staging.lstm_research import run_lstm_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/lstm-mps.yaml")
    )
    parser.add_argument(
        "--reference-campaign",
        type=Path,
        default=Path("var/research/sc-w30-v2/campaign.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("var/research/sc-w30-lstm-mps-v1")
    )
    parser.add_argument(
        "--mode", choices=["preflight", "benchmark", "run"], default="run"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_lstm_campaign(
        args.config, args.reference_campaign, args.output_root, args.mode, args.resume
    )
    print(
        json.dumps(
            result
            if args.mode != "run"
            else {
                k: result[k]
                for k in [
                    "overall_winner",
                    "best_neural",
                    "delta_cv_f1",
                    "mlflow_parent_run_id",
                    "experiment_id",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
