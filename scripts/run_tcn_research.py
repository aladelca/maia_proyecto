"""Run each of the three sleep-staging comparisons on MPS, CUDA or explicit CPU."""

import argparse
import json
import os
from pathlib import Path

import torch

from sleep_staging.models.tcn import research_device
from sleep_staging.tcn_research import run
from sleep_staging.tcn_training import Settings

ROOT = Path(__file__).resolve().parents[1]


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combination", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short technical run on five real development nights",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Directory containing data/manifests and data/raw",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Research features.parquet, alongside its metadata.json",
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "configs/splits/sc_subjects_v1.csv"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "var/research/sc-tcn-v1")
    parser.add_argument(
        "--champion",
        type=Path,
        help="Historical refinement lgbm.json with its runs directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs", type=positive, help="Maximum temporal-head epochs (default 20)"
    )
    parser.add_argument(
        "--encoder-epochs",
        type=positive,
        help="Fixed single-epoch CNN training epochs (default 5)",
    )
    parser.add_argument("--batch-size", type=positive, default=128)
    args = parser.parse_args()
    if args.seed < 0 or args.seed > 2**32 - 5:
        parser.error("seed must be in 0..4294967291")
    device = research_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    reference = ROOT / "var/research/sc-w30-v2"
    features = args.features
    if features is None:
        campaign_path = reference / "campaign.json"
        if not campaign_path.is_file():
            parser.error(
                "supply --features pointing to the Sleep Cassette research cache"
            )
        saved = Path(json.loads(campaign_path.read_text())["dataset"])
        features = reference / "datasets" / saved.parent.name / saved.name
    if not features.is_file():
        parser.error(f"feature cache missing: {features}")
    source = args.source_root or os.getenv("SLEEP_RESEARCH_DATA_ROOT") or ROOT
    if (
        source == ROOT
        and not (ROOT / "data").exists()
        and (reference / "work-state.json").exists()
    ):
        source = (
            json.loads((reference / "work-state.json").read_text()).get("source_root")
            or ROOT
        )
    config = Settings(
        seed=args.seed,
        batch_size=args.batch_size,
        width=16 if args.smoke else 64,
        epochs=args.epochs or (1 if args.smoke else 20),
        encoder_epochs=args.encoder_epochs or (1 if args.smoke else 5),
    )
    print(
        f"Device={device}, combination={args.combination}, smoke={args.smoke}",
        flush=True,
    )
    champion = args.champion
    default_champion = ROOT / "var/research/sc-w30-refinement-v1/lgbm.json"
    if champion is None and default_champion.is_file():
        champion = default_champion
    output = run(
        args.combination,
        features,
        args.split,
        Path(source),
        args.output,
        config,
        device,
        args.smoke,
        champion,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
