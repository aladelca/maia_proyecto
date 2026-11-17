"""Run development-only literature-driven ablations and grouped CatBoost tuning."""

import argparse
import json
from pathlib import Path

from sleep_staging.config import project_root, read_yaml
from sleep_staging.research import SPECS, TUNING_CANDIDATES, run_research_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("var/research/sc-w30-v2")
    )
    parser.add_argument(
        "--experiment", action="append", choices=[s.name for s in SPECS]
    )
    parser.add_argument("--no-refit", action="store_true")
    parser.add_argument(
        "--tuning-config",
        type=Path,
        default=project_root() / "configs/research/catboost-moderate.yaml",
    )
    args = parser.parse_args()
    candidates = read_yaml(args.tuning_config).get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not all(isinstance(p, dict) for p in candidates)
    ):
        parser.error(
            "The tuning configuration requires a non-empty list of candidates."
        )
    TUNING_CANDIDATES[:] = candidates
    result = run_research_campaign(
        args.source_root.resolve(),
        project_root() / "configs/splits/sc_subjects_v1.csv",
        args.output_root.resolve(),
        names=args.experiment,
        refit=not args.no_refit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
