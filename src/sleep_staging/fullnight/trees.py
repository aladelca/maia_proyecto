"""One CPU LightGBM baseline per supported sensor configuration."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from sleep_staging.fullnight.data import PROFILES, Inputs, profile_bits, valid_blocks
from sleep_staging.provenance import atomic_path


def tree_features(x: Inputs, profile: str) -> np.ndarray:
    mask = profile_bits(profile)
    f = x.features[:, mask].reshape(len(x.available), -1)
    presence = x.available[:, mask].astype("float32")
    avg = np.full_like(f, np.nan)
    for begin, end in valid_blocks(x.eligible(mask)):
        values = f[begin:end]
        finite = np.isfinite(values)
        total = uniform_filter1d(
            np.where(finite, values, 0), size=15, axis=0, mode="nearest"
        )
        denom = uniform_filter1d(
            finite.astype("float32"), size=15, axis=0, mode="nearest"
        )
        np.divide(total, denom, out=avg[begin:end], where=denom > 1e-6)
    return np.column_stack((f, avg, presence))


def fit_predict(request: dict) -> None:
    import lightgbm as lgb

    folder = Path(request["output"])

    def arrays(record: str) -> tuple[Inputs, np.ndarray, np.ndarray]:
        path = Path(request["paths"][record])
        return (
            Inputs.load(path),
            np.load(path / "labels.npy"),
            np.load(path / "w30.npy"),
        )

    for profile in PROFILES:
        filename = folder / (profile + ".txt")
        if not filename.exists():
            features = []
            labels = []
            for rid in request["train"]:
                x, y, crop = arrays(rid)
                use = x.eligible(profile_bits(profile)) & (y >= 0) & (y < 5)
                if request["window"] == "w30":
                    use &= crop
                features.append(tree_features(x, profile)[use])
                labels.append(y[use])
            target = np.concatenate(labels)
            if not np.array_equal(np.unique(target), np.arange(5)):
                raise ValueError(
                    "Se requieren los cinco estadios en el entrenamiento LightGBM."
                )
            counts = np.bincount(target, minlength=5)
            model = lgb.LGBMClassifier(
                n_estimators=request["trees"],
                num_leaves=31,
                max_bin=63,
                learning_rate=0.08,
                n_jobs=3,
                min_child_samples=20,
                random_state=request["seed"],
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            model.fit(
                np.concatenate(features),
                target,
                sample_weight=np.sqrt(counts.max() / counts)[target],
            )
            with atomic_path(filename) as tmp:
                model.booster_.save_model(str(tmp))
            del features, labels, model
        booster = lgb.Booster(model_file=str(filename))
        for rid in request["validation"]:
            x = Inputs.load(Path(request["paths"][rid]))
            p = np.asarray(
                booster.predict(tree_features(x, profile), num_threads=2),
                dtype=np.float32,
            )
            p[~x.eligible(profile_bits(profile))] = np.nan
            with atomic_path(folder / rid / (profile + ".npy")) as tmp:
                np.save(tmp, p)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    job = p.add_mutually_exclusive_group(required=True)
    job.add_argument("request", type=Path, nargs="?")
    job.add_argument(
        "--features", type=Path, nargs=3, metavar=("TREE", "INPUT", "OUTPUT")
    )
    args = p.parse_args()
    if args.features:
        import lightgbm as lgb

        model_file, input_file, output_file = args.features
        booster = lgb.Booster(model_file=str(model_file))
        np.save(
            output_file,
            np.asarray(
                booster.predict(np.load(input_file), num_threads=2), dtype=np.float32
            ),
        )
    else:
        fit_predict(json.loads(args.request.read_text()))
