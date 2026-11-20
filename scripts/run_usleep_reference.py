"""Retrain U-Sleep on the same Sleep Cassette folds and lock it before transfer."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch import nn

from sleep_staging.fullnight.cohort import prepare, research_root
from sleep_staging.fullnight.experiment import reports
from sleep_staging.fullnight.metrics import score
from sleep_staging.fullnight.training import Night
from sleep_staging.models import usleep_reference as ref
from sleep_staging.models.tcn import research_device
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def check_model(
    folder: Path,
    nights: list[Night],
    cfg: ref.Budget,
    seed: int,
    device: torch.device,
    identity: str,
) -> nn.Module:
    checkpoint = folder / "model.pt"
    people = sorted({n.subject for n in nights})
    if not checkpoint.is_file():
        net = ref.fit(nights, cfg, seed, device)
        with atomic_path(checkpoint) as tmp:
            torch.save(
                dict(
                    subjects=people,
                    run=identity,
                    weights={k: v.cpu() for k, v in net.state_dict().items()},
                ),
                tmp,
            )
        return net
    item = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if item["subjects"] != people or item["run"] != identity:
        raise ValueError("Pesos U-Sleep incompatibles con los participantes.")
    net = ref.build()
    net.load_state_dict(item["weights"])
    return net.to(device).eval()


def evaluate(
    nights: list[Night],
    folder: Path,
    net: nn.Module,
    cfg: ref.Budget,
    device: torch.device,
    conditions: dict,
) -> list:
    rows = []
    for n in nights:
        p = ref.predict(net, n.inp, cfg, device)
        with atomic_path(folder / "predictions" / (n.record + ".npz")) as tmp:
            np.savez_compressed(tmp, probabilities=p, onset_s=np.arange(len(p)) * 30)
        rows.append(
            dict(
                record=n.record,
                subject=n.subject,
                fold=n.fold,
                cohort=conditions.get(n.record, "development"),
                profile=ref.PROFILE,
                model="usleep",
                windows=score(n.y, p, n.w30),
            )
        )
    write_json(folder / "counts.json", rows)
    return rows


def run(
    base: Path, src: Path, cfg: ref.Budget, device: torch.device, reserved: bool
) -> Path:
    protocol = json.loads((base / "protocol.json").read_text())
    parent = json.loads((base / "freeze.json").read_text())
    split = ROOT / "configs/splits/sc_subjects_v1.csv"
    if (
        parent["smoke"]
        or protocol["split_sha256"] != hash_file(split)
        or fingerprint(protocol) != parent["run"]
    ):
        raise ValueError("La referencia requiere la campaña completa congelada.")
    spec = dict(
        campaign=parent["run"],
        settings=asdict(cfg),
        profile=ref.PROFILE,
        hz=50,
        library=version("braindecode"),
        torch=version("torch"),
        device=str(device),
        code=hash_file(Path(ref.__file__)),
        runner=hash_file(Path(__file__)),
        classes=protocol["classes"][:5],
        blocks="sample 35-epoch class-centred windows; score every eligible complete epoch",
    )
    identity = fingerprint(spec)
    folder = base / "usleep" / identity
    folder.mkdir(parents=True, exist_ok=True)
    people = pd.read_csv(split).set_index("subject_id")
    cache = base.parents[1] / "cache"
    dev = []
    for rid, key in protocol["nights"].items():
        path = cache / "nights" / rid / key
        subject = json.loads((path / "metadata.json").read_text())["subject"]
        dev.append(Night.load(rid, subject, int(people.loc[subject, "fold"]), path))
    if set(n.subject for n in dev) != set(parent["subjects"]):
        raise ValueError("Diferencia en los participantes de desarrollo.")
    print(f"U-Sleep {folder} · {device}", flush=True)
    with FileLock(str(folder / "run.lock")):
        write_json(folder / "protocol.json", spec)
        if not reserved:
            counts = []
            for f in range(5):
                dest = folder / f"fold-{f}"
                train, valid = (
                    [n for n in dev if n.fold != f],
                    [n for n in dev if n.fold == f],
                )
                if set(n.subject for n in train) & set(n.subject for n in valid):
                    raise ValueError(
                        "Coincidencia entre participantes de entrenamiento y validación."
                    )
                if (dest / "counts.json").is_file():
                    counts.extend(json.loads((dest / "counts.json").read_text()))
                else:
                    net = check_model(dest, train, cfg, cfg.seed + f, device, identity)
                    counts.extend(evaluate(valid, dest, net, cfg, device, {}))
                print(f"U-Sleep pliegue {f + 1}/5 completo", flush=True)
            reports(folder, counts, True)
            check_model(folder / "refit", dev, cfg, cfg.seed + 5, device, identity)
            write_json(
                folder / "freeze.json",
                dict(
                    run=identity,
                    subjects=parent["subjects"],
                    model_sha256=hash_file(folder / "refit/model.pt"),
                ),
            )
            return folder
        freeze = json.loads((folder / "freeze.json").read_text())
        if (
            freeze["run"] != identity
            or hash_file(folder / "refit/model.pt") != freeze["model_sha256"]
        ):
            raise ValueError("Se requiere el modelo congelado de U-Sleep.")
        dest = folder / "reserved"
        if (dest / "completed.json").is_file():
            return dest
        original = json.loads((base / "reserved/completed.json").read_text())
        conditions_file = ROOT / "configs/splits/st_conditions_v1.csv"
        if original["run"] != parent["run"] or original[
            "conditions_sha256"
        ] != hash_file(conditions_file):
            raise ValueError("Las condiciones difieren de la evaluación principal.")
        sc = pd.read_parquet(src / "data/manifests/sleep-cassette/recordings.parquet")
        sc = sc.loc[
            sc.subject_id.isin(people.index[people.partition.eq("test")])
        ].assign(condition="cassette_test")
        st = pd.read_parquet(
            src / "data/manifests/sleep-telemetry/recordings.parquet"
        ).merge(pd.read_csv(conditions_file), on="recording_id", validate="one_to_one")
        selected = pd.concat([sc, st], ignore_index=True)
        if len(selected) != original["nights"] or set(selected.subject_id) & set(
            parent["subjects"]
        ):
            raise ValueError("Participantes de prueba incompatibles.")
        paths = prepare(src, cache, selected)
        nights = [
            Night.load(r.recording_id, r.subject_id, -1, paths[r.recording_id])
            for r in selected.itertuples()
        ]
        net = check_model(folder / "refit", dev, cfg, cfg.seed + 5, device, identity)
        rows = evaluate(
            nights,
            dest,
            net,
            cfg,
            device,
            dict(zip(selected.recording_id, selected.condition, strict=True)),
        )
        telemetry = [
            {**r, "cohort": "telemetry"}
            for r in rows
            if r["cohort"] in {"placebo", "temazepam"}
        ]
        reports(dest, rows + telemetry, False)
        write_json(
            dest / "completed.json",
            dict(run=identity, nights=len(nights), weights=freeze["model_sha256"]),
        )
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--evaluate-frozen", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    print(
        run(
            a.run,
            research_root(ROOT, a.source_root),
            ref.Budget(),
            research_device(a.device),
            a.evaluate_frozen,
        )
    )
