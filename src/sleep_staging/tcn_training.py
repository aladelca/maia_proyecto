"""Supervised single-epoch representation learning and grouped temporal-head fits."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sleep_staging.models.tcn import SequenceHead, WaveEncoder
from sleep_staging.provenance import atomic_path, hash_file, write_json
from sleep_staging.refinement_data import class_weights, macro_f1_eval


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    width: int = 64
    batch_size: int = 128
    encoder_epochs: int = 5
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 0.001
    dropout: float = 0.2
    aux_weight: float = 0.2


def tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def save_network(model: nn.Module, path: Path, extra: dict) -> None:
    with atomic_path(path) as tmp:
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                **extra,
            },
            tmp,
        )


def batches(rows: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def learn_embeddings(
    waves: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    needed: np.ndarray,
    directory: Path,
    cfg: Settings,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Train exclusively on this phase's subjects. Reuse identical frozen embeddings across heads."""
    path = directory / "embeddings.npy"
    weights_file = directory / "encoder.pt"
    manifest = directory / "encoder.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["embeddings_sha256"] != hash_file(path) or saved[
            "model_sha256"
        ] != hash_file(weights_file):
            raise ValueError("Encoder cache checksum mismatch.")
        return np.load(path, mmap_mode="r")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = WaveEncoder(cfg.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    weights = tensor(class_weights(labels[train], 0.5).astype("float32"), device)
    for _ in range(cfg.encoder_epochs):
        for rows in batches(rng.permutation(train), cfg.batch_size):
            opt.zero_grad(set_to_none=True)
            logits = net(tensor(waves[rows], device))
            loss = F.cross_entropy(logits, tensor(labels[rows], device), weight=weights)
            loss.backward()
            opt.step()
    net.eval()
    with atomic_path(path) as tmp, torch.no_grad():
        output = np.lib.format.open_memmap(
            tmp, mode="w+", dtype="float32", shape=(len(labels), cfg.width)
        )
        for rows in batches(needed, cfg.batch_size):
            output[rows] = net.embed(tensor(waves[rows], device)).cpu().numpy()
        output.flush()
        del output
    save_network(net, weights_file, {"width": cfg.width})
    write_json(
        manifest,
        {"model_sha256": hash_file(weights_file), "embeddings_sha256": hash_file(path)},
    )
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def predict(
    model: nn.Module,
    values: np.ndarray,
    windows: np.ndarray,
    rows: np.ndarray,
    size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for batch in batches(rows, size):
        logits, _ = model(tensor(values[windows[batch]], device))
        parts.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(parts)


def fit_head(
    kind: str,
    values: np.ndarray,
    windows: np.ndarray,
    labels: np.ndarray,
    boundary: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    cfg: Settings,
    device: torch.device,
    seed: int,
    aux: bool,
    epochs: int,
) -> tuple[SequenceHead, int, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SequenceHead(
        kind, values.shape[1], cfg.width, windows.shape[1], cfg.dropout
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    weights = tensor(class_weights(labels[training], 0.5).astype("float32"), device)
    history: list[float] = []
    best = 0
    for step in range(1, epochs + 1):
        net.train()
        for batch in batches(rng.permutation(training), cfg.batch_size):
            opt.zero_grad(set_to_none=True)
            logits, auxiliary = net(tensor(values[windows[batch]], device))
            loss = F.cross_entropy(
                logits, tensor(labels[batch], device), weight=weights
            )
            if aux:
                assert auxiliary is not None
                valid = boundary[batch] >= 0
                if valid.any():
                    indices = tensor(np.flatnonzero(valid), device)
                    loss = loss + cfg.aux_weight * F.binary_cross_entropy_with_logits(
                        auxiliary.index_select(0, indices),
                        tensor(boundary[batch][valid].astype("float32"), device),
                    )
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if len(validation):
            prob = predict(net, values, windows, validation, cfg.batch_size, device)
            score = macro_f1_eval(labels[validation], prob)[1]
            history.append(score)
            if not best or score > history[best - 1]:
                best = step
            if step - best >= cfg.patience:
                break
    return net, best or epochs, history
