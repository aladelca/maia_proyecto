"""Train with subject-disjoint nights and score complete contiguous EEG blocks."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sleep_staging.fullnight.data import (
    LABELS,
    PROFILES,
    Inputs,
    profile_bits,
    valid_blocks,
)
from sleep_staging.fullnight.model import FullNightNet
from sleep_staging.provenance import atomic_path

HALO = 7
REFERENCE = "eeg_fpz_cz+eeg_pz_oz+eog"
ALL = REFERENCE + "+emg"
NETWORKS = {
    "wave": ("wave", "five"),
    "spectrogram": ("spectrum", "five"),
    "six_direct": ("wave", "six"),
    "six_movement": ("wave", "movement"),
}


@dataclass(frozen=True)
class Settings:
    epochs: int = 4
    width: int = 24
    chunk: int = 192
    chunks_per_night: int = 0
    trees: int = 64
    seed: int = 42
    training_window: str = "full"
    smoke: bool = False

    def __post_init__(self) -> None:
        if (
            min(self.epochs, self.width, self.chunk, self.trees) < 1
            or self.chunks_per_night < 0
        ):
            raise ValueError("Los tamaños y presupuestos deben ser positivos.")
        if self.training_window not in {"full", "w30"}:
            raise ValueError("Ventana de entrenamiento inválida.")


@dataclass
class Night:
    record: str
    subject: str
    fold: int
    path: Path
    inp: Inputs
    y: np.ndarray
    w30: np.ndarray

    @classmethod
    def load(cls, record: str, subject: str, fold: int, path: Path) -> "Night":
        return cls(
            record,
            subject,
            fold,
            path,
            Inputs.load(path),
            np.load(path / "labels.npy", mmap_mode="r"),
            np.load(path / "w30.npy", mmap_mode="r"),
        )


def tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(x, dtype=np.float32, copy=True, order="C")).to(
        device
    )


def encode_slice(
    net: FullNightNet, x: Inputs, lo: int, hi: int, device: torch.device
) -> torch.Tensor:
    values = x.wave if net.view == "wave" else x.spectrum
    return net.encode(tensor(values[lo:hi], device), tensor(x.emg[lo:hi], device))


def supervised(night: Night, settings: Settings, nclasses: int) -> np.ndarray:
    mask = (night.y >= 0) & (night.y < nclasses)
    if settings.training_window == "w30":
        mask &= night.w30
    return mask


def train_network(
    name: str, nights: list[Night], cfg: Settings, device: torch.device, seed: int
) -> FullNightNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = FullNightNet(*NETWORKS[name], width=cfg.width).to(device)
    nclasses = 5 if net.head == "five" else 6
    profile_masks = [profile_bits(p) for p in PROFILES]
    masks = [supervised(n, cfg, nclasses) for n in nights]
    counts = sum(
        (
            np.bincount(n.y[m & n.inp.eligible(profile_bits(ALL))], minlength=nclasses)
            for n, m in zip(nights, masks, strict=True)
        ),
        np.zeros(nclasses),
    )
    if counts.sum() == 0:
        raise ValueError("Sin etiquetas de entrenamiento.")
    weights = tensor(np.sqrt(counts.max() / np.maximum(counts, 1)).clip(1, 40), device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001)
    net.train()
    for _ in range(cfg.epochs):
        tasks: list[tuple[int, int, int]] = []
        for i, n in enumerate(nights):
            starts = np.arange(0, len(n.y), cfg.chunk)
            if cfg.chunks_per_night and len(starts) > cfg.chunks_per_night:
                # Technical/limited-budget runs draw the same seeded blocks across arms.
                starts = rng.choice(starts, cfg.chunks_per_night, replace=False)
            tasks.extend(
                (i, int(s), int(rng.integers(len(profile_masks)))) for s in starts
            )
        rng.shuffle(tasks)
        for i, start, profile in tasks:
            n = nights[i]
            end = min(start + cfg.chunk, len(n.y))
            lo = max(0, start - HALO)
            hi = min(len(n.y), end + HALO)
            present = n.inp.available[lo:hi] & profile_masks[profile]
            terms: list[torch.Tensor] = []
            masses: list[torch.Tensor] = []
            for a, b in valid_blocks(present[:, :2].any(1)):
                left = lo + a
                right = lo + b
                use = masks[i][left:right].copy()
                positions = np.arange(left, right)
                use &= (positions >= start) & (positions < end)
                if not use.any():
                    continue
                h = encode_slice(net, n.inp, left, right, device)
                lp = net(h, torch.from_numpy(present[a:b]).to(device))
                idx = torch.from_numpy(np.flatnonzero(use)).to(device)
                y = torch.from_numpy(n.y[left:right][use].astype(np.int64)).to(device)
                terms.append(F.nll_loss(lp[idx], y, weight=weights, reduction="sum"))
                masses.append(weights[y].sum())
            if terms:
                opt.zero_grad(set_to_none=True)
                loss = torch.stack(terms).sum() / torch.stack(masses).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
    return net.eval()


@torch.inference_mode()
def encode_night(net: FullNightNet, x: Inputs, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            encode_slice(net, x, j, min(j + 256, len(x.wave)), device)
            for j in range(0, len(x.wave), 256)
        ]
    )


@torch.inference_mode()
def predict_encoded(
    net: FullNightNet, enc: torch.Tensor, x: Inputs, profile: str
) -> np.ndarray:
    present = x.available & profile_bits(profile)
    probs = np.full(
        (len(enc), 5 if net.head == "five" else 6), np.nan, dtype=np.float32
    )
    for start, end in valid_blocks(present[:, :2].any(1)):
        mask = torch.from_numpy(present[start:end]).to(enc.device)
        probs[start:end] = net(enc[start:end], mask).exp().cpu().numpy()
    return probs


def save_model(
    path: Path, net: FullNightNet, settings: Settings, subjects: list[str], run_id: str
) -> None:
    bundle = {
        "schema": "ibio-edf-v1",
        "view": net.view,
        "head": net.head,
        "width": net.width,
        "classes": LABELS[: 5 if net.head == "five" else 6],
        "settings": asdict(settings),
        "training_subjects": sorted(set(subjects)),
        "run_id": run_id,
        "weights": {k: v.detach().cpu() for k, v in net.state_dict().items()},
    }
    with atomic_path(path) as tmp:
        torch.save(bundle, tmp)


def load_model(path: Path, device: torch.device) -> tuple[FullNightNet, dict]:
    item = torch.load(path, map_location="cpu", weights_only=True)
    net = FullNightNet(item["view"], item["head"], item["width"])
    if (
        item["schema"] != "ibio-edf-v1"
        or item["classes"] != LABELS[: 5 if net.head == "five" else 6]
    ):
        raise ValueError("Contrato de modelo EDF incompatible.")
    net.load_state_dict(item.pop("weights"), strict=True)
    return net.to(device).eval(), item
