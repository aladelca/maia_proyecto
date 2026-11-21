"""Figura cualitativa: hipnograma real frente al predicho para una noche de prueba.

Usa las predicciones guardadas por scripts/evaluate_test.py (no vuelve a evaluar).
Se elige un sujeto representativo: el de kappa más cercano a la mediana por sujeto.

Uso (desde la raíz del repositorio):
    uv run --locked python docs/articulo/figura_hipnograma.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "var/test-evaluation"
OUT = Path(__file__).with_name("figura_hipnograma.png")
ORDER = ["N3", "N2", "N1", "REM", "W"]  # de abajo hacia arriba, convención clínica

per_subject = pd.read_csv(EVAL / "test_per_subject.csv")
median = per_subject.kappa.median()
subject = per_subject.iloc[(per_subject.kappa - median).abs().argsort().iloc[0]]
pred = pd.read_parquet(EVAL / "test_predictions.parquet")
night = pred[pred.subject_id.eq(subject.subject_id)]
recording = sorted(night.recording_id.unique())[0]
night = night[night.recording_id.eq(recording)].copy()
night["onset_h"] = night.epoch_id.str.split(":").str[1].astype(int) / 100 / 3600
night = night.sort_values("onset_h")
night["onset_h"] -= night.onset_h.iloc[0]
level = {s: i for i, s in enumerate(ORDER)}
real = night.task_5.map(level).to_numpy()
hat = night.prediction.map(level).to_numpy()
t = night.onset_h.to_numpy()
agree = float((real == hat).mean())

fig, axes = plt.subplots(2, 1, figsize=(16, 7.2), sharex=True, constrained_layout=True)
for ax, values, title in [
    (axes[0], real, "Hipnograma de referencia (experto)"),
    (axes[1], hat, "Hipnograma predicho (LightGBM)"),
]:
    ax.step(t, values, where="post", color="#14213D", linewidth=1.6)
    wrong = real != hat
    if ax is axes[1]:
        ax.scatter(t[wrong], values[wrong], s=10, color="#C2621D", zorder=3,
                   label="Época en desacuerdo")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=13, frameon=False)
    ax.set_yticks(range(len(ORDER)), ORDER, fontsize=14)
    ax.set_ylim(-0.5, len(ORDER) - 0.5)
    ax.set_title(title, fontsize=16, loc="left")
    ax.grid(axis="y", color="#DDDDDD")
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("Horas desde el inicio del registro analizado", fontsize=14)
axes[1].tick_params(axis="x", labelsize=13)
fig.savefig(OUT, dpi=200, facecolor="white")
print(f"{OUT} · {recording} · sujeto {subject.subject_id} · kappa {subject.kappa:.3f} · "
      f"F1 {subject.f1_macro:.3f} · concordancia {agree:.3f} · épocas {len(night)}")
