"""Genera la figura de arquitectura del artículo (ciclo de ML + despliegue).

Rótulos grandes y paleta legible en escala de grises (observación de la rúbrica 7
del avance). Uso:
    uv run --locked python docs/articulo/figura_arquitectura.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("figura_arquitectura.png")

LIGHT, MID, DARK = "#f2f2f2", "#d9d9d9", "#bfbfbf"


def box(ax, x, y, w, h, title, body, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fill, ec="black", lw=1.4,
        )
    )
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
            fontsize=15, weight="bold")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top",
            fontsize=12.5, linespacing=1.3)


def arrow(ax, p, q, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=22,
                                 lw=1.8, color="black", linestyle=ls))


fig, ax = plt.subplots(figsize=(16, 9.2), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.2)
ax.axis("off")

# Franja superior: ciclo de aprendizaje de máquina (investigación)
ax.text(0.1, 9.05, "A. Ciclo de aprendizaje de máquina (desarrollo, 62 sujetos)",
        fontsize=16, weight="bold", va="top")
w, h, y = 2.65, 2.6, 5.75
xs = [0.1, 3.3, 6.5, 9.7, 12.9]
box(ax, xs[0], y, w, h, "1. Datos",
    "Sleep-EDFx\n(PhysioNet)\nSleep Cassette:\n153 noches, 78 sujetos\nSHA-256 verificado", LIGHT)
box(ax, xs[1], y, w, h, "2. Preparación",
    "Épocas de 30 s\nFiltro 0,3–35 Hz\nRecorte de vigilia\n±30 min\nControl de calidad", LIGHT)
box(ax, xs[2], y, w, h, "3. Características",
    "145 por época:\nFpz-Cz, Pz-Oz,\nEOG, EMG\n+ 290 de contexto\ntemporal = 435", LIGHT)
box(ax, xs[3], y, w, h, "4. Modelado",
    "RF, CatBoost,\nLightGBM,\nLSTM / BiLSTM\n5 folds por sujeto\n+ ajuste interno", LIGHT)
box(ax, xs[4], y, w, h, "5. Registro",
    "MLflow: parámetros,\nmétricas, OOF\nModelo campeón:\nLightGBM ajustado\n16 sujetos reservados", LIGHT)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + w, y + h / 2), (b, y + h / 2))

# Franja inferior: despliegue
ax.text(0.1, 5.2, "B. Despliegue (AWS us-west-2 + Vercel, infraestructura como código con Terraform)",
        fontsize=16, weight="bold", va="top")
y2, h2 = 1.75, 2.75
box(ax, 0.1, y2, 2.65, h2, "Usuario",
    "Navegador web\nCarga una noche\no usa el ejemplo\npúblico", MID)
box(ax, 3.3, y2, 2.65, h2, "Frontend",
    "React + Vite\npublicado en\nVercel\nHipnograma,\ntiempos, CSV", MID)
box(ax, 6.5, y2, 2.65, h2, "API",
    "API Gateway\n+ Lambda\n(contenedor)\nCarga firmada a S3\nIdempotencia", MID)
box(ax, 9.7, y2, 2.65, h2, "Orquestación",
    "Step Functions\n→ Lambda worker\nValida SHA-256,\nderiva 435 variables,\npredice", MID)
box(ax, 12.9, y2, 2.65, h2, "Almacenamiento",
    "S3 versionado:\ncargas, resultados\n(expiran en 7 días)\ny versiones\ndel modelo", MID)
for a, b in [(2.75, 3.3), (5.95, 6.5), (9.15, 9.7), (12.35, 12.9)]:
    arrow(ax, (a, y2 + h2 / 2), (b, y2 + h2 / 2), style="<|-|>")

box(ax, 9.7, 0.0, 5.85, 1.55, "Reentrenamiento bajo demanda",
    "EC2 efímera (imagen en ECR) + SageMaker MLflow\n→ candidato verificado → promoción", DARK)
arrow(ax, (14.2, 1.55), (14.2, y2), ls="--")
arrow(ax, (xs[4] + w / 2, y), (14.2, y2 + h2), ls="--")
ax.text(14.4, 5.0, "exporta\nmodelo", fontsize=12, va="center")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
