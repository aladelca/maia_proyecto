"""Tabla de inclusión/exclusión de épocas y distribución por clase (Sleep Cassette).

Responde a la observación de la rúbrica 2 del avance: reunir en tablas cuántas
épocas se incluyen y descartan, y cuántas quedan de cada clase en el conjunto
realmente analizado. Solo lee el índice de épocas (``epochs.parquet``) y la
partición fija; no lee señales ni modelos.

Por defecto la distribución por clase se reporta únicamente para desarrollo,
para no inspeccionar las etiquetas del conjunto de prueba reservado. La opción
``--include-test-classes`` la añade de forma explícita.

Uso:
    uv run --locked python scripts/epoch_inventory.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CLASSES = ["W", "N1", "N2", "N3", "REM"]
PARTITIONS = {"development": "Desarrollo", "test": "Prueba reservada"}


def _fmt(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def _pct(n: float, d: float) -> str:
    return f"{100 * n / d:.2f}".replace(".", ",") + " %" if d else "—"


def find_interim(root: Path, wake_margin: int | None) -> Path:
    """Localiza la versión de preprocesamiento de Sleep Cassette completa."""
    for meta_path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prep = meta.get("preprocessing", {})
        if (
            meta.get("data", {}).get("source") == "sleep-cassette"
            and prep.get("max_epochs_per_recording") is None
            and prep.get("wake_margin_minutes") == wake_margin
        ):
            return meta_path.parent
    raise FileNotFoundError(
        f"No hay preprocesamiento Sleep Cassette completo con wake_margin={wake_margin} en {root}"
    )


def load(interim: Path, splits: Path) -> pd.DataFrame:
    epochs = pd.read_parquet(interim / "epochs.parquet")
    split = pd.read_csv(splits)
    out = epochs.merge(split, on="subject_id", how="left", validate="many_to_one")
    if out.partition.isna().any():
        missing = out.loc[out.partition.isna(), "subject_id"].unique()
        raise ValueError(f"Sujetos sin partición: {missing[:5]}")
    out["reason"] = out.reason.fillna("")
    out["any_flat"] = out.filter(like="flat_").any(axis=1)
    out["eligible5"] = out.reason.eq("") & out.task_5.isin(CLASSES)
    return out


def flow_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    groups = [(PARTITIONS[p], df[df.partition == p]) for p in PARTITIONS]
    groups.append(("Total", df))
    for name, g in groups:
        movement = g.reason.eq("") & g.task_6.eq("M")
        rows[name] = {
            "Sujetos": g.subject_id.nunique(),
            "Registros (noches)": g.recording_id.nunique(),
            "Épocas de 30 s en los hipnogramas": len(g),
            "Excluidas: sin puntuar (?)": int(g.reason.eq("unscored").sum()),
            "Excluidas: recorte de vigilia (±30 min)": int(
                g.reason.eq("wake_crop").sum()
            ),
            "Excluidas como objetivo: movimiento (M), conservadas como contexto": int(
                movement.sum()
            ),
            "Épocas elegibles, tarea de 5 clases": int(g.eligible5.sum()),
            "  de ellas, con algún canal marcado como plano": int(
                (g.eligible5 & g.any_flat).sum()
            ),
        }
    return pd.DataFrame(rows)


def class_table(df: pd.DataFrame, partitions: list[str]) -> pd.DataFrame:
    cols = {}
    for p in partitions:
        g = df[(df.partition == p)]
        full = g[g.task_5.isin(CLASSES) & g.reason.isin(["", "wake_crop"])]
        kept = g[g.eligible5]
        n_full = full.task_5.value_counts().reindex(CLASSES, fill_value=0)
        n_kept = kept.task_5.value_counts().reindex(CLASSES, fill_value=0)
        label = PARTITIONS[p]
        cols[(label, "Noche completa")] = n_full
        cols[(label, "Analizadas")] = n_kept
        cols[(label, "% analizadas")] = (100 * n_kept / n_kept.sum()).round(2)
    out = pd.DataFrame(cols)
    out.loc["Total"] = out.sum()
    for c in out.columns:
        if c[1] == "% analizadas":
            out.loc["Total", c] = 100.0
    return out


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    dev = df[(df.partition == "development") & df.eligible5]
    t = pd.crosstab(dev.fold, dev.task_5).reindex(columns=CLASSES, fill_value=0)
    t.insert(0, "Sujetos", dev.groupby("fold").subject_id.nunique())
    t.insert(1, "Registros", dev.groupby("fold").recording_id.nunique())
    t["Total"] = t[CLASSES].sum(axis=1)
    t.loc["Total"] = t.sum()
    t.index = [f"Fold {i}" if i != "Total" else i for i in t.index]
    return t


def to_markdown(flow: pd.DataFrame, classes: pd.DataFrame, folds: pd.DataFrame) -> str:
    lines = [
        "# Inventario de épocas de Sleep Cassette",
        "",
        "Generado con `scripts/epoch_inventory.py` a partir del índice de épocas del "
        "preprocesamiento completo con recorte de vigilia de 30 minutos "
        "(`wake_margin_minutes: 30`) y la partición `configs/splits/sc_subjects_v1.csv`.",
        "",
        "## Tabla A. Flujo de inclusión y exclusión de épocas",
        "",
        "| Concepto | " + " | ".join(flow.columns) + " |",
        "| --- | " + " | ".join("---:" for _ in flow.columns) + " |",
    ]
    for idx, row in flow.iterrows():
        lines.append(f"| {idx.strip()} | " + " | ".join(_fmt(v) for v in row) + " |")
    lines += [
        "",
        "Las épocas de movimiento no son objetivo en la tarea de cinco clases, pero se "
        "conservan en la secuencia para calcular las variables de contexto temporal. "
        "La marca de canal plano es informativa: no excluye épocas.",
        "",
        "## Tabla B. Distribución por clase antes y después del recorte de vigilia",
        "",
    ]
    parts = list(dict.fromkeys(c[0] for c in classes.columns))
    header = ["Estadio"]
    for p in parts:
        header += [f"{p}: noche completa", f"{p}: analizadas", f"{p}: % analizadas"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in header[1:]) + " |")
    for idx, row in classes.iterrows():
        cells = []
        for p in parts:
            cells += [
                _fmt(row[(p, "Noche completa")]),
                _fmt(row[(p, "Analizadas")]),
                f"{row[(p, '% analizadas')]:.2f}".replace(".", ",") + " %",
            ]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Tabla C. Épocas analizadas por fold de desarrollo",
        "",
        "| Fold | " + " | ".join(folds.columns) + " |",
        "| --- | " + " | ".join("---:" for _ in folds.columns) + " |",
    ]
    for idx, row in folds.iterrows():
        lines.append(f"| {idx} | " + " | ".join(_fmt(v) for v in row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--splits", type=Path, default=Path("configs/splits/sc_subjects_v1.csv")
    )
    parser.add_argument("--wake-margin", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("docs/epoch-inventory.md"))
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("var/reports/epoch-inventory")
    )
    parser.add_argument("--include-test-classes", action="store_true")
    args = parser.parse_args()

    interim = find_interim(args.interim_root, args.wake_margin)
    df = load(interim, args.splits)
    partitions = ["development"] + (["test"] if args.include_test_classes else [])

    flow = flow_table(df)
    classes = class_table(df, partitions)
    folds = fold_table(df)

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    flow.to_csv(args.csv_dir / "flow.csv")
    classes.to_csv(args.csv_dir / "classes.csv")
    folds.to_csv(args.csv_dir / "folds.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(to_markdown(flow, classes, folds), encoding="utf-8")
    print(f"Preprocesamiento: {interim.name}")
    print(f"Escrito {args.output} y CSV en {args.csv_dir}")


if __name__ == "__main__":
    main()
