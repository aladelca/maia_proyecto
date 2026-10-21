"""Ejecuta los notebooks con el kernel local y los EDF reales ya descargados."""

import nbformat
from nbclient import NotebookClient

from sleep_staging.config import project_root


def main() -> None:
    root = project_root()
    for path in sorted((root / "notebook").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="sleep-staging-local",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        nbformat.write(notebook, path)
        print(f"Verificado: {path.name}", flush=True)


if __name__ == "__main__":
    main()
