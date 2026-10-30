"""Acceso al dataset de Kaggle.

Responsable: Integrante C.

Carga el dataset (archivo local en DATASET_PATH) con pandas y ofrece consultas
acotadas que `agent` expone como herramientas. Las funciones deben devolver
resultados pequeños y serializables (texto o dict), aptos para pasarle al LLM,
nunca el DataFrame completo.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Lee el dataset desde disco."""
    raise NotImplementedError


# Las funciones de consulta concretas se definen cuando se elija el dataset.
