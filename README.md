# tt2-agentic-rag

Obligatorio de **Taller de Tecnologías 2** (Universidad ORT Uruguay, 2026): un sistema
**Agentic RAG** en Python construido con LangGraph. El agente responde preguntas
consultando papers académicos (RAG sobre ChromaDB) y un dataset de Kaggle, decide
qué herramientas usar, mantiene múltiples conversaciones persistentes y tiene memoria
a corto y largo plazo.

## Documentación

La documentación académica (arquitectura, decisiones e informe) se mantiene en un
Google Docs compartido, no en este repositorio.

- Google Docs: _(agregar enlace cuando esté disponible)_

## Tecnologías

- Python 3.12
- LangGraph / LangChain
- ChromaDB (base vectorial local)
- pypdf, pandas
- LLM y embeddings vía API externa (proveedor a definir)
- Jupyter Notebook (entregable principal)
- pytest

## Estructura

```text
tt2-agentic-rag/
├── src/tt2_agentic_rag/
│   ├── ingest.py      # PDFs -> chunks -> embeddings -> Chroma
│   ├── rag.py         # búsqueda semántica
│   ├── dataset.py     # dataset de Kaggle
│   ├── agent.py       # grafo LangGraph y herramientas
│   └── memory.py      # memoria corto/largo plazo
├── notebooks/obligatorio.ipynb   # integración y demo
├── data/
│   ├── papers/        # PDFs (no se versionan)
│   └── datasets/      # dataset de Kaggle (no se versiona)
├── tests/
├── .env.example
├── pyproject.toml
└── requirements.txt
```

`storage/` (índice de Chroma y base SQLite de memoria) se genera localmente y está en `.gitignore`.

## Configuración en Fedora

### 1. Python 3.12

```bash
sudo dnf install python3.12
python3.12 --version
```

### 2. Entorno virtual

```bash
cd tt2-agentic-rag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Dependencias

```bash
pip install -e .
```

Instala el paquete en modo editable junto con todo lo listado en `requirements.txt`
(`pyproject.toml` lee las dependencias de ese archivo). Al agregar una dependencia,
editar solo `requirements.txt` y volver a correr el comando.

### 4. Variables de entorno

```bash
cp .env.example .env
# completar proveedor, modelo y clave
```

### 5. Datos

Copiar los PDFs a `data/papers/` y el dataset de Kaggle a `data/datasets/`.
No se suben al repositorio: cada integrante los descarga localmente.

En VS Code: `Python: Select Interpreter` → `.venv`.

## Uso

```bash
# Notebook
jupyter notebook notebooks/obligatorio.ipynb
# (o abrirlo en VS Code y elegir el kernel de .venv)

# Pruebas
pytest
```

## Colaboración

Distribución inicial:

| Integrante | Archivos |
|------------|----------|
| A | `ingest.py`, `rag.py` |
| B | `agent.py` |
| C | `dataset.py`, `memory.py` |

El notebook integra los módulos; lo edita una persona a la vez (avisar antes en el grupo).

Flujo de trabajo:

1. `main` siempre debe funcionar. No se commitea directo a `main`.
2. Una rama por tarea, creada desde `main` actualizado:
   ```bash
   git switch main && git pull
   git switch -c feature/rag-search
   ```
   Nombres: `feature/<módulo>-<descripción>` (ej. `feature/memory-checkpointer`).
3. Commits chicos con prefijo: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.
4. Antes del PR: `git pull --rebase origin main` y `pytest`.
5. Pull request a `main` con al menos una aprobación de otro integrante.
6. Si se cambia una firma pública de un módulo, avisarlo en el PR: otros módulos dependen de ella.
7. En la revisión de `agent.py`, verificar que no se use `create_agent()` ni `create_react_agent()`
   (el enunciado exige armar el grafo a mano).
8. Notebook: limpiar outputs antes de commitear (*Clear All Outputs*) para evitar conflictos.
9. Nunca commitear `.env`, claves ni datos pesados.

## Estado actual

Scaffolding inicial: estructura del paquete, stubs con docstrings (`NotImplementedError`),
notebook base y pruebas de importación. Pendiente: elegir proveedor de LLM, elegir dataset
e implementar los módulos.
