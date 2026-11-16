# tt2-agentic-rag

Obligatorio de **Taller de Tecnologías 2** (Universidad ORT Uruguay, 2026): un sistema
**Agentic RAG** en Python construido con LangGraph. El agente responde preguntas
consultando papers académicos y un dataset de Kaggle (Netflix Titles), ambos con RAG sobre Pinecone, decide
qué herramientas usar, mantiene múltiples conversaciones persistentes y tiene memoria
a corto y largo plazo.

## Documentación

La documentación académica (arquitectura, decisiones e informe) se mantiene en un
Google Docs compartido, no en este repositorio.

- Google Docs: _(agregar enlace cuando esté disponible)_

## Tecnologías

- Python 3.12
- LangGraph / LangChain
- Pinecone (base vectorial serverless)
- pypdf, pandas
- LLM: Groq (`qwen/qwen3.8-27b`); embeddings locales `all-MiniLM-L6-v2` (sentence-transformers)
- Jupyter Notebook (entregable principal)

## Estructura

```text
tt2-agentic-rag/
├── src/tt2_agentic_rag/
│   ├── ingest.py      # PDFs -> chunks -> embeddings -> Pinecone
│   ├── rag.py         # búsqueda semántica en papers (tool)
│   ├── dataset.py     # dataset Netflix -> Pinecone (tool)
│   ├── agent.py       # grafo LangGraph y herramientas
│   └── memory.py      # memoria corto/largo plazo
├── notebooks/
│   ├── obligatorio.ipynb        # integración y demo (importa los módulos de src/)
│   └── obligatorio_colab.ipynb  # versión autocontenida para Google Colab
├── data/
│   ├── papers/        # PDFs
│   └── datasets/      # netflix_titles.csv
├── .env.example
├── pyproject.toml
└── requirements.txt
```

`storage/` (base SQLite de la memoria a corto plazo) se genera localmente y está en `.gitignore`.
Los índices vectoriales (`research-papers`, `netflix-catalog`, `user-memory`) viven en Pinecone
y se crean automáticamente la primera vez.

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
# completar GROQ_API_KEY y PINECONE_API_KEY
```

### 5. Datos

Los PDFs están en `data/papers/` y el dataset en `data/datasets/netflix_titles.csv`.

En VS Code: `Python: Select Interpreter` → `.venv`.

## Uso

```bash
jupyter notebook notebooks/obligatorio.ipynb
# (o abrirlo en VS Code y elegir el kernel de .venv)
```

## Estado actual

Módulos implementados: ingesta e indexación de papers y del dataset Netflix en Pinecone,
tools de RAG, agente LangGraph armado a mano, memoria a corto plazo (SQLite) y a largo
plazo (hechos del usuario en Pinecone). `notebooks/obligatorio.ipynb` integra los módulos;
`notebooks/obligatorio_colab.ipynb` es la versión autocontenida para Colab (código copiado
en el notebook, claves desde los Secretos de Colab).
