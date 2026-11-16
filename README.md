# tt2-agentic-rag

Obligatorio de **Taller de Tecnologías 2** (Universidad ORT Uruguay, 2026): un sistema
**Agentic RAG** en Python construido con LangGraph. El agente responde preguntas
consultando papers académicos y un dataset de Kaggle (Netflix Titles), ambos con RAG sobre Pinecone, decide
qué herramientas usar, mantiene múltiples conversaciones persistentes y tiene memoria
a corto y largo plazo.

## Documentación

La documentación académica (arquitectura, decisiones e informe) se mantiene en un
Google Docs compartido, no en este repositorio.

- [Documentación del obligatorio (Google Docs)](https://docs.google.com/document/d/12Ru1_duFPMg2JqzT2VOpIqiN1tn-C2oBJoJYEMDlwy8)

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
│   ├── memory.py      # memoria corto/largo plazo
│   └── ui.py          # interfaz web Gradio
├── notebooks/
│   ├── obligatorio.ipynb        # integración y demo (importa los módulos de src/)
│   └── obligatorio_colab.ipynb  # versión autocontenida para Google Colab
├── data/
│   ├── papers/        # PDFs
│   └── datasets/      # netflix_titles.csv
├── app.py             # interfaz Gradio sin notebook
├── .env.example
├── pyproject.toml
└── requirements.txt
```

`storage/` (base SQLite de la memoria a corto plazo) se genera localmente y está en `.gitignore`.
Los índices vectoriales (`research-papers`, `netflix-catalog`, `user-memory`) viven en Pinecone
y se crean automáticamente la primera vez.

## Cómo ejecutar el notebook localmente

Instrucciones para correr `notebooks/obligatorio.ipynb` en una máquina propia (Linux, macOS o Windows).
Si se prefiere no instalar nada, `notebooks/obligatorio_colab.ipynb` corre en Google Colab.

### Requisitos previos

- **Python 3.12** (`python3 --version` o `py --version` en Windows) y **git**.
- **API key de Groq** (LLM): crear una gratis en <https://console.groq.com/keys>.
- **API key de Pinecone** (base vectorial): crear una gratis en <https://app.pinecone.io>, sección *API Keys*.
  El plan gratuito alcanza: el proyecto usa 3 índices.

Los datos ya vienen en el repositorio: los 6 PDFs en `data/papers/` y el dataset en
`data/datasets/netflix_titles.csv`. No hay que descargar nada aparte.

### 1. Clonar el repositorio

```bash
git clone https://github.com/ScheckGuillermo/tt2-agentic-rag.git
cd tt2-agentic-rag
```

### 2. Crear el entorno virtual e instalar dependencias

Linux / macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

`pip install -e .` instala todas las dependencias de `requirements.txt` y además el paquete
`tt2_agentic_rag` (el código de `src/`), que es lo que importa el notebook.

### 3. Configurar las API keys

Copiar la plantilla y completar las dos claves:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

```dotenv
GROQ_API_KEY=gsk_...
PINECONE_API_KEY=pcsk_...
```

El resto de las variables de `.env` ya tiene valores por defecto que funcionan.

### 4. Abrir y ejecutar el notebook

Con el entorno virtual activado:

```bash
jupyter notebook notebooks/obligatorio.ipynb
```

En el navegador, ejecutar todo con **Run → Run All Cells** (o *Kernel → Restart Kernel and Run All Cells*).
En VS Code: abrir el notebook, elegir el kernel de `.venv` (*Select Kernel* → *Python Environments*) y *Run All*.

### Qué esperar en la primera ejecución

- Se descarga una única vez el modelo de embeddings `all-MiniLM-L6-v2` (~90 MB).
- Los índices `research-papers`, `netflix-catalog` y `user-memory` se crean solos en la
  cuenta de Pinecone indicada en `.env`. Si están vacíos, el notebook calcula los embeddings
  en la CPU y los sube (~3.100 chunks de papers y ~8.800 títulos de Netflix): esto puede
  demorar varios minutos. Las ejecuciones siguientes detectan que los índices ya tienen
  datos y no reindexan.
- El historial de conversaciones se guarda en `storage/memory.sqlite`, que se crea automáticamente.
- La última celda (sección 8, opcional) levanta la interfaz Gradio en <http://127.0.0.1:7860>.
  La misma interfaz se puede abrir sin el notebook con `python app.py`.

### Problemas frecuentes

| Síntoma | Solución |
|---------|----------|
| `ValueError: Falta GROQ_API_KEY en .env` (o `PINECONE_API_KEY`) | Revisar que exista `.env` en la raíz del repo con ambas claves completas. |
| `ModuleNotFoundError: No module named 'tt2_agentic_rag'` | El notebook no está usando el kernel de `.venv`, o faltó `pip install -e .`. |
| Error de rate limit de Groq (`429`) | Límite del plan gratuito; esperar un minuto y volver a ejecutar la celda. |
