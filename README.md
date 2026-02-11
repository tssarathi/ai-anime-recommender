# AI Anime Recommender

A production-grade **Retrieval-Augmented Generation (RAG)** anime recommendation system built with LangChain, ChromaDB, Groq LLM, and Streamlit. Users describe their anime preferences in natural language, and the system retrieves semantically relevant anime from a vector store, then generates personalized recommendations using an LLM.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Data Pipeline — Medallion Architecture](#data-pipeline--medallion-architecture)
- [RAG Workflow](#rag-workflow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker](#docker)
  - [Kubernetes](#kubernetes)
- [Environment Variables](#environment-variables)
- [Key Design Decisions](#key-design-decisions)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit Chat UI                             │
│                     (src/app/app.py · port 8501)                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ user query
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 AnimeRecommendationPipeline                          │
│                   (src/pipeline/pipeline.py)                         │
│                                                                      │
│   ┌──────────────────┐      ┌─────────────────────────────────────┐  │
│   │  ChromaDB        │─────▶│  LangChain RetrievalQA Chain        │  │
│   │  Vector Store    │      │  ┌──────────────┐  ┌─────────────┐  │  │
│   │  (data/gold/)    │      │  │ HuggingFace  │  │ Groq LLM    │  │  │
│   │                  │      │  │ Embeddings   │  │ (Llama 3.1) │  │  │
│   └──────────────────┘      │  └──────────────┘  └─────────────┘  │  │
│                             └─────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Personalized Anime
                    Recommendations + Sources
```

---

## Tech Stack

| Layer                      | Technology                                                        |
| -------------------------- | ----------------------------------------------------------------- |
| **LLM**              | Groq Cloud — Llama 3.1 8B Instant (low-latency inference)        |
| **Orchestration**    | LangChain (RetrievalQA chain, prompt templates, document loaders) |
| **Embeddings**       | HuggingFace `all-MiniLM-L6-v2` via Sentence Transformers        |
| **Vector Store**     | ChromaDB (persistent, on-disk)                                    |
| **Frontend**         | Streamlit (chat-based conversational UI)                          |
| **Data Processing**  | Pandas (CSV ETL pipeline)                                         |
| **Containerization** | Docker (Python 3.11-slim)                                         |
| **Orchestration**    | Kubernetes (Deployment + LoadBalancer Service)                    |
| **Language**         | Python 3.11                                                       |
| **Linting**          | Ruff (linter + formatter)                                         |
| **Config**           | python-dotenv (environment variable management)                   |

---

## Data Pipeline — Medallion Architecture

The ETL pipeline follows a **Bronze → Silver → Gold** medallion pattern, a data engineering best practice for incremental data quality improvement:

```
Bronze (Raw)                    Silver (Cleaned)                Gold (Serving)
─────────────                   ────────────────                ──────────────
anime_with_synopsis.csv   ───▶  anime_updated.csv         ───▶  ChromaDB
                                                                Vector Store
  • Raw CSV with             • Structured "document"          • Embedded with
    Name, Genres,              column combining                 all-MiniLM-L6-v2
    Synopsis fields            Title + Genres + Synopsis      • Chunked (1000 chars)
  • Dropped NaN rows         • Ready for embedding            • Persisted to disk
```

**`build_pipeline.py`** orchestrates this full ETL:

1. **`AnimeDataLoader`** — reads the bronze CSV, validates required columns (`Name`, `Genres`, `sypnopsis`), concatenates fields into a single `document` column, and writes to silver.
2. **`VectorStoreBuilder`** — loads the silver CSV via LangChain's `CSVLoader`, splits documents with `CharacterTextSplitter`, generates embeddings, and persists to ChromaDB.

---

## RAG Workflow

When a user sends a message, the following happens end-to-end:

```
1. User Input          "I like dark psychological anime with plot twists"
       │
       ▼
2. Embedding           Query embedded via all-MiniLM-L6-v2
       │
       ▼
3. Retrieval           Top-10 most similar anime retrieved from ChromaDB
       │
       ▼
4. Prompt Assembly     Retrieved docs injected into a custom PromptTemplate
       │                with strict rules (context-only, ranked, max 3)
       ▼
5. LLM Generation      Groq (Llama 3.1 8B) generates personalized response
       │
       ▼
6. Response Display    Streamlit renders recommendations + expandable sources
```

The prompt template enforces:

- **Context-grounded responses** — the LLM can only recommend anime present in the retrieved documents
- **Ranked output** — best match first, up to 3 recommendations
- **Structured format** — Title, Genres, Synopsis, and a personalized "Why you'll love it" explanation
- **Graceful fallback** — if no good match exists, the system suggests rephrasing

---

## Project Structure

```
ai-anime-recommender/
│
├── src/
│   ├── app/
│   │   └── app.py                 # Streamlit chat UI & session management
│   │
│   ├── config/
│   │   └── config.py              # Centralized env variable loading
│   │
│   ├── etl/
│   │   ├── data_loader.py         # Bronze → Silver CSV transformation
│   │   └── vector_store.py        # Silver → Gold vector store builder
│   │
│   ├── llm/
│   │   ├── prompt_template.py     # LangChain PromptTemplate with rules
│   │   └── recommender.py         # RetrievalQA chain setup & invocation
│   │
│   ├── pipeline/
│   │   ├── build_pipeline.py      # One-time ETL: CSV → Vector Store
│   │   └── pipeline.py            # Runtime pipeline: query → recommendations
│   │
│   └── utils/
│       ├── custom_exception.py    # Exception with file/line detail
│       └── logger.py              # Timestamped file-based logging
│
├── data/
│   ├── bronze/                    # Raw anime dataset (CSV)
│   ├── silver/                    # Processed documents (CSV)
│   └── gold/                      # ChromaDB vector store (persisted)
│
├── logs/                          # Auto-generated runtime logs
│
├── Dockerfile                     # Multi-stage Python 3.11-slim container
├── deployment.yaml                # Kubernetes Deployment manifest
├── service.yaml                   # Kubernetes LoadBalancer Service
├── setup.py                       # Editable package install
├── requirements.txt               # Pinned Python dependencies
└── .env                           # API keys (not committed)
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Groq API Key** — [Get one free at groq.com](https://console.groq.com)
- **HuggingFace API Token** — [Generate at huggingface.co](https://huggingface.co/settings/tokens)
- **Docker** (optional, for containerized deployment)
- **kubectl + Minikube/Kind** (optional, for Kubernetes deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/sarathi/ai-anime-recommender.git
cd ai-anime-recommender

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies (editable mode)
pip install -e .

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see Environment Variables section)

# 5. Build the vector store (one-time ETL)
python -m src.pipeline.build_pipeline

# 6. Launch the application
streamlit run src/app/app.py
```

The app will be available at **http://localhost:8501**.

### Docker

```bash
# Build the image
docker build -t ai-anime-recommender:latest .

# Run the container
docker run -p 8501:8501 --env-file .env ai-anime-recommender:latest
```

### Kubernetes

```bash
# 1. Create secrets from your .env file
kubectl create secret generic llmops-secrets \
  --from-env-file=.env

# 2. Deploy the application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 3. Access the service
kubectl get svc llmops-service
# Use the EXTERNAL-IP on port 80, or use port-forward for local clusters:
kubectl port-forward svc/llmops-service 8501:80
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.1-8b-instant
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

| Variable                     | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `GROQ_API_KEY`             | API key for Groq cloud LLM inference                     |
| `MODEL_NAME`               | Groq model identifier (default:`llama-3.1-8b-instant`) |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace token for embedding model access             |

---

## Key Design Decisions

| Decision                             | Rationale                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **RAG over fine-tuning**       | Keeps the LLM general-purpose while grounding responses in a curated anime dataset — cheaper, faster to update, and avoids hallucination |
| **Medallion Architecture**     | Separates raw ingestion, transformation, and serving layers for clean data lineage and reproducibility                                    |
| **ChromaDB (persistent)**      | Lightweight, embedded vector store that persists to disk — no external database server required                                          |
| **Groq (Llama 3.1 8B)**        | Sub-second inference latency for real-time chat; free tier available for development                                                      |
| **all-MiniLM-L6-v2**           | Fast, high-quality sentence embeddings (384-dim) balancing accuracy and performance                                                       |
| **Streamlit chat UI**          | Rapid prototyping of conversational interfaces with built-in session state management                                                     |
| **Context-grounded prompting** | Custom prompt template prevents hallucination by restricting recommendations to retrieved documents only                                  |
| **Kubernetes-ready**           | Deployment and Service manifests enable single-command production deployment on any K8s cluster                                           |
| **Modular package structure**  | Each concern (ETL, LLM, pipeline, config, utils) is isolated for testability and maintainability                                          |
| **Custom exception handling**  | Exceptions include file name and line number for efficient debugging in production                                                        |

---

## License

This project is open source and available under the [MIT License](LICENSE).
