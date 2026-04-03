# CRAG System

A Corrective RAG (Retrieval-Augmented Generation) system built with FastAPI and LangGraph. Upload documents, ask questions, and get answers grounded in your content — with automatic web search fallback when the documents aren't enough.

---

## How it works

The pipeline runs in six stages: it generates a hypothetical answer to improve retrieval (HyDE), scores the retrieved documents for relevance, rewrites the query for web search if needed, filters the context down to relevant sentences, and generates a final answer. The verdict — CORRECT, INCORRECT, or AMBIGUOUS — tells you whether the answer came from your documents, the web, or both.

---

## Stack

- **Backend** — FastAPI, LangGraph, LangChain
- **Vector store** — FAISS with BM25 ensemble and FlashRank reranking
- **Auth** — Google OAuth, JWT, Redis sessions, encrypted API key storage
- **Infrastructure** — Docker, Redis, Ollama

---

## Supported providers

| Type | Providers |
|---|---|
| LLM | Ollama, Groq, Anthropic, Google, HuggingFace API, HuggingFace Local |
| Embeddings | Ollama, HuggingFace (local), Google |

Groq and Anthropic have no embedding API and fall back to the configured `EMBEDDING_FALLBACK` provider automatically.

---

## Getting started

**Prerequisites** — Docker and Docker Compose.

Copy the example env file and fill in your values:

```bash
cp backend/.env.example backend/.env
```

Then start everything:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Ollama pulls the default models automatically on first startup.

---

## Environment variables

All variables go in `backend/.env`. The full list of required and optional variables is in `backend/.env.example`.

The key ones to set are `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` for OAuth, and `JWT_SECRET_KEY` and `ENCRYPTION_KEY` for session security. Provider API keys (Groq, Anthropic, Google, HuggingFace, Tavily) are stored per-user in Redis via `POST /auth/set_keys` — they do not go in the env file.

---

## API overview

```
GET  /auth/login              Google OAuth login
GET  /auth/callback           OAuth callback
POST /auth/set_keys           Store provider API keys for the session
POST /auth/logout

POST /documents/upload        Upload a PDF
POST /documents/process       Chunk, embed, and index uploaded PDFs
POST /documents/query         Test retrieval directly

POST /crag/chat               Run the CRAG pipeline

GET  /ollama/health           Check Ollama status
GET  /ollama/models/llm       List allowed LLM models
POST /ollama/models/llm/pull  Pull a model
GET  /ollama/models/embedding
POST /ollama/models/embedding/pull
```

---

## Running a chat request

After logging in and uploading + processing a document, send:

```json
POST /crag/chat
{
  "question": "your question here",
  "llm_provider": "groq",
  "llm_model": "llama-3.3-70b-versatile",
  "embedding_provider": "ollama",
  "embedding_model": "embeddinggemma:300m"
}
```

The response includes the answer, the verdict, and whether web search was used.

---

## Development

The backend source is in `backend/`. Dependencies are managed with two files — `requirements.txt` for unpinned development installs and `requirements.lock` for reproducible production builds. The Dockerfile uses `requirements.lock`.

To update the lock file after adding a new dependency, create a fresh virtual environment, install from `requirements.txt`, and freeze:

```bash
python -m venv freeze_env
source freeze_env/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock
deactivate && rm -rf freeze_env
```

---

## Future scope

- Frontend interface
- Logging
- Web search source citations in the response
- Async LangGraph execution + Background document indexing
- Conversation history
- Streaming responses with real-time pipeline stage updates
- Multi-modal support and additional document types
- MCP tools integration
- Workspaces for team and organisational use
- Function calling for users

---

## Project structure

```
backend/
  app/
    routers/        API endpoints
    services/       Business logic (CRAG, documents, LLM factory, Ollama, auth)
    models/         Pydantic schemas and SQLAlchemy models
    middleware/      JWT auth
    config.py       Settings via pydantic-settings
    main.py         FastAPI app and lifespan
  requirements.txt
  requirements.lock
  Dockerfile
docker-compose.yml
```