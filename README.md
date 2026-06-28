---
title: SkyWings Airline Support
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SkyWings AI-Powered Airline Customer Support

FastAPI + Streamlit application that answers airline customer-support questions
using a Groq-hosted LLM, a Supabase PostgreSQL database for flight data, and a
Pinecone vector store for policy / FAQ retrieval.

## Architecture

- `backend.py` — FastAPI service exposing `POST /chat` with input/output guardrails.
- `streamlit_app.py` — Streamlit UI that calls the FastAPI backend.
- `ingest_pdf.py` — one-time PDF ingestion into Pinecone (run before first deploy).
- `start.sh` — boots uvicorn (8000) and Streamlit (7860) inside the container.

## Required secrets

Add the following under **Settings → Variables and secrets** in this Space:

| Name                  | Purpose                                  |
|-----------------------|------------------------------------------|
| `GROQ_API_KEY`        | LLM (Groq OpenAI-compatible endpoint)    |
| `PINECONE_API_KEY`    | Vector store                             |
| `PINECONE_INDEX_NAME` | Defaults to `airline-faq-index`          |
| `DB_HOST`             | Supabase pooler host                     |
| `DB_PORT`             | `5432`                                   |
| `DB_USER`             | Supabase username                        |
| `DB_PASSWORD`         | Supabase password                        |
| `DB_NAME`             | `postgres`                               |

## Local run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in real values
python ingest_pdf.py  # one-time
./start.sh
```

Then open `http://localhost:7860`.
