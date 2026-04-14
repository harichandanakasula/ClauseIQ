# ClauseIQ — Regulatory Document Intelligence

Detects semantic contradictions across banking compliance documents using 
RAG + NLI. Targets the $6.6B annual US bank compliance fine problem.

## What it does
Banks operate under multiple overlapping regulatory frameworks (Fed SR 11-7, 
OCC guidelines, etc.). Conflicting requirements often go unnoticed, leading 
to costly compliance failures. ClauseIQ surfaces these contradictions 
automatically.

## How it works
1. **Ingestion** — PDFs are chunked with structure-aware splitting and 
   embedded using BAAI/bge-base-en-v1.5, stored in Qdrant vector DB
2. **Retrieval** — User query retrieves top-k semantically similar clause 
   pairs across documents
3. **Contradiction Detection** — DeBERTa NLI cross-encoder scores each 
   pair for contradiction, entailment, or neutrality
4. **Answer Generation** — Claude (claude-haiku) synthesizes findings with 
   source attribution and flags conflicts explicitly

## Tech Stack
- **Embeddings:** BAAI/bge-base-en-v1.5
- **Vector DB:** Qdrant
- **NLI Model:** cross-encoder/nli-deberta-v3-base
- **LLM:** Claude via Anthropic API (LangChain)
- **Backend:** FastAPI
- **Frontend:** Streamlit

## Documents Supported
- Fed SR 11-7 (Federal Reserve Model Risk Management)
- OCC Model Risk 2021
- Add any banking compliance PDF to /data

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python pipeline/loader.py  # ingest documents
streamlit run app.py  # launch UI
```

## Environment Variables
```
ANTHROPIC_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
```
