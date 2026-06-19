# AKTU Syllabus RAG Assistant

A Retrieval-Augmented Generation (RAG) system that lets students ask natural language questions about their AKTU syllabus and get grounded, source-cited answers — instead of manually searching through 40+ page PDFs.

> Ask: *"What topics are covered in Unit 3 of Software Engineering?"*
> Get: an answer pulled directly from the actual syllabus, with the source document cited — not a hallucinated guess.

---

## Why this exists

AKTU syllabus PDFs are long, dense, and repetitive across subjects (every subject has 5 units, similar formatting, similar phrasing). Finding a specific topic means scrolling through pages of near-identical structure. This project turns that into a question-answering interface grounded in the actual document content.

## How it works
PDF → structure-aware chunking → embeddings → MongoDB Atlas Vector Search → hybrid retrieval → Groq LLM → grounded answer

1. **Ingestion** (`ingest.py`) — extracts text from syllabus PDFs and splits it using regex-based structural detection (subject code + unit boundaries), not blind character-count chunking. Falls back to paragraph-based chunking for documents without a detectable subject/unit structure (e.g. PYQs).
2. **Embeddings** — each chunk is embedded locally using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions), no external API cost.
3. **Vector storage** — chunks + embeddings + metadata (subject code, subject name, unit) are stored in MongoDB Atlas with a Vector Search index (cosine similarity).
4. **Hybrid retrieval** (`retriever.py`) — vector search pulls a wide candidate pool (`numCandidates: 100`), then results are re-ranked: chunks whose `subject_name` appears in the user's query get boosted. This solves a real failure mode — multiple subjects in the same syllabus share near-identical "Unit III" phrasing, which confuses pure semantic search.
5. **Generation** (`main.py`) — retrieved chunks are passed to Groq's LLM with an explicit grounding instruction: *answer only from context, say so if the answer isn't there.* This is the core anti-hallucination guardrail.

## Key technical decisions (and why)

| Decision | Reasoning |
|---|---|
| Structure-aware chunking over fixed character splitting | Keeps a full concept (e.g. one Unit) in a single chunk instead of cutting it mid-thought |
| Cosine similarity over Euclidean distance | Embedding magnitude can reflect text length/density, not meaning. Cosine compares direction only, which better reflects semantic similarity |
| Retrieve-then-rerank (wide candidate pool, then narrow) | Re-ranking every document in the collection isn't feasible at scale; narrowing with cheap approximate search first, then doing precise re-ranking on a small pool, is the standard tradeoff |
| Subject-name boost instead of a hardcoded keyword→subject map | A hardcoded mapping breaks the moment a different college's syllabus or a new subject is uploaded. Boosting based on the subject name *extracted from the document itself* generalizes to any syllabus |
| Local embeddings (`sentence-transformers`) instead of an embedding API | No per-query cost, fully offline after model download |

## Tech stack

- **Backend:** FastAPI, Python
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector DB:** MongoDB Atlas Vector Search
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **PDF parsing:** `pypdf`

## Project structure
aktu-rag/

├── backend/

│   ├── ingest.py        # PDF → structured chunks → embeddings → MongoDB

│   ├── retriever.py      # hybrid vector search + re-ranking

│   ├── main.py            # FastAPI endpoints

│   ├── requirements.txt

│   ├── .env.example

│   └── data/               # place your syllabus/PYQ PDFs here (not committed)

└── README.md

## Setup

```bash
# clone and enter backend
cd backend

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# fill in MONGO_URI and GROQ_API_KEY in .env

# add a syllabus PDF to backend/data/

# ingest the PDF(s)
python ingest.py

# run the API
uvicorn main:app --reload
```

Test it at `http://127.0.0.1:8000/docs`.

### MongoDB Atlas Vector Search index

Create a Vector Search index named `vector_index` on the `aktu_rag.documents` collection:

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine" },
    { "type": "filter", "path": "metadata.source" }
  ]
}
```

## Known limitations / future work

- Currently handles one syllabus PDF format well; PYQ-specific chunking is not yet implemented
- No frontend yet — interaction is via the FastAPI `/docs` interface
- Re-ranking is a simple keyword-presence boost, not a learned re-ranker — a cross-encoder re-ranking model would likely improve precision further
- No conversational memory across questions (each query is independent)
