import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

from retriever import retrieve_chunks, format_context

load_dotenv()

app = FastAPI(title="AKTU RAG Assistant")

# --- Allow frontend (React) to call this backend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your Vercel URL once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq client ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# --- Request schema ---
class QueryRequest(BaseModel):
    question: str
    source_filter: str | None = None   # optional: restrict to one PDF


# --- Health check ---
@app.get("/")
def root():
    return {"status": "AKTU RAG Assistant is running"}


# --- Main RAG endpoint ---
@app.post("/ask")
def ask_question(request: QueryRequest):
    # Step 1: Retrieve relevant chunks from MongoDB
    chunks = retrieve_chunks(
        query=request.question,
        source_filter=request.source_filter,
        top_k=5
    )

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": []
        }

    # Step 2: Build context from retrieved chunks
    context = format_context(chunks)

    # Step 3: Build the prompt — this is where we ground the LLM in real data
    prompt = f"""You are a helpful study assistant for AKTU engineering students.
Answer the question using ONLY the context below. If the answer isn't in the context, say so clearly — do not make up information.

Context:
{context}

Question: {request.question}

Answer:"""

    # Step 4: Call Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content

    # Step 5: Return answer + which sources were used (for transparency)
    sources = list(set(chunk["metadata"]["source"] for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources
    }