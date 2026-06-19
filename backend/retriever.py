import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["aktu_rag"]
collection = db["documents"]

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query: str, source_filter: str = None, top_k: int = 10):
    query_embedding = model.encode(query).tolist()
    query_lower = query.lower()

    filter_conditions = {}
    if source_filter:
        filter_conditions["metadata.source"] = source_filter

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,   # cast a wider net
            "limit": 20,            # pull more candidates before re-ranking
            "filter": filter_conditions if filter_conditions else {}
        }
    }

    project_stage = {
        "$project": {
            "text": 1,
            "metadata": 1,
            "score": {"$meta": "vectorSearchScore"},
            "_id": 0
        }
    }

    results = list(collection.aggregate([vector_search_stage, project_stage]))

    # Re-rank: boost chunks whose subject_name appears in the query
    def relevance_boost(chunk):
        subject_name = (chunk["metadata"].get("subject_name") or "").lower()
        boost = 0
        if subject_name and subject_name in query_lower:
            boost = 1.0  # strong boost if subject explicitly named in query
        return chunk["score"] + boost

    results.sort(key=relevance_boost, reverse=True)
    return results[:top_k]


def format_context(chunks: list) -> str:
    # Joins retrieved chunks into one context block for the LLM
    context = ""
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        subject = chunk["metadata"].get("subject_name") or "Unknown subject"
        unit = chunk["metadata"].get("unit")
        label = f"{subject} (Unit {unit})" if unit else subject
        context += f"[{label} | Source: {source}]\n{chunk['text']}\n\n"
    return context.strip()