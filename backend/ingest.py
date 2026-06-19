import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

load_dotenv()

# --- 1. Connect to MongoDB Atlas ---
client = MongoClient(os.getenv("MONGO_URI"))
db = client["aktu_rag"]
collection = db["documents"]

# --- 2. Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 3. Extract text from PDF ---
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# --- 4. Structure-aware chunking ---
SUBJECT_HEADER_PATTERN = re.compile(
    r'\n([A-Z]{2,6}\d{3,4})\s+([A-Z][A-Z\s&,\-]{5,80})\n'
)

UNIT_PATTERN = re.compile(
    r'\n(I{1,3}|IV|V)\s*\n'  
)

def chunk_by_subject_and_unit(text: str, source_name: str):
    """
    Splits text into chunks based on subject headers first,
    then by Unit within each subject. Falls back to paragraph
    chunking if structure isn't detected.
    """
    chunks = []

    # Step 1: Split text into subject blocks
    subject_matches = list(SUBJECT_HEADER_PATTERN.finditer(text))

    if not subject_matches:
        return fallback_paragraph_chunking(text, source_name)

    for i, match in enumerate(subject_matches):
        subject_code = match.group(1).strip()
        subject_name = match.group(2).strip()
        start = match.start()
        end = subject_matches[i + 1].start() if i + 1 < len(subject_matches) else len(text)
        subject_block = text[start:end]

        # Step 2: Within this subject block, split by Unit
        unit_matches = list(UNIT_PATTERN.finditer(subject_block))

        if not unit_matches:
            chunks.append({
                "text": subject_block.strip()[:2000],  
                "subject_code": subject_code,
                "subject_name": subject_name,
                "unit": None
            })
            continue

        for j, umatch in enumerate(unit_matches):
            unit_label = umatch.group(1)
            ustart = umatch.start()
            uend = unit_matches[j + 1].start() if j + 1 < len(unit_matches) else len(subject_block)
            unit_text = subject_block[ustart:uend].strip()

            if len(unit_text) < 30:
                continue

            chunks.append({
                "text": f"{subject_code} {subject_name} - Unit {unit_label}\n{unit_text}"[:2000],
                "subject_code": subject_code,
                "subject_name": subject_name,
                "unit": unit_label
            })

    return build_documents(chunks, source_name)


def fallback_paragraph_chunking(text: str, source_name: str):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunks = []
    for p in paragraphs:
        chunks.append({
            "text": p[:2000],
            "subject_code": None,
            "subject_name": None,
            "unit": None
        })
    return build_documents(chunks, source_name)


def build_documents(chunks: list, source_name: str):
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk["text"],
            "embedding": model.encode(chunk["text"]).tolist(),
            "metadata": {
                "source": source_name,
                "chunk_index": i,
                "subject_code": chunk["subject_code"],
                "subject_name": chunk["subject_name"],
                "unit": chunk["unit"]
            }
        })
    return documents

# --- 5. Ingest a PDF into MongoDB ---
def ingest_pdf(pdf_path: str):
    source_name = Path(pdf_path).stem
    print(f"Ingesting: {source_name}")

    text = extract_text_from_pdf(pdf_path)
    documents = chunk_by_subject_and_unit(text, source_name)

    collection.delete_many({"metadata.source": source_name})
    collection.insert_many(documents)

    print(f"✅ Inserted {len(documents)} chunks from {source_name}")
    
    
    print("\n--- Sample chunks ---")
    for doc in documents[:3]:
        print(f"[{doc['metadata']['subject_code']} Unit {doc['metadata']['unit']}] {doc['text'][:80]}...")
    print()

# --- 6. Run this file directly to ingest all PDFs in /data folder ---
if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in /data folder. Add your syllabus/PYQ PDFs there.")
    else:
        for pdf_file in pdf_files:
            ingest_pdf(str(pdf_file))