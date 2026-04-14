import os
import fitz  # pymupdf
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── clients ──────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# free, local, no API key needed — very good for legal text
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

COLLECTION_NAME = "clauseiq"
VECTOR_DIM = 768  # bge-base outputs 768-dim vectors

# ── create collection if it doesn't exist ────────────────────────────────
def create_collection():
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"✓ Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"✓ Collection already exists: {COLLECTION_NAME}")

# ── extract text from PDF ─────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# ── infer metadata from filename ──────────────────────────────────────────
def extract_metadata(pdf_path: str) -> dict:
    filename = Path(pdf_path).stem.lower()

    # regulator detection from filename
    if any(k in filename for k in ["fed", "sr1", "frb", "federal"]):
        regulator = "Federal Reserve"
    elif "occ" in filename:
        regulator = "OCC"
    elif "cfpb" in filename:
        regulator = "CFPB"
    elif any(k in filename for k in ["bis", "basel"]):
        regulator = "BIS/Basel"
    elif "fincen" in filename:
        regulator = "FinCEN"
    else:
        regulator = "Other"

    # year detection from filename
    year = 0
    for part in filename.replace("-", " ").replace("_", " ").split():
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break

    # topic tagging based on keywords in filename
    topic_map = {
        "model": "Model Risk",
        "aml": "AML",
        "fair": "Fair Lending",
        "capital": "Capital Requirements",
        "stress": "Stress Testing",
        "ai": "AI Governance",
        "cyber": "Cybersecurity",
        "data": "Data Governance",
    }
    topics = [label for keyword, label in topic_map.items() if keyword in filename]
    if not topics:
        topics = ["General"]

    return {
        "source_doc": Path(pdf_path).stem,
        "regulator": regulator,
        "year": year,
        "filename": Path(pdf_path).name,
        "topics": topics
    }

# ── embed a list of text chunks ───────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    # bge models work best with this prefix for retrieval
    prefixed = [f"Represent this regulatory document passage: {t}" for t in texts]
    embeddings = embedding_model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()

# ── chunk and ingest a single PDF ─────────────────────────────────────────
def ingest_pdf(pdf_path: str):
    print(f"\n→ Ingesting: {Path(pdf_path).name}")

    # extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print(f"  ✗ No text extracted — skipping (is the PDF scanned?)")
        return
    print(f"  Extracted {len(raw_text):,} characters")

    # chunk with overlap to preserve context across boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(raw_text)
    print(f"  Created {len(chunks)} chunks")

    # extract metadata
    metadata = extract_metadata(pdf_path)
    print(f"  Regulator: {metadata['regulator']} | Year: {metadata['year']} | Topics: {metadata['topics']}")

    # embed and upload in batches of 16
    batch_size = 16
    all_points = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = embed_texts(batch)

        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    **metadata,
                    "text": chunk,
                    "chunk_index": i + j
                }
            )
            all_points.append(point)

        print(f"  Embedded chunks {i+1}–{min(i+batch_size, len(chunks))} of {len(chunks)}")

    # upload all points to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=all_points
    )
    print(f"  ✓ Uploaded {len(all_points)} chunks → Qdrant")

# ── ingest all PDFs in data/ folder ──────────────────────────────────────
def ingest_all():
    print("=" * 50)
    print("ClauseIQ — Document Ingestion Pipeline")
    print("=" * 50)

    create_collection()

    data_dir = Path("data")
    pdfs = list(data_dir.glob("*.pdf"))

    if not pdfs:
        print("\n✗ No PDFs found in data/ folder.")
        print("  Add regulatory PDFs and run again.")
        return

    print(f"\nFound {len(pdfs)} PDF(s) to ingest:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")

    for pdf in pdfs:
        ingest_pdf(str(pdf))

    print("\n" + "=" * 50)
    print(f"✓ Done. Ingested {len(pdfs)} document(s) into ClauseIQ.")
    print("=" * 50)

if __name__ == "__main__":
    ingest_all()