import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from itertools import combinations

load_dotenv()

# ── clients ──────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# embedding model — same one used in ingestion
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# NLI cross-encoder — this is the contradiction detector
# outputs scores for: contradiction, entailment, neutral
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

COLLECTION_NAME = "clauseiq"
CONTRADICTION_THRESHOLD = 0.35  # flag pairs scoring above this


# ── embed a query ─────────────────────────────────────────────────────────
def embed_query(query: str) -> list[float]:
    prefixed = f"Represent this regulatory document passage: {query}"
    embedding = embedding_model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()


# ── retrieve top-k chunks from Qdrant ────────────────────────────────────
def retrieve_chunks(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    query_vector = embed_query(query)

    # build optional metadata filter
    qdrant_filter = None
    if filters:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if filters.get("regulator"):
            conditions.append(
                FieldCondition(key="regulator", match=MatchValue(value=filters["regulator"]))
            )
        if filters.get("year_from"):
            from qdrant_client.models import Range
            conditions.append(
                FieldCondition(key="year", range=Range(gte=filters["year_from"]))
            )
        if conditions:
            from qdrant_client.models import Must
            qdrant_filter = Filter(must=conditions)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter
    )

    chunks = []
    for r in results:
        chunks.append({
            "text": r.payload["text"],
            "source_doc": r.payload["source_doc"],
            "regulator": r.payload["regulator"],
            "year": r.payload["year"],
            "score": r.score
        })

    return chunks


# ── run NLI on a pair of chunks ───────────────────────────────────────────
def score_pair(text_a: str, text_b: str) -> dict:
    """
    DeBERTa NLI outputs 3 scores: [contradiction, entailment, neutral]
    We return all three plus the predicted label.
    """
    scores = nli_model.predict([(text_a, text_b)])[0]

    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    predicted_label = label_map[scores.argmax()]

    return {
        "contradiction": float(scores[0]),
        "entailment": float(scores[1]),
        "neutral": float(scores[2]),
        "predicted_label": predicted_label
    }


# ── detect contradictions across retrieved chunks ─────────────────────────
def detect_contradictions(chunks: list[dict]) -> list[dict]:
    """
    Run pairwise NLI scoring across all retrieved chunks.
    Flag pairs where contradiction score > CONTRADICTION_THRESHOLD
    AND the chunks come from different source documents.
    """
    contradictions = []

    # only check pairs from different documents — same doc conflicts are less interesting
    pairs = [
        (a, b) for a, b in combinations(range(len(chunks)), 2)
        if chunks[a]["source_doc"] != chunks[b]["source_doc"]
    ]

    if not pairs:
        # if only one document, check all pairs anyway
        pairs = list(combinations(range(len(chunks)), 2))

    for i, j in pairs:
        chunk_a = chunks[i]
        chunk_b = chunks[j]

        nli_scores = score_pair(chunk_a["text"], chunk_b["text"])

        if nli_scores["contradiction"] >= CONTRADICTION_THRESHOLD:
            contradictions.append({
                "chunk_a": {
                    "text": chunk_a["text"],
                    "source": chunk_a["source_doc"],
                    "regulator": chunk_a["regulator"],
                    "year": chunk_a["year"]
                },
                "chunk_b": {
                    "text": chunk_b["text"],
                    "source": chunk_b["source_doc"],
                    "regulator": chunk_b["regulator"],
                    "year": chunk_b["year"]
                },
                "contradiction_score": round(nli_scores["contradiction"], 3),
                "entailment_score": round(nli_scores["entailment"], 3),
                "neutral_score": round(nli_scores["neutral"], 3)
            })

    # sort by contradiction score descending
    contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
    return contradictions


# ── main function: retrieve + detect ─────────────────────────────────────
def find_contradictions(query: str, top_k: int = 10, filters: dict = None) -> dict:
    print(f"\nQuery: {query}")
    print("Retrieving chunks...")
    chunks = retrieve_chunks(query, top_k=top_k, filters=filters)
    print(f"Retrieved {len(chunks)} chunks")

    print("Running NLI contradiction detection...")
    contradictions = detect_contradictions(chunks)
    print(f"Found {len(contradictions)} contradiction(s) above threshold {CONTRADICTION_THRESHOLD}")

    return {
        "query": query,
        "chunks_retrieved": len(chunks),
        "contradictions_found": len(contradictions),
        "contradictions": contradictions,
        "chunks": chunks
    }


# ── test it ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ClauseIQ — Contradiction Detector Test")
    print("=" * 50)
    queries = [
        "model validation independence requirements",
        "board oversight and governance responsibilities",
        "validation frequency and periodic review",
        "model risk appetite and limits",
    ]

    for query in queries:
        result = find_contradictions(query, top_k=10)
        print(f"\nQuery: {query}")
        print(f"Contradictions found: {result['contradictions_found']}")

        if result["contradictions"]:
            for c in result["contradictions"]:
                print(f"\n  Score: {c['contradiction_score']}")
                print(f"  A [{c['chunk_a']['source']}]: {c['chunk_a']['text'][:150]}")
                print(f"  B [{c['chunk_b']['source']}]: {c['chunk_b']['text'][:150]}")
        else:
            print("\n✓ No contradictions found above threshold.")
            print("  (Add more regulatory documents to data/ for richer results)")