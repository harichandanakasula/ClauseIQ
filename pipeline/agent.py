import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from pipeline.contradiction_detector import retrieve_chunks, find_contradictions

load_dotenv()

# ── Claude model ──────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

# ── prompt template ───────────────────────────────────────────────────────
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are ClauseIQ, an expert regulatory compliance analyst for banking and financial institutions.

Your job is to answer questions about banking regulations clearly and accurately, based ONLY on the provided regulatory document excerpts.

Rules:
- Always cite your sources using [Document Name, Regulator, Year] format
- If the context does not contain enough information, say so clearly
- Use plain professional English — avoid unnecessary jargon
- If you detect tension or ambiguity between passages, flag it explicitly
- Never make up information not present in the context"""),

    ("human", """Here are the relevant regulatory document excerpts:

{context}

Question: {question}

Please answer the question with citations. If you notice any tensions or ambiguities across the excerpts, flag them clearly.""")
])

# ── format chunks into context string ────────────────────────────────────
def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i}]\n"
            f"Source: {chunk['source_doc']} | "
            f"Regulator: {chunk['regulator']} | "
            f"Year: {chunk['year']}\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)

# ── format contradiction results ──────────────────────────────────────────
def format_contradictions(contradictions: list[dict]) -> list[dict]:
    formatted = []
    for c in contradictions:
        formatted.append({
            "score": c["contradiction_score"],
            "source_a": f"{c['chunk_a']['source']} ({c['chunk_a']['regulator']}, {c['chunk_a']['year']})",
            "text_a": c["chunk_a"]["text"],
            "source_b": f"{c['chunk_b']['source']} ({c['chunk_b']['regulator']}, {c['chunk_b']['year']})",
            "text_b": c["chunk_b"]["text"],
        })
    return formatted

# ── main query function ───────────────────────────────────────────────────
def query_clauseiq(question: str, filters: dict = None) -> dict:
    print(f"\nProcessing query: {question}")

    # step 1 — retrieve relevant chunks
    print("Step 1: Retrieving relevant chunks...")
    chunks = retrieve_chunks(question, top_k=8, filters=filters)
    print(f"  Retrieved {len(chunks)} chunks")

    # step 2 — detect contradictions
    print("Step 2: Running contradiction detection...")
    contradiction_result = find_contradictions(question, top_k=8, filters=filters)
    contradictions = format_contradictions(contradiction_result["contradictions"])
    print(f"  Found {len(contradictions)} contradiction(s)")

    # step 3 — generate answer with Claude
    print("Step 3: Generating answer with Claude...")
    context = format_context(chunks)

    chain = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    print("  Answer generated")

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": c["source_doc"],
                "regulator": c["regulator"],
                "year": c["year"],
                "excerpt": c["text"][:200] + "..."
            }
            for c in chunks
        ],
        "contradictions": contradictions,
        "chunks_retrieved": len(chunks)
    }


# ── test it ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ClauseIQ — Agent Test")
    print("=" * 50)

    result = query_clauseiq(
        "What are the requirements for model validation independence?"
    )

    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result["answer"])

    print("\n" + "=" * 50)
    print("SOURCES USED:")
    print("=" * 50)
    for s in result["sources"][:3]:
        print(f"  - {s['source']} ({s['regulator']}, {s['year']})")

    if result["contradictions"]:
        print("\n" + "=" * 50)
        print(f"CONTRADICTIONS DETECTED: {len(result['contradictions'])}")
        print("=" * 50)
        for c in result["contradictions"]:
            print(f"\n  Score: {c['score']}")
            print(f"  A [{c['source_a']}]: {c['text_a'][:150]}...")
            print(f"  B [{c['source_b']}]: {c['text_b'][:150]}...")
    else:
        print("\n✓ No contradictions detected for this query")