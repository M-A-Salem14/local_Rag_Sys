import torch
from retriever  import retrieve_and_rerank
from generator  import generate

def is_reasoning_query(question: str) -> bool:
    """Simple heuristic: trigger DeepSeek for complex queries."""
    keywords = ["why", "compare", "analyze", "reason",
                "explain", "how does", "difference between"]
    q = question.lower()
    return any(kw in q for kw in keywords)

def run_rag(question: str, verbose: bool = False) -> dict:
    print(f"\n🔍 Query: {question}")

    # Step 1: Hybrid retrieve + rerank
    docs = retrieve_and_rerank(question)

    if verbose:
        print(f"   Retrieved {len(docs)} chunks after reranking")
        for i, d in enumerate(docs):
            print(f"   [{i+1}] {d['source']} | chunk {d['chunk_idx']}")

    # Step 2: Free VRAM from reranker before generation
    torch.cuda.empty_cache()

    # Step 3: Generate
    use_r = is_reasoning_query(question)
    answer = generate(question, docs, use_reasoning=use_r)

    model_used = "deepseek-r1:14b" if use_r else "qwen3:14b"
    print(f"   Model: {model_used}")

    return {
        "question":   question,
        "answer":     answer,
        "sources":    [d["source"] for d in docs],
        "model":      model_used,
    }