import torch
import time
from retriever import retrieve_and_rerank
from generator import generate

def format_duration(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"

def print_separator(char="─", width=60):
    print(char * width)

def is_reasoning_query(question: str) -> bool:
    keywords = ["why", "compare", "analyze", "reason",
                "explain", "how does", "difference between"]
    q = question.lower()
    return any(kw in q for kw in keywords)

def run_rag(question: str, verbose: bool = False) -> dict:
    total_start = time.perf_counter()

    print_separator("═")
    print(f"  QUERY: {question}")
    print_separator("═")

    # ── Stage 1: Hybrid Retrieval ──────────────────────────
    print("\n[1/3] HYBRID RETRIEVAL (dense ANN + BM25 FTS + RRF fusion)")
    print(f"      Input : query string ({len(question)} chars)")
    t0 = time.perf_counter()

    from retriever import retrieve  # import here to time separately
    candidates = retrieve(question)

    retrieval_time = time.perf_counter() - t0
    print(f"      Output: {len(candidates)} candidates after RRF fusion")
    print(f"      Time  : {format_duration(retrieval_time)}")

    if verbose:
        print("\n      Top candidates before reranking:")
        for i, d in enumerate(candidates[:5]):
            preview = d['text'][:80].replace('\n', ' ')
            print(f"        [{i+1}] {d['source']} | chunk {d['chunk_idx']}")
            print(f"             \"{preview}...\"")

    # ── Stage 2: Reranking ────────────────────────────────
    print("\n[2/3] RERANKING (BGE-Reranker-v2-M3 cross-encoder)")
    print(f"      Input : {len(candidates)} candidates")
    t0 = time.perf_counter()

    from reranker import reranker as bge_reranker
    from config import RERANK_TOP_K
    passages = [c["text"] for c in candidates]
    ranked   = bge_reranker.rerank(question, passages)
    docs     = [candidates[idx] for idx, _score in ranked]

    rerank_time = time.perf_counter() - t0
    print(f"      Output: {len(docs)} docs kept (top {RERANK_TOP_K})")
    print(f"      Time  : {format_duration(rerank_time)}")

    if verbose:
        print("\n      Reranked docs (score → source):")
        for (idx, score), doc in zip(ranked, docs):
            preview = doc['text'][:80].replace('\n', ' ')
            print(f"        score={score:.4f} | {doc['source']} | chunk {doc['chunk_idx']}")
            print(f"             \"{preview}...\"")

    # ── Free VRAM before LLM ──────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 3: Generation ───────────────────────────────
    use_r      = is_reasoning_query(question)
    model_used = "deepseek-r1:14b" if use_r else "qwen3:14b"

    print(f"\n[3/3] GENERATION (model: {model_used})")
    context_chars = sum(len(d['text']) for d in docs)
    print(f"      Input : {len(docs)} chunks, ~{context_chars} chars of context")
    print(f"      Reason: {'reasoning query detected' if use_r else 'standard query'}")
    t0 = time.perf_counter()

    answer = generate(question, docs, use_reasoning=use_r)

    gen_time = time.perf_counter() - t0
    print(f"      Output: {len(answer)} chars generated")
    print(f"      Time  : {format_duration(gen_time)}")

    # ── Summary ───────────────────────────────────────────
    total_time = time.perf_counter() - total_start
    print_separator("─", 60)    
    print(f"  TIMING SUMMARY")
    print(f"  {'Retrieval':<20} {format_duration(retrieval_time):>10}")
    print(f"  {'Reranking':<20} {format_duration(rerank_time):>10}")
    print(f"  {'Generation':<20} {format_duration(gen_time):>10}")
    print(f"  {'─'*30}")
    print(f"  {'TOTAL':<20} {format_duration(total_time):>10}")
    print_separator("═")

    return {
        "question": question,
        "answer":   answer,
        "sources":  [d["source"] for d in docs],
        "model":    model_used,
        "timing": {
            "retrieval_s":  retrieval_time,
            "reranking_s":  rerank_time,
            "generation_s": gen_time,
            "total_s":      total_time,
        }
    }