from collections import defaultdict
from embedder import embedder
from db_init import get_db, get_or_create_table
from config import RETRIEVE_TOP_K, DENSE_WEIGHT, SPARSE_WEIGHT
from reranker import reranker as bge_reranker


RRF_K = 60   # standard RRF constant

def rrf_fusion(
    dense_results: list,
    sparse_results: list,
    dense_w: float = DENSE_WEIGHT,
    sparse_w: float = SPARSE_WEIGHT,
) -> list[dict]:
    scores = defaultdict(float)
    docs   = {}

    for rank, row in enumerate(dense_results):
        scores[row["id"]] += dense_w / (RRF_K + rank + 1)
        docs[row["id"]] = row

    for rank, row in enumerate(sparse_results):
        scores[row["id"]] += sparse_w / (RRF_K + rank + 1)
        docs[row["id"]] = row

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[doc_id] for doc_id, _ in ranked[:RETRIEVE_TOP_K]]

def retrieve(query: str) -> list[dict]:
    db    = get_db()
    table = get_or_create_table(db)

    # --- Dense (semantic) search ---
    q_vec = embedder.encode_query(query)["dense"][0].tolist()
    dense = (
        table.search(q_vec, query_type="vector")
             .limit(RETRIEVE_TOP_K)
             .select(["id", "text", "source", "chunk_idx"])
             .to_list()
    )

    # --- Sparse (BM25 / FTS) search ---
    sparse = (
        table.search(query, query_type="fts")
             .limit(RETRIEVE_TOP_K)
             .select(["id", "text", "source", "chunk_idx"])
             .to_list()
    )

    return rrf_fusion(dense, sparse)

def retrieve_and_rerank(query: str) -> list[dict]:
    candidates = retrieve(query)          # top-20 fused
    passages   = [c["text"] for c in candidates]

    ranked = bge_reranker.rerank(query, passages)
    # ranked = [(original_idx, score), ...]

    top_docs = [candidates[idx] for idx, _score in ranked]
    return top_docs   # top RERANK_TOP_K (default 5) docs