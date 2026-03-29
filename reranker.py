from FlagEmbedding import FlagReranker
from config import RERANKER_MODEL, RERANK_TOP_K

class BGEReranker:
    def __init__(self):
        self.model = FlagReranker(
            RERANKER_MODEL,
            use_fp16=True,
            device="cuda",
        )

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """
        Returns indices of top-K passages sorted by relevance score (descending).
        """
        pairs = [[query, p] for p in passages]
        scores = self.model.compute_score(pairs, normalize=True)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:RERANK_TOP_K]  # (original_idx, score)

reranker = BGEReranker()