import sys
from FlagEmbedding import FlagReranker
from config import RERANKER_MODEL, RERANK_TOP_K, DEVICE


class BGEReranker:
    def __init__(self, device: str = DEVICE):
        self.model = FlagReranker(
            RERANKER_MODEL,
            use_fp16=device == "cuda",
            device=device,
        )
        if device != "cuda":
            print(
                "\033[91m[WARNING] Reranker running on CPU — "
                "expect slower reranking.\033[0m",
                file=sys.stderr,
            )

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        pairs = [[query, p] for p in passages]
        scores = self.model.compute_score(pairs, normalize=True)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:RERANK_TOP_K]


reranker = BGEReranker()
