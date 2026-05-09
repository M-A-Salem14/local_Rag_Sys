from FlagEmbedding import BGEM3FlagModel
import numpy as np
import torch
from config import EMBED_MODEL

use_cuda = torch.cuda.is_available()
if use_cuda:
    try:
        _ = torch.zeros(1, device="cuda")
    except RuntimeError:
        use_cuda = False


class BGEEmbedder:
    def __init__(self):
        self.model = BGEM3FlagModel(
            EMBED_MODEL,
            use_fp16=use_cuda,
            device="cuda" if use_cuda else "cpu",
        )

    def encode(self, texts: list[str], batch_size: int = 16) -> dict:
        if not texts:
            return {"dense": np.array([], dtype=np.float32), "sparse": []}
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense":  np.array(output["dense_vecs"], dtype=np.float32),
            "sparse": output["lexical_weights"],
        }

    def encode_query(self, query: str) -> dict:
        return self.encode([query])

# Module-level singleton
embedder = BGEEmbedder()
