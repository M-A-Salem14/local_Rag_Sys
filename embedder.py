from FlagEmbedding import BGEM3FlagModel
import numpy as np
from config import EMBED_MODEL

class BGEEmbedder:
    def __init__(self):
        # use_fp16=True halves VRAM — safe on RTX 5070
        self.model = BGEM3FlagModel(
            EMBED_MODEL,
            use_fp16=True,
            device="cuda",
        )

    def encode(self, texts: list[str], batch_size: int = 16) -> dict:
        """
        Returns a dict with:
          'dense_vecs'  : np.ndarray (N, 1024)  — for ANN search
          'lexical_weights': list[dict]          — for sparse search
        """
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,   # off unless you need ColBERT
        )
        return {
            "dense":  np.array(output["dense_vecs"], dtype=np.float32),
            "sparse": output["lexical_weights"],  # list of {token_id: weight}
        }

    def encode_query(self, query: str) -> dict:
        return self.encode([query])

# Module-level singleton
embedder = BGEEmbedder()