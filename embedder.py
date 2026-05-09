import sys
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from config import EMBED_MODEL, DEVICE


class BGEEmbedder:
    def __init__(self, device: str = DEVICE):
        use_fp16 = device == "cuda"
        self.model = BGEM3FlagModel(
            EMBED_MODEL,
            use_fp16=use_fp16,
            device=device,
        )
        if device != "cuda":
            print(
                "\033[91m[WARNING] Embedder running on CPU — "
                "expect slower indexing and queries.\033[0m",
                file=sys.stderr,
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


embedder = BGEEmbedder()
