import uuid, re
from pathlib import Path
from tqdm import tqdm
from embedder import embedder
from db_init import get_db, get_or_create_table
from config import (
    DATA_DIR, CHUNKING_LOG_MODE, CHUNKING_LOG_DIR, CHUNKING_STRATEGY,
    CHUNK_MAX_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_SENT,
)
from chunking_logger import ChunkingLogger
from markdown_chunker import chunk_markdown


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64
) -> list[str]:
    """Token-approximate sliding window chunking (legacy)."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _chunk_with_strategy(text: str) -> list[str]:
    if CHUNKING_STRATEGY == "markdown":
        return chunk_markdown(text, CHUNK_MAX_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_SENT)
    return chunk_text(text)


def index_directory(data_dir: Path = DATA_DIR, batch_size: int = 32):
    db    = get_db()
    table = get_or_create_table(db)

    files = list(data_dir.glob("**/*.txt")) + list(data_dir.glob("**/*.md"))
    print(f"Found {len(files)} files to index | Strategy: {CHUNKING_STRATEGY}")

    logger = ChunkingLogger(CHUNKING_LOG_MODE, CHUNKING_LOG_DIR, strategy=CHUNKING_STRATEGY)
    all_chunks, all_meta = [], []

    for file_idx, fpath in enumerate(tqdm(files, desc="Chunking")):
        text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            logger.log_file(fpath, total_words=0, chunk_count=0, skipped=True)
            continue

        words = text.split()
        logger.log_file(fpath, total_words=len(words), chunk_count=0, skipped=False)
        chunks = _chunk_with_strategy(text)

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            all_chunks.append(chunk)
            all_meta.append({
                "id":        str(uuid.uuid4()),
                "source":    str(fpath),
                "chunk_idx": idx,
            })

            logger.log_chunk(idx, chunk)

            # Overlap check only for legacy strategy
            if CHUNKING_STRATEGY == "legacy" and idx > 0:
                prev_words = chunks[idx - 1].split()
                curr_words = chunk.split()
                logger.log_overlap(file_idx, prev_words, curr_words, 64)

        # Update file entry with actual chunk count
        logger.files[-1]["chunk_count"] = idx + 1 if chunks else 0

    # Embed in batches
    records = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch_texts = all_chunks[i : i + batch_size]
        batch_meta  = all_meta[i : i + batch_size]
        vecs        = embedder.encode(batch_texts)["dense"]

        for j, (text, meta, vec) in enumerate(zip(batch_texts, batch_meta, vecs)):
            records.append({
                **meta,
                "text":   text,
                "vector": vec.tolist(),
            })

    table.add(records)
    print(f"Indexed {len(records)} chunks into LanceDB")
    logger.write()

if __name__ == "__main__":
    index_directory()
