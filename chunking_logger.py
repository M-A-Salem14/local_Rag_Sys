import statistics
from datetime import datetime
from pathlib import Path


class ChunkingLogger:
    def __init__(self, mode: str, output_dir: Path, strategy: str = "legacy"):
        self.mode = mode  # "metadata" | "full"
        self.strategy = strategy  # "legacy" | "markdown"
        self.output_dir = output_dir
        self.files = []
        self.anomalies = []
        self.all_chunk_word_counts = []
        self.all_chunk_char_counts = []

    def log_file(self, path: Path, total_words: int, chunk_count: int, skipped: bool):
        self.files.append({
            "path": path,
            "total_words": total_words,
            "chunk_count": chunk_count,
            "skipped": skipped,
            "chunks": [],
        })

    def log_chunk(self, idx: int, content: str):
        words = content.split()
        word_count = len(words)
        char_count = len(content)
        self.all_chunk_word_counts.append(word_count)
        self.all_chunk_char_counts.append(char_count)

        entry = {
            "idx": idx,
            "word_count": word_count,
            "char_count": char_count,
            "first_50_chars": content[:50].replace("\n", " "),
            "last_50_chars": content[-50:].replace("\n", " ") if len(content) > 50 else "",
        }

        if self.mode == "full":
            entry["content"] = content

        current_file = self.files[-1]
        current_file["chunks"].append(entry)

        if word_count < 50:
            self.anomalies.append(
                f"[!] {current_file['path']} chunk {idx}: "
                f"only {word_count} words (possible formatting issue)"
            )

    def log_overlap(self, file_idx: int, chunk_a_words: list[str], chunk_b_words: list[str], overlap_size: int):
        expected = chunk_a_words[-overlap_size:]
        actual = chunk_b_words[:overlap_size]
        matched = expected == actual
        if not matched:
            current_file = self.files[file_idx]
            chunk_idx = len(current_file["chunks"]) - 1
            self.anomalies.append(
                f"[!] {current_file['path']} chunk {chunk_idx}: "
                f"overlap mismatch with next chunk (expected {overlap_size} words)"
            )

    def write(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"chunking_{timestamp}.md"

        total_chunks = sum(f["chunk_count"] for f in self.files)
        skipped_count = sum(1 for f in self.files if f["skipped"])

        lines = [
            f"# Chunking Debug Log",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mode: {self.mode} | Strategy: {self.strategy}",
            f"Files scanned: {len(self.files)} | Skipped: {skipped_count} | Total chunks: {total_chunks}",
            "",
        ]

        # Distribution summary
        if self.all_chunk_word_counts:
            lines.append("## Distribution Summary")
            lines.append("")
            lines.append("| Metric  | Words  | Chars    |")
            lines.append("|---------|--------|----------|")
            lines.append(f"| Min     | {min(self.all_chunk_word_counts):<6} | {min(self.all_chunk_char_counts):<8} |")
            lines.append(f"| Max     | {max(self.all_chunk_word_counts):<6} | {max(self.all_chunk_char_counts):<8} |")
            lines.append(f"| Mean    | {statistics.mean(self.all_chunk_word_counts):<6.0f} | {statistics.mean(self.all_chunk_char_counts):<8.0f} |")
            lines.append(f"| Median  | {statistics.median(self.all_chunk_word_counts):<6.0f} | {statistics.median(self.all_chunk_char_counts):<8.0f} |")
            lines.append("")

        # Anomalies
        if self.anomalies:
            lines.append(f"## Anomalies ({len(self.anomalies)})")
            lines.append("")
            for a in self.anomalies:
                lines.append(f"- {a}")
            lines.append("")

        # Per-file details
        lines.append("## Files")
        lines.append("")

        for f in self.files:
            if f["skipped"]:
                lines.append(f"### {f['path']}")
                lines.append(f"- **SKIPPED** (empty or whitespace-only)")
                lines.append("")
                continue

            lines.append(f"### {f['path']}")
            lines.append(f"- Words: {f['total_words']:,} | Chunks: {f['chunk_count']}")
            lines.append("")

            for c in f["chunks"]:
                lines.append(f"#### Chunk {c['idx']}")
                lines.append(f"- Words: {c['word_count']} | Chars: {c['char_count']}")
                lines.append(f"- Starts: `{c['first_50_chars']}...`")
                if c['last_50_chars']:
                    lines.append(f"- Ends: `...{c['last_50_chars']}`")

                if self.mode == "full" and "content" in c:
                    lines.append("")
                    lines.append("**Content:**")
                    lines.append(f"> {c['content'].replace(chr(10), chr(10) + '> ')}")

                lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        print(f"Chunking log written to {filepath}")
        return filepath
