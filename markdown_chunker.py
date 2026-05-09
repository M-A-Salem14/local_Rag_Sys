import re


def chunk_markdown(
    text: str,
    max_words: int = 512,
    min_words: int = 50,
    overlap_sent: int = 1,
) -> list[str]:
    if not text.strip():
        return []

    blocks = _parse_blocks(text)
    sections = _group_into_sections(blocks)

    if not sections:
        return []

    sections = _merge_small(sections, min_words, max_words)

    chunks = []
    for section in sections:
        if section["word_count"] <= max_words:
            chunks.append(section["content"].strip())
        else:
            chunks.extend(_split_oversized(section, max_words, overlap_sent))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Block parser — splits raw markdown into typed blocks
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"(`{3,})")
_TABLE_ROW_RE = re.compile(r"^\|.*\|$", re.MULTILINE)


def _parse_blocks(text: str) -> list[dict]:
    lines = text.split("\n")
    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            blocks.append({"type": "heading", "content": line, "level": level})
            i += 1
            continue

        # Fenced code block
        fence_m = _CODE_FENCE_RE.match(line.strip())
        if fence_m:
            fence = fence_m.group(1)
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith(fence) and i > 0:
                    i += 1
                    break
                i += 1
            blocks.append({"type": "code", "content": "\n".join(code_lines)})
            continue

        # Table (consecutive | rows)
        if _TABLE_ROW_RE.match(line.strip()):
            table_lines = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "content": "\n".join(table_lines)})
            continue

        # Empty line — skip
        if not line.strip():
            i += 1
            continue

        # Paragraph (collect until blank line or structural element)
        para_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                break
            if _HEADING_RE.match(next_line):
                break
            if _CODE_FENCE_RE.match(next_line.strip()):
                break
            if _TABLE_ROW_RE.match(next_line.strip()):
                break
            para_lines.append(next_line)
            i += 1
        blocks.append({"type": "paragraph", "content": "\n".join(para_lines)})

    return blocks


# ---------------------------------------------------------------------------
# Section builder — groups blocks under their nearest heading
# ---------------------------------------------------------------------------

def _group_into_sections(blocks: list[dict]) -> list[dict]:
    sections = []
    current = None

    for block in blocks:
        if block["type"] == "heading":
            if current is not None:
                sections.append(current)
            current = {
                "heading": block["content"],
                "content": block["content"],
                "word_count": len(block["content"].split()),
            }
        else:
            if current is None:
                current = {"heading": None, "content": "", "word_count": 0}
            current["content"] += "\n" + block["content"]
            current["word_count"] += len(block["content"].split())

    if current is not None:
        sections.append(current)

    return sections


# ---------------------------------------------------------------------------
# Small-section merger — combines sections below min_words threshold
# ---------------------------------------------------------------------------

def _merge_small(sections: list[dict], min_words: int, max_words: int) -> list[dict]:
    if not sections:
        return []

    merged = [sections[0]]

    for section in sections[1:]:
        prev = merged[-1]
        combined_words = prev["word_count"] + section["word_count"]

        if combined_words <= max_words * 1.5:
            # Merge if previous is small OR if current section is also small
            if prev["word_count"] < min_words or section["word_count"] < min_words:
                prev["content"] += "\n\n" + section["content"]
                prev["word_count"] = combined_words
                continue

        merged.append(section)

    return merged


# ---------------------------------------------------------------------------
# Oversized splitter — paragraph → sentence → character fallback
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_oversized(section: dict, max_words: int, overlap_sent: int) -> list[str]:
    content = section["content"]
    words = content.split()

    # If it's a single code block or table that's oversized, keep it intact
    if content.strip().startswith("```") or content.strip().startswith("|"):
        return [content.strip()]

    # Level 1: Split by paragraphs
    paragraphs = _split_paragraphs(content)
    chunks = _accumulate(paragraphs, max_words, by="paragraph")

    # Check if any chunk is still oversized
    final = []
    for chunk in chunks:
        if len(chunk.split()) <= max_words:
            final.append(chunk)
        else:
            # Level 2: Split by sentences
            sentences = _split_sentences(chunk)
            sub = _accumulate_sentences(sentences, max_words, overlap_sent)
            final.extend(sub)

    return final


def _accumulate(parts: list[str], max_words: int, by: str = "paragraph") -> list[str]:
    chunks = []
    current = ""

    for part in parts:
        candidate = (current + "\n\n" + part).strip() if current else part
        if len(candidate.split()) <= max_words:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def _accumulate_sentences(sentences: list[str], max_words: int, overlap_sent: int) -> list[str]:
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current:
            chunks.append(" ".join(current))
            # Keep last N sentences as overlap
            current = current[-overlap_sent:] if overlap_sent > 0 else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    return chunks
