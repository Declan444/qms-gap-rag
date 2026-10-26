"""Cut markdown documents into chunks at their headings, keeping where each chunk came from."""
import re
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


def split_markdown(path: Path, max_words: int) -> list[dict]:
    """Return chunks with metadata: doc, doc_title, section, text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_title, section_path, buffer, chunks = path.stem, [], [], []

    def flush():
        text = " ".join(l.strip() for l in buffer if l.strip())
        if not text:
            return
        words = text.split()
        for i in range(0, len(words), max_words):
            part = " ".join(words[i:i + max_words])
            chunks.append({
                "doc": path.stem,
                "doc_title": doc_title,
                "section": " > ".join(section_path) or "(intro)",
                "text": part,
            })

    for line in lines:
        m = HEADING.match(line)
        if m:
            flush()
            buffer = []
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:
                doc_title, section_path = title, []
            else:
                section_path = section_path[: level - 2] + [title]
        else:
            buffer.append(line)
    flush()
    return chunks


def parse_checklist(path: Path) -> list[dict]:
    """Each '## (x) Title' heading plus its paragraph becomes one requirement."""
    items, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+\((\w)\)\s+(.*)$", line)
        if m:
            current = {"id": m.group(1), "title": m.group(2).strip(), "text": ""}
            items.append(current)
        elif current and line.strip():
            current["text"] += (" " if current["text"] else "") + line.strip()
    return items
