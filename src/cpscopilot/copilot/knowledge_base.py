"""Security knowledge base: Markdown playbooks with YAML front matter.

Each ``*.md`` file in the knowledge-base directory looks like::

    ---
    id: PB-SPOOF
    title: Sensor spoofing / false data injection
    tags: [sensor, spoofing, integrity]
    references: [...]
    ---
    ## Indicators
    ...
    ## Investigation steps
    1. ...

Documents are split into one chunk per ``##`` section (long sections are
further split into overlapping word windows). Chunk ids are deterministic
(``PB-SPOOF#investigation-steps``), so citations are stable across runs. The
retrieval index is built from the knowledge base only - never from telemetry
- so it cannot leak test data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cpscopilot.utils.io import sha256_file

__all__ = ["KBChunk", "kb_fingerprint", "load_knowledge_base", "numbered_steps"]

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_STEP = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")


@dataclass(frozen=True)
class KBChunk:
    """One retrievable unit of guidance."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)

    @property
    def requires_approval(self) -> bool:
        """Sections about containment/response always need analyst approval."""
        s = self.section.lower()
        return any(k in s for k in ("approval", "containment", "response", "recovery"))

    def retrieval_text(self) -> str:
        return " ".join([self.title, self.section, " ".join(self.tags), self.text])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def numbered_steps(text: str) -> list[str]:
    """Extract ``1. ...`` list items from chunk text."""
    return [m.group(1) for line in text.splitlines() if (m := _STEP.match(line))]


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    lines = text.splitlines()
    if len(text.split()) <= size:
        return [text]
    # keep list items intact: window over lines, not raw words
    windows: list[str] = []
    current: list[str] = []
    count = 0
    for line in lines:
        n = len(line.split())
        if current and count + n > size:
            windows.append("\n".join(current))
            tail: list[str] = []
            tail_count = 0
            for prev in reversed(current):
                tail_count += len(prev.split())
                if tail_count > overlap:
                    break
                tail.insert(0, prev)
            current, count = tail, sum(len(x.split()) for x in tail)
        current.append(line)
        count += n
    if current:
        windows.append("\n".join(current))
    return windows


def load_knowledge_base(directory: Path, chunk_words: int = 140,
                        chunk_overlap_words: int = 30) -> list[KBChunk]:
    """Parse and chunk every Markdown playbook in ``directory`` (sorted by name)."""
    directory = Path(directory)
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no knowledge-base documents in {directory}")
    chunks: list[KBChunk] = []
    seen: set[str] = set()
    for path in files:
        if path.name.lower() == "readme.md":
            continue
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER.match(raw)
        if not match:
            raise ValueError(f"{path} lacks YAML front matter")
        meta = yaml.safe_load(match.group(1)) or {}
        body = match.group(2)
        doc_id = str(meta["id"])
        title = str(meta.get("title", doc_id))
        tags = tuple(str(t) for t in meta.get("tags", []))
        refs = tuple(str(r) for r in meta.get("references", []))
        sections = re.split(r"^##\s+", body, flags=re.MULTILINE)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading, _, content = section.partition("\n")
            parts = _split_words(content.strip(), chunk_words, chunk_overlap_words)
            for k, part in enumerate(parts):
                suffix = "" if len(parts) == 1 else f"-{k + 1}"
                chunk_id = f"{doc_id}#{_slug(heading)}{suffix}"
                if chunk_id in seen:
                    raise ValueError(f"duplicate chunk id {chunk_id}")
                seen.add(chunk_id)
                chunks.append(KBChunk(chunk_id, doc_id, title, heading.strip(), part.strip(),
                                      tags, refs))
    return chunks


def kb_fingerprint(directory: Path) -> dict[str, str]:
    """SHA-256 of every knowledge-base file (recorded with each run)."""
    return {p.name: sha256_file(p) for p in sorted(Path(directory).glob("*.md"))}
