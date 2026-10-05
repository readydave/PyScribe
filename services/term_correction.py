"""Names / terms correction for ASR backends that have no hotword biasing (Nemotron streaming).

Each listed term is fuzzy-matched against runs of transcript words; near misses ("Peter son" or
"Petersen" for "Peterson") are replaced by the term as typed, keeping the run's timing and any
surrounding punctuation.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

MIN_TERM_CHARS = 4
# Allowed edit distance between a transcript run and a term: one per 5 letters (at least one).
EDIT_LETTERS = 5
_EDGE_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def parse_terms(text: str | None) -> list[str]:
    """Split a comma/newline separated names-and-terms field into unique terms long enough to match safely."""
    seen: set[str] = set()
    terms: list[str] = []
    for part in re.split(r"[,\n]", str(text or "")):
        term = " ".join(part.split())
        key = _norm(term)
        if len(key) >= MIN_TERM_CHARS and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def apply_terms(words: list[dict[str, Any]], terms: Sequence[str]) -> list[dict[str, Any]]:
    """Return `words` (dicts with `text`/`start`/`end`) with near-miss spellings of `terms` replaced."""
    prepared = [(term, _norm(term), max(1, len(term.split()))) for term in terms if len(_norm(term)) >= MIN_TERM_CHARS]
    if not prepared or not words:
        return words
    normalized = {key for _, key, _ in prepared}
    out: list[dict[str, Any]] = []
    index = 0
    while index < len(words):
        best: tuple[int, str, int] | None = None
        for term, key, count in prepared:
            for size in sorted({max(1, count - 1), count, count + 1}):
                run = words[index : index + size]
                if len(run) < size:
                    continue
                candidate = _norm("".join(w["text"] for w in run))
                if candidate in normalized and candidate != key:
                    continue  # already another listed term
                distance = _edit_distance(candidate, key)
                if distance <= max(1, len(key) // EDIT_LETTERS) and (best is None or (distance, size) < (best[0], best[2])):
                    best = (distance, term, size)
        if best is None:
            out.append(words[index])
            index += 1
            continue
        _, term, size = best
        run = words[index : index + size]
        lead = _EDGE_RE.match(run[0]["text"]).group(1)  # type: ignore[union-attr]
        trail = _EDGE_RE.match(run[-1]["text"]).group(3)  # type: ignore[union-attr]
        out.append({**run[0], "text": f"{lead}{term}{trail}", "end": run[-1]["end"]})
        index += size
    return out
