"""Lightweight document index with chunk-level retrieval.

Implements a self-contained TF-IDF + cosine-similarity index over text
chunks carved from corpus documents.  Each chunk preserves provenance
metadata (doc_id, page/section estimate, character offsets) so that
downstream test generators can attach traceability to every testcase.

No external vector-DB or embedding-model dependency — runs 100 % locally
with only the Python standard library + a minimal math layer.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from promptlab.utils.doc_corpus import CorpusDocument

console = Console()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text chunk with provenance metadata."""
    chunk_id: str
    doc_id: str
    doc_url: str
    doc_title: str
    text: str
    char_start: int
    char_end: int
    page_estimate: int  # rough page number (chars / ~3000)
    section_hint: str   # nearest heading or empty string
    token_count: int    # rough word count


@dataclass
class RetrievalResult:
    """A scored chunk returned by the index."""
    chunk: Chunk
    score: float        # cosine similarity
    evidence_span: str  # short excerpt highlighting the match


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    """Split a document into overlapping chunks with metadata."""

    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.chunk_size = chunk_size   # in *words*
        self.overlap = overlap

    # Heading regex — common Markdown / plain-text heading patterns.
    _HEADING_RE = re.compile(
        r"^(?:#{1,4}\s+|[A-Z][A-Z ]{3,}$|[A-Z][a-z].{5,60}:?\s*$)", re.MULTILINE
    )

    def chunk_document(self, doc: CorpusDocument) -> list[Chunk]:
        """Split *doc.text* into Chunk objects."""
        text = doc.text
        if not text:
            return []

        words = text.split()
        total_words = len(words)
        chunks: list[Chunk] = []
        start_word = 0

        while start_word < total_words:
            end_word = min(start_word + self.chunk_size, total_words)
            chunk_words = words[start_word:end_word]
            chunk_text = " ".join(chunk_words)

            # Character offsets (approximate via word count proportions).
            char_start = int(start_word / total_words * len(text)) if total_words else 0
            char_end = int(end_word / total_words * len(text)) if total_words else len(text)

            # Page estimate (~3000 chars per page).
            page_estimate = max(1, char_start // 3000 + 1)

            # Section hint: last heading seen before this chunk.
            section_hint = self._find_nearest_heading(text, char_start)

            chunk_id = hashlib.md5(f"{doc.doc_id}:{start_word}".encode()).hexdigest()[:10]

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                doc_url=doc.url,
                doc_title=doc.title,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                page_estimate=page_estimate,
                section_hint=section_hint,
                token_count=len(chunk_words),
            ))

            step = max(1, self.chunk_size - self.overlap)
            start_word += step

        return chunks

    def _find_nearest_heading(self, text: str, pos: int) -> str:
        best = ""
        for m in self._HEADING_RE.finditer(text):
            if m.start() <= pos:
                best = m.group(0).strip().lstrip("#").strip()
            else:
                break
        return best[:120]


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could to of in for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once here there when where "
    "why how all each every both few more most other some such no nor "
    "not only own same so than too very i me my we our you your he him "
    "his she her it its they them their what which who whom this that "
    "these those am and but if or because until while about against "
    "also just don t s d ll ve re".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenizer with stop-word removal."""
    return [w for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in _STOP]


class DocumentIndex:
    """TF-IDF index over document chunks for lightweight RAG retrieval."""

    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.chunks: list[Chunk] = []
        self._tfidf_vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build(self, docs: list[CorpusDocument]) -> int:
        """Build the index from corpus documents.  Returns chunk count."""
        self.chunks = []
        for doc in docs:
            self.chunks.extend(self.chunker.chunk_document(doc))

        if not self.chunks:
            return 0

        console.print(f"  [green]✓[/green] Chunked {len(docs)} docs → {len(self.chunks)} chunks")

        # Build TF vectors per chunk.
        tf_vectors: list[Counter] = []
        for chunk in self.chunks:
            tf_vectors.append(Counter(_tokenize(chunk.text)))

        # Compute IDF.
        n = len(self.chunks)
        df: Counter = Counter()
        for tf in tf_vectors:
            for term in tf:
                df[term] += 1
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}

        # Compute TF-IDF vectors.
        self._tfidf_vectors = []
        for tf in tf_vectors:
            vec: dict[str, float] = {}
            for term, count in tf.items():
                vec[term] = (1 + math.log(count)) * self._idf.get(term, 1.0)
            self._tfidf_vectors.append(vec)

        return len(self.chunks)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Return the *top_k* most relevant chunks for *query*."""
        if not self.chunks:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_vec: dict[str, float] = {}
        for term, count in q_tf.items():
            q_vec[term] = (1 + math.log(count)) * self._idf.get(term, 1.0)

        scored: list[tuple[int, float]] = []
        for idx, d_vec in enumerate(self._tfidf_vectors):
            score = _cosine(q_vec, d_vec)
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for idx, score in scored[:top_k]:
            chunk = self.chunks[idx]
            evidence = self._extract_evidence(chunk.text, q_tokens)
            results.append(RetrievalResult(chunk=chunk, score=score, evidence_span=evidence))

        return results

    def retrieve_multi(self, queries: list[str], top_k_per_query: int = 5) -> list[RetrievalResult]:
        """Run multiple queries, deduplicate by chunk_id, return union."""
        seen: set[str] = set()
        results: list[RetrievalResult] = []
        for q in queries:
            for r in self.retrieve(q, top_k=top_k_per_query):
                if r.chunk.chunk_id not in seen:
                    seen.add(r.chunk.chunk_id)
                    results.append(r)
        # sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_evidence(text: str, q_tokens: list[str], max_len: int = 250) -> str:
        """Extract a short span of *text* that best overlaps with *q_tokens*."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return text[:max_len]

        best_sent = ""
        best_overlap = 0
        q_set = set(q_tokens)
        for s in sentences:
            s_tokens = set(_tokenize(s))
            overlap = len(s_tokens & q_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent = s
        evidence = best_sent or sentences[0]
        if len(evidence) > max_len:
            evidence = evidence[:max_len] + "..."
        return evidence


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(a[k] * b[k] for k in a if k in b)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
