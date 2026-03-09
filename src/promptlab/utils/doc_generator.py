"""Document-grounded testcase generator.

Uses the local document index to generate provenance-rich testcases:
1. Extract **domain concepts** from the BSP (not just keywords).
2. Build concept-targeted *retrieval intents*.
3. Retrieve relevant chunks and score domain-relevance.
4. Use a local LLM to turn evidence spans into atomic testcases.
5. Attach provenance metadata (source_doc_id, page, chunk_id, excerpt).
6. Validate provenance completeness and evidence cleanliness.
7. Also generates *constraint-driven* testcases directly from BSP text.
8. Iteratively generate until 100-150 accepted cases.

Falls back to a no-LLM heuristic generator when LLM is unavailable so
the pipeline always produces output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from promptlab.utils.doc_corpus import is_blocked_domain
from promptlab.utils.doc_index import Chunk, DocumentIndex, RetrievalResult

console = Console()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceInfo:
    """Traceability metadata for a single testcase."""
    source_doc_id: str = ""
    source_url: str = ""
    page_number: int = 0
    chunk_id: str = ""
    section: str = ""
    evidence_span: str = ""


@dataclass
class GeneratedCase:
    """A single generated testcase with provenance."""
    case_id: str
    prompt: str
    assertions: list[dict]
    tags: list[str]
    provenance: ProvenanceInfo
    style: str = "qa"  # "qa" | "cloze" | "constraint"
    concept_bucket: str = ""  # which BSP concept this case tests


@dataclass
class RejectionRecord:
    """Tracks why a candidate was rejected (for diagnostics)."""
    reason: str
    prompt_snippet: str = ""
    source_url: str = ""


# ---------------------------------------------------------------------------
# BSP concept extractor
# ---------------------------------------------------------------------------

# Domain-specific concept dictionaries.  The key is the detected domain;
# the value is a list of (concept_label, keywords_for_search) tuples.
_DOMAIN_CONCEPTS: dict[str, list[tuple[str, list[str]]]] = {
    "legal": [
        ("contract_formation", ["contract formation", "offer and acceptance", "consideration", "capacity", "legality"]),
        ("breach_remedies", ["breach of contract", "material breach", "anticipatory breach", "damages", "specific performance"]),
        ("nda", ["non-disclosure agreement", "NDA", "confidentiality obligations", "trade secret"]),
        ("indemnification", ["indemnification clause", "limitation of liability", "hold harmless"]),
        ("force_majeure", ["force majeure", "impossibility of performance", "frustration of purpose"]),
        ("arbitration", ["arbitration clause", "dispute resolution", "mediation", "alternative dispute resolution"]),
        ("employment", ["employment contract", "non-compete clause", "severance", "at-will employment"]),
        ("ip_contracts", ["intellectual property assignment", "licensing terms", "work for hire"]),
        ("jurisdiction", ["choice of law", "jurisdiction clause", "governing law", "forum selection"]),
        ("termination", ["termination clause", "termination for cause", "termination for convenience"]),
        ("enforceability", ["enforceability", "unconscionable contract", "statute of frauds", "parol evidence rule"]),
        ("ucc", ["Uniform Commercial Code", "UCC", "sale of goods", "Article 2"]),
    ],
}

# Fallback: extract concepts ad-hoc from BSP text when no domain-specific
# dictionary is available.
_BSP_CONCEPT_RE = re.compile(
    r"(?:^|\n)\s*\d+\.\s*\*{0,2}([^*:\n]{3,60})\*{0,2}\s*:", re.MULTILINE
)


def extract_bsp_concepts(bsp: str, domain: str) -> list[tuple[str, list[str]]]:
    """Extract domain concepts from the BSP.

    Returns a list of (concept_label, search_keywords) tuples.
    Uses the curated dictionary when available, augmented by any
    additional concepts found directly in the BSP.
    """
    concepts: list[tuple[str, list[str]]] = []

    # Start from curated domain dictionary if available
    if domain in _DOMAIN_CONCEPTS:
        concepts.extend(_DOMAIN_CONCEPTS[domain])

    # Mine the BSP for additional concepts (numbered list items with headings)
    for m in _BSP_CONCEPT_RE.finditer(bsp):
        raw = m.group(1).strip().strip("*").strip()
        # Skip if too short or generic
        if len(raw) < 4 or raw.lower() in {"rules", "format", "response", "structure", "quality", "example"}:
            continue
        label = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        # Avoid duplicates
        existing_labels = {c[0] for c in concepts}
        if label not in existing_labels:
            # Use the heading itself plus words from the subsequent colon description
            keywords = [raw.lower()]
            concepts.append((label, keywords))

    return concepts


# ---------------------------------------------------------------------------
# Retrieval-intent builder
# ---------------------------------------------------------------------------

_INTENT_TEMPLATES: list[tuple[str, str]] = [
    ("definition", "{concept} definition explanation"),
    ("procedure", "{concept} step by step procedure process"),
    ("rule", "{concept} rules requirements legal principles"),
    ("exception", "{concept} exceptions edge cases special circumstances"),
    ("best_practice", "{concept} best practices guidelines"),
    ("pitfall", "{concept} common mistakes pitfalls"),
    ("example", "{concept} examples real world scenarios case study"),
    ("comparison", "{concept} comparison alternatives differences"),
]


def build_retrieval_intents(
    domain: str,
    role: str,
    keywords: list[str],
    capabilities: list[str],
    constraints: list[str],
    *,
    bsp_concepts: list[tuple[str, list[str]]] | None = None,
) -> list[tuple[str, str]]:
    """Return list of (intent_label, query_string) pairs.

    When *bsp_concepts* is provided, build concept-targeted queries
    instead of generic domain queries.
    """
    intents: list[tuple[str, str]] = []
    domain_clean = domain.replace("_", " ")

    if bsp_concepts:
        # Concept-targeted intents (high quality)
        for concept_label, concept_kws in bsp_concepts:
            concept_phrase = " ".join(concept_kws[:3])
            for tmpl_label, tmpl in _INTENT_TEMPLATES:
                intents.append(
                    (f"{concept_label}:{tmpl_label}",
                     tmpl.format(concept=concept_phrase))
                )
    else:
        # Fallback: generic domain queries (original behaviour)
        for label, template in _INTENT_TEMPLATES:
            intents.append((label, template.format(concept=domain_clean)))

    # Keyword-specific queries (limited)
    for kw in keywords[:4]:
        intents.append(("keyword", f"{kw} {domain_clean}"))

    # Capability-based queries
    for cap in capabilities[:3]:
        intents.append(("capability", f"how to {cap}"))

    # Constraint-based queries
    for con in constraints[:3]:
        intents.append(("constraint", f"{domain_clean} {con}"))

    return intents


# ---------------------------------------------------------------------------
# Evidence cleaning
# ---------------------------------------------------------------------------

_ARTIFACT_RE = re.compile(
    r"(/\*\s*Lines?\s*\d+.*?omitted\s*\*/|"
    r"\[\.\.\.?\]|"
    r"<[^>]{1,40}>|"
    r"\{%.*?%\}|"
    r"\{\{.*?\}\}|"
    r"\\[nrt]|"
    r"[\x00-\x08\x0b\x0c\x0e-\x1f])",
    re.IGNORECASE | re.DOTALL,
)

_NOISE_ARTIFACTS_RE = re.compile(
    r"(cookie\s*(consent|policy|banner|hub|preferences|settings)|"
    r"accept\s+all\s+cookies|"
    r"we\s+use\s+cookies|"
    r"sign\s+in\s+to\s+your\s+account|"
    r"subscribe\s+to\s+our\s+newsletter|"
    r"follow\s+us\s+on|"
    r"©\s*\d{4}|"
    r"all\s+rights\s+reserved|"
    r"privacy\s+policy|"
    r"terms\s+(of\s+)?(service|use)|"
    r"sponsored\s+products?|"
    r"your\s+recently\s+viewed|"
    r"customers\s+who\s+bought)",
    re.IGNORECASE,
)


def clean_evidence_span(text: str) -> str:
    """Strip rendering artifacts, control chars, and noise from evidence."""
    if not text:
        return ""
    text = _ARTIFACT_RE.sub("", text)
    text = _NOISE_ARTIFACTS_RE.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Remove trailing ellipsis from truncation
    text = re.sub(r"\.\.\.\s*$", "", text).strip()
    return text


def evidence_has_artifacts(text: str) -> bool:
    """Return True if evidence still contains rendering artifacts."""
    return bool(re.search(r"/\*\s*Lines?\s*\d+.*?omitted\s*\*/", text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------

def provenance_is_complete(prov: "ProvenanceInfo") -> bool:
    """Return True if all mandatory provenance fields are populated.

    BSP-sourced cases (source_doc_id == 'bsp') are exempt from URL/chunk checks.
    """
    if prov.source_doc_id == "bsp":
        return True  # BSP constraints don't need external provenance
    return bool(
        prov.source_doc_id
        and prov.source_url
        and prov.chunk_id
        and prov.evidence_span
    )


# ---------------------------------------------------------------------------
# Domain-relevance scoring
# ---------------------------------------------------------------------------

def _build_concept_keyword_set(concepts: list[tuple[str, list[str]]]) -> set[str]:
    """Flatten concept keywords into a lower-cased set."""
    kw_set: set[str] = set()
    for _, kws in concepts:
        for kw in kws:
            for w in kw.lower().split():
                if len(w) > 2:
                    kw_set.add(w)
    return kw_set


def compute_domain_relevance(text: str, concept_keywords: set[str]) -> float:
    """Score how domain-relevant a piece of text is (0.0-1.0)."""
    if not text or not concept_keywords:
        return 0.5  # neutral when no concepts available
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    if not words:
        return 0.0
    overlap = words & concept_keywords
    return min(1.0, len(overlap) / max(3, len(concept_keywords) * 0.15))


# ---------------------------------------------------------------------------
# LLM-based testcase generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a testcase generation assistant.  Given a DOMAIN, an EVIDENCE snippet from a source document, and an INTENT type, you MUST produce exactly one JSON object with these fields:
- "prompt": a clear, standalone question or fill-in-the-blank sentence that can be used to test an AI assistant in this domain.
- "answer_key": the expected correct answer or phrase (brief, factual).
- "style": either "qa" (question-answer) or "cloze" (fill-in-the-blank with ____ placeholder).
- "difficulty": "easy", "medium", or "hard".

Rules:
1. The testcase MUST be answerable using ONLY the evidence provided.
2. Do NOT include cookie policies, marketing text, or UI copy.
3. Use professional, domain-appropriate language.
4. For "cloze" style, replace exactly one important term in a factual sentence with ____.
5. Return ONLY the JSON object, no markdown fences, no extra text.
"""

_USER_TEMPLATE = """\
DOMAIN: {domain}
INTENT: {intent}
EVIDENCE:
{evidence}

Generate one testcase (JSON):"""


_CONSTRAINT_SYSTEM = """\
You are a testcase generation assistant.  Given a BSP (behavior specification prompt) and a specific CONSTRAINT extracted from it, generate a testcase that verifies the AI follows this constraint.

Return EXACTLY one JSON object:
- "prompt": a question or scenario that would test whether the AI respects this constraint.
- "answer_key": description of the expected behavior (what the AI should or should not do).
- "style": "constraint"
- "difficulty": "medium"

Return ONLY the JSON object, no markdown fences, no extra text.
"""

_CONSTRAINT_USER = """\
BSP CONSTRAINT: {constraint}

BSP context (abbreviated):
{bsp_excerpt}

Generate one constraint testcase (JSON):"""


async def _llm_generate_case(
    runner,
    model: str,
    domain: str,
    intent: str,
    evidence: str,
) -> Optional[dict]:
    """Call local LLM to generate a single testcase dict from evidence."""
    try:
        result = await runner.complete(
            prompt=_USER_TEMPLATE.format(domain=domain, intent=intent, evidence=evidence),
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=500,
        )
        text = result.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception:
        return None


async def _llm_generate_constraint_case(
    runner,
    model: str,
    constraint: str,
    bsp_excerpt: str,
) -> Optional[dict]:
    """Call local LLM to generate a constraint testcase."""
    try:
        result = await runner.complete(
            prompt=_CONSTRAINT_USER.format(constraint=constraint, bsp_excerpt=bsp_excerpt[:800]),
            model=model,
            system_prompt=_CONSTRAINT_SYSTEM,
            temperature=0.3,
            max_tokens=500,
        )
        text = result.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM)
# ---------------------------------------------------------------------------

# Regex for extracting concrete answer phrases from evidence
_CONCRETE_ANSWER_RE = re.compile(
    r"(?:"
    r"(?:known\s+as|called|termed|referred\s+to\s+as|defined\s+as)\s+(?:a |an |the )?([A-Z][A-Za-z0-9 ]{3,50})"
    r"|\"([^\"]{4,50})\""
    r"|'([^']{4,50})'"
    r"|\(([A-Z][A-Z0-9]{1,10})\)"
    r")"
)


def _extract_concrete_answer(evidence: str) -> Optional[str]:
    """Try to extract a concrete named term/phrase from evidence text."""
    m = _CONCRETE_ANSWER_RE.search(evidence)
    if m:
        return next((g for g in m.groups() if g), None)
    # Fallback: look for capitalized multi-word phrases
    cap = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", evidence)
    if cap:
        return max(cap, key=len)[:50]
    return None


def _heuristic_case_from_evidence(
    evidence: str,
    domain: str,
    intent: str,
    idx: int,
) -> Optional[dict]:
    """Create a testcase from evidence using regex heuristics (no LLM)."""
    # Clean evidence before processing
    evidence = clean_evidence_span(evidence)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", evidence) if len(s.strip()) > 30]
    if not sentences:
        return None

    sent = sentences[0]
    # Try definition pattern: "X is/are Y"
    m = re.match(r"^((?:The\s+)?[A-Z][A-Za-z0-9\s]{2,40}?) (?:is|are) (?:a |an |the )?(.{15,})", sent)
    if m:
        subject, definition = m.group(1).strip(), m.group(2).strip()
        return {
            "prompt": f"What is {subject}?",
            "answer_key": definition[:180],
            "style": "qa",
            "difficulty": "easy",
        }

    # Cloze: mask an important word
    words = sent.split()
    candidates = [w for w in words if len(re.sub(r"\W", "", w)) >= 5 and w[0].isupper()]
    if not candidates:
        candidates = [w for w in words if len(re.sub(r"\W", "", w)) >= 6]
    if candidates:
        mask_word = random.choice(candidates)
        clean_mask = re.sub(r"\W$", "", mask_word)
        masked = sent.replace(mask_word, "____", 1)
        return {
            "prompt": f"Fill in the blank with the correct word or term:\n\n{masked}",
            "answer_key": clean_mask,
            "style": "cloze",
            "difficulty": "medium",
        }

    return None


# ---------------------------------------------------------------------------
# Static validator (Stage 1 cheap checks)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"(TODO|FIXME|PLACEHOLDER|INSERT|EXAMPLE|LOREM)", re.IGNORECASE)
_NOISE_RE = re.compile(
    r"(cookie|tracker|analytics|sign\s+in|subscribe|newsletter|free\s+trial|"
    r"marketing|youtube|facebook|instagram|twitter|tiktok|copyright|©|"
    r"add\s+to\s+cart|buy\s+now|shop\s+now|product\s+details|"
    r"recently\s+viewed|browsing\s+history|sponsored\s+products|"
    r"customers\s+who\s+bought|people\s+found\s+this\s+helpful)",
    re.IGNORECASE,
)

# Tautological / self-referential patterns
_TAUTOLOGY_RE = re.compile(
    r"(what is the (primary )?(purpose|function|goal) of (the |this |a )?"
    r"('|\")?[^?]{0,30}('|\")?\s+feature|"
    r"(according|based) (on|upon) the (evidence|text|passage|source|document)|"
    r"as (stated|mentioned|described|noted|indicated) in the|"
    r"per the (evidence|text|passage|document)|"
    r"(from|in) the (evidence|passage|text),?\ |"
    r"the (passage|text|evidence|document) (states?|says?|mentions?|suggests?|indicates?)|"
    r"what does the (evidence|text|passage) (suggest|say|state|mention)|"
    r"what type of (career|job|resource|tool|platform) is (the|this))",
    re.IGNORECASE,
)

# Product/vendor trivia patterns
_PRODUCT_TRIVIA_RE = re.compile(
    r"(what award did|which (award|recognition)|"
    r"according to the \d{4} .{0,30} survey|"
    r"what is the name of the (organization|company|platform|product) that|"
    r"what feature of .{0,40} (platform|software|tool|product)|"
    r"what is the title of the .{0,40} (practice note|product|guide|article)\b)",
    re.IGNORECASE,
)

# Patterns that identify sidebar / PR boilerplate / navigation-only chunks
_SIDEBAR_RE = re.compile(
    r"(recognized in legal \d{3,4}|"  # Law firm award blurbs
    r"advises? .{0,40} on (its|their|the) .{0,30}(billion|million)\.?\s|"  # Deal advisory
    r"ranked #?\d\s|"  # Award rankings
    r"easy,?\s+no.hassle\s+way|"  # Marketing invitations
    r"we can help defend your|"  # Law firm CTA
    r"discover key insights|"  # Marketing
    r"\.docx\s+[A-Z][a-z]|"  # .docx filename artifact (student notes)
    r"(spring|fall|winter|summer)\s+20\d{2}\b|"  # Semester/year metadata
    r"us\$\d+\s*(billion|million)|"  # Deal value announcements
    r"skip to (main content|navigation)|"  # Page-chrome navigation
    r"all rights reserved\.?\s*$)",  # Copyright footer
    re.IGNORECASE,
)


def _chunk_is_substantive(text: str) -> bool:
    """Return True if the chunk contains real educational/informational content.

    Rejects: PR sidebar blurbs, marketing CTAs, table-of-contents heading
    lists, student-note filename artifacts, law firm recognition announcements.
    """
    if _SIDEBAR_RE.search(text):
        return False
    # Reject if chunk is mostly just short headings with no real sentences
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        long_lines = [line for line in lines if len(line) > 60]
        if len(long_lines) / len(lines) < 0.2:  # <20% substantive lines → headings only
            return False
    return True


def _passes_static_validation(case_dict: dict) -> bool:
    """Stage 1: cheap schema + noise filter.  Returns True if acceptable."""
    prompt = case_dict.get("prompt", "")
    answer = case_dict.get("answer_key", "")

    # Length bounds
    if len(prompt) < 15 or len(prompt) > 1500:
        return False
    if len(answer) < 3 or len(answer) > 500:
        return False

    # Noise / placeholder filter
    if _PLACEHOLDER_RE.search(prompt) or _PLACEHOLDER_RE.search(answer):
        return False
    if _NOISE_RE.search(prompt) or _NOISE_RE.search(answer):
        return False

    # Tautological / self-referential filter
    if _TAUTOLOGY_RE.search(prompt):
        return False

    # Content-echo: reject if the answer appears verbatim in the question
    answer_lower = answer.lower().strip()
    if len(answer_lower) >= 8 and answer_lower in prompt.lower():
        return False

    # Product/vendor trivia filter
    if _PRODUCT_TRIVIA_RE.search(prompt):
        return False

    # Style-specific checks
    style = case_dict.get("style", "qa")
    if style == "qa" and "?" not in prompt:
        return False
    if style == "cloze" and "____" not in prompt:
        return False

    return True


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

def _tokenize_for_dedup(text: str) -> set[str]:
    """Extract lower-cased >=3-char word tokens for similarity."""
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup_cases(cases: list[GeneratedCase]) -> list[GeneratedCase]:
    """Remove exact and near-duplicate testcases.

    Three-tier deduplication:
    1. Same source chunk (doc_id + chunk_id already used) → always a dup
    2. Exact normalised prompt match
    3. Jaccard similarity > 0.45 (catches synonym swaps like 'typically'
       vs 'commonly', 'categories' vs 'types')
    """
    seen_keys: set[str] = set()
    seen_chunk_ids: set[tuple[str, str]] = set()  # (doc_id, chunk_id)
    seen_token_sets: list[set[str]] = []
    unique: list[GeneratedCase] = []

    for c in cases:
        # Same-chunk dedup: two cases from the same chunk are always near-dups
        chunk_key = (c.provenance.source_doc_id, c.provenance.chunk_id)
        if chunk_key[1] and chunk_key in seen_chunk_ids:
            continue

        # Exact dedup
        key = re.sub(r"\s+", " ", c.prompt.lower().strip())[:200]
        if key in seen_keys:
            continue

        # Near-duplicate dedup via Jaccard similarity (lowered threshold)
        tokens = _tokenize_for_dedup(c.prompt)
        is_dup = False
        for existing_tokens in seen_token_sets:
            if _jaccard_similarity(tokens, existing_tokens) > 0.45:
                is_dup = True
                break
        if is_dup:
            continue

        if chunk_key[1]:
            seen_chunk_ids.add(chunk_key)
        seen_keys.add(key)
        seen_token_sets.append(tokens)
        unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class DocGroundedGenerator:
    """Orchestrates retrieval → generation → validation → YAML output.

    Quality controls:
    - Concept-balanced retrieval (coverage across BSP concepts)
    - Domain-relevance scoring per chunk
    - Evidence cleaning (artifact removal)
    - Provenance completeness enforcement
    - Blocked-domain rejection
    - Acceptance scoring with rejection tracking
    - Iterative generation until target met
    """

    def __init__(
        self,
        index: DocumentIndex,
        runner=None,
        model: str = "ollama/llama3.1:8b",
        retrieval_top_k: int = 10,
    ):
        self.index = index
        self.runner = runner   # LLMRunner (optional)
        self.model = model
        self.retrieval_top_k = retrieval_top_k
        # Diagnostics
        self.rejections: list[RejectionRecord] = []
        self.concept_coverage: Counter = Counter()

    async def generate(
        self,
        bsp: str,
        domain: str,
        role: str,
        keywords: list[str],
        capabilities: list[str],
        constraints: list[str],
        target_count: int = 100,
        output_dir: Optional[Path] = None,
        output_type: Literal["benchmark", "cloze", "all"] = "all",
    ) -> tuple[list[GeneratedCase], Path]:
        """Generate document-grounded testcases.

        Returns (cases, output_file_path).
        """
        all_cases: list[GeneratedCase] = []
        self.rejections = []
        self.concept_coverage = Counter()

        # ------- Step 0: Extract BSP concepts -------
        bsp_concepts = extract_bsp_concepts(bsp, domain)
        concept_kw_set = _build_concept_keyword_set(bsp_concepts)
        console.print(f"  [green]✓[/green] Extracted {len(bsp_concepts)} BSP concepts: "
                       f"{', '.join(c[0] for c in bsp_concepts[:8])}...")

        # ------- Step 1: Build concept-targeted retrieval intents -------
        console.print("[bold cyan]  Retrieval: building concept-targeted intents...[/bold cyan]")
        intents = build_retrieval_intents(
            domain, role, keywords, capabilities, constraints,
            bsp_concepts=bsp_concepts,
        )
        console.print(f"  [green]✓[/green] {len(intents)} retrieval intents")

        # ------- Step 2: Concept-balanced retrieval -------
        console.print("[bold cyan]  Retrieval: querying index (concept-balanced)...[/bold cyan]")
        retrieved = self._concept_balanced_retrieve(intents, bsp_concepts)
        console.print(f"  [green]✓[/green] {len(retrieved)} unique chunks retrieved")

        if not retrieved:
            console.print("  [yellow]⚠ No chunks retrieved — corpus may be too small[/yellow]")

        # ------- Step 3: Iterative generation with quality gates -------
        console.print("[bold cyan]  Generating testcases from evidence (quality-gated)...[/bold cyan]")

        intent_map = self._map_chunks_to_intents(retrieved, intents)
        use_llm = self.runner is not None

        # We may need multiple passes over chunks with different strategies
        max_iterations = 3
        for iteration in range(max_iterations):
            if len(all_cases) >= target_count:
                break

            if iteration > 0:
                console.print(f"  [cyan]Iteration {iteration+1}: have {len(all_cases)}/{target_count}, "
                              f"need {target_count - len(all_cases)} more...[/cyan]")

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                task = progress.add_task(f"Generating (iter {iteration+1})...", total=len(retrieved))
                for r in retrieved:
                    if len(all_cases) >= target_count:
                        break
                    intent_label = intent_map.get(r.chunk.chunk_id, "knowledge")
                    progress.update(task, description=f"[{intent_label}] {r.chunk.doc_title[:35]}...")

                    # --- Quality gate 1: Blocked domain ---
                    if is_blocked_domain(r.chunk.doc_url):
                        self.rejections.append(RejectionRecord(
                            reason="blocked_domain",
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 1.5: Chunk substantiveness ---
                    if not _chunk_is_substantive(r.chunk.text):
                        self.rejections.append(RejectionRecord(
                            reason="non_substantive_chunk",
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 2: Domain relevance ---
                    relevance = compute_domain_relevance(
                        r.evidence_span + " " + r.chunk.text[:200],
                        concept_kw_set,
                    )
                    if relevance < 0.15:
                        self.rejections.append(RejectionRecord(
                            reason=f"low_domain_relevance ({relevance:.2f})",
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Clean evidence ---
                    cleaned_evidence = clean_evidence_span(r.evidence_span)
                    if len(cleaned_evidence) < 20:
                        self.rejections.append(RejectionRecord(
                            reason="evidence_too_short_after_cleaning",
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Generate case ---
                    case_dict: Optional[dict] = None
                    if use_llm:
                        case_dict = await _llm_generate_case(
                            self.runner, self.model, domain, intent_label, cleaned_evidence,
                        )
                        await asyncio.sleep(0.5)

                    if case_dict is None:
                        # On later iterations, try with more of the chunk text
                        ev = cleaned_evidence if iteration == 0 else clean_evidence_span(r.chunk.text[:500])
                        case_dict = _heuristic_case_from_evidence(ev, domain, intent_label, len(all_cases))

                    if case_dict is None:
                        self.rejections.append(RejectionRecord(
                            reason="generation_failed",
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 3: Static validation ---
                    if not _passes_static_validation(case_dict):
                        self.rejections.append(RejectionRecord(
                            reason="failed_static_validation",
                            prompt_snippet=case_dict.get("prompt", "")[:80],
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 4: Output type filter ---
                    style = case_dict.get("style", "qa")
                    if output_type == "benchmark" and style == "cloze":
                        progress.advance(task)
                        continue
                    if output_type == "cloze" and style == "qa":
                        progress.advance(task)
                        continue

                    # --- Build case with cleaned evidence ---
                    gc = self._dict_to_case(
                        case_dict, r, domain, len(all_cases) + 1,
                        cleaned_evidence=cleaned_evidence,
                        concept_bucket=intent_label.split(":")[0] if ":" in intent_label else intent_label,
                    )

                    # _dict_to_case returns None on invalid metadata or weak assertions
                    if gc is None:
                        self.rejections.append(RejectionRecord(
                            reason="invalid_metadata_or_weak_assertion",
                            prompt_snippet=case_dict.get("prompt", "")[:80],
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 5: Provenance completeness ---
                    if not provenance_is_complete(gc.provenance):
                        self.rejections.append(RejectionRecord(
                            reason="incomplete_provenance",
                            prompt_snippet=gc.prompt[:80],
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 6: Evidence artifact check ---
                    if evidence_has_artifacts(gc.provenance.evidence_span):
                        self.rejections.append(RejectionRecord(
                            reason="evidence_has_artifacts",
                            prompt_snippet=gc.prompt[:80],
                            source_url=r.chunk.doc_url,
                        ))
                        progress.advance(task)
                        continue

                    # --- Quality gate 6.5: Assertion grounding check ---
                    # Verify the 'contains' assertion value is actually present
                    # in the evidence span (prevents hallucinated assertions).
                    if gc.assertions:
                        first = gc.assertions[0]
                        if first.get("type") == "contains":
                            val = first.get("value", "").lower().strip()
                            evidence_lower = gc.provenance.evidence_span.lower()
                            # Accept if value is substring OR has ≥0.4 Jaccard with evidence
                            val_tokens = _tokenize_for_dedup(val)
                            ev_tokens = _tokenize_for_dedup(evidence_lower)
                            grounded = (
                                val in evidence_lower
                                or (val_tokens and _jaccard_similarity(val_tokens, ev_tokens) >= 0.4)
                            )
                            if not grounded:
                                self.rejections.append(RejectionRecord(
                                    reason="ungrounded_assertion",
                                    prompt_snippet=gc.prompt[:80],
                                    source_url=r.chunk.doc_url,
                                ))
                                progress.advance(task)
                                continue

                    all_cases.append(gc)
                    self.concept_coverage[gc.concept_bucket] += 1
                    progress.advance(task)

        console.print(f"  [green]✓[/green] Evidence-based: {len(all_cases)} testcases")

        # ------- Step 4: Constraint-driven testcases -------
        if constraints and len(all_cases) < target_count:
            console.print("[bold cyan]  Generating constraint testcases from BSP...[/bold cyan]")
            constraint_cases = await self._generate_constraint_cases(
                bsp, constraints, domain, target_count - len(all_cases),
            )
            all_cases.extend(constraint_cases)
            console.print(f"  [green]✓[/green] Constraint-based: {len(constraint_cases)} testcases")

        # ------- Step 5: Deduplicate -------
        all_cases = _dedup_cases(all_cases)

        # Trim
        all_cases = all_cases[:target_count]

        # ------- Diagnostics -------
        self._print_quality_scorecard(all_cases, target_count, domain, concept_kw_set)

        if len(all_cases) < target_count:
            console.print(
                f"  [yellow]⚠ Produced {len(all_cases)}/{target_count} "
                f"(need more source documents for full target)[/yellow]"
            )

        # ------- Step 6: Write YAML -------
        output_file = self._write_yaml(all_cases, domain, role, keywords, output_dir)

        return all_cases, output_file

    # ------------------------------------------------------------------
    # Concept-balanced retrieval
    # ------------------------------------------------------------------

    def _concept_balanced_retrieve(
        self,
        intents: list[tuple[str, str]],
        bsp_concepts: list[tuple[str, list[str]]],
    ) -> list[RetrievalResult]:
        """Retrieve chunks ensuring coverage across all BSP concepts.

        Instead of a flat retrieve_multi, we retrieve per concept bucket
        and interleave results to ensure diversity.
        """
        if not bsp_concepts:
            # Fallback: flat retrieval
            intent_queries = [q for _, q in intents]
            return self.index.retrieve_multi(intent_queries, top_k_per_query=self.retrieval_top_k)

        # Group intents by concept
        concept_intents: dict[str, list[str]] = {}
        other_intents: list[str] = []
        for label, query in intents:
            concept_key = label.split(":")[0] if ":" in label else None
            if concept_key:
                concept_intents.setdefault(concept_key, []).append(query)
            else:
                other_intents.append(query)

        # Retrieve per concept bucket
        per_concept: dict[str, list[RetrievalResult]] = {}
        seen_chunks: set[str] = set()

        for concept_key, queries in concept_intents.items():
            results = self.index.retrieve_multi(queries, top_k_per_query=self.retrieval_top_k)
            deduped = []
            for r in results:
                if r.chunk.chunk_id not in seen_chunks:
                    seen_chunks.add(r.chunk.chunk_id)
                    deduped.append(r)
            per_concept[concept_key] = deduped

        # Also retrieve for non-concept intents
        if other_intents:
            other_results = self.index.retrieve_multi(other_intents, top_k_per_query=self.retrieval_top_k)
            for r in other_results:
                if r.chunk.chunk_id not in seen_chunks:
                    seen_chunks.add(r.chunk.chunk_id)
                    per_concept.setdefault("_other", []).append(r)

        # Per-concept cap to prevent over-representation
        num_concepts = max(len(concept_intents), 1)
        per_concept_cap = max(5, int((self.retrieval_top_k * 2) / num_concepts))

        # Round-robin interleave across concepts for balanced coverage
        final: list[RetrievalResult] = []
        concept_keys = list(per_concept.keys())
        concept_emitted: Counter = Counter()
        max_len = max((len(v) for v in per_concept.values()), default=0)
        for i in range(max_len):
            for key in concept_keys:
                if i < len(per_concept[key]) and concept_emitted[key] < per_concept_cap:
                    final.append(per_concept[key][i])
                    concept_emitted[key] += 1

        return final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_chunks_to_intents(
        self,
        retrieved: list[RetrievalResult],
        intents: list[tuple[str, str]],
    ) -> dict[str, str]:
        """Map each chunk_id → best matching intent label (simple overlap)."""
        intent_map: dict[str, str] = {}
        from promptlab.utils.doc_index import _tokenize

        intent_token_sets = [(label, set(_tokenize(q))) for label, q in intents]

        for r in retrieved:
            chunk_tokens = set(_tokenize(r.chunk.text[:500]))
            best_label = "knowledge"
            best_overlap = 0
            for label, itokens in intent_token_sets:
                overlap = len(chunk_tokens & itokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label
            intent_map[r.chunk.chunk_id] = best_label
        return intent_map

    def _dict_to_case(
        self,
        d: dict,
        r: RetrievalResult,
        domain: str,
        idx: int,
        *,
        cleaned_evidence: str = "",
        concept_bucket: str = "",
    ) -> Optional[GeneratedCase]:
        """Convert raw LLM/heuristic dict + retrieval result → GeneratedCase.

        Returns None if the case fails provenance or artifact validation.
        """
        style = d.get("style", "qa")
        prompt = clean_evidence_span(d.get("prompt", ""))  # clean artifacts from prompt
        answer_key = d.get("answer_key", "")

        # Reject if essential retrieval metadata is missing
        if not r.chunk.doc_id or not r.chunk.doc_url or not r.chunk.chunk_id:
            return None

        # Build assertions — prefer concrete over min_length
        assertions: list[dict] = []
        if style == "cloze":
            assertions.append({"type": "contains", "value": answer_key, "case_sensitive": False})
        elif len(answer_key) < 100 and len(answer_key) >= 3:
            assertions.append({"type": "contains", "value": answer_key[:50], "case_sensitive": False})
        else:
            # Try to extract a concrete phrase from evidence instead of min_length
            concrete = _extract_concrete_answer(cleaned_evidence or r.evidence_span)
            if concrete:
                assertions.append({"type": "contains", "value": concrete[:50], "case_sensitive": False})
            else:
                # No concrete answer possible — reject rather than emit weak assertion
                return None

        tags = [domain.lower(), style]
        difficulty = d.get("difficulty", "medium")
        if difficulty:
            tags.append(difficulty)
        if concept_bucket:
            tags.append(concept_bucket)

        # Use cleaned evidence if provided
        ev_span = cleaned_evidence or clean_evidence_span(r.evidence_span)

        provenance = ProvenanceInfo(
            source_doc_id=r.chunk.doc_id,
            source_url=r.chunk.doc_url,
            page_number=r.chunk.page_estimate,
            chunk_id=r.chunk.chunk_id,
            section=r.chunk.section_hint,
            evidence_span=ev_span[:300],
        )

        case_id = f"{domain.lower().replace(' ', '-')}-{style}-{idx}"
        return GeneratedCase(
            case_id=case_id,
            prompt=prompt,
            assertions=assertions,
            tags=tags,
            provenance=provenance,
            style=style,
            concept_bucket=concept_bucket,
        )

    async def _generate_constraint_cases(
        self,
        bsp: str,
        constraints: list[str],
        domain: str,
        max_cases: int,
    ) -> list[GeneratedCase]:
        """Generate testcases from BSP constraints."""
        cases: list[GeneratedCase] = []
        bsp_excerpt = bsp[:1200]

        for i, constraint in enumerate(constraints):
            if len(cases) >= max_cases:
                break

            case_dict: Optional[dict] = None
            if self.runner:
                case_dict = await _llm_generate_constraint_case(
                    self.runner, self.model, constraint, bsp_excerpt,
                )
                await asyncio.sleep(0.5)

            if case_dict is None:
                # Heuristic fallback
                case_dict = {
                    "prompt": f"Given the constraint '{constraint}', how should the assistant respond?",
                    "answer_key": constraint,
                    "style": "constraint",
                    "difficulty": "medium",
                }

            if case_dict and _passes_static_validation(case_dict):
                provenance = ProvenanceInfo(
                    source_doc_id="bsp",
                    source_url="",
                    page_number=0,
                    chunk_id=f"bsp-constraint-{i}",
                    section="BSP Constraints",
                    evidence_span=clean_evidence_span(constraint[:300]),
                )
                gc = GeneratedCase(
                    case_id=f"{domain.lower().replace(' ', '-')}-constraint-{i+1}",
                    prompt=case_dict.get("prompt", ""),
                    assertions=[{"type": "min_length", "value": "30"}],
                    tags=[domain.lower(), "constraint"],
                    provenance=provenance,
                    style="constraint",
                    concept_bucket="constraint",
                )
                cases.append(gc)
        return cases

    # ------------------------------------------------------------------
    # Quality scorecard
    # ------------------------------------------------------------------

    def _print_quality_scorecard(
        self,
        cases: list[GeneratedCase],
        target: int,
        domain: str,
        concept_keywords: set[str],
    ) -> None:
        """Print a diagnostic quality scorecard to the console."""
        total = len(cases)
        rejected = len(self.rejections)
        rejection_reasons = Counter(r.reason for r in self.rejections)

        # Domain-specific ratio
        domain_specific = 0
        for c in cases:
            relevance = compute_domain_relevance(c.prompt + " " + c.provenance.evidence_span, concept_keywords)
            if relevance >= 0.3:
                domain_specific += 1
        domain_pct = (domain_specific / total * 100) if total else 0

        # Source diversity
        source_domains = Counter()
        for c in cases:
            from urllib.parse import urlparse
            if c.provenance.source_url:
                source_domains[urlparse(c.provenance.source_url).netloc] += 1

        console.print("\n[bold cyan]  ── Quality Scorecard ──[/bold cyan]")
        console.print(f"  Accepted:       {total}/{target}")
        console.print(f"  Rejected:       {rejected}")
        for reason, count in rejection_reasons.most_common(5):
            console.print(f"    └ {reason}: {count}")
        console.print(f"  Domain-specific: {domain_specific}/{total} ({domain_pct:.0f}%)")
        console.print(f"  Concept buckets: {dict(self.concept_coverage)}")
        console.print(f"  Source domains:  {len(source_domains)} unique")
        for dom, cnt in source_domains.most_common(5):
            console.print(f"    └ {dom}: {cnt}")

    # ------------------------------------------------------------------
    # YAML output
    # ------------------------------------------------------------------

    def _write_yaml(
        self,
        cases: list[GeneratedCase],
        domain: str,
        role: str,
        keywords: list[str],
        output_dir: Optional[Path],
    ) -> Path:
        """Write testcases as a backward-compatible YAML file + provenance."""
        if output_dir is None:
            output_dir = Path.cwd() / "temp"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_slug = domain.lower().replace(" ", "_")

        yaml_cases = []
        for c in cases:
            entry: dict = {
                "id": c.case_id,
                "prompt": clean_evidence_span(c.prompt),  # final sanitisation
                "assertions": c.assertions,
                "tags": c.tags,
            }
            # Provenance in a non-breaking sub-key — final sanitisation pass
            entry["provenance"] = {
                "source_doc_id": c.provenance.source_doc_id,
                "source_url": c.provenance.source_url,
                "page_number": c.provenance.page_number,
                "chunk_id": c.provenance.chunk_id,
                "section": clean_evidence_span(c.provenance.section),
                "evidence_span": clean_evidence_span(c.provenance.evidence_span),
            }
            yaml_cases.append(entry)

        yaml_content = {
            "metadata": {
                "name": f"Auto-Generated Tests: {domain.title()}",
                "description": (
                    f"Document-grounded tests from BSP analysis. "
                    f"Domain: {domain}. Role: {role[:80]}"
                ),
                "generated_at": datetime.now().isoformat(),
                "source": "promptlab-docs-web-generator",
                "generation_mode": "docs_web",
                "test_count": len(yaml_cases),
                "keywords": keywords[:5],
            },
            "defaults": {"temperature": 0},
            "cases": yaml_cases,
        }

        output_file = output_dir / f"auto_generated_{domain_slug}_{timestamp}.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return output_file
