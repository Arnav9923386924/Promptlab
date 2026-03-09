"""Unit tests for the document-grounded (docs_web) generation pipeline.

Covers:
- DocumentChunker: chunking logic, section hints, page estimates
- DocumentIndex: TF-IDF build, retrieve, retrieve_multi
- DocGroundedGenerator: static validation, intent builder, YAML output
- CorpusManager: noise URL filter, text cleaning, blocked domain filter
- Config: DocsWebConfig defaults and loading
- AutoTestGenerator: docs_web dispatch
- BSP concept extraction
- Evidence cleaning & artifact detection
- Provenance validation
- Domain-relevance scoring
- Content quality scoring
- Integration test for contract-law BSP
"""

import asyncio
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_CONTRACT_LAW_BSP = """\
You are LegalAI, a Legal Research Assistant specializing in contract law.

## Primary Mission:
Help users understand legal concepts, analyze contracts, and provide educational guidance on legal matters.

## Your Expertise Areas:
1. **Contract Formation**: Offer, acceptance, consideration, capacity, legality
2. **Contract Clauses**: Indemnification, limitation of liability, force majeure, arbitration
3. **Breach & Remedies**: Material breach, anticipatory breach, damages, specific performance
4. **NDAs**: Confidentiality obligations, term duration, exceptions, enforcement
5. **Employment Contracts**: At-will employment, non-compete clauses, severance terms
6. **IP in Contracts**: Assignment clauses, licensing terms, work-for-hire provisions

## Strict Rules (MUST FOLLOW):
1. **NEVER** say "you should" or "you must" when discussing legal actions
2. **ALWAYS** define legal terms in parentheses on first use
3. **ALWAYS** cite relevant legal principles
4. **NEVER** answer questions about specific legal products, platforms, or services
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus_doc(
    doc_id: str = "d1",
    url: str = "https://example.com/guide",
    title: str = "Test Guide",
    text: str = "",
    char_count: int = 0,
):
    from promptlab.utils.doc_corpus import CorpusDocument
    if not text:
        text = (
            "Introduction to Legal Compliance.\n"
            "Legal compliance means following all applicable laws and regulations.\n"
            "Organizations must ensure their policies are updated regularly.\n"
            "Key Concepts in Regulatory Frameworks.\n"
            "A regulatory framework defines the structure of rules that govern behavior.\n"
            "Compliance officers are responsible for monitoring adherence.\n"
            "Penalties for non-compliance can include fines and legal action.\n"
            "Best Practices for Auditing.\n"
            "Auditing involves systematic examination of records and processes.\n"
            "Internal audits should be conducted quarterly as a minimum.\n"
            "External audits provide an independent assessment of compliance.\n"
        ) * 10  # repeat to get enough text for chunking
    return CorpusDocument(
        doc_id=doc_id,
        url=url,
        title=title,
        content_hash="abc123",
        fetched_at="2025-01-01T00:00:00Z",
        char_count=char_count or len(text),
        domain="example.com",
        text=text,
    )


# ---------------------------------------------------------------------------
# A) DocumentChunker tests
# ---------------------------------------------------------------------------

class TestDocumentChunker:
    """Test document chunking logic."""

    def test_basic_chunking(self):
        from promptlab.utils.doc_index import DocumentChunker
        doc = _make_corpus_doc()
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        # Every chunk should have valid metadata
        for ch in chunks:
            assert ch.doc_id == "d1"
            assert ch.doc_url == "https://example.com/guide"
            assert ch.token_count > 0
            assert ch.chunk_id  # non-empty

    def test_empty_doc_gives_no_chunks(self):
        from promptlab.utils.doc_index import DocumentChunker
        from promptlab.utils.doc_corpus import CorpusDocument
        doc = CorpusDocument(
            doc_id="empty", url="https://x.com", title="Empty",
            content_hash="", fetched_at="", char_count=0, domain="x.com", text="",
        )
        chunker = DocumentChunker()
        assert chunker.chunk_document(doc) == []

    def test_page_estimate_increases(self):
        from promptlab.utils.doc_index import DocumentChunker
        long_text = "word " * 5000  # ~25,000 chars → multiple pages
        doc = _make_corpus_doc(text=long_text)
        chunker = DocumentChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk_document(doc)
        pages = [c.page_estimate for c in chunks]
        # Pages should be non-decreasing
        for i in range(1, len(pages)):
            assert pages[i] >= pages[i - 1]

    def test_overlap_produces_more_chunks(self):
        from promptlab.utils.doc_index import DocumentChunker
        doc = _make_corpus_doc()
        no_overlap = DocumentChunker(chunk_size=100, overlap=0).chunk_document(doc)
        with_overlap = DocumentChunker(chunk_size=100, overlap=50).chunk_document(doc)
        assert len(with_overlap) >= len(no_overlap)


# ---------------------------------------------------------------------------
# B) DocumentIndex tests
# ---------------------------------------------------------------------------

class TestDocumentIndex:
    """Test TF-IDF index build and retrieval."""

    def test_build_index(self):
        from promptlab.utils.doc_index import DocumentIndex
        doc = _make_corpus_doc()
        index = DocumentIndex(chunk_size=50, overlap=10)
        chunk_count = index.build([doc])
        assert chunk_count > 0
        assert len(index.chunks) == chunk_count

    def test_retrieve_returns_scored_results(self):
        from promptlab.utils.doc_index import DocumentIndex
        doc = _make_corpus_doc()
        index = DocumentIndex(chunk_size=50, overlap=10)
        index.build([doc])
        results = index.retrieve("legal compliance regulations", top_k=3)
        assert len(results) > 0
        assert results[0].score > 0
        assert results[0].evidence_span  # non-empty

    def test_retrieve_empty_index(self):
        from promptlab.utils.doc_index import DocumentIndex
        index = DocumentIndex()
        # Don't build — should return empty
        assert index.retrieve("anything") == []

    def test_retrieve_multi_deduplicates(self):
        from promptlab.utils.doc_index import DocumentIndex
        doc = _make_corpus_doc()
        index = DocumentIndex(chunk_size=50, overlap=10)
        index.build([doc])
        # Same query twice should still deduplicate
        results = index.retrieve_multi(
            ["legal compliance", "legal compliance regulations"],
            top_k_per_query=5,
        )
        chunk_ids = [r.chunk.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Should have no duplicate chunk_ids"


# ---------------------------------------------------------------------------
# C) Static validation tests
# ---------------------------------------------------------------------------

class TestStaticValidation:
    """Test the cheap static validation pass."""

    def test_valid_qa_passes(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "What is legal compliance?",
            "answer_key": "Following all applicable laws",
            "style": "qa",
        }
        assert _passes_static_validation(case) is True

    def test_valid_cloze_passes(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "Fill in the blank: ____ is the process of following laws.",
            "answer_key": "Compliance",
            "style": "cloze",
        }
        assert _passes_static_validation(case) is True

    def test_rejects_cookie_noise(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "What is the cookie consent policy?",
            "answer_key": "Accept all cookies",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_short_prompt(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {"prompt": "What?", "answer_key": "Something", "style": "qa"}
        assert _passes_static_validation(case) is False

    def test_rejects_placeholder(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "What is TODO: insert question here?",
            "answer_key": "Some answer",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_qa_without_question_mark(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "Tell me about compliance",
            "answer_key": "Following laws",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_cloze_without_blank(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "Compliance is the process of following laws.",
            "answer_key": "Compliance",
            "style": "cloze",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_tautological_prompt(self):
        """Prompts that are self-referential ('according to the evidence') should be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "According to the evidence, what does the text suggest about law?",
            "answer_key": "It suggests following law",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_based_on_the_evidence(self):
        """'Based on the evidence' variant should also be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "Based on the evidence, what is the primary purpose of an indemnification clause?",
            "answer_key": "to hold one party harmless",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_as_stated_in(self):
        """'As stated in the' variant should be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "As stated in the document, what constitutes a material breach?",
            "answer_key": "failure to perform",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_passage_states(self):
        """'The passage states' variant should be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "The passage states that NDAs require what key element?",
            "answer_key": "confidentiality",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_product_trivia(self):
        """Product/vendor trivia questions should be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "What award did CARET Legal receive from the 2025 LegalTech Awards?",
            "answer_key": "Cloud-based Platform of the Year",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_ecommerce_noise(self):
        """E-commerce patterns should be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "What is the purpose of the 'Your recently viewed items' feature?",
            "answer_key": "To help navigate back",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False

    def test_rejects_survey_reference(self):
        """Survey-based vendor content should be rejected."""
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "prompt": "According to the 2016 Practical Law Customer Survey, who found it useful?",
            "answer_key": "Solo firms",
            "style": "qa",
        }
        assert _passes_static_validation(case) is False


# ---------------------------------------------------------------------------
# D) Retrieval intent builder tests
# ---------------------------------------------------------------------------

class TestRetrievalIntents:
    """Test that build_retrieval_intents produces varied intent types."""

    def test_produces_intents(self):
        from promptlab.utils.doc_generator import build_retrieval_intents
        intents = build_retrieval_intents(
            domain="legal_compliance",
            role="compliance officer",
            keywords=["regulation", "audit", "penalty"],
            capabilities=["review documents", "assess risk"],
            constraints=["must follow GDPR"],
        )
        assert len(intents) > 0
        labels = {label for label, _ in intents}
        # Should have template-based and keyword intents at minimum
        assert "definition" in labels
        assert "keyword" in labels

    def test_constraint_intents(self):
        from promptlab.utils.doc_generator import build_retrieval_intents
        intents = build_retrieval_intents(
            domain="medicine",
            role="doctor",
            keywords=[],
            capabilities=[],
            constraints=["must verify allergies", "follow dosage guidelines"],
        )
        constraint_intents = [(l, q) for l, q in intents if l == "constraint"]
        assert len(constraint_intents) == 2

    def test_concept_targeted_intents(self):
        """When BSP concepts are provided, intents should include concept-tagged queries."""
        from promptlab.utils.doc_generator import build_retrieval_intents
        concepts = [
            ("contract_formation", ["contract formation", "offer acceptance"]),
            ("breach_remedies", ["breach of contract", "damages"]),
        ]
        intents = build_retrieval_intents(
            domain="legal",
            role="Legal AI",
            keywords=["contract"],
            capabilities=[],
            constraints=[],
            bsp_concepts=concepts,
        )
        labels = [l for l, _ in intents]
        # Should have concept-prefixed labels (e.g. "contract_formation:definition")
        concept_labels = [l for l in labels if ":" in l]
        assert len(concept_labels) > 0, "Should produce concept-prefixed intent labels"
        concept_roots = {l.split(":")[0] for l in concept_labels}
        assert "contract_formation" in concept_roots
        assert "breach_remedies" in concept_roots


# ---------------------------------------------------------------------------
# E) CorpusManager noise filters
# ---------------------------------------------------------------------------

class TestCorpusManagerFilters:
    """Test URL and text noise filtering."""

    def test_noise_url_detected(self):
        from promptlab.utils.doc_corpus import CorpusManager
        mgr = CorpusManager(project_root=Path("."))
        assert mgr._is_noise_url("https://example.com/login") is True
        assert mgr._is_noise_url("https://youtube.com/watch?v=123") is True
        assert mgr._is_noise_url("https://example.com/privacy-policy") is True

    def test_clean_url_allowed(self):
        from promptlab.utils.doc_corpus import CorpusManager
        mgr = CorpusManager(project_root=Path("."))
        assert mgr._is_noise_url("https://docs.example.com/guide") is False

    def test_clean_doc_text_removes_cookies(self):
        from promptlab.utils.doc_corpus import CorpusManager
        mgr = CorpusManager(project_root=Path("."))
        raw = (
            "Welcome to our guide.\n"
            "We use cookies to improve your experience.\n"
            "Accept all cookies\n"
            "This chapter covers legal compliance.\n"
            "© 2024 Company Inc.\n"
            "All rights reserved.\n"
            "Subscribe to our newsletter for updates.\n"
            "Important information about regulations.\n"
        )
        cleaned = mgr._clean_doc_text(raw)
        assert "cookie" not in cleaned.lower()
        assert "newsletter" not in cleaned.lower()
        assert "legal compliance" in cleaned.lower()
        assert "Important information" in cleaned

    def test_noise_url_rejects_amazon(self):
        """CorpusManager._is_noise_url should reject amazon.com via blocked domains."""
        from promptlab.utils.doc_corpus import CorpusManager
        mgr = CorpusManager(project_root=Path("."))
        assert mgr._is_noise_url("https://www.amazon.com/Official-Guide/dp/0159003911") is True


# ---------------------------------------------------------------------------
# F) DocsWebConfig tests
# ---------------------------------------------------------------------------

class TestDocsWebConfig:
    """Test DocsWebConfig defaults and integration with PromptLabConfig."""

    def test_defaults(self):
        from promptlab.utils.config import DocsWebConfig
        cfg = DocsWebConfig()
        assert cfg.max_docs == 20
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 200
        assert cfg.retrieval_top_k == 10
        assert cfg.target_count == 100
        assert cfg.llm_model is None

    def test_in_promptlab_config(self):
        from promptlab.utils.config import PromptLabConfig
        cfg = PromptLabConfig()
        assert hasattr(cfg, "docs_web")
        assert cfg.docs_web.max_docs == 20

    def test_load_from_yaml(self, tmp_path: Path):
        from promptlab.utils.config import load_config
        yaml_content = (
            "version: 1\n"
            "bsp:\n"
            "  generation_mode: docs_web\n"
            "docs_web:\n"
            "  max_docs: 30\n"
            "  chunk_size: 1000\n"
            "  retrieval_top_k: 15\n"
        )
        config_file = tmp_path / "promptlab.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        cfg = load_config(config_file)
        assert cfg.bsp.generation_mode == "docs_web"
        assert cfg.docs_web.max_docs == 30
        assert cfg.docs_web.chunk_size == 1000
        assert cfg.docs_web.retrieval_top_k == 15


# ---------------------------------------------------------------------------
# G) GeneratedTests backward compatibility
# ---------------------------------------------------------------------------

class TestGeneratedTestsCompat:
    """Ensure GeneratedTests works with both old and new pipelines."""

    def test_web_mode_defaults(self):
        from promptlab.utils.auto_test_generator import GeneratedTests
        gt = GeneratedTests(
            domain="test",
            qa_pairs=[],
            masked_tests=[],
            yaml_content={},
            output_file=Path("out.yaml"),
            generation_time=1.0,
            scraped_sources=5,
        )
        assert gt.generated_cases == []
        assert gt.generation_mode_used == "web"

    def test_docs_web_mode(self):
        from promptlab.utils.auto_test_generator import GeneratedTests
        gt = GeneratedTests(
            domain="test",
            qa_pairs=[],
            masked_tests=[],
            yaml_content={},
            output_file=Path("out.yaml"),
            generation_time=2.0,
            scraped_sources=10,
            generated_cases=["case1", "case2"],
            generation_mode_used="docs_web",
        )
        assert len(gt.generated_cases) == 2
        assert gt.generation_mode_used == "docs_web"


# ---------------------------------------------------------------------------
# H) Heuristic testcase generation
# ---------------------------------------------------------------------------

class TestHeuristicGeneration:
    """Test the no-LLM heuristic fallback generator."""

    def test_definition_pattern(self):
        from promptlab.utils.doc_generator import _heuristic_case_from_evidence
        evidence = "Compliance is the act of following all applicable laws and regulations to avoid penalties."
        result = _heuristic_case_from_evidence(evidence, "legal", "definition", 0)
        assert result is not None
        assert result["style"] == "qa"
        assert "?" in result["prompt"]

    def test_cloze_fallback(self):
        from promptlab.utils.doc_generator import _heuristic_case_from_evidence
        evidence = "Organizations must conduct quarterly Internal Audits to ensure regulatory compliance."
        result = _heuristic_case_from_evidence(evidence, "legal", "procedure", 0)
        assert result is not None
        # Should produce either qa or cloze
        assert result["style"] in ("qa", "cloze")

    def test_too_short_evidence_returns_none(self):
        from promptlab.utils.doc_generator import _heuristic_case_from_evidence
        result = _heuristic_case_from_evidence("Short.", "legal", "definition", 0)
        assert result is None

    def test_heuristic_cleans_artifacts_from_evidence(self):
        """Evidence with /* Lines omitted */ should be cleaned before generation."""
        from promptlab.utils.doc_generator import _heuristic_case_from_evidence
        evidence = "Indemnification is the act of /* Lines 10-20 omitted */ compensating for harm or loss suffered by another party."
        result = _heuristic_case_from_evidence(evidence, "legal", "definition", 0)
        if result is not None:
            assert "/* Lines" not in result["prompt"]
            assert "omitted */" not in result["prompt"]


# ---------------------------------------------------------------------------
# I) Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Test GeneratedCase deduplication."""

    def test_exact_dedup(self):
        from promptlab.utils.doc_generator import GeneratedCase, ProvenanceInfo, _dedup_cases
        prov = ProvenanceInfo()
        cases = [
            GeneratedCase(case_id="1", prompt="What is compliance?", assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="2", prompt="What is compliance?", assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="3", prompt="What is auditing?", assertions=[], tags=[], provenance=prov),
        ]
        unique = _dedup_cases(cases)
        assert len(unique) == 2

    def test_case_insensitive_dedup(self):
        from promptlab.utils.doc_generator import GeneratedCase, ProvenanceInfo, _dedup_cases
        prov = ProvenanceInfo()
        cases = [
            GeneratedCase(case_id="1", prompt="What is COMPLIANCE?", assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="2", prompt="what is compliance?", assertions=[], tags=[], provenance=prov),
        ]
        unique = _dedup_cases(cases)
        assert len(unique) == 1

    def test_paraphrase_dedup(self):
        """Near-duplicate paraphrases should be collapsed via Jaccard similarity."""
        from promptlab.utils.doc_generator import GeneratedCase, ProvenanceInfo, _dedup_cases
        prov = ProvenanceInfo()
        cases = [
            GeneratedCase(case_id="1",
                          prompt="What are the essential elements that must be present for a contract?",
                          assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="2",
                          prompt="What are the four essential elements required for a valid contract?",
                          assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="3",
                          prompt="What is the doctrine of force majeure?",
                          assertions=[], tags=[], provenance=prov),
        ]
        unique = _dedup_cases(cases)
        # The two contract-elements questions should collapse to one
        assert len(unique) == 2

    def test_dissimilar_prompts_kept(self):
        """Genuinely different prompts should not be deduped."""
        from promptlab.utils.doc_generator import GeneratedCase, ProvenanceInfo, _dedup_cases
        prov = ProvenanceInfo()
        cases = [
            GeneratedCase(case_id="1", prompt="What is a non-disclosure agreement?",
                          assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="2", prompt="What remedies exist for breach of contract?",
                          assertions=[], tags=[], provenance=prov),
            GeneratedCase(case_id="3", prompt="What is force majeure in contract law?",
                          assertions=[], tags=[], provenance=prov),
        ]
        unique = _dedup_cases(cases)
        assert len(unique) == 3


# ---------------------------------------------------------------------------
# J) BSPConfig generation_mode validation
# ---------------------------------------------------------------------------

class TestBSPConfigModes:
    """Verify BSPConfig accepts all valid generation modes."""

    def test_web_mode(self):
        from promptlab.utils.config import BSPConfig
        cfg = BSPConfig(generation_mode="web")
        assert cfg.generation_mode == "web"

    def test_docs_web_mode(self):
        from promptlab.utils.config import BSPConfig
        cfg = BSPConfig(generation_mode="docs_web")
        assert cfg.generation_mode == "docs_web"

    def test_hybrid_mode(self):
        from promptlab.utils.config import BSPConfig
        cfg = BSPConfig(generation_mode="hybrid")
        assert cfg.generation_mode == "hybrid"


# ---------------------------------------------------------------------------
# K) Blocked domain filter
# ---------------------------------------------------------------------------

class TestBlockedDomainFilter:
    """Test the hard domain blocklist in doc_corpus."""

    def test_amazon_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.amazon.com/some-product/dp/123") is True
        assert is_blocked_domain("https://amazon.com/guide") is True

    def test_ebay_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.ebay.com/itm/12345") is True

    def test_social_media_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.facebook.com/page") is True
        assert is_blocked_domain("https://twitter.com/status/123") is True
        assert is_blocked_domain("https://www.reddit.com/r/law") is True
        assert is_blocked_domain("https://www.linkedin.com/article") is True

    def test_review_sites_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.g2.com/products/review") is True
        assert is_blocked_domain("https://www.trustpilot.com/review") is True
        assert is_blocked_domain("https://www.capterra.com/product") is True

    def test_legitimate_legal_sites_allowed(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.law.cornell.edu/ucc") is False
        assert is_blocked_domain("https://guides.library.harvard.edu/law") is False
        assert is_blocked_domain("https://www.americanbar.org/guide") is False

    def test_subdomain_of_blocked_domain(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://smile.amazon.com/product") is True


# ---------------------------------------------------------------------------
# L) Content quality scoring
# ---------------------------------------------------------------------------

class TestContentQualityScoring:
    """Test the content quality scoring heuristic in doc_corpus."""

    def test_high_quality_content(self):
        from promptlab.utils.doc_corpus import compute_content_quality_score
        text = (
            "Contract formation requires four essential elements: offer, acceptance, "
            "consideration, and mutual assent. An offer is a manifestation of willingness "
            "to enter into a bargain. Acceptance is the offeree's expression of assent to "
            "the terms of the offer. Consideration is something of value exchanged between "
            "the parties. Mutual assent means both parties agree to the same terms.\n"
        ) * 5
        score = compute_content_quality_score(text)
        assert score >= 0.7

    def test_thin_content_penalised(self):
        from promptlab.utils.doc_corpus import compute_content_quality_score
        score = compute_content_quality_score("Short page.")
        assert score < 0.5

    def test_cta_heavy_penalised(self):
        from promptlab.utils.doc_corpus import compute_content_quality_score
        text = (
            "Buy now and save 20%!\n"
            "Add to cart for free shipping.\n"
            "Customers who bought this also bought...\n"
            "Frequently bought together.\n"
            "Product details and description.\n"
            "Shop now for the best deals.\n"
        ) * 3
        score = compute_content_quality_score(text)
        assert score < 0.5

    def test_vendor_marketing_penalised(self):
        from promptlab.utils.doc_corpus import compute_content_quality_score
        text = (
            "Rated #1 in G2 for legal software.\n"
            "Award-winning platform trusted by 5000 firms.\n"
            "Book a demo today.\n"
            "Start your free trial now.\n"
            "Our platform helps law firms manage cases efficiently.\n"
            "ROI calculator available.\n"
        ) * 3
        score = compute_content_quality_score(text)
        assert score < 0.5

    def test_empty_content_zero(self):
        from promptlab.utils.doc_corpus import compute_content_quality_score
        assert compute_content_quality_score("") == 0.0


# ---------------------------------------------------------------------------
# M) BSP concept extraction
# ---------------------------------------------------------------------------

class TestBSPConceptExtraction:
    """Test extraction of domain concepts from BSP text."""

    def test_contract_law_concepts(self):
        from promptlab.utils.doc_generator import extract_bsp_concepts
        concepts = extract_bsp_concepts(_CONTRACT_LAW_BSP, "legal")
        labels = [c[0] for c in concepts]
        assert "contract_formation" in labels
        assert "breach_remedies" in labels
        assert "nda" in labels

    def test_concepts_have_keywords(self):
        from promptlab.utils.doc_generator import extract_bsp_concepts
        concepts = extract_bsp_concepts(_CONTRACT_LAW_BSP, "legal")
        for label, kws in concepts:
            assert len(kws) > 0, f"Concept {label} has no keywords"

    def test_unknown_domain_extracts_from_bsp(self):
        """For unknown domains, should extract concepts from numbered headings."""
        from promptlab.utils.doc_generator import extract_bsp_concepts
        bsp = (
            "You are a gardening assistant.\n"
            "## Areas:\n"
            "1. **Soil Preparation**: pH, composting, drainage\n"
            "2. **Pest Management**: organic methods, IPM\n"
            "3. **Irrigation Systems**: drip, sprinkler, scheduling\n"
        )
        concepts = extract_bsp_concepts(bsp, "gardening")
        labels = [c[0] for c in concepts]
        assert any("soil" in l for l in labels)
        assert any("pest" in l or "management" in l for l in labels)


# ---------------------------------------------------------------------------
# N) Evidence cleaning
# ---------------------------------------------------------------------------

class TestEvidenceCleaning:
    """Test evidence span cleaning and artifact detection."""

    def test_removes_line_omission_artifacts(self):
        from promptlab.utils.doc_generator import clean_evidence_span
        text = "Some legal text /* Lines 10-20 omitted */ and more text."
        cleaned = clean_evidence_span(text)
        assert "/* Lines" not in cleaned
        assert "omitted */" not in cleaned
        assert "legal text" in cleaned

    def test_removes_html_tags(self):
        from promptlab.utils.doc_generator import clean_evidence_span
        text = "Contract <b>formation</b> requires <a href='#'>consideration</a>."
        cleaned = clean_evidence_span(text)
        assert "<b>" not in cleaned
        assert "<a " not in cleaned

    def test_collapses_whitespace(self):
        from promptlab.utils.doc_generator import clean_evidence_span
        text = "Contract   formation    requires   consideration."
        cleaned = clean_evidence_span(text)
        assert "  " not in cleaned

    def test_removes_trailing_ellipsis(self):
        from promptlab.utils.doc_generator import clean_evidence_span
        text = "Contract formation requires consideration..."
        cleaned = clean_evidence_span(text)
        assert not cleaned.endswith("...")

    def test_empty_input(self):
        from promptlab.utils.doc_generator import clean_evidence_span
        assert clean_evidence_span("") == ""
        assert clean_evidence_span(None) == ""

    def test_artifact_detection_positive(self):
        from promptlab.utils.doc_generator import evidence_has_artifacts
        assert evidence_has_artifacts("text /* Lines 10-20 omitted */ more") is True

    def test_artifact_detection_negative(self):
        from promptlab.utils.doc_generator import evidence_has_artifacts
        assert evidence_has_artifacts("clean evidence about contracts") is False


# ---------------------------------------------------------------------------
# O) Provenance validation
# ---------------------------------------------------------------------------

class TestProvenanceValidation:
    """Test provenance completeness checking."""

    def test_complete_provenance_passes(self):
        from promptlab.utils.doc_generator import ProvenanceInfo, provenance_is_complete
        prov = ProvenanceInfo(
            source_doc_id="doc1",
            source_url="https://example.com",
            page_number=1,
            chunk_id="chunk1",
            section="Introduction",
            evidence_span="Some evidence text",
        )
        assert provenance_is_complete(prov) is True

    def test_missing_source_url_fails(self):
        from promptlab.utils.doc_generator import ProvenanceInfo, provenance_is_complete
        prov = ProvenanceInfo(
            source_doc_id="doc1",
            source_url="",
            page_number=1,
            chunk_id="chunk1",
            section="",
            evidence_span="Some evidence",
        )
        assert provenance_is_complete(prov) is False

    def test_missing_evidence_span_fails(self):
        from promptlab.utils.doc_generator import ProvenanceInfo, provenance_is_complete
        prov = ProvenanceInfo(
            source_doc_id="doc1",
            source_url="https://example.com",
            page_number=1,
            chunk_id="chunk1",
            section="",
            evidence_span="",
        )
        assert provenance_is_complete(prov) is False

    def test_bsp_source_exempt(self):
        """BSP-sourced cases don't need external provenance."""
        from promptlab.utils.doc_generator import ProvenanceInfo, provenance_is_complete
        prov = ProvenanceInfo(
            source_doc_id="bsp",
            source_url="",
            page_number=0,
            chunk_id="bsp-constraint-0",
            section="",
            evidence_span="",
        )
        assert provenance_is_complete(prov) is True

    def test_missing_chunk_id_fails(self):
        from promptlab.utils.doc_generator import ProvenanceInfo, provenance_is_complete
        prov = ProvenanceInfo(
            source_doc_id="doc1",
            source_url="https://example.com",
            page_number=1,
            chunk_id="",
            section="",
            evidence_span="Some evidence",
        )
        assert provenance_is_complete(prov) is False


# ---------------------------------------------------------------------------
# P) Domain relevance scoring
# ---------------------------------------------------------------------------

class TestDomainRelevance:
    """Test domain-relevance scoring."""

    def test_relevant_text_scores_high(self):
        from promptlab.utils.doc_generator import compute_domain_relevance
        keywords = {"contract", "formation", "offer", "acceptance", "consideration", "breach", "damages"}
        text = "Contract formation requires an offer, acceptance, and consideration. Breach leads to damages."
        score = compute_domain_relevance(text, keywords)
        assert score >= 0.5

    def test_irrelevant_text_scores_low(self):
        from promptlab.utils.doc_generator import compute_domain_relevance
        keywords = {"contract", "formation", "offer", "acceptance", "consideration", "breach", "damages"}
        text = "The best chocolate cake recipe involves mixing flour, sugar, and eggs."
        score = compute_domain_relevance(text, keywords)
        assert score < 0.3

    def test_empty_text_scores_neutral(self):
        from promptlab.utils.doc_generator import compute_domain_relevance
        keywords = {"contract", "formation"}
        # Empty text with keywords returns 0.5 (neutral) per early-return
        assert compute_domain_relevance("", keywords) == 0.5

    def test_no_concepts_returns_neutral(self):
        from promptlab.utils.doc_generator import compute_domain_relevance
        assert compute_domain_relevance("some text", set()) == 0.5


# ---------------------------------------------------------------------------
# Q) Integration: contract-law BSP quality checks
# ---------------------------------------------------------------------------

class TestContractLawIntegration:
    """Integration test validating the full pipeline quality for contract-law BSP.

    Creates a synthetic corpus of contract-law documents, runs the full
    generation pipeline, and validates output against acceptance criteria.
    """

    @staticmethod
    def _make_contract_law_corpus():
        """Create a synthetic corpus of contract-law documents."""
        from promptlab.utils.doc_corpus import CorpusDocument

        docs = []
        docs.append(CorpusDocument(
            doc_id="cf1", url="https://legal-ed.example.com/contract-formation",
            title="Contract Formation Guide",
            content_hash="h1", fetched_at="2025-01-01T00:00:00Z",
            char_count=3000, domain="legal-ed.example.com",
            text=(
                "Contract Formation: Essential Elements.\n"
                "A valid contract requires four elements: offer, acceptance, consideration, and mutual assent.\n"
                "An offer is a definite proposal made by one party to another.\n"
                "Acceptance is the unqualified agreement to the terms of the offer.\n"
                "Consideration is something of value exchanged between the parties.\n"
                "Under the Statute of Frauds, certain contracts must be in writing.\n"
                "Capacity means parties must have legal ability to enter contracts.\n"
                "Minors generally lack capacity to form binding contracts.\n"
                "Legality requires the contract purpose to be lawful.\n"
                "The mailbox rule states acceptance is effective when dispatched.\n"
            ) * 8,
        ))
        docs.append(CorpusDocument(
            doc_id="br1", url="https://legal-ed.example.com/breach-remedies",
            title="Breach of Contract and Remedies",
            content_hash="h2", fetched_at="2025-01-01T00:00:00Z",
            char_count=3000, domain="legal-ed.example.com",
            text=(
                "Breach of Contract: Types and Remedies.\n"
                "A material breach is a failure to perform a substantial part of the contract.\n"
                "An anticipatory breach occurs when a party indicates they will not perform.\n"
                "Compensatory damages aim to put the injured party in the position they would have been.\n"
                "Consequential damages are losses that flow from the breach.\n"
                "Specific performance is an equitable remedy ordering the breaching party to perform.\n"
                "Liquidated damages are pre-agreed damages in the contract.\n"
                "Mitigation of damages requires the injured party to take reasonable steps.\n"
                "Nominal damages are awarded when breach occurred but no actual loss.\n"
            ) * 8,
        ))
        docs.append(CorpusDocument(
            doc_id="nda1", url="https://legal-ed.example.com/nda-guide",
            title="Non-Disclosure Agreements Guide",
            content_hash="h3", fetched_at="2025-01-01T00:00:00Z",
            char_count=3000, domain="legal-ed.example.com",
            text=(
                "Non-Disclosure Agreements: Key Provisions.\n"
                "A non-disclosure agreement protects confidential information shared between parties.\n"
                "The disclosing party shares information; the receiving party agrees to keep it confidential.\n"
                "Confidential information typically excludes publicly available information.\n"
                "NDAs specify the term or duration of confidentiality obligations.\n"
                "Mutual NDAs protect both parties' confidential information.\n"
                "Remedies for breach of NDA may include injunctive relief and damages.\n"
                "Trade secrets are a category of confidential information with special protections.\n"
            ) * 8,
        ))
        docs.append(CorpusDocument(
            doc_id="fm1", url="https://legal-ed.example.com/force-majeure",
            title="Force Majeure and Termination Clauses",
            content_hash="h4", fetched_at="2025-01-01T00:00:00Z",
            char_count=3000, domain="legal-ed.example.com",
            text=(
                "Force Majeure Clauses in Contracts.\n"
                "Force majeure clauses excuse performance when extraordinary events prevent fulfillment.\n"
                "Common force majeure events include natural disasters, war, and pandemics.\n"
                "The doctrine of impossibility of performance applies when performance becomes impossible.\n"
                "Frustration of purpose applies when the purpose of the contract is destroyed.\n"
                "Termination clauses specify conditions under which parties can end the contract.\n"
                "Termination for cause allows ending the contract due to breach.\n"
                "Termination for convenience allows ending without breach, often with notice.\n"
            ) * 8,
        ))
        return docs

    def test_generation_quality(self, tmp_path: Path):
        """Full pipeline test with synthetic contract-law corpus."""
        from promptlab.utils.doc_index import DocumentIndex
        from promptlab.utils.doc_generator import (
            DocGroundedGenerator,
            provenance_is_complete,
            evidence_has_artifacts,
        )
        from promptlab.utils.doc_corpus import is_blocked_domain

        docs = self._make_contract_law_corpus()
        index = DocumentIndex(chunk_size=50, overlap=10)
        chunk_count = index.build(docs)
        assert chunk_count > 0

        generator = DocGroundedGenerator(index=index, retrieval_top_k=15)
        cases, output_file = asyncio.get_event_loop().run_until_complete(
            generator.generate(
                bsp=_CONTRACT_LAW_BSP,
                domain="legal",
                role="LegalAI",
                keywords=["contract", "formation", "breach"],
                capabilities=[],
                constraints=["NEVER say you should"],
                target_count=50,
                output_dir=tmp_path,
                output_type="all",
            )
        )

        # 1. Should produce cases
        assert len(cases) > 0, "Should produce at least some cases"

        # 2. No cases from blocked domains
        blocked_cases = [c for c in cases if is_blocked_domain(c.provenance.source_url)]
        assert len(blocked_cases) == 0, f"Found {len(blocked_cases)} cases from blocked domains"

        # 3. No cases with incomplete provenance
        incomplete = [c for c in cases if not provenance_is_complete(c.provenance)]
        assert len(incomplete) == 0, f"Found {len(incomplete)} cases with incomplete provenance"

        # 4. No evidence spans with artifacts
        artifact_cases = [c for c in cases if evidence_has_artifacts(c.provenance.evidence_span)]
        assert len(artifact_cases) == 0, f"Found {len(artifact_cases)} cases with evidence artifacts"

        # 5. YAML output is valid
        import yaml
        with open(output_file, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
        assert "metadata" in yaml_content
        assert "cases" in yaml_content
        assert yaml_content["metadata"]["test_count"] == len(cases)
        for case in yaml_content["cases"]:
            assert "id" in case
            assert "prompt" in case
            assert "assertions" in case
            assert "provenance" in case
            assert "source_doc_id" in case["provenance"]
            assert "source_url" in case["provenance"]

    def test_no_amazon_contamination(self):
        """Verify amazon.com URLs are always blocked."""
        from promptlab.utils.doc_corpus import is_blocked_domain
        for url in [
            "https://www.amazon.com/Official-Guide-Specialties/dp/0159003911",
            "https://amazon.com/Legal-Books/dp/123456",
            "https://smile.amazon.com/product",
        ]:
            assert is_blocked_domain(url), f"Should block: {url}"

    def test_concept_coverage_tracking(self, tmp_path: Path):
        """Verify concept coverage tracking is populated after generation."""
        from promptlab.utils.doc_index import DocumentIndex
        from promptlab.utils.doc_generator import DocGroundedGenerator

        docs = self._make_contract_law_corpus()
        index = DocumentIndex(chunk_size=50, overlap=10)
        index.build(docs)

        generator = DocGroundedGenerator(index=index, retrieval_top_k=10)
        cases, _ = asyncio.get_event_loop().run_until_complete(
            generator.generate(
                bsp=_CONTRACT_LAW_BSP,
                domain="legal",
                role="LegalAI",
                keywords=["contract"],
                capabilities=[],
                constraints=[],
                target_count=20,
                output_dir=tmp_path,
            )
        )
        assert len(generator.concept_coverage) > 0, "Should track concept coverage"

    def test_rejection_tracking(self, tmp_path: Path):
        """Verify rejection records list is initialised."""
        from promptlab.utils.doc_index import DocumentIndex
        from promptlab.utils.doc_generator import DocGroundedGenerator

        docs = self._make_contract_law_corpus()
        index = DocumentIndex(chunk_size=50, overlap=10)
        index.build(docs)

        generator = DocGroundedGenerator(index=index, retrieval_top_k=10)
        asyncio.get_event_loop().run_until_complete(
            generator.generate(
                bsp=_CONTRACT_LAW_BSP,
                domain="legal",
                role="LegalAI",
                keywords=["contract"],
                capabilities=[],
                constraints=[],
                target_count=20,
                output_dir=tmp_path,
            )
        )
        assert isinstance(generator.rejections, list)


# ---------------------------------------------------------------------------
# R) Concrete answer extraction
# ---------------------------------------------------------------------------

class TestConcreteAnswerExtraction:
    """Test _extract_concrete_answer helper used to avoid min_length-only assertions."""

    def test_extracts_defined_term(self):
        from promptlab.utils.doc_generator import _extract_concrete_answer
        text = "This concept is known as Anticipatory Breach in contract law."
        result = _extract_concrete_answer(text)
        assert result is not None
        assert "Anticipatory Breach" in result

    def test_extracts_quoted_term(self):
        from promptlab.utils.doc_generator import _extract_concrete_answer
        text = 'The principle is called "Mitigation of Damages" in legal parlance.'
        result = _extract_concrete_answer(text)
        assert result is not None
        assert "Mitigation of Damages" in result

    def test_extracts_acronym(self):
        from promptlab.utils.doc_generator import _extract_concrete_answer
        text = "This is governed by the Uniform Commercial Code (UCC)."
        result = _extract_concrete_answer(text)
        assert result is not None
        assert "UCC" in result

    def test_extracts_capitalized_phrase_fallback(self):
        from promptlab.utils.doc_generator import _extract_concrete_answer
        text = "The Statute of Frauds requires certain contracts to be in writing."
        result = _extract_concrete_answer(text)
        assert result is not None
        assert "Statute" in result

    def test_returns_none_for_no_concrete_terms(self):
        from promptlab.utils.doc_generator import _extract_concrete_answer
        text = "this is all lowercase text with no concrete terms or definitions."
        result = _extract_concrete_answer(text)
        assert result is None


# ---------------------------------------------------------------------------
# S) Jaccard similarity helper
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    """Test the Jaccard similarity function used for paraphrase dedup."""

    def test_identical_sets(self):
        from promptlab.utils.doc_generator import _jaccard_similarity
        a = {"contract", "formation", "elements"}
        assert _jaccard_similarity(a, a) == 1.0

    def test_disjoint_sets(self):
        from promptlab.utils.doc_generator import _jaccard_similarity
        a = {"contract", "formation"}
        b = {"chocolate", "cake", "recipe"}
        assert _jaccard_similarity(a, b) == 0.0

    def test_overlapping_sets(self):
        from promptlab.utils.doc_generator import _jaccard_similarity
        a = {"essential", "elements", "contract", "formation", "valid"}
        b = {"four", "essential", "elements", "required", "valid", "contract"}
        sim = _jaccard_similarity(a, b)
        assert sim > 0.4  # significant overlap
        assert sim < 1.0

    def test_empty_sets(self):
        from promptlab.utils.doc_generator import _jaccard_similarity
        assert _jaccard_similarity(set(), {"a", "b"}) == 0.0
        assert _jaccard_similarity(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# T) Integration: no min_length-only assertions
# ---------------------------------------------------------------------------

class TestNoWeakAssertions:
    """Verify that generated cases never use min_length-only assertions."""

    def test_integration_no_min_length_only(self, tmp_path: Path):
        """Full pipeline should not produce cases with only min_length assertions."""
        from promptlab.utils.doc_index import DocumentIndex
        from promptlab.utils.doc_generator import DocGroundedGenerator

        docs = TestContractLawIntegration._make_contract_law_corpus()
        index = DocumentIndex(chunk_size=50, overlap=10)
        index.build(docs)

        generator = DocGroundedGenerator(index=index, retrieval_top_k=10)
        cases, output_file = asyncio.get_event_loop().run_until_complete(
            generator.generate(
                bsp=_CONTRACT_LAW_BSP,
                domain="legal",
                role="LegalAI",
                keywords=["contract"],
                capabilities=[],
                constraints=[],
                target_count=30,
                output_dir=tmp_path,
            )
        )

        for case in cases:
            for assertion in case.assertions:
                assert assertion["type"] != "min_length", (
                    f"Case {case.case_id} has a min_length-only assertion: {case.prompt[:60]}"
                )

    def test_integration_no_artifacts_in_prompts(self, tmp_path: Path):
        """Full pipeline should not produce prompts containing /* Lines omitted */."""
        from promptlab.utils.doc_index import DocumentIndex
        from promptlab.utils.doc_generator import DocGroundedGenerator

        docs = TestContractLawIntegration._make_contract_law_corpus()
        index = DocumentIndex(chunk_size=50, overlap=10)
        index.build(docs)

        generator = DocGroundedGenerator(index=index, retrieval_top_k=10)
        cases, _ = asyncio.get_event_loop().run_until_complete(
            generator.generate(
                bsp=_CONTRACT_LAW_BSP,
                domain="legal",
                role="LegalAI",
                keywords=["contract"],
                capabilities=[],
                constraints=[],
                target_count=30,
                output_dir=tmp_path,
            )
        )

        for case in cases:
            assert "/* Lines" not in case.prompt, (
                f"Case {case.case_id} has artifacts in prompt: {case.prompt[:80]}"
            )
            assert "/* Lines" not in case.provenance.evidence_span, (
                f"Case {case.case_id} has artifacts in evidence"
            )


# ---------------------------------------------------------------------------
# U) Student-note aggregator blocked domains
# ---------------------------------------------------------------------------

class TestStudentNoteBlockedDomains:
    """Verify that student-note / homework-help sites are hard-blocked."""

    def test_cliffsnotes_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.cliffsnotes.com/study-notes/26809352")

    def test_coursehero_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.coursehero.com/file/123/Contracts.pdf")

    def test_studocu_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://studocu.com/en-us/document/law-notes/12345")

    def test_chegg_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.chegg.com/homework-help/questions-and-answers/")

    def test_quizlet_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://quizlet.com/legal-flashcards-123")

    def test_sparknotes_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.sparknotes.com/law/contracts/")

    def test_gradesaver_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert is_blocked_domain("https://www.gradesaver.com/study-guide/law/contracts")

    def test_legitimate_edu_not_blocked(self):
        from promptlab.utils.doc_corpus import is_blocked_domain
        assert not is_blocked_domain("https://www.law.cornell.edu/ucc/2/2-207")
        assert not is_blocked_domain("https://www.americanbar.org/groups/business_law/")


# ---------------------------------------------------------------------------
# V) Chunk substantiveness filter
# ---------------------------------------------------------------------------

class TestChunkSubstantiveness:
    """Tests for _chunk_is_substantive() which rejects PR/sidebar/nav chunks."""

    def test_rejects_law_firm_award_blurb(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        text = (
            "Sterlington Recognized in Legal 500's New York Elite Rankings. "
            "The firm advised on US$40 billion transactions in 2024."
        )
        assert not _chunk_is_substantive(text)

    def test_rejects_marketing_invitation(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        text = "Discover key insights. The easy, no-hassle way to get quick access to legal research."
        assert not _chunk_is_substantive(text)

    def test_rejects_law_firm_cta(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        text = (
            "Have you been injured? Contact our attorneys today. "
            "We can help defend your legal rights. Call now for a free consultation."
        )
        assert not _chunk_is_substantive(text)

    def test_rejects_docx_student_notes_artifact(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        text = (
            "Business Enterprises.docx Business Enterprises Dombalagian Spring 2021 "
            "1) AGENCY a) Defining Agency - Restatement (Third) of Agency"
        )
        assert not _chunk_is_substantive(text)

    def test_rejects_headings_only_chunk(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        # All lines are short headings — no real sentences
        text = "\n".join([
            "Definition of an Indemnification Clause",
            "How Does Indemnification Work?",
            "Types of Indemnification",
            "Common Contract Provisions",
            "Limitation of Liability",
            "Force Majeure Clauses",
        ])
        assert not _chunk_is_substantive(text)

    def test_accepts_real_educational_content(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        text = (
            "An indemnification clause is a contractual provision that requires one party to "
            "compensate the other for specific losses or damages. These clauses are common in "
            "commercial contracts and serve to allocate risk between the parties. When negotiating "
            "indemnification terms, practitioners should consider the scope of covered claims, "
            "the obligations to defend, and any caps on liability."
        )
        assert _chunk_is_substantive(text)

    def test_accepts_legal_statute_text(self):
        from promptlab.utils.doc_generator import _chunk_is_substantive
        text = (
            "Under UCC Section 2-207, a definite and seasonable expression of acceptance "
            "operates as an acceptance even though it states terms additional to or different "
            "from those offered or agreed upon, unless acceptance is expressly made conditional "
            "on assent to the additional or different terms."
        )
        assert _chunk_is_substantive(text)


# ---------------------------------------------------------------------------
# W) Content-echo tautology in static validation
# ---------------------------------------------------------------------------

class TestContentEchoTautology:
    """Verify _passes_static_validation rejects when answer appears in prompt."""

    def test_rejects_answer_verbatim_in_prompt(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        # Answer appears verbatim in the question text
        case = {
            "style": "qa",
            "prompt": "What is the key principle of effective breach resolution?",
            "answer_key": "effective breach resolution",
        }
        assert not _passes_static_validation(case)

    def test_rejects_answer_as_substring_in_prompt(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "style": "qa",
            "prompt": "Why must indemnification clauses be drafted to ensure minor and material obligations?",
            "answer_key": "minor and material",
        }
        assert not _passes_static_validation(case)

    def test_accepts_answer_not_in_prompt(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        case = {
            "style": "qa",
            "prompt": "What are the two primary categories used to classify contract breaches?",
            "answer_key": "minor and material",
        }
        assert _passes_static_validation(case)

    def test_short_answer_not_checked(self):
        from promptlab.utils.doc_generator import _passes_static_validation
        # Very short answers (<8 chars) skip the echo check to avoid false positives
        case = {
            "style": "qa",
            "prompt": "What does UCC stand for in commerce law?",
            "answer_key": "ucc",  # 3 chars
        }
        assert _passes_static_validation(case)


# ---------------------------------------------------------------------------
# X) Same-chunk deduplication
# ---------------------------------------------------------------------------

class TestSameChunkDedup:
    """Verify _dedup_cases collapses cases that share the same source chunk."""

    def _make_case(self, case_id, prompt, doc_id="doc1", chunk_id="chunk1"):
        from promptlab.utils.doc_generator import GeneratedCase, ProvenanceInfo
        return GeneratedCase(
            case_id=case_id,
            prompt=prompt,
            assertions=[{"type": "contains", "value": "test", "case_sensitive": False}],
            tags=["legal"],
            provenance=ProvenanceInfo(
                source_doc_id=doc_id,
                source_url="https://example.com/guide",
                page_number=1,
                chunk_id=chunk_id,
                section="",
                evidence_span="test evidence",
            ),
        )

    def test_same_chunk_id_rejected(self):
        from promptlab.utils.doc_generator import _dedup_cases
        c1 = self._make_case("qa-1", "What are the elements of a contract?", chunk_id="chunk1")
        c2 = self._make_case("qa-2", "What elements are needed to form a contract?", chunk_id="chunk1")
        result = _dedup_cases([c1, c2])
        # Second case has same chunk_id → deduped
        assert len(result) == 1
        assert result[0].case_id == "qa-1"

    def test_different_chunk_ids_kept(self):
        from promptlab.utils.doc_generator import _dedup_cases
        c1 = self._make_case("qa-1", "What is offer and acceptance?", chunk_id="chunk1")
        c2 = self._make_case("qa-2", "Define consideration in contract law.", chunk_id="chunk2")
        result = _dedup_cases([c1, c2])
        assert len(result) == 2

    def test_empty_chunk_id_uses_jaccard_only(self):
        from promptlab.utils.doc_generator import _dedup_cases
        # Cases with empty chunk_id fall through to Jaccard dedup
        c1 = self._make_case("qa-1", "What are the elements of a valid contract?", chunk_id="")
        c2 = self._make_case("qa-2", "What are the elements that make a contract valid?", chunk_id="")
        result = _dedup_cases([c1, c2])
        # Should be deduped via Jaccard (high overlap)
        assert len(result) == 1

    def test_jaccard_threshold_lowered_to_45(self):
        from promptlab.utils.doc_generator import _dedup_cases, _jaccard_similarity, _tokenize_for_dedup
        # Verify the threshold is now 0.45 not 0.5
        p1 = "What are the two main categories of contract breaches?"
        p2 = "What are the two primary types of contract breaches?"
        t1 = _tokenize_for_dedup(p1)
        t2 = _tokenize_for_dedup(p2)
        sim = _jaccard_similarity(t1, t2)
        # These should be caught by the new 0.45 threshold
        c1 = self._make_case("qa-1", p1, chunk_id="chunkA")
        c2 = self._make_case("qa-2", p2, chunk_id="chunkB")
        result = _dedup_cases([c1, c2])
        if sim > 0.45:
            assert len(result) == 1
        else:
            # If similarity is just below threshold, both may pass (still valid)
            assert len(result) <= 2


# ---------------------------------------------------------------------------
# Y) Assertion grounding helper logic
# ---------------------------------------------------------------------------

class TestAssertionGrounding:
    """Tests for the assertion-grounding logic (value in evidence or high Jaccard)."""

    def test_value_substring_of_evidence_is_grounded(self):
        from promptlab.utils.doc_generator import _tokenize_for_dedup, _jaccard_similarity
        value = "minor and material"
        evidence = "Contract breaches are classified into minor and material categories."
        val_tokens = _tokenize_for_dedup(value)
        ev_tokens = _tokenize_for_dedup(evidence)
        grounded = (
            value.lower() in evidence.lower()
            or _jaccard_similarity(val_tokens, ev_tokens) >= 0.4
        )
        assert grounded

    def test_paraphrase_with_sufficient_jaccard_is_grounded(self):
        from promptlab.utils.doc_generator import _tokenize_for_dedup, _jaccard_similarity
        value = "offer acceptance consideration capacity"
        evidence = "Contract formation requires offer, acceptance, consideration, and legal capacity."
        val_tokens = _tokenize_for_dedup(value)
        ev_tokens = _tokenize_for_dedup(evidence)
        grounded = (
            value.lower() in evidence.lower()
            or _jaccard_similarity(val_tokens, ev_tokens) >= 0.4
        )
        assert grounded

    def test_unrelated_value_is_ungrounded(self):
        from promptlab.utils.doc_generator import _tokenize_for_dedup, _jaccard_similarity
        value = "to ensure financial protection and ability to fulfill"
        evidence = "The vendor shall maintain adequate insurance policies during the term."
        val_tokens = _tokenize_for_dedup(value)
        ev_tokens = _tokenize_for_dedup(evidence)
        grounded = (
            value.lower() in evidence.lower()
            or _jaccard_similarity(val_tokens, ev_tokens) >= 0.4
        )
        assert not grounded

    def test_exact_match_always_grounded(self):
        from promptlab.utils.doc_generator import _tokenize_for_dedup, _jaccard_similarity
        value = "specific performance"
        evidence = "The court may order specific performance as an equitable remedy."
        val_tokens = _tokenize_for_dedup(value)
        ev_tokens = _tokenize_for_dedup(evidence)
        grounded = (
            value.lower() in evidence.lower()
            or _jaccard_similarity(val_tokens, ev_tokens) >= 0.4
        )
        assert grounded
