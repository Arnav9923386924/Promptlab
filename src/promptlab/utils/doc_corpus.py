"""Document Corpus Manager — discovery, download, and caching of domain docs.

Flow:
1. Analyze BSP to infer domain/role.
2. Generate targeted search queries for authoritative documents.
3. Download HTML pages, extract clean text.
4. Save to .promptlab/corpus/ with metadata manifest (doc_id, URL, title,
   fetched_at, content_hash).
5. Skip re-download when content hash unchanged.

This module uses the existing WebScraper for HTTP + HTML parsing, so no new
network-layer dependencies are needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from promptlab.utils.scraper import ScrapedContent, ScraperConfig, WebScraper

console = Console()

# ---------------------------------------------------------------------------
# Domain blocklist & content-quality heuristics
# ---------------------------------------------------------------------------

# Hard-blocked domains: e-commerce, social media, forums, review aggregators
_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    # E-commerce
    "amazon.com", "www.amazon.com", "amazon.co.uk", "www.amazon.co.uk",
    "ebay.com", "www.ebay.com", "etsy.com", "www.etsy.com",
    "walmart.com", "www.walmart.com", "alibaba.com", "www.alibaba.com",
    "shopify.com", "www.shopify.com", "target.com", "www.target.com",
    # Social media
    "facebook.com", "www.facebook.com", "twitter.com", "www.twitter.com",
    "x.com", "www.x.com", "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com", "linkedin.com", "www.linkedin.com",
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "pinterest.com", "www.pinterest.com", "snapchat.com",
    # Video / streaming
    "youtube.com", "www.youtube.com", "youtu.be",
    "vimeo.com", "www.vimeo.com", "twitch.tv", "www.twitch.tv",
    # Forums / Q&A
    "quora.com", "www.quora.com", "answers.yahoo.com",
    "stackexchange.com", "stackoverflow.com",
    # Review / rating aggregators
    "yelp.com", "www.yelp.com", "trustpilot.com", "www.trustpilot.com",
    "g2.com", "www.g2.com", "capterra.com", "www.capterra.com",
    "glassdoor.com", "www.glassdoor.com",
    # News aggregators / low-signal
    "buzzfeed.com", "www.buzzfeed.com",
    "medium.com",  # often paywalled / opinion pieces
    # Student-note aggregators / homework-help sites
    "cliffsnotes.com", "www.cliffsnotes.com",
    "coursehero.com", "www.coursehero.com",
    "studocu.com", "www.studocu.com",
    "chegg.com", "www.chegg.com",
    "quizlet.com", "www.quizlet.com",
    "sparknotes.com", "www.sparknotes.com",
    "gradesaver.com", "www.gradesaver.com",
    "schmoop.com", "www.schmoop.com",
    "litcharts.com", "www.litcharts.com",
})

# Regex patterns that identify vendor-marketing / product-pitch pages
_VENDOR_MARKETING_RE = re.compile(
    r"(book\s+a\s+demo|schedule\s+a\s+demo|request\s+a\s+demo|"
    r"free\s+trial|start\s+your\s+free|"
    r"award[- ]winning|breakthrough\s+award|"
    r"customer\s+survey|trusted\s+by\s+\d|"
    r"sign\s+up\s+(now|today|free)|get\s+started\s+(for\s+)?free|"
    r"pricing\s+plans?|compare\s+plans|"
    r"rated\s+#?\d|leader\s+in\s+g2|"
    r"our\s+platform\s+(helps|enables|empowers)|"
    r"roi\s+calculator|total\s+cost\s+of\s+ownership)",
    re.IGNORECASE,
)

# CTA / thin-content signals
_CTA_HEAVY_RE = re.compile(
    r"(buy\s+now|add\s+to\s+cart|shop\s+now|order\s+now|"
    r"customers\s+who\s+bought|frequently\s+bought\s+together|"
    r"sponsored\s+products|related\s+products|"
    r"product\s+details|product\s+description|"
    r"your\s+recently\s+viewed|browsing\s+history|"
    r"people\s+found\s+this\s+helpful)",
    re.IGNORECASE,
)


def is_blocked_domain(url: str) -> bool:
    """Return True if the URL belongs to a hard-blocked domain."""
    netloc = urlparse(url).netloc.lower()
    # Strip port
    if ":" in netloc:
        netloc = netloc.split(":")[0]
    # Check exact match
    if netloc in _BLOCKED_DOMAINS:
        return True
    # Check if it's a subdomain of a blocked domain (e.g. smile.amazon.com)
    for bd in _BLOCKED_DOMAINS:
        if netloc.endswith("." + bd):
            return True
    return False


def compute_content_quality_score(text: str) -> float:
    """Score page content quality from 0.0 (junk) to 1.0 (high substance).

    Penalises:
    - CTA-heavy / product-page patterns
    - Vendor-marketing language
    - Very short content
    - High ratio of short lines (nav fragments)
    """
    if not text:
        return 0.0

    score = 1.0
    lines = text.split("\n")
    total_lines = len(lines) or 1

    # Penalise thin content
    if len(text) < 500:
        score -= 0.4
    elif len(text) < 1000:
        score -= 0.2

    # Penalise high ratio of short lines (nav/boilerplate)
    short_lines = sum(1 for l in lines if len(l.strip()) < 30)
    short_ratio = short_lines / total_lines
    if short_ratio > 0.6:
        score -= 0.3
    elif short_ratio > 0.4:
        score -= 0.15

    # Penalise CTA-heavy content
    cta_hits = len(_CTA_HEAVY_RE.findall(text))
    if cta_hits >= 3:
        score -= 0.4
    elif cta_hits >= 1:
        score -= 0.15

    # Penalise vendor-marketing
    vendor_hits = len(_VENDOR_MARKETING_RE.findall(text))
    if vendor_hits >= 3:
        score -= 0.4
    elif vendor_hits >= 1:
        score -= 0.15

    return max(0.0, min(1.0, score))

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    """A single downloaded document in the corpus."""
    doc_id: str
    url: str
    title: str
    content_hash: str
    fetched_at: str  # ISO-8601
    char_count: int
    domain: str  # netloc
    text: str = ""  # full clean text (loaded lazily from file when needed)

    # paths are relative to corpus_dir
    text_file: str = ""       # e.g. "doc_abc123.txt"
    metadata_file: str = ""   # e.g. "doc_abc123.meta.json"


@dataclass
class CorpusManifest:
    """Top-level manifest tracking all documents in the corpus."""
    created_at: str = ""
    updated_at: str = ""
    bsp_domain: str = ""
    bsp_role: str = ""
    documents: list[dict] = field(default_factory=list)  # list of serialised CorpusDocument (minus text)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CorpusManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Corpus Manager
# ---------------------------------------------------------------------------

class CorpusManager:
    """Downloads and caches domain-relevant documents for testcase generation."""

    # Extra search suffixes for richer content.
    _QUERY_SUFFIXES = [
        "best practices guide",
        "documentation official",
        "FAQ frequently asked questions",
        "tutorial introduction",
        "common mistakes pitfalls",
        "rules regulations overview",
        "terminology glossary",
        "case study examples",
    ]

    def __init__(
        self,
        project_root: Path,
        serpapi_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        max_docs: int = 20,
        max_pages_per_doc: int = 1,
        scraper_timeout: float = 30.0,
    ):
        self.project_root = project_root
        self.corpus_dir = project_root / ".promptlab" / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.corpus_dir / "manifest.json"
        self.serpapi_key = serpapi_key
        self.brave_api_key = brave_api_key
        self.max_docs = max_docs
        self.max_pages_per_doc = max_pages_per_doc
        self.scraper_timeout = scraper_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_corpus(
        self,
        domain: str,
        role: str,
        search_queries: list[str],
        keywords: list[str],
    ) -> list[CorpusDocument]:
        """Discover, download, and cache documents.  Returns corpus list."""

        manifest = self._load_manifest()
        existing_hashes: dict[str, CorpusDocument] = {}
        for doc_dict in manifest.documents:
            existing_hashes[doc_dict["url"]] = CorpusDocument(**doc_dict)

        console.print("[bold cyan]  Phase 1/3: Discovering documents...[/bold cyan]")
        urls = await self._discover_urls(domain, role, search_queries, keywords)
        console.print(f"  [green]✓[/green] Found {len(urls)} candidate URLs")

        console.print("[bold cyan]  Phase 2/3: Downloading & caching...[/bold cyan]")
        docs = await self._download_docs(urls, existing_hashes)
        console.print(f"  [green]✓[/green] Corpus: {len(docs)} documents ({sum(d.char_count for d in docs):,} chars)")

        console.print("[bold cyan]  Phase 3/3: Saving manifest...[/bold cyan]")
        self._save_manifest(docs, domain, role)
        console.print(f"  [green]✓[/green] Manifest saved to {self.manifest_path}")

        return docs

    def load_corpus(self) -> list[CorpusDocument]:
        """Load an existing corpus from disk (no network)."""
        manifest = self._load_manifest()
        docs: list[CorpusDocument] = []
        for doc_dict in manifest.documents:
            doc = CorpusDocument(**doc_dict)
            text_path = self.corpus_dir / doc.text_file
            if text_path.exists():
                doc.text = text_path.read_text(encoding="utf-8")
                docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def _discover_urls(
        self,
        domain: str,
        role: str,
        search_queries: list[str],
        keywords: list[str],
    ) -> list[str]:
        """Use search engines to find relevant document URLs."""
        scraper_config = ScraperConfig(
            max_pages=self.max_docs,
            serpapi_key=self.serpapi_key,
            brave_api_key=self.brave_api_key,
            timeout_seconds=self.scraper_timeout,
        )
        scraper = WebScraper(scraper_config)

        all_urls: list[str] = []
        seen_domains: set[str] = set()

        # Build a wider set of queries.
        queries: list[str] = list(search_queries)
        domain_clean = domain.replace("_", " ")
        for suffix in self._QUERY_SUFFIXES:
            queries.append(f"{domain_clean} {suffix}")
        if role and role.lower() != "ai assistant":
            queries.append(f"{role} professional guide")

        # Deduplicate queries.
        seen_q: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            q_low = q.lower().strip()
            if q_low not in seen_q:
                seen_q.add(q_low)
                unique_queries.append(q)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("Searching...", total=len(unique_queries))
            for query in unique_queries:
                if len(all_urls) >= self.max_docs:
                    break
                progress.update(task, description=f"Search: {query[:45]}...")
                try:
                    found = await scraper.search(query, num_results=5, silent=True)
                    for u in (found or []):
                        netloc = urlparse(u).netloc
                        if netloc not in seen_domains and len(all_urls) < self.max_docs:
                            # skip obviously bad URLs
                            if not self._is_noise_url(u):
                                seen_domains.add(netloc)
                                all_urls.append(u)
                except Exception:
                    pass  # individual query failure is non-fatal
                progress.advance(task)
                await asyncio.sleep(0.3)

        await scraper.close()
        return all_urls

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def _download_docs(
        self,
        urls: list[str],
        existing: dict[str, CorpusDocument],
    ) -> list[CorpusDocument]:
        """Download URLs, cache text files, skip unchanged content."""
        scraper_config = ScraperConfig(
            max_pages=len(urls),
            serpapi_key=self.serpapi_key,
            brave_api_key=self.brave_api_key,
            timeout_seconds=self.scraper_timeout,
        )
        scraper = WebScraper(scraper_config)
        docs: list[CorpusDocument] = []

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("Downloading...", total=len(urls))
            for url in urls:
                progress.update(task, description=f"Fetching: {urlparse(url).netloc}...")
                try:
                    content_list = await scraper.crawl([url], max_pages=self.max_pages_per_doc, show_progress=False)
                    if not content_list:
                        progress.advance(task)
                        continue
                    content: ScrapedContent = content_list[0]
                    text = self._clean_doc_text(content.text)
                    if len(text) < 200:
                        progress.advance(task)
                        continue  # too thin

                    # Content quality gate
                    quality = compute_content_quality_score(text)
                    if quality < 0.35:
                        console.print(f"  [dim]Skip (quality={quality:.2f}): {urlparse(url).netloc}[/dim]")
                        progress.advance(task)
                        continue

                    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

                    # Check cache
                    if url in existing and existing[url].content_hash == content_hash:
                        old_doc = existing[url]
                        old_doc.text = text
                        docs.append(old_doc)
                        progress.advance(task)
                        continue

                    doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
                    text_file = f"doc_{doc_id}.txt"
                    meta_file = f"doc_{doc_id}.meta.json"
                    (self.corpus_dir / text_file).write_text(text, encoding="utf-8")

                    doc = CorpusDocument(
                        doc_id=doc_id,
                        url=url,
                        title=content.title or urlparse(url).path,
                        content_hash=content_hash,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        char_count=len(text),
                        domain=urlparse(url).netloc,
                        text=text,
                        text_file=text_file,
                        metadata_file=meta_file,
                    )
                    # Persist per-doc metadata
                    meta = {k: v for k, v in asdict(doc).items() if k != "text"}
                    (self.corpus_dir / meta_file).write_text(
                        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    docs.append(doc)
                except Exception as e:
                    console.print(f"  [yellow]Skip {urlparse(url).netloc}: {e}[/yellow]")
                progress.advance(task)
                await asyncio.sleep(0.2)

        await scraper.close()
        return docs

    # ------------------------------------------------------------------
    # Cleaning / filtering
    # ------------------------------------------------------------------

    _NOISE_URL_PATTERNS = re.compile(
        r"(login|signup|register|cart|checkout|pricing|cookie|privacy-policy|terms-of-service"
        r"|\.pdf$|\.jpg$|\.png$|\.gif$|youtube\.com|facebook\.com|twitter\.com|instagram\.com)",
        re.IGNORECASE,
    )

    def _is_noise_url(self, url: str) -> bool:
        # Hard-blocked domain check first
        if is_blocked_domain(url):
            return True
        return bool(self._NOISE_URL_PATTERNS.search(url))

    _NOISE_TEXT_PATTERNS = re.compile(
        r"(cookie\s*(consent|policy|banner|hub|preferences|settings)|"
        r"accept\s+all\s+cookies|"
        r"we\s+use\s+cookies|"
        r"sign\s+in\s+to\s+your\s+account|"
        r"free\s+trial|"
        r"subscribe\s+to\s+our\s+newsletter|"
        r"follow\s+us\s+on|"
        r"©\s*\d{4}|"
        r"all\s+rights\s+reserved|"
        r"privacy\s+policy|"
        r"terms\s+(of\s+)?(service|use))",
        re.IGNORECASE,
    )

    def _clean_doc_text(self, raw: str) -> str:
        """Strip boilerplate / noise from scraped HTML text."""
        lines = raw.split("\n")
        clean: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) < 15:
                continue  # tiny nav fragments
            if self._NOISE_TEXT_PATTERNS.search(stripped):
                continue
            clean.append(stripped)
        return "\n".join(clean)

    # ------------------------------------------------------------------
    # Manifest persistence
    # ------------------------------------------------------------------

    def _load_manifest(self) -> CorpusManifest:
        if self.manifest_path.exists():
            try:
                return CorpusManifest.load(self.manifest_path)
            except Exception:
                pass
        return CorpusManifest(created_at=datetime.now(timezone.utc).isoformat())

    def _save_manifest(self, docs: list[CorpusDocument], domain: str, role: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        manifest = CorpusManifest(
            created_at=now,
            updated_at=now,
            bsp_domain=domain,
            bsp_role=role,
            documents=[
                {k: v for k, v in asdict(d).items() if k != "text"}
                for d in docs
            ],
        )
        manifest.save(self.manifest_path)
