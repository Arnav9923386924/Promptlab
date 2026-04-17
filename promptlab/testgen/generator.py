"""Automatic Test Generator - Creates tests from BSP via web scraping or document-grounded retrieval.

This module analyzes the Behavior Specification Prompt (BSP) to:
1. Extract domain keywords and context
2. Generate smart search queries
3. Scrape relevant content (web mode) OR download/index/retrieve docs (docs_web mode)
4. Auto-generate 50-100 test cases with optional provenance metadata

Generation modes:
- "web"      — scrape + regex extract (original pipeline)
- "docs_web" — download docs → TF-IDF index → retrieve → LLM/heuristic generate
- "hybrid"   — docs_web first, web fallback if target not met
"""

import asyncio
import re
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from promptlab.testgen.scraper import (
    WebScraper, 
    ScraperConfig, 
    ScrapedContent,
    scrape_for_domain,
)
from promptlab.testgen.data_processor import (
    DataProcessor,
    QAPair,
    MaskedTest,
    generate_test_cases_yaml,
    generate_masked_test_cases_yaml,
)

console = Console()


@dataclass
class BSPAnalysis:
    """Analysis of a Behavior Specification Prompt."""
    domain: str
    keywords: list[str]
    role: str
    capabilities: list[str]
    constraints: list[str]
    search_queries: list[str]
    test_categories: list[str]
    
    
@dataclass
class GeneratedTests:
    """Collection of auto-generated tests."""
    domain: str
    qa_pairs: list[QAPair]
    masked_tests: list[MaskedTest]
    yaml_content: dict
    output_file: Path
    generation_time: float
    scraped_sources: int
    # docs_web pipeline populates this; web mode leaves it empty
    generated_cases: list = field(default_factory=list)
    generation_mode_used: str = "web"
    

class BSPAnalyzer:
    """Analyzes BSP to extract domain, keywords, and generate search queries."""
    
    # Common role patterns
    ROLE_PATTERNS = [
        r"you are (?:a |an )?(.+?)(?:\.|,|who|that|$)",
        r"act as (?:a |an )?(.+?)(?:\.|,|who|that|$)",
        r"behave as (?:a |an )?(.+?)(?:\.|,|who|that|$)",
        r"assume the role of (?:a |an )?(.+?)(?:\.|,|who|that|$)",
        r"your role is (?:to be )?(?:a |an )?(.+?)(?:\.|,|$)",
    ]
    
    # Domain keywords to look for (ordered by specificity - most specific first)
    DOMAIN_KEYWORDS = {
        "code_review": ["code review", "review code", "code quality", "pull request", "pr review", "code feedback"],
        "code": ["code", "programming", "developer", "software", "engineer", "debug", "script", "function", "class", "api", "coding", "programmer"],
        "medical": ["medical", "health", "doctor", "patient", "diagnosis", "symptom", "treatment", "clinical", "healthcare"],
        "legal": ["legal", "law", "attorney", "lawyer", "contract", "court", "compliance", "regulation", "litigation"],
        "finance": ["finance", "financial", "banking", "investment", "trading", "stock", "money", "accounting", "tax"],
        "education": ["education", "teacher", "student", "learning", "course", "curriculum", "academic", "tutor"],
        "support": ["customer support", "help desk", "ticket", "troubleshoot", "customer service"],
        "writing": ["writing", "writer", "content", "article", "blog", "copywriting", "editing", "creative"],
        "data": ["data", "analysis", "analytics", "statistics", "database", "sql", "visualization", "metrics"],
        "security": ["security", "cybersecurity", "hacking", "vulnerability", "encryption", "firewall", "threat"],
        "devops": ["devops", "deployment", "ci/cd", "docker", "kubernetes", "infrastructure", "cloud", "aws", "azure"],
    }
    
    # Capability patterns
    CAPABILITY_PATTERNS = [
        r"(?:can|able to|capable of) (.+?)(?:\.|,|and|$)",
        r"(?:will|should) (.+?)(?:\.|,|and|$)",
        r"(?:helps?|assists?) (?:users? )?(?:to |with )?(.+?)(?:\.|,|and|$)",
        r"(?:provides?|offers?) (.+?)(?:\.|,|and|$)",
    ]
    
    # Constraint patterns
    CONSTRAINT_PATTERNS = [
        r"(?:don't|do not|never|avoid|must not) (.+?)(?:\.|,|$)",
        r"(?:only|exclusively|solely) (.+?)(?:\.|,|$)",
        r"(?:limited to|restricted to) (.+?)(?:\.|,|$)",
        r"(?:within|inside) (.+?)(?:\.|,|$)",
    ]
    
    def analyze(self, bsp: str) -> BSPAnalysis:
        """Analyze BSP and extract structured information."""
        bsp_lower = bsp.lower()
        
        # Extract role
        role = self._extract_role(bsp)
        
        # Detect domain
        domain = self._detect_domain(bsp_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(bsp)
        
        # Extract capabilities
        capabilities = self._extract_capabilities(bsp)
        
        # Extract constraints
        constraints = self._extract_constraints(bsp)
        
        # Generate search queries
        search_queries = self._generate_search_queries(domain, role, keywords, capabilities, bsp=bsp)
        
        # Determine test categories
        test_categories = self._determine_test_categories(domain, capabilities, constraints)
        
        return BSPAnalysis(
            domain=domain,
            keywords=keywords,
            role=role,
            capabilities=capabilities,
            constraints=constraints,
            search_queries=search_queries,
            test_categories=test_categories,
        )
    
    def _extract_role(self, bsp: str) -> str:
        """Extract the role description from BSP."""
        for pattern in self.ROLE_PATTERNS:
            match = re.search(pattern, bsp, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                # Clean up common suffixes
                role = re.sub(r'\s+who\s.*$', '', role, flags=re.IGNORECASE)
                role = re.sub(r'\s+that\s.*$', '', role, flags=re.IGNORECASE)
                return role[:100]  # Limit length
        
        # Fallback: use first sentence
        first_sentence = bsp.split('.')[0]
        return first_sentence[:100] if first_sentence else "AI Assistant"
    
    def _detect_domain(self, bsp_lower: str) -> str:
        """Detect the primary domain from BSP."""
        scores = {}
        
        # Check for multi-word phrases first (more specific)
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                # Give more weight to longer/more specific matches
                if kw in bsp_lower:
                    score += len(kw.split())  # Multi-word phrases score higher
            if score > 0:
                scores[domain] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def _extract_keywords(self, bsp: str) -> list[str]:
        """Extract important keywords from BSP."""
        keywords = []
        
        # Skip common filler words
        skip_words = {'your', 'you', 'the', 'and', 'for', 'with', 'that', 'this', 
                      'are', 'have', 'will', 'should', 'must', 'can', 'may',
                      'rules', 'format', 'response', 'end', 'start', 'list'}
        
        # Extract capitalized terms (likely important) - get multi-word phrases
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', bsp)
        for c in caps:
            c_lower = c.lower()
            if len(c) > 3 and c_lower not in skip_words:
                keywords.append(c_lower)
        
        # Extract technical terms (snake_case, camelCase)
        tech = re.findall(r'\b([a-z]+_[a-z]+)\b|\b([a-z]+[A-Z][a-z]+)\b', bsp)
        for t in tech:
            keywords.extend([x.lower() for x in t if x])
        
        # Extract quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', bsp)
        keywords.extend([q.lower() for q in quoted if len(q) > 2 and len(q) < 50])
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen and len(kw) > 2 and kw not in skip_words:
                seen.add(kw)
                unique.append(kw)
        
        return unique[:20]
    
    def _extract_capabilities(self, bsp: str) -> list[str]:
        """Extract capabilities mentioned in BSP."""
        capabilities = []
        for pattern in self.CAPABILITY_PATTERNS:
            matches = re.findall(pattern, bsp, re.IGNORECASE)
            capabilities.extend([m.strip()[:100] for m in matches if len(m.strip()) > 5])
        return capabilities[:10]
    
    def _extract_constraints(self, bsp: str) -> list[str]:
        """Extract constraints mentioned in BSP."""
        constraints = []
        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, bsp, re.IGNORECASE)
            constraints.extend([m.strip()[:100] for m in matches if len(m.strip()) > 5])
        return constraints[:10]
    
    def _generate_search_queries(
        self, 
        domain: str, 
        role: str, 
        keywords: list[str],
        capabilities: list[str],
        bsp: str = "",
    ) -> list[str]:
        """Generate smart search queries based on BSP analysis.

        When BSP concepts are available, generates concept-targeted queries
        instead of generic domain queries.
        """
        from promptlab.testgen.doc_generator import extract_bsp_concepts

        queries = []
        
        # Clean domain for search (replace underscores with spaces)
        domain_clean = domain.replace("_", " ")

        # --- Concept-targeted queries (high value) ---
        if bsp:
            concepts = extract_bsp_concepts(bsp, domain)
            for concept_label, concept_kws in concepts[:10]:
                # Use the first keyword phrase as the search query seed
                phrase = concept_kws[0] if concept_kws else concept_label.replace("_", " ")
                queries.append(f"{phrase} legal principles")
                queries.append(f"{phrase} explained guide")
        
        # --- Domain-wide queries (fallback / supplement) ---
        queries.append(f"{domain_clean} best practices")
        queries.append(f"{domain_clean} key concepts guide")
        queries.append(f"{domain_clean} common mistakes")
        queries.append(f"{domain_clean} guide")
        
        # Role-specific queries
        if role and role != "AI Assistant":
            role_clean = role.replace("a ", "").replace("an ", "")[:50]
            queries.append(f"{role_clean} guide")
        
        # More targeted keyword queries
        useful_keywords = [kw for kw in keywords if len(kw) > 4][:3]
        for kw in useful_keywords:
            queries.append(f"{kw} best practices")
        
        # Remove duplicates
        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        
        return unique[:20]  # Allow more queries for concept coverage
    
    def _determine_test_categories(
        self,
        domain: str,
        capabilities: list[str],
        constraints: list[str]
    ) -> list[str]:
        """Determine what categories of tests to generate."""
        categories = ["knowledge"]  # Always test knowledge
        
        if capabilities:
            categories.append("capability")
        
        if constraints:
            categories.append("constraint")  # Test that constraints are followed
        
        # Domain-specific test categories
        domain_tests = {
            "code": ["code_generation", "debugging", "explanation"],
            "medical": ["accuracy", "safety", "disclaimer"],
            "legal": ["accuracy", "disclaimer", "jurisdiction"],
            "finance": ["accuracy", "disclaimer", "risk"],
            "support": ["helpfulness", "clarity", "resolution"],
            "writing": ["creativity", "grammar", "style"],
        }
        
        if domain in domain_tests:
            categories.extend(domain_tests[domain])
        
        return list(set(categories))


class AutoTestGenerator:
    """Generates tests automatically from BSP using web scraping or document-grounded retrieval."""
    
    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        max_pages: int = 20,
        # docs_web pipeline params
        project_root: Optional[Path] = None,
        llm_runner=None,
        llm_model: str = "ollama/llama3.1:8b",
        max_docs: int = 20,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        retrieval_top_k: int = 10,
        scraper_timeout: float = 30.0,
    ):
        """Initialize the auto test generator.
        
        Args:
            serpapi_key: SerpAPI key for Google search (recommended)
            brave_api_key: Brave Search API key (optional)
            max_pages: Maximum pages to scrape
            project_root: Project root for corpus cache (docs_web mode)
            llm_runner: LLMRunner instance for LLM-based generation (optional)
            llm_model: Model identifier for LLM calls
            max_docs: Maximum documents to download (docs_web mode)
            chunk_size: Chunk size in words for document indexing
            chunk_overlap: Overlap in words between chunks
            retrieval_top_k: Top-K chunks per retrieval query
        """
        self.serpapi_key = serpapi_key
        self.brave_api_key = brave_api_key
        self.max_pages = max_pages
        self.analyzer = BSPAnalyzer()
        # docs_web params
        self.project_root = project_root or Path.cwd()
        self.llm_runner = llm_runner
        self.llm_model = llm_model
        self.max_docs = max_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_top_k = retrieval_top_k
        self.scraper_timeout = scraper_timeout
    
    async def generate_tests(
        self,
        bsp: str,
        target_count: int = 50,
        output_dir: Optional[Path] = None,
        output_type: Literal["benchmark", "cloze", "all"] = "all",
        generation_mode: str = "web",
    ) -> GeneratedTests:
        """Generate tests automatically from BSP.
        
        Generation modes:
        - "web":      Scrape web → regex extract (original pipeline).
        - "docs_web": Download docs → index → retrieve → LLM/heuristic gen.
        - "hybrid":   docs_web first, web fallback if target not met.
        
        Args:
            bsp: The Behavior Specification Prompt
            target_count: Target number of accepted test cases
            output_dir: Directory to save generated tests
            output_type: Type of tests to generate
            generation_mode: "web", "docs_web", or "hybrid"
            
        Returns:
            GeneratedTests object with all generated test data
        """
        start_time = asyncio.get_event_loop().time()
        
        # Step 1: Analyze BSP
        console.print("[bold cyan]Step 1/4: Analyzing BSP...[/bold cyan]")
        analysis = self.analyzer.analyze(bsp)
        
        console.print(f"  [green]✓[/green] Domain: [bold]{analysis.domain}[/bold]")
        console.print(f"  [green]✓[/green] Role: {analysis.role[:60]}...")
        console.print(f"  [green]✓[/green] Keywords: {', '.join(analysis.keywords[:5])}")
        console.print(f"  [green]✓[/green] Search queries: {len(analysis.search_queries)}")
        console.print(f"  [green]✓[/green] Mode: {generation_mode}")
        
        # ------ docs_web / hybrid dispatch ------
        if generation_mode in ("docs_web", "hybrid"):
            result = await self._generate_docs_web(
                bsp=bsp,
                analysis=analysis,
                target_count=target_count,
                output_dir=output_dir,
                output_type=output_type,
                start_time=start_time,
            )
            # In hybrid mode, fall back to web if docs_web didn't reach target
            total_docs = len(result.generated_cases)
            if generation_mode == "hybrid" and total_docs < target_count:
                console.print(
                    f"\n[bold cyan]Hybrid fallback: docs_web produced {total_docs}/{target_count}. "
                    f"Running web scraping for remainder...[/bold cyan]"
                )
                web_result = await self._generate_web(
                    bsp=bsp,
                    analysis=analysis,
                    target_count=target_count - total_docs,
                    output_dir=output_dir,
                    output_type=output_type,
                    generation_mode="web",
                    start_time=start_time,
                )
                # Merge web results into docs_web result
                result.qa_pairs = web_result.qa_pairs
                result.masked_tests = web_result.masked_tests
                result.scraped_sources = web_result.scraped_sources
                result.generation_mode_used = "hybrid"
                # Rebuild combined yaml_content
                end_time = asyncio.get_event_loop().time()
                result.generation_time = end_time - start_time
                # Re-write YAML with combined content — append web cases
                combined_yaml = dict(result.yaml_content)
                web_cases = web_result.yaml_content.get("cases", [])
                combined_yaml["cases"] = combined_yaml.get("cases", []) + web_cases
                combined_yaml["metadata"]["test_count"] = len(combined_yaml["cases"])
                combined_yaml["metadata"]["generation_mode"] = "hybrid"
                result.yaml_content = combined_yaml
                # Overwrite output file
                import yaml as _yaml
                with open(result.output_file, "w", encoding="utf-8") as f:
                    _yaml.dump(combined_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return result
        
        # ------ web (original) pipeline ------
        return await self._generate_web(
            bsp=bsp,
            analysis=analysis,
            target_count=target_count,
            output_dir=output_dir,
            output_type=output_type,
            generation_mode=generation_mode,
            start_time=start_time,
        )
    
    # ------------------------------------------------------------------
    # docs_web pipeline
    # ------------------------------------------------------------------
    
    async def _generate_docs_web(
        self,
        bsp: str,
        analysis: BSPAnalysis,
        target_count: int,
        output_dir: Optional[Path],
        output_type: Literal["benchmark", "cloze", "all"],
        start_time: float,
    ) -> GeneratedTests:
        """Document-grounded generation: download → index → retrieve → generate."""
        from promptlab.testgen.doc_corpus import CorpusManager
        from promptlab.testgen.doc_index import DocumentIndex
        from promptlab.testgen.doc_generator import DocGroundedGenerator
        
        # Phase 1: Build / update corpus
        console.print("\n[bold cyan]Step 2/4: Building document corpus...[/bold cyan]")
        corpus_mgr = CorpusManager(
            project_root=self.project_root,
            serpapi_key=self.serpapi_key,
            brave_api_key=self.brave_api_key,
            max_docs=self.max_docs,
            scraper_timeout=self.scraper_timeout,
        )
        docs = await corpus_mgr.build_corpus(
            domain=analysis.domain,
            role=analysis.role,
            search_queries=analysis.search_queries,
            keywords=analysis.keywords,
        )
        
        if not docs:
            raise ValueError(
                "No documents could be downloaded. "
                "Check your search API keys or try 'web' mode."
            )
        
        # Phase 2: Index
        console.print("\n[bold cyan]Step 3/4: Indexing & generating testcases...[/bold cyan]")
        index = DocumentIndex(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        chunk_count = index.build(docs)
        console.print(f"  [green]✓[/green] Index: {chunk_count} chunks from {len(docs)} documents")
        
        if chunk_count == 0:
            raise ValueError("No indexable content — corpus documents are too small.")
        
        # Phase 3: Generate
        generator = DocGroundedGenerator(
            index=index,
            runner=self.llm_runner,
            model=self.llm_model,
            retrieval_top_k=self.retrieval_top_k,
        )
        
        cases, output_file = await generator.generate(
            bsp=bsp,
            domain=analysis.domain,
            role=analysis.role,
            keywords=analysis.keywords,
            capabilities=analysis.capabilities,
            constraints=analysis.constraints,
            target_count=target_count,
            output_dir=output_dir,
            output_type=output_type,
        )
        
        end_time = asyncio.get_event_loop().time()
        
        console.print(f"\n[bold cyan]Step 4/4: Summary[/bold cyan]")
        console.print(f"  [green]✓[/green] Generated {len(cases)} testcases (docs_web)")
        console.print(f"  [green]✓[/green] Saved to: {output_file}")
        console.print(f"  [green]✓[/green] Time: {end_time - start_time:.1f}s")
        
        # Read the YAML back so we can populate yaml_content
        yaml_content: dict = {}
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f) or {}
        except Exception:
            pass
        
        return GeneratedTests(
            domain=analysis.domain,
            qa_pairs=[],        # docs_web uses GeneratedCase, not QAPair
            masked_tests=[],
            yaml_content=yaml_content,
            output_file=output_file,
            generation_time=end_time - start_time,
            scraped_sources=len(docs),
            generated_cases=cases,
            generation_mode_used="docs_web",
        )
    
    # ------------------------------------------------------------------
    # web (original) pipeline
    # ------------------------------------------------------------------
    
    async def _generate_web(
        self,
        bsp: str,
        analysis: BSPAnalysis,
        target_count: int,
        output_dir: Optional[Path],
        output_type: Literal["benchmark", "cloze", "all"],
        generation_mode: str,
        start_time: float,
    ) -> GeneratedTests:
        """Original web scraping + regex extraction pipeline."""
        MAX_ROUNDS = 4
        
        console.print("\n[bold cyan]Step 2/4: Scraping relevant web content...[/bold cyan]")
        
        scraper_config = ScraperConfig(
            max_pages=self.max_pages,
            serpapi_key=self.serpapi_key,
            brave_api_key=self.brave_api_key,
            timeout_seconds=self.scraper_timeout,
        )
        scraper = WebScraper(scraper_config)
        
        processor = DataProcessor()
        all_qa_pairs: list[QAPair] = []
        all_masked: list[MaskedTest] = []
        scraped_count = 0
        
        # Oversample factor: scrape ~1.5× target to account for dedup losses
        oversample = int(target_count * 1.5)
        
        # Calculate per-type targets up front
        if output_type == "all":
            qa_target = int(target_count * 0.6)
            masked_target = target_count - qa_target
        elif output_type == "benchmark":
            qa_target = target_count
            masked_target = 0
        else:
            qa_target = 0
            masked_target = target_count
        
        queries_used: set[str] = set()
        scraped_domains: set[str] = set()
        
        for round_num in range(1, MAX_ROUNDS + 1):
            accepted = len(all_qa_pairs) + len(all_masked)
            if accepted >= target_count:
                break
            
            # Pick queries not yet used; if exhausted, re-use with different page depth
            remaining_queries = [q for q in analysis.search_queries if q not in queries_used]
            if not remaining_queries:
                # Augment queries with round-specific suffixes for variety
                remaining_queries = [
                    f"{q} examples round {round_num}"
                    for q in analysis.search_queries[:4]
                ]
            
            # Scale scraping effort to gap
            gap = target_count - accepted
            pages_this_round = max(3, min(self.max_pages, int(gap * 0.4)))
            pages_per_query = max(2, pages_this_round // max(1, len(remaining_queries)))
            
            if round_num > 1:
                console.print(f"  [cyan]Round {round_num}: need {gap} more tests, scraping {pages_this_round} pages...[/cyan]")
            
            round_content: list[ScrapedContent] = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Searching & scraping...", 
                    total=len(remaining_queries)
                )
                
                for query in remaining_queries:
                    if len(round_content) >= pages_this_round:
                        break
                    queries_used.add(query)
                    progress.update(task, description=f"Searching: {query[:40]}...")
                    
                    try:
                        urls = await scraper.search(query, num_results=pages_per_query + 2, silent=True)
                        if urls:
                            urls = [
                                u for u in urls 
                                if self._get_domain(u) not in scraped_domains
                            ][:pages_per_query]
                            
                            if urls:
                                content = await scraper.crawl(
                                    urls, 
                                    max_pages=pages_per_query,
                                    show_progress=False
                                )
                                for c in content:
                                    scraped_domains.add(self._get_domain(c.url))
                                round_content.extend(content)
                    except Exception as e:
                        console.print(f"  [yellow]Warning: {e}[/yellow]")
                    
                    progress.advance(task)
                    await asyncio.sleep(0.3)
            
            # Fallback for first round if search APIs are limited
            if round_num == 1 and len(round_content) < 3:
                console.print("[yellow]Search APIs limited. Trying direct documentation URLs...[/yellow]")
                fallback_urls = self._get_fallback_urls(analysis.domain, analysis.role)
                if fallback_urls:
                    try:
                        fallback_content = await scraper.crawl(fallback_urls[:5], show_progress=True)
                        round_content.extend(fallback_content)
                    except Exception:
                        pass
            
            scraped_count += len(round_content)
            
            if not round_content:
                if round_num == 1:
                    raise ValueError("No content could be scraped. Check your search API keys.")
                break  # no new content available
            
            console.print(f"  [green]✓[/green] Scraped {len(round_content)} pages (total: {scraped_count})")
            
            # Step 3 (per round): Extract tests from new content
            for content in round_content:
                if output_type in ("benchmark", "all"):
                    qa_pairs = await processor.extract_qa_pairs(content, analysis.domain)
                    all_qa_pairs.extend(qa_pairs)
                if output_type in ("cloze", "all"):
                    masked = await processor.extract_masked_tests(content, analysis.domain)
                    all_masked.extend(masked)
            
            # Deduplicate after each round
            all_qa_pairs = self._deduplicate_qa(all_qa_pairs)
            all_masked = self._deduplicate_masked(all_masked)
        
        await scraper.close()
        
        # --- Hybrid mode: generate synthetic variants to fill remaining gap ---
        accepted = len(all_qa_pairs) + len(all_masked)
        if generation_mode == "hybrid" and accepted < target_count:
            console.print("\n[bold cyan]Step 3b: Generating synthetic variants (hybrid mode)...[/bold cyan]")
            gap = target_count - accepted
            synthetic_qa, synthetic_masked = self._generate_synthetic_variants(
                all_qa_pairs, all_masked, gap, output_type,
            )
            all_qa_pairs.extend(synthetic_qa)
            all_masked.extend(synthetic_masked)
            all_qa_pairs = self._deduplicate_qa(all_qa_pairs)
            all_masked = self._deduplicate_masked(all_masked)
            console.print(f"  [green]✓[/green] +{len(synthetic_qa)} Q&A variants, +{len(synthetic_masked)} cloze variants")
        
        # Trim to target per type
        all_qa_pairs = all_qa_pairs[:qa_target]
        all_masked = all_masked[:masked_target]
        
        total_tests = len(all_qa_pairs) + len(all_masked)
        console.print(f"\n[bold cyan]Step 3/4: Test generation summary[/bold cyan]")
        console.print(f"  [green]✓[/green] Generated {len(all_qa_pairs)} Q&A tests")
        console.print(f"  [green]✓[/green] Generated {len(all_masked)} cloze tests")
        console.print(f"  [green]✓[/green] Total: {total_tests} / {target_count} requested")
        
        if total_tests < target_count:
            console.print(f"  [yellow]⚠ Could only produce {total_tests} of {target_count} requested tests (limited source material)[/yellow]")
        
        # Step 4: Create YAML output
        console.print("\n[bold cyan]Step 4/4: Creating test file...[/bold cyan]")
        
        yaml_content = self._build_yaml(
            analysis.domain,
            all_qa_pairs,
            all_masked,
            bsp,
            analysis,
        )
        
        if output_dir is None:
            output_dir = Path.cwd() / "temp"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        domain_slug = analysis.domain.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"auto_generated_{domain_slug}_{timestamp}.yaml"
        
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        console.print(f"  [green]✓[/green] Saved to: {output_file}")
        
        end_time = asyncio.get_event_loop().time()
        
        return GeneratedTests(
            domain=analysis.domain,
            qa_pairs=all_qa_pairs,
            masked_tests=all_masked,
            yaml_content=yaml_content,
            output_file=output_file,
            generation_time=end_time - start_time,
            scraped_sources=scraped_count,
        )
    
    # ------------------------------------------------------------------
    # Hybrid mode: synthetic variant generation (no LLM call required)
    # ------------------------------------------------------------------
    
    _PARAPHRASE_PREFIXES = [
        "Explain", "Describe", "What is", "Can you clarify",
        "Summarize", "Tell me about", "Define", "Elaborate on",
    ]
    
    def _generate_synthetic_variants(
        self,
        qa_pairs: list[QAPair],
        masked_tests: list[MaskedTest],
        count: int,
        output_type: str,
    ) -> tuple[list[QAPair], list[MaskedTest]]:
        """Create lightweight variants from existing accepted tests.
        
        Strategies (all local, no API call):
        - Rephrase questions using alternate prefixes
        - Negate constraint tests ("What should you NOT do...")
        - Insert edge-case qualifiers ("in an unusual situation", "with missing data")
        """
        syn_qa: list[QAPair] = []
        syn_masked: list[MaskedTest] = []
        
        edge_qualifiers = [
            "in an unusual situation",
            "when data is missing",
            "under time pressure",
            "for a beginner",
            "in a formal context",
            "with conflicting requirements",
        ]
        
        if output_type in ("benchmark", "all") and qa_pairs:
            seeds = list(qa_pairs)
            random.shuffle(seeds)
            for seed in seeds:
                if len(syn_qa) >= count:
                    break
                # Strategy 1: prefix swap
                prefix = random.choice(self._PARAPHRASE_PREFIXES)
                # Strip leading question words from original
                q = re.sub(r'^(what|how|why|when|where|who|can you|explain|describe|define)\s+', '', seed.question, flags=re.IGNORECASE).strip()
                new_q = f"{prefix} {q}"
                if new_q.lower().strip() != seed.question.lower().strip():
                    syn_qa.append(QAPair(
                        question=new_q,
                        answer=seed.answer,
                        source_url=seed.source_url,
                        tags=(seed.tags or []) + ["synthetic"],
                    ))
                
                if len(syn_qa) >= count:
                    break
                
                # Strategy 2: edge-case qualifier
                qualifier = random.choice(edge_qualifiers)
                syn_qa.append(QAPair(
                    question=f"{seed.question.rstrip('?')} {qualifier}?",
                    answer=seed.answer,
                    source_url=seed.source_url,
                    tags=(seed.tags or []) + ["synthetic", "edge-case"],
                ))
        
        if output_type in ("cloze", "all") and masked_tests:
            seeds = list(masked_tests)
            random.shuffle(seeds)
            for seed in seeds:
                if len(syn_masked) >= count:
                    break
                # Swap mask position: if the sentence has multiple key terms, mask a different one
                words = seed.masked_text.replace("___", seed.answer).split()
                # Pick a different word to mask (min 4 chars, not the original answer)
                candidates = [w for w in words if len(w) >= 4 and w.lower() != seed.answer.lower()]
                if candidates:
                    new_mask_word = random.choice(candidates)
                    new_text = seed.masked_text.replace("___", seed.answer).replace(new_mask_word, "___", 1)
                    if "___" in new_text:
                        syn_masked.append(MaskedTest(
                            masked_text=new_text,
                            answer=new_mask_word,
                            original_text=seed.original_text,
                            mask_position=0,
                            source_url=seed.source_url,
                            tags=(seed.tags or []) + ["synthetic"],
                        ))
        
        return syn_qa[:count], syn_masked[:count]
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    def _get_fallback_urls(self, domain: str, role: str) -> list[str]:
        """Get fallback documentation URLs when search fails."""
        domain_lower = domain.lower().replace("_", " ")
        role_lower = role.lower() if role else ""
        
        # Known reliable documentation sites based on domain
        fallback_map = {
            "code": [
                "https://google.github.io/styleguide/",
                "https://docs.python.org/3/tutorial/",
                "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
                "https://wiki.c2.com/?CodeSmell",
            ],
            "code_review": [
                "https://google.github.io/eng-practices/review/reviewer/",
                "https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/",
                "https://wiki.c2.com/?CodeReviewPatterns",
                "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews",
            ],
            "programming": [
                "https://docs.python.org/3/faq/programming.html",
                "https://docs.python.org/3/tutorial/errors.html",
                "https://developer.mozilla.org/en-US/docs/Learn/JavaScript",
            ],
            "python": [
                "https://docs.python.org/3/faq/programming.html",
                "https://docs.python.org/3/tutorial/",
                "https://wiki.python.org/moin/BeginnersGuide/NonProgrammers",
                "https://realpython.com/python-best-practices/",
            ],
            "javascript": [
                "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
                "https://developer.mozilla.org/en-US/docs/Learn/JavaScript/First_steps",
                "https://javascript.info/first-steps",
            ],
            "support": [
                "https://www.zendesk.com/blog/customer-service-skills/",
                "https://help.zendesk.com/hc/en-us/articles/4408832479386-Best-practices-for-agents",
            ],
            "writing": [
                "https://developers.google.com/style",
                "https://docs.microsoft.com/en-us/style-guide/welcome/",
            ],
            "security": [
                "https://owasp.org/www-project-top-ten/",
                "https://cheatsheetseries.owasp.org/",
            ],
            "medical": [
                "https://www.mayoclinic.org/diseases-conditions",
                "https://www.cdc.gov/health-topics.html",
            ],
        }
        
        # Try exact domain match first
        if domain_lower.replace(" ", "_") in fallback_map:
            return fallback_map[domain_lower.replace(" ", "_")]
        
        # Try partial match
        for key, urls in fallback_map.items():
            if key in domain_lower or domain_lower in key:
                return urls
        
        # Default fallback - general programming/tech docs
        return [
            "https://docs.python.org/3/faq/programming.html",
            "https://developer.mozilla.org/en-US/docs/Learn",
            "https://wiki.c2.com/?DesignPatterns",
        ]
    
    def _deduplicate_qa(self, qa_pairs: list[QAPair]) -> list[QAPair]:
        """Remove duplicate Q&A pairs."""
        seen = set()
        unique = []
        for qa in qa_pairs:
            key = qa.question.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(qa)
        return unique
    
    def _deduplicate_masked(self, masked: list[MaskedTest]) -> list[MaskedTest]:
        """Remove duplicate masked tests."""
        seen = set()
        unique = []
        for m in masked:
            key = m.masked_text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(m)
        return unique
    
    def _build_yaml(
        self,
        domain: str,
        qa_pairs: list[QAPair],
        masked_tests: list[MaskedTest],
        bsp: str,
        analysis: BSPAnalysis,
    ) -> dict:
        """Build the combined YAML test file."""
        cases = []
        
        # Add Q&A test cases
        for i, qa in enumerate(qa_pairs, 1):
            case = {
                "id": f"{domain.lower().replace(' ', '-')}-qa-{i}",
                "prompt": qa.question,
                "assertions": [],
                "tags": qa.tags or [domain.lower()],
            }
            
            # Add smart assertions based on answer
            answer = qa.answer.strip()
            
            # If answer is short, use exact contains
            if len(answer) < 100:
                # Extract key terms for assertion (first 50 chars or first sentence)
                key_part = answer.split('.')[0][:50]
                case["assertions"].append({
                    "type": "contains",
                    "value": key_part,
                    "case_sensitive": False,
                })
            else:
                # For longer answers, use min_length assertion
                case["assertions"].append({
                    "type": "min_length",
                    "value": "50",
                })
            
            cases.append(case)
        
        # Add cloze (fill-in-blank) test cases
        for i, masked in enumerate(masked_tests, 1):
            case = {
                "id": f"{domain.lower().replace(' ', '-')}-cloze-{i}",
                "prompt": f"Fill in the blank with the correct word or term:\n\n{masked.masked_text}",
                "assertions": [
                    {
                        "type": "contains",
                        "value": masked.answer,
                        "case_sensitive": False,
                    }
                ],
                "tags": masked.tags or [domain.lower(), "cloze"],
            }
            cases.append(case)
        
        # Shuffle for variety
        random.shuffle(cases)
        
        return {
            "metadata": {
                "name": f"Auto-Generated Tests: {domain.title()}",
                "description": f"Automatically generated from BSP analysis. Domain: {domain}. Role: {analysis.role[:80]}",
                "generated_at": datetime.now().isoformat(),
                "source": "promptlab-auto-generator",
                "test_count": len(cases),
                "keywords": analysis.keywords[:5],
            },
            "defaults": {
                "temperature": 0,
            },
            "cases": cases,
        }


async def generate_tests_from_bsp(
    bsp: str,
    output_dir: Optional[Path] = None,
    target_count: int = 50,
    serpapi_key: Optional[str] = None,
    max_pages: int = 20,
    generation_mode: str = "web",
    project_root: Optional[Path] = None,
    llm_runner=None,
    llm_model: str = "ollama/llama3.1:8b",
) -> GeneratedTests:
    """Convenience function to generate tests from BSP.
    
    Args:
        bsp: The Behavior Specification Prompt
        output_dir: Directory to save generated tests
        target_count: Target number of test cases
        serpapi_key: SerpAPI key for better search results
        max_pages: Maximum pages to scrape
        generation_mode: "web", "docs_web", or "hybrid"
        project_root: Project root for corpus cache (docs_web/hybrid)
        llm_runner: LLMRunner instance for LLM-based generation
        llm_model: Model identifier for LLM calls
        
    Returns:
        GeneratedTests object
    """
    import os
    
    if serpapi_key is None:
        serpapi_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
    
    generator = AutoTestGenerator(
        serpapi_key=serpapi_key,
        max_pages=max_pages,
        project_root=project_root,
        llm_runner=llm_runner,
        llm_model=llm_model,
    )
    
    return await generator.generate_tests(
        bsp=bsp,
        target_count=target_count,
        output_dir=output_dir,
        generation_mode=generation_mode,
    )
