"""Automatic Test Generator - Creates tests from BSP via web scraping.

This module analyzes the Behavior Specification Prompt (BSP) to:
1. Extract domain keywords and context
2. Generate smart search queries
3. Scrape relevant content from the web
4. Auto-generate 50-100 test cases

The goal is to make manual test writing unnecessary - just provide a BSP
and tests are generated automatically via intelligent web scraping.
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

from promptlab.utils.scraper import (
    WebScraper, 
    ScraperConfig, 
    ScrapedContent,
    scrape_for_domain,
)
from promptlab.utils.data_processor import (
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
        search_queries = self._generate_search_queries(domain, role, keywords, capabilities)
        
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
        capabilities: list[str]
    ) -> list[str]:
        """Generate smart search queries based on BSP analysis."""
        queries = []
        
        # Clean domain for search (replace underscores with spaces)
        domain_clean = domain.replace("_", " ")
        
        # Domain-specific queries - simpler and more effective
        queries.append(f"{domain_clean} best practices")
        queries.append(f"{domain_clean} checklist")
        queries.append(f"{domain_clean} common mistakes")
        queries.append(f"{domain_clean} tips and tricks")
        queries.append(f"how to do {domain_clean}")
        queries.append(f"{domain_clean} guide")
        queries.append(f"{domain_clean} examples")
        
        # Role-specific queries
        if role and role != "AI Assistant":
            role_clean = role.replace("a ", "").replace("an ", "")[:50]
            queries.append(f"{role_clean} guide")
            queries.append(f"{role_clean} tips")
            queries.append(f"what does a {role_clean} do")
        
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
        
        return unique[:12]  # Limit to 12 queries
    
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
    """Generates tests automatically from BSP using web scraping."""
    
    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        max_pages: int = 20,
    ):
        """Initialize the auto test generator.
        
        Args:
            serpapi_key: SerpAPI key for Google search (recommended)
            brave_api_key: Brave Search API key (optional)
            max_pages: Maximum pages to scrape
        """
        self.serpapi_key = serpapi_key
        self.brave_api_key = brave_api_key
        self.max_pages = max_pages
        self.analyzer = BSPAnalyzer()
    
    async def generate_tests(
        self,
        bsp: str,
        target_count: int = 50,
        output_dir: Optional[Path] = None,
        output_type: Literal["benchmark", "cloze", "all"] = "all",
    ) -> GeneratedTests:
        """Generate tests automatically from BSP.
        
        Args:
            bsp: The Behavior Specification Prompt
            target_count: Target number of test cases to generate
            output_dir: Directory to save generated tests
            output_type: Type of tests to generate
            
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
        
        # Step 2: Scrape content
        console.print("\n[bold cyan]Step 2/4: Scraping relevant web content...[/bold cyan]")
        
        scraper_config = ScraperConfig(
            max_pages=self.max_pages,
            serpapi_key=self.serpapi_key,
            brave_api_key=self.brave_api_key,
        )
        scraper = WebScraper(scraper_config)
        
        all_content: list[ScrapedContent] = []
        pages_per_query = max(2, self.max_pages // len(analysis.search_queries))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Searching & scraping...", 
                total=len(analysis.search_queries)
            )
            
            for query in analysis.search_queries:
                if len(all_content) >= self.max_pages:
                    break
                
                progress.update(task, description=f"Searching: {query[:40]}...")
                
                try:
                    urls = await scraper.search(query, num_results=pages_per_query + 2, silent=True)
                    
                    if urls:
                        # Filter out already scraped domains for variety
                        scraped_domains = {
                            self._get_domain(c.url) for c in all_content
                        }
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
                            all_content.extend(content)
                            
                except Exception as e:
                    console.print(f"  [yellow]Warning: {e}[/yellow]")
                
                progress.advance(task)
                await asyncio.sleep(0.3)  # Brief pause between queries
        
        # If no search results, try direct URLs for common documentation sites
        if len(all_content) < 3:
            console.print("[yellow]Search APIs limited. Trying direct documentation URLs...[/yellow]")
            fallback_urls = self._get_fallback_urls(analysis.domain, analysis.role)
            if fallback_urls:
                try:
                    fallback_content = await scraper.crawl(fallback_urls[:5], show_progress=True)
                    all_content.extend(fallback_content)
                except Exception:
                    pass
        
        await scraper.close()
        
        if not all_content:
            raise ValueError("No content could be scraped. Check your search API keys.")
        
        console.print(f"  [green]✓[/green] Scraped {len(all_content)} pages")
        
        # Step 3: Generate Q&A pairs
        console.print("\n[bold cyan]Step 3/4: Generating test cases...[/bold cyan]")
        
        processor = DataProcessor()
        all_qa_pairs: list[QAPair] = []
        all_masked: list[MaskedTest] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing content...", total=len(all_content))
            
            for content in all_content:
                progress.update(
                    task, 
                    description=f"Processing: {content.title[:40]}..."
                )
                
                if output_type in ("benchmark", "all"):
                    qa_pairs = await processor.extract_qa_pairs(content, analysis.domain)
                    all_qa_pairs.extend(qa_pairs)
                
                if output_type in ("cloze", "all"):
                    masked = await processor.extract_masked_tests(content, analysis.domain)
                    all_masked.extend(masked)
                
                progress.advance(task)
        
        # Deduplicate and limit
        all_qa_pairs = self._deduplicate_qa(all_qa_pairs)
        all_masked = self._deduplicate_masked(all_masked)
        
        # Calculate how many of each type we need
        if output_type == "all":
            qa_target = int(target_count * 0.6)  # 60% Q&A
            masked_target = int(target_count * 0.4)  # 40% cloze
        elif output_type == "benchmark":
            qa_target = target_count
            masked_target = 0
        else:
            qa_target = 0
            masked_target = target_count
        
        all_qa_pairs = all_qa_pairs[:qa_target]
        all_masked = all_masked[:masked_target]
        
        total_tests = len(all_qa_pairs) + len(all_masked)
        console.print(f"  [green]✓[/green] Generated {len(all_qa_pairs)} Q&A tests")
        console.print(f"  [green]✓[/green] Generated {len(all_masked)} cloze tests")
        console.print(f"  [green]✓[/green] Total: {total_tests} test cases")
        
        # Step 4: Create YAML output
        console.print("\n[bold cyan]Step 4/4: Creating test file...[/bold cyan]")
        
        # Build combined YAML
        yaml_content = self._build_yaml(
            analysis.domain,
            all_qa_pairs,
            all_masked,
            bsp,
            analysis,
        )
        
        # Determine output path
        if output_dir is None:
            output_dir = Path.cwd() / "temp"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on domain and timestamp
        domain_slug = analysis.domain.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"auto_generated_{domain_slug}_{timestamp}.yaml"
        
        # Save YAML
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        console.print(f"  [green]✓[/green] Saved to: {output_file}")
        
        end_time = asyncio.get_event_loop().time()
        generation_time = end_time - start_time
        
        return GeneratedTests(
            domain=analysis.domain,
            qa_pairs=all_qa_pairs,
            masked_tests=all_masked,
            yaml_content=yaml_content,
            output_file=output_file,
            generation_time=generation_time,
            scraped_sources=len(all_content),
        )
    
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
) -> GeneratedTests:
    """Convenience function to generate tests from BSP.
    
    Args:
        bsp: The Behavior Specification Prompt
        output_dir: Directory to save generated tests
        target_count: Target number of test cases
        serpapi_key: SerpAPI key for better search results
        max_pages: Maximum pages to scrape
        
    Returns:
        GeneratedTests object
    """
    import os
    
    # Try to get API key from environment if not provided
    if serpapi_key is None:
        serpapi_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
    
    generator = AutoTestGenerator(
        serpapi_key=serpapi_key,
        max_pages=max_pages,
    )
    
    return await generator.generate_tests(
        bsp=bsp,
        target_count=target_count,
        output_dir=output_dir,
    )
