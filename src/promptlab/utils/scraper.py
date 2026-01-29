"""Web scraper for collecting domain-specific content."""

import asyncio
import json
import re
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
_temp_dir: Optional[Path] = None


def get_temp_dir(project_root: Optional[Path] = None) -> Path:
    """Get or create the temp directory for scraped content."""
    global _temp_dir
    if _temp_dir is None or not _temp_dir.exists():
        if project_root:
            _temp_dir = project_root / "temp"
        else:
            _temp_dir = Path.cwd() / "temp"
        _temp_dir.mkdir(exist_ok=True)
        console.print(f"[dim]Using temp directory: {_temp_dir}[/dim]")
    return _temp_dir


def cleanup_temp_dir() -> None:
    """Clean up all files in the temp directory."""
    global _temp_dir
    if _temp_dir and _temp_dir.exists():
        try:
            for f in _temp_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            console.print(f"[dim]Cleaned up temp directory: {_temp_dir}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not clean up temp dir: {e}[/yellow]")
        _temp_dir = None


@dataclass
class ScrapedContent:
    """Represents scraped content from a web page."""
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _temp_file: Optional[Path] = field(default=None, repr=False)
    
    def save_to_temp(self) -> Path:
        """Save content to temp file to reduce memory."""
        temp_dir = get_temp_dir()
        import hashlib
        url_hash = hashlib.md5(self.url.encode()).hexdigest()[:12]
        temp_file = temp_dir / f"content_{url_hash}.json"
        
        data = {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "links": self.links,
            "metadata": self.metadata,
        }
        
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        
        self._temp_file = temp_file
        # Clear the text from memory (keep only essential info)
        self._text_backup = self.text
        return temp_file
    
    def load_from_temp(self) -> str:
        """Load the full text content from temp file."""
        if self._temp_file and self._temp_file.exists():
            with open(self._temp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("text", "")
        return getattr(self, "_text_backup", self.text)
    
    @classmethod
    def from_temp_file(cls, temp_file: Path) -> "ScrapedContent":
        """Load a ScrapedContent instance from a temp file."""
        with open(temp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        content = cls(
            url=data["url"],
            title=data["title"],
            text=data["text"],
            links=data.get("links", []),
            metadata=data.get("metadata", {}),
        )
        content._temp_file = temp_file
        return content


@dataclass
class ScraperConfig:
    """Configuration for the web scraper."""
    max_pages: int = 10
    max_depth: int = 2
    delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    user_agent: str = "PromptLab/1.0 (Data Collection Bot)"
    respect_robots: bool = True
    allowed_domains: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=lambda: [
        r"\.pdf$", r"\.jpg$", r"\.png$", r"\.gif$", r"\.css$", r"\.js$",
        r"login", r"signup", r"register", r"logout", r"cart", r"checkout"
    ])


class WebScraper:
    """Async web scraper using httpx and BeautifulSoup."""
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._visited_urls: set[str] = set()
        self._scraped_content: list[ScrapedContent] = []
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers={"User-Agent": self.config.user_agent},
                follow_redirects=True,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be scraped."""
        parsed = urlparse(url)
        
        # Check protocol
        if parsed.scheme not in ("http", "https"):
            return False
        
        # Check domain restrictions
        if self.config.allowed_domains:
            if not any(domain in parsed.netloc for domain in self.config.allowed_domains):
                return False
        
        # Check exclusion patterns
        for pattern in self.config.exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def _normalize_url(self, url: str, base_url: str) -> str:
        """Normalize a URL relative to a base URL."""
        # Handle relative URLs
        if not url.startswith(("http://", "https://")):
            url = urljoin(base_url, url)
        
        # Remove fragments
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        return normalized
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from BeautifulSoup object."""
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", 
                            "meta", "link", "noscript", "iframe", "svg", "button"]):
            element.decompose()
        
        # Remove elements by class/id patterns (navigation, menus, etc.)
        for element in soup.find_all(class_=re.compile(r'nav|menu|sidebar|footer|header|breadcrumb', re.I)):
            element.decompose()
        for element in soup.find_all(id=re.compile(r'nav|menu|sidebar|footer|header', re.I)):
            element.decompose()
        
        # Try to find main content area first
        main_content = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'content|main|article', re.I))
        if main_content:
            soup = main_content
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            # Skip short lines (likely navigation) and lines with just symbols
            if len(line) > 15 and not re.match(r'^[\|\-\•\>\<\»\«\s]+$', line):
                lines.append(line)
        
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all valid links from a page."""
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            url = self._normalize_url(href, base_url)
            if self._is_valid_url(url) and url not in self._visited_urls:
                links.append(url)
        return list(set(links))  # Deduplicate
    
    async def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent or None if scraping failed
        """
        if url in self._visited_urls:
            return None
        
        if not self._is_valid_url(url):
            return None
        
        self._visited_urls.add(url)
        
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Extract text content
            text = self._extract_text(soup)
            
            # Extract links for crawling
            links = self._extract_links(soup, url)
            
            content = ScrapedContent(
                url=url,
                title=title,
                text=text,
                links=links,
                metadata={"content_length": len(text)},
            )
            
            self._scraped_content.append(content)
            return content
            
        except httpx.HTTPError as e:
            console.print(f"[yellow]HTTP error scraping {url}: {e}[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error scraping {url}: {e}[/red]")
            return None
    
    async def crawl(
        self, 
        start_urls: list[str], 
        max_pages: Optional[int] = None,
        show_progress: bool = True
    ) -> list[ScrapedContent]:
        """Crawl starting from given URLs.
        
        Args:
            start_urls: List of URLs to start crawling from
            max_pages: Maximum pages to scrape (uses config default if not specified)
            show_progress: Whether to show progress indicator
            
        Returns:
            List of scraped content
        """
        max_pages = max_pages or self.config.max_pages
        to_visit = list(start_urls)
        depth_map = {url: 0 for url in start_urls}
        
        self._visited_urls.clear()
        self._scraped_content.clear()
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Scraping...", total=None)
                
                while to_visit and len(self._scraped_content) < max_pages:
                    url = to_visit.pop(0)
                    current_depth = depth_map.get(url, 0)
                    
                    if current_depth > self.config.max_depth:
                        continue
                    
                    progress.update(
                        task, 
                        description=f"Scraping ({len(self._scraped_content)}/{max_pages}): {url[:50]}..."
                    )
                    
                    content = await self.scrape_url(url)
                    
                    if content and current_depth < self.config.max_depth:
                        for link in content.links[:10]:  # Limit links per page
                            if link not in depth_map:
                                to_visit.append(link)
                                depth_map[link] = current_depth + 1
                    
                    # Rate limiting
                    await asyncio.sleep(self.config.delay_seconds)
        else:
            while to_visit and len(self._scraped_content) < max_pages:
                url = to_visit.pop(0)
                current_depth = depth_map.get(url, 0)
                
                if current_depth > self.config.max_depth:
                    continue
                
                content = await self.scrape_url(url)
                
                if content and current_depth < self.config.max_depth:
                    for link in content.links[:10]:
                        if link not in depth_map:
                            to_visit.append(link)
                            depth_map[link] = current_depth + 1
                
                await asyncio.sleep(self.config.delay_seconds)
        
        await self.close()
        return self._scraped_content
    
    async def scrape_search_results(
        self, 
        query: str, 
        num_results: int = 10
    ) -> list[ScrapedContent]:
        """Scrape content based on a search query.
        
        Uses DuckDuckGo HTML search (no API key required).
        
        Args:
            query: Search query
            num_results: Number of results to scrape
            
        Returns:
            List of scraped content
        """
        # DuckDuckGo HTML search
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        try:
            client = await self._get_client()
            response = await client.get(search_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract result URLs - try multiple selectors
            result_urls = []
            
            # Try different DuckDuckGo selectors
            selectors = [
                ".result__url",           # Standard DDG
                ".result__a",             # Link elements
                "a.result__url",          # Direct link
                ".results .result a",     # Generic result links
            ]
            
            for selector in selectors:
                for element in soup.select(selector):
                    # Get href or text content
                    url = element.get("href") or element.get_text(strip=True)
                    if url:
                        # Handle DDG redirect URLs
                        if "uddg=" in url:
                            import urllib.parse
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                            if "uddg" in parsed:
                                url = parsed["uddg"][0]
                        
                        if not url.startswith("http"):
                            url = "https://" + url
                        
                        if self._is_valid_url(url) and url not in result_urls:
                            result_urls.append(url)
                
                if result_urls:
                    break
            
            # Limit results
            result_urls = result_urls[:num_results]
            
            console.print(f"[cyan]Found {len(result_urls)} search results for '{query}'[/cyan]")
            
            # Scrape each result
            return await self.crawl(result_urls, max_pages=num_results)
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return []


# Convenience function for simple scraping
async def scrape_urls(urls: list[str], max_pages: int = 10) -> list[ScrapedContent]:
    """Simple function to scrape a list of URLs.
    
    Args:
        urls: List of URLs to scrape
        max_pages: Maximum pages to scrape
        
    Returns:
        List of scraped content
    """
    scraper = WebScraper(ScraperConfig(max_pages=max_pages))
    return await scraper.crawl(urls)


async def scrape_for_domain(
    domain: str, 
    num_pages: int = 10,
    search_queries: Optional[list[str]] = None
) -> list[ScrapedContent]:
    """Scrape content for a specific domain/topic.
    
    Args:
        domain: Domain or topic (e.g., "legal contracts", "medical diagnosis")
        num_pages: Number of pages to scrape
        search_queries: Optional list of search queries (auto-generated if not provided)
        
    Returns:
        List of scraped content
    """
    # Domain-specific seed URLs for common topics
    DOMAIN_SEEDS = {
        "python": [
            "https://docs.python.org/3/faq/general.html",
            "https://docs.python.org/3/faq/programming.html",
            "https://wiki.python.org/moin/BeginnersGuide/Programmers",
        ],
        "javascript": [
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
            "https://javascript.info/",
        ],
        "machine learning": [
            "https://developers.google.com/machine-learning/crash-course/ml-intro",
            "https://www.ibm.com/topics/machine-learning",
            "https://scikit-learn.org/stable/getting_started.html",
        ],
        "ai": [
            "https://www.ibm.com/topics/artificial-intelligence",
            "https://developers.google.com/machine-learning/crash-course",
        ],
        "legal": [
            "https://www.law.cornell.edu/wex",
            "https://www.nolo.com/legal-encyclopedia",
        ],
        "medical": [
            "https://medlineplus.gov/healthtopics.html",
            "https://www.mayoclinic.org/diseases-conditions",
        ],
        "finance": [
            "https://www.investopedia.com/terms/",
        ],
        "cooking": [
            "https://www.allrecipes.com/recipes/",
            "https://www.simplyrecipes.com/",
        ],
        "react": [
            "https://react.dev/learn",
            "https://react.dev/reference/react",
        ],
        "typescript": [
            "https://www.typescriptlang.org/docs/handbook/2/basic-types.html",
            "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
        ],
    }
    
    # Find matching seed URLs
    seed_urls = []
    domain_lower = domain.lower()
    for key, urls in DOMAIN_SEEDS.items():
        if key in domain_lower or domain_lower in key:
            seed_urls.extend(urls)
    
    scraper = WebScraper(ScraperConfig(max_pages=num_pages))
    all_content: list[ScrapedContent] = []
    
    # First try seed URLs if available
    if seed_urls:
        console.print(f"[cyan]Using {len(seed_urls)} seed URLs for '{domain}'[/cyan]")
        content = await scraper.crawl(seed_urls[:3], max_pages=min(5, num_pages))
        all_content.extend(content)
    
    # Then try search if we need more
    if len(all_content) < num_pages:
        if search_queries is None:
            search_queries = [
                f"{domain} FAQ",
                f"{domain} Q&A",
                f"{domain} questions and answers",
                f"{domain} tutorial",
                f"{domain} guide",
            ]
        
        for query in search_queries:
            if len(all_content) >= num_pages:
                break
            remaining = num_pages - len(all_content)
            content = await scraper.scrape_search_results(query, num_results=remaining)
            all_content.extend(content)
    
    return all_content
