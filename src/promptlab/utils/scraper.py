"""Modern web scraper with hybrid httpx + Playwright approach.

Tech stack:
- httpx (async) for fast HTTP requests
- Playwright (async) with stealth for JS-rendered/bot-protected pages
- selectolax + lxml for fast HTML parsing
- SerpAPI (Google results, 100 free/month, no CC required)
- Brave Search API (requires credit card even for free tier)
- SearXNG (completely free, no auth needed)
- DuckDuckGo/Google scrape as fallbacks
"""

import asyncio
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, quote

import httpx
from selectolax.parser import HTMLParser
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
_temp_dir: Optional[Path] = None

# Browser instance (lazy loaded)
_browser = None
_playwright = None


def get_temp_dir(project_root: Optional[Path] = None) -> Path:
    """Get or create the temp directory for scraped content."""
    global _temp_dir
    if _temp_dir is None or not _temp_dir.exists():
        if project_root:
            _temp_dir = project_root / "temp"
        else:
            _temp_dir = Path.cwd() / "temp"
        _temp_dir.mkdir(exist_ok=True)
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
    delay_seconds: float = 0.5
    timeout_seconds: float = 30.0
    # Realistic browser User-Agent
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    allowed_domains: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=lambda: [
        r"\.pdf$", r"\.jpg$", r"\.png$", r"\.gif$", r"\.css$", r"\.js$",
        r"login", r"signup", r"register", r"logout", r"cart", r"checkout"
    ])
    # SerpAPI key (RECOMMENDED - 100 free searches/month, no credit card required)
    # Get free key at: https://serpapi.com/
    serpapi_key: Optional[str] = None
    # Brave Search API key (optional - requires credit card even for free tier)
    brave_api_key: Optional[str] = None
    # Use browser for all requests (slower but more reliable)
    force_browser: bool = False


class WebScraper:
    """Async web scraper with hybrid httpx + Playwright approach."""
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._visited_urls: set[str] = set()
        self._scraped_content: list[ScrapedContent] = []
        self._failed_with_httpx: set[str] = set()  # URLs that need browser
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with browser-like headers."""
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=headers,
                follow_redirects=True,
                http2=True,
            )
        return self._client
    
    async def _get_browser(self):
        """Get or create Playwright browser with stealth."""
        global _browser, _playwright
        
        if _browser is None:
            try:
                from playwright.async_api import async_playwright
                
                _playwright = await async_playwright().start()
                _browser = await _playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                    ]
                )
                console.print("[dim]Browser initialized for JS rendering[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not initialize browser: {e}[/yellow]")
                return None
        
        return _browser
    
    async def _scrape_with_browser(self, url: str) -> Optional[tuple[str, str]]:
        """Scrape a URL using Playwright browser (for JS-rendered pages)."""
        browser = await self._get_browser()
        if not browser:
            return None
        
        try:
            from playwright_stealth import Stealth
            
            context = await browser.new_context(
                user_agent=self.config.user_agent,
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()
            
            # Apply stealth to avoid detection
            stealth = Stealth()
            await stealth.apply_stealth_async(page)
            
            await page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_seconds * 1000)
            await page.wait_for_timeout(1000)  # Wait for JS to render
            
            html = await page.content()
            await context.close()
            
            return html, "browser"
            
        except Exception as e:
            console.print(f"[yellow]Browser scrape failed for {url}: {e}[/yellow]")
            return None
    
    async def close(self):
        """Close HTTP client and browser."""
        global _browser, _playwright
        
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        
        if _browser:
            await _browser.close()
            _browser = None
        if _playwright:
            await _playwright.stop()
            _playwright = None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be scraped."""
        parsed = urlparse(url)
        
        if parsed.scheme not in ("http", "https"):
            return False
        
        if self.config.allowed_domains:
            if not any(domain in parsed.netloc for domain in self.config.allowed_domains):
                return False
        
        for pattern in self.config.exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def _normalize_url(self, url: str, base_url: str) -> str:
        """Normalize a URL relative to a base URL."""
        if not url.startswith(("http://", "https://")):
            url = urljoin(base_url, url)
        
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        return normalized
    
    def _extract_text_selectolax(self, html: str) -> str:
        """Extract clean text using selectolax (fast)."""
        parser = HTMLParser(html)
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside', 
                    'meta', 'link', 'noscript', 'iframe', 'svg', 'button', 'form']:
            for node in parser.css(tag):
                node.decompose()
        
        # Remove by class/id patterns
        for selector in ['.nav', '.menu', '.sidebar', '.footer', '.header', 
                        '.breadcrumb', '.advertisement', '.ad', '.cookie',
                        '#nav', '#menu', '#sidebar', '#footer', '#header']:
            for node in parser.css(selector):
                node.decompose()
        
        # Try to find main content
        main = parser.css_first('main, article, .content, .main, #content, #main')
        if main:
            text = main.text(separator="\n", strip=True)
        else:
            body = parser.css_first('body')
            text = body.text(separator="\n", strip=True) if body else ""
        
        # Clean up
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 20 and not re.match(r'^[\|\-\•\>\<\»\«\s]+$', line):
                lines.append(line)
        
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text
    
    def _extract_title_selectolax(self, html: str) -> str:
        """Extract title using selectolax."""
        parser = HTMLParser(html)
        title_node = parser.css_first('title')
        if title_node:
            return title_node.text(strip=True)
        
        h1_node = parser.css_first('h1')
        if h1_node:
            return h1_node.text(strip=True)
        
        return ""
    
    def _extract_links_selectolax(self, html: str, base_url: str) -> list[str]:
        """Extract all valid links using selectolax."""
        parser = HTMLParser(html)
        links = []
        
        for node in parser.css('a[href]'):
            href = node.attributes.get('href', '')
            if href:
                url = self._normalize_url(href, base_url)
                if self._is_valid_url(url) and url not in self._visited_urls:
                    links.append(url)
        
        return list(set(links))
    
    async def scrape_url(self, url: str, use_browser: bool = False) -> Optional[ScrapedContent]:
        """Scrape a single URL with hybrid approach.
        
        Args:
            url: URL to scrape
            use_browser: Force browser usage for this URL
            
        Returns:
            ScrapedContent or None if scraping failed
        """
        if url in self._visited_urls:
            return None
        
        if not self._is_valid_url(url):
            return None
        
        self._visited_urls.add(url)
        html = None
        method = "httpx"
        
        # Try httpx first (fast)
        if not use_browser and not self.config.force_browser:
            try:
                client = await self._get_client()
                response = await client.get(url)
                
                # Check for bot detection / JS requirement
                if response.status_code == 403 or response.status_code == 429:
                    self._failed_with_httpx.add(url)
                    use_browser = True
                elif response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type.lower():
                        return None
                    html = response.text
                    
                    # Check if page requires JS (common patterns)
                    if self._needs_javascript(html):
                        use_browser = True
                        html = None
                else:
                    response.raise_for_status()
                    
            except httpx.HTTPError as e:
                # Fallback to browser on HTTP errors
                use_browser = True
        
        # Browser fallback (for JS-rendered or blocked pages)
        if (use_browser or self.config.force_browser) and html is None:
            result = await self._scrape_with_browser(url)
            if result:
                html, method = result
        
        if not html:
            return None
        
        # Parse with selectolax (fast)
        title = self._extract_title_selectolax(html)
        text = self._extract_text_selectolax(html)
        links = self._extract_links_selectolax(html, url)
        
        if not text or len(text) < 100:
            return None
        
        content = ScrapedContent(
            url=url,
            title=title,
            text=text,
            links=links,
            metadata={"content_length": len(text), "method": method},
        )
        
        self._scraped_content.append(content)
        return content
    
    def _needs_javascript(self, html: str) -> bool:
        """Check if page likely needs JavaScript to render content."""
        indicators = [
            'id="__next"',  # Next.js
            'id="app"',  # Vue/React apps
            'id="root"',  # React
            'Please enable JavaScript',
            'This site requires JavaScript',
            'noscript',
            'loading...',
            'cloudflare',
        ]
        html_lower = html.lower()
        
        # Check for empty body or minimal content
        parser = HTMLParser(html)
        body = parser.css_first('body')
        if body:
            text = body.text(strip=True)
            if len(text) < 200:
                return True
        
        return any(ind.lower() in html_lower for ind in indicators)
    
    async def crawl(
        self, 
        start_urls: list[str], 
        max_pages: Optional[int] = None,
        show_progress: bool = True
    ) -> list[ScrapedContent]:
        """Crawl starting from given URLs."""
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
                        for link in content.links[:10]:
                            if link not in depth_map:
                                to_visit.append(link)
                                depth_map[link] = current_depth + 1
                    
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
    
    async def search_serpapi(self, query: str, num_results: int = 10) -> list[str]:
        """Search using SerpAPI (Google results, 100 free/month, no CC required).
        
        Get your free API key at: https://serpapi.com/
        """
        api_key = self.config.serpapi_key
        if not api_key:
            # Try environment variable
            import os
            api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
        
        if not api_key:
            return []
        
        try:
            client = await self._get_client()
            response = await client.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "api_key": api_key,
                    "num": num_results,
                    "engine": "google",
                }
            )
            response.raise_for_status()
            
            data = response.json()
            urls = []
            for result in data.get("organic_results", []):
                url = result.get("link")
                if url and self._is_valid_url(url):
                    urls.append(url)
            
            return urls[:num_results]
            
        except Exception as e:
            console.print(f"[yellow]SerpAPI search error: {e}[/yellow]")
            return []
    
    async def search_searxng(self, query: str, num_results: int = 10) -> list[str]:
        """Search using SearXNG public instances (completely free, no auth)."""
        searxng_instances = [
            "https://searx.be",
            "https://search.sapti.me",
            "https://searx.tiekoetter.com",
            "https://searx.ninja",
        ]
        
        client = await self._get_client()
        
        for instance in searxng_instances:
            try:
                response = await client.get(
                    f"{instance}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "categories": "general",
                        "language": "en",
                    },
                    headers={"User-Agent": "PromptLab/1.0"},
                    timeout=15.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    urls = []
                    for result in data.get("results", []):
                        url = result.get("url")
                        if url and self._is_valid_url(url):
                            urls.append(url)
                    
                    if urls:
                        return urls[:num_results]
            
            except Exception:
                continue  # Try next instance
        
        return []
    
    async def search_brave(self, query: str, num_results: int = 10) -> list[str]:
        """Search using Brave Search API (requires credit card for signup)."""
        api_key = self.config.brave_api_key
        if not api_key:
            # Try environment variable
            import os
            api_key = os.environ.get("BRAVE_API_KEY")
        
        if not api_key:
            return []
        
        try:
            client = await self._get_client()
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={"X-Subscription-Token": api_key}
            )
            response.raise_for_status()
            
            data = response.json()
            urls = []
            for result in data.get("web", {}).get("results", []):
                url = result.get("url")
                if url and self._is_valid_url(url):
                    urls.append(url)
            
            return urls[:num_results]
            
        except Exception as e:
            console.print(f"[yellow]Brave search error: {e}[/yellow]")
            return []
    
    async def search_duckduckgo(self, query: str, num_results: int = 10) -> list[str]:
        """Fallback search using DuckDuckGo library (free, no auth)."""
        try:
            from duckduckgo_search import DDGS
            import warnings
            warnings.filterwarnings("ignore")
            
            blocked_domains = {
                'weforum.org', 'medium.com', 'linkedin.com', 'facebook.com', 
                'twitter.com', 'instagram.com', 'reddit.com', 'zhihu.com',
                'quora.com', 'pinterest.com', 'youtube.com', 'tiktok.com',
                'x.com', 'threads.net', 'tumblr.com', 'snapchat.com',
            }
            
            # Block non-English TLDs
            blocked_tlds = {'.jp', '.cn', '.kr', '.ru', '.de', '.fr', '.es', '.it', '.br', '.nl'}
            
            urls = []
            
            # Run synchronous DDG in executor to avoid blocking
            def _search():
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, region='us-en', max_results=num_results * 4))
                        return results
                except Exception:
                    return []
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _search)
            
            for r in results:
                url = r.get("href") or r.get("link")
                if not url:
                    continue
                if any(d in url for d in blocked_domains):
                    continue
                # Skip non-English TLDs
                if any(url.lower().endswith(tld) or tld + '/' in url.lower() for tld in blocked_tlds):
                    continue
                if self._is_valid_url(url) and url not in urls:
                    urls.append(url)
            
            return urls[:num_results]
            
        except ImportError:
            # duckduckgo-search not installed
            return []
        except Exception as e:
            # Rate limited or other error - fail silently
            return []
    
    async def search_google_scrape(self, query: str, num_results: int = 10) -> list[str]:
        """Scrape Google search results using browser (fallback)."""
        browser = await self._get_browser()
        if not browser:
            return []
        
        try:
            from playwright_stealth import Stealth
            
            context = await browser.new_context(
                user_agent=self.config.user_agent,
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()
            
            stealth = Stealth()
            await stealth.apply_stealth_async(page)
            
            search_url = f"https://www.google.com/search?q={quote(query)}&num={num_results}"
            await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(2000)
            
            # Extract result URLs
            urls = []
            links = await page.query_selector_all('a[href]')
            for link in links:
                href = await link.get_attribute('href')
                if href and href.startswith('http') and 'google.com' not in href:
                    if self._is_valid_url(href) and href not in urls:
                        urls.append(href)
            
            await context.close()
            return urls[:num_results]
            
        except Exception as e:
            console.print(f"[dim]Google scrape failed: {e}[/dim]")
            return []
    
    async def search(self, query: str, num_results: int = 10, silent: bool = False) -> list[str]:
        """Search for URLs using available search providers.
        
        Priority (optimized for reliability):
        1. SerpAPI (Google results, 100 free/month, no CC required) ⭐ BEST
        2. DuckDuckGo library (free, usually works well)
        3. SearXNG (free, no auth, multiple instances)
        4. Brave Search API (good quality, requires CC for signup)
        5. Google scrape via Playwright (last resort, often blocked)
        
        Args:
            query: Search query
            num_results: Number of results to return
            silent: If True, don't print status messages
        """
        def log(msg: str):
            if not silent:
                console.print(msg)
        
        # Try SerpAPI first (best quality, free tier without CC)
        urls = await self.search_serpapi(query, num_results)
        if urls:
            log(f"[dim]Found {len(urls)} results via SerpAPI[/dim]")
            return urls
        
        # Try DuckDuckGo (moved up - more reliable than SearXNG)
        urls = await self.search_duckduckgo(query, num_results)
        if urls:
            log(f"[dim]Found {len(urls)} results via DuckDuckGo[/dim]")
            return urls
        
        # Try SearXNG (completely free, no auth)
        urls = await self.search_searxng(query, num_results)
        if urls:
            log(f"[dim]Found {len(urls)} results via SearXNG[/dim]")
            return urls
        
        # Try Brave Search (requires CC for signup)
        urls = await self.search_brave(query, num_results)
        if urls:
            log(f"[dim]Found {len(urls)} results via Brave Search[/dim]")
            return urls
        
        # Last resort: Google scrape via Playwright
        urls = await self.search_google_scrape(query, num_results)
        if urls:
            log(f"[dim]Found {len(urls)} results via Google scrape[/dim]")
            return urls
        
        if not silent:
            console.print(f"[yellow]No search results for '{query}'[/yellow]")
        return []
    
    async def scrape_search_results(
        self, 
        query: str, 
        num_results: int = 10
    ) -> list[ScrapedContent]:
        """Search and scrape results."""
        urls = await self.search(query, num_results)
        if not urls:
            return []
        
        return await self.crawl(urls, max_pages=num_results)


# Convenience functions
async def scrape_urls(urls: list[str], max_pages: int = 10) -> list[ScrapedContent]:
    """Simple function to scrape a list of URLs."""
    scraper = WebScraper(ScraperConfig(max_pages=max_pages))
    return await scraper.crawl(urls)


async def scrape_for_domain(
    domain: str, 
    num_pages: int = 10,
    search_queries: Optional[list[str]] = None
) -> list[ScrapedContent]:
    """Scrape content for a specific domain/topic using dynamic URL discovery.
    
    This is fully dynamic - no static/hardcoded URLs. Uses search engines
    to discover relevant content for any topic.
    
    Args:
        domain: Domain or topic (e.g., "quantum computing", "blockchain")
        num_pages: Number of pages to scrape
        search_queries: Optional list of search queries (auto-generated if not provided)
        
    Returns:
        List of scraped content
    """
    scraper = WebScraper(ScraperConfig(max_pages=num_pages))
    all_content: list[ScrapedContent] = []
    
    # Generate smart search queries for better coverage
    if search_queries is None:
        search_queries = [
            f"what is {domain} explained",
            f"{domain} tutorial beginner guide",
            f"{domain} FAQ frequently asked questions",
            f"{domain} documentation official",
            f"learn {domain} basics",
        ]
    
    console.print(f"[cyan]Searching for '{domain}' content dynamically...[/cyan]")
    
    # Search and scrape
    for query in search_queries:
        if len(all_content) >= num_pages:
            break
        
        remaining = num_pages - len(all_content)
        urls = await scraper.search(query, num_results=remaining + 3)
        
        if urls:
            # Filter out already scraped domains to get variety
            scraped_domains = {urlparse(c.url).netloc for c in all_content}
            urls = [u for u in urls if urlparse(u).netloc not in scraped_domains][:remaining]
            
            if urls:
                content = await scraper.crawl(urls, max_pages=remaining)
                all_content.extend(content)
                
                if content:
                    console.print(f"[green]✓[/green] Got {len(content)} pages from '{query}'")
        
        # Brief pause between searches
        await asyncio.sleep(0.5)
    
    return all_content
