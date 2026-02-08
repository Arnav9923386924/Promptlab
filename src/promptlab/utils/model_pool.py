"""Dynamic Model Pool — auto-discovers free models from multiple providers.

Fetches available free models from OpenRouter API and Google AI Studio,
ranks by parameter count, and provides automatic fallback when models get
rate-limited. Only used for council judges — the chairman stays static.
"""

import re
import time
import httpx
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from rich.console import Console

console = Console()


# How long to consider a model "rate limited" before retrying (seconds)
RATE_LIMIT_COOLDOWN = 90  # 1.5 minutes

# Cache lifetime for the model list (seconds)
MODEL_CACHE_TTL = 600  # 10 minutes

# Minimum context length to be considered a viable judge
MIN_CONTEXT_LENGTH = 4000

# Regex patterns to extract parameter count from model names/IDs
# Matches: 70b, 4b, 235b, 27b, 8b, 1.5b, 0.5b, etc.
_PARAM_PATTERNS = [
    re.compile(r'(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_:])'),  # "70b", "4b", "1.5b"
    re.compile(r'(\d+(?:\.\d+)?)\s*[bB]$'),               # ends with "70b"
]


def _extract_param_count(model_id: str) -> float:
    """Extract parameter count in billions from a model ID or name.
    
    Examples:
        'meta-llama/llama-3.3-70b-instruct:free' → 70.0
        'google/gemma-3-27b-it:free' → 27.0
        'qwen/qwen3-4b:free' → 4.0
        'qwen/qwen3-235b-a22b:free' → 235.0
        'some-model-without-params' → 0.0
    """
    text = model_id.lower()
    
    # Find all parameter-like matches, take the largest
    # (handles cases like "qwen3-235b-a22b" where 235b is the real param count)
    candidates = []
    for pattern in _PARAM_PATTERNS:
        for match in pattern.finditer(text):
            try:
                candidates.append(float(match.group(1)))
            except ValueError:
                continue
    
    if candidates:
        return max(candidates)  # Largest match is likely the real param count
    return 0.0


@dataclass
class FreeModel:
    """A free model discovered from a provider."""
    id: str                    # e.g. "google/gemma-3-27b-it:free" or "gemini-2.5-flash"
    name: str                  # e.g. "Gemini 2.5 Flash"
    context_length: int        # e.g. 1048576
    param_count: float = 0.0   # billions of parameters (extracted from name)
    provider: str = ""         # "openrouter" or "google"
    full_id: str = ""          # with provider prefix for runner

    def __post_init__(self):
        if not self.full_id:
            if self.provider == "google":
                self.full_id = f"google/{self.id}"
            else:
                self.full_id = f"openrouter/{self.id}"
        if self.param_count == 0.0:
            self.param_count = _extract_param_count(self.id)


@dataclass
class ModelPool:
    """Dynamic pool of free models from multiple providers.
    
    Models are ranked by parameter count (extracted from model names).
    Higher parameter models are tried first as they tend to be more capable judges.
    Supports both OpenRouter and Google AI Studio.
    
    Usage:
        pool = ModelPool(openrouter_api_key="sk-or-...", google_api_key="AIza...")
        await pool.initialize()
        judges = pool.get_available_judges(preferred=["google/gemini-2.5-flash"])
    """
    openrouter_api_key: str = ""
    google_api_key: str = ""
    _models: list[FreeModel] = field(default_factory=list)
    _rate_limited: dict[str, float] = field(default_factory=dict)  # model_id → unblock_timestamp
    _used_in_session: set[str] = field(default_factory=set)
    _last_fetch: float = 0.0
    _initialized: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized and len(self._models) > 0

    async def initialize(self) -> None:
        """Fetch free models from all configured providers. Safe to call multiple times (cached)."""
        now = time.time()
        if self._initialized and (now - self._last_fetch) < MODEL_CACHE_TTL:
            return  # Still fresh
        
        all_models: list[FreeModel] = []
        
        # Fetch from Google AI Studio
        if self.google_api_key:
            try:
                google_models = await self._fetch_google_models()
                all_models.extend(google_models)
            except Exception as e:
                console.print(f"[yellow]  ⚠ Google model fetch failed: {str(e)[:60]}[/yellow]")
        
        # Fetch from OpenRouter
        if self.openrouter_api_key:
            try:
                or_models = await self._fetch_openrouter_models()
                all_models.extend(or_models)
            except Exception as e:
                console.print(f"[yellow]  ⚠ OpenRouter model fetch failed: {str(e)[:60]}[/yellow]")
        
        if all_models:
            # Sort all models by param count descending, context length secondary
            all_models.sort(key=lambda m: (m.param_count, m.context_length), reverse=True)
            self._models = all_models
            self._initialized = True
            self._last_fetch = now
            
            google_count = sum(1 for m in all_models if m.provider == "google")
            or_count = sum(1 for m in all_models if m.provider == "openrouter")
            top3 = [f"{m.name[:25]} ({m.param_count:.0f}B)" for m in all_models[:3]]
            console.print(f"[dim]  Model pool: {len(all_models)} models (Google: {google_count}, OpenRouter: {or_count}) — top: {', '.join(top3)}[/dim]")
        else:
            console.print("[yellow]  ⚠ Model pool: no models found, falling back to config members[/yellow]")

    async def _fetch_google_models(self) -> list[FreeModel]:
        """Fetch text generation models from Google AI Studio.
        
        All models returned by the API are free tier.
        Filters to only models that support generateContent (text generation).
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={self.google_api_key}",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

        models = []
        # Skip non-text models (imagen, veo, embedding, aqa, tts, audio, robotics, etc.)
        skip_keywords = ["imagen", "veo", "embedding", "aqa", "tts", "audio", "predict",
                         "robotics", "computer-use", "deep-research", "image"]
        
        for entry in data.get("models", []):
            name = entry.get("name", "")  # e.g. "models/gemini-2.5-flash"
            display_name = entry.get("displayName", "")
            methods = entry.get("supportedGenerationMethods", [])
            
            # Must support text generation
            if "generateContent" not in methods:
                continue
            
            # Skip image/video/audio/embedding models
            name_lower = name.lower()
            if any(kw in name_lower for kw in skip_keywords):
                continue
            
            input_limit = entry.get("inputTokenLimit", 0) or 0
            if input_limit < MIN_CONTEXT_LENGTH:
                continue
            
            # Extract model ID (strip "models/" prefix)
            model_id = name.replace("models/", "")
            
            models.append(FreeModel(
                id=model_id,
                name=display_name or model_id,
                context_length=input_limit,
                provider="google",
            ))

        return models

    async def _fetch_openrouter_models(self) -> list[FreeModel]:
        """Fetch and filter free models from OpenRouter /api/v1/models."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {}
            if self.openrouter_api_key:
                headers["Authorization"] = f"Bearer {self.openrouter_api_key}"
                headers["HTTP-Referer"] = "https://github.com/promptlab"
                headers["X-Title"] = "PromptLab"

            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        models = []
        for entry in data.get("data", []):
            pricing = entry.get("pricing", {})
            prompt_cost = str(pricing.get("prompt", "1"))
            completion_cost = str(pricing.get("completion", "1"))

            # Only free models (both prompt and completion cost == "0")
            if prompt_cost != "0" or completion_cost != "0":
                continue

            ctx_len = entry.get("context_length", 0) or 0
            if ctx_len < MIN_CONTEXT_LENGTH:
                continue

            model_id = entry.get("id", "")
            if not model_id:
                continue

            models.append(FreeModel(
                id=model_id,
                name=entry.get("name", model_id),
                context_length=ctx_len,
                provider="openrouter",
            ))

        # Sort by parameter count descending (primary), context_length (secondary)
        # Higher param models are more capable judges
        models.sort(key=lambda m: (m.param_count, m.context_length), reverse=True)
        return models

    def get_available_judges(
        self,
        preferred: list[str],
    ) -> list[str]:
        """Get the FULL ordered list of available judge model IDs.
        
        Order: preferred models first, then ALL discovered pool models sorted
        by parameter count descending. Rate-limited models are pushed to the
        end (as last-resort fallbacks) rather than excluded.
        
        Args:
            preferred: Config-defined preferred models (e.g. ["openrouter/qwen/qwen3-4b:free"])
            
        Returns:
            List of ALL available model IDs, highest-capability first
        """
        now = time.time()
        result: list[str] = []
        rate_limited_fallbacks: list[str] = []
        seen: set[str] = set()

        # 1. Preferred models first (if not currently rate-limited)
        for model in preferred:
            if model in self._rate_limited and self._rate_limited[model] > now:
                rate_limited_fallbacks.append(model)
            elif model not in seen:
                result.append(model)
                seen.add(model)

        # 2. ALL discovered pool models, sorted by param count (already sorted)
        if self._initialized:
            for fm in self._models:
                if fm.full_id in seen:
                    continue
                if fm.full_id in self._rate_limited and self._rate_limited[fm.full_id] > now:
                    rate_limited_fallbacks.append(fm.full_id)
                    continue
                result.append(fm.full_id)
                seen.add(fm.full_id)

        # 3. Append rate-limited models at the end as last resort
        for model in rate_limited_fallbacks:
            if model not in seen:
                result.append(model)
                seen.add(model)

        return result

    def mark_rate_limited(self, model_id: str, cooldown: float = RATE_LIMIT_COOLDOWN) -> None:
        """Mark a model as rate-limited. It won't be returned by get_available_judges
        until the cooldown expires.
        
        Args:
            model_id: Full model ID (e.g. "openrouter/qwen/qwen3-4b:free")
            cooldown: Seconds to wait before retrying this model
        """
        self._rate_limited[model_id] = time.time() + cooldown
        model_short = model_id.split("/")[-1][:25]
        console.print(f"[yellow]    → {model_short} rate-limited, cooling down {cooldown:.0f}s[/yellow]")

    def mark_used(self, model_id: str) -> None:
        """Track that a model was used in this session (for stats)."""
        self._used_in_session.add(model_id)

    def reset_session(self) -> None:
        """Reset per-session tracking (call between evaluations if desired)."""
        self._used_in_session.clear()

    def get_pool_stats(self) -> dict:
        """Return stats about the pool for diagnostics."""
        now = time.time()
        active_limits = {k: v for k, v in self._rate_limited.items() if v > now}
        return {
            "total_discovered": len(self._models),
            "rate_limited": len(active_limits),
            "used_in_session": len(self._used_in_session),
            "initialized": self._initialized,
            "top_models": [
                f"{m.id.split('/')[-1][:25]} ({m.param_count:.0f}B)"
                for m in self._models[:5]
            ] if self._models else [],
        }
