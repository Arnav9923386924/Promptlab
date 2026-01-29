"""LLM Runner - Unified interface to multiple LLM providers."""

import httpx
from typing import Optional
from pydantic import BaseModel
import time


class CompletionResult(BaseModel):
    """Result from an LLM completion."""
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model: str = ""
    cost_usd: float = 0.0


class LLMRunner:
    """Unified interface to run completions against various LLM providers.
    
    Supported providers:
    - ollama: Local Ollama instance
    - openrouter: OpenRouter API (access to many models)
    - openai: OpenAI API (GPT-4, GPT-3.5)
    - anthropic: Anthropic API (Claude)
    - google: Google AI API (Gemini)
    - xai: xAI API (Grok)
    """
    
    # API endpoints for direct providers
    ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "google": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "xai": "https://api.x.ai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }
    
    def __init__(self, config: dict):
        """Initialize LLM runner with config.
        
        Args:
            config: Provider configuration from promptlab.yaml
        """
        self.config = config
        self.providers = config.get("providers", {})
        self.default_model = config.get("default", "ollama/llama3.1:8b")
        # Reusable HTTP client for connection pooling
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1000,
    ) -> CompletionResult:
        """Run a completion against the specified model.
        
        Args:
            prompt: The user prompt
            model: Model identifier (e.g., "ollama/llama3.1:8b", "openai/gpt-4o")
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            CompletionResult with response and metadata
        """
        model = model or self.default_model
        provider, model_name = self._parse_model(model)
        
        start_time = time.time()
        
        # Route to appropriate provider
        if provider == "ollama":
            result = await self._complete_ollama(
                model_name, prompt, system_prompt, temperature, max_tokens
            )
        elif provider == "openrouter":
            result = await self._complete_openai_compatible(
                "openrouter", model_name, prompt, system_prompt, temperature, max_tokens
            )
        elif provider == "openai":
            result = await self._complete_openai_compatible(
                "openai", model_name, prompt, system_prompt, temperature, max_tokens
            )
        elif provider == "xai":
            result = await self._complete_openai_compatible(
                "xai", model_name, prompt, system_prompt, temperature, max_tokens
            )
        elif provider == "anthropic":
            result = await self._complete_anthropic(
                model_name, prompt, system_prompt, temperature, max_tokens
            )
        elif provider == "google":
            result = await self._complete_google(
                model_name, prompt, system_prompt, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: ollama, openrouter, openai, anthropic, google, xai")
        
        result.latency_ms = int((time.time() - start_time) * 1000)
        result.model = model
        
        return result
    
    def _parse_model(self, model: str) -> tuple[str, str]:
        """Parse model string into provider and model name."""
        if "/" in model:
            parts = model.split("/", 1)
            return parts[0], parts[1]
        return "ollama", model
    
    def _get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        api_key = self.providers.get(provider, {}).get("api_key")
        if not api_key:
            raise ValueError(f"{provider.title()} API key not configured. Add to promptlab.yaml: providers.{provider}.api_key")
        return api_key
    
    async def _complete_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Complete using Ollama."""
        endpoint = self.providers.get("ollama", {}).get("endpoint", "http://localhost:11434")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        client = await self._get_client()
        response = await client.post(
            f"{endpoint}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return CompletionResult(
            text=data.get("message", {}).get("content", ""),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            cost_usd=0.0,  # Ollama is free
        )
    
    async def _complete_openai_compatible(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Complete using OpenAI-compatible API (OpenAI, OpenRouter, xAI)."""
        api_key = self._get_api_key(provider)
        endpoint = self.ENDPOINTS[provider]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # OpenRouter needs additional headers
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/promptlab"
            headers["X-Title"] = "PromptLab"
        
        client = await self._get_client()
        response = await client.post(
            endpoint,
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        
        return CompletionResult(
            text=text,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            cost_usd=0.0,
        )
    
    async def _complete_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Complete using Anthropic Claude API."""
        api_key = self._get_api_key("anthropic")
        
        messages = [{"role": "user", "content": prompt}]
        
        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINTS["anthropic"],
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract text from content blocks
        content = data.get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text += block.get("text", "")
        
        usage = data.get("usage", {})
        
        return CompletionResult(
            text=text,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            cost_usd=0.0,
        )
    
    async def _complete_google(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Complete using Google Gemini API."""
        api_key = self._get_api_key("google")
        endpoint = self.ENDPOINTS["google"].format(model=model)
        
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {system_prompt}"}]
            })
            contents.append({
                "role": "model", 
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        client = await self._get_client()
        response = await client.post(
            f"{endpoint}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract text from response
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                text += part.get("text", "")
        
        usage = data.get("usageMetadata", {})
        
        return CompletionResult(
            text=text,
            tokens_in=usage.get("promptTokenCount", 0),
            tokens_out=usage.get("candidatesTokenCount", 0),
            cost_usd=0.0,
        )
    
    async def check_health(self, model: Optional[str] = None) -> bool:
        """Check if the model is available."""
        model = model or self.default_model
        provider, model_name = self._parse_model(model)
        
        try:
            if provider == "ollama":
                endpoint = self.providers.get("ollama", {}).get("endpoint", "http://localhost:11434")
                client = await self._get_client()
                response = await client.get(f"{endpoint}/api/tags")
                return response.status_code == 200
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
