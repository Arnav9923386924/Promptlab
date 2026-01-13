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
    """Unified interface to run completions against various LLM providers."""
    
    def __init__(self, config: dict):
        """Initialize LLM runner with config.
        
        Args:
            config: Provider configuration from promptlab.yaml
        """
        self.config = config
        self.providers = config.get("providers", {})
        self.default_model = config.get("default", "ollama/llama3.1:8b")
    
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
            model: Model identifier (e.g., "ollama/llama3.1:8b")
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            CompletionResult with response and metadata
        """
        model = model or self.default_model
        provider, model_name = self._parse_model(model)
        
        start_time = time.time()
        
        if provider == "ollama":
            result = await self._complete_ollama(
                model_name, prompt, system_prompt, temperature, max_tokens
            )
        elif provider == "openrouter":
            result = await self._complete_openrouter(
                model_name, prompt, system_prompt, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        result.latency_ms = int((time.time() - start_time) * 1000)
        result.model = model
        
        return result
    
    def _parse_model(self, model: str) -> tuple[str, str]:
        """Parse model string into provider and model name."""
        if "/" in model:
            parts = model.split("/", 1)
            return parts[0], parts[1]
        return "ollama", model
    
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
        
        async with httpx.AsyncClient(timeout=60.0) as client:
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
    
    async def _complete_openrouter(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Complete using OpenRouter."""
        api_key = self.providers.get("openrouter", {}).get("api_key")
        if not api_key:
            raise ValueError("OpenRouter API key not configured")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract response
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        
        return CompletionResult(
            text=text,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            cost_usd=0.0,  # TODO: Calculate from model pricing
        )
    
    async def check_health(self, model: Optional[str] = None) -> bool:
        """Check if the model is available."""
        model = model or self.default_model
        provider, model_name = self._parse_model(model)
        
        try:
            if provider == "ollama":
                endpoint = self.providers.get("ollama", {}).get("endpoint", "http://localhost:11434")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{endpoint}/api/tags")
                    return response.status_code == 200
            return True
        except Exception:
            return False
