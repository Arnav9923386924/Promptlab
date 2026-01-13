"""Configuration loader for PromptLab."""

from pathlib import Path
from typing import Optional
import os
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""
    endpoint: Optional[str] = None
    api_key: Optional[str] = None


class ModelsConfig(BaseModel):
    """Models configuration."""
    default: str = "ollama/llama3.1:8b"
    providers: dict[str, ProviderConfig] = {}


class CouncilConfig(BaseModel):
    """Council configuration."""
    enabled: bool = True
    mode: str = "fast"
    members: list[str] = []
    chairman: Optional[str] = None


class TestingConfig(BaseModel):
    """Testing configuration."""
    parallelism: int = 4
    timeout_ms: int = 30000
    retries: int = 2


class PromptLabConfig(BaseModel):
    """Complete PromptLab configuration."""
    version: int = 1
    models: ModelsConfig = ModelsConfig()
    council: CouncilConfig = CouncilConfig()
    testing: TestingConfig = TestingConfig()


def load_config(path: Optional[Path] = None) -> PromptLabConfig:
    """Load configuration from promptlab.yaml.
    
    Args:
        path: Path to config file (default: ./promptlab.yaml)
        
    Returns:
        PromptLabConfig object
    """
    if path is None:
        path = Path.cwd() / "promptlab.yaml"
    
    if not path.exists():
        return PromptLabConfig()
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # Expand environment variables in API keys
    if "models" in data and "providers" in data["models"]:
        for provider, config in data["models"]["providers"].items():
            if isinstance(config, dict) and "api_key" in config:
                api_key = config["api_key"]
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    config["api_key"] = os.environ.get(env_var, "")
    
    return PromptLabConfig(**data)


def get_project_root() -> Optional[Path]:
    """Find the project root by looking for promptlab.yaml."""
    current = Path.cwd()
    
    while current != current.parent:
        if (current / "promptlab.yaml").exists():
            return current
        current = current.parent
    
    return None
