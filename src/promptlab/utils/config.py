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
    generator: Optional[str] = None  # LLM for generating test cases
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


class BSPConfig(BaseModel):
    """Behavior Specification Prompt (BSP) configuration."""
    # Path to the BSP file or inline BSP content
    prompt: Optional[str] = None
    prompt_file: Optional[str] = None  # Path to file containing BSP
    # Minimum score threshold for passing
    min_score: float = 0.7
    # Whether to use council for BSP evaluation
    use_council: bool = True
    # Version tracking for BSP changes
    version: str = "1.0.0"
    # Auto-generation settings
    auto_generate: bool = True  # Enable auto test generation via scraping
    auto_generate_count: int = 50  # Number of tests to generate


class BaselineConfig(BaseModel):
    """Baseline score management configuration."""
    # Directory to store baseline scores
    storage_dir: str = ".promptlab/baselines"
    # Whether to auto-update baseline on improvement
    auto_update: bool = False
    # Minimum improvement required to update baseline (percentage)
    min_improvement: float = 0.0


class GitConfig(BaseModel):
    """Git integration configuration."""
    # Whether to enable git push on score improvement
    enabled: bool = False
    # Branch to push to
    branch: str = "main"
    # Commit message template (supports {score}, {previous_score}, {improvement})
    commit_template: str = "chore: BSP validation passed (score: {score:.2f}, improvement: +{improvement:.2f})"
    # Whether to push automatically or just commit
    auto_push: bool = False


class ScraperConfig(BaseModel):
    """Web scraper configuration."""
    serpapi_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    fallback_search: str = "searxng"
    max_pages: int = 5
    timeout: int = 30


class PromptLabConfig(BaseModel):
    """Complete PromptLab configuration."""
    version: int = 1
    models: ModelsConfig = ModelsConfig()
    council: CouncilConfig = CouncilConfig()
    testing: TestingConfig = TestingConfig()
    bsp: BSPConfig = BSPConfig()
    baseline: BaselineConfig = BaselineConfig()
    git: GitConfig = GitConfig()
    scraper: ScraperConfig = ScraperConfig()


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
                if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    config["api_key"] = os.environ.get(env_var, "")
    
    # Load BSP from file if specified
    if "bsp" in data and "prompt_file" in data["bsp"]:
        bsp_file = Path(data["bsp"]["prompt_file"])
        if bsp_file.exists():
            data["bsp"]["prompt"] = bsp_file.read_text(encoding="utf-8")
    
    return PromptLabConfig(**data)


def get_project_root() -> Optional[Path]:
    """Find the project root by looking for promptlab.yaml."""
    current = Path.cwd()
    
    while current != current.parent:
        if (current / "promptlab.yaml").exists():
            return current
        current = current.parent
    
    return None


def load_bsp(config: PromptLabConfig, project_root: Optional[Path] = None) -> Optional[str]:
    """Load the Behavior Specification Prompt from config or file.
    
    Args:
        config: PromptLab configuration
        project_root: Project root directory
        
    Returns:
        BSP string or None if not configured
    """
    if config.bsp.prompt:
        return config.bsp.prompt
    
    if config.bsp.prompt_file:
        bsp_path = Path(config.bsp.prompt_file)
        if not bsp_path.is_absolute() and project_root:
            bsp_path = project_root / bsp_path
        
        if bsp_path.exists():
            return bsp_path.read_text(encoding="utf-8")
    
    return None
