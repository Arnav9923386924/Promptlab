"""Configuration loader for PromptLab.

Supports ${VAR} env-var expansion in YAML values.
Auto-loads .env files from the project root via python-dotenv.
"""

from pathlib import Path
from typing import Optional
import os
import re
import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover
    _load_dotenv = None  # type: ignore


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
    model_roles: dict[str, str] = {}
    use_fixed_judges: bool = False
    debug_judge_responses: bool = False
    required_judges: int = 2  # Minimum number of successful judge scores required
    verbose_attempts: bool = False  # Show detailed per-model attempt logs on console
    log_attempts: bool = True  # Write detailed attempt logs to file
    log_attempts_path: Optional[str] = None  # Custom log path (auto-generated if None)
    
    @field_validator('required_judges')
    @classmethod
    def validate_required_judges(cls, v: int) -> int:
        if v < 2:
            raise ValueError(
                f"council.required_judges must be >= 2 (got {v}). "
                "Council evaluation requires at least 2 judges for meaningful consensus."
            )
        return v


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
    # Generation mode: "web" | "docs_web" | "hybrid"
    #   web      — scrape + regex extract (original pipeline)
    #   docs_web — download docs → TF-IDF index → retrieve → LLM/heuristic gen
    #   hybrid   — docs_web first, web fallback if target not met
    generation_mode: str = "web"


class DocsWebConfig(BaseModel):
    """Configuration for the document-grounded (docs_web) generation pipeline."""
    max_docs: int = 20          # Max documents to download
    chunk_size: int = 800       # Chunk size in words for indexing
    chunk_overlap: int = 200    # Overlap in words between consecutive chunks
    retrieval_top_k: int = 10   # Top-K chunks per retrieval query
    target_count: int = 100     # Default target number of testcases
    llm_model: Optional[str] = None  # Override model for testcase generation (uses models.default if None)


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
    # Branch to push to (null → auto-detect current branch)
    branch: Optional[str] = "main"
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


class GuardrailConfig(BaseModel):
    """Guardrail testing configuration."""
    enabled: bool = False
    categories: list[str] = ["prompt_injection", "jailbreak", "system_prompt_extraction", "role_breaking", "data_exfiltration"]
    evaluator_model: Optional[str] = None  # Model for ambiguous evaluations
    rate_limit_delay: float = 3.0


class OptimizerConfig(BaseModel):
    """BSP optimizer configuration."""
    max_iterations: int = 3
    target_lint_score: float = 0.90
    plateau_threshold: float = 0.02
    optimizer_model: Optional[str] = None


class MultiTurnConfig(BaseModel):
    """Multi-turn evaluation configuration."""
    num_turns: int = 6
    rate_limit_delay: float = 3.0
    evaluator_model: Optional[str] = None


class TrainingDataConfig(BaseModel):
    """Training data generation configuration."""
    min_quality_score: float = 0.7
    output_format: str = "openai"  # openai, alpaca, sharegpt
    output_dir: str = ".promptlab/training_data"


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
    docs_web: DocsWebConfig = DocsWebConfig()
    guardrail: GuardrailConfig = GuardrailConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    multi_turn: MultiTurnConfig = MultiTurnConfig()
    training_data: TrainingDataConfig = TrainingDataConfig()


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: str) -> str:
    """Expand all ${VAR} references in a string to os.environ values.
    
    Supports ${VAR} anywhere in the string (not just whole-value).
    Returns the original placeholder if the env var is unset.
    """
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return _ENV_VAR_RE.sub(_replace, value)


def _deep_expand(obj: object) -> object:
    """Recursively expand ${VAR} in all string values of a dict/list tree."""
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _deep_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_expand(v) for v in obj]
    return obj


def load_config(path: Optional[Path] = None) -> PromptLabConfig:
    """Load configuration from promptlab.yaml.
    
    Automatically loads .env from the config file's directory (if present)
    before expanding ${VAR} placeholders throughout the YAML.
    
    Args:
        path: Path to config file (default: ./promptlab.yaml)
        
    Returns:
        PromptLabConfig object
    """
    if path is None:
        path = Path.cwd() / "promptlab.yaml"
    
    if not path.exists():
        return PromptLabConfig()
    
    # Auto-load .env from the same directory as promptlab.yaml
    # override=True ensures .env values always take precedence over stale
    # shell env vars (e.g. leftover OPENROUTER_API_KEY from a prior session).
    env_file = path.parent / ".env"
    if env_file.exists():
        if _load_dotenv is not None:
            _load_dotenv(env_file, override=True)
        else:
            # Minimal fallback: parse KEY=VALUE lines when python-dotenv is missing
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip()
                    if key:
                        os.environ[key] = val
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # Recursively expand ${VAR} in the entire config tree
    data = _deep_expand(data)
    
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
