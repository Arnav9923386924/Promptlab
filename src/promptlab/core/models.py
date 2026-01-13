"""Data models for PromptLab."""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class AssertionType(str, Enum):
    """Types of assertions supported."""
    CONTAINS = "contains"
    EXACT_MATCH = "exact_match"
    REGEX = "regex"
    ONE_OF = "one_of"
    MAX_LENGTH = "max_length"
    MIN_LENGTH = "min_length"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    COUNCIL_JUDGE = "council_judge"
    JSON_VALID = "json_valid"
    JSON_SCHEMA = "json_schema"


class Assertion(BaseModel):
    """A single assertion to validate LLM response."""
    type: AssertionType
    value: Optional[str] = None
    values: Optional[list[str]] = None
    pattern: Optional[str] = None
    min_score: Optional[float] = None
    max_chars: Optional[int] = Field(None, alias="chars")
    case_sensitive: bool = True
    criteria: Optional[str] = None
    mode: Optional[Literal["full", "fast", "vote"]] = None
    schema_: Optional[dict] = Field(None, alias="schema")


class TestCase(BaseModel):
    """A single test case."""
    id: str
    prompt: str
    input: Optional[str] = None
    input_file: Optional[str] = None
    expected: Optional[str] = None
    assertions: list[Assertion] = []
    tags: list[str] = []
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_ms: Optional[int] = None


class TestSuiteDefaults(BaseModel):
    """Default settings for a test suite."""
    model: Optional[str] = None
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = None
    timeout_ms: Optional[int] = 30000
    system_prompt: Optional[str] = None


class TestSuiteMetadata(BaseModel):
    """Metadata for a test suite."""
    name: str
    version: Optional[str] = None
    tags: list[str] = []
    owner: Optional[str] = None


class TestSuite(BaseModel):
    """A complete test suite loaded from YAML."""
    metadata: Optional[TestSuiteMetadata] = None
    defaults: Optional[TestSuiteDefaults] = None
    stage: Optional[str] = None
    cases: list[TestCase]


class AssertionResult(BaseModel):
    """Result of a single assertion."""
    type: AssertionType
    passed: bool
    expected: Optional[str] = None
    actual: Optional[str] = None
    message: Optional[str] = None
    score: Optional[float] = None


class TestResult(BaseModel):
    """Result of a single test case."""
    case_id: str
    status: Literal["passed", "failed", "skipped", "error"]
    response: Optional[str] = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    assertions: list[AssertionResult] = []
    error: Optional[str] = None
    council_scores: Optional[dict] = None


class RunSummary(BaseModel):
    """Summary of a test run."""
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    pass_rate: float
    duration_ms: int
    cost_usd: float = 0.0


class TestRun(BaseModel):
    """Complete test run result."""
    run_id: str
    timestamp: str
    config_path: str
    summary: RunSummary
    results: list[TestResult]
    baseline_run_id: Optional[str] = None
    regressions: list[str] = []
