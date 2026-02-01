"""Behavior Specification Prompt (BSP) Validator.

This module handles the complete BSP validation workflow:
1. Load BSP (system prompt that defines LLM behavior)
2. Run tests with BSP prepended to all prompts
3. Collect all outputs into a single file
4. Pass outputs to LLM Council for batch evaluation
5. Compare scores with baseline
6. Push to git if improved
"""

import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from promptlab.utils.config import PromptLabConfig, load_bsp
from promptlab.llm_council.llm_runner.runner import LLMRunner
from promptlab.llm_council.council.council import Council
from promptlab.orchestrators.parser import parse_test_file, discover_test_files
from promptlab.orchestrators.models import TestCase, TestSuite

console = Console()


@dataclass
class TestOutput:
    """Single test output for council review."""
    test_id: str
    prompt: str
    bsp: str
    response: str
    expected: Optional[str] = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class BatchOutput:
    """Complete batch of test outputs for council evaluation."""
    run_id: str
    timestamp: str
    bsp: str
    bsp_hash: str
    bsp_version: str
    model: str
    total_tests: int
    outputs: list[TestOutput] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "bsp": self.bsp,
            "bsp_hash": self.bsp_hash,
            "bsp_version": self.bsp_version,
            "model": self.model,
            "total_tests": self.total_tests,
            "outputs": [asdict(o) for o in self.outputs],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BatchOutput":
        """Load from dictionary."""
        outputs = [TestOutput(**o) for o in data.get("outputs", [])]
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            bsp=data["bsp"],
            bsp_hash=data["bsp_hash"],
            bsp_version=data["bsp_version"],
            model=data["model"],
            total_tests=data["total_tests"],
            outputs=outputs,
        )


@dataclass 
class CouncilBatchResult:
    """Result from council batch evaluation."""
    final_score: float
    passed: bool
    confidence: Literal["high", "medium", "low"]
    individual_scores: list[dict]
    summary: str
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    """Complete validation result."""
    run_id: str
    timestamp: str
    bsp_hash: str
    bsp_version: str
    model: str
    total_tests: int
    council_score: float
    passed: bool
    baseline_score: Optional[float] = None
    improvement: Optional[float] = None
    should_push: bool = False
    council_result: Optional[CouncilBatchResult] = None
    outputs_file: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        if self.council_result:
            result["council_result"] = self.council_result.to_dict()
        return result


class BSPValidator:
    """Validates Behavior Specification Prompts using council evaluation."""
    
    BATCH_EVALUATION_PROMPT = """You are evaluating an LLM's performance with a specific Behavior Specification Prompt (BSP).

## BEHAVIOR SPECIFICATION PROMPT (BSP):
{bsp}

## EVALUATION CRITERIA:
1. **Role Adherence** (0-1): Does the LLM consistently act according to the BSP?
2. **Response Quality** (0-1): Are responses accurate, helpful, and well-formatted?
3. **Consistency** (0-1): Are responses consistent across similar questions?
4. **Appropriateness** (0-1): Does the LLM stay within its defined scope?

## TEST OUTPUTS TO EVALUATE:
{outputs}

## YOUR TASK:
Evaluate the overall performance of this BSP configuration.

Respond in this EXACT format:
ROLE_ADHERENCE: [0.0-1.0]
RESPONSE_QUALITY: [0.0-1.0]
CONSISTENCY: [0.0-1.0]
APPROPRIATENESS: [0.0-1.0]
FINAL_SCORE: [0.0-1.0]
CONFIDENCE: [high/medium/low]
SUMMARY: [2-3 sentence summary of performance]
RECOMMENDATIONS: [Comma-separated list of improvement suggestions]
"""

    def __init__(self, config: PromptLabConfig, project_root: Optional[Path] = None):
        """Initialize BSP validator.
        
        Args:
            config: PromptLab configuration
            project_root: Project root directory
        """
        self.config = config
        self.project_root = project_root or Path.cwd()
        
        # Initialize LLM runner
        self.llm_runner = LLMRunner({
            "default": config.models.default,
            "providers": {
                name: {"endpoint": p.endpoint, "api_key": p.api_key}
                for name, p in config.models.providers.items()
            },
        })
        
        # Initialize council if enabled
        if config.council.enabled:
            self.council = Council(
                {
                    "members": config.council.members,
                    "chairman": config.council.chairman,
                    "mode": config.council.mode,
                },
                self.llm_runner,
            )
        else:
            self.council = None
        
        # Load BSP
        self.bsp = load_bsp(config, project_root)
        self.bsp_hash = self._compute_bsp_hash(self.bsp) if self.bsp else ""
    
    def _compute_bsp_hash(self, bsp: str) -> str:
        """Compute hash of BSP for versioning."""
        return hashlib.sha256(bsp.encode()).hexdigest()[:12]
    
    async def run_tests_with_bsp(
        self,
        test_files: list[Path],
        show_progress: bool = True,
    ) -> BatchOutput:
        """Run all tests with BSP prepended to prompts.
        
        Includes rate limiting to avoid 429 errors from API providers.
        
        Args:
            test_files: List of test file paths
            show_progress: Whether to show progress bar
            
        Returns:
            BatchOutput containing all test outputs
        """
        import asyncio
        
        run_id = f"bsp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        outputs: list[TestOutput] = []
        
        # Collect all test cases
        all_cases: list[tuple[TestCase, TestSuite]] = []
        for test_file in test_files:
            suite = parse_test_file(test_file)
            for case in suite.cases:
                all_cases.append((case, suite))
        
        console.print(f"[cyan]Running {len(all_cases)} tests with BSP (rate-limited)...[/cyan]")
        
        # Rate limiting: delay between requests to avoid 429 errors
        # OpenRouter free tier: ~20 req/min, so ~3 seconds between requests
        rate_limit_delay = 3.0  # seconds between API calls
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running tests...", total=len(all_cases))
                
                for i, (case, suite) in enumerate(all_cases):
                    progress.update(task, description=f"Running: {case.id}")
                    output = await self._run_single_test(case, suite)
                    outputs.append(output)
                    progress.advance(task)
                    
                    # Rate limiting: wait between requests (skip for last one)
                    if i < len(all_cases) - 1:
                        await asyncio.sleep(rate_limit_delay)
        else:
            for i, (case, suite) in enumerate(all_cases):
                output = await self._run_single_test(case, suite)
                outputs.append(output)
                
                # Rate limiting
                if i < len(all_cases) - 1:
                    await asyncio.sleep(rate_limit_delay)
        
        return BatchOutput(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            bsp=self.bsp or "",
            bsp_hash=self.bsp_hash,
            bsp_version=self.config.bsp.version,
            model=self.config.models.default,
            total_tests=len(outputs),
            outputs=outputs,
        )
    
    async def _run_single_test(self, case: TestCase, suite: TestSuite) -> TestOutput:
        """Run a single test case with BSP prepended."""
        # Build full prompt with BSP
        full_prompt = case.prompt
        
        # Get system prompt (BSP takes precedence)
        system_prompt = self.bsp
        if not system_prompt and suite.defaults:
            system_prompt = suite.defaults.system_prompt
        
        # Get model
        model = case.model or (suite.defaults.model if suite.defaults else None) or self.config.models.default
        temperature = case.temperature if case.temperature is not None else (suite.defaults.temperature if suite.defaults else 0)
        
        try:
            completion = await self.llm_runner.complete(
                prompt=full_prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=case.max_tokens or 1000,
            )
            
            return TestOutput(
                test_id=case.id,
                prompt=full_prompt,
                bsp=system_prompt or "",
                response=completion.text,
                expected=case.expected,
                latency_ms=completion.latency_ms,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
            )
        except Exception as e:
            return TestOutput(
                test_id=case.id,
                prompt=full_prompt,
                bsp=system_prompt or "",
                response=f"ERROR: {str(e)}",
                expected=case.expected,
            )
    
    def save_outputs(self, batch: BatchOutput, output_dir: Optional[Path] = None) -> Path:
        """Save batch outputs to JSON file.
        
        Args:
            batch: Batch output to save
            output_dir: Directory to save to (default: .promptlab/runs)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = self.project_root / ".promptlab" / "runs"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{batch.run_id}_outputs.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(batch.to_dict(), f, indent=2, ensure_ascii=False)
        
        return output_file
    
    async def evaluate_batch_with_council(self, batch: BatchOutput) -> CouncilBatchResult:
        """Evaluate batch outputs using LLM Council with BATCH processing.
        
        This uses batch evaluation to dramatically reduce API calls:
        - Old: n tests × m judges = n×m API calls (e.g., 30×3 = 90 calls)
        - New: 1 batch × m judges = m API calls (e.g., 1×3 = 3 calls)
        
        Args:
            batch: Batch output to evaluate
            
        Returns:
            CouncilBatchResult with scores and feedback
        """
        if not self.council:
            # If no council, use single LLM evaluation
            return await self._evaluate_with_single_llm(batch)
        
        console.print("[cyan]Submitting outputs to LLM Council for BATCH evaluation...[/cyan]")
        console.print(f"[dim]  → {len(batch.outputs)} tests evaluated in {len(self.council.members)} API calls (batch mode)[/dim]")
        
        # Prepare outputs for batch evaluation
        outputs_for_batch = [
            {
                "test_id": out.test_id,
                "prompt": out.prompt,
                "response": out.response,
                "expected": out.expected,
            }
            for out in batch.outputs
        ]
        
        # Use batch evaluation - ONE API call per judge
        batch_result = await self.council.evaluate_batch(
            outputs=outputs_for_batch,
            bsp=batch.bsp,
            min_score=self.config.bsp.min_score,
        )
        
        return CouncilBatchResult(
            final_score=batch_result.final_score,
            passed=batch_result.passed,
            confidence=batch_result.confidence,
            individual_scores=[
                {
                    "model": s.model,
                    "score": s.overall_score,
                    "reasoning": s.reasoning,
                    "role_adherence": s.role_adherence,
                    "response_quality": s.response_quality,
                    "consistency": s.consistency,
                }
                for s in batch_result.member_scores
            ],
            summary=batch_result.summary,
            recommendations=batch_result.recommendations,
        )
    
    async def _evaluate_with_single_llm(self, batch: BatchOutput) -> CouncilBatchResult:
        """Fallback evaluation with single LLM if council is disabled."""
        outputs_text = self._format_outputs_for_evaluation(batch.outputs[:20])
        
        eval_prompt = self.BATCH_EVALUATION_PROMPT.format(
            bsp=batch.bsp[:1000] if batch.bsp else "No BSP specified",
            outputs=outputs_text,
        )
        
        result = await self.llm_runner.complete(
            prompt=eval_prompt,
            model=self.config.models.default,
            temperature=0,
        )
        
        scores = self._parse_batch_scores(result.text)
        
        return CouncilBatchResult(
            final_score=scores.get("final_score", 0.5),
            passed=scores.get("final_score", 0.5) >= self.config.bsp.min_score,
            confidence=scores.get("confidence", "medium"),
            individual_scores=[],
            summary=scores.get("summary", result.text[:500]),
            recommendations=scores.get("recommendations", []),
        )
    
    def _format_outputs_for_evaluation(self, outputs: list[TestOutput]) -> str:
        """Format test outputs for council evaluation."""
        formatted = []
        for i, out in enumerate(outputs, 1):
            formatted.append(f"""
### Test {i}: {out.test_id}
**Prompt:** {out.prompt[:200]}
**Response:** {out.response[:500]}
{"**Expected:** " + out.expected[:200] if out.expected else ""}
---""")
        return "\n".join(formatted)
    
    def _parse_batch_scores(self, text: str) -> dict:
        """Parse scores from council evaluation response."""
        scores = {
            "role_adherence": 0.5,
            "response_quality": 0.5,
            "consistency": 0.5,
            "appropriateness": 0.5,
            "final_score": 0.5,
            "confidence": "medium",
            "summary": "",
            "recommendations": [],
        }
        
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("ROLE_ADHERENCE:"):
                try:
                    scores["role_adherence"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("RESPONSE_QUALITY:"):
                try:
                    scores["response_quality"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("CONSISTENCY:"):
                try:
                    scores["consistency"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("APPROPRIATENESS:"):
                try:
                    scores["appropriateness"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("FINAL_SCORE:"):
                try:
                    scores["final_score"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("CONFIDENCE:"):
                conf = line.split(":")[-1].strip().lower()
                if conf in ["high", "medium", "low"]:
                    scores["confidence"] = conf
            elif line.startswith("SUMMARY:"):
                scores["summary"] = line.split(":", 1)[-1].strip()
            elif line.startswith("RECOMMENDATIONS:"):
                recs = line.split(":", 1)[-1].strip()
                scores["recommendations"] = [r.strip() for r in recs.split(",") if r.strip()]
        
        return scores
    
    async def validate(
        self,
        test_dir: Optional[Path] = None,
        test_files: Optional[list[str]] = None,
        auto_generate: bool = True,
        generate_count: int = 50,
    ) -> ValidationResult:
        """Run complete BSP validation workflow.
        
        If no test files exist and auto_generate is True, tests will be
        automatically generated via web scraping based on BSP analysis.
        
        Args:
            test_dir: Directory containing test files
            test_files: Specific test files to run
            auto_generate: Whether to auto-generate tests if none exist
            generate_count: Number of tests to generate if auto-generating
            
        Returns:
            ValidationResult with scores and comparison
        """
        from promptlab.orchestrators.baseline import BaselineManager
        
        if test_dir is None:
            test_dir = self.project_root / "temp"
        
        # Discover test files
        if test_files:
            files = [Path(f) for f in test_files]
        else:
            files = discover_test_files(test_dir)
        
        # AUTO-GENERATE TESTS if none exist
        if not files and auto_generate and self.bsp:
            console.print("[yellow]No test files found. Auto-generating tests from BSP...[/yellow]\n")
            
            try:
                from promptlab.utils.auto_test_generator import AutoTestGenerator
                
                # Get API keys from config
                serpapi_key = self.config.scraper.serpapi_key if self.config.scraper else None
                
                generator = AutoTestGenerator(
                    serpapi_key=serpapi_key,
                    max_pages=20,
                )
                
                generated = await generator.generate_tests(
                    bsp=self.bsp,
                    target_count=generate_count,
                    output_dir=test_dir,
                )
                
                console.print(f"\n[green]✓ Auto-generated {len(generated.qa_pairs) + len(generated.masked_tests)} tests in {generated.generation_time:.1f}s[/green]")
                console.print(f"[green]✓ Tests saved to: {generated.output_file}[/green]\n")
                
                # Re-discover test files after generation
                files = discover_test_files(test_dir)
                
            except Exception as e:
                console.print(f"[red]Auto-generation failed: {e}[/red]")
                console.print("[yellow]Falling back to manual test requirement.[/yellow]")
        
        if not files:
            raise ValueError(f"No test files found in {test_dir}. Create tests or enable auto-generation.")
        
        console.print(Panel(
            f"[bold]BSP Version:[/bold] {self.config.bsp.version}\n"
            f"[bold]BSP Hash:[/bold] {self.bsp_hash[:8]}...\n"
            f"[bold]Model:[/bold] {self.config.models.default}\n"
            f"[bold]Test Files:[/bold] {len(files)}\n"
            f"[bold]Council:[/bold] {'enabled' if self.council else 'disabled'}",
            title="BSP Validation",
            border_style="blue",
        ))
        
        # Step 1: Run tests with BSP
        console.print("\n[bold cyan]Step 1: Running tests with BSP...[/bold cyan]")
        batch = await self.run_tests_with_bsp(files)
        console.print(f"[green]✓ Completed {batch.total_tests} tests[/green]")
        
        # Step 2: Save outputs
        console.print("\n[bold cyan]Step 2: Saving outputs...[/bold cyan]")
        outputs_file = self.save_outputs(batch)
        console.print(f"[green]✓ Saved to {outputs_file}[/green]")
        
        # Step 3: Council evaluation
        console.print("\n[bold cyan]Step 3: Council evaluation...[/bold cyan]")
        council_result = await self.evaluate_batch_with_council(batch)
        console.print(f"[green]✓ Council score: {council_result.final_score:.2f}[/green]")
        
        # Step 4: Compare with baseline
        console.print("\n[bold cyan]Step 4: Comparing with baseline...[/bold cyan]")
        baseline_manager = BaselineManager(self.project_root / ".promptlab")
        baseline = baseline_manager.get_latest_baseline()
        
        baseline_score = baseline.score if baseline else None
        improvement = None
        should_push = False
        
        if baseline_score is not None:
            improvement = council_result.final_score - baseline_score
            should_push = improvement > self.config.baseline.min_improvement
            
            if improvement > 0:
                console.print(f"[green]✓ Improvement: +{improvement:.2f} ({baseline_score:.2f} → {council_result.final_score:.2f})[/green]")
            elif improvement < 0:
                console.print(f"[red]✗ Regression: {improvement:.2f} ({baseline_score:.2f} → {council_result.final_score:.2f})[/red]")
            else:
                console.print(f"[yellow]= No change: {council_result.final_score:.2f}[/yellow]")
        else:
            console.print("[yellow]No baseline found. This will be the first baseline.[/yellow]")
            should_push = council_result.passed
        
        result = ValidationResult(
            run_id=batch.run_id,
            timestamp=batch.timestamp,
            bsp_hash=batch.bsp_hash,
            bsp_version=batch.bsp_version,
            model=batch.model,
            total_tests=batch.total_tests,
            council_score=council_result.final_score,
            passed=council_result.passed,
            baseline_score=baseline_score,
            improvement=improvement,
            should_push=should_push,
            council_result=council_result,
            outputs_file=str(outputs_file),
        )
        
        # Print summary
        self._print_validation_summary(result)
        
        return result
    
    def _print_validation_summary(self, result: ValidationResult):
        """Print validation summary to console."""
        console.print()
        
        # Create results table
        table = Table(title="Validation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("Council Score", f"{result.council_score:.2f}")
        table.add_row("Passed", "[green]Yes[/green]" if result.passed else "[red]No[/red]")
        table.add_row("Confidence", result.council_result.confidence if result.council_result else "N/A")
        
        if result.baseline_score is not None:
            table.add_row("Baseline Score", f"{result.baseline_score:.2f}")
            improvement_str = f"{result.improvement:+.2f}" if result.improvement else "0.00"
            improvement_color = "green" if result.improvement and result.improvement > 0 else "red" if result.improvement and result.improvement < 0 else "yellow"
            table.add_row("Improvement", f"[{improvement_color}]{improvement_str}[/{improvement_color}]")
        
        table.add_row("Should Push", "[green]Yes[/green]" if result.should_push else "[red]No[/red]")
        
        console.print(table)
        
        # Print recommendations
        if result.council_result and result.council_result.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in result.council_result.recommendations:
                console.print(f"  • {rec}")
