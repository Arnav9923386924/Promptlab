"""Test runner - orchestrates test execution with parallel processing."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import time

from rich.console import Console
from rich.table import Table

from promptlab.core.models import (
    TestSuite, TestCase, TestResult, TestRun, RunSummary, AssertionType
)
from promptlab.core.parser import parse_test_file, discover_test_files
from promptlab.core.assertions import run_all_assertions
from promptlab.llm_council.llm_runner.runner import LLMRunner
from promptlab.llm_council.council.council import Council
from promptlab.utils.config import PromptLabConfig


console = Console()


class TestRunner:
    """Orchestrates test execution."""
    
    def __init__(self, config: PromptLabConfig):
        """Initialize test runner.
        
        Args:
            config: PromptLab configuration
        """
        self.config = config
        self.llm_runner = LLMRunner({
            "default": config.models.default,
            "providers": {
                name: {"endpoint": p.endpoint, "api_key": p.api_key}
                for name, p in config.models.providers.items()
            },
        })
        
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
    
    async def run_test_file(self, path: Path, semaphore: asyncio.Semaphore = None) -> list[TestResult]:
        """Run all tests in a file.
        
        Args:
            path: Path to test file
            semaphore: Optional semaphore for concurrency control
            
        Returns:
            List of test results
        """
        suite = parse_test_file(path)
        
        # Run all cases in parallel with semaphore
        async def run_with_semaphore(case):
            if semaphore:
                async with semaphore:
                    return await self.run_test_case(case, suite)
            return await self.run_test_case(case, suite)
        
        results = await asyncio.gather(*[run_with_semaphore(case) for case in suite.cases])
        return list(results)
    
    async def run_test_case(self, case: TestCase, suite: TestSuite) -> TestResult:
        """Run a single test case.
        
        Args:
            case: Test case to run
            suite: Parent test suite (for defaults)
            
        Returns:
            TestResult
        """
        start_time = time.time()
        
        # Merge defaults
        model = case.model or (suite.defaults.model if suite.defaults else None) or self.config.models.default
        temperature = case.temperature if case.temperature is not None else (suite.defaults.temperature if suite.defaults else 0)
        system_prompt = suite.defaults.system_prompt if suite.defaults else None
        
        try:
            # Run LLM completion
            completion = await self.llm_runner.complete(
                prompt=case.prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=case.max_tokens or 1000,
            )
            
            response = completion.text
            
            # Run basic assertions
            assertion_results = run_all_assertions(case.assertions, response)
            
            # Handle council judge assertions
            council_scores = None
            for i, assertion in enumerate(case.assertions):
                if assertion.type == AssertionType.COUNCIL_JUDGE and self.council:
                    council_result = await self.council.evaluate(
                        response=response,
                        criteria=assertion.criteria or "",
                        min_score=assertion.min_score or 0.7,
                        mode=assertion.mode,
                    )
                    
                    # Update assertion result
                    assertion_results[i].passed = council_result.passed
                    assertion_results[i].score = council_result.final_score
                    assertion_results[i].message = council_result.consensus_summary
                    
                    council_scores = {
                        "final_score": council_result.final_score,
                        "confidence": council_result.confidence,
                        "members": [
                            {"model": s.model, "score": s.score, "reasoning": s.reasoning}
                            for s in council_result.member_scores
                        ],
                    }
            
            # Determine overall status
            all_passed = all(r.passed for r in assertion_results)
            
            return TestResult(
                case_id=case.id,
                status="passed" if all_passed else "failed",
                response=response,
                latency_ms=completion.latency_ms,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                assertions=assertion_results,
                council_scores=council_scores,
            )
            
        except Exception as e:
            return TestResult(
                case_id=case.id,
                status="error",
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000),
            )
    
    async def run_all(
        self,
        test_dir: Path,
        files: Optional[list[str]] = None,
        stage: Optional[str] = None,
    ) -> TestRun:
        """Run all tests.
        
        Args:
            test_dir: Directory containing test files
            files: Specific files to run (optional)
            stage: Filter by pipeline stage (optional)
            
        Returns:
            TestRun with all results
        """
        start_time = time.time()
        
        # Discover test files
        if files:
            test_files = [Path(f) for f in files]
        else:
            test_files = discover_test_files(test_dir)
        
        # Filter by stage if specified
        if stage:
            filtered = []
            for f in test_files:
                suite = parse_test_file(f)
                if suite.stage == stage:
                    filtered.append(f)
            test_files = filtered
        
        # Create semaphore for controlled parallelism
        parallelism = self.config.testing.parallelism if hasattr(self.config, 'testing') and self.config.testing else 4
        semaphore = asyncio.Semaphore(parallelism)
        
        all_results = []
        
        console.print(f"[dim]Running with parallelism={parallelism}...[/dim]")
        
        # Run all test files in parallel
        async def run_file(test_file):
            return await self.run_test_file(test_file, semaphore)
        
        file_results = await asyncio.gather(*[run_file(f) for f in test_files])
        
        for results in file_results:
            all_results.extend(results)
        
        # Calculate summary
        total = len(all_results)
        passed = sum(1 for r in all_results if r.status == "passed")
        failed = sum(1 for r in all_results if r.status == "failed")
        skipped = sum(1 for r in all_results if r.status == "skipped")
        errors = sum(1 for r in all_results if r.status == "error")
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return TestRun(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            config_path=str(test_dir.parent / "promptlab.yaml"),
            summary=RunSummary(
                total=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                pass_rate=passed / total if total > 0 else 0,
                duration_ms=duration_ms,
            ),
            results=all_results,
        )


def print_results(run: TestRun):
    """Print test results to console."""
    console.print()
    
    # Results table
    table = Table(title="Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Time", justify="right")
    table.add_column("Details")
    
    for result in run.results:
        status_style = {
            "passed": "[green]✓ PASSED[/green]",
            "failed": "[red]✗ FAILED[/red]",
            "skipped": "[yellow]⊘ SKIPPED[/yellow]",
            "error": "[red]⚠ ERROR[/red]",
        }
        
        details = ""
        if result.status == "failed":
            failed_assertions = [a for a in result.assertions if not a.passed]
            if failed_assertions:
                details = failed_assertions[0].message or ""
        elif result.status == "error":
            details = result.error or ""
        elif result.council_scores:
            details = f"Council: {result.council_scores['final_score']:.2f}"
        
        table.add_row(
            result.case_id,
            status_style.get(result.status, result.status),
            f"{result.latency_ms}ms",
            details[:50] + "..." if len(details) > 50 else details,
        )
    
    console.print(table)
    
    # Summary
    console.print()
    summary = run.summary
    pass_rate_color = "green" if summary.pass_rate >= 0.9 else "yellow" if summary.pass_rate >= 0.7 else "red"
    
    console.print(f"[bold]Results:[/bold] {summary.passed}/{summary.total} passed ([{pass_rate_color}]{summary.pass_rate:.1%}[/{pass_rate_color}])")
    console.print(f"[bold]Duration:[/bold] {summary.duration_ms}ms")
    
    if summary.failed > 0:
        console.print(f"[red]Failed: {summary.failed}[/red]")
    if summary.errors > 0:
        console.print(f"[red]Errors: {summary.errors}[/red]")
