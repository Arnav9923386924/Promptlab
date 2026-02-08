"""BSP Optimizer - Iteratively improves BSP based on evaluation feedback.

Given a BSP and failed test results, uses an LLM to rewrite
the BSP to address weaknesses. Runs the BSP Linter (free) before
and after each iteration. Minimizes API calls by:
- Pre-validating with linter (zero cost)
- Single LLM call per optimization iteration
- Early stopping when score plateaus
"""

import asyncio
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from promptlab.llm_council.llm_runner.runner import LLMRunner
from promptlab.utils.bsp_linter import BSPLinter, LintResult


@dataclass
class OptimizationStep:
    """One iteration of BSP optimization."""
    iteration: int
    bsp_text: str
    bsp_hash: str
    lint_score: float
    lint_issues_count: int
    changes_made: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    original_bsp: str
    optimized_bsp: str
    original_lint_score: float
    final_lint_score: float
    iterations_run: int
    max_iterations: int
    steps: list[OptimizationStep] = field(default_factory=list)
    api_calls_used: int = 0
    stopped_early: bool = False
    stop_reason: str = ""

    @property
    def improvement(self) -> float:
        return round(self.final_lint_score - self.original_lint_score, 4)


class BSPOptimizer:
    """Iteratively optimizes BSP quality with minimal API usage.
    
    Workflow:
    1. Lint BSP (free) to find issues
    2. If issues found, call LLM ONCE to rewrite BSP addressing issues
    3. Lint again (free) to verify improvement
    4. Repeat until no issues remain or max iterations reached
    5. Early stop if lint score plateaus
    
    Typical API cost: 1-5 calls total (one per iteration).
    """

    OPTIMIZE_PROMPT = """You are an expert prompt engineer. Improve this Behavior Specification Prompt (BSP) to fix the issues listed below.

## CURRENT BSP:
{bsp}

## ISSUES FOUND BY LINTER:
{issues}

## WEAK AREAS FROM EVALUATION (if any):
{weak_areas}

## RULES:
1. Keep the original intent and role intact
2. Fix ALL listed issues
3. Add missing sections (role, constraints, format, examples) if flagged
4. Replace vague language with specific, measurable instructions
5. Resolve any contradictions
6. Add edge case handling if missing
7. Keep the BSP concise (200-800 words ideal)
8. Use clear markdown structure with ## headers

## OUTPUT:
Return ONLY the improved BSP text. No explanations, no commentary — just the BSP.
"""

    def __init__(
        self,
        llm_runner: LLMRunner,
        optimizer_model: Optional[str] = None,
        max_iterations: int = 3,
        target_lint_score: float = 0.90,
        plateau_threshold: float = 0.02,
    ):
        """Initialize BSP optimizer.
        
        Args:
            llm_runner: LLM runner for API calls
            optimizer_model: Model to use for rewriting (uses default if None)
            max_iterations: Maximum optimization iterations (each = 1 API call)
            target_lint_score: Stop when lint score reaches this
            plateau_threshold: Stop if improvement < this between iterations
        """
        self.llm_runner = llm_runner
        self.optimizer_model = optimizer_model
        self.max_iterations = max_iterations
        self.target_lint_score = target_lint_score
        self.plateau_threshold = plateau_threshold
        self.linter = BSPLinter()

    async def optimize(
        self,
        bsp: str,
        weak_areas: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize a BSP iteratively.
        
        Args:
            bsp: The BSP text to optimize
            weak_areas: Optional list of weak areas from council evaluation
            model: Override model for optimization
            
        Returns:
            OptimizationResult with original and optimized BSP
        """
        opt_model = model or self.optimizer_model
        if not opt_model:
            raise ValueError("No model specified for optimization. Provide optimizer_model or model parameter.")

        current_bsp = bsp
        api_calls = 0
        steps: list[OptimizationStep] = []

        # Initial lint (free)
        initial_lint = self.linter.lint(current_bsp)
        initial_score = initial_lint.score

        steps.append(OptimizationStep(
            iteration=0,
            bsp_text=current_bsp,
            bsp_hash=self._hash(current_bsp),
            lint_score=initial_lint.score,
            lint_issues_count=len(initial_lint.issues),
            changes_made="Original BSP",
        ))

        # Early exit if already at target
        if initial_lint.score >= self.target_lint_score and not initial_lint.errors:
            return OptimizationResult(
                original_bsp=bsp,
                optimized_bsp=current_bsp,
                original_lint_score=initial_score,
                final_lint_score=initial_lint.score,
                iterations_run=0,
                max_iterations=self.max_iterations,
                steps=steps,
                api_calls_used=0,
                stopped_early=True,
                stop_reason="BSP already meets target lint score",
            )

        previous_score = initial_lint.score

        for iteration in range(1, self.max_iterations + 1):
            # Format issues for prompt
            issues_text = self._format_issues(initial_lint if iteration == 1 else lint_result)
            weak_areas_text = ", ".join(weak_areas) if weak_areas else "None specified"

            # Single API call per iteration
            prompt = self.OPTIMIZE_PROMPT.format(
                bsp=current_bsp,
                issues=issues_text,
                weak_areas=weak_areas_text,
            )

            try:
                result = await self.llm_runner.complete(
                    prompt=prompt,
                    model=opt_model,
                    temperature=0.3,  # Slight creativity for rewrites
                    max_tokens=2000,
                )
                api_calls += 1
                new_bsp = result.text.strip()
            except Exception as e:
                steps.append(OptimizationStep(
                    iteration=iteration,
                    bsp_text=current_bsp,
                    bsp_hash=self._hash(current_bsp),
                    lint_score=previous_score,
                    lint_issues_count=0,
                    changes_made=f"API call failed: {str(e)[:100]}",
                ))
                return OptimizationResult(
                    original_bsp=bsp,
                    optimized_bsp=current_bsp,
                    original_lint_score=initial_score,
                    final_lint_score=previous_score,
                    iterations_run=iteration,
                    max_iterations=self.max_iterations,
                    steps=steps,
                    api_calls_used=api_calls,
                    stopped_early=True,
                    stop_reason=f"API error: {str(e)[:100]}",
                )

            # Strip markdown code fences if model wrapped output
            new_bsp = self._clean_output(new_bsp)

            # Validate the new BSP isn't garbage
            if len(new_bsp) < 20:
                steps.append(OptimizationStep(
                    iteration=iteration,
                    bsp_text=current_bsp,
                    bsp_hash=self._hash(current_bsp),
                    lint_score=previous_score,
                    lint_issues_count=0,
                    changes_made="LLM returned empty/too-short response — skipped",
                ))
                continue

            # Lint the new BSP (free)
            lint_result = self.linter.lint(new_bsp)

            steps.append(OptimizationStep(
                iteration=iteration,
                bsp_text=new_bsp,
                bsp_hash=self._hash(new_bsp),
                lint_score=lint_result.score,
                lint_issues_count=len(lint_result.issues),
                changes_made=f"Score: {previous_score:.2f} → {lint_result.score:.2f}",
            ))

            # Only accept if improved
            if lint_result.score >= previous_score:
                current_bsp = new_bsp
            else:
                # Revert — LLM made it worse
                steps[-1].changes_made += " (REVERTED — score decreased)"

            # Check for target reached
            if lint_result.score >= self.target_lint_score:
                return OptimizationResult(
                    original_bsp=bsp,
                    optimized_bsp=current_bsp,
                    original_lint_score=initial_score,
                    final_lint_score=lint_result.score,
                    iterations_run=iteration,
                    max_iterations=self.max_iterations,
                    steps=steps,
                    api_calls_used=api_calls,
                    stopped_early=True,
                    stop_reason="Target lint score reached",
                )

            # Check for plateau
            improvement = lint_result.score - previous_score
            if abs(improvement) < self.plateau_threshold and iteration > 1:
                return OptimizationResult(
                    original_bsp=bsp,
                    optimized_bsp=current_bsp,
                    original_lint_score=initial_score,
                    final_lint_score=lint_result.score,
                    iterations_run=iteration,
                    max_iterations=self.max_iterations,
                    steps=steps,
                    api_calls_used=api_calls,
                    stopped_early=True,
                    stop_reason=f"Score plateaued (Δ={improvement:.3f})",
                )

            previous_score = max(previous_score, lint_result.score)

            # Rate limiting between iterations
            await asyncio.sleep(3.0)

        # Final lint
        final_lint = self.linter.lint(current_bsp)

        return OptimizationResult(
            original_bsp=bsp,
            optimized_bsp=current_bsp,
            original_lint_score=initial_score,
            final_lint_score=final_lint.score,
            iterations_run=self.max_iterations,
            max_iterations=self.max_iterations,
            steps=steps,
            api_calls_used=api_calls,
        )

    def _format_issues(self, lint_result: LintResult) -> str:
        """Format lint issues for the optimization prompt."""
        lines = []
        for issue in lint_result.issues:
            prefix = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
            lines.append(f"{prefix} [{issue.severity.upper()}] {issue.message}")
            if issue.suggestion:
                lines.append(f"   Fix: {issue.suggestion}")
        return "\n".join(lines) if lines else "No issues found"

    def _clean_output(self, text: str) -> str:
        """Remove markdown code fences if the model wrapped the output."""
        text = text.strip()
        # Remove ```markdown ... ``` or ``` ... ```
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
        return text.strip()

    def _hash(self, text: str) -> str:
        """Generate short hash of text."""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def format_report(self, result: OptimizationResult) -> str:
        """Format optimization result as readable report."""
        lines = []
        lines.append("BSP Optimization Report")
        lines.append("=" * 40)
        lines.append(f"Iterations: {result.iterations_run}/{result.max_iterations}")
        lines.append(f"API Calls Used: {result.api_calls_used}")
        lines.append(f"Lint Score: {result.original_lint_score:.2f} → {result.final_lint_score:.2f} ({'+' if result.improvement >= 0 else ''}{result.improvement:.2f})")

        if result.stopped_early:
            lines.append(f"Early Stop: {result.stop_reason}")
        lines.append("")

        lines.append("Iteration History:")
        for step in result.steps:
            lines.append(f"  [{step.iteration}] Score: {step.lint_score:.2f} | Issues: {step.lint_issues_count} | {step.changes_made}")

        return "\n".join(lines)
