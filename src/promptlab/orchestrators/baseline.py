"""Baseline management for regression detection and BSP score tracking."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from rich.console import Console

from promptlab.orchestrators.models import TestRun, TestResult

console = Console()


class Baseline(BaseModel):
    """A saved baseline for comparison."""
    tag: str
    run_id: str
    timestamp: str
    results: dict[str, dict]  # case_id -> {status, response_hash, score}


class BSPBaseline(BaseModel):
    """A saved BSP validation baseline."""
    run_id: str
    timestamp: str
    score: float
    bsp_hash: str
    bsp_version: str
    model: str
    total_tests: int
    passed: bool
    confidence: str = "medium"
    tag: Optional[str] = None


class RegressionResult(BaseModel):
    """Result of comparing against a baseline."""
    case_id: str
    regression_type: str  # status, latency, semantic
    baseline_value: str
    current_value: str
    severity: str  # high, medium, low


class BaselineManager:
    """Manages test baselines for regression detection."""
    
    def __init__(self, promptlab_dir: Path):
        """Initialize baseline manager.
        
        Args:
            promptlab_dir: Path to .promptlab directory
        """
        self.baselines_dir = promptlab_dir / "baselines"
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
    
    def create_baseline(self, run: TestRun, tag: str) -> Baseline:
        """Create a baseline from a test run.
        
        Args:
            run: Test run to save as baseline
            tag: Tag name for the baseline
            
        Returns:
            Created Baseline object
        """
        results = {}
        for result in run.results:
            results[result.case_id] = {
                "status": result.status,
                "response_hash": hash(result.response) if result.response else None,
                "latency_ms": result.latency_ms,
                "score": result.council_scores.get("final_score") if result.council_scores else None,
            }
        
        baseline = Baseline(
            tag=tag,
            run_id=run.run_id,
            timestamp=run.timestamp,
            results=results,
        )
        
        # Save to file
        baseline_path = self.baselines_dir / f"{tag}.json"
        baseline_path.write_text(baseline.model_dump_json(indent=2))
        
        return baseline
    
    def load_baseline(self, tag: str) -> Optional[Baseline]:
        """Load a baseline by tag.
        
        Args:
            tag: Tag name of the baseline
            
        Returns:
            Baseline object or None if not found
        """
        baseline_path = self.baselines_dir / f"{tag}.json"
        
        if not baseline_path.exists():
            return None
        
        data = json.loads(baseline_path.read_text())
        return Baseline(**data)
    
    def list_baselines(self) -> list[str]:
        """List all available baseline tags."""
        return [f.stem for f in self.baselines_dir.glob("*.json")]
    
    def delete_baseline(self, tag: str) -> bool:
        """Delete a baseline by tag."""
        baseline_path = self.baselines_dir / f"{tag}.json"
        if baseline_path.exists():
            baseline_path.unlink()
            return True
        return False
    
    def compare(self, run: TestRun, baseline: Baseline) -> list[RegressionResult]:
        """Compare a test run against a baseline.
        
        Args:
            run: Current test run
            baseline: Baseline to compare against
            
        Returns:
            List of regressions found
        """
        regressions = []
        
        for result in run.results:
            baseline_result = baseline.results.get(result.case_id)
            
            if not baseline_result:
                continue  # New test, no comparison
            
            # Status regression (passed -> failed)
            if baseline_result["status"] == "passed" and result.status == "failed":
                regressions.append(RegressionResult(
                    case_id=result.case_id,
                    regression_type="status",
                    baseline_value="passed",
                    current_value="failed",
                    severity="high",
                ))
            
            # Latency regression (>50% increase)
            baseline_latency = baseline_result.get("latency_ms", 0)
            if baseline_latency > 0 and result.latency_ms > baseline_latency * 1.5:
                regressions.append(RegressionResult(
                    case_id=result.case_id,
                    regression_type="latency",
                    baseline_value=f"{baseline_latency}ms",
                    current_value=f"{result.latency_ms}ms",
                    severity="medium",
                ))
            
            # Score regression (significant drop)
            baseline_score = baseline_result.get("score")
            current_score = result.council_scores.get("final_score") if result.council_scores else None
            
            if baseline_score and current_score and current_score < baseline_score - 0.15:
                regressions.append(RegressionResult(
                    case_id=result.case_id,
                    regression_type="score",
                    baseline_value=f"{baseline_score:.2f}",
                    current_value=f"{current_score:.2f}",
                    severity="medium",
                ))
        
        return regressions
    
    # ============ BSP Score Management ============
    
    def get_latest_bsp_baseline(self, tag: Optional[str] = None) -> Optional[BSPBaseline]:
        """Get the most recent BSP validation baseline.
        
        Args:
            tag: Optional tag filter
            
        Returns:
            Latest BSP baseline or None
        """
        baselines = self._load_bsp_baselines()
        
        if tag:
            baselines = [b for b in baselines if b.tag == tag]
        
        if not baselines:
            return None
        
        # Sort by timestamp descending
        baselines.sort(key=lambda b: b.timestamp, reverse=True)
        return baselines[0]
    
    # Alias for backward compatibility
    def get_latest_baseline(self, tag: Optional[str] = None) -> Optional[BSPBaseline]:
        """Alias for get_latest_bsp_baseline."""
        return self.get_latest_bsp_baseline(tag)
    
    def _load_bsp_baselines(self) -> list[BSPBaseline]:
        """Load all BSP baselines from storage."""
        bsp_file = self.baselines_dir / "bsp_baselines.json"
        
        if not bsp_file.exists():
            return []
        
        try:
            with open(bsp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [BSPBaseline(**b) for b in data]
        except (json.JSONDecodeError, KeyError):
            return []
    
    def _save_bsp_baselines(self, baselines: list[BSPBaseline]):
        """Save all BSP baselines to storage."""
        bsp_file = self.baselines_dir / "bsp_baselines.json"
        
        with open(bsp_file, "w", encoding="utf-8") as f:
            json.dump([b.model_dump() for b in baselines], f, indent=2)
    
    def save_bsp_baseline(
        self,
        run_id: str,
        score: float,
        bsp_hash: str,
        bsp_version: str,
        model: str,
        total_tests: int,
        passed: bool,
        confidence: str = "medium",
        tag: Optional[str] = None,
    ) -> BSPBaseline:
        """Save a new BSP validation baseline.
        
        Args:
            run_id: Validation run ID
            score: Council score
            bsp_hash: Hash of the BSP
            bsp_version: BSP version string
            model: Model used
            total_tests: Number of tests run
            passed: Whether validation passed
            confidence: Council confidence level
            tag: Optional tag for this baseline
            
        Returns:
            Created baseline
        """
        baseline = BSPBaseline(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            score=score,
            bsp_hash=bsp_hash,
            bsp_version=bsp_version,
            model=model,
            total_tests=total_tests,
            passed=passed,
            confidence=confidence,
            tag=tag,
        )
        
        baselines = self._load_bsp_baselines()
        baselines.append(baseline)
        self._save_bsp_baselines(baselines)
        
        console.print(f"[green]✓ Saved BSP baseline: {score:.2f} (run: {run_id})[/green]")
        return baseline
    
    def update_bsp_baseline_if_improved(
        self,
        run_id: str,
        score: float,
        bsp_hash: str,
        bsp_version: str,
        model: str,
        total_tests: int,
        passed: bool,
        confidence: str = "medium",
        min_improvement: float = 0.0,
        tag: Optional[str] = None,
    ) -> tuple[bool, Optional[BSPBaseline]]:
        """Update BSP baseline only if score improved.
        
        Args:
            run_id: Validation run ID
            score: New score
            bsp_hash: BSP hash
            bsp_version: BSP version
            model: Model used
            total_tests: Number of tests
            passed: Whether passed
            confidence: Confidence level
            min_improvement: Minimum improvement required
            tag: Optional tag
            
        Returns:
            Tuple of (was_updated, new_baseline)
        """
        current = self.get_latest_bsp_baseline(tag)
        
        if current is None:
            # No baseline yet, save this one
            baseline = self.save_bsp_baseline(
                run_id, score, bsp_hash, bsp_version, model,
                total_tests, passed, confidence, tag
            )
            return True, baseline
        
        improvement = score - current.score
        
        if improvement > min_improvement:
            baseline = self.save_bsp_baseline(
                run_id, score, bsp_hash, bsp_version, model,
                total_tests, passed, confidence, tag
            )
            console.print(f"[green]✓ BSP baseline updated: {current.score:.2f} → {score:.2f} (+{improvement:.2f})[/green]")
            return True, baseline
        else:
            console.print(f"[yellow]BSP baseline not updated: improvement {improvement:.2f} < min required {min_improvement:.2f}[/yellow]")
            return False, None
    
    def compare_bsp_score(
        self,
        score: float,
        tag: Optional[str] = None,
    ) -> dict:
        """Compare a score with the current BSP baseline.
        
        Args:
            score: Score to compare
            tag: Optional baseline tag
            
        Returns:
            Comparison dict with baseline info and improvement
        """
        baseline = self.get_latest_bsp_baseline(tag)
        
        if baseline is None:
            return {
                "has_baseline": False,
                "baseline_score": None,
                "improvement": None,
                "is_improvement": True,
                "recommendation": "No baseline exists. This score will become the first baseline.",
            }
        
        improvement = score - baseline.score
        is_improvement = improvement > 0
        
        if is_improvement:
            recommendation = f"Score improved by {improvement:.2f}. Consider updating baseline."
        elif improvement == 0:
            recommendation = "Score unchanged from baseline."
        else:
            recommendation = f"Score regressed by {abs(improvement):.2f}. BSP changes may have introduced issues."
        
        return {
            "has_baseline": True,
            "baseline_score": baseline.score,
            "baseline_run_id": baseline.run_id,
            "baseline_timestamp": baseline.timestamp,
            "baseline_bsp_version": baseline.bsp_version,
            "improvement": improvement,
            "is_improvement": is_improvement,
            "recommendation": recommendation,
        }
    
    def get_bsp_score_history(self, limit: int = 10, tag: Optional[str] = None) -> list[dict]:
        """Get BSP score history for trending.
        
        Args:
            limit: Maximum number of records
            tag: Optional tag filter
            
        Returns:
            List of score records
        """
        baselines = self._load_bsp_baselines()
        
        if tag:
            baselines = [b for b in baselines if b.tag == tag]
        
        # Sort by timestamp descending
        baselines.sort(key=lambda b: b.timestamp, reverse=True)
        baselines = baselines[:limit]
        
        return [
            {
                "timestamp": b.timestamp,
                "score": b.score,
                "bsp_version": b.bsp_version,
                "passed": b.passed,
                "run_id": b.run_id,
            }
            for b in baselines
        ]
