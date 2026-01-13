"""Baseline management for regression detection."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from promptlab.core.models import TestRun, TestResult


class Baseline(BaseModel):
    """A saved baseline for comparison."""
    tag: str
    run_id: str
    timestamp: str
    results: dict[str, dict]  # case_id -> {status, response_hash, score}


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
