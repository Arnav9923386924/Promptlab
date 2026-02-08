"""Evaluation History Tracker - Tracks scores over time locally.

Stores evaluation results in a local JSON file for:
- Trend analysis
- Regression detection
- Score comparison across BSP versions
- No external dependencies or API calls needed
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HistoryEntry:
    """A single evaluation history entry."""
    timestamp: str
    bsp_version: str
    bsp_hash: str
    model: str
    overall_score: float
    role_adherence: float = 0.0
    response_quality: float = 0.0
    consistency: float = 0.0
    confidence: str = "medium"
    total_tests: int = 0
    weak_areas: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class RegressionAlert:
    """Alert when regression is detected."""
    detected: bool
    metric: str
    current_value: float
    previous_value: float
    delta: float
    message: str


@dataclass
class TrendReport:
    """Score trend over time."""
    entries: list[HistoryEntry]
    trend_direction: str  # "improving", "declining", "stable"
    average_score: float
    best_score: float
    worst_score: float
    total_evaluations: int


class EvaluationHistory:
    """Tracks evaluation results over time in a local JSON file.
    
    Completely free - no API calls, stores everything locally.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize history tracker.
        
        Args:
            project_root: Project root directory (default: current directory)
        """
        self.project_root = project_root or Path.cwd()
        self.history_dir = self.project_root / ".promptlab"
        self.history_file = self.history_dir / "evaluation_history.json"
        self._history: list[HistoryEntry] = []
        self._load()

    def _load(self):
        """Load history from disk."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                self._history = [
                    HistoryEntry(**entry)
                    for entry in data.get("evaluations", [])
                ]
            except (json.JSONDecodeError, TypeError):
                self._history = []
        else:
            self._history = []

    def _save(self):
        """Persist history to disk."""
        self.history_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "total_evaluations": len(self._history),
            "evaluations": [asdict(e) for e in self._history],
        }
        self.history_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def record(
        self,
        overall_score: float,
        bsp_version: str = "unknown",
        bsp_hash: str = "",
        model: str = "",
        role_adherence: float = 0.0,
        response_quality: float = 0.0,
        consistency: float = 0.0,
        confidence: str = "medium",
        total_tests: int = 0,
        weak_areas: Optional[list[str]] = None,
        notes: str = "",
    ):
        """Record an evaluation result.
        
        Args:
            overall_score: The overall evaluation score (0.0-1.0)
            bsp_version: Version of the BSP
            bsp_hash: Hash of the BSP content
            model: Model being evaluated
            role_adherence: Role adherence score
            response_quality: Response quality score
            consistency: Consistency score
            confidence: Confidence level
            total_tests: Number of tests run
            weak_areas: List of identified weak areas
            notes: Optional notes
        """
        entry = HistoryEntry(
            timestamp=datetime.now().isoformat(),
            bsp_version=bsp_version,
            bsp_hash=bsp_hash,
            model=model,
            overall_score=round(overall_score, 4),
            role_adherence=round(role_adherence, 4),
            response_quality=round(response_quality, 4),
            consistency=round(consistency, 4),
            confidence=confidence,
            total_tests=total_tests,
            weak_areas=weak_areas or [],
            notes=notes,
        )
        self._history.append(entry)
        self._save()

    def detect_regression(self, threshold: float = 0.05) -> list[RegressionAlert]:
        """Compare latest result with previous to detect regression.
        
        Args:
            threshold: Minimum score drop to trigger alert
            
        Returns:
            List of regression alerts (empty if no regression)
        """
        if len(self._history) < 2:
            return []

        current = self._history[-1]
        previous = self._history[-2]
        alerts = []

        metrics = {
            "overall_score": (current.overall_score, previous.overall_score),
            "role_adherence": (current.role_adherence, previous.role_adherence),
            "response_quality": (current.response_quality, previous.response_quality),
            "consistency": (current.consistency, previous.consistency),
        }

        for metric, (curr_val, prev_val) in metrics.items():
            delta = curr_val - prev_val
            if delta < -threshold:
                alerts.append(RegressionAlert(
                    detected=True,
                    metric=metric,
                    current_value=curr_val,
                    previous_value=prev_val,
                    delta=round(delta, 4),
                    message=(
                        f"{metric} dropped by {abs(delta):.2f} "
                        f"({prev_val:.2f} → {curr_val:.2f})"
                    ),
                ))

        return alerts

    def get_trend(self, last_n: int = 10) -> TrendReport:
        """Get score trend over recent evaluations.
        
        Args:
            last_n: Number of recent evaluations to analyze
            
        Returns:
            TrendReport with trend analysis
        """
        recent = self._history[-last_n:] if self._history else []

        if not recent:
            return TrendReport(
                entries=[],
                trend_direction="stable",
                average_score=0.0,
                best_score=0.0,
                worst_score=0.0,
                total_evaluations=0,
            )

        scores = [e.overall_score for e in recent]
        avg_score = sum(scores) / len(scores)

        # Determine trend direction
        if len(scores) >= 3:
            first_half = scores[:len(scores) // 2]
            second_half = scores[len(scores) // 2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            if second_avg - first_avg > 0.03:
                direction = "improving"
            elif first_avg - second_avg > 0.03:
                direction = "declining"
            else:
                direction = "stable"
        else:
            direction = "stable"

        return TrendReport(
            entries=recent,
            trend_direction=direction,
            average_score=round(avg_score, 4),
            best_score=round(max(scores), 4),
            worst_score=round(min(scores), 4),
            total_evaluations=len(self._history),
        )

    def get_history(self) -> list[HistoryEntry]:
        """Get full evaluation history."""
        return list(self._history)

    def get_latest(self) -> Optional[HistoryEntry]:
        """Get most recent evaluation entry."""
        return self._history[-1] if self._history else None

    def compare_bsp_versions(self) -> dict[str, dict]:
        """Compare scores across different BSP versions.
        
        Returns:
            Dict mapping bsp_version to average scores
        """
        versions: dict[str, list[HistoryEntry]] = {}
        for entry in self._history:
            if entry.bsp_version not in versions:
                versions[entry.bsp_version] = []
            versions[entry.bsp_version].append(entry)

        comparison = {}
        for version, entries in versions.items():
            scores = [e.overall_score for e in entries]
            comparison[version] = {
                "average_score": round(sum(scores) / len(scores), 4),
                "best_score": round(max(scores), 4),
                "evaluations": len(entries),
                "latest_timestamp": entries[-1].timestamp,
            }

        return comparison

    def format_report(self) -> str:
        """Format history as readable report."""
        trend = self.get_trend()
        lines = []

        lines.append(f"Evaluation History ({trend.total_evaluations} total evaluations)")
        lines.append(f"Trend: {trend.trend_direction}")
        lines.append(f"Average Score: {trend.average_score:.2f}")
        lines.append(f"Best: {trend.best_score:.2f} | Worst: {trend.worst_score:.2f}")
        lines.append("")

        # Recent entries
        lines.append("Recent Evaluations:")
        for entry in trend.entries[-5:]:
            ts = entry.timestamp[:19]  # Trim microseconds
            lines.append(
                f"  {ts} | Score: {entry.overall_score:.2f} | "
                f"BSP: {entry.bsp_version} | Model: {entry.model}"
            )

        # Regressions
        alerts = self.detect_regression()
        if alerts:
            lines.append("\nRegression Alerts:")
            for alert in alerts:
                lines.append(f"  ⚠ {alert.message}")

        return "\n".join(lines)
