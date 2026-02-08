"""BSP Linter - Validates BSP quality offline without API calls.

Checks for common issues like vague language, missing sections,
no examples, and structural problems. This is a FREE pre-check
before running expensive API evaluations.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LintIssue:
    """A single lint issue found in BSP."""
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    suggestion: str = ""


@dataclass
class LintResult:
    """Complete lint result for a BSP."""
    score: float  # 0.0-1.0 quality score
    issues: list[LintIssue] = field(default_factory=list)
    word_count: int = 0
    has_examples: bool = False
    has_constraints: bool = False
    has_format: bool = False
    has_role: bool = False
    ready_for_evaluation: bool = False

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "info"]


class BSPLinter:
    """Validates BSP quality without any API calls (completely free)."""

    # Section keywords to detect
    ROLE_KEYWORDS = [
        "you are", "act as", "your role", "behave as", "assume the role",
        "role:", "## role", "# role",
    ]
    CAPABILITY_KEYWORDS = [
        "capabilities", "can do", "able to", "features", "functions",
        "## capabilities", "# capabilities", "what you do",
    ]
    CONSTRAINT_KEYWORDS = [
        "constraints", "limitations", "don't", "do not", "never",
        "must not", "avoid", "restricted", "## constraints", "# constraints",
        "rules:", "## rules", "# rules",
    ]
    FORMAT_KEYWORDS = [
        "format", "structure", "output", "respond with", "response format",
        "## format", "# format", "template", "## response",
    ]
    EXAMPLE_KEYWORDS = [
        "example", "e.g.", "for instance", "sample", "## example",
        "# example", "user:", "assistant:", "input:", "output:",
    ]

    # Vague terms that should be replaced with specific criteria
    VAGUE_TERMS = {
        "be helpful": "Define specific helpful behaviors (e.g., 'provide step-by-step instructions')",
        "be good": "Specify what 'good' means (e.g., 'respond within 100 words')",
        "do your best": "Define measurable success criteria",
        "try to": "Use definitive language ('always' or 'must' instead of 'try to')",
        "be nice": "Specify tone requirements (e.g., 'use professional, empathetic language')",
        "be professional": "Define professional behavior (e.g., 'use formal tone, avoid slang')",
        "be concise": "Set word/sentence limits (e.g., 'respond in 2-3 sentences')",
        "be accurate": "Specify accuracy requirements (e.g., 'cite sources when stating facts')",
        "as needed": "Define explicit triggers for actions",
        "when appropriate": "Specify exact conditions for when to act",
    }

    # Contradicting instruction patterns
    CONTRADICTION_PAIRS = [
        ("always provide detailed", "keep responses short"),
        ("be concise", "provide comprehensive"),
        ("never refuse", "decline inappropriate"),
        ("answer everything", "stay within scope"),
        ("be brief", "explain thoroughly"),
    ]

    def lint(self, bsp: str) -> LintResult:
        """Analyze BSP for quality issues without any API calls.

        Args:
            bsp: The Behavior Specification Prompt text

        Returns:
            LintResult with score, issues, and metadata
        """
        if not bsp or not bsp.strip():
            return LintResult(
                score=0.0,
                issues=[LintIssue(
                    severity="error",
                    category="structure",
                    message="BSP is empty",
                    suggestion="Provide a behavior specification with role, capabilities, and constraints",
                )],
                ready_for_evaluation=False,
            )

        issues: list[LintIssue] = []
        bsp_lower = bsp.lower()
        word_count = len(bsp.split())

        # --- Structure Checks ---
        has_role = self._check_section(bsp_lower, self.ROLE_KEYWORDS)
        has_constraints = self._check_section(bsp_lower, self.CONSTRAINT_KEYWORDS)
        has_format = self._check_section(bsp_lower, self.FORMAT_KEYWORDS)
        has_examples = self._check_section(bsp_lower, self.EXAMPLE_KEYWORDS)
        has_capabilities = self._check_section(bsp_lower, self.CAPABILITY_KEYWORDS)

        if not has_role:
            issues.append(LintIssue(
                severity="error",
                category="structure",
                message="No role definition found",
                suggestion="Add 'You are a [role]' or a '## Role' section at the beginning",
            ))

        if not has_constraints:
            issues.append(LintIssue(
                severity="warning",
                category="structure",
                message="No constraints or rules defined",
                suggestion="Add a '## Constraints' section defining what the model should NOT do",
            ))

        if not has_format:
            issues.append(LintIssue(
                severity="warning",
                category="structure",
                message="No response format specified",
                suggestion="Add a '## Response Format' section defining expected output structure",
            ))

        if not has_examples:
            issues.append(LintIssue(
                severity="warning",
                category="completeness",
                message="No examples provided",
                suggestion="Add 2-3 input/output examples to guide model behavior",
            ))

        if not has_capabilities:
            issues.append(LintIssue(
                severity="info",
                category="structure",
                message="No explicit capabilities section found",
                suggestion="Add a '## Capabilities' section listing what the model can do",
            ))

        # --- Length Checks ---
        if word_count < 30:
            issues.append(LintIssue(
                severity="error",
                category="length",
                message=f"BSP too short ({word_count} words)",
                suggestion="Minimum recommended: 50 words. Add role, constraints, and examples",
            ))
        elif word_count < 80:
            issues.append(LintIssue(
                severity="warning",
                category="length",
                message=f"BSP quite short ({word_count} words)",
                suggestion="Consider adding more detail. Recommended: 100-500 words",
            ))
        elif word_count > 2000:
            issues.append(LintIssue(
                severity="warning",
                category="length",
                message=f"BSP very long ({word_count} words)",
                suggestion="Consider condensing. Long BSPs can confuse models. Target: 200-800 words",
            ))

        # --- Vague Language Checks ---
        for term, suggestion in self.VAGUE_TERMS.items():
            if term in bsp_lower:
                issues.append(LintIssue(
                    severity="warning",
                    category="clarity",
                    message=f"Vague instruction found: '{term}'",
                    suggestion=suggestion,
                ))

        # --- Contradiction Detection ---
        for pattern_a, pattern_b in self.CONTRADICTION_PAIRS:
            if pattern_a in bsp_lower and pattern_b in bsp_lower:
                issues.append(LintIssue(
                    severity="error",
                    category="contradiction",
                    message=f"Potential contradiction: '{pattern_a}' vs '{pattern_b}'",
                    suggestion="Resolve conflicting instructions. Be specific about when each applies",
                ))

        # --- Edge Case Handling Check ---
        edge_case_indicators = [
            "if the user", "when the input", "in case of",
            "edge case", "error", "invalid", "empty",
            "unknown", "ambiguous", "unclear",
        ]
        has_edge_cases = any(ind in bsp_lower for ind in edge_case_indicators)
        if not has_edge_cases:
            issues.append(LintIssue(
                severity="info",
                category="completeness",
                message="No edge case handling detected",
                suggestion="Add rules for: empty input, ambiguous questions, off-topic requests",
            ))

        # --- Measurability Check ---
        measurable_indicators = [
            r'\d+\s*words?', r'\d+\s*sentences?', r'\d+\s*points?',
            r'maximum\s+\d+', r'minimum\s+\d+', r'at least\s+\d+',
            r'no more than\s+\d+', r'between\s+\d+\s+and\s+\d+',
        ]
        has_measurable = any(re.search(p, bsp_lower) for p in measurable_indicators)
        if not has_measurable:
            issues.append(LintIssue(
                severity="info",
                category="measurability",
                message="No measurable criteria found",
                suggestion="Add quantifiable limits (e.g., 'respond in 2-3 sentences', 'max 200 words')",
            ))

        # --- Calculate Score ---
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])
        info_count = len([i for i in issues if i.severity == "info"])

        score = max(0.0, 1.0 - (error_count * 0.20) - (warning_count * 0.08) - (info_count * 0.03))
        score = round(min(1.0, score), 2)

        return LintResult(
            score=score,
            issues=issues,
            word_count=word_count,
            has_examples=has_examples,
            has_constraints=has_constraints,
            has_format=has_format,
            has_role=has_role,
            ready_for_evaluation=error_count == 0,
        )

    def _check_section(self, bsp_lower: str, keywords: list[str]) -> bool:
        """Check if any of the section keywords are present."""
        return any(kw in bsp_lower for kw in keywords)

    def format_report(self, result: LintResult) -> str:
        """Format lint result as readable report."""
        lines = []
        lines.append(f"BSP Lint Score: {result.score:.2f}/1.00")
        lines.append(f"Word Count: {result.word_count}")
        lines.append(f"Ready for Evaluation: {'Yes' if result.ready_for_evaluation else 'No'}")
        lines.append("")

        sections = {
            "Role Definition": result.has_role,
            "Constraints": result.has_constraints,
            "Response Format": result.has_format,
            "Examples": result.has_examples,
        }
        lines.append("Sections Found:")
        for name, found in sections.items():
            status = "✓" if found else "✗"
            lines.append(f"  {status} {name}")
        lines.append("")

        if result.errors:
            lines.append("ERRORS (must fix):")
            for issue in result.errors:
                lines.append(f"  ✗ [{issue.category}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")

        if result.warnings:
            lines.append("\nWARNINGS (should fix):")
            for issue in result.warnings:
                lines.append(f"  ⚠ [{issue.category}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")

        if result.infos:
            lines.append("\nINFO (consider):")
            for issue in result.infos:
                lines.append(f"  ℹ [{issue.category}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")

        return "\n".join(lines)
