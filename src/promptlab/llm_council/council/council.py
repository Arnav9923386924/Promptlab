"""LLM Council - Multi-model evaluation with cross-critique and consensus."""

from typing import Optional, Literal
from pydantic import BaseModel
import asyncio

from promptlab.llm_council.llm_runner.runner import LLMRunner, CompletionResult


class JudgeScore(BaseModel):
    """Score from a single judge."""
    model: str
    score: float
    reasoning: str
    passed: bool


class CouncilResult(BaseModel):
    """Result from council evaluation."""
    final_score: float
    passed: bool
    confidence: Literal["high", "medium", "low"]
    member_scores: list[JudgeScore]
    consensus_summary: str
    disagreements: list[str] = []


class Council:
    """LLM Council for multi-model evaluation."""
    
    JUDGE_PROMPT = """You are evaluating an LLM response. Score it from 0.0 to 1.0 based on the criteria.

CRITERIA:
{criteria}

RESPONSE TO EVALUATE:
{response}

Provide your evaluation in this exact format:
SCORE: [0.0-1.0]
REASONING: [Your explanation in 2-3 sentences]
"""

    CRITIQUE_PROMPT = """You are reviewing other judges' evaluations. Consider if their scores are fair.

ORIGINAL RESPONSE:
{response}

CRITERIA:
{criteria}

JUDGE EVALUATIONS:
{evaluations}

Do you agree with these evaluations? Note any concerns briefly.
ASSESSMENT: [Your brief assessment]
"""

    SYNTHESIS_PROMPT = """You are the chairman synthesizing a final verdict.

ORIGINAL RESPONSE:
{response}

CRITERIA:
{criteria}

JUDGE SCORES AND REASONING:
{evaluations}

{critiques}

Synthesize a final verdict:
FINAL_SCORE: [0.0-1.0]
CONFIDENCE: [high/medium/low]
SUMMARY: [1-2 sentence consensus summary]
"""

    def __init__(self, config: dict, llm_runner: LLMRunner):
        """Initialize council.
        
        Args:
            config: Council configuration from promptlab.yaml
            llm_runner: LLM runner instance
        """
        self.config = config
        self.llm_runner = llm_runner
        self.members = config.get("members", [])
        self.chairman = config.get("chairman", self.members[0] if self.members else None)
        self.mode = config.get("mode", "fast")
    
    async def evaluate(
        self,
        response: str,
        criteria: str,
        min_score: float = 0.7,
        mode: Optional[str] = None,
    ) -> CouncilResult:
        """Evaluate a response using the council.
        
        Args:
            response: The LLM response to evaluate
            criteria: Evaluation criteria
            min_score: Minimum score to pass
            mode: Override council mode (full/fast/vote)
            
        Returns:
            CouncilResult with scores and consensus
        """
        mode = mode or self.mode
        
        # Stage 1: Independent judging
        judge_scores = await self._stage1_judge(response, criteria)
        
        if mode == "vote":
            # Simple majority vote
            return self._vote_result(judge_scores, min_score)
        
        if mode == "full":
            # Stage 2: Cross-critique
            critiques = await self._stage2_critique(response, criteria, judge_scores)
        else:
            critiques = ""
        
        # Stage 3: Chairman synthesis
        result = await self._stage3_synthesize(response, criteria, judge_scores, critiques, min_score)
        
        return result
    
    async def _stage1_judge(self, response: str, criteria: str) -> list[JudgeScore]:
        """Stage 1: Each council member judges independently."""
        tasks = []
        for model in self.members:
            prompt = self.JUDGE_PROMPT.format(criteria=criteria, response=response)
            tasks.append(self._get_judge_score(model, prompt))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scores = []
        for model, result in zip(self.members, results):
            if isinstance(result, Exception):
                scores.append(JudgeScore(
                    model=model,
                    score=0.5,
                    reasoning=f"Error: {result}",
                    passed=False,
                ))
            else:
                scores.append(result)
        
        return scores
    
    async def _get_judge_score(self, model: str, prompt: str) -> JudgeScore:
        """Get score from a single judge."""
        result = await self.llm_runner.complete(prompt, model=model, temperature=0)
        
        # Parse response
        score = 0.5
        reasoning = result.text
        
        for line in result.text.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                    score = max(0.0, min(1.0, score))  # Clamp to 0-1
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        return JudgeScore(
            model=model,
            score=score,
            reasoning=reasoning,
            passed=score >= 0.7,
        )
    
    async def _stage2_critique(
        self, response: str, criteria: str, scores: list[JudgeScore]
    ) -> str:
        """Stage 2: Cross-critique (anonymized)."""
        evaluations = "\n".join([
            f"Judge {i+1}: Score {s.score:.2f} - {s.reasoning}"
            for i, s in enumerate(scores)
        ])
        
        prompt = self.CRITIQUE_PROMPT.format(
            response=response,
            criteria=criteria,
            evaluations=evaluations,
        )
        
        # Use chairman for critique synthesis
        result = await self.llm_runner.complete(prompt, model=self.chairman, temperature=0)
        return result.text
    
    async def _stage3_synthesize(
        self,
        response: str,
        criteria: str,
        scores: list[JudgeScore],
        critiques: str,
        min_score: float,
    ) -> CouncilResult:
        """Stage 3: Chairman synthesizes final verdict."""
        evaluations = "\n".join([
            f"- {s.model}: Score {s.score:.2f} - {s.reasoning}"
            for s in scores
        ])
        
        critique_section = f"CRITIQUES:\n{critiques}" if critiques else ""
        
        prompt = self.SYNTHESIS_PROMPT.format(
            response=response,
            criteria=criteria,
            evaluations=evaluations,
            critiques=critique_section,
        )
        
        result = await self.llm_runner.complete(prompt, model=self.chairman, temperature=0)
        
        # Parse synthesis
        final_score = sum(s.score for s in scores) / len(scores) if scores else 0.5
        confidence = "medium"
        summary = result.text
        
        for line in result.text.split("\n"):
            if line.startswith("FINAL_SCORE:"):
                try:
                    final_score = float(line.replace("FINAL_SCORE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("CONFIDENCE:"):
                conf = line.replace("CONFIDENCE:", "").strip().lower()
                if conf in ["high", "medium", "low"]:
                    confidence = conf
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
        
        # Detect disagreements
        disagreements = []
        score_std = self._calculate_std([s.score for s in scores])
        if score_std > 0.15:
            disagreements.append(f"High score variance: {score_std:.2f}")
        
        return CouncilResult(
            final_score=final_score,
            passed=final_score >= min_score,
            confidence=confidence,
            member_scores=scores,
            consensus_summary=summary,
            disagreements=disagreements,
        )
    
    def _vote_result(self, scores: list[JudgeScore], min_score: float) -> CouncilResult:
        """Simple majority vote result."""
        passed_count = sum(1 for s in scores if s.score >= min_score)
        total = len(scores)
        passed = passed_count > total / 2
        
        avg_score = sum(s.score for s in scores) / total if total else 0
        
        return CouncilResult(
            final_score=avg_score,
            passed=passed,
            confidence="medium",
            member_scores=scores,
            consensus_summary=f"{passed_count}/{total} judges passed",
            disagreements=[],
        )
    
    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
