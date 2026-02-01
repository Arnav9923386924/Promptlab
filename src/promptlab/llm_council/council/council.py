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

    # Batch evaluation prompt - evaluates ALL outputs in ONE call
    BATCH_JUDGE_PROMPT = """You are evaluating an LLM's performance across multiple test outputs.

## BEHAVIOR SPECIFICATION (BSP):
{bsp}

## EVALUATION CRITERIA:
1. **Role Adherence**: Does the LLM consistently act according to the BSP?
2. **Response Quality**: Are responses accurate, helpful, and well-formatted?
3. **Consistency**: Are responses consistent across similar questions?
4. **Appropriateness**: Does the LLM stay within its defined scope?

## TEST OUTPUTS TO EVALUATE ({total_tests} total):
{outputs}

## YOUR TASK:
Evaluate the OVERALL performance across ALL outputs. Consider patterns, strengths, and weaknesses.

Respond in this EXACT format:
OVERALL_SCORE: [0.0-1.0]
ROLE_ADHERENCE: [0.0-1.0]
RESPONSE_QUALITY: [0.0-1.0]
CONSISTENCY: [0.0-1.0]
REASONING: [2-3 sentence summary of performance]
WEAK_AREAS: [Comma-separated list of areas needing improvement, or "none"]
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

    # ========================================================================
    # BATCH EVALUATION - Reduces API calls from O(n*m) to O(m)
    # ========================================================================
    
    async def evaluate_batch(
        self,
        outputs: list[dict],
        bsp: str,
        min_score: float = 0.7,
    ) -> "BatchEvaluationResult":
        """Evaluate multiple outputs in a SINGLE batch call per judge.
        
        This dramatically reduces API calls:
        - Old: 30 tests × 3 judges = 90 API calls
        - New: 1 batch × 3 judges = 3 API calls
        
        Args:
            outputs: List of test outputs [{test_id, prompt, response, expected}, ...]
            bsp: The Behavior Specification Prompt being evaluated
            min_score: Minimum score to pass
            
        Returns:
            BatchEvaluationResult with aggregated scores
        """
        # Format all outputs into a single evaluation text
        outputs_text = self._format_batch_outputs(outputs)
        
        # Stage 1: Each judge evaluates the ENTIRE batch in ONE call
        judge_results = await self._batch_stage1_judge(
            outputs_text=outputs_text,
            bsp=bsp,
            total_tests=len(outputs),
        )
        
        # Stage 2: Synthesize final score from all judges
        final_result = self._synthesize_batch_result(judge_results, min_score)
        
        return final_result
    
    def _format_batch_outputs(self, outputs: list[dict], max_outputs: int = 25) -> str:
        """Format outputs for batch evaluation, respecting token limits."""
        formatted = []
        
        # Limit to prevent token overflow (prioritize diversity)
        limited_outputs = outputs[:max_outputs]
        
        for i, out in enumerate(limited_outputs, 1):
            # Truncate long responses to prevent token overflow
            prompt = out.get("prompt", "")[:200]
            response = out.get("response", "")[:400]
            expected = out.get("expected", "")
            
            entry = f"""
### Test {i}: {out.get('test_id', f'test_{i}')}
**Prompt:** {prompt}
**Response:** {response}"""
            
            if expected:
                entry += f"\n**Expected:** {expected[:150]}"
            
            formatted.append(entry)
        
        if len(outputs) > max_outputs:
            formatted.append(f"\n... and {len(outputs) - max_outputs} more tests (showing representative sample)")
        
        return "\n---".join(formatted)
    
    async def _batch_stage1_judge(
        self,
        outputs_text: str,
        bsp: str,
        total_tests: int,
    ) -> list["BatchJudgeScore"]:
        """Stage 1: Each judge evaluates the ENTIRE batch in ONE API call."""
        tasks = []
        
        for model in self.members:
            prompt = self.BATCH_JUDGE_PROMPT.format(
                bsp=bsp[:1500] if bsp else "No BSP specified",
                outputs=outputs_text,
                total_tests=total_tests,
            )
            tasks.append(self._get_batch_judge_score(model, prompt))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scores = []
        for model, result in zip(self.members, results):
            if isinstance(result, Exception):
                scores.append(BatchJudgeScore(
                    model=model,
                    overall_score=0.5,
                    role_adherence=0.5,
                    response_quality=0.5,
                    consistency=0.5,
                    reasoning=f"Error: {result}",
                    weak_areas=[],
                ))
            else:
                scores.append(result)
        
        return scores
    
    async def _get_batch_judge_score(self, model: str, prompt: str) -> "BatchJudgeScore":
        """Get batch score from a single judge (ONE API call for ALL outputs)."""
        result = await self.llm_runner.complete(prompt, model=model, temperature=0, max_tokens=1500)
        
        # Parse response
        score_data = {
            "overall_score": 0.5,
            "role_adherence": 0.5,
            "response_quality": 0.5,
            "consistency": 0.5,
            "reasoning": "",
            "weak_areas": [],
        }
        
        for line in result.text.split("\n"):
            line = line.strip()
            if line.startswith("OVERALL_SCORE:"):
                try:
                    score_data["overall_score"] = max(0.0, min(1.0, float(line.split(":")[-1].strip())))
                except ValueError:
                    pass
            elif line.startswith("ROLE_ADHERENCE:"):
                try:
                    score_data["role_adherence"] = max(0.0, min(1.0, float(line.split(":")[-1].strip())))
                except ValueError:
                    pass
            elif line.startswith("RESPONSE_QUALITY:"):
                try:
                    score_data["response_quality"] = max(0.0, min(1.0, float(line.split(":")[-1].strip())))
                except ValueError:
                    pass
            elif line.startswith("CONSISTENCY:"):
                try:
                    score_data["consistency"] = max(0.0, min(1.0, float(line.split(":")[-1].strip())))
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                score_data["reasoning"] = line.split(":", 1)[-1].strip()
            elif line.startswith("WEAK_AREAS:"):
                areas = line.split(":", 1)[-1].strip()
                if areas.lower() != "none":
                    score_data["weak_areas"] = [a.strip() for a in areas.split(",") if a.strip()]
        
        return BatchJudgeScore(model=model, **score_data)
    
    def _synthesize_batch_result(
        self,
        judge_scores: list["BatchJudgeScore"],
        min_score: float,
    ) -> "BatchEvaluationResult":
        """Synthesize final result from all judges (NO additional API call)."""
        if not judge_scores:
            return BatchEvaluationResult(
                final_score=0.0,
                passed=False,
                confidence="low",
                member_scores=judge_scores,
                summary="No judges available",
                recommendations=[],
            )
        
        # Calculate weighted averages
        total = len(judge_scores)
        final_score = sum(s.overall_score for s in judge_scores) / total
        role_adherence = sum(s.role_adherence for s in judge_scores) / total
        response_quality = sum(s.response_quality for s in judge_scores) / total
        consistency = sum(s.consistency for s in judge_scores) / total
        
        # Determine confidence based on judge agreement
        score_std = self._calculate_std([s.overall_score for s in judge_scores])
        if score_std < 0.1:
            confidence = "high"
        elif score_std < 0.2:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Aggregate weak areas from all judges
        all_weak_areas = []
        for s in judge_scores:
            all_weak_areas.extend(s.weak_areas)
        
        # Count frequency of weak areas
        weak_area_counts = {}
        for area in all_weak_areas:
            weak_area_counts[area] = weak_area_counts.get(area, 0) + 1
        
        # Recommendations = weak areas mentioned by multiple judges
        recommendations = [area for area, count in weak_area_counts.items() if count >= 2]
        if not recommendations and all_weak_areas:
            recommendations = list(set(all_weak_areas))[:3]
        
        # Create summary
        summary = f"Score: {final_score:.2f} | Role: {role_adherence:.2f} | Quality: {response_quality:.2f} | Consistency: {consistency:.2f}"
        
        return BatchEvaluationResult(
            final_score=final_score,
            passed=final_score >= min_score,
            confidence=confidence,
            member_scores=judge_scores,
            summary=summary,
            recommendations=recommendations,
            breakdown={
                "role_adherence": role_adherence,
                "response_quality": response_quality,
                "consistency": consistency,
            },
        )


# ============================================================================
# Batch Evaluation Data Classes
# ============================================================================

class BatchJudgeScore(BaseModel):
    """Score from a single judge for batch evaluation."""
    model: str
    overall_score: float
    role_adherence: float
    response_quality: float
    consistency: float
    reasoning: str
    weak_areas: list[str] = []


class BatchEvaluationResult(BaseModel):
    """Result from batch council evaluation."""
    final_score: float
    passed: bool
    confidence: Literal["high", "medium", "low"]
    member_scores: list[BatchJudgeScore]
    summary: str
    recommendations: list[str] = []
    breakdown: dict = {}
