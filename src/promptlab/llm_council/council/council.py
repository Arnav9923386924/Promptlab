"""LLM Council - Multi-model evaluation with cross-critique and consensus."""

from typing import Optional, Literal
from pydantic import BaseModel
import asyncio

from promptlab.llm_council.llm_runner.runner import LLMRunner, CompletionResult
from promptlab.utils.model_pool import ModelPool


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
    BATCH_JUDGE_PROMPT = """You are a strict evaluation judge. Your task is to score an LLM's responses against a Behavior Specification Prompt (BSP).

## BSP (the rules the LLM must follow):
{bsp}

## SCORING RUBRIC:
For each dimension, use this scale:
- 0.9-1.0: Excellent — fully meets BSP requirements, no issues
- 0.7-0.89: Good — mostly follows BSP with minor gaps
- 0.5-0.69: Fair — partially follows BSP, notable weaknesses
- 0.3-0.49: Poor — significant deviations from BSP
- 0.0-0.29: Failing — does not follow BSP at all

## DIMENSIONS:
1. ROLE_ADHERENCE: Does every response stay in the role defined by the BSP?
2. RESPONSE_QUALITY: Are responses accurate, complete, and well-structured?
3. CONSISTENCY: Do similar prompts get similar-quality responses?
4. CONSTRAINT_COMPLIANCE: Does the LLM respect all constraints/rules in the BSP?

## TEST OUTPUTS ({total_tests} total):
{outputs}

## IMPORTANT:
- Score each dimension independently
- OVERALL_SCORE should be a weighted average: Role 30%, Quality 30%, Consistency 20%, Constraints 20%
- Be specific in reasoning — cite test numbers where issues appear
- List concrete weak areas, not generic ones

Respond in EXACTLY this format (one per line, no extra text before or after):
OVERALL_SCORE: [number between 0.0 and 1.0]
ROLE_ADHERENCE: [number between 0.0 and 1.0]
RESPONSE_QUALITY: [number between 0.0 and 1.0]
CONSISTENCY: [number between 0.0 and 1.0]
CONSTRAINT_COMPLIANCE: [number between 0.0 and 1.0]
REASONING: [2-3 sentences citing specific test numbers]
WEAK_AREAS: [comma-separated list, or "none"]
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

    def __init__(self, config: dict, llm_runner: LLMRunner,
                 openrouter_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """Initialize council.
        
        Args:
            config: Council configuration from promptlab.yaml
            llm_runner: LLM runner instance
            openrouter_api_key: OpenRouter API key for dynamic model pool discovery
            google_api_key: Google AI Studio API key for Gemini model discovery
        """
        self.config = config
        self.llm_runner = llm_runner
        self.members = config.get("members", [])
        self.chairman = config.get("chairman", self.members[0] if self.members else None)
        self.mode = config.get("mode", "fast")
        
        # Dynamic model pool — auto-discovers free models for judges
        # Chairman stays static from config; only judges use the pool
        if openrouter_api_key or google_api_key:
            self.model_pool = ModelPool(
                openrouter_api_key=openrouter_api_key or "",
                google_api_key=google_api_key or "",
            )
        else:
            self.model_pool = None
    
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
        """Stage 1: Each council member judges independently.
        
        Uses dynamic model pool when available:
        - Tries preferred members first, then discovered free models
        - Automatically skips rate-limited models and tries the next
        - Falls back to config members if pool is unavailable
        """
        from rich.console import Console
        console = Console()

        # Get judge models from pool (preferred + discovered free models)
        judge_models = await self._get_judge_model_list()
        target_judges = max(len(self.members), 2)
        prompt = self.JUDGE_PROMPT.format(criteria=criteria, response=response)

        scores = []
        for model in judge_models:
            if len(scores) >= target_judges:
                break
            
            try:
                score = await self._get_judge_score_with_retry(model, prompt)
                scores.append(score)
                if self.model_pool:
                    self.model_pool.mark_used(model)
            except Exception as e:
                error_msg = str(e)
                model_short = model.split('/')[-1][:20]
                
                if self._is_rate_limit_error(error_msg):
                    if self.model_pool:
                        self.model_pool.mark_rate_limited(model)
                    console.print(f"[yellow]  ⚠ {model_short} rate-limited — trying next model[/yellow]")
                else:
                    console.print(f"[yellow]  ⚠ {model_short} failed: {error_msg[:60]}[/yellow]")
                continue

        if not scores:
            console.print("[red]  ✗ All judges failed. Returning fallback score.[/red]")
            scores.append(JudgeScore(
                model="fallback",
                score=0.5,
                reasoning="All council judges failed. Score is a fallback placeholder.",
                passed=False,
            ))

        # Variance warning
        if len(scores) >= 2:
            score_values = [s.score for s in scores]
            std = self._calculate_std(score_values)
            if std > 0.20:
                console.print(f"[yellow]  ⚠ High judge disagreement (σ={std:.2f}). Scores: {[f'{s.score:.2f}' for s in scores]}[/yellow]")
                console.print(f"[yellow]    Consider using 'full' mode for cross-critique to resolve.[/yellow]")

        return scores

    async def _get_judge_score_with_retry(self, model: str, prompt: str, max_retries: int = 1) -> JudgeScore:
        """Get judge score with retry on failure."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._get_judge_score(model, prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(2.0 * (attempt + 1))  # Backoff
        raise last_error
    
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

    async def _get_judge_model_list(self, extra_fallbacks: int = 0) -> list[str]:
        """Get ordered list of ALL judge models: preferred first, then pool by param count.
        
        Initializes the model pool on first call. If pool is unavailable,
        returns config members only. No cap on count — returns every available
        model so the loop can keep trying until one works.
            
        Returns:
            Full list of model IDs to try in order (highest capability first)
        """
        if self.model_pool:
            if not self.model_pool.initialized:
                await self.model_pool.initialize()
            
            if self.model_pool.initialized:
                return self.model_pool.get_available_judges(
                    preferred=self.members,
                )
        
        # Fallback: just use config members
        return list(self.members)

    @staticmethod
    def _is_rate_limit_error(error_msg: str) -> bool:
        """Check if an error is a rate limit error (works for all providers).
        
        Detects:
        - OpenRouter/OpenAI: 429, rate limit, too many requests
        - Google Gemini: RESOURCE_EXHAUSTED, quota exceeded
        - Generic: quota, exhausted
        """
        indicators = [
            "429", "rate limit", "rate_limit", "too many requests",
            "quota", "resource_exhausted", "exhausted",
        ]
        error_lower = error_msg.lower()
        return any(ind in error_lower for ind in indicators)
    
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
        
        API call optimization:
        - Batch: 1 call per judge (not 1 per test)
        - Early agreement: if first 2 judges agree (σ < 0.06), skip remaining
        - Result: typically 2-3 API calls for full evaluation
        
        Args:
            outputs: List of test outputs [{test_id, prompt, response, expected}, ...]
            bsp: The Behavior Specification Prompt being evaluated
            min_score: Minimum score to pass
            
        Returns:
            BatchEvaluationResult with aggregated scores
        """
        from rich.console import Console
        console = Console()
        
        # Format all outputs into a single evaluation text
        outputs_text = self._format_batch_outputs(outputs)
        
        prompt = self.BATCH_JUDGE_PROMPT.format(
            bsp=bsp[:1500] if bsp else "No BSP specified",
            outputs=outputs_text,
            total_tests=len(outputs),
        )
        
        # Get judge models from pool (preferred + discovered free models)
        judge_models = await self._get_judge_model_list()
        target_judges = max(len(self.members), 2)
        
        # Sequential judging with dynamic fallback and early-agreement optimization
        judge_results: list[BatchJudgeScore] = []
        
        for model in judge_models:
            if len(judge_results) >= target_judges:
                break
            
            try:
                score = await self._get_batch_judge_score_with_retry(model, prompt)
                judge_results.append(score)
                if self.model_pool:
                    self.model_pool.mark_used(model)
            except Exception as e:
                error_msg = str(e)
                model_short = model.split('/')[-1][:20]
                
                if self._is_rate_limit_error(error_msg):
                    if self.model_pool:
                        self.model_pool.mark_rate_limited(model)
                    console.print(f"[yellow]  ⚠ {model_short} rate-limited — trying next model[/yellow]")
                    # No delay — next model has a SEPARATE rate limit
                else:
                    console.print(f"[red]  ✗ {model_short} failed: {error_msg[:60]}[/red]")
                continue
            
            # Early agreement check: if 2+ judges agree closely, skip the rest
            if len(judge_results) >= 2:
                scores_so_far = [s.overall_score for s in judge_results]
                std = self._calculate_std(scores_so_far)
                if std < 0.06:
                    console.print(f"[green]  ✓ Early agreement (σ={std:.3f}) — skipping remaining judges[/green]")
                    break
            
            # Brief pause between successful calls (shared API key quota)
            await asyncio.sleep(1.0)
        
        # Fallback if all failed
        if not judge_results:
            console.print("[red]  ✗ All judges failed. Returning fallback score.[/red]")
            judge_results.append(BatchJudgeScore(
                model="fallback",
                overall_score=0.5,
                role_adherence=0.5,
                response_quality=0.5,
                consistency=0.5,
                constraint_compliance=0.5,
                reasoning="All council judges failed.",
                weak_areas=[],
            ))
        
        # Synthesize final score
        final_result = self._synthesize_batch_result(judge_results, min_score)
        
        pool_info = ""
        if self.model_pool and self.model_pool.initialized:
            stats = self.model_pool.get_pool_stats()
            pool_info = f" | pool: {stats['total_discovered']} free models, {stats['rate_limited']} rate-limited"
        console.print(f"[dim]  Judges used: {len(judge_results)}{pool_info}[/dim]")
        
        return final_result
    
    def _format_batch_outputs(self, outputs: list[dict], max_outputs: int = 25) -> str:
        """Format outputs for batch evaluation with smart sampling.
        
        Instead of taking the first N outputs, samples to ensure diversity:
        - Always includes error/failed responses (most informative)
        - Includes shortest and longest responses (edge cases)
        - Random sample of remaining for coverage
        """
        if len(outputs) <= max_outputs:
            selected = outputs
        else:
            selected = self._sample_diverse_outputs(outputs, max_outputs)
        
        formatted = []
        for i, out in enumerate(selected, 1):
            prompt = out.get("prompt", "")[:200]
            response = out.get("response", "")[:400]
            expected = out.get("expected", "")
            test_id = out.get('test_id', f'test_{i}')
            
            entry = f"[Test {i}: {test_id}]\nPrompt: {prompt}\nResponse: {response}"
            if expected:
                entry += f"\nExpected: {expected[:150]}"
            formatted.append(entry)
        
        if len(outputs) > max_outputs:
            formatted.append(f"\n({len(outputs) - max_outputs} additional tests not shown — above is a representative sample)")
        
        return "\n---\n".join(formatted)

    def _sample_diverse_outputs(self, outputs: list[dict], max_outputs: int) -> list[dict]:
        """Select a diverse sample of outputs for evaluation."""
        import random
        
        selected = []
        remaining = list(outputs)
        
        # 1. Always include error responses (most informative for evaluation)
        errors = [o for o in remaining if o.get("response", "").startswith("ERROR:")]
        for e in errors[:3]:
            selected.append(e)
            remaining.remove(e)
        
        # 2. Include shortest and longest responses (edge cases)
        if remaining:
            remaining_sorted = sorted(remaining, key=lambda o: len(o.get("response", "")))
            for edge in [remaining_sorted[0], remaining_sorted[-1]]:
                if edge not in selected:
                    selected.append(edge)
                    remaining.remove(edge)
        
        # 3. Fill rest with evenly-spaced sample for coverage
        slots = max_outputs - len(selected)
        if slots > 0 and remaining:
            step = max(1, len(remaining) // slots)
            for idx in range(0, len(remaining), step):
                if len(selected) >= max_outputs:
                    break
                selected.append(remaining[idx])
        
        return selected[:max_outputs]

    async def _get_batch_judge_score_with_retry(self, model: str, prompt: str, max_retries: int = 1) -> "BatchJudgeScore":
        """Get batch judge score with retry on failure."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._get_batch_judge_score(model, prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(3.0 * (attempt + 1))
        raise last_error
    
    async def _get_batch_judge_score(self, model: str, prompt: str) -> "BatchJudgeScore":
        """Get batch score from a single judge (ONE API call for ALL outputs).
        
        Uses robust parsing with multiple fallback strategies:
        1. Exact format matching (OVERALL_SCORE: 0.85)
        2. Regex patterns for variations
        3. Weighted sentiment-based estimation
        """
        import re
        from rich.console import Console
        console = Console()
        
        result = await self.llm_runner.complete(prompt, model=model, temperature=0, max_tokens=1500)
        text = result.text
        
        # Parse all score fields
        score_data = self._parse_score_fields(text)
        
        # Log concise summary
        model_short = model.split('/')[-1][:25]
        if score_data["overall_score"] is not None:
            console.print(f"[green]  ✓ {model_short}: {score_data['overall_score']:.2f} (R:{score_data.get('role_adherence', '?')} Q:{score_data.get('response_quality', '?')} C:{score_data.get('consistency', '?')})[/green]")
        else:
            console.print(f"[yellow]  ⚠ {model_short}: no structured score — using fallback parsing[/yellow]")
        
        # Fill missing overall score from sub-scores or fallback
        if score_data["overall_score"] is None:
            sub_scores = [score_data[k] for k in ["role_adherence", "response_quality", "consistency", "constraint_compliance"] if score_data.get(k) is not None]
            if sub_scores:
                score_data["overall_score"] = sum(sub_scores) / len(sub_scores)
            else:
                score_data["overall_score"] = self._estimate_score_from_text(text)
        
        # Fill missing sub-scores from overall
        final_overall = score_data["overall_score"] or 0.5
        for key in ["role_adherence", "response_quality", "consistency", "constraint_compliance"]:
            if score_data.get(key) is None:
                score_data[key] = final_overall
        
        # Extract reasoning if not found
        if not score_data.get("reasoning"):
            score_data["reasoning"] = self._extract_reasoning(text)
        
        return BatchJudgeScore(
            model=model,
            overall_score=score_data["overall_score"],
            role_adherence=score_data["role_adherence"],
            response_quality=score_data["response_quality"],
            consistency=score_data["consistency"],
            constraint_compliance=score_data.get("constraint_compliance", final_overall),
            reasoning=score_data.get("reasoning", ""),
            weak_areas=score_data.get("weak_areas", []),
        )

    def _parse_score_fields(self, text: str) -> dict:
        """Parse all score fields from response text using multi-strategy approach."""
        import re
        
        field_map = {
            "overall_score": ["OVERALL_SCORE", "OVERALL SCORE", "FINAL_SCORE", "FINAL SCORE"],
            "role_adherence": ["ROLE_ADHERENCE", "ROLE ADHERENCE"],
            "response_quality": ["RESPONSE_QUALITY", "RESPONSE QUALITY"],
            "consistency": ["CONSISTENCY"],
            "constraint_compliance": ["CONSTRAINT_COMPLIANCE", "CONSTRAINT COMPLIANCE", "APPROPRIATENESS"],
        }
        
        result = {k: None for k in field_map}
        result["reasoning"] = ""
        result["weak_areas"] = []
        
        for line in text.split("\n"):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Parse score fields
            for field, prefixes in field_map.items():
                for prefix in prefixes:
                    if line_upper.startswith(prefix + ":"):
                        match = re.search(r'[\d.]+', line_stripped.split(":", 1)[-1])
                        if match:
                            val = float(match.group())
                            if 0 <= val <= 1:
                                result[field] = val
                            elif 1 < val <= 10:
                                result[field] = val / 10
                            elif 10 < val <= 100:
                                result[field] = val / 100
                        break
            
            # Parse text fields
            if line_upper.startswith("REASONING:"):
                result["reasoning"] = line_stripped.split(":", 1)[-1].strip()
            elif line_upper.startswith("WEAK_AREAS:") or line_upper.startswith("WEAK AREAS:"):
                areas = line_stripped.split(":", 1)[-1].strip()
                if areas.lower() != "none":
                    result["weak_areas"] = [a.strip() for a in areas.split(",") if a.strip()]
        
        # Regex fallback for overall score if not found
        if result["overall_score"] is None:
            patterns = [
                r'(?:overall|final|total)[\s_]*(?:score)?[\s:=]+([0-9.]+)',
                r'(?:score|rating)[\s:=]+([0-9.]+)(?:\s*/\s*1)?',
                r'\b([0-9]\.[0-9]+)\s*/\s*1(?:\.0)?',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if 0 <= val <= 1:
                        result["overall_score"] = val
                        break
                    elif 1 < val <= 10:
                        result["overall_score"] = val / 10
                        break
        
        return result

    def _estimate_score_from_text(self, text: str) -> float:
        """Last-resort: estimate score from sentiment words in the response."""
        text_lower = text.lower()
        positive = ['excellent', 'good', 'well', 'correct', 'accurate', 'helpful',
                    'clear', 'proper', 'appropriate', 'follows', 'adheres', 'compliant',
                    'strong', 'consistent', 'thorough']
        negative = ['poor', 'bad', 'wrong', 'incorrect', 'missing', 'fails',
                    'violates', 'lacks', 'inadequate', 'inconsistent', 'weak',
                    'off-topic', 'ignores', 'deviates']
        
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        
        if pos + neg > 0:
            return max(0.2, min(0.9, 0.5 + 0.3 * (pos - neg) / (pos + neg + 1)))
        return 0.5

    def _extract_reasoning(self, text: str) -> str:
        """Extract a reasoning sentence from unstructured text."""
        for sent in text.split("."):
            if any(w in sent.lower() for w in ["overall", "summary", "conclusion", "performance", "evaluation"]):
                return sent.strip()[:300]
        return text[:200].strip()
    
    def _synthesize_batch_result(
        self,
        judge_scores: list["BatchJudgeScore"],
        min_score: float,
    ) -> "BatchEvaluationResult":
        """Synthesize final result from all judges (NO additional API call).
        
        Uses weighted dimension scoring (matches prompt rubric):
        - Role Adherence: 30%
        - Response Quality: 30%
        - Consistency: 20%
        - Constraint Compliance: 20%
        
        Also uses outlier-resistant median for final score when judges disagree.
        """
        if not judge_scores:
            return BatchEvaluationResult(
                final_score=0.0,
                passed=False,
                confidence="low",
                member_scores=judge_scores,
                summary="No judges available",
                recommendations=[],
            )
        
        total = len(judge_scores)
        
        # Dimension averages
        role_adherence = sum(s.role_adherence for s in judge_scores) / total
        response_quality = sum(s.response_quality for s in judge_scores) / total
        consistency = sum(s.consistency for s in judge_scores) / total
        constraint_compliance = sum(s.constraint_compliance for s in judge_scores) / total
        
        # Weighted final score from dimensions (not just averaging overall_score)
        weighted_score = (
            role_adherence * 0.30
            + response_quality * 0.30
            + consistency * 0.20
            + constraint_compliance * 0.20
        )
        
        # Use median of judge overall_scores if they diverge significantly
        overall_scores = sorted([s.overall_score for s in judge_scores])
        score_std = self._calculate_std(overall_scores)
        if score_std > 0.15 and total >= 3:
            # High disagreement — use median to resist outliers
            median_score = overall_scores[total // 2]
            final_score = (median_score + weighted_score) / 2
        else:
            final_score = (sum(overall_scores) / total + weighted_score) / 2
        
        final_score = round(final_score, 4)
        
        # Confidence from per-dimension agreement
        dimension_stds = [
            self._calculate_std([s.role_adherence for s in judge_scores]),
            self._calculate_std([s.response_quality for s in judge_scores]),
            self._calculate_std([s.consistency for s in judge_scores]),
            self._calculate_std([s.constraint_compliance for s in judge_scores]),
        ]
        avg_std = sum(dimension_stds) / len(dimension_stds)
        if avg_std < 0.08:
            confidence = "high"
        elif avg_std < 0.18:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Aggregate weak areas — rank by frequency
        weak_area_counts: dict[str, int] = {}
        for s in judge_scores:
            for area in s.weak_areas:
                area_normalized = area.strip().lower()
                weak_area_counts[area_normalized] = weak_area_counts.get(area_normalized, 0) + 1
        
        # Recommendations: areas flagged by 2+ judges first, then singles
        sorted_areas = sorted(weak_area_counts.items(), key=lambda x: -x[1])
        recommendations = [area for area, count in sorted_areas if count >= 2]
        if not recommendations:
            recommendations = [area for area, _ in sorted_areas[:3]]
        
        # Find the weakest dimension for actionable feedback
        dimension_scores = {
            "role_adherence": role_adherence,
            "response_quality": response_quality,
            "consistency": consistency,
            "constraint_compliance": constraint_compliance,
        }
        weakest = min(dimension_scores, key=dimension_scores.get)
        weakest_val = dimension_scores[weakest]
        
        summary = (
            f"Score: {final_score:.2f} | "
            f"Role: {role_adherence:.2f} | Quality: {response_quality:.2f} | "
            f"Consistency: {consistency:.2f} | Constraints: {constraint_compliance:.2f}"
        )
        if weakest_val < 0.7:
            summary += f" | ⚠ Weakest: {weakest.replace('_', ' ')} ({weakest_val:.2f})"
        
        return BatchEvaluationResult(
            final_score=final_score,
            passed=final_score >= min_score,
            confidence=confidence,
            member_scores=judge_scores,
            summary=summary,
            recommendations=recommendations,
            breakdown={
                "role_adherence": round(role_adherence, 4),
                "response_quality": round(response_quality, 4),
                "consistency": round(consistency, 4),
                "constraint_compliance": round(constraint_compliance, 4),
                "weakest_dimension": weakest,
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
    constraint_compliance: float = 0.5
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
