"""Multi-Turn Conversation Evaluator - Evaluate conversation threads.

Tests multi-turn conversations for:
- Context retention across turns
- Role consistency throughout dialogue
- Personality drift detection
- Response coherence

Optimized for free API usage:
- Sends entire conversation as ONE batch to council
- Uses local analysis for drift detection (free)
- Single evaluation call per conversation
"""

import asyncio
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Literal

from promptlab.llm_council.llm_runner.runner import LLMRunner


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    turn_number: int = 0


@dataclass
class TurnAnalysis:
    """Analysis of a single turn."""
    turn_number: int
    role_consistent: bool
    context_referenced: bool
    tone_shift_detected: bool
    notes: str = ""


@dataclass
class ConversationEvaluation:
    """Complete evaluation of a multi-turn conversation."""
    overall_score: float
    context_retention_score: float
    role_consistency_score: float
    coherence_score: float
    personality_drift_score: float  # 1.0 = no drift, 0.0 = full drift
    turn_analyses: list[TurnAnalysis] = field(default_factory=list)
    total_turns: int = 0
    drift_detected_at: list[int] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    api_calls_used: int = 0


class MultiTurnEvaluator:
    """Evaluates multi-turn conversations for quality and consistency.
    
    Cost optimization strategy:
    1. Local analysis (FREE): tone markers, context references, length patterns
    2. Single batch API call: entire conversation evaluated in one call
    3. Optional per-turn deep analysis only when flagged by local analysis
    """

    CONVERSATION_EVAL_PROMPT = """You are evaluating a multi-turn conversation between a user and an AI assistant.

## BSP (Expected Behavior):
{bsp}

## CONVERSATION ({total_turns} turns):
{conversation}

## EVALUATION CRITERIA:
1. **Context Retention**: Does the assistant remember and reference previous turns?
2. **Role Consistency**: Does the assistant stay in character per the BSP?
3. **Response Coherence**: Are responses logically connected to the conversation flow?
4. **Personality Drift**: Does the assistant's tone/behavior change unexpectedly?

## RESPOND IN THIS EXACT FORMAT:
CONTEXT_RETENTION: [0.0-1.0]
ROLE_CONSISTENCY: [0.0-1.0]
COHERENCE: [0.0-1.0]
PERSONALITY_DRIFT: [0.0-1.0] (1.0 = no drift)
DRIFT_TURNS: [comma-separated turn numbers where drift occurred, or "none"]
SUMMARY: [2-3 sentence evaluation]
RECOMMENDATIONS: [comma-separated list, or "none"]
"""

    # Common tone markers for local drift detection
    FORMAL_MARKERS = [
        "therefore", "consequently", "furthermore", "nevertheless",
        "however", "pursuant", "accordingly", "hereby",
    ]
    INFORMAL_MARKERS = [
        "lol", "haha", "gonna", "wanna", "kinda", "btw",
        "omg", "yeah", "nope", "awesome", "cool",
    ]

    def __init__(
        self,
        llm_runner: LLMRunner,
        config: Optional[dict] = None,
    ):
        """Initialize multi-turn evaluator.
        
        Args:
            llm_runner: LLM runner for API calls
            config: Optional configuration
        """
        self.llm_runner = llm_runner
        self.config = config or {}
        self.rate_limit_delay = self.config.get("rate_limit_delay", 3.0)

    async def evaluate_conversation(
        self,
        turns: list[ConversationTurn],
        bsp: str,
        evaluator_model: str,
    ) -> ConversationEvaluation:
        """Evaluate a multi-turn conversation.
        
        Cost: 1 API call for the entire conversation.
        
        Args:
            turns: List of conversation turns
            bsp: The BSP that should govern assistant behavior
            evaluator_model: Model to use for evaluation
            
        Returns:
            ConversationEvaluation with scores and analysis
        """
        if not turns:
            return ConversationEvaluation(
                overall_score=0.0,
                context_retention_score=0.0,
                role_consistency_score=0.0,
                coherence_score=0.0,
                personality_drift_score=1.0,
                summary="No conversation turns provided",
            )

        # Step 1: Local analysis (FREE)
        local_analyses = self._analyze_locally(turns, bsp)

        # Step 2: API evaluation (1 call)
        api_result = await self._evaluate_with_api(turns, bsp, evaluator_model)

        # Step 3: Combine local and API results
        return self._combine_results(local_analyses, api_result, turns)

    def _analyze_locally(
        self,
        turns: list[ConversationTurn],
        bsp: str,
    ) -> list[TurnAnalysis]:
        """Analyze conversation locally for free.
        
        Checks:
        - Tone consistency (formal vs informal markers)
        - Context references (mentions of previous turns)
        - Response length consistency
        """
        analyses = []
        assistant_turns = [t for t in turns if t.role == "assistant"]

        if not assistant_turns:
            return analyses

        # Establish baseline tone from first assistant turn
        first_response = assistant_turns[0].content.lower()
        baseline_formal = sum(1 for m in self.FORMAL_MARKERS if m in first_response)
        baseline_informal = sum(1 for m in self.INFORMAL_MARKERS if m in first_response)
        baseline_tone = "formal" if baseline_formal > baseline_informal else "informal"

        # Track average response length
        avg_length = sum(len(t.content) for t in assistant_turns) / len(assistant_turns)

        for i, turn in enumerate(turns):
            if turn.role != "assistant":
                continue

            content_lower = turn.content.lower()

            # Check tone shift
            formal_count = sum(1 for m in self.FORMAL_MARKERS if m in content_lower)
            informal_count = sum(1 for m in self.INFORMAL_MARKERS if m in content_lower)
            current_tone = "formal" if formal_count > informal_count else "informal"
            tone_shift = current_tone != baseline_tone and (formal_count + informal_count) > 0

            # Check context references (does it mention things from earlier?)
            context_ref = False
            if i > 1:  # Not the first exchange
                # Check if response references content from any previous user turn
                prev_user_turns = [t for t in turns[:turn.turn_number] if t.role == "user"]
                for prev in prev_user_turns:
                    # Check for shared significant words (>4 chars to avoid articles)
                    prev_words = set(w for w in prev.content.lower().split() if len(w) > 4)
                    curr_words = set(w for w in content_lower.split() if len(w) > 4)
                    if prev_words & curr_words:
                        context_ref = True
                        break

            # Check role consistency (basic BSP keyword check)
            bsp_lower = bsp.lower()
            bsp_keywords = set(w for w in bsp_lower.split() if len(w) > 5)
            response_keywords = set(w for w in content_lower.split() if len(w) > 5)
            role_overlap = len(bsp_keywords & response_keywords) / max(len(bsp_keywords), 1)
            role_consistent = role_overlap > 0.02 or len(bsp_keywords) < 5

            notes = []
            if tone_shift:
                notes.append(f"Tone shifted from {baseline_tone} to {current_tone}")
            if not context_ref and i > 2:
                notes.append("No reference to previous context")
            if abs(len(turn.content) - avg_length) > avg_length * 0.7:
                notes.append("Significant length deviation from average")

            analyses.append(TurnAnalysis(
                turn_number=turn.turn_number,
                role_consistent=role_consistent,
                context_referenced=context_ref or i <= 1,
                tone_shift_detected=tone_shift,
                notes="; ".join(notes) if notes else "OK",
            ))

        return analyses

    async def _evaluate_with_api(
        self,
        turns: list[ConversationTurn],
        bsp: str,
        evaluator_model: str,
    ) -> dict:
        """Evaluate entire conversation with a single API call."""
        # Format conversation
        conversation_text = self._format_conversation(turns)

        prompt = self.CONVERSATION_EVAL_PROMPT.format(
            bsp=bsp[:800],
            conversation=conversation_text,
            total_turns=len(turns),
        )

        try:
            result = await self.llm_runner.complete(
                prompt=prompt,
                model=evaluator_model,
                temperature=0,
                max_tokens=500,
            )
            return self._parse_api_response(result.text)
        except Exception as e:
            return {
                "context_retention": 0.5,
                "role_consistency": 0.5,
                "coherence": 0.5,
                "personality_drift": 0.8,
                "drift_turns": [],
                "summary": f"API evaluation failed: {str(e)[:100]}",
                "recommendations": [],
                "error": True,
            }

    def _parse_api_response(self, text: str) -> dict:
        """Parse the API evaluation response."""
        result = {
            "context_retention": 0.5,
            "role_consistency": 0.5,
            "coherence": 0.5,
            "personality_drift": 0.8,
            "drift_turns": [],
            "summary": "",
            "recommendations": [],
        }

        for line in text.split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("CONTEXT_RETENTION:") or upper.startswith("CONTEXT RETENTION:"):
                match = re.search(r'[\d.]+', line.split(":")[-1])
                if match:
                    result["context_retention"] = max(0, min(1, float(match.group())))

            elif upper.startswith("ROLE_CONSISTENCY:") or upper.startswith("ROLE CONSISTENCY:"):
                match = re.search(r'[\d.]+', line.split(":")[-1])
                if match:
                    result["role_consistency"] = max(0, min(1, float(match.group())))

            elif upper.startswith("COHERENCE:"):
                match = re.search(r'[\d.]+', line.split(":")[-1])
                if match:
                    result["coherence"] = max(0, min(1, float(match.group())))

            elif upper.startswith("PERSONALITY_DRIFT:") or upper.startswith("PERSONALITY DRIFT:"):
                match = re.search(r'[\d.]+', line.split(":")[-1])
                if match:
                    result["personality_drift"] = max(0, min(1, float(match.group())))

            elif upper.startswith("DRIFT_TURNS:") or upper.startswith("DRIFT TURNS:"):
                value = line.split(":", 1)[-1].strip().lower()
                if value != "none":
                    nums = re.findall(r'\d+', value)
                    result["drift_turns"] = [int(n) for n in nums]

            elif upper.startswith("SUMMARY:"):
                result["summary"] = line.split(":", 1)[-1].strip()

            elif upper.startswith("RECOMMENDATIONS:"):
                value = line.split(":", 1)[-1].strip()
                if value.lower() != "none":
                    result["recommendations"] = [r.strip() for r in value.split(",") if r.strip()]

        return result

    def _combine_results(
        self,
        local_analyses: list[TurnAnalysis],
        api_result: dict,
        turns: list[ConversationTurn],
    ) -> ConversationEvaluation:
        """Combine local and API analysis into final evaluation."""
        # Local drift detection
        local_drift_turns = [
            a.turn_number for a in local_analyses if a.tone_shift_detected
        ]
        local_no_context = [
            a.turn_number for a in local_analyses if not a.context_referenced
        ]

        # Combine drift detections
        all_drift_turns = sorted(set(local_drift_turns + api_result.get("drift_turns", [])))

        # Adjust scores based on local analysis
        context_score = api_result.get("context_retention", 0.5)
        if local_no_context:
            # Penalize if local analysis found missing context references
            penalty = len(local_no_context) / max(len(local_analyses), 1) * 0.2
            context_score = max(0, context_score - penalty)

        role_score = api_result.get("role_consistency", 0.5)
        local_inconsistent = sum(1 for a in local_analyses if not a.role_consistent)
        if local_inconsistent > 0:
            penalty = local_inconsistent / max(len(local_analyses), 1) * 0.15
            role_score = max(0, role_score - penalty)

        coherence_score = api_result.get("coherence", 0.5)
        drift_score = api_result.get("personality_drift", 0.8)

        if local_drift_turns:
            drift_penalty = len(local_drift_turns) / max(len(local_analyses), 1) * 0.2
            drift_score = max(0, drift_score - drift_penalty)

        # Overall score
        overall = (
            context_score * 0.30
            + role_score * 0.30
            + coherence_score * 0.25
            + drift_score * 0.15
        )

        # Recommendations
        recommendations = list(api_result.get("recommendations", []))
        if local_no_context and len(local_no_context) > 1:
            recommendations.append(
                "BSP should instruct the model to reference previous conversation context"
            )
        if local_drift_turns:
            recommendations.append(
                "BSP should include tone consistency rules (e.g., 'maintain formal tone throughout')"
            )

        return ConversationEvaluation(
            overall_score=round(overall, 4),
            context_retention_score=round(context_score, 4),
            role_consistency_score=round(role_score, 4),
            coherence_score=round(coherence_score, 4),
            personality_drift_score=round(drift_score, 4),
            turn_analyses=local_analyses,
            total_turns=len(turns),
            drift_detected_at=all_drift_turns,
            summary=api_result.get("summary", ""),
            recommendations=recommendations,
            api_calls_used=0 if api_result.get("error") else 1,
        )

    def _format_conversation(self, turns: list[ConversationTurn]) -> str:
        """Format conversation for the evaluation prompt."""
        lines = []
        for turn in turns:
            role = turn.role.upper()
            content = turn.content[:300]  # Truncate to save tokens
            lines.append(f"[Turn {turn.turn_number}] {role}: {content}")
        return "\n".join(lines)

    async def generate_test_conversation(
        self,
        bsp: str,
        model: str,
        num_turns: int = 6,
        test_prompts: Optional[list[str]] = None,
    ) -> list[ConversationTurn]:
        """Generate a test conversation by interacting with the model.
        
        Cost: num_turns/2 API calls (one per user-assistant exchange).
        
        Args:
            bsp: BSP to test against
            model: Model to converse with
            num_turns: Total number of turns (user + assistant)
            test_prompts: Optional user prompts to use (auto-generated if None)
            
        Returns:
            List of conversation turns
        """
        default_prompts = [
            "Hello, can you help me?",
            "Can you explain what you do?",
            "What about edge cases?",
            "Can you give me an example?",
            "Thanks, one more question — what are your limitations?",
        ]

        user_prompts = test_prompts or default_prompts[:num_turns // 2]
        conversation: list[ConversationTurn] = []
        message_history = []
        turn_num = 0

        for user_prompt in user_prompts:
            # User turn
            turn_num += 1
            conversation.append(ConversationTurn(
                role="user",
                content=user_prompt,
                turn_number=turn_num,
            ))
            message_history.append(f"User: {user_prompt}")

            # Build conversation context for the model
            context = "\n".join(message_history)
            full_prompt = f"Previous conversation:\n{context}\n\nRespond as the assistant."

            try:
                result = await self.llm_runner.complete(
                    prompt=full_prompt,
                    model=model,
                    system_prompt=bsp,
                    temperature=0.3,
                    max_tokens=500,
                )
                assistant_response = result.text
            except Exception as e:
                assistant_response = f"[Error: {str(e)[:100]}]"

            # Assistant turn
            turn_num += 1
            conversation.append(ConversationTurn(
                role="assistant",
                content=assistant_response,
                turn_number=turn_num,
            ))
            message_history.append(f"Assistant: {assistant_response}")

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

        return conversation
