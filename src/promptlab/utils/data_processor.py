"""Data processor for extracting Q&A pairs and masked tests from scraped content - NO LLM REQUIRED."""

import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import yaml
from rich.console import Console

from promptlab.utils.scraper import ScrapedContent

console = Console()


@dataclass
class QAPair:
    """A question-answer pair extracted from content."""
    question: str
    answer: str
    source_url: str = ""
    domain: str = ""
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)


@dataclass
class MaskedTest:
    """A fill-in-the-blank (cloze) test extracted from content."""
    masked_text: str
    answer: str
    original_text: str
    mask_position: int
    source_url: str = ""
    domain: str = ""
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)


@dataclass
class RolePlayPersona:
    """A role-play persona generated from domain content."""
    name: str
    description: str
    system_prompt: str
    example_dialogues: list[dict] = field(default_factory=list)
    domain: str = ""
    traits: list[str] = field(default_factory=list)


class DataProcessor:
    """Process scraped content using text extraction - NO LLM NEEDED."""
    
    def __init__(self, llm_runner=None, model: Optional[str] = None):
        """Initialize the data processor. LLM runner is optional and not used."""
        pass  # No LLM needed!
    
    def _is_valid_qa(self, question: str, answer: str) -> bool:
        """Check if Q&A pair is valid and not noise."""
        # Skip navigation/UI text
        skip_patterns = [
            r'^what is what\??',  # Broken patterns
            r'^what is if\b',  # "What is If a variable..."
            r'^what is it\b',  # "What is It is..."
            r'^what is there\b',
            r'^what is this\b',
            r'^what is that\b',
            r'^what is several\b',  # Weird patterns
            r'(navigation|index|next|previous|theme|auto|light|dark|menu|home)',
            r'^(click|press|select|choose|see|explain\s+\w+\s+in\s+the\s+context)',
            r'^\d+$',  # Just numbers
            r'^[\W\s]+$',  # Just symbols
            r'^(documentation|general)',  # Common noise
            r'\n',  # Contains newlines (likely noisy)
            r'advertising',  # Ads
        ]
        q_lower = question.lower().strip()
        a_lower = answer.lower().strip()
        for pattern in skip_patterns:
            if re.search(pattern, q_lower, re.I) or re.search(pattern, a_lower, re.I):
                return False
        # Question should be a proper question ending with ?
        if not question.strip().endswith('?'):
            return False
        # Question should not have newlines or weird chars
        if '\n' in question or '|' in question:
            return False
        return len(question) > 15 and len(answer) > 10 and len(answer) < 250
    
    def _extract_sentences(self, text: str, max_sentences: int = 50) -> list[str]:
        """Extract clean sentences from text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean = []
        for s in sentences:
            s = s.strip()
            # Skip short, navigation-like, or symbol-only lines
            if len(s) > 30 and len(s) < 500 and not re.match(r'^[\|\-\>\<\»\«\s]+', s):
                clean.append(s)
        return clean[:max_sentences]
    
    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms (capitalized words, technical terms)."""
        # Skip common non-informative words
        skip_words = {'Navigation', 'Index', 'Next', 'Previous', 'Home', 'Menu', 'Search', 'Click', 'See', 'Also'}
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        caps = [c for c in caps if c not in skip_words and len(c) > 3]
        tech = re.findall(r'\b[a-z]+_[a-z]+\b|\b[a-z]+[A-Z][a-z]+\b', text)
        terms = list(set(caps + tech))
        return terms[:20]
    
    def _clean_text(self, text: str) -> str:
        """Remove navigation artifacts from text."""
        # Remove common nav patterns
        text = re.sub(r'—[^—]+documentation', '', text, flags=re.I)
        text = re.sub(r'\d+\.\d+(\.\d+)?\s+Documentation', '', text)
        text = re.sub(r'Navigation\s+index\s+modules', '', text, flags=re.I)
        text = re.sub(r'\|\s*(next|previous|home|index)\s*\|', '', text, flags=re.I)
        text = re.sub(r'Theme\s+(Auto|Light|Dark)', '', text, flags=re.I)
        return text.strip()
    
    def _find_qa_patterns(self, text: str) -> list[tuple[str, str]]:
        """Find Q&A patterns in text (FAQ sections, Q: A: format, etc.)."""
        text = self._clean_text(text)
        qa_pairs = []
        
        # Pattern 1: Q: ... A: ... format
        qa_matches = re.findall(
            r'(?:Q|Question)[:\s]+(.+?)(?:A|Answer)[:\s]+(.+?)(?=(?:Q|Question)[:\s]|$)',
            text, re.IGNORECASE | re.DOTALL
        )
        for q, a in qa_matches:
            q, a = q.strip()[:200], a.strip()[:500]
            if self._is_valid_qa(q, a):
                qa_pairs.append((q, a))
        
        # Pattern 2: "What is X?" followed by definition
        what_matches = re.findall(
            r'(What (?:is|are|does|do) [^?]+\?)\s*([^.!?]+[.!?])',
            text, re.IGNORECASE
        )
        for q, a in what_matches:
            q, a = q.strip(), a.strip()
            if self._is_valid_qa(q, a):
                qa_pairs.append((q, a))
        
        # Pattern 3: "How to X?" followed by explanation
        how_matches = re.findall(
            r'(How (?:to|do|does|can) [^?]+\?)\s*([^.!?]+[.!?](?:[^.!?]+[.!?])?)',
            text, re.IGNORECASE
        )
        for q, a in how_matches:
            q, a = q.strip(), a.strip()
            if self._is_valid_qa(q, a):
                qa_pairs.append((q, a))
        
        # Pattern 4: "Why X?" followed by explanation
        why_matches = re.findall(
            r'(Why (?:is|are|do|does|should|would|can) [^?]+\?)\s*([^.!?]+[.!?](?:[^.!?]+[.!?])?)',
            text, re.IGNORECASE
        )
        for q, a in why_matches:
            q, a = q.strip(), a.strip()
            if self._is_valid_qa(q, a):
                qa_pairs.append((q, a))
        
        # Pattern 5: "When X?" questions
        when_matches = re.findall(
            r'(When (?:should|do|does|is|are|to) [^?]+\?)\s*([^.!?]+[.!?](?:[^.!?]+[.!?])?)',
            text, re.IGNORECASE
        )
        for q, a in when_matches:
            q, a = q.strip(), a.strip()
            if self._is_valid_qa(q, a):
                qa_pairs.append((q, a))
        
        # Pattern 6: "Which X?" questions
        which_matches = re.findall(
            r'(Which [^?]+\?)\s*([^.!?]+[.!?])',
            text, re.IGNORECASE
        )
        for q, a in which_matches:
            q, a = q.strip(), a.strip()
            if self._is_valid_qa(q, a):
                qa_pairs.append((q, a))
        
        return qa_pairs[:20]  # Increased from 10 to 20
    
    def _generate_questions_from_content(self, text: str, domain: str) -> list[QAPair]:
        """Generate Q&A pairs from content using text analysis."""
        qa_pairs = []
        seen_questions = set()
        
        # 1. Try to find existing Q&A patterns (already validated in _find_qa_patterns)
        found_qa = self._find_qa_patterns(text)
        for q, a in found_qa:
            q_norm = q.lower().strip()
            if q_norm in seen_questions:
                continue
            
            # Validate again with the full answer
            if not self._is_valid_qa(q, a):
                continue
                
            seen_questions.add(q_norm)
            
            # Clean and truncate the answer to first 2 sentences
            sentences = re.split(r'(?<=[.!?])\s+', a)
            clean_answer = ' '.join(sentences[:2]).strip()
            if len(clean_answer) > 200:
                clean_answer = clean_answer[:200] + '...'
            
            qa_pairs.append(QAPair(
                question=q,
                answer=clean_answer,
                domain=domain,
                difficulty="medium",
                tags=[domain.lower()]
            ))
        
        # 2. Generate questions from key sentences (definitions)
        sentences = self._extract_sentences(text)
        
        # Find "X is Y" definition patterns
        for sentence in sentences[:30]:
            # Match patterns like "Python is a programming language"
            match = re.match(r'^((?:The\s+)?[A-Z][A-Za-z0-9\s]+?) (?:is|are) (?:a |an |the )?(.+)', sentence)
            if match:
                subject, definition = match.groups()
                subject = subject.strip()
                definition = definition.strip()
                
                # Skip if subject is too short/long or generic
                if len(subject) < 4 or len(subject) > 50 or len(definition) < 20:
                    continue
                    
                # Skip navigation-like or generic subjects
                skip_subjects = {'this', 'it', 'there', 'here', 'that', 'these', 'those', 
                                'the page', 'the section', 'the following', 'this page',
                                'if', 'if a', 'several', 'what', 'it is'}
                if subject.lower() in skip_subjects:
                    continue
                
                question = f"What is {subject}?"
                
                # Use the full definition as answer (truncated if too long)
                answer = definition if len(definition) <= 150 else definition[:150] + '...'
                
                # Validate the generated Q&A
                if not self._is_valid_qa(question, answer):
                    continue
                
                if question.lower() in seen_questions:
                    continue
                seen_questions.add(question.lower())
                
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    domain=domain,
                    difficulty="easy",
                    tags=[domain.lower(), "definition"]
                ))
        
        return qa_pairs[:25]  # Increased from 15 to 25 for more tests
    
    async def extract_qa_pairs(self, content: ScrapedContent, domain: str) -> list[QAPair]:
        """Extract Q&A pairs from scraped content - NO LLM NEEDED."""
        qa_pairs = self._generate_questions_from_content(content.text, domain)
        for qa in qa_pairs:
            qa.source_url = content.url
        return qa_pairs
    
    async def generate_persona(self, contents: list[ScrapedContent], domain: str) -> Optional[RolePlayPersona]:
        """Generate a basic persona from domain content - NO LLM NEEDED."""
        return RolePlayPersona(
            name=f"{domain.title()} Expert",
            description=f"An AI assistant specialized in {domain}",
            system_prompt=f"You are a helpful {domain} expert. Answer questions accurately and concisely.",
            example_dialogues=[
                {"user": f"Tell me about {domain}", "assistant": f"I'd be happy to help you learn about {domain}. What would you like to know?"},
            ],
            domain=domain,
            traits=["knowledgeable", "helpful", "precise"]
        )
    
    def _generate_masked_tests(self, text: str, domain: str) -> list[MaskedTest]:
        """Generate fill-in-the-blank (cloze) tests from text."""
        masked_tests = []
        sentences = self._extract_sentences(text)
        
        # Common technical/domain terms we want to mask
        important_pos_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
            r'\b([a-z]+_[a-z]+)\b',  # snake_case terms
            r'\b([a-z]+[A-Z][a-z]+)\b',  # camelCase terms
            r'\b(\d+(?:\.\d+)?)\b',  # Numbers
        ]
        
        for sentence in sentences[:30]:
            # Skip too short or too long sentences
            if len(sentence) < 40 or len(sentence) > 300:
                continue
            
            # Find potential words to mask
            words = sentence.split()
            if len(words) < 6:
                continue
            
            # Try to find important terms to mask
            candidates = []
            
            # Find capitalized words (likely important terms)
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', word)
                
                # Skip common words
                skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                              'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                              'could', 'should', 'may', 'might', 'must', 'can', 'this',
                              'that', 'these', 'those', 'it', 'its', 'for', 'with', 'to',
                              'of', 'in', 'on', 'at', 'by', 'from', 'up', 'about', 'into',
                              'over', 'after', 'and', 'but', 'or', 'if', 'then', 'so', 'as'}
                
                if clean_word.lower() in skip_words:
                    continue
                
                # Good candidates: proper nouns, technical terms, numbers
                if len(clean_word) >= 4:
                    if clean_word[0].isupper() or '_' in clean_word or any(c.isupper() for c in clean_word[1:]):
                        candidates.append((i, word, clean_word, "term"))
                    elif clean_word.isdigit() or re.match(r'\d+\.\d+', clean_word):
                        candidates.append((i, word, clean_word, "number"))
                    elif len(clean_word) >= 6:  # Longer words are often important
                        candidates.append((i, word, clean_word, "word"))
            
            if not candidates:
                continue
            
            # Pick a random candidate to mask
            idx, original_word, clean_answer, word_type = random.choice(candidates)
            
            # Create masked sentence
            masked_words = words.copy()
            masked_words[idx] = "____"
            masked_sentence = " ".join(masked_words)
            
            # Determine difficulty based on word type
            difficulty = "easy" if word_type == "number" else "medium" if word_type == "term" else "hard"
            
            masked_tests.append(MaskedTest(
                masked_text=masked_sentence,
                answer=clean_answer,
                original_text=sentence,
                mask_position=idx,
                domain=domain,
                difficulty=difficulty,
                tags=[domain.lower(), "cloze", word_type],
            ))
        
        return masked_tests[:15]  # Increased from 10 to 15 masked tests per content
    
    async def extract_masked_tests(self, content: ScrapedContent, domain: str) -> list[MaskedTest]:
        """Extract fill-in-the-blank tests from scraped content."""
        masked_tests = self._generate_masked_tests(content.text, domain)
        for test in masked_tests:
            test.source_url = content.url
        return masked_tests
    
    async def process_content(
        self,
        contents: list[ScrapedContent],
        domain: str,
        output_type: Literal["benchmark", "roleplay", "finetune", "masked", "all"] = "all"
    ) -> dict:
        """Process scraped content - NO LLM NEEDED."""
        results = {
            "domain": domain,
            "sources": len(contents),
            "qa_pairs": [],
            "masked_tests": [],
            "persona": None,
            "finetune_data": [],
        }
        
        if output_type in ("benchmark", "finetune", "all"):
            console.print(f"[cyan]Extracting Q&A pairs from {len(contents)} sources...[/cyan]")
            for content in contents:
                pairs = await self.extract_qa_pairs(content, domain)
                results["qa_pairs"].extend(pairs)
                if pairs:
                    console.print(f"  [green]✓[/green] {len(pairs)} pairs from {content.title[:50]}")
        
        if output_type in ("masked", "all"):
            console.print(f"[cyan]Generating masked (fill-in-blank) tests...[/cyan]")
            for content in contents:
                masked = await self.extract_masked_tests(content, domain)
                results["masked_tests"].extend(masked)
                if masked:
                    console.print(f"  [green]✓[/green] {len(masked)} cloze tests from {content.title[:50]}")
        
        if output_type in ("roleplay", "all"):
            console.print(f"[cyan]Generating persona for {domain}...[/cyan]")
            persona = await self.generate_persona(contents, domain)
            if persona:
                results["persona"] = persona
                console.print(f"  [green]✓[/green] Created persona: {persona.name}")
        
        if output_type in ("finetune", "all"):
            for qa in results["qa_pairs"]:
                results["finetune_data"].append({
                    "messages": [
                        {"role": "user", "content": qa.question},
                        {"role": "assistant", "content": qa.answer},
                    ]
                })
            # Also add masked tests to finetune data
            for masked in results["masked_tests"]:
                results["finetune_data"].append({
                    "messages": [
                        {"role": "user", "content": f"Fill in the blank: {masked.masked_text}"},
                        {"role": "assistant", "content": masked.answer},
                    ]
                })
        
        return results


def generate_test_cases_yaml(qa_pairs: list[QAPair], domain: str) -> dict:
    """Generate PromptLab test cases YAML from Q&A pairs."""
    cases = []
    for i, qa in enumerate(qa_pairs, 1):
        cases.append({
            "id": f"{domain.lower().replace(' ', '-')}-{i}",
            "prompt": qa.question,
            "assertions": [{"type": "contains", "value": qa.answer.strip(), "case_sensitive": False}],
            "tags": qa.tags or [domain.lower()],
        })
    return {
        "metadata": {"name": f"{domain.title()} - Knowledge Test"},
        "defaults": {"temperature": 0},
        "cases": cases,
    }


def generate_masked_test_cases_yaml(masked_tests: list[MaskedTest], domain: str) -> dict:
    """Generate PromptLab test cases YAML from masked (cloze) tests."""
    cases = []
    for i, test in enumerate(masked_tests, 1):
        cases.append({
            "id": f"{domain.lower().replace(' ', '-')}-cloze-{i}",
            "prompt": f"Fill in the blank with the correct word or phrase:\n\n{test.masked_text}",
            "expected": test.answer,
            "assertions": [
                {"type": "contains", "value": test.answer, "case_sensitive": False}
            ],
            "tags": test.tags or [domain.lower(), "cloze"],
        })
    return {
        "metadata": {
            "name": f"{domain.title()} - Fill-in-the-Blank Tests",
            "description": "Cloze tests to evaluate knowledge completion ability",
        },
        "defaults": {"temperature": 0},
        "cases": cases,
    }


def generate_persona_yaml(persona: RolePlayPersona) -> dict:
    """Generate role-play test cases YAML from persona."""
    cases = []
    for i, dialogue in enumerate(persona.example_dialogues, 1):
        cases.append({
            "id": f"{persona.domain.lower().replace(' ', '-')}-roleplay-{i}",
            "system_prompt": persona.system_prompt,
            "prompt": dialogue.get("user", ""),
            "assertions": [{"type": "min_length", "value": 50}],
            "tags": ["roleplay", persona.domain.lower()],
        })
    return {
        "metadata": {
            "name": f"{persona.name} - Role-play Tests",
            "persona": {"name": persona.name, "description": persona.description, "traits": persona.traits}
        },
        "defaults": {"temperature": 0.7, "system_prompt": persona.system_prompt},
        "cases": cases,
    }


def save_outputs(results: dict, output_dir: Path, domain: str) -> dict[str, Path]:
    """Save processed results to files."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    domain_slug = domain.lower().replace(" ", "_")
    
    if results["qa_pairs"]:
        test_yaml = generate_test_cases_yaml(results["qa_pairs"], domain)
        test_path = output_dir / f"{domain_slug}_benchmark.yaml"
        with open(test_path, "w", encoding="utf-8") as f:
            yaml.dump(test_yaml, f, default_flow_style=False, allow_unicode=True)
        saved_files["benchmark"] = test_path
        console.print(f"[green]✓[/green] Saved benchmark: {test_path}")
    
    if results.get("masked_tests"):
        masked_yaml = generate_masked_test_cases_yaml(results["masked_tests"], domain)
        masked_path = output_dir / f"{domain_slug}_cloze.yaml"
        with open(masked_path, "w", encoding="utf-8") as f:
            yaml.dump(masked_yaml, f, default_flow_style=False, allow_unicode=True)
        saved_files["cloze"] = masked_path
        console.print(f"[green]✓[/green] Saved cloze tests: {masked_path}")
    
    if results["persona"]:
        persona_yaml = generate_persona_yaml(results["persona"])
        persona_path = output_dir / f"{domain_slug}_roleplay.yaml"
        with open(persona_path, "w", encoding="utf-8") as f:
            yaml.dump(persona_yaml, f, default_flow_style=False, allow_unicode=True)
        saved_files["roleplay"] = persona_path
        console.print(f"[green]✓[/green] Saved roleplay: {persona_path}")
    
    if results["finetune_data"]:
        finetune_path = output_dir / f"{domain_slug}_finetune.jsonl"
        with open(finetune_path, "w", encoding="utf-8") as f:
            for entry in results["finetune_data"]:
                f.write(json.dumps(entry) + "\n")
        saved_files["finetune"] = finetune_path
        console.print(f"[green]✓[/green] Saved {len(results['finetune_data'])} finetune examples")
    
    return saved_files
