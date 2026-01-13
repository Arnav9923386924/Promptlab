<p align="center">
  <h1 align="center">🧪 PromptLab</h1>
</p>

<p align="center">
  <strong>CI/CD for LLM Applications</strong><br>
  Test prompts at every pipeline stage. Catch regressions before production.<br>
  Evaluate with an LLM Council, not a single biased judge.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MCP-powered-purple.svg" alt="MCP">
  <img src="https://img.shields.io/badge/council-evaluation-gold.svg" alt="Council">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT">
</p>

---

## 🎯 What Problem Does This Solve?

You're building an LLM-powered app. You have prompts in production. You make changes.

**The Problem:**
- How do you know your changes didn't break something?
- How do you test each stage of your LLM pipeline?
- How do you get unbiased evaluation of LLM outputs?

**PromptLab is the answer:** Automated testing for LLM applications with council-based evaluation.

---

## 🏛️ How We're Different

### PromptLab vs LLM Arenas (Chatbot Arena, LMSYS)

| | LLM Arenas | PromptLab |
|---|------------|-----------|
| **Question answered** | "Which model is best?" | "Did my code change break my app?" |
| **Who uses it** | Researchers | App developers |
| **Test data** | Generic benchmarks | YOUR specific use cases |
| **Goal** | Global model ranking | Prevent YOUR regressions |
| **When it runs** | Crowdsourced, async | On every git push |
| **Output** | Leaderboard | CI/CD pass/fail |

> **Arenas tell you which model to use.**  
> **PromptLab tells you if your prompts still work.**

---

## 🔑 Core Concepts

### 1. Multi-Stage Pipeline Testing

LLM apps aren't just one prompt → one response. They're pipelines:

```
User Query
    │
    ▼
┌──────────────────────┐
│ STAGE 1: Intent      │ ◄── TEST: Classification accuracy
│ Classification       │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ STAGE 2: RAG         │ ◄── TEST: Retrieval relevance
│ Retrieval            │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ STAGE 3: Response    │ ◄── TEST: Quality, safety, accuracy
│ Generation           │
└──────────────────────┘
    │
    ▼
Final Response
```

**PromptLab injects test checkpoints at each stage — not just the final output.**

---

### 2. Regression Detection Across Code Changes

```
git commit: "Updated system prompt for friendlier tone"

PromptLab runs automatically:

  v1.0 (baseline)      →  Pass rate: 94%
  v1.1 (your change)   →  Pass rate: 87%  ⚠️ REGRESSION DETECTED
  
  Failing tests:
  - refund-request: Expected empathy, got formal response
  - complaint-handling: Tone mismatch
  
  ❌ PR blocked until fixed
```

---

### 3. LLM Council Evaluation (Inspired by Karpathy)

Single LLM-as-judge = biased toward its own style.

**LLM Council = multiple models deliberate, cross-critique, reach consensus.**

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM COUNCIL PROCESS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STAGE 1: Independent Judging                               │
│  ┌────────┐  ┌────────┐  ┌────────┐                        │
│  │ Claude │  │ GPT-4  │  │ Llama  │                        │
│  │ 0.82   │  │ 0.75   │  │ 0.78   │                        │
│  └────────┘  └────────┘  └────────┘                        │
│                     │                                       │
│  STAGE 2: Cross-Critique (Anonymized)                       │
│  "Judge A's score seems high because..."                    │
│  "Judge B missed the context about..."                      │
│                     │                                       │
│  STAGE 3: Chairman Synthesis                                │
│  ┌─────────────────────────────────────┐                   │
│  │ Final Score: 0.78                   │                   │
│  │ Confidence: HIGH                    │                   │
│  │ Consensus: "Response is accurate    │                   │
│  │ but could be more concise"          │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**When council members disagree → Test flagged for human review.**

---

## 📁 Project Structure

```
your-project/
├── promptlab.yaml              # Configuration
├── prompts/                    # Your prompt templates
│   ├── intent_classifier.txt
│   ├── rag_query.txt
│   └── response_generator.txt
├── tests/
│   ├── pipeline/
│   │   ├── stage1_intent.yaml  # Intent classification tests
│   │   ├── stage2_rag.yaml     # RAG retrieval tests
│   │   └── stage3_response.yaml
│   ├── regression/
│   │   └── prod_failures.yaml  # Captured from production
│   └── edge_cases/
│       └── sarcasm.yaml
└── .promptlab/                 # Local state (gitignore)
    ├── baselines/
    └── runs/
```

---

## ⚡ Quick Start

### 1. Install

```bash
pip install promptlab
```

### 2. Initialize

```bash
cd your-llm-project
promptlab init
```

### 3. Configure Council

```yaml
# promptlab.yaml
version: 1

# Models for running your prompts
models:
  default: ollama/llama3.1:8b
  providers:
    ollama:
      endpoint: http://localhost:11434
    openrouter:
      api_key: ${OPENROUTER_API_KEY}

# LLM Council for evaluation
council:
  enabled: true
  mode: fast  # full | fast | vote
  members:
    - ollama/llama3.1:8b      # Free, local
    - ollama/mistral:7b       # Free, local
    - openrouter/gpt-4o-mini  # Cheap, diverse perspective
  chairman: ollama/llama3.1:8b

# Pipeline stages to test
pipeline:
  stages:
    - name: intent
      prompt_file: prompts/intent_classifier.txt
    - name: retrieval
      prompt_file: prompts/rag_query.txt
    - name: response
      prompt_file: prompts/response_generator.txt
```

### 4. Write Tests

```yaml
# tests/pipeline/stage1_intent.yaml
stage: intent

cases:
  - id: support-request
    input: "I need help with my order"
    assertions:
      - type: contains
        value: SUPPORT
        
  - id: sales-inquiry
    input: "What's the pricing for enterprise?"
    assertions:
      - type: contains
        value: SALES
      - type: council_judge
        criteria: "Correctly identifies sales intent"
        min_score: 0.7
```

### 5. Run Tests

```bash
promptlab test
```

```
  tests/pipeline/stage1_intent.yaml
    ✓ support-request (124ms)
    ✓ sales-inquiry (156ms)
      Council: 3/3 agreed (avg: 0.85)
      
  tests/pipeline/stage3_response.yaml
    ✓ greeting (234ms)
    ⚠️ refund-request
      Council: SPLIT (2/3)
      → Flagged for human review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Results: 8/9 passed | 1 needs review
 Regressions vs main: 0
 Cost: $0.004
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔄 Practical Test Generation

**You don't write hundreds of YAML files manually.**

### Import from CSV (Business team can contribute!)

```bash
promptlab import tests.csv
```

```csv
input,expected_label,stage,tags
"I need help",SUPPORT,intent,basic
"What's your price?",SALES,intent,basic
"This sucks",COMPLAINT,intent,edge-case
```

### Capture from Production Logs

```bash
promptlab capture --from-logs ./logs/llm_calls.jsonl
```

Automatically creates test cases from successful production calls.

### Generate with LLM Assistance

```bash
promptlab generate --examples "I love it -> POSITIVE, I hate it -> NEGATIVE"
```

LLM generates 20 similar test cases automatically.

### Golden Test Pattern

```bash
promptlab golden create
```

Interactive: Run prompts, review outputs, approve as "golden" expected values.

---

## 📊 Regression Workflow

### Create Baseline

```bash
promptlab baseline create --tag main
```

### Test Against Baseline

```bash
promptlab test --baseline main
```

### CI/CD Integration

```yaml
# .github/workflows/prompt-tests.yml
name: Prompt Tests

on:
  pull_request:
    paths: ['prompts/**', 'tests/**']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install promptlab
      - run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama pull llama3.1:8b
      - run: promptlab test --ci --baseline main
```

**Exit codes:**
- `0` = All passed
- `1` = Failures or regressions
- `2` = Config error

---

## 🏗️ Architecture (MCP-Based)

```
┌─────────────────────────────────────────────────────────────┐
│                      PromptLab CLI                          │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  mcp-llm-runner │  │  mcp-council    │  │  mcp-pipeline   │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • Ollama        │  │ • Stage 1: Judge│  │ • Stage routing │
│ • OpenRouter    │  │ • Stage 2: Crit │  │ • Checkpoint    │
│ • Any provider  │  │ • Stage 3: Synth│  │   injection     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Why MCP?**
- Swap models without changing test code
- Add new evaluation metrics as plugins
- Pipeline stages are composable
- Servers are shareable across projects

---

## 📋 Commands Reference

| Command | Description |
|---------|-------------|
| `promptlab init` | Initialize project |
| `promptlab test` | Run all tests |
| `promptlab test --stage intent` | Test specific pipeline stage |
| `promptlab test --baseline main` | Compare against baseline |
| `promptlab test --watch` | Re-run on file changes |
| `promptlab baseline create --tag v1` | Save current as baseline |
| `promptlab import data.csv` | Import test cases from CSV |
| `promptlab capture --from-logs` | Generate tests from logs |
| `promptlab council explain <id>` | Show council deliberation |

---

## 🎚️ Council Modes

| Mode | Stages | Cost | Use Case |
|------|--------|------|----------|
| `full` | All 3 | High | Critical/production tests |
| `fast` | 2 (skip critique) | Medium | Regular testing |
| `vote` | Just pass/fail | Low | Quick sanity checks |

```yaml
# Per-test override
assertions:
  - type: council_judge
    mode: full  # Override for this critical test
    criteria: "..."
```

---

## 🚀 Roadmap

- [ ] **Phase 1:** Core framework + Ollama integration
- [ ] **Phase 2:** Council evaluation engine
- [ ] **Phase 3:** Pipeline stage testing
- [ ] **Phase 4:** CI/CD + baseline management
- [ ] **Phase 5:** Import/capture tools
- [ ] **Phase 6:** VS Code extension

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 License

MIT

---

<p align="center">
  <b>Stop testing prompts manually. Start shipping with confidence.</b>
</p>
