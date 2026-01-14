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
  <img src="https://img.shields.io/badge/council-evaluation-gold.svg" alt="Council">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT">
</p>

---

## ⚡ Quick Start (60 seconds)

### 1. Install

```bash
pip install promptlab
```

### 2. Setup (Choose One)

**Using Ollama (Local, Free):**
```bash
promptlab setup ollama
```

**Using OpenRouter (Cloud):**
```bash
promptlab setup openrouter --api-key YOUR_KEY
```

### 3. Run Tests

**Test with existing YAML files:**
```bash
promptlab run roleplay
```

**Dynamic test generation (auto-creates tests for any role):**
```bash
promptlab run roleplay --role "You are a Python expert"
```

**Run HuggingFace benchmarks:**
```bash
promptlab run performance
```

---

## 🎯 What Problem Does This Solve?

You're building an LLM-powered app. You have prompts in production. You make changes.

**The Problem:**
- How do you know your changes didn't break something?
- How do you test each stage of your LLM pipeline?
- How do you get unbiased evaluation of LLM outputs?

**PromptLab is the answer:** Automated testing for LLM applications with council-based evaluation.

---

## 🔥 Key Features

| Feature | Description |
|---------|-------------|
| **Dynamic Test Generation** | `--role "You are X"` → Auto-generates relevant tests |
| **LLM Council** | Multiple models evaluate responses, reach consensus |
| **HuggingFace Benchmarks** | Import MMLU, GSM8K, TruthfulQA with one command |
| **Parallel Execution** | Run tests concurrently for speed |
| **OpenRouter + Ollama** | Use local or cloud models seamlessly |

---

## 📁 Project Structure

```
promptlab/
├── promptlab.yaml          # Your config (API keys, models)
├── promptlab.example.yaml  # Example config to copy
├── tests/                  # Test YAML files
│   ├── sentiment.yaml
│   ├── code_review.yaml
│   ├── council_test.yaml   # Tests using LLM Council
│   └── generated_*.yaml    # Auto-generated tests
└── src/                    # Source code
```

---

## ⚙️ Configuration

Copy `promptlab.example.yaml` to `promptlab.yaml` and customize:

```yaml
# promptlab.yaml
version: 1

models:
  default: ollama/llama3.1:8b           # Model to test
  generator: ollama/llama3.1:8b         # Model for generating tests
  
  providers:
    ollama:
      endpoint: http://localhost:11434
    openrouter:
      api_key: your-openrouter-api-key

council:
  enabled: true
  mode: fast  # full | fast | vote
  members:
    - ollama/llama3.1:8b
    - ollama/llama3.2:3b
  chairman: ollama/llama3.1:8b

testing:
  parallelism: 4
  timeout_ms: 30000
```

---

## 🏛️ LLM Council (Inspired by Karpathy)

Single LLM-as-judge = biased toward its own style.

**LLM Council = multiple models deliberate, cross-critique, reach consensus.**

```
STAGE 1: Independent Judging
┌────────┐  ┌────────┐  ┌────────┐
│ Llama  │  │ Gemma  │  │ Mistral│
│ 0.82   │  │ 0.75   │  │ 0.78   │
└────────┘  └────────┘  └────────┘
                │
STAGE 2: Cross-Critique (Optional)
"Judge A's score seems high because..."
                │
STAGE 3: Chairman Synthesis
┌─────────────────────────────────────┐
│ Final Score: 0.78                   │
│ Confidence: HIGH                    │
│ Consensus: "Accurate but verbose"   │
└─────────────────────────────────────┘
```

### Council Test Example:

```yaml
# tests/council_test.yaml
cases:
  - id: quality-check
    prompt: "Explain Python decorators in 2 sentences."
    assertions:
      - type: council_judge
        criteria: "Response correctly explains decorators"
        min_score: 0.7
```

---

## 📊 Free OpenRouter Models

For users without paid API access:

| Model ID | Provider |
|----------|----------|
| `openrouter/meta-llama/llama-3.1-8b-instruct:free` | Meta |
| `openrouter/google/gemma-2-9b-it:free` | Google |
| `openrouter/mistralai/mistral-7b-instruct:free` | Mistral |

Use in config:
```yaml
generator: openrouter/google/gemma-2-9b-it:free
```

---

## 📋 Commands Reference

| Command | Description |
|---------|-------------|
| `promptlab init` | Initialize project with example config |
| `promptlab setup ollama` | Quick setup for Ollama |
| `promptlab setup openrouter -k KEY` | Quick setup for OpenRouter |
| `promptlab test` | Run all YAML tests |
| `promptlab test tests/file.yaml` | Run specific test file |
| `promptlab run roleplay` | Run existing roleplay tests |
| `promptlab run roleplay --role "..."` | Generate + run tests for role |
| `promptlab run performance` | Run HuggingFace benchmarks |
| `promptlab benchmark gsm8k` | Download GSM8K benchmark |

---

## 🎚️ Council Modes

| Mode | Stages | Speed | Use Case |
|------|--------|-------|----------|
| `full` | All 3 | Slow | Critical tests |
| `fast` | 2 (skip critique) | Medium | Regular testing |
| `vote` | Just majority | Fast | Quick sanity checks |

---

## 📊 HuggingFace Benchmarks

Download and test with standard LLM benchmarks:

```bash
# Download benchmark
promptlab benchmark gsm8k --samples 20

# Run performance test
promptlab run performance
```

**Available benchmarks:** `gsm8k`, `mmlu`, `truthfulqa`, `hellaswag`

---

## 🔄 CI/CD Integration

```yaml
# .github/workflows/prompt-tests.yml
name: Prompt Tests

on:
  pull_request:
    paths: ['tests/**', 'promptlab.yaml']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install promptlab
      - run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama serve &
          sleep 5
          ollama pull llama3.1:8b
      - run: promptlab test --ci
```

---

## 🚀 Roadmap

- [x] Core framework + Ollama integration
- [x] Council evaluation engine
- [x] Dynamic test generation
- [x] HuggingFace benchmark import
- [x] Parallel test execution
- [ ] VS Code extension
- [ ] Production log capture

---

## 🤝 Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 License

MIT

---

<p align="center">
  <b>Stop testing prompts manually. Start shipping with confidence.</b>
</p>
