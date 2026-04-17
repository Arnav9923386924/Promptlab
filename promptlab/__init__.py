"""PromptLab - CI/CD for LLM Applications.

Top-level programmatic API — import directly from ``promptlab``:

    from promptlab import BSPValidator, BSPLinter, BSPOptimizer
    from promptlab import HyperparamGovernanceCouncil
    from promptlab import GuardrailTester, MultiTurnEvaluator
    from promptlab import AutoTestGenerator
"""

__version__ = "0.1.0"

# ── Lazy imports for programmatic use ────────────────────────────────────
# These are wrapped in a function so that heavy dependencies (httpx, pydantic,
# etc.) are only loaded when the user actually accesses the symbols.


def __getattr__(name: str):
    """Lazy-load public API symbols on first access."""
    _api_map = {
        # BSP
        "BSPValidator": ("promptlab.bsp.validator", "BSPValidator"),
        "BSPLinter": ("promptlab.bsp.linter", "BSPLinter"),
        "BSPOptimizer": ("promptlab.bsp.optimizer", "BSPOptimizer"),
        "BaselineManager": ("promptlab.bsp.baseline", "BaselineManager"),
        "TestRunner": ("promptlab.bsp.runner", "TestRunner"),
        # Fine-tuning governance
        "HyperparamGovernanceCouncil": ("promptlab.finetuning.governance", "HyperparamGovernanceCouncil"),
        "HyperparamProposer": ("promptlab.finetuning.proposer", "HyperparamProposer"),
        "RunEvaluator": ("promptlab.finetuning.evaluator", "RunEvaluator"),
        "ChairmanSynthesizer": ("promptlab.finetuning.chairman", "ChairmanSynthesizer"),
        # Guard
        "GuardrailTester": ("promptlab.guard.guardrail", "GuardrailTester"),
        "MultiTurnEvaluator": ("promptlab.guard.multi_turn", "MultiTurnEvaluator"),
        # Test generation
        "AutoTestGenerator": ("promptlab.testgen.generator", "AutoTestGenerator"),
    }

    if name in _api_map:
        module_path, attr = _api_map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module 'promptlab' has no attribute {name!r}")


__all__ = [
    "__version__",
    # BSP
    "BSPValidator",
    "BSPLinter",
    "BSPOptimizer",
    "BaselineManager",
    "TestRunner",
    # Fine-tuning
    "HyperparamGovernanceCouncil",
    "HyperparamProposer",
    "RunEvaluator",
    "ChairmanSynthesizer",
    # Guard
    "GuardrailTester",
    "MultiTurnEvaluator",
    # Test generation
    "AutoTestGenerator",
]
