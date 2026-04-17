"""CLI commands — history, guardrail, conversation evaluation, training data export."""

import typer
from rich.panel import Panel
from pathlib import Path
import yaml

from promptlab.cli.helpers import console, is_interactive, ensure_initialized


def register(app: typer.Typer):
    """Register general-purpose commands on the root app."""

    @app.command("history")
    def show_history(
        last_n: int = typer.Option(10, "--last", "-n", help="Number of recent entries to show"),
        compare: bool = typer.Option(False, "--compare", "-c", help="Compare BSP versions"),
    ):
        """Show evaluation history and score trends — completely FREE.

        Examples:
          promptlab history               # Show recent evaluation history
          promptlab history --last 20     # Show last 20 entries
          promptlab history --compare     # Compare scores across BSP versions
        """
        from promptlab.utils.history import EvaluationHistory

        cwd = Path.cwd()
        tracker = EvaluationHistory(cwd)

        if not tracker.get_history():
            console.print("[yellow]No evaluation history found. Run 'promptlab validate' first.[/yellow]")
            raise typer.Exit(0)

        if compare:
            comparison = tracker.compare_bsp_versions()
            console.print(Panel.fit("[bold]BSP Version Comparison[/bold]", border_style="blue"))
            for version, data in comparison.items():
                console.print(f"  {version}: avg={data['average_score']:.2f}, best={data['best_score']:.2f}, runs={data['evaluations']}")
        else:
            report = tracker.format_report()
            trend = tracker.get_trend(last_n)
            border = "green" if trend.trend_direction == "improving" else "red" if trend.trend_direction == "declining" else "blue"
            console.print(Panel(report, title="Evaluation History", border_style=border))

    @app.command("guard")
    @app.command("guardrail", hidden=True)
    def run_guardrail(
        bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
        categories: list[str] = typer.Option(None, "--category", "-c", help="Attack categories to test"),
        evaluator: str = typer.Option(None, "--evaluator", "-e", help="Model for ambiguous evaluations"),
    ):
        """Run adversarial security tests against your BSP.

        Tests prompt injection, jailbreaks, system prompt extraction,
        role breaking, and data exfiltration attacks.

        Examples:
          promptlab guardrail                                    # Run all attack categories
          promptlab guardrail -c prompt_injection -c jailbreak   # Test specific categories
          promptlab guardrail -e openrouter/google/gemini-2.0-flash-001  # Use evaluator for ambiguous results
        """
        import asyncio as aio
        from promptlab.utils.config import load_config, load_bsp
        from promptlab.llm_council.llm_runner.runner import LLMRunner
        from promptlab.guard.guardrail import GuardrailTester

        cwd = Path.cwd()
        config_path = ensure_initialized(cwd)

        config = load_config(config_path)

        if bsp_file:
            bsp = Path(bsp_file).read_text(encoding="utf-8")
        else:
            bsp = load_bsp(config, cwd)
            if not bsp:
                console.print("[red]✗ No BSP configured.[/red]")
                raise typer.Exit(1)

        async def _run():
            runner = LLMRunner(config.models.dict())
            tester = GuardrailTester(runner, {"rate_limit_delay": config.guardrail.rate_limit_delay})

            test_categories = list(categories) if categories else config.guardrail.categories
            eval_model = evaluator or config.guardrail.evaluator_model

            console.print(Panel(
                f"[bold]Model:[/bold] {config.models.default}\n"
                f"[bold]Categories:[/bold] {', '.join(test_categories)}\n"
                f"[bold]Evaluator:[/bold] {eval_model or 'pattern-matching only (free)'}",
                title="Guardrail Test",
                border_style="blue",
            ))

            report = await tester.run_guardrail_tests(
                bsp=bsp,
                model=config.models.default,
                categories=test_categories,
                evaluator_model=eval_model,
            )

            # Display results
            console.print(f"\n[bold]Safety Score:[/bold] {report.overall_safety_score:.2f}/1.00")
            console.print(f"[bold]Total Attacks:[/bold] {report.total_attacks}")
            console.print(f"[bold]Vulnerabilities:[/bold] {report.vulnerabilities_found}")

            if report.critical_count:
                console.print(f"[bold red]  Critical: {report.critical_count}[/bold red]")
            if report.high_count:
                console.print(f"[red]  High: {report.high_count}[/red]")
            if report.medium_count:
                console.print(f"[yellow]  Medium: {report.medium_count}[/yellow]")

            if report.recommendations:
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in report.recommendations:
                    console.print(f"  → {rec}")

            if report.vulnerabilities_found == 0:
                console.print(Panel("[bold green]✓ No vulnerabilities found![/bold green]", border_style="green"))
            else:
                console.print(Panel(f"[bold red]✗ {report.vulnerabilities_found} vulnerabilities found[/bold red]", border_style="red"))

            return report

        try:
            report = aio.run(_run())
            if report.critical_count > 0:
                raise typer.Exit(1)
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    @app.command("chat")
    @app.command("evaluate-conversation", hidden=True)
    def evaluate_conversation(
        bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
        conversation_file: str = typer.Option(None, "--conversation", "-c", help="Path to conversation JSON file"),
        model: str = typer.Option(None, "--model", "-m", help="Model to generate test conversation"),
        evaluator: str = typer.Option(None, "--evaluator", "-e", help="Model for evaluation"),
        num_turns: int = typer.Option(6, "--turns", "-n", help="Number of conversation turns to generate"),
    ):
        """Evaluate multi-turn conversation quality.

        Either provide a conversation file or auto-generate a test conversation.
        Cost: 1 API call for evaluation + N/2 calls if generating conversation.

        Examples:
          promptlab evaluate-conversation                              # Generate & evaluate conversation
          promptlab evaluate-conversation -c conversation.json         # Evaluate existing conversation
          promptlab evaluate-conversation --turns 10 -m ollama/llama3  # Generate 10-turn conversation
        """
        import asyncio as aio
        import json as json_module
        from promptlab.utils.config import load_config, load_bsp
        from promptlab.llm_council.llm_runner.runner import LLMRunner
        from promptlab.guard.multi_turn import MultiTurnEvaluator, ConversationTurn

        cwd = Path.cwd()
        config_path = ensure_initialized(cwd)

        config = load_config(config_path)

        if bsp_file:
            bsp = Path(bsp_file).read_text(encoding="utf-8")
        else:
            bsp = load_bsp(config, cwd)
            if not bsp:
                console.print("[red]✗ No BSP configured.[/red]")
                raise typer.Exit(1)

        target_model = model or config.models.default
        eval_model = evaluator or config.multi_turn.evaluator_model or (config.council.members[0] if config.council.members else target_model)

        async def _run():
            runner = LLMRunner(config.models.dict())
            evaluator_instance = MultiTurnEvaluator(runner, {"rate_limit_delay": config.multi_turn.rate_limit_delay})

            if conversation_file:
                # Load existing conversation
                conv_data = json_module.loads(Path(conversation_file).read_text(encoding="utf-8"))
                turns = [
                    ConversationTurn(role=t["role"], content=t["content"], turn_number=i + 1)
                    for i, t in enumerate(conv_data)
                ]
                console.print(f"[green]✓ Loaded {len(turns)} turns from {conversation_file}[/green]")
            else:
                # Generate test conversation
                console.print(f"[blue]Generating {num_turns}-turn test conversation with {target_model}...[/blue]")
                turns = await evaluator_instance.generate_test_conversation(
                    bsp=bsp,
                    model=target_model,
                    num_turns=num_turns,
                )
                console.print(f"[green]✓ Generated {len(turns)} turns[/green]")

            # Show conversation
            console.print("\n[bold]Conversation:[/bold]")
            for turn in turns:
                role_color = "cyan" if turn.role == "user" else "green"
                console.print(f"  [{role_color}][{turn.role.upper()}][/{role_color}] {turn.content[:150]}...")

            # Evaluate
            console.print(f"\n[blue]Evaluating with {eval_model}...[/blue]")
            result = await evaluator_instance.evaluate_conversation(turns, bsp, eval_model)

            # Display results
            console.print(Panel(
                f"[bold]Overall Score:[/bold] {result.overall_score:.5f}\n"
                f"[bold]Context Retention:[/bold] {result.context_retention_score:.5f}\n"
                f"[bold]Role Consistency:[/bold] {result.role_consistency_score:.5f}\n"
                f"[bold]Coherence:[/bold] {result.coherence_score:.5f}\n"
                f"[bold]Personality Drift:[/bold] {result.personality_drift_score:.5f} (1.0 = no drift)\n"
                f"\n{result.summary}",
                title="Conversation Evaluation",
                border_style="green" if result.overall_score >= 0.7 else "red",
            ))

            if result.drift_detected_at:
                console.print(f"[yellow]  ⚠ Drift detected at turns: {result.drift_detected_at}[/yellow]")

            if result.recommendations:
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in result.recommendations:
                    console.print(f"  → {rec}")

            console.print(f"\n[dim]API calls used: {result.api_calls_used}[/dim]")

        try:
            aio.run(_run())
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    @app.command("export")
    @app.command("generate-training-data", hidden=True)
    def generate_training_data(
        bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
        outputs_dir: str = typer.Option("temp", "--outputs-dir", "-d", help="Directory with test outputs"),
        min_score: float = typer.Option(0.7, "--min-score", "-s", help="Minimum quality score filter"),
        format: str = typer.Option("openai", "--format", "-f", help="Output format: openai, alpaca, sharegpt"),
        preview: bool = typer.Option(False, "--preview", "-p", help="Preview without saving"),
    ):
        """Generate fine-tuning training data from BSP + evaluation outputs — completely FREE.

        Converts your BSP and high-scoring test outputs into JSONL format
        suitable for fine-tuning models. No API calls needed.

        Examples:
          promptlab export                            # Generate from default outputs
          promptlab export --format alpaca            # Alpaca format
          promptlab export --min-score 0.8 --preview  # Preview high-quality examples only
        """
        import json as json_module
        from promptlab.utils.config import load_config, load_bsp
        from promptlab.finetuning.training_data import TrainingDataGenerator

        cwd = Path.cwd()
        config_path = ensure_initialized(cwd)

        config = load_config(config_path)

        if bsp_file:
            bsp = Path(bsp_file).read_text(encoding="utf-8")
        else:
            bsp = load_bsp(config, cwd)
            if not bsp:
                console.print("[red]✗ No BSP configured.[/red]")
                raise typer.Exit(1)

        # Load outputs from evaluation history
        outputs_path = cwd / outputs_dir
        outputs = []

        if outputs_path.is_dir():
            for f in sorted(outputs_path.glob("*.yaml")) + sorted(outputs_path.glob("*.json")):
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "cases" in data:
                        for case in data["cases"]:
                            outputs.append({
                                "prompt": case.get("prompt", ""),
                                "response": case.get("response", case.get("expected", "")),
                                "test_id": case.get("id", f.stem),
                                "score": min_score,
                            })
                except Exception:
                    continue

        if not outputs:
            console.print("[yellow]No test outputs found. Run 'promptlab bsp' first to generate outputs.[/yellow]")
            raise typer.Exit(0)

        generator = TrainingDataGenerator(
            project_root=cwd,
            min_quality_score=min_score,
            output_format=format,
        )

        dataset = generator.generate_from_outputs(
            bsp=bsp,
            outputs=outputs,
            overall_score=min_score,
            bsp_version=config.bsp.version,
        )

        report = generator.format_report(dataset)
        console.print(Panel(report, title="Training Data", border_style="blue"))

        if preview:
            console.print("\n[bold]Preview (first 3 examples):[/bold]")
            for ex in dataset.examples[:3]:
                console.print(f"  [cyan]User:[/cyan] {ex.user[:100]}")
                console.print(f"  [green]Assistant:[/green] {ex.assistant[:100]}")
                console.print()
        else:
            output_file = generator.export_jsonl(dataset)
            preview_file = generator.export_json(dataset)
            console.print(f"[green]✓ Training data saved to {output_file}[/green]")
            console.print(f"[green]✓ Preview saved to {preview_file}[/green]")
            console.print(f"\n[dim]Total examples: {dataset.total_examples} | Format: {format} | Estimated tokens: {dataset.total_tokens_estimate:,}[/dim]")
