"""CLI commands — fine-tuning hyperparameter governance."""

import typer
from rich.panel import Panel
from pathlib import Path

from promptlab.cli.helpers import console


def register(ft_app: typer.Typer):
    """Register fine-tuning commands on the ft sub-app."""

    @ft_app.callback(invoke_without_command=True)
    def ft_default(ctx: typer.Context):
        """Fine-tuning hyperparameter governance council."""
        if ctx.invoked_subcommand is None:
            # Default: run governance
            ctx.invoke(ft_run)

    @ft_app.command("run")
    def ft_run(
        config_file: str = typer.Option("ft_config.json", "--config", "-c", help="Path to FT governance config"),
        artifacts_file: str = typer.Option(None, "--artifacts", "-a", help="Path to run artifacts JSON"),
        propose_only: bool = typer.Option(False, "--propose", help="Only propose next config (skip evaluation)"),
        output_json: str = typer.Option(None, "--output", "-o", help="Save governance output to JSON file"),
    ):
        """Run the fine-tuning hyperparameter governance council.

        This command orchestrates the consensus-based LLM council that
        proposes, evaluates, and selects hyperparameter configurations.

        Modes:
          --propose    Just propose the next config (no artifacts needed)
          (default)    Evaluate run artifacts and produce a governance decision

        Examples:
          promptlab ft                                 # Run with default config
          promptlab ft --propose                       # Just propose next config
          promptlab ft -a run_artifacts.json           # Evaluate a training run
          promptlab ft -c custom_config.json -o out.json
        """
        import asyncio as aio
        import json as json_module

        cwd = Path.cwd()
        config_path = cwd / config_file

        if not config_path.exists():
            console.print(f"[red]✗ Config not found: {config_path}[/red]")
            console.print("[dim]Run 'promptlab init --mode ft' to create ft_config.json[/dim]")
            raise typer.Exit(1)

        try:
            ft_config = json_module.loads(config_path.read_text(encoding="utf-8"))
        except json_module.JSONDecodeError as e:
            console.print(f"[red]✗ Invalid JSON in {config_file}: {e}[/red]")
            raise typer.Exit(1)

        from promptlab.utils.config import load_config
        from promptlab.llm_council.llm_runner.runner import LLMRunner
        from promptlab.finetuning.governance import HyperparamGovernanceCouncil
        from promptlab.finetuning.models import (
            SearchSpace, HardwareProfile, DatasetProfile,
            RunArtifact, CandidateConfig, TrainingMetrics, RunHistoryEntry,
        )

        # Load promptlab.yaml for LLM runner config
        promptlab_yaml = cwd / "promptlab.yaml"
        if not promptlab_yaml.exists():
            console.print("[red]✗ No promptlab.yaml found. Run 'promptlab init' first.[/red]")
            raise typer.Exit(1)

        plconfig = load_config(promptlab_yaml)

        async def _run():
            runner = LLMRunner(plconfig.models.dict())

            # Build governance council from ft_config
            search_space = SearchSpace(**ft_config.get("search_space", {}))
            hardware = HardwareProfile(**ft_config.get("hardware_profile", {}))
            dataset = DatasetProfile(**ft_config.get("dataset_profile", {"name": "default", "num_samples": 0}))

            # Load run history
            history_path = cwd / ".promptlab" / "governance" / "history.json"
            run_history = []
            if history_path.exists():
                try:
                    entries = json_module.loads(history_path.read_text())
                    run_history = [RunHistoryEntry(**e) for e in entries]
                except Exception:
                    pass

            council = HyperparamGovernanceCouncil(
                llm_runner=runner,
                judge_models=ft_config.get("judge_models", []),
                search_space=search_space,
                hardware_profile=hardware,
                dataset_profile=dataset,
                tuning_goal=ft_config.get("tuning_goal", ""),
                task_domain=ft_config.get("task_domain", "general"),
                base_model=ft_config.get("base_model", ""),
                chairman_model=ft_config.get("chairman_model"),
                run_history=run_history,
            )

            if propose_only:
                # Step 1: Just propose
                console.print(Panel(
                    f"[bold]Goal:[/bold] {ft_config.get('tuning_goal', 'N/A')}\n"
                    f"[bold]Base model:[/bold] {ft_config.get('base_model', 'N/A')}\n"
                    f"[bold]History:[/bold] {len(run_history)} previous runs",
                    title="[bold cyan]Hyperparameter Proposer[/bold cyan]",
                    border_style="cyan",
                ))

                config_proposal = await council.propose_config()
                proposed_json = config_proposal.model_dump_json(indent=2)

                console.print(Panel(
                    proposed_json,
                    title="[bold green]Proposed Configuration[/bold green]",
                    border_style="green",
                ))

                if output_json:
                    out_path = Path(output_json)
                    out_path.write_text(proposed_json)
                    console.print(f"[green]✓ Saved to {out_path}[/green]")

                return None

            # Full evaluation mode — need artifacts
            if not artifacts_file:
                console.print("[red]✗ --artifacts is required for evaluation mode.[/red]")
                console.print("[dim]Use --propose to just get the next config suggestion.[/dim]")
                raise typer.Exit(1)

            artifacts_path = Path(artifacts_file)
            if not artifacts_path.exists():
                console.print(f"[red]✗ Artifacts file not found: {artifacts_path}[/red]")
                raise typer.Exit(1)

            artifact_data = json_module.loads(artifacts_path.read_text())
            artifact = RunArtifact(**artifact_data)

            console.print(Panel(
                f"[bold]Goal:[/bold] {ft_config.get('tuning_goal', 'N/A')}\n"
                f"[bold]Config ID:[/bold] {artifact.config.config_id}\n"
                f"[bold]Eval loss:[/bold] {artifact.metrics.final_eval_loss:.4f}\n"
                f"[bold]Judges:[/bold] {', '.join(ft_config.get('judge_models', [])[:3])}",
                title="[bold cyan]Governance Council[/bold cyan]",
                border_style="cyan",
            ))

            output = await council.evaluate_run(artifact)

            # Display verdict
            verdict_color = {
                "pass": "green",
                "conditional_pass": "yellow",
                "fail": "red",
            }.get(output.decision.verdict.value, "white")

            console.print(Panel(
                f"[bold]Verdict:[/bold] [{verdict_color}]{output.decision.verdict.value.upper()}[/{verdict_color}]\n"
                f"[bold]Score:[/bold] {output.decision.scores.final_score:.3f}\n"
                f"[bold]Confidence:[/bold] {output.decision.confidence:.3f}\n"
                f"[bold]Next:[/bold] {output.next_step.action.value}\n\n"
                f"{output.decision.summary}",
                title=f"[bold {verdict_color}]Governance Decision[/bold {verdict_color}]",
                border_style=verdict_color,
            ))

            # Save governance output
            if output_json:
                out_path = Path(output_json)
                out_path.write_text(output.to_json())
                console.print(f"[green]✓ Full output saved to {out_path}[/green]")

            # Save to history
            gov_dir = cwd / ".promptlab" / "governance"
            gov_dir.mkdir(parents=True, exist_ok=True)
            history = [e.model_dump() for e in council.run_history]
            history_path.write_text(json_module.dumps(history, indent=2))
            console.print(f"[dim]History updated: {len(history)} runs[/dim]")

            return output

        try:
            aio.run(_run())
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
