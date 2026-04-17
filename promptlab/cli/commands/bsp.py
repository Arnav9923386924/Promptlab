"""CLI commands — BSP validation, linting, optimization & test generation."""

import typer
from rich.panel import Panel
from pathlib import Path
from typing import Optional

from promptlab.cli.helpers import console, is_interactive, ensure_initialized


def register(bsp_app: typer.Typer, app: typer.Typer):
    """Register BSP commands on both the bsp sub-app and root app (hidden aliases)."""

    @bsp_app.callback(invoke_without_command=True)
    def bsp_default(ctx: typer.Context):
        """BSP validation, linting, optimization & test generation."""
        if ctx.invoked_subcommand is None:
            # Default: run validation
            ctx.invoke(validate_bsp)

    @bsp_app.command("run")
    @app.command("validate", hidden=True)
    def validate_bsp(
        test_dir: str = typer.Option("temp", "--dir", "-d", help="Directory containing test files"),
        files: list[str] = typer.Option(None, "--file", "-f", help="Specific test files to run"),
        save_baseline: bool = typer.Option(True, "--save-baseline/--no-save-baseline", help="Save as new baseline if improved"),
        auto_push: bool = typer.Option(False, "--push", "-p", help="Auto-push to git if score improves"),
        ci: bool = typer.Option(False, "--ci", help="CI mode - exit with code based on validation result"),
        output_json: str = typer.Option(None, "--output", "-o", help="Save validation result to JSON file"),
        generate: int = typer.Option(None, "--generate", "-g", help="Auto-generate N test cases via web scraping (default: 50 if no tests exist)"),
        no_generate: bool = typer.Option(False, "--no-generate", help="Disable auto-generation even if no tests exist"),
        no_interactive: bool = typer.Option(False, "--no-interactive", help="Skip interactive model selection (use config defaults)"),
    ):
        """Validate your Behavior Specification Prompt (BSP) against test cases.

        This command runs the complete BSP validation workflow:
        1. Load BSP from config (promptlab.yaml)
        2. Auto-generate tests if none exist (via web scraping)
        3. Run all test cases with BSP prepended
        4. Collect outputs for council review
        5. Submit to LLM Council for batch evaluation
        6. Compare score with baseline
        7. Optionally push to git if improved

        Examples:
          promptlab validate                        # Run validation (auto-generates if no tests)
          promptlab validate --generate 100         # Generate 100 tests via scraping
          promptlab validate --no-generate          # Skip auto-generation, require manual tests
          promptlab validate --push                 # Push to git if improved
          promptlab validate --ci                   # CI mode for GitHub Actions
          promptlab validate -f temp/my_test.yaml   # Validate specific file
        """
        import asyncio
        import json as json_module
        from promptlab.utils.config import load_config, load_bsp
        from promptlab.bsp.validator import BSPValidator
        from promptlab.bsp.baseline import BaselineManager
        from promptlab.utils.git_integration import GitIntegration
        from promptlab.utils.model_selector import discover_models, run_model_selection

        cwd = Path.cwd()
        config_path = ensure_initialized(cwd)
        test_path = cwd / test_dir

        # Load config
        config = load_config(config_path)

        # ── Interactive model/role selection ──
        # In CI mode or with --no-interactive, skip the interactive UI
        skip_interactive = ci or no_interactive
        if config.council.enabled and not skip_interactive and is_interactive():
            try:
                # Step 1: Discover models (async — needs an event loop)
                _loop = asyncio.new_event_loop()
                try:
                    available = _loop.run_until_complete(discover_models(config))
                finally:
                    _loop.close()

                # Step 2: InquirerPy interactive selection (sync — NO event loop running)
                judges, chairman, model_roles, require_all_selected, selected_required_judges = run_model_selection(config, available)

                # Apply user selections back to config
                if judges:
                    config.council.members = judges
                    config.council.required_judges = selected_required_judges
                    config.council.use_fixed_judges = require_all_selected
                    mode_text = "strict" if require_all_selected else "flexible"
                    console.print(
                        f"[dim]Updated: required_judges={selected_required_judges}, "
                        f"use_fixed_judges={str(require_all_selected)} ({mode_text} mode)[/dim]"
                    )
                if chairman:
                    config.council.chairman = chairman
                if model_roles:
                    config.council.model_roles = model_roles
            except Exception as e:
                console.print(f"[yellow]Interactive selection failed ({e}) — using config defaults[/yellow]")

        # Check if BSP is configured
        bsp = load_bsp(config, cwd)
        if not bsp:
            console.print("[yellow]⚠️  No BSP configured. Add 'bsp' section to promptlab.yaml[/yellow]")
            console.print("[dim]Example:[/dim]")
            console.print("[dim]  bsp:[/dim]")
            console.print("[dim]    prompt: 'You are a helpful assistant...'[/dim]")
            console.print("[dim]    min_score: 0.7[/dim]")
            raise typer.Exit(1)

        # Resolve effective test count: --generate flag > config > default 50
        if generate is not None:
            effective_count = generate
        elif config.bsp and config.bsp.auto_generate_count:
            effective_count = config.bsp.auto_generate_count
        else:
            effective_count = 50

        gen_mode = config.bsp.generation_mode if config.bsp else "web"

        # ── Interactive generation-mode selection (InquirerPy) ──
        if not no_generate and not skip_interactive and is_interactive():
            try:
                from InquirerPy import inquirer
                from InquirerPy.separator import Separator

                mode_choices = [
                    {
                        "name": "web       — Scrape web pages -> regex extract Q&A / cloze tests (fast)",
                        "value": "web",
                    },
                    {
                        "name": "docs_web  — Download docs -> index -> retrieve -> generate (higher quality)",
                        "value": "docs_web",
                    },
                    {
                        "name": "hybrid    — docs_web first, web scraping fallback if target not met",
                        "value": "hybrid",
                    },
                ]

                selected_mode = inquirer.select(
                    message="Select test generation mode:",
                    choices=mode_choices,
                    default=gen_mode,
                    cycle=True,
                    instruction="(arrow keys to navigate, Enter to confirm)",
                ).execute()

                if selected_mode:
                    gen_mode = selected_mode
                    if config.bsp:
                        config.bsp.generation_mode = gen_mode
                    console.print(f"[dim]Generation mode: {gen_mode}[/dim]\n")
            except ImportError:
                console.print("[dim]InquirerPy not installed — using config default mode[/dim]")
            except (KeyboardInterrupt, EOFError):
                console.print("[dim]Mode selection cancelled — using config default[/dim]")
            except Exception:
                pass  # fall through to config default

        # ── Interactive: ask whether to regenerate tests if they already exist ──
        force_regenerate = False
        if not no_generate and not skip_interactive and is_interactive():
            from promptlab.bsp.parser import discover_test_files
            existing_tests = discover_test_files(test_path)
            if existing_tests:
                try:
                    from InquirerPy import inquirer

                    console.print(f"\n[bold]Found {len(existing_tests)} existing test file(s) in {test_path}[/bold]")
                    regen = inquirer.confirm(
                        message="Generate new test cases? (existing tests will be removed)",
                        default=False,
                    ).execute()

                    if regen:
                        # Ask how many tests to generate
                        count_input = inquirer.number(
                            message="How many test cases to generate?",
                            default=effective_count,
                            min_allowed=15,
                            max_allowed=500,
                        ).execute()
                        effective_count = int(count_input)
                        force_regenerate = True

                        # Remove old test files
                        for f in existing_tests:
                            try:
                                f.unlink()
                            except Exception:
                                pass
                        console.print(f"[dim]Removed {len(existing_tests)} old test file(s). Will generate {effective_count} new tests.[/dim]\n")
                    else:
                        console.print(f"[dim]Using {len(existing_tests)} existing test file(s).[/dim]\n")
                        no_generate = True  # skip auto-generation since user wants existing tests
                except ImportError:
                    pass
                except (KeyboardInterrupt, EOFError):
                    console.print("[dim]Cancelled — using existing tests[/dim]")
                    no_generate = True
                except Exception:
                    pass
            else:
                # No existing tests — ask for count
                try:
                    from InquirerPy import inquirer

                    console.print(f"\n[bold]No existing test files in {test_path}[/bold]")
                    count_input = inquirer.number(
                        message="How many test cases to generate?",
                        default=effective_count,
                        min_allowed=15,
                        max_allowed=500,
                    ).execute()
                    effective_count = int(count_input)
                except ImportError:
                    pass
                except (KeyboardInterrupt, EOFError):
                    console.print("[dim]Using default count[/dim]")
                except Exception:
                    pass

        console.print(Panel(
            f"[bold]BSP Version:[/bold] {config.bsp.version if config.bsp else 'default'}\n"
            f"[bold]Model:[/bold] {config.models.default}\n"
            f"[bold]Council:[/bold] {'enabled' if config.council.enabled else 'disabled'}\n"
            f"[bold]Test Dir:[/bold] {test_path}\n"
            f"[bold]Auto Generate:[/bold] {'disabled' if no_generate else f'{effective_count} tests ({gen_mode} mode)'}\n"
            f"[bold]Auto Push:[/bold] {'yes' if auto_push else 'no'}",
            title="BSP Validation",
            border_style="blue",
        ))

        async def run_validation():
            # Create validator
            validator = BSPValidator(config, cwd)

            # Determine auto-generation settings
            should_auto_generate = not no_generate
            target_count = effective_count

            # Run validation
            result = await validator.validate(
                test_dir=test_path,
                test_files=list(files) if files else None,
                auto_generate=should_auto_generate,
                generate_count=target_count,
            )

            # Ask chairman for BSP improvements (same event loop), regardless of pass/fail.
            bsp_suggestion = None
            if not ci and validator.council:
                batch = getattr(validator, '_last_batch', None)
                council_res = result.council_result
                if batch and council_res:
                    try:
                        bsp_suggestion = await validator.get_bsp_improvement(batch, council_res)
                    except Exception as e:
                        from rich.console import Console as _C
                        _C().print(f"[yellow]  BSP improvement suggestion failed: {e}[/yellow]")

            return result, validator, bsp_suggestion

        try:
            # Use a fresh event loop to avoid "Event loop is closed" on Windows
            _val_loop = asyncio.new_event_loop()
            try:
                result, validator, bsp_suggestion = _val_loop.run_until_complete(run_validation())
            finally:
                _val_loop.close()

            # Record evaluation history
            from promptlab.utils.history import EvaluationHistory
            history = EvaluationHistory(cwd)

            # Extract detailed scores if available
            instruction_following = 0.0
            helpfulness = 0.0
            coherence = 0.0
            confidence = "medium"
            weak_areas = []

            if result.council_result:
                confidence = result.council_result.confidence
                # Try to extract subscores if they exist
                if hasattr(result.council_result, 'instruction_following'):
                    instruction_following = result.council_result.instruction_following
                if hasattr(result.council_result, 'helpfulness'):
                    helpfulness = result.council_result.helpfulness
                if hasattr(result.council_result, 'coherence'):
                    coherence = result.council_result.coherence
                if hasattr(result.council_result, 'weak_areas'):
                    weak_areas = result.council_result.weak_areas or []

            history.record(
                overall_score=result.council_score,
                bsp_version=result.bsp_version or config.bsp.version if config.bsp else "1.0.0",
                bsp_hash=result.bsp_hash[:16] if result.bsp_hash else "",
                model=result.model or config.models.default,
                instruction_following=instruction_following,
                helpfulness=helpfulness,
                coherence=coherence,
                confidence=confidence,
                total_tests=result.total_tests,
                weak_areas=weak_areas,
                notes=f"Baseline: {result.baseline_score:.5f}" if result.baseline_score else "",
            )

            # Save baseline if improved
            if save_baseline and result.should_push:
                baseline_manager = BaselineManager(cwd / ".promptlab")
                baseline_manager.save_bsp_baseline(
                    run_id=result.run_id,
                    score=result.council_score,
                    bsp_hash=result.bsp_hash,
                    bsp_version=result.bsp_version,
                    model=result.model,
                    total_tests=result.total_tests,
                    passed=result.passed,
                    confidence=result.council_result.confidence if result.council_result else "medium",
                    tag=f"bsp_{result.bsp_version}_{result.bsp_hash[:8]}",
                )
                console.print("[green]✓ Saved new baseline[/green]")

            # Auto-push to git if enabled
            if auto_push and result.should_push:
                git = GitIntegration(cwd)
                commit_template = config.git.commit_template if config.git else "chore: BSP validation passed (score: {score:.2f})"
                git.commit_and_push(
                    score=result.council_score,
                    previous_score=result.baseline_score,
                    commit_template=commit_template,
                    branch=config.git.branch if config.git else None,
                    auto_push=config.git.auto_push if config.git else True,
                )
            elif auto_push and not result.should_push:
                console.print("[yellow]Not pushing - score did not improve sufficiently[/yellow]")

            # Save JSON output if requested
            if output_json:
                output_path = Path(output_json)
                with open(output_path, "w", encoding="utf-8") as f:
                    json_module.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                console.print(f"[green]✓ Saved result to {output_path}[/green]")

            # Final summary
            console.print()
            if result.passed:
                console.print(Panel(
                    f"[bold green]✓ Validation PASSED[/bold green]\n\n"
                    f"Score: {result.council_score:.5f}\n"
                    f"{'Improvement: ' + f'+{result.improvement:.5f}' if result.improvement and result.improvement > 0 else ''}\n"
                    f"Outputs: {result.outputs_file}",
                    title="Success",
                    border_style="green",
                ))
            else:
                console.print(Panel(
                    f"[bold red]✗ Validation FAILED[/bold red]\n\n"
                    f"Score: {result.council_score:.5f}\n"
                    f"Min Required: {(config.bsp.min_score if config.bsp else 0.7):.5f}\n"
                    f"Outputs: {result.outputs_file}",
                    title="❌ Failed",
                    border_style="red",
                ))

            # ----------------------------------------------------------------
            # BSP IMPROVEMENT SUGGESTION (non-CI only, regardless of pass/fail)
            # ----------------------------------------------------------------
            if not ci and bsp_suggestion:
                console.print()
                # Display suggested changes
                console.print(Panel(
                    "\n".join(f"  • {c}" for c in bsp_suggestion.changes),
                    title="Suggested Changes",
                    border_style="yellow",
                ))

                # Show a preview of the improved BSP (first 500 chars)
                preview = bsp_suggestion.improved_bsp[:500]
                if len(bsp_suggestion.improved_bsp) > 500:
                    preview += "\n... (truncated)"
                console.print(Panel(
                    preview,
                    title="Improved BSP Preview",
                    border_style="cyan",
                ))

                # Prompt the user
                bsp_file_path = None
                if config.bsp and config.bsp.prompt_file:
                    bsp_file_path = cwd / config.bsp.prompt_file

                if bsp_file_path:
                    update_bsp = typer.confirm(
                        f"\nWould you like to update {config.bsp.prompt_file} with the improved BSP?",
                        default=False,
                    )

                    if update_bsp:
                        # Backup current BSP
                        backup_path = bsp_file_path.with_suffix(".bsp.bak")
                        if bsp_file_path.exists():
                            backup_path.write_text(
                                bsp_file_path.read_text(encoding="utf-8"),
                                encoding="utf-8",
                            )
                            console.print(f"[dim]  Backed up current BSP to {backup_path.name}[/dim]")

                        bsp_file_path.write_text(bsp_suggestion.improved_bsp, encoding="utf-8")
                        console.print(f"[green]✓ Updated {config.bsp.prompt_file} with improved BSP[/green]")
                        console.print("[yellow]  Run 'promptlab validate' again to check the new score.[/yellow]")
                    else:
                        console.print("[dim]BSP update declined - terminating without changes.[/dim]")
                        raise typer.Exit(0)
                else:
                    console.print("[yellow]  ⚠ BSP is inline (not a file) — cannot auto-update. Copy the improved BSP manually.[/yellow]")
                    console.print(Panel(
                        bsp_suggestion.improved_bsp,
                        title="Full Improved BSP (copy this)",
                        border_style="green",
                    ))

            # Exit code for CI
            if ci:
                raise typer.Exit(0 if result.passed else 1)

        except Exception as e:
            console.print(f"[red]Error during validation: {e}[/red]")
            if ci:
                raise typer.Exit(2)
            raise typer.Exit(1)

    @bsp_app.command("lint")
    @app.command("lint-bsp", hidden=True)
    def lint_bsp(
        bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file (overrides promptlab.yaml)"),
    ):
        """Lint your BSP for quality issues — completely FREE, no API calls.

        Checks for vague language, missing sections, contradictions,
        and structural problems. Returns a quality score and actionable fixes.

        Examples:
          promptlab lint-bsp                  # Lint BSP from promptlab.yaml
          promptlab lint-bsp -b my_bsp.txt    # Lint a specific BSP file
        """
        from promptlab.utils.config import load_config, load_bsp
        from promptlab.bsp.linter import BSPLinter

        cwd = Path.cwd()

        # Load BSP
        if bsp_file:
            bsp_path = Path(bsp_file)
            if not bsp_path.exists():
                console.print(f"[red]✗ BSP file not found: {bsp_file}[/red]")
                raise typer.Exit(1)
            bsp = bsp_path.read_text(encoding="utf-8")
        else:
            config_path = ensure_initialized(cwd)
            config = load_config(config_path)
            bsp = load_bsp(config, cwd)
            if not bsp:
                console.print("[red]✗ No BSP configured. Add 'bsp' section to promptlab.yaml[/red]")
                raise typer.Exit(1)

        linter = BSPLinter()
        result = linter.lint(bsp)
        report = linter.format_report(result)

        if result.ready_for_evaluation:
            console.print(Panel(report, title="BSP Lint — PASSED ✓", border_style="green"))
        else:
            console.print(Panel(report, title="BSP Lint — FIX REQUIRED ✗", border_style="red"))
            raise typer.Exit(1)

    @bsp_app.command("opt")
    @app.command("optimize-bsp", hidden=True)
    def optimize_bsp(
        bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
        model: str = typer.Option(None, "--model", "-m", help="Model for optimization"),
        max_iterations: int = typer.Option(3, "--iterations", "-i", help="Max optimization iterations"),
        target_score: float = typer.Option(0.90, "--target", "-t", help="Target lint score"),
        save: bool = typer.Option(True, "--save/--no-save", help="Save optimized BSP to file"),
    ):
        """Optimize your BSP iteratively using AI + linting feedback.

        Cost: 1-3 API calls total (one per iteration). Pre-validates with
        the free BSP linter before and after each optimization.

        Examples:
          promptlab optimize-bsp                       # Optimize BSP from config
          promptlab optimize-bsp -b bsp.txt -i 5       # 5 iterations on specific file
          promptlab optimize-bsp --target 0.95         # Aim for 0.95 lint score
        """
        import asyncio as aio
        from promptlab.utils.config import load_config, load_bsp
        from promptlab.llm_council.llm_runner.runner import LLMRunner
        from promptlab.bsp.optimizer import BSPOptimizer

        cwd = Path.cwd()
        config_path = ensure_initialized(cwd)

        config = load_config(config_path)

        if bsp_file:
            bsp_path = Path(bsp_file)
            bsp = bsp_path.read_text(encoding="utf-8")
        else:
            bsp = load_bsp(config, cwd)
            bsp_path = None
            if not bsp:
                console.print("[red]✗ No BSP configured.[/red]")
                raise typer.Exit(1)

        opt_model = model or config.optimizer.optimizer_model or config.models.default

        async def _run():
            runner = LLMRunner(config.models.dict())
            optimizer = BSPOptimizer(
                llm_runner=runner,
                optimizer_model=opt_model,
                max_iterations=max_iterations,
                target_lint_score=target_score,
            )

            console.print(Panel(
                f"[bold]Model:[/bold] {opt_model}\n"
                f"[bold]Max Iterations:[/bold] {max_iterations}\n"
                f"[bold]Target Score:[/bold] {target_score}",
                title="BSP Optimizer",
                border_style="blue",
            ))

            result = await optimizer.optimize(bsp)
            report = optimizer.format_report(result)
            console.print(Panel(report, title="Optimization Result", border_style="green" if result.improvement > 0 else "yellow"))

            return result

        try:
            result = aio.run(_run())

            if save and result.improvement > 0:
                output_path = bsp_path or (cwd / "bsp_optimized.txt")
                if bsp_path:
                    # Backup original
                    backup = bsp_path.with_suffix(".bak")
                    backup.write_text(bsp, encoding="utf-8")
                    console.print(f"[dim]Original backed up to {backup}[/dim]")
                output_path.write_text(result.optimized_bsp, encoding="utf-8")
                console.print(f"[green]✓ Optimized BSP saved to {output_path}[/green]")
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    @bsp_app.command("gen")
    @app.command("scraper", hidden=True)
    def scraper_generate_tests(
        bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file (default: bsp.txt in cwd, then config)"),
        count: Optional[int] = typer.Option(None, "--count", "-n", help="Number of test cases to generate (default: auto_generate_count from config, else 50)"),
        output_dir: str = typer.Option("temp", "--output-dir", "-d", help="Directory to save generated test cases"),
        mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Generation mode: 'web', 'docs_web', or 'hybrid' (prompted if omitted)"),
        output_type: str = typer.Option("all", "--type", "-t", help="Test output type: 'benchmark' (Q&A), 'cloze' (masked), or 'all'"),
        max_pages: int = typer.Option(20, "--max-pages", help="Maximum web pages to scrape (web/hybrid modes)"),
        max_docs: int = typer.Option(20, "--max-docs", help="Maximum documents to download (docs_web/hybrid modes)"),
        chunk_size: int = typer.Option(800, "--chunk-size", help="Chunk size in words for document indexing"),
        retrieval_top_k: int = typer.Option(10, "--top-k", help="Top-K chunks per retrieval query"),
    ):
        """Generate test cases from your BSP — no validation run needed.

        Reads the Behavior Specification Prompt (BSP) from bsp.txt (or promptlab.yaml),
        analyzes it to extract domain/keywords, then generates ready-to-use YAML test
        files saved to the output directory.

        If --mode is not supplied you will be prompted to choose interactively.

        Generation modes:
          web      — scrape web pages → regex extract Q&A / cloze tests (original)
          docs_web — download docs → TF-IDF index → retrieve → LLM/heuristic generate
                     (higher quality, provenance metadata on every case)
          hybrid   — docs_web first, web fallback if target not met

        Examples:
          promptlab scraper                              # interactive mode picker
          promptlab scraper --mode web --count 50        # skip prompt, web mode
          promptlab scraper --mode docs_web --count 100  # 100 doc-grounded tests
          promptlab scraper --mode hybrid                # docs_web + web fallback
          promptlab scraper --bsp-file path/to/bsp.txt   # Use a specific BSP file
          promptlab scraper --mode docs_web --top-k 15   # More retrieval diversity
          promptlab scraper --type benchmark             # Q&A pairs only
          promptlab scraper --output-dir my_tests        # Save to custom directory
        """
        import asyncio as aio
        import yaml

        # ------------------------------------------------------------------ #
        # 0. Interactive mode selection (only when --mode not passed)          #
        # ------------------------------------------------------------------ #
        if mode is None:
            console.print()
            console.print(Panel(
                "[bold]1.[/bold] [cyan]web[/cyan]      — Scrape web pages → regex extract Q&A / cloze tests\n"
                "              Fast, no downloads, original pipeline.\n\n"
                "[bold]2.[/bold] [green]docs_web[/green] — Download docs → index → retrieve → generate\n"
                "              Higher quality, provenance on every testcase.\n\n"
                "[bold]3.[/bold] [yellow]hybrid[/yellow]   — docs_web first, web scraping fallback if target not met.",
                title="[bold cyan]Choose generation mode[/bold cyan]",
                border_style="cyan",
            ))
            choice = typer.prompt(
                "Enter choice",
                default="1",
            ).strip()
            mode = {"1": "web", "2": "docs_web", "3": "hybrid"}.get(
                choice, choice.lower()
            )
            if mode not in ("web", "docs_web", "hybrid"):
                console.print(f"[red]✗ Unknown mode '{mode}'. Choose 1, 2, or 3.[/red]")
                raise typer.Exit(1)
            console.print(f"[dim]Mode selected: {mode}[/dim]\n")

        from promptlab.utils.config import load_config, load_bsp
        from promptlab.testgen.generator import AutoTestGenerator

        cwd = Path.cwd()

        # ------------------------------------------------------------------ #
        # 1. Resolve BSP                                                       #
        # ------------------------------------------------------------------ #
        bsp: str = ""

        if bsp_file:
            bsp_path = Path(bsp_file)
            if not bsp_path.exists():
                console.print(f"[red]✗ BSP file not found: {bsp_path}[/red]")
                raise typer.Exit(1)
            bsp = bsp_path.read_text(encoding="utf-8").strip()
            console.print(f"[dim]Using BSP from: {bsp_path}[/dim]")
        else:
            # Prefer an explicit bsp.txt in the current directory
            default_bsp_txt = cwd / "bsp.txt"
            if default_bsp_txt.exists():
                bsp = default_bsp_txt.read_text(encoding="utf-8").strip()
                console.print(f"[dim]Using BSP from: {default_bsp_txt}[/dim]")
            else:
                # Fall back to promptlab.yaml config
                config_path = cwd / "promptlab.yaml"
                if config_path.exists():
                    config = load_config(config_path)
                    bsp = load_bsp(config, cwd) or ""
                    if bsp:
                        console.print("[dim]Using BSP from promptlab.yaml[/dim]")

        if not bsp:
            console.print(
                "[red]✗ No BSP found.[/red]\n"
                "[dim]Provide one via --bsp-file, place a bsp.txt in the current directory, "
                "or configure 'bsp.prompt' / 'bsp.prompt_file' in promptlab.yaml.[/dim]"
            )
            raise typer.Exit(1)

        # ------------------------------------------------------------------ #
        # 2. Resolve output directory & API keys                               #
        # ------------------------------------------------------------------ #
        out_dir = cwd / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        serpapi_key: str | None = None
        brave_api_key: str | None = None
        llm_model: str = "ollama/llama3.1:8b"
        scraper_timeout: float = 30.0
        config_path = cwd / "promptlab.yaml"
        config = None
        if config_path.exists():
            try:
                config = load_config(config_path)
                serpapi_key = config.scraper.serpapi_key if config.scraper else None
                brave_api_key = config.scraper.brave_api_key if config.scraper else None
                scraper_timeout = float(config.scraper.timeout) if config.scraper and config.scraper.timeout is not None else 30.0
                llm_model = config.docs_web.llm_model or config.models.default
                # Pull count from config when not supplied on CLI
                if count is None and config.bsp and config.bsp.auto_generate_count:
                    count = config.bsp.auto_generate_count
                # Pull docs_web defaults from config when not overridden on CLI
                if max_docs == 20 and config.docs_web:
                    max_docs = config.docs_web.max_docs
                if chunk_size == 800 and config.docs_web:
                    chunk_size = config.docs_web.chunk_size
                if retrieval_top_k == 10 and config.docs_web:
                    retrieval_top_k = config.docs_web.retrieval_top_k
            except Exception:
                pass

        import os
        serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
        brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        # Final fallback for count
        count = count or 50

        # ------------------------------------------------------------------ #
        # 3. Display plan                                                      #
        # ------------------------------------------------------------------ #
        mode_detail = {
            "web": "web scraping → regex extract",
            "docs_web": "download docs → index → retrieve → generate (provenance)",
            "hybrid": "docs_web first, web fallback",
        }.get(mode, mode)

        plan_lines = [
            f"[bold]BSP preview:[/bold] {bsp[:120].strip()}{'...' if len(bsp) > 120 else ''}",
            f"[bold]Target tests:[/bold] {count}",
            f"[bold]Mode:[/bold] {mode} — {mode_detail}",
            f"[bold]Output type:[/bold] {output_type}",
        ]
        if mode in ("web", "hybrid"):
            plan_lines.append(f"[bold]Max pages:[/bold] {max_pages}")
        if mode in ("docs_web", "hybrid"):
            plan_lines.append(f"[bold]Max docs:[/bold] {max_docs}")
            plan_lines.append(f"[bold]Chunk size:[/bold] {chunk_size} words")
            plan_lines.append(f"[bold]Top-K:[/bold] {retrieval_top_k}")
            plan_lines.append(f"[bold]LLM model:[/bold] {llm_model}")
        plan_lines.append(f"[bold]Output dir:[/bold] {out_dir}")
        plan_lines.append(f"[bold]SerpAPI:[/bold] {'configured' if serpapi_key else 'not set (using fallback search)'}")

        console.print(Panel(
            "\n".join(plan_lines),
            title="[bold cyan]Scraper — Test Case Generation[/bold cyan]",
            border_style="cyan",
        ))

        # ------------------------------------------------------------------ #
        # 4. Run generation                                                    #
        # ------------------------------------------------------------------ #
        async def _run():
            # Build LLM runner for docs_web / hybrid when a model config exists
            llm_runner = None
            if mode in ("docs_web", "hybrid") and config is not None:
                try:
                    from promptlab.llm_council.llm_runner.runner import LLMRunner
                    runner_config = {
                        "default": llm_model,
                        "providers": {
                            name: {"endpoint": p.endpoint, "api_key": p.api_key}
                            for name, p in config.models.providers.items()
                        } if config.models.providers else {},
                    }
                    llm_runner = LLMRunner(runner_config)
                except Exception:
                    pass  # fall back to heuristic generation

            generator = AutoTestGenerator(
                serpapi_key=serpapi_key,
                brave_api_key=brave_api_key,
                max_pages=max_pages,
                project_root=cwd,
                llm_runner=llm_runner,
                llm_model=llm_model,
                max_docs=max_docs,
                chunk_size=chunk_size,
                retrieval_top_k=retrieval_top_k,
                scraper_timeout=scraper_timeout,
            )
            return await generator.generate_tests(
                bsp=bsp,
                target_count=count,
                output_dir=out_dir,
                output_type=output_type,
                generation_mode=mode,
            )

        try:
            result = aio.run(_run())
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            console.print(f"[red]✗ Scraper failed: {e}[/red]")
            raise typer.Exit(1)

        # ------------------------------------------------------------------ #
        # 5. Summary                                                           #
        # ------------------------------------------------------------------ #
        if result.generation_mode_used in ("docs_web", "hybrid"):
            doc_count = len(result.generated_cases)
            web_count = len(result.qa_pairs) + len(result.masked_tests)
            total = doc_count + web_count
            summary_lines = [
                f"[bold green]✓ Generated {total} test cases[/bold green]",
                "",
                f"  Doc-grounded cases  : {doc_count}",
            ]
            if web_count:
                summary_lines.append(f"  Web Q&A + cloze     : {web_count}")
            summary_lines += [
                f"  Sources downloaded   : {result.scraped_sources}",
                f"  Mode                : {result.generation_mode_used}",
                f"  Generation time     : {result.generation_time:.1f}s",
                "",
                f"  Saved to: [bold]{result.output_file}[/bold]",
                "",
                "[dim]Run 'promptlab validate' to evaluate your BSP against these tests.[/dim]",
            ]
        else:
            total = len(result.qa_pairs) + len(result.masked_tests)
            summary_lines = [
                f"[bold green]✓ Generated {total} test cases[/bold green]",
                "",
                f"  Q&A (benchmark) pairs : {len(result.qa_pairs)}",
                f"  Masked (cloze) tests  : {len(result.masked_tests)}",
                f"  Web pages scraped     : {result.scraped_sources}",
                f"  Generation time       : {result.generation_time:.1f}s",
                "",
                f"  Saved to: [bold]{result.output_file}[/bold]",
                "",
                "[dim]Run 'promptlab validate' to evaluate your BSP against these tests.[/dim]",
            ]

        console.print()
        console.print(Panel(
            "\n".join(summary_lines),
            title="[bold green]Scraper Complete[/bold green]",
            border_style="green",
        ))
