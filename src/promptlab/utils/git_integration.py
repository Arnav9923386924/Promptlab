"""Git integration for automated push based on BSP validation scores.

This module handles:
1. Checking git status
2. Committing baseline updates
3. Pushing to remote when score improves
"""

import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class GitStatus:
    """Git repository status."""
    is_repo: bool
    branch: str = ""
    has_changes: bool = False
    changed_files: list[str] = None
    remote: str = ""
    
    def __post_init__(self):
        if self.changed_files is None:
            self.changed_files = []


class GitIntegration:
    """Handles git operations for BSP validation workflow."""
    
    def __init__(self, project_root: Path):
        """Initialize git integration.
        
        Args:
            project_root: Root directory of the git repository
        """
        self.project_root = project_root
    
    def _run_git(self, *args) -> tuple[bool, str]:
        """Run a git command.
        
        Args:
            *args: Git command arguments
            
        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0, result.stdout.strip()
        except FileNotFoundError:
            return False, "Git not found"
        except Exception as e:
            return False, str(e)
    
    def get_status(self) -> GitStatus:
        """Get current git repository status.
        
        Returns:
            GitStatus object
        """
        # Check if it's a git repo
        success, _ = self._run_git("rev-parse", "--is-inside-work-tree")
        if not success:
            return GitStatus(is_repo=False)
        
        # Get current branch
        success, branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if not success:
            branch = "unknown"
        
        # Check for changes
        success, status = self._run_git("status", "--porcelain")
        changed_files = [line[3:] for line in status.split("\n") if line.strip()]
        
        # Get remote
        success, remote = self._run_git("remote", "get-url", "origin")
        if not success:
            remote = ""
        
        return GitStatus(
            is_repo=True,
            branch=branch,
            has_changes=len(changed_files) > 0,
            changed_files=changed_files,
            remote=remote,
        )
    
    def stage_files(self, files: list[str]) -> bool:
        """Stage specific files for commit.
        
        Args:
            files: List of file paths to stage
            
        Returns:
            True if successful
        """
        for file in files:
            success, _ = self._run_git("add", file)
            if not success:
                console.print(f"[red]Failed to stage: {file}[/red]")
                return False
        return True
    
    def stage_baseline_files(self) -> bool:
        """Stage baseline-related files.
        
        Returns:
            True if successful
        """
        files_to_stage = [
            ".promptlab/baselines/",
            "promptlab.yaml",
        ]
        
        for file in files_to_stage:
            success, _ = self._run_git("add", file)
            # Don't fail if file doesn't exist
        
        return True
    
    def commit(
        self,
        message: str,
        score: Optional[float] = None,
        previous_score: Optional[float] = None,
    ) -> bool:
        """Create a commit with the given message.
        
        Args:
            message: Commit message (supports {score}, {previous_score}, {improvement} placeholders)
            score: Current score for placeholder
            previous_score: Previous score for placeholder
            
        Returns:
            True if successful
        """
        # Format message with placeholders
        improvement = (score - previous_score) if score and previous_score else 0
        message = message.format(
            score=score or 0,
            previous_score=previous_score or 0,
            improvement=improvement,
        )
        
        success, output = self._run_git("commit", "-m", message)
        
        if success:
            console.print(f"[green]✓ Committed: {message}[/green]")
        else:
            if "nothing to commit" in output:
                console.print("[yellow]Nothing to commit[/yellow]")
                return True
            console.print(f"[red]Commit failed: {output}[/red]")
        
        return success
    
    def push(self, branch: Optional[str] = None, force: bool = False) -> bool:
        """Push to remote.
        
        Args:
            branch: Branch to push (default: current branch)
            force: Whether to force push
            
        Returns:
            True if successful
        """
        args = ["push"]
        
        if branch:
            args.extend(["origin", branch])
        
        if force:
            args.append("--force")
        
        success, output = self._run_git(*args)
        
        if success:
            console.print("[green]✓ Pushed to remote[/green]")
        else:
            console.print(f"[red]Push failed: {output}[/red]")
        
        return success
    
    def commit_and_push(
        self,
        score: float,
        previous_score: Optional[float] = None,
        commit_template: str = "chore: BSP validation passed (score: {score:.2f})",
        branch: Optional[str] = None,
        auto_push: bool = True,
    ) -> bool:
        """Stage, commit, and optionally push baseline updates.
        
        Args:
            score: Current validation score
            previous_score: Previous baseline score
            commit_template: Commit message template
            branch: Branch to push to
            auto_push: Whether to push after commit
            
        Returns:
            True if all operations successful
        """
        status = self.get_status()
        
        if not status.is_repo:
            console.print("[yellow]Not a git repository. Skipping git operations.[/yellow]")
            return False
        
        # Stage baseline files
        console.print("[cyan]Staging baseline files...[/cyan]")
        if not self.stage_baseline_files():
            return False
        
        # Commit
        console.print("[cyan]Creating commit...[/cyan]")
        if not self.commit(commit_template, score, previous_score):
            return False
        
        # Push
        if auto_push:
            console.print("[cyan]Pushing to remote...[/cyan]")
            return self.push(branch)
        
        return True
    
    def should_push(
        self,
        score: float,
        previous_score: Optional[float],
        min_improvement: float = 0.0,
    ) -> tuple[bool, str]:
        """Determine if we should push based on score improvement.
        
        Args:
            score: Current score
            previous_score: Previous baseline score
            min_improvement: Minimum improvement required
            
        Returns:
            Tuple of (should_push, reason)
        """
        if previous_score is None:
            return True, "No baseline exists. This will be the first baseline."
        
        improvement = score - previous_score
        
        if improvement > min_improvement:
            return True, f"Score improved by {improvement:.2f} (threshold: {min_improvement:.2f})"
        elif improvement == 0:
            return False, "Score unchanged from baseline."
        else:
            return False, f"Score regressed by {abs(improvement):.2f}. Not pushing."
