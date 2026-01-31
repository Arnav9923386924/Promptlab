"""Pipeline manager for multi-stage LLM testing."""

from typing import Optional
from pydantic import BaseModel
from pathlib import Path


class PipelineStage(BaseModel):
    """Configuration for a single pipeline stage."""
    name: str
    prompt_file: Optional[str] = None
    prompt_template: Optional[str] = None
    model: Optional[str] = None
    description: Optional[str] = None


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    stages: list[PipelineStage] = []


class PipelineManager:
    """Manages multi-stage LLM pipeline testing."""
    
    def __init__(self, config: dict, project_root: Path):
        """Initialize pipeline manager.
        
        Args:
            config: Pipeline configuration from promptlab.yaml
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.stages: dict[str, PipelineStage] = {}
        
        # Parse stages from config
        for stage_data in config.get("stages", []):
            stage = PipelineStage(**stage_data)
            self.stages[stage.name] = stage
    
    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a pipeline stage by name."""
        return self.stages.get(name)
    
    def get_stage_names(self) -> list[str]:
        """Get list of all stage names in order."""
        return list(self.stages.keys())
    
    def get_prompt_for_stage(self, stage_name: str) -> Optional[str]:
        """Get the prompt template for a stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Prompt template string or None
        """
        stage = self.get_stage(stage_name)
        if not stage:
            return None
        
        # If template is inline, return it
        if stage.prompt_template:
            return stage.prompt_template
        
        # If file is specified, load it
        if stage.prompt_file:
            prompt_path = self.project_root / stage.prompt_file
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
        
        return None
    
    def validate_pipeline(self) -> list[str]:
        """Validate pipeline configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for name, stage in self.stages.items():
            # Check that prompt is defined
            if not stage.prompt_file and not stage.prompt_template:
                errors.append(f"Stage '{name}' has no prompt_file or prompt_template")
            
            # Check that prompt file exists
            if stage.prompt_file:
                prompt_path = self.project_root / stage.prompt_file
                if not prompt_path.exists():
                    errors.append(f"Stage '{name}' prompt file not found: {stage.prompt_file}")
        
        return errors


def filter_tests_by_stage(test_files: list[Path], stage: str) -> list[Path]:
    """Filter test files to only those matching a specific stage.
    
    Args:
        test_files: List of test file paths
        stage: Stage name to filter by
        
    Returns:
        Filtered list of test files
    """
    from promptlab.orchestrators.parser import parse_test_file
    
    filtered = []
    for f in test_files:
        try:
            suite = parse_test_file(f)
            if suite.stage == stage:
                filtered.append(f)
        except Exception:
            pass  # Skip files that can't be parsed
    
    return filtered
