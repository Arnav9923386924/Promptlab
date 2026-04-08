# Council Logging Changes - Quiet Console with File Diagnostics

## Overview
Implemented quiet-by-default console output for council evaluations while preserving full diagnostics in separate log files under `.promptlab/runs/`.

## Changes Made

### 1. New Config Fields (config.py)
Added to `CouncilConfig`:
- **`verbose_attempts`** (bool, default=False): Show detailed per-model attempt logs on console
- **`log_attempts`** (bool, default=True): Write detailed attempt logs to file
- **`log_attempts_path`** (Optional[str], default=None): Custom log path (auto-generated if None)

### 2. New Logging Infrastructure (council.py)

#### CouncilAttemptsLogger Class
File-based logger for detailed judge attempt tracking:
- `log_attempt_start(model, is_configured)`: Log each model attempt
- `log_success(model, score)`: Log successful evaluations
- `log_failure(model, error)`: Log failed evaluations
- `log_rate_limit(model)`: Log rate-limit errors
- `log_collection_complete(total, configured, fallback)`: Log completion summary
- `log_insufficient_judges(got, required, configured_tried, total_available)`: Log errors
- `log_chunk_start(chunk_size)`: Log batch chunk starts

Logs are written to: `.promptlab/runs/<run_id>_judge_attempts.log`

#### Council Constructor Updates
- Added `run_id: Optional[str]` parameter
- Added `project_root: Optional[Path]` parameter
- Initialize `self.verbose_attempts` from config
- Initialize `self.attempts_logger` with auto-generated path

### 3. Console Output Updates

#### _stage1_judge() Method
**Before:** Every model attempt printed to console (noisy)
```
⚠ model-a rate-limited — trying next model
⚠ model-b failed: error message
✓ model-c scored 0.85
```

**After (default):** Only summary lines shown
```
🎯 Target: 2 judge scores (required minimum)
✓ Collected 2 judge scores (2 configured, 0 fallback)
```

**After (verbose_attempts=true):** Full output like before

#### _evaluate_single_batch() Method
Same pattern applied for batch evaluations.

### 4. Integration Updates

Updated Council instantiation in:
- **bsp_validator.py**: Passes `project_root` and new config fields
- **runner.py**: Passes new config fields (project_root=None)
- **All test files**: Pass `run_id=None, project_root=None` for tests

### 5. Config Template Updates

Updated with new fields and comments:
- `testing_env/tester/promptlab.yaml`
- `promptlab/promptlab.example.yaml`

## Usage Examples

### Default (Quiet Console)
```yaml
council:
  enabled: true
  members: [model-a, model-b]
  verbose_attempts: false  # Default
  log_attempts: true       # Default
```

**Console Output:**
```
🎯 Target: 2 judge scores (required minimum)
✓ Collected 2 judge scores (2 configured, 0 fallback)
```

**Log File** (`.promptlab/runs/bsp_20260102_143022_judge_attempts.log`):
```
=== Council Judge Attempts Log ===
Started: 2026-01-02T14:30:22

[ATTEMPT] model-a (source: configured)
[RATE_LIMIT] model-a → skipping to next model
[ATTEMPT] model-b (source: configured)
[SUCCESS] model-b → score=0.850
[ATTEMPT] model-c (source: fallback)
[SUCCESS] model-c → score=0.780

[COMPLETE] Collected 2 scores (1 configured, 1 fallback)
```

### Verbose Mode (Full Console)
```yaml
council:
  verbose_attempts: true  # Show all attempts on console
```

**Console Output:**
```
🎯 Target: 2 judge scores (required minimum)
📋 Configured members: 2, Available candidates: 10
⚠ model-a rate-limited — trying next model
✓ model-b scored 0.85
✓ model-c scored 0.78
✓ Collected 2 judge scores (1 configured, 1 fallback)
```

### Disable File Logging
```yaml
council:
  log_attempts: false  # No log file
```

### Custom Log Path
```yaml
council:
  log_attempts_path: /custom/path/my_log.txt
```

## Testing

All 48 tests passing:
- 12 required_judges tests (unchanged logic)
- 20 hardening tests
- 16 scaling tests

## Benefits

1. **Cleaner Console**: No more spam from rate-limited/failed models
2. **Full Diagnostics**: All attempt details preserved in log files
3. **Zero Breaking Changes**: Existing configs work with defaults
4. **Flexible**: Users can enable verbose mode when debugging
5. **Reproducible**: Log files track exact evaluation sequence

## Files Changed

- `src/promptlab/utils/config.py` - Added 3 config fields
- `src/promptlab/llm_council/council/council.py` - Added logger class, updated methods
- `src/promptlab/orchestrators/bsp_validator.py` - Updated Council instantiation
- `src/promptlab/orchestrators/runner.py` - Updated Council instantiation
- `tests/test_required_judges.py` - Updated test Council instantiations
- `tests/test_scaling.py` - Updated test Council instantiations
- `testing_env/tester/promptlab.yaml` - Added logging config
- `promptlab/promptlab.example.yaml` - Added logging config

## Backward Compatibility

✅ Fully backward compatible:
- `verbose_attempts` defaults to `false` (quiet by default)
- `log_attempts` defaults to `true` (diagnostics preserved)
- `log_attempts_path` defaults to `None` (auto-generates safe path)
- All existing configs work without modification
