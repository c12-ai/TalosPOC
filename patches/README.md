# Third-Party Package Patches

This directory contains patches for bugs in third-party dependencies.

## Patches Included

### 1. CopilotKit emit_messages fix

**Issue**: [CopilotKit #2066](https://github.com/CopilotKit/CopilotKit/issues/2066)

**Problem**: `copilotkit_customize_config(config, emit_messages=False)` does not work -
TEXT_MESSAGE_* events are still emitted despite the configuration.

**Root Causes**:

1. **Metadata Reading Bug**: In `langgraph_agui_agent.py`, the code uses
   `getattr(raw_event, 'metadata', {})` to read metadata, but `raw_event` is a dict,
   not an object. This should use `.get()` instead.

2. **Encoder Crash Bug**: When the filtering logic works (after fixing bug #1), it returns
   `""` (empty string) to skip dispatching events. However, this empty string gets yielded
   to the event encoder which expects event objects with `model_dump_json()` method,
   causing an `AttributeError`.

**Fixes**:

1. Changed metadata reading to handle both dict and object cases.
2. Changed return value from `""` to `None` for filtered events.
3. Added `run()` method override to filter out `None` values before yielding to encoder.

## How to Apply Patches

Run after installing dependencies:

```bash
# After uv sync or pip install
uv run patches/apply_patches.py
# or
python patches/apply_patches.py
```

## Automatic Application

Add this to your workflow:

```bash
# Full install command
uv sync && uv run patches/apply_patches.py
```

Or add a Makefile target:

```makefile
install:
	uv sync
	uv run patches/apply_patches.py
```

## When to Remove

Remove patches when the upstream fix is released. Check:
- [CopilotKit #2066](https://github.com/CopilotKit/CopilotKit/issues/2066) - emit_messages fix
