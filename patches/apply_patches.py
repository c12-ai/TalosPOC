#!/usr/bin/env python3
"""
Apply patches to third-party packages in the virtual environment.

This script should be run after `uv sync` or `pip install` to apply
necessary bug fixes to dependencies.

Usage:
    python patches/apply_patches.py
    # or
    uv run patches/apply_patches.py
"""

import sys
from pathlib import Path


def get_site_packages() -> Path:
    """Get the site-packages directory for the current environment."""
    for path in sys.path:
        p = Path(path)
        if p.name == "site-packages" and p.exists():
            return p
    raise RuntimeError("Could not find site-packages directory")


def apply_copilotkit_emit_messages_fix(site_packages: Path) -> bool:
    """
    Fix CopilotKit emit_messages metadata bug.

    Bug 1: langgraph_agui_agent.py uses getattr() on a dict instead of .get()
           This causes copilotkit:emit-messages metadata to be ignored.

    Bug 2: Returning "" for filtered events breaks the encoder which expects
           event objects with model_dump_json() method.

    Returns True if patch was applied, False if already patched.
    """
    target_file = site_packages / "copilotkit" / "langgraph_agui_agent.py"

    if not target_file.exists():
        print(f"  [SKIP] {target_file} not found (copilotkit not installed?)")
        return False

    content = target_file.read_text()

    # Check if already patched (check for both fixes)
    if "raw_event.get('metadata', {})" in content and "async def run(self, input):" in content:
        print(f"  [OK] Already patched: {target_file.name}")
        return False

    patched = False

    # Fix 0: Update docstring to explain None return for filtered events
    old_signature = '''    def _dispatch_event(self, event) -> str:
        """Override the dispatch event method to handle custom CopilotKit events and filtering"""'''
    new_signature = '''    def _dispatch_event(self, event) -> str:
        """Override the dispatch event method to handle custom CopilotKit events and filtering.

        Note: Returns None for filtered events (which violates the str return type annotation,
        but the base class also violates it by returning event objects). The None values are
        filtered out in run() before reaching the encoder.
        """'''

    if old_signature in content:
        content = content.replace(old_signature, new_signature)
        patched = True

    # Fix 1: metadata reading bug
    old_metadata_code = "metadata = getattr(raw_event, 'metadata', {}) or {}"
    new_metadata_code = """# FIX: raw_event can be a dict or an object, handle both cases
            # See: https://github.com/CopilotKit/CopilotKit/issues/2066
            metadata = (raw_event.get('metadata', {}) if isinstance(raw_event, dict)
                        else getattr(raw_event, 'metadata', {})) or {}"""

    if old_metadata_code in content:
        content = content.replace(old_metadata_code, new_metadata_code)
        patched = True

    # Fix 2: return None instead of "" and add run() override to filter None events
    # Replace return "" with return None
    old_return = 'return ""  # Don\'t dispatch this event'
    new_return = "return None  # Don't dispatch this event"

    if old_return in content:
        content = content.replace(old_return, new_return)
        patched = True

    # Add run() method override if not present
    if "async def run(self, input):" not in content:
        # Find the EXACT end of _dispatch_event by looking for the unique pattern
        # that includes the method boundary (before _handle_single_event)
        old_dispatch_end = """        return super()._dispatch_event(event)

    async def _handle_single_event"""
        new_dispatch_end = """        return super()._dispatch_event(event)

    async def run(self, input):
        \"\"\"Override run to filter out None events from _dispatch_event filtering.\"\"\"
        async for event in super().run(input):
            if event is not None:
                yield event

    async def _handle_single_event"""

        if old_dispatch_end in content:
            content = content.replace(old_dispatch_end, new_dispatch_end)
            patched = True

    if patched:
        target_file.write_text(content)
        print(f"  [PATCHED] {target_file.name}")
        return True
    else:
        print(f"  [WARN] Expected code not found in {target_file.name}")
        print(f"         CopilotKit version may have changed. Manual review needed.")
        return False


def main():
    print("=" * 60)
    print("Applying patches to third-party packages")
    print("=" * 60)

    try:
        site_packages = get_site_packages()
        print(f"Site-packages: {site_packages}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    patches_applied = 0

    # Apply all patches
    print("[1/1] CopilotKit emit_messages fix...")
    if apply_copilotkit_emit_messages_fix(site_packages):
        patches_applied += 1

    print()
    print("=" * 60)
    print(f"Done. {patches_applied} patch(es) applied.")
    print("=" * 60)


if __name__ == "__main__":
    main()
