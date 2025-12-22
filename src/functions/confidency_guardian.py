"""
Independent validating each determination that AI made is trustworthy or not.

This is a Guardian as guardrail for each determination that AI made
"""

from pydantic import BaseModel


class ConfidencyGuardianAgent(BaseModel):
    def run(self) -> None:
        """Validate whether an AI determination can be trusted."""
        return
