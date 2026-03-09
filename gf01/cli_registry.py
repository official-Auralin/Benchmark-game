"""Parser registration facade for grouped GF-01 CLI command families."""

from __future__ import annotations

from .cli_registry_p0 import register_p0_commands
from .cli_registry_pilot import register_pilot_commands
from .cli_registry_playback import register_playback_commands
from .cli_registry_q033 import register_q033_commands
from .cli_registry_quality import register_quality_commands
from .cli_registry_workflows import register_workflow_commands

__all__ = [
    "register_p0_commands",
    "register_pilot_commands",
    "register_playback_commands",
    "register_q033_commands",
    "register_quality_commands",
    "register_workflow_commands",
]
