"""
Shared repository-scope detection for private source vs public mirror tests.

Tests that assert private research_pack artifacts should use this helper so
scope behavior remains consistent across files and changes are centralized.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Bobby Veihman"
__copyright__ = "Academic Commons"
__license__ = "License Name"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

import os
from pathlib import Path


def is_public_mirror(root: Path) -> bool:
    """
    Return True when running in public-mirror scope.

    Precedence:
    1. `GF01_REPO_SCOPE` env override (`public|public_mirror|private|source`).
    2. Fallback marker: private source repo contains `Spec.tex`; public mirror
       intentionally does not.
    """

    scope = os.environ.get("GF01_REPO_SCOPE", "").strip().lower()
    if scope in {"public", "public_mirror"}:
        return True
    if scope in {"private", "source"}:
        return False
    return not (root / "Spec.tex").exists()
