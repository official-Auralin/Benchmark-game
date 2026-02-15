"""
Executable module entry point for `python -m gf01`.

This file is intentionally small: it forwards execution to the CLI main
function so command behavior is centralized in one place.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

__author__ = "Bobby Veihman"
__copyright__ = "Academic Commons"
__license__ = "License Name"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
