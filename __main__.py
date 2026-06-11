"""Thin repo-checkout shim for the OpenDV-HCI CLI.

Allows ``python __main__.py <subcommand>`` (or ``python . <subcommand>``)
from a clone without installing the package. The real dispatcher lives in
``scripts.cli`` so that the installed ``opendv`` console script resolves a
proper importable module — entry points cannot target a top-level
``__main__`` module (it collides with the interpreter's own ``__main__``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
