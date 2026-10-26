"""Alias of :mod:`opendv.value_labels` (moved there in 3.1).

Keeps the old import path, ``python -m`` and direct runs of this file working
in a source checkout; new code should import the module from ``opendv``.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # run directly as a file
    sys.path.insert(0, _ROOT)

from opendv._compat import alias  # noqa: E402

alias(__name__, "opendv.value_labels")
