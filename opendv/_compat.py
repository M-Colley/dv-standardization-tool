"""Aliases that keep the pre-3.1 module paths (``scripts.*``, ``analyses.*``) working.

Each old module file calls :func:`alias`, which makes the old name refer to
the *same* module object as its new :mod:`opendv` location, so attribute
access and ``unittest.mock.patch("scripts.x.y")`` act on the real module.
"""

from __future__ import annotations

import importlib
import runpy
import sys

# Modules whose CLI starts worker processes: calling main() of the imported
# module keeps their functions picklable as ``opendv.<module>.<function>``.
CALL_MAIN = frozenset({"opendv.run_batch_standardization", "opendv.run_catalog_meta_analysis"})


def alias(name: str, target: str) -> None:
    """Make module ``name`` an alias of ``target``, or run ``target`` when executed as a script."""
    if name == "__main__":
        if target in CALL_MAIN:
            importlib.import_module(target).main()
        else:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        return
    if name == "__mp_main__":  # a worker process re-importing the main script
        return
    sys.modules[name] = importlib.import_module(target)
