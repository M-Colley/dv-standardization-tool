"""Plugins: extend OpenDV-HCI from another installed package, without forking it.

A package declares entry points in its ``pyproject.toml``; OpenDV-HCI loads
them on first use (:func:`load_plugins`, also called by the batch runner, the
file loader and the instrument registry).

=========================  ==================  ===============================================
group                      entry-point name    object
=========================  ==================  ===============================================
``opendv.sources``         manifest            ``materialize(source, target, refresh)`` returning
                           ``source_type``     ``(base_dir, provenance_id)``, as for
                                               :func:`opendv.run_batch_standardization.register_source_type`
``opendv.loaders``         file suffix         ``load(path) -> pandas.DataFrame``
                           without the dot
``opendv.instruments``     any                 path of an instrument registry YAML (same format as
                                               ``schemas/instruments.yaml``), or a callable
                                               returning one
=========================  ==================  ===============================================

Example ``pyproject.toml`` of a plugin package::

    [project.entry-points."opendv.sources"]
    my_archive = "my_plugin:materialize"

    [project.entry-points."opendv.loaders"]
    sav2 = "my_plugin:load_sav2"

    [project.entry-points."opendv.instruments"]
    my_scales = "my_plugin:REGISTRY_PATH"

A plugin that fails to load is logged and skipped; it never stops a run.
"""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GROUPS = ("opendv.sources", "opendv.loaders", "opendv.instruments")

_loaded: dict[str, list[str]] | None = None
_instrument_paths: list[Path] = []


def _load(ep: EntryPoint) -> Any | None:
    try:
        return ep.load()
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop a run
        logger.warning("Skipping OpenDV plugin %s (%s): %s", ep.name, ep.value, exc)
        return None


def load_plugins(force: bool = False) -> dict[str, list[str]]:
    """Register every installed plugin once; return ``{group: [names]}``."""
    global _loaded
    if _loaded is not None and not force:
        return _loaded
    from opendv import archive_utils, data_loaders
    from opendv.run_batch_standardization import register_source_type

    loaded: dict[str, list[str]] = {group: [] for group in GROUPS}
    _instrument_paths.clear()
    for ep in entry_points(group="opendv.sources"):
        materializer = _load(ep)
        if callable(materializer):
            register_source_type(ep.name)(materializer)
            loaded["opendv.sources"].append(ep.name)
    for ep in entry_points(group="opendv.loaders"):
        loader = _load(ep)
        if callable(loader):
            suffix = "." + ep.name.lower().lstrip(".")
            data_loaders._register_format(suffix)(loader)
            archive_utils.DATA_FILE_SUFFIXES.add(suffix)  # make the runner discover such files
            loaded["opendv.loaders"].append(suffix)
    for ep in entry_points(group="opendv.instruments"):
        target = _load(ep)
        path = target() if callable(target) else target
        if path is not None and Path(path).is_file():
            _instrument_paths.append(Path(path))
            loaded["opendv.instruments"].append(ep.name)
        elif path is not None:
            logger.warning("OpenDV instrument plugin %s points to a missing file: %s", ep.name, path)
    _loaded = loaded
    if any(loaded.values()):
        logger.info("Loaded OpenDV plugins: %s", {g: n for g, n in loaded.items() if n})
    return loaded


def instrument_registry_paths() -> list[Path]:
    """Instrument registries contributed by plugins."""
    load_plugins()
    return list(_instrument_paths)


def reset() -> None:
    """Forget loaded plugins so the next call rescans (for tests)."""
    global _loaded
    _loaded = None
    _instrument_paths.clear()


__all__ = ["GROUPS", "instrument_registry_paths", "load_plugins", "reset"]
