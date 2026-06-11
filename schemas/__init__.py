"""Schema data package for OpenDV-HCI.

This package intentionally contains no Python code — it exists so the YAML
schema files are included in built distributions (``[tool.setuptools.packages.find]``
only collects packages) and resolve next to ``scripts/`` after installation,
matching the ``Path(__file__).parents[1] / "schemas"`` lookups used across
the codebase.
"""
