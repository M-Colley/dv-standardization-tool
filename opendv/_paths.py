"""Where OpenDV-HCI finds its schemas and, in a source checkout, the repository.

Schemas ship inside the installed package (``opendv/schemas``, mapped from
the repository's ``schemas/`` folder by the build configuration). In a source
checkout without installation they are read from ``<repository>/schemas``.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
# The repository root in a source checkout (data/ and the analysis output
# folders live there); in an installed package it is site-packages.
REPO_ROOT = PACKAGE_DIR.parent


def _schemas_dir() -> Path:
    bundled = PACKAGE_DIR / "schemas"
    if (bundled / "standard_dv_mapping.yaml").is_file():
        return bundled
    checkout = REPO_ROOT / "schemas"
    if (checkout / "standard_dv_mapping.yaml").is_file():
        return checkout
    try:  # editable install that maps opendv.schemas to the checkout
        import opendv.schemas as schemas_pkg

        return Path(schemas_pkg.__file__).resolve().parent
    except ImportError:
        return checkout


SCHEMAS_DIR = _schemas_dir()

__all__ = ["PACKAGE_DIR", "REPO_ROOT", "SCHEMAS_DIR"]
