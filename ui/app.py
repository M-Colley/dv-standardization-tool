"""Moved to ``opendv/ui/app.py``; ``streamlit run ui/app.py`` keeps working through this file.
Prefer ``opendv ui`` or ``streamlit run opendv/ui/app.py``."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "opendv" / "ui" / "app.py"), run_name="__main__")
