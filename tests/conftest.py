"""Shared pytest configuration."""

import matplotlib

# Tests render figures headlessly. Without this, matplotlib may pick an
# interactive backend (TkAgg on Windows) and fail on machines without a
# working Tk installation, depending on which test first opens a figure.
matplotlib.use("Agg")
