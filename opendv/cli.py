"""
OpenDV-HCI CLI dispatcher.

Backs the installed ``opendv`` console script and ``python __main__.py`` /
``python .`` invocations from a repository checkout. Subcommands delegate to
the existing entry-point modules.

Subcommands
-----------
standardize  Single-study DV standardization (opendv.convert_dv)
batch        Batch standardization across multiple studies (opendv.run_batch_standardization)
analyze      Multi-study meta-analysis (opendv.analysis.multi_study_analysis)
validate     Schema validation (opendv.validate_schema)
review       Review LLM and unmapped column mappings (opendv.mapping_review)
ui           Start the Streamlit interface (opendv/ui/app.py)
plugins      List installed plugins (opendv.plugins)
"""

from __future__ import annotations

import argparse
import sys

from opendv import __version__

EXAMPLES = """\
examples:
  opendv standardize --help
  opendv batch --manifest sources_manifest.yaml
  opendv analyze --output-dir results/
  opendv validate schemas/standard_dv_mapping.yaml
  opendv review queue --batch-dir data/processed/catalog_meta_analysis
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opendv",
        description=(
            "OpenDV-HCI: schema-driven standardization and meta-analysis "
            "of dependent variables in HCI research."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", title="commands")

    # Delegating commands have no options of their own (add_help=False), so
    # -h/--help reaches the delegate's parser along with the other arguments.
    subparsers.add_parser(
        "standardize",
        help="Run single-study DV standardization",
        add_help=False,
    )
    subparsers.add_parser(
        "batch",
        help="Run batch standardization across multiple studies",
        add_help=False,
    )
    subparsers.add_parser(
        "analyze",
        help="Run multi-study meta-analysis",
        add_help=False,
    )
    subparsers.add_parser(
        "validate",
        help="Validate a DV standardization schema",
        usage="opendv validate [-h] path/to/schema.yaml",
    )
    subparsers.add_parser(
        "ui",
        help="Start the Streamlit interface",
        description="Start the Streamlit app; further arguments go to `streamlit run`.",
    )
    subparsers.add_parser("plugins", help="List installed OpenDV plugins")
    subparsers.add_parser(
        "review",
        help="Review LLM and unmapped column mappings (queue / apply / summary)",
        add_help=False,
    )

    return parser


def _run_standardize(argv: list[str]) -> None:
    try:
        from opendv.convert_dv import main
    except ImportError as exc:
        print(
            f"Error: could not import opendv.convert_dv ({exc}).\n"
            "Make sure the package is installed: pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.argv = ["opendv-standardize", *argv]
    main()


def _run_batch(argv: list[str]) -> None:
    try:
        from opendv.run_batch_standardization import main
    except ImportError as exc:
        print(
            f"Error: could not import opendv.run_batch_standardization ({exc}).\n"
            "Make sure the package is installed: pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.argv = ["opendv-batch", *argv]
    main()


def _run_analyze(argv: list[str]) -> None:
    try:
        from opendv.analysis.multi_study_analysis import main
    except ImportError as exc:
        print(
            f"Error: could not import opendv.analysis.multi_study_analysis ({exc}).\n"
            "Make sure the package is installed: pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.argv = ["opendv-analyze", *argv]
    main()


def _run_validate(argv: list[str]) -> None:
    try:
        from opendv.validate_schema import load_schema, validate
    except ImportError as exc:
        print(
            f"Error: could not import opendv.validate_schema ({exc}).\n"
            "Make sure the package is installed: pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)

    if not argv:
        print("Usage: opendv validate <path/to/schema.yaml>", file=sys.stderr)
        sys.exit(1)

    schema_path = argv[0]
    print("=" * 60)
    print("OpenDV-HCI Schema Validator")
    print("=" * 60)
    print(f"Validating: {schema_path}\n")

    try:
        schema = load_schema(schema_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {schema_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Could not load schema: {exc}", file=sys.stderr)
        sys.exit(1)

    issues = validate(schema, schema_path)
    if issues:
        print(f"[ERROR] Schema validation FAILED with {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        sys.exit(1)

    print("[OK] Schema is valid!\n")


def _run_review(argv: list[str]) -> None:
    from opendv.mapping_review import main

    sys.argv = ["opendv-review", *argv]
    main(argv)


def _run_ui(argv: list[str]) -> None:
    import subprocess
    from pathlib import Path

    app = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *argv]))


def _run_plugins(argv: list[str]) -> None:
    import json

    from opendv.plugins import load_plugins

    print(json.dumps(load_plugins(), indent=2))


_DISPATCH = {
    "standardize": _run_standardize,
    "batch": _run_batch,
    "analyze": _run_analyze,
    "validate": _run_validate,
    "review": _run_review,
    "ui": _run_ui,
    "plugins": _run_plugins,
}


def main() -> None:
    parser = _build_parser()

    # Parse only the first argument to identify the subcommand;
    # remaining argv is forwarded to the delegate so each module
    # can use its own argparse independently.
    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _DISPATCH[args.command](remaining)


if __name__ == "__main__":
    main()
