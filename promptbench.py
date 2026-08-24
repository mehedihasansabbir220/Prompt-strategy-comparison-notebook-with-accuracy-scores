"""Top-level entry point so the CLI can be run as ``python -m promptbench``.

The implementation lives in :mod:`src.cli`, alongside the rest of the library;
this shim only exists to give the command its documented name.
"""

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
