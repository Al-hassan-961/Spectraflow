"""``python -m spectraflow.server`` entry point.

Keeping this separate from ``spectraflow.server.app`` means the documented
short invocation runs the CLI without runpy executing the module twice.
"""

from __future__ import annotations

from spectraflow.server.app import main

if __name__ == "__main__":
    raise SystemExit(main())
