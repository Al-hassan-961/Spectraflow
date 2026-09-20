"""Native-core discovery.

The C++ extension in ``core/`` is an *accelerator*, never a requirement. This
module locates it once at import time and exposes it as :data:`NATIVE`
(``None`` when unavailable), so callers can write::

    from spectraflow._native import NATIVE, HAVE_NATIVE
    if HAVE_NATIVE:
        ...  # native path
    else:
        ...  # NumPy path

Search order:

1. a normal ``import spectraflow_core`` — this is what ``pip install .`` yields;
2. ``<repo root>/spectraflow_core<suffix>`` — ``build_ext --inplace``;
3. anything matching ``<repo root>/build/**/spectraflow_core<suffix>`` — a plain
   ``cmake --build build`` drops the module there.

Loading from a file path uses :mod:`importlib.util` rather than mutating
``sys.path``, so a stale build directory elsewhere on the path can never shadow
the expected module.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterator

_MODULE_NAME = "spectraflow_core"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _candidate_paths() -> Iterator[Path]:
    """Yield plausible on-disk locations of the compiled extension."""
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _REPO_ROOT / f"{_MODULE_NAME}{suffix}"
        if candidate.is_file():
            yield candidate

    build_dir = _REPO_ROOT / "build"
    if build_dir.is_dir():
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            yield from sorted(build_dir.rglob(f"{_MODULE_NAME}{suffix}"))


def load_native() -> Any | None:
    """Return the native module, or ``None`` when it cannot be loaded.

    Never raises: a missing or ABI-incompatible build must simply select the
    NumPy fallback.
    """
    try:
        return importlib.import_module(_MODULE_NAME)
    except ImportError:
        pass
    except Exception:
        # A present-but-unloadable module (e.g. linked against a different
        # libpython) must not take the whole package down with it.
        pass

    for path in _candidate_paths():
        try:
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Register before exec so that module-level imports resolve.
            sys.modules[_MODULE_NAME] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop(_MODULE_NAME, None)
            continue
    return None


NATIVE: Any | None = load_native()
HAVE_NATIVE: bool = NATIVE is not None


def backend_description() -> str:
    """Human-readable summary of which implementation is active."""
    if HAVE_NATIVE:
        return f"native C++ core (spectraflow_core {_MODULE_NAME})"
    return "pure NumPy fallback (native core not built)"
