"""Shared pytest fixtures.

The suite runs end to end with no CSI hardware attached: every stimulus comes
from :mod:`tests.csi_fixtures`, which drives the production physics-based
simulator. Tests are written to be meaningful on a host where the native core
may or may not be built and where ``onnxruntime`` / ``fastapi`` may be absent --
the pipeline is required to degrade gracefully in all of those cases, and the
suite asserts that it does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repository root importable when pytest is invoked from elsewhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spectraflow._native import HAVE_NATIVE  # noqa: E402
from spectraflow.config import default_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    """The default, validated pipeline configuration."""
    return default_config()


@pytest.fixture(scope="session")
def has_native() -> bool:
    """Whether the native C++ core is available in this environment."""
    return HAVE_NATIVE


@pytest.fixture(scope="session")
def fast_config(config):
    """A configuration tuned for fast tests: short window, coarser FFT."""
    return config.replace(window_seconds=4.0, nfft=256)
