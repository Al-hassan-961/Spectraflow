"""Async streaming server: UDP ingestion, DSP/pose pipeline and WebSocket API.

Attributes are resolved lazily (PEP 562) rather than imported eagerly. That is
not just a load-time optimisation: an eager ``import spectraflow.server.app``
here puts the module in ``sys.modules`` before runpy executes it, so
``python -m spectraflow.server.app`` runs the module twice and emits

    RuntimeWarning: 'spectraflow.server.app' found in sys.modules after import
    of package 'spectraflow.server', but prior to execution of ...

Resolving on first access keeps ``import spectraflow.server`` cheap and makes
both ``python -m spectraflow.server`` and ``... -m spectraflow.server.app``
clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "SensingEngine",
    "SpectraflowServer",
    "StreamHub",
    "create_app",
    "create_asgi_app",
    "create_fastapi_app",
    "main",
]

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from spectraflow.server.app import (
        SensingEngine,
        SpectraflowServer,
        StreamHub,
        create_app,
        create_asgi_app,
        create_fastapi_app,
        main,
    )

_LAZY = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Import :mod:`spectraflow.server.app` only when a name is first used."""
    if name in _LAZY:
        from spectraflow import server as _server

        module = __import__("spectraflow.server.app", fromlist=["_"])
        value = getattr(module, name)
        setattr(_server, name, value)  # cache so later access skips this
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
