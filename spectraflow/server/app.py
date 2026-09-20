"""Async streaming server: UDP CSI ingestion -> DSP/pose -> WebSocket clients.

Endpoint contract:

* ``GET /``            -- the WebGL visualiser (``static/index.html``)
* ``GET /css/*``, ``/js/*`` -- static assets
* ``GET /api/status``  -- JSON engine/backend/receiver diagnostics
* ``WS  /ws``          -- one JSON sensing frame per message (schema below)

Frame schema (the frozen contract shared with ``static/js/``)::

    {
      "type": "frame",
      "t": 1712345678.123,          # wall-clock seconds
      "seq": 12345,                 # node sequence number
      "node_id": 1,
      "presence": true,
      "motion": 0.12,
      "vitals": {"bpm": 72.4, "rpm": 15.2, "confidence": 0.74,
                 "bpm_snr": 4.1, "rpm_snr": 3.2,
                 "bpm_confidence": 0.6, "rpm_confidence": 0.8},
      "keypoints": [[x, y, z], ... 17 entries ...],   # or null
      "power": [-42.1, -41.0, ...]                    # dB per subcarrier
    }

Any field may be ``null`` on a degraded frame; the client is required to render
``--`` rather than invent a value.

Two application flavours are provided and share all of their logic:

* :func:`create_fastapi_app` -- used when ``fastapi`` is importable;
* :func:`create_asgi_app`    -- a dependency-free ASGI application.

FastAPI is a routing/validation layer over ASGI, not a server, so both are
served by the same ``uvicorn`` process. This matters in practice: FastAPI
depends on pydantic-core, whose Rust build fails on Android/aarch64, so on the
reference Termux deployment the dependency-free flavour is the one that runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np

from spectraflow._native import HAVE_NATIVE
from spectraflow.config import SpectraflowConfig, default_config
from spectraflow.dsp import BACKEND, VitalSignsExtractor
from spectraflow.inference import PoseEstimator
from spectraflow.ingestion import (
    CsiFrame,
    SyntheticCsiSource,
    UdpCsiReceiver,
)

logger = logging.getLogger("spectraflow.server")

__all__ = [
    "SensingEngine",
    "StreamHub",
    "SpectraflowServer",
    "create_app",
    "create_asgi_app",
    "create_fastapi_app",
    "main",
]

#: Repository root, used to locate ``static/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATIC_ROOT = _REPO_ROOT / "static"

#: How often pose inference runs. It costs far more than the vital-sign DSP, so
#: it is decoupled from the frame rate and interpolated by the front-end.
_POSE_INTERVAL_SECONDS = 0.2
#: Frames retained for the pose window.
_POSE_WINDOW_FRAMES = 20


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EngineStats:
    """Throughput counters for diagnostics."""

    frames: int = 0
    published: int = 0
    pose_runs: int = 0
    started_at: float = field(default_factory=time.time)
    last_frame_at: float = 0.0

    @property
    def uptime(self) -> float:
        return max(0.0, time.time() - self.started_at)


class SensingEngine:
    """Turns CSI frames into WebSocket payloads.

    Owns the DSP and pose estimators and keeps the sliding CSI window pose
    inference needs. One engine instance serves every connected client.
    """

    def __init__(
        self,
        config: SpectraflowConfig | None = None,
        *,
        pose_estimator: PoseEstimator | None = None,
    ) -> None:
        self.config = config or default_config()
        self.vitals = VitalSignsExtractor(self.config)
        self.pose = pose_estimator or PoseEstimator(self.config)
        self.stats = EngineStats()

        self._pose_buffer: deque[np.ndarray] = deque(maxlen=_POSE_WINDOW_FRAMES)
        self._last_pose_at = 0.0
        self._last_pose: Any = None
        self._last_power: list[float] = []

    def process(self, frame: CsiFrame) -> dict[str, Any]:
        """Consume one frame and build the JSON payload for it."""
        self.stats.frames += 1
        self.stats.last_frame_at = time.time()

        vitals = self.vitals.update(frame)

        self._pose_buffer.append(np.asarray(frame.csi, dtype=np.complex64))
        self._last_power = [
            round(float(v), 2) for v in frame.amplitude_db()
        ]

        now = time.monotonic()
        if (
            self._last_pose is None
            or (now - self._last_pose_at) >= _POSE_INTERVAL_SECONDS
        ):
            window = np.stack(list(self._pose_buffer))
            self._last_pose = self.pose.estimate(window)
            self._last_pose_at = now
            self.stats.pose_runs += 1

        pose = self._last_pose
        payload = {
            "type": "frame",
            "t": round(frame.received_at, 3),
            "seq": int(frame.sequence),
            "node_id": int(frame.node_id),
            "presence": bool(vitals.presence),
            "motion": round(float(vitals.motion), 4),
            "vitals": vitals.as_dict(),
            "keypoints": pose.as_list() if pose is not None else None,
            "power": self._last_power,
        }
        self.stats.published += 1
        return payload

    def hello(self) -> dict[str, Any]:
        """Handshake message describing the engine to a new client."""
        return {
            "type": "hello",
            "version": "1.0.0",
            "sample_rate_hz": self.config.sample_rate_hz,
            "respiration_band_hz": list(self.config.respiration_band_hz),
            "heart_band_hz": list(self.config.heart_band_hz),
            "window_seconds": self.config.window_seconds,
            "keypoint_format": "coco17",
            "dsp_backend": BACKEND,
            "native_core": HAVE_NATIVE,
            "pose_backend": self.pose.backend,
            "simulation": self.config.simulation,
        }

    def status(self) -> dict[str, Any]:
        """Diagnostics for ``GET /api/status``."""
        vitals = self.vitals.result
        return {
            "uptime_seconds": round(self.stats.uptime, 2),
            "frames": self.stats.frames,
            "published": self.stats.published,
            "pose_runs": self.stats.pose_runs,
            "pose_backend": self.pose.backend,
            "dsp_backend": BACKEND,
            "native_core": HAVE_NATIVE,
            "simulation": self.config.simulation,
            "vitals": vitals.as_dict(),
            "presence": bool(vitals.presence),
            "motion": round(float(vitals.motion), 4),
            "frames_buffered": self.vitals.frames_seen,
        }

    def reset(self) -> None:
        self.vitals.reset()
        self.pose.reset()
        self._pose_buffer.clear()
        self._last_pose = None
        self._last_pose_at = 0.0


class StreamHub:
    """Broadcast fan-out to WebSocket clients with per-client backpressure.

    Each client owns a bounded queue. When a client cannot keep up, the *oldest*
    frame is discarded rather than blocking the producer -- a slow browser must
    never apply backpressure to the sensing pipeline, and stale CSI is worthless
    anyway.
    """

    def __init__(self, queue_depth: int = 8) -> None:
        self.queue_depth = max(1, int(queue_depth))
        self._queues: list[asyncio.Queue[dict[str, Any]]] = []
        self.dropped = 0
        self.published = 0

    def register(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_depth)
        self._queues.append(queue)
        return queue

    def unregister(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._queues.remove(queue)
        except ValueError:
            pass

    def publish(self, payload: dict[str, Any]) -> None:
        """Non-blocking broadcast; never awaits, never raises."""
        self.published += 1
        for queue in list(self._queues):
            if queue.full():
                try:
                    queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:  # pragma: no cover - race
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover - race
                self.dropped += 1

    @property
    def client_count(self) -> int:
        return len(self._queues)


# ---------------------------------------------------------------------------
# Server core
# ---------------------------------------------------------------------------


class SpectraflowServer:
    """Wires the frame source through the engine into the hub.

    Transport-agnostic: the ASGI layers below only handle I/O.
    """

    def __init__(self, config: SpectraflowConfig | None = None) -> None:
        self.config = config or default_config()
        self.config.validate()
        self.engine = SensingEngine(self.config)
        self.hub = StreamHub(self.config.client_queue_depth)
        self.receiver: UdpCsiReceiver | None = None
        self.synthetic: SyntheticCsiSource | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.rate_limited = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Begin ingesting frames and publishing payloads."""
        if self.running:
            return

        if self.config.simulation:
            self.synthetic = SyntheticCsiSource(
                n_subcarriers=64,
                rate_hz=self.config.simulation_rate_hz,
            )
            logger.info(
                "Spectraflow: simulation mode -- synthesising CSI at %.1f Hz",
                self.config.simulation_rate_hz,
            )
        else:
            self.receiver = UdpCsiReceiver(
                host=self.config.udp_host,
                port=self.config.udp_port,
                max_datagram_bytes=self.config.max_datagram_bytes,
                stale_after_seconds=self.config.stale_frame_seconds,
                capacity=self.config.ring_capacity,
            )
            bound = await self.receiver.start()
            logger.info("Spectraflow: listening for CSI on udp/%s:%d", self.config.udp_host, bound)

        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="spectraflow-engine")

    async def stop(self) -> None:
        """Stop ingestion and tear down the task and socket."""
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        if self.receiver is not None:
            await self.receiver.stop()
            self.receiver = None

    async def _frames(self) -> AsyncIterator[CsiFrame]:
        """Yield frames from whichever source is configured."""
        if self.synthetic is not None:
            async for frame in self.synthetic.stream():
                if self._stopping.is_set():
                    return
                yield frame
            return

        assert self.receiver is not None
        while not self._stopping.is_set():
            yield await self.receiver.get()

    async def _run(self) -> None:
        """Main loop: frame -> engine -> hub, rate-limited to the stream rate."""
        period = 1.0 / max(self.config.stream_rate_hz, 1e-6)
        next_publish = time.monotonic()

        try:
            async for frame in self._frames():
                payload = self.engine.process(frame)

                now = time.monotonic()
                if now < next_publish:
                    # Rate-limit the outbound stream: the DSP runs on every
                    # frame, but clients neither need nor can render more than
                    # a few tens of frames per second.
                    self.rate_limited += 1
                    continue
                next_publish = now + period
                self.hub.publish(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Spectraflow: engine loop terminated unexpectedly")
            raise


# ---------------------------------------------------------------------------
# ASGI helpers
# ---------------------------------------------------------------------------

_HTTP_STATUS = {
    200: "OK",
    404: "Not Found",
    405: "Method Not Allowed",
    500: "Internal Server Error",
}


def _json_response(payload: Any, status: int = 200) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode()),
        (b"cache-control", b"no-store"),
    ]
    return status, headers, body


def _resolve_static(path: str) -> Path | None:
    """Map a URL path onto a file inside ``static/``, or ``None``.

    Rejects traversal outside the static root, which matters because this
    process is reachable from the LAN.
    """
    relative = path.lstrip("/") or "index.html"
    if relative.endswith("/"):
        relative += "index.html"
    candidate = (_STATIC_ROOT / relative).resolve()
    try:
        candidate.relative_to(_STATIC_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _static_response(path: str) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    file_path = _resolve_static(path)
    if file_path is None:
        body = b"404 Not Found"
        return 404, [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
        ], body

    body = file_path.read_bytes()
    mime, _ = mimetypes.guess_type(str(file_path))
    headers = [
        (b"content-type", f"{mime or 'application/octet-stream'}".encode()),
        (b"content-length", str(len(body)).encode()),
        (b"cache-control", b"no-cache"),
    ]
    return 200, headers, body


async def _asgi_send_response(
    send: Callable[[dict[str, Any]], Any],
    status: int,
    headers: list[tuple[bytes, bytes]],
    body: bytes,
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_asgi_app(server: SpectraflowServer) -> Callable[..., Any]:
    """Dependency-free ASGI application.

    Serves HTTP and WebSocket scopes and manages the engine's lifecycle through
    the ASGI lifespan protocol, so it works under ``uvicorn`` with no FastAPI,
    Starlette or pydantic involved.
    """

    async def app(scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        scope_type = scope.get("type")

        if scope_type == "lifespan":
            await _lifespan(server, receive, send)
            return

        if scope_type == "http":
            await _http(server, scope, send)
            return

        if scope_type == "websocket":
            await _websocket(server, scope, receive, send)
            return

        raise RuntimeError(f"unsupported ASGI scope type: {scope_type!r}")

    return app


async def _lifespan(
    server: SpectraflowServer,
    receive: Callable[..., Any],
    send: Callable[..., Any],
) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            try:
                await server.start()
                await send({"type": "lifespan.startup.complete"})
            except Exception as exc:
                logger.exception("Spectraflow: startup failed")
                await send({"type": "lifespan.startup.failed", "message": str(exc)})
                return
        elif message["type"] == "lifespan.shutdown":
            await server.stop()
            await send({"type": "lifespan.shutdown.complete"})
            return


async def _http(
    server: SpectraflowServer,
    scope: dict[str, Any],
    send: Callable[..., Any],
) -> None:
    path = scope.get("path", "/")
    method = scope.get("method", "GET")

    if method not in ("GET", "HEAD"):
        await _asgi_send_response(
            send, 405, [(b"content-type", b"text/plain")], b"405 Method Not Allowed"
        )
        return

    if path == "/api/status":
        status, headers, body = _json_response(server.engine.status())
    elif path == "/health":
        status, headers, body = _json_response({"status": "ok"})
    else:
        status, headers, body = _static_response(path)

    if method == "HEAD":
        body = b""
    await _asgi_send_response(send, status, headers, body)


async def _websocket(
    server: SpectraflowServer,
    scope: dict[str, Any],
    receive: Callable[..., Any],
    send: Callable[..., Any],
) -> None:
    path = scope.get("path", "/")
    if path != "/ws":
        await send({"type": "websocket.close", "code": 4404})
        return

    await send({"type": "websocket.accept"})
    queue = server.hub.register()
    disconnected = asyncio.Event()

    async def watch_disconnect() -> None:
        """Drain the receive channel so a closed socket is noticed promptly."""
        try:
            while True:
                message = await receive()
                if message["type"] == "websocket.disconnect":
                    return
        except Exception:  # noqa: BLE001 - any failure means "gone"
            return
        finally:
            disconnected.set()

    watcher = asyncio.create_task(watch_disconnect())
    try:
        await send(
            {
                "type": "websocket.send",
                "text": json.dumps(server.engine.hello()),
            }
        )

        while not disconnected.is_set():
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {getter, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                getter.cancel()
                break
            payload = getter.result()
            try:
                await send(
                    {"type": "websocket.send", "text": json.dumps(payload)}
                )
            except Exception:  # noqa: BLE001 - client vanished mid-send
                break
    finally:
        watcher.cancel()
        server.hub.unregister(queue)


def create_fastapi_app(server: SpectraflowServer) -> Any:
    """FastAPI flavour, used when ``fastapi`` is importable.

    Shares :class:`SpectraflowServer` with the dependency-free flavour, so the
    two cannot drift apart in behaviour.
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: PLC0415
    from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: PLC0415

    app = FastAPI(title="Spectraflow", version="1.0.0")

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - requires fastapi
        await server.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - requires fastapi
        await server.stop()

    @app.get("/api/status")
    async def _status() -> JSONResponse:  # pragma: no cover - requires fastapi
        return JSONResponse(server.engine.status())

    @app.get("/health")
    async def _health() -> JSONResponse:  # pragma: no cover - requires fastapi
        return JSONResponse({"status": "ok"})

    @app.get("/")
    async def _index() -> Response:  # pragma: no cover - requires fastapi
        return _fastapi_static("index.html", FileResponse, Response)

    @app.get("/{asset_path:path}")
    async def _assets(asset_path: str) -> Response:  # pragma: no cover
        return _fastapi_static(asset_path, FileResponse, Response)

    @app.websocket("/ws")
    async def _ws(websocket: WebSocket) -> None:  # pragma: no cover
        await websocket.accept()
        queue = server.hub.register()
        try:
            await websocket.send_text(json.dumps(server.engine.hello()))
            while True:
                payload = await queue.get()
                await websocket.send_text(json.dumps(payload))
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("Spectraflow: websocket closed", exc_info=True)
        finally:
            server.hub.unregister(queue)

    return app


def _fastapi_static(asset_path: str, file_response: Any, response_cls: Any) -> Any:
    file_path = _resolve_static("/" + asset_path.lstrip("/"))
    if file_path is None:
        return response_cls(content="404 Not Found", status_code=404)
    return file_response(file_path)


def create_app(
    config: SpectraflowConfig | None = None,
    *,
    prefer_fastapi: bool = True,
) -> tuple[Any, SpectraflowServer]:
    """Build the ASGI application, choosing the richest available flavour.

    Returns:
        ``(app, server)`` -- the server object is returned so tests can start,
        stop and inspect the engine directly.
    """
    server = SpectraflowServer(config)
    if prefer_fastapi:
        try:
            import fastapi  # noqa: F401,PLC0415
        except ImportError:
            logger.info(
                "Spectraflow: fastapi unavailable; serving the dependency-free "
                "ASGI application (identical routes and payloads)."
            )
        else:
            return create_fastapi_app(server), server
    return create_asgi_app(server), server


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m spectraflow.server.app``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="spectraflow", description="Wi-Fi CSI 3D sensing server"
    )
    parser.add_argument("--host", default=None, help="bind address")
    parser.add_argument("--port", type=int, default=None, help="bind port")
    parser.add_argument("--udp-port", type=int, default=None, help="CSI UDP port")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="generate synthetic CSI instead of listening on UDP",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help="analysis window in seconds (default 5.0)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    changes: dict[str, Any] = {}
    if args.host is not None:
        changes["host"] = args.host
    if args.port is not None:
        changes["port"] = args.port
    if args.udp_port is not None:
        changes["udp_port"] = args.udp_port
    if args.window is not None:
        changes["window_seconds"] = args.window
    if args.simulate:
        changes["simulation"] = True

    config = default_config().replace(**changes)
    app, _server = create_app(config)

    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "uvicorn is required to serve Spectraflow:\n"
            "    pip install uvicorn",
            file=__import__("sys").stderr,
        )
        return 2

    print(f"\n  Spectraflow listening on http://{config.host}:{config.port}")
    print(f"  DSP backend: {BACKEND} (native core: {HAVE_NATIVE})")
    print(f"  Mode: {'SIMULATION' if config.simulation else f'UDP {config.udp_host}:{config.udp_port}'}\n")

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
