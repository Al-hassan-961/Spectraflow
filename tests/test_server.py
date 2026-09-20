"""Server tests: HTTP routes, the WebSocket stream and client backpressure.

The ASGI application is driven directly (no network, no uvicorn) by feeding it
the scope/receive/send triple, which exercises the real routing, the real
lifespan and the real WebSocket handshake path that uvicorn will use.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from spectraflow.config import default_config
from spectraflow.server import (
    SensingEngine,
    StreamHub,
    create_app,
    create_asgi_app,
)
from spectraflow.server.app import SpectraflowServer, _static_response

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STATIC_ROOT = _REPO_ROOT / "static"


def _config(**changes):
    base = default_config().replace(
        simulation=True, simulation_rate_hz=40.0, stream_rate_hz=20.0
    )
    return base.replace(**changes) if changes else base


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_http_routes():
    app, server = create_app(_config())

    async def call(path: str, method: str = "GET"):
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            sent.append(message)

        await app({"type": "http", "path": path, "method": method, "headers": []}, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return start["status"], dict(start["headers"]), body

    status, headers, body = asyncio.run(call("/"))
    assert status == 200
    assert headers[b"content-type"] == b"text/html"
    assert b"<html" in body.lower() or b"<!doctype" in body.lower()

    status, headers, body = asyncio.run(call("/api/status"))
    assert status == 200
    payload = json.loads(body)
    assert payload["dsp_backend"] in ("native", "numpy")
    assert "pose_backend" in payload and "frames" in payload

    status, _, body = asyncio.run(call("/health"))
    assert status == 200 and json.loads(body) == {"status": "ok"}

    status, _, _ = asyncio.run(call("/does-not-exist.js"))
    assert status == 404

    status, _, _ = asyncio.run(call("/", method="POST"))
    assert status == 405


def test_static_assets_are_served_with_correct_types():
    app, _ = create_app(_config())

    async def fetch(path: str):
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            sent.append(message)

        await app({"type": "http", "path": path, "method": "GET", "headers": []}, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return start["status"], dict(start["headers"]), body

    for path, expected_prefix in (
        ("/css/style.css", b"text/css"),
        ("/js/app.js", b"text/javascript"),
        ("/js/scene.js", b"text/javascript"),
        ("/js/avatar.js", b"text/javascript"),
        ("/js/hud.js", b"text/javascript"),
        ("/js/config.js", b"text/javascript"),
    ):
        status, headers, body = asyncio.run(fetch(path))
        assert status == 200, path
        assert headers[b"content-type"].startswith(expected_prefix), path
        assert len(body) > 0, path


@pytest.mark.parametrize(
    "path",
    [
        "/../setup.py",
        "/../../etc/passwd",
        "/js/../../setup.py",
        "/%2e%2e/setup.py",
    ],
)
def test_path_traversal_outside_static_is_refused(path):
    """This process is reachable from the LAN, so traversal must be blocked."""
    app, _ = create_app(_config())

    async def fetch():
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            sent.append(message)

        await app({"type": "http", "path": path, "method": "GET", "headers": []}, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return start["status"], body

    status, body = asyncio.run(fetch())
    assert status == 404
    assert b"import" not in body[:200], "source file leaked"


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def _ws_session(duration: float = 2.5):
    """Run a full lifespan + websocket session and collect what was sent."""
    app, server = create_app(_config())

    async def scenario():
        life_in: asyncio.Queue = asyncio.Queue()
        life_out: list[dict] = []
        await life_in.put({"type": "lifespan.startup"})

        async def life_receive():
            return await life_in.get()

        async def life_send(message):
            life_out.append(message)

        lifespan = asyncio.create_task(app({"type": "lifespan"}, life_receive, life_send))
        await asyncio.sleep(0.3)

        ws_in: asyncio.Queue = asyncio.Queue()
        ws_out: list[dict] = []
        await ws_in.put({"type": "websocket.connect"})

        async def ws_receive():
            return await ws_in.get()

        async def ws_send(message):
            ws_out.append(message)

        socket = asyncio.create_task(
            app({"type": "websocket", "path": "/ws"}, ws_receive, ws_send)
        )
        await asyncio.sleep(duration)
        await ws_in.put({"type": "websocket.disconnect"})
        await asyncio.sleep(0.4)

        for task in (socket, lifespan):
            if not task.done():
                task.cancel()
        await life_in.put({"type": "lifespan.shutdown"})
        return life_out, ws_out, server

    return asyncio.run(scenario())


def test_websocket_streams_frames_matching_the_frozen_contract():
    lifespan, messages, server = _ws_session()

    assert lifespan and lifespan[0]["type"] == "lifespan.startup.complete"

    kinds = [m["type"] for m in messages]
    assert kinds.count("websocket.accept") == 1

    payloads = [json.loads(m["text"]) for m in messages if m["type"] == "websocket.send"]
    hello = [p for p in payloads if p["type"] == "hello"]
    frames = [p for p in payloads if p["type"] == "frame"]

    assert len(hello) == 1
    assert hello[0]["keypoint_format"] == "coco17"
    assert hello[0]["dsp_backend"] in ("native", "numpy")

    assert len(frames) >= 20, f"expected a continuous stream, got {len(frames)}"

    frame = frames[-1]
    assert isinstance(frame["t"], float)
    assert isinstance(frame["seq"], int)
    assert isinstance(frame["node_id"], int)
    assert isinstance(frame["presence"], bool)
    assert isinstance(frame["motion"], float)
    assert set(frame["vitals"]) == {
        "bpm", "rpm", "confidence",
        "bpm_confidence", "rpm_confidence", "bpm_snr", "rpm_snr",
    }
    for value in frame["vitals"].values():
        assert value is None or isinstance(value, float)

    assert all(isinstance(v, float) for v in frame["power"])
    assert len(frame["power"]) == 64

    keypoints = frame["keypoints"]
    if keypoints is not None:
        assert len(keypoints) == 17
        assert all(len(p) == 3 for p in keypoints)
        assert all(isinstance(v, float) for p in keypoints for v in p)

    sequences = [f["seq"] for f in frames]
    assert sequences == sorted(sequences), "sequence numbers must advance"

    # The client must be deregistered when it goes away.
    assert server.hub.client_count == 0


def test_websocket_rejects_unknown_paths():
    app, _ = create_app(_config())

    async def scenario():
        sent: list[dict] = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        await app({"type": "websocket", "path": "/nope"}, receive, send)
        return sent

    sent = asyncio.run(scenario())
    assert sent[0]["type"] == "websocket.close"
    assert sent[0]["code"] == 4404


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


def test_hub_drops_oldest_for_a_slow_client():
    async def scenario():
        hub = StreamHub(queue_depth=3)
        queue = hub.register()

        for i in range(10):
            hub.publish({"type": "frame", "seq": i})

        assert queue.qsize() == 3, "queue must stay bounded"
        assert hub.dropped == 7
        assert hub.published == 10

        # The frames that survived are the NEWEST ones: stale CSI is worthless.
        drained = [queue.get_nowait()["seq"] for _ in range(3)]
        assert drained == [7, 8, 9]
        return hub

    hub = asyncio.run(scenario())
    assert hub.client_count == 1
    hub.unregister(hub._queues[0])
    assert hub.client_count == 0


def test_hub_publish_with_no_clients_is_a_no_op():
    async def scenario():
        hub = StreamHub()
        hub.publish({"type": "frame"})
        assert hub.published == 1 and hub.client_count == 0
        return True

    assert asyncio.run(scenario())


def test_hub_unregister_is_idempotent():
    async def scenario():
        hub = StreamHub()
        queue = hub.register()
        hub.unregister(queue)
        hub.unregister(queue)
        return hub.client_count

    assert asyncio.run(scenario()) == 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_produces_a_payload_per_frame():
    engine = SensingEngine(_config())
    from csi_fixtures import synthetic_source

    source = synthetic_source(seed=3)
    payload = None
    for frame in source.frames(3.0):
        payload = engine.process(frame)

    assert payload["type"] == "frame"
    assert engine.stats.frames == 60
    assert engine.stats.published == 60
    assert len(payload["power"]) == 64
    assert engine.status()["frames"] == 60


def test_engine_hello_advertises_the_active_backends():
    engine = SensingEngine(_config())
    hello = engine.hello()
    assert hello["dsp_backend"] in ("native", "numpy")
    assert hello["pose_backend"] in ("onnx", "analytic")
    assert hello["keypoint_format"] == "coco17"
    assert hello["sample_rate_hz"] == 20.0


def test_engine_reset_clears_accumulated_state():
    engine = SensingEngine(_config())
    from csi_fixtures import synthetic_source

    for frame in synthetic_source(seed=4).frames(20.0):
        engine.process(frame)
    assert engine.vitals.frames_seen > 0

    engine.reset()
    assert engine.vitals.frames_seen == 0


def test_server_start_stop_is_idempotent():
    async def scenario():
        server = SpectraflowServer(_config(simulation=True))
        await server.start()
        assert server.running
        await server.start()  # no-op
        assert server.running
        await asyncio.sleep(0.3)
        assert server.engine.stats.frames > 0
        await server.stop()
        assert not server.running
        await server.stop()  # no-op

    asyncio.run(scenario())


def test_app_factory_prefers_available_flavour_without_failing():
    """Whatever is installed, create_app must return a working application."""
    app, server = create_app(_config())
    assert callable(app)
    assert isinstance(server, SpectraflowServer)

    explicit = create_asgi_app(server)
    assert callable(explicit)


# ---------------------------------------------------------------------------
# Offline readiness of the served page
# ---------------------------------------------------------------------------


def test_index_page_makes_no_external_requests():
    """The UI must render with no internet access at all.

    A page that silently depends on a CDN renders nothing useful when the
    network is blocked, which is the normal state on the reference deployment.
    Every asset, including Three.js, is therefore vendored and served locally.
    """
    html = (_REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    external = re.findall(r"https?://[^\s\"'<>)]+", html)
    assert not external, f"index.html still requests remote URLs: {external}"


def test_importmap_targets_exist_on_disk_and_are_served():
    """Resolve the importmap the way a browser does and check every target."""
    html = (_REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="importmap">\s*(\{.*?\})\s*</script>', html, re.S
    )
    assert match, "index.html must declare an import map"
    imports = json.loads(match.group(1))["imports"]

    assert not any(v.startswith("http") for v in imports.values()), imports

    # The specifiers the JavaScript actually uses.
    for specifier in ("three", "three/addons/controls/OrbitControls.js"):
        resolved = None
        for key in sorted(imports, key=len, reverse=True):
            if specifier.startswith(key):
                resolved = imports[key] + specifier[len(key) :]
                break
        assert resolved, f"no import-map entry resolves {specifier!r}"

        target = (_STATIC_ROOT / resolved.replace("./", "", 1)).resolve()
        assert target.is_file(), f"{specifier!r} -> {resolved} does not exist"

        # ...and that the server actually serves it.
        relative = str(target.relative_to(_STATIC_ROOT.resolve()))
        status, headers, body = _static_response("/" + relative)
        assert status == 200, (specifier, status)
        assert len(body) > 0


def test_every_script_and_stylesheet_reference_is_served():
    html = (_REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', html)
    assert refs, "index.html should reference assets"
    for ref in refs:
        assert not ref.startswith("http"), ref
        status, _, body = _static_response("/" + ref.lstrip("./"))
        assert status == 200, ref
        assert len(body) > 0, ref
