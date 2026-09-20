/**
 * Spectraflow — application entry point.
 *
 * Responsibilities:
 *   - load Three.js and the rendering modules, surfacing a clear in-DOM banner
 *     when the CDN is unreachable instead of leaving a blank screen
 *   - own the WebSocket client (exponential backoff with jitter, auto-reconnect)
 *   - run the requestAnimationFrame loop and feed the renderers
 *   - decouple render rate from network rate: the newest frame wins and stale
 *     frames are dropped, never queued
 *   - drive the HUD and the live stats readout
 *
 * Startup is deliberately split: `config.js` and `hud.js` are static imports
 * because they have no CDN dependency, while Three.js, `scene.js` and
 * `avatar.js` are pulled in with dynamic `import()` so a network failure is a
 * catchable rejection rather than a dead module graph.
 */

import { APP, RENDER, VOLUME, WS, numOr, safeDelta, summarizeNumbers } from './config.js';
import { createHud } from './hud.js';

/** Disarm the boot watchdog in index.html as early as possible. */
if (typeof window !== 'undefined') window.__SF_BOOTED__ = true;

/* -------------------------------------------------------------------------- */
/* DOM helpers                                                                */
/* -------------------------------------------------------------------------- */

function byId(id) {
  if (typeof document === 'undefined' || typeof document.getElementById !== 'function') return null;
  return document.getElementById(id);
}

/**
 * Reveal the fatal-error banner. Reuses the markup shipped in index.html (so
 * the watchdog script and this function share one banner) and creates it when
 * the element is missing.
 */
function showBanner(title, message, detail) {
  if (typeof document === 'undefined') return;

  let banner = byId('fatal-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'fatal-banner';
    banner.className = 'banner glass';
    banner.setAttribute('role', 'alert');
    banner.innerHTML = ''
      + '<h2 class="banner-title" data-banner-title></h2>'
      + '<p class="banner-message" data-banner-message></p>'
      + '<p class="banner-detail" data-banner-detail></p>'
      + '<button class="btn" type="button" data-banner-retry>Reload</button>';
    document.body.appendChild(banner);
  }

  const setText = (selector, value) => {
    const node = banner.querySelector(selector);
    if (node && value) node.textContent = value;
  };
  setText('[data-banner-title]', title || 'Spectraflow could not start');
  setText('[data-banner-message]', message || '');
  setText('[data-banner-detail]', detail || '');

  const detailNode = banner.querySelector('[data-banner-detail]');
  if (detailNode) detailNode.hidden = !detail;

  const retry = banner.querySelector('[data-banner-retry]');
  if (retry && !retry.__sfBound) {
    retry.__sfBound = true;
    retry.addEventListener('click', () => {
      if (typeof window !== 'undefined' && window.location && window.location.reload) {
        window.location.reload();
      }
    });
  }

  banner.hidden = false;
  if (document.body && document.body.classList) document.body.classList.add('has-banner');
}

function hideBanner() {
  const banner = byId('fatal-banner');
  if (banner) banner.hidden = true;
  if (typeof document !== 'undefined' && document.body && document.body.classList) {
    document.body.classList.remove('has-banner');
  }
}

/* -------------------------------------------------------------------------- */
/* Live counters                                                              */
/* -------------------------------------------------------------------------- */

const counters = {
  received: 0,
  dropped: 0,
  malformed: 0,
  reconnects: 0,
  controlMessages: 0,
  fps: 0,
  frameWindowStartMs: 0,
  frameWindowCount: 0,
};

/** Newest frame waiting to be consumed by the render loop (newest wins). */
let pendingFrame = null;
/** Newest frame already consumed; retained so the HUD keeps showing vitals. */
let latestFrame = null;
let lastFrameMs = 0;

let scene = null;
let avatar = null;
let hud = null;

/** HUD writes are routed through here so a HUD failure never breaks telemetry. */
function setStatus(text, state) {
  if (hud) hud.setStatus(text, state);
}

/* -------------------------------------------------------------------------- */
/* Banner-safe startup                                                        */
/* -------------------------------------------------------------------------- */

function boot() {
  const overlay = byId('overlay') || document.body;

  // The HUD is built defensively: a DOM problem here must not stop the socket
  // client or the render loop from running.
  try {
    hud = createHud(overlay);
    setStatus('connecting', 'connecting');
  } catch (error) {
    hud = null;
    showBanner(
      'The HUD could not be created',
      'Vitals and stats will not be displayed, but the 3D view and the sensor connection keep running.',
      error && error.message ? error.message : String(error),
    );
  }

  window.addEventListener('online', () => {
    if (!isSocketOpen()) {
      clearReconnectTimer();
      reconnectAttempt = 0;
      connect();
    }
  });

  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('visibilitychange', () => {
      // A background tab pauses rAF; come back to a live socket rather than a
      // stale one that already timed out server-side.
      if (!document.hidden && !isSocketOpen()) {
        clearReconnectTimer();
        connect();
      }
    });
  }

  loadRenderers()
    .catch((error) => {
      // loadRenderers handles its own messaging; this is the last-resort net.
      showBanner(
        '3D view unavailable',
        'The WebGL renderer could not be started, so the volumetric view is disabled. Telemetry and vitals keep running.',
        error && error.message ? error.message : String(error),
      );
    })
    .then(() => {
      startLoop();
      connect();
    });
}

/**
 * Import Three.js and the rendering modules. Any failure (offline, blocked
 * CDN, no import-map support, no WebGL) is reported in the DOM and leaves the
 * HUD + telemetry working.
 */
async function loadRenderers() {
  const viewport = byId('viewport') || document.body;

  try {
    await import('three');
  } catch (error) {
    showBanner(
      'Three.js could not be loaded',
      'The 3D library is served from a CDN (cdn.jsdelivr.net) and could not be fetched, so the volumetric view is unavailable. Check the network connection, then reload. Telemetry and vitals below keep updating from the sensor.',
      `import("three") failed: ${error && error.message ? error.message : String(error)}`,
    );
    return;
  }

  try {
    const [sceneModule, avatarModule] = await Promise.all([
      import('./scene.js'),
      import('./avatar.js'),
    ]);

    scene = sceneModule.createScene(viewport, {
      onError: (message) => {
        showBanner('3D rendering stopped', message, 'Reload the page to rebuild the WebGL context.');
      },
    });
    avatar = avatarModule.createAvatar(scene);

    hud.onResetView = () => {
      if (scene && typeof scene.resetView === 'function') scene.resetView();
    };

    hideBanner();
    console.info(
      `[${APP.name}] viewport ready — sensing volume ${VOLUME.width} m x ${VOLUME.depth} m x ${VOLUME.height} m`,
    );
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    const isWebgl = /webgl/i.test(message);
    showBanner(
      isWebgl ? 'WebGL is unavailable' : 'The 3D view failed to start',
      isWebgl
        ? 'This browser could not create a WebGL context, so the 3D view is disabled. Try closing other tabs or restarting the browser. Telemetry and vitals keep updating.'
        : 'The rendering modules could not be initialised. Telemetry and vitals keep updating.',
      message,
    );
  }
}

/* -------------------------------------------------------------------------- */
/* WebSocket client                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Reconnect schedule: 500 ms, doubling per attempt, capped at 10 s, with
 * symmetric jitter so a fleet of clients never retries in lockstep.
 */
function backoffDelay(attemptIndex) {
  const exponent = Math.max(0, attemptIndex);
  const nominal = Math.min(WS.maxDelayMs, WS.baseDelayMs * Math.pow(2, exponent));
  const spread = nominal * WS.jitterRatio;              // 0.5 -> +/- 25 %
  const jittered = nominal + (Math.random() - 0.5) * spread;
  return Math.round(Math.max(WS.baseDelayMs * 0.5, Math.min(WS.maxDelayMs, jittered)));
}

function websocketUrl() {
  const location = (typeof window !== 'undefined' && window.location) ? window.location : null;
  const protocol = location && location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = (location && location.host) ? location.host : WS.fallbackHost;
  return `${protocol}//${host}${WS.path}`;
}

let socket = null;
let reconnectAttempt = 0;
let reconnectTimer = 0;
let reconnectAtMs = 0;
let socketState = 'idle'; // 'idle' | 'connecting' | 'open' | 'closed'
let lastCountdownMs = -Infinity;

function isSocketOpen() {
  return socketState === 'open' && socket !== null;
}

function clearReconnectTimer() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = 0;
  }
  reconnectAtMs = 0;
}

function connect() {
  clearReconnectTimer();

  const url = websocketUrl();
  socketState = 'connecting';
  setStatus(reconnectAttempt > 0 ? `reconnecting (attempt ${reconnectAttempt})` : 'connecting', 'connecting');

  let ws;
  try {
    ws = new WebSocket(url);
  } catch (error) {
    scheduleReconnect(`invalid WebSocket URL ${url}`);
    return;
  }
  socket = ws;

  ws.onopen = () => {
    if (socket !== ws) return;
    socketState = 'open';
    reconnectAttempt = 0;
    clearReconnectTimer();
    setStatus('online', 'online');
  };

  ws.onmessage = (event) => {
    if (socket !== ws) return;
    handleMessage(event);
  };

  ws.onerror = () => {
    if (socket !== ws) return;
    // `close` always follows an error; the reconnect is scheduled there so the
    // backoff timer is never armed twice.
    setStatus('socket error', 'offline');
  };

  ws.onclose = (event) => {
    if (socket === ws) {
      socket = null;
      socketState = 'closed';
    }
    const reason = event && typeof event.reason === 'string' && event.reason
      ? event.reason
      : (event && Number.isFinite(event.code) ? `code ${event.code}` : 'closed');
    scheduleReconnect(reason);
  };
}

function scheduleReconnect(reason) {
  if (reconnectTimer) return; // already counting down

  const delay = backoffDelay(reconnectAttempt);
  reconnectAttempt += 1;
  counters.reconnects += 1;
  reconnectAtMs = Date.now() + delay;

  setStatus(`offline · ${reason}`, 'offline');
  if (typeof console !== 'undefined' && console.info) {
    console.info(`[${APP.name}] socket closed (${reason}); retrying in ${delay} ms`);
  }
  reconnectTimer = setTimeout(() => {
    reconnectTimer = 0;
    reconnectAtMs = 0;
    connect();
  }, delay);
}

/** Treat a payload without a `type` as a frame when it carries frame fields. */
function looksLikeFrame(message) {
  return message.seq !== undefined
    || message.keypoints !== undefined
    || message.power !== undefined
    || message.vitals !== undefined
    || message.presence !== undefined;
}

function handleMessage(event) {
  const data = event ? event.data : null;
  if (typeof data !== 'string') return; // binary payloads are not part of the contract
  if (data.length > WS.maxPayloadBytes) {
    counters.malformed += 1;
    return;
  }

  let message;
  try {
    message = JSON.parse(data);
  } catch (error) {
    counters.malformed += 1;
    return;
  }
  if (!message || typeof message !== 'object') {
    counters.malformed += 1;
    return;
  }

  const type = typeof message.type === 'string' ? message.type : '';

  if (type === 'frame' || (type === '' && looksLikeFrame(message))) {
    ingestFrame(message);
    return;
  }
  if (type === 'hello') {
    handleHello(message);
    return;
  }
  if (type === 'stats') {
    // Server-side telemetry is intentionally not rendered yet; counted only.
    counters.controlMessages += 1;
    return;
  }

  // Unknown `type` values are ignored on purpose, so a newer server can add
  // message kinds without breaking this client.
  counters.controlMessages += 1;
}

function handleHello(message) {
  counters.controlMessages += 1;
  if (scene && Array.isArray(message.nodes)) {
    scene.setNodes(message.nodes);
  }
}

function ingestFrame(frame) {
  counters.received += 1;
  // Newest wins: an unconsumed frame is dropped rather than queued, so the
  // render loop can never fall behind the network.
  if (pendingFrame !== null) counters.dropped += 1;
  pendingFrame = frame;
}

/* -------------------------------------------------------------------------- */
/* Render loop                                                                */
/* -------------------------------------------------------------------------- */

const powerSummary = { count: 0, mean: 0, std: 0, min: 0, max: 0, span: 0 };
let lastTickMs = 0;
let rafId = 0;

function buildStats(nowMs, dt, consumed) {
  const frameAgeMs = lastFrameMs > 0 ? nowMs - lastFrameMs : null;

  let seq = null;
  let nodeId = null;
  let subcarriers = null;
  if (latestFrame) {
    seq = Number.isFinite(latestFrame.seq) ? latestFrame.seq : null;
    nodeId = Number.isFinite(latestFrame.node_id) ? latestFrame.node_id : null;
    summarizeNumbers(latestFrame.power, powerSummary);
    subcarriers = powerSummary.count > 0 ? powerSummary.count : null;
  }

  return {
    dt,
    fps: counters.fps,
    received: counters.received,
    dropped: counters.dropped,
    malformed: counters.malformed,
    seq,
    nodeId,
    subcarriers,
    frameAgeMs,
    stale: frameAgeMs === null || frameAgeMs > WS.staleFrameMs,
    reconnects: counters.reconnects,
    nodes: scene ? scene.nodeCount : null,
    /** True when this tick consumed a fresh frame, so the HUD updates at once. */
    frameFresh: consumed !== null,
  };
}

function tick(nowMs) {
  rafId = requestAnimationFrame(tick);

  const dt = safeDelta((nowMs - lastTickMs) / 1000, RENDER.maxDeltaSeconds);
  lastTickMs = nowMs;

  // Consume at most one frame per rendered frame: the newest one.
  let consumed = null;
  if (pendingFrame !== null) {
    consumed = pendingFrame;
    pendingFrame = null;
    latestFrame = consumed;
    lastFrameMs = nowMs;
  }

  // Rolling FPS over a 500 ms window, so the readout is readable. This counts
  // rendered frames (rAF ticks), which is what "FPS" means; network throughput
  // is reported separately as "Frames".
  counters.frameWindowCount += 1;
  if (counters.frameWindowStartMs === 0) counters.frameWindowStartMs = nowMs;
  const windowMs = nowMs - counters.frameWindowStartMs;
  if (windowMs >= 500) {
    counters.fps = (counters.frameWindowCount * 1000) / windowMs;
    counters.frameWindowCount = 0;
    counters.frameWindowStartMs = nowMs;
  }

  if (scene) scene.update(consumed);
  if (avatar) {
    // `null` keypoints always reach the avatar so a degraded frame fades the
    // skeleton out instead of freezing it on screen.
    avatar.update(latestFrame ? latestFrame.keypoints : null, dt);
  }
  if (scene) scene.render();

  if (hud) {
    hud.update(latestFrame, buildStats(nowMs, dt, consumed));

    // Live countdown while a reconnect is pending. The "offline ·" prefix is
    // kept so the chip always states the connection state plainly.
    if (reconnectTimer && nowMs - lastCountdownMs > 250) {
      lastCountdownMs = nowMs;
      const remainingMs = Math.max(0, reconnectAtMs - Date.now());
      setStatus(`offline · retry in ${(remainingMs / 1000).toFixed(1)} s`, 'offline');
    }
  }
}

function startLoop() {
  if (rafId) return;
  lastTickMs = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  counters.frameWindowStartMs = lastTickMs;
  rafId = requestAnimationFrame(tick);
}

/* -------------------------------------------------------------------------- */
/* Global error surfacing                                                     */
/* -------------------------------------------------------------------------- */

window.addEventListener('error', (event) => {
  const message = event && event.message ? event.message : 'unknown script error';
  const source = event && event.filename ? ` (${event.filename}:${numOr(event.lineno, 0)})` : '';
  showBanner('Runtime error', 'An unexpected error interrupted Spectraflow. Reloading usually clears it.', `${message}${source}`);
});

window.addEventListener('unhandledrejection', (event) => {
  const reason = event ? event.reason : null;
  const message = reason && reason.message ? reason.message : String(reason);
  showBanner('Runtime error', 'An unexpected error interrupted Spectraflow. Reloading usually clears it.', message);
});

/* -------------------------------------------------------------------------- */
/* Go                                                                         */
/* -------------------------------------------------------------------------- */

try {
  boot();
} catch (error) {
  showBanner(
    'Spectraflow could not start',
    'The interface failed to initialise. This usually means the browser is very old or a required file is missing from the static bundle.',
    error && error.message ? error.message : String(error),
  );
}

/** Small debug handle: inspect live state from the browser console. */
window.__spectraflow__ = {
  get frame() {
    return latestFrame;
  },
  get counters() {
    return counters;
  },
  get socketState() {
    return socketState;
  },
  get hud() {
    return hud;
  },
  get scene() {
    return scene;
  },
  get avatar() {
    return avatar;
  },
  reconnect() {
    clearReconnectTimer();
    reconnectAttempt = 0;
    connect();
  },
  dispose() {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
    clearReconnectTimer();
    if (socket) {
      const ws = socket;
      socket = null;
      socketState = 'closed';
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      try {
        ws.close();
      } catch (error) {
        // Already closing.
      }
    }
    if (avatar && typeof avatar.dispose === 'function') avatar.dispose();
    if (scene && typeof scene.dispose === 'function') scene.dispose();
    if (hud && typeof hud.dispose === 'function') hud.dispose();
    avatar = null;
    scene = null;
    hud = null;
  },
};
