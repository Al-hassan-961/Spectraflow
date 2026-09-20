/**
 * Spectraflow — vitals HUD overlay.
 *
 * Builds and owns two glassmorphism panels inside the supplied root element:
 *   - vitals (pinned top-left): BPM, RPM, signal confidence bar, motion,
 *     presence and the live connection status
 *   - stats (pinned bottom-right): frames received, FPS, dropped frames, last
 *     sequence number, node id, subcarrier count and frame age
 *
 * Every accessor is null-safe: a degraded frame (`vitals: null`, `bpm: null`,
 * `presence: null`, ...) renders as `--` and the confidence bar collapses to
 * zero width instead of producing `NaN`/`null` text.
 *
 * This module imports nothing but `config.js`, so the HUD still works when the
 * rendering layer is unavailable.
 */

import { APP, HUD, clamp, confidenceColor, damp, isFiniteNumber } from './config.js';

const STATUS_STATES = ['connecting', 'online', 'offline'];

/* -------------------------------------------------------------------------- */
/* Small DOM / formatting helpers                                             */
/* -------------------------------------------------------------------------- */

function resolveRoot(rootElement) {
  if (typeof rootElement === 'string') {
    const found = document.querySelector(rootElement);
    if (found) return { node: found, owned: false };
  }
  if (rootElement && typeof rootElement.appendChild === 'function') {
    return { node: rootElement, owned: false };
  }

  // No usable root: mount a self-managed overlay so the HUD is never lost.
  const overlay = document.createElement('div');
  overlay.className = 'overlay';
  document.body.appendChild(overlay);
  return { node: overlay, owned: true };
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

/** Fixed-precision number, or `--` when the value is null/NaN/Infinity. */
function fmt(value, digits) {
  if (!isFiniteNumber(value)) return '--';
  return value.toFixed(digits);
}

/** Rounded integer, or `--`. */
function fmtInt(value) {
  if (!isFiniteNumber(value)) return '--';
  return String(Math.round(value));
}

/** Plain counter, or `--`. */
function fmtCount(value) {
  if (!isFiniteNumber(value)) return '--';
  return String(value);
}

/** Seconds with one decimal, or `--`. */
function fmtAge(ms) {
  if (!isFiniteNumber(ms)) return '--';
  return `${(ms / 1000).toFixed(1)} s`;
}

/* -------------------------------------------------------------------------- */
/* HUD factory                                                                */
/* -------------------------------------------------------------------------- */

/**
 * Create the HUD.
 *
 * @param {HTMLElement|string} rootElement element the panels are appended to
 * @returns {{
 *   update: (frame: object|null, stats?: object) => void,
 *   setStatus: (text: string, state: 'connecting'|'online'|'offline') => void,
 *   dispose: () => void,
 *   onResetView: (() => void)|null,
 *   root: HTMLElement, vitals: HTMLElement, stats: HTMLElement,
 * }}
 */
export function createHud(rootElement) {
  const resolved = resolveRoot(rootElement);
  const root = resolved.node;
  const ownsRoot = resolved.owned;

  /* ------------------------------------------------------------ vitals panel */

  const vitals = el('section', 'panel glass hud-vitals');
  vitals.setAttribute('role', 'status');
  vitals.setAttribute('aria-live', 'polite');

  const head = el('header', 'panel-head');
  const brand = el('div', 'brand');
  brand.appendChild(el('span', 'brand-mark'));
  brand.appendChild(el('span', 'brand-name', APP.name));
  head.appendChild(brand);

  const statusChip = el('div', 'status');
  statusChip.setAttribute('data-status', 'connecting');
  statusChip.appendChild(el('span', 'status-dot'));
  const statusText = el('span', 'status-text', 'connecting');
  statusChip.appendChild(statusText);
  head.appendChild(statusChip);
  vitals.appendChild(head);

  /** Field nodes keyed by data-field, so writes stay O(1) and cached. */
  const fields = Object.create(null);

  function field(key, className, text) {
    const node = el('span', className, text);
    node.setAttribute('data-field', key);
    fields[key] = node;
    return node;
  }

  const metrics = el('div', 'metrics');

  const heartMetric = el('div', 'metric');
  heartMetric.appendChild(el('div', 'metric-label', 'Heart rate'));
  const heartValue = el('div', 'metric-value');
  heartValue.appendChild(field('bpm', 'metric-number', '--'));
  heartValue.appendChild(el('span', 'metric-unit', 'bpm'));
  heartMetric.appendChild(heartValue);
  const heartSub = el('div', 'metric-sub');
  heartSub.appendChild(el('span', 'metric-sub-label', 'SNR'));
  heartSub.appendChild(field('bpm_snr', 'metric-sub-value', '--'));
  heartMetric.appendChild(heartSub);
  metrics.appendChild(heartMetric);

  const respMetric = el('div', 'metric');
  respMetric.appendChild(el('div', 'metric-label', 'Respiration'));
  const respValue = el('div', 'metric-value');
  respValue.appendChild(field('rpm', 'metric-number', '--'));
  respValue.appendChild(el('span', 'metric-unit', 'rpm'));
  respMetric.appendChild(respValue);
  const respSub = el('div', 'metric-sub');
  respSub.appendChild(el('span', 'metric-sub-label', 'SNR'));
  respSub.appendChild(field('rpm_snr', 'metric-sub-value', '--'));
  respMetric.appendChild(respSub);
  metrics.appendChild(respMetric);

  vitals.appendChild(metrics);

  /* -------------------------------------------------------- confidence meter */

  const confidence = el('div', 'confidence');
  const confidenceHead = el('div', 'confidence-head');
  confidenceHead.appendChild(el('span', 'confidence-label', 'Signal confidence'));
  const confidenceValue = field('confidence', 'confidence-value', '--');
  confidenceHead.appendChild(confidenceValue);
  confidence.appendChild(confidenceHead);

  const confidenceTrack = el('div', 'confidence-track');
  const confidenceFill = el('div', 'confidence-fill');
  confidenceFill.setAttribute('data-role', 'confidence-fill');
  confidenceFill.style.width = '0%';
  confidenceTrack.appendChild(confidenceFill);
  confidence.appendChild(confidenceTrack);
  vitals.appendChild(confidence);

  /* ----------------------------------------------------------------- chips */

  const chips = el('div', 'chips');

  const motionChip = el('div', 'chip');
  motionChip.appendChild(el('span', 'chip-label', 'Motion'));
  motionChip.appendChild(field('motion', 'chip-value', '--'));
  chips.appendChild(motionChip);

  const presenceChip = el('div', 'chip');
  presenceChip.appendChild(el('span', 'chip-label', 'Presence'));
  presenceChip.appendChild(field('presence', 'chip-value', '--'));
  chips.appendChild(presenceChip);

  vitals.appendChild(chips);
  root.appendChild(vitals);

  /* ------------------------------------------------------------- stats panel */

  const statsPanel = el('section', 'panel glass hud-stats');
  const statsGrid = el('div', 'stats-grid');
  statsPanel.appendChild(statsGrid);

  const statNodes = Object.create(null);
  function statRow(key, label) {
    const row = el('div', 'stat');
    row.appendChild(el('span', 'stat-label', label));
    const value = el('span', 'stat-value', '--');
    row.appendChild(value);
    statsGrid.appendChild(row);
    statNodes[key] = value;
    return value;
  }

  statRow('fps', 'FPS');
  statRow('received', 'Frames');
  statRow('dropped', 'Dropped');
  statRow('malformed', 'Malformed');
  statRow('seq', 'Seq');
  statRow('node', 'Node');
  statRow('subcarriers', 'Subcarriers');
  statRow('age', 'Frame age');

  const resetButton = el('button', 'btn btn-reset', 'Reset view');
  resetButton.setAttribute('type', 'button');
  statsPanel.appendChild(resetButton);
  root.appendChild(statsPanel);

  /* ------------------------------------------------------------ state + cache */

  const written = Object.create(null);
  let confDisplay = 0;
  let lastStatsWriteMs = -Infinity;
  let disposed = false;
  const api = {
    root,
    vitals,
    stats: statsPanel,
    /** Assigned by the host app; invoked when the reset button is pressed. */
    onResetView: null,
  };

  function writeText(key, node, value) {
    if (written[key] === value) return;
    written[key] = value;
    node.textContent = value;
  }

  const handleResetClick = (event) => {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    if (typeof api.onResetView === 'function') api.onResetView();
  };
  resetButton.addEventListener('click', handleResetClick);

  /* ------------------------------------------------------------- public API */

  /**
   * Push one frame (or `null` when nothing has arrived yet) plus loop stats.
   *
   * @param {object|null} frame newest sensing frame
   * @param {object} [stats] { dt, fps, received, dropped, seq, nodeId,
   *                           subcarriers, frameAgeMs, stale }
   */
  function update(frame, stats) {
    if (disposed) return;

    const safeFrame = frame && typeof frame === 'object' ? frame : null;
    const safeStats = stats && typeof stats === 'object' ? stats : {};
    const dt = isFiniteNumber(safeStats.dt) && safeStats.dt > 0 ? safeStats.dt : 1 / 60;

    /* ------------------------------------------------------------- vitals */

    const vitalsData = safeFrame && safeFrame.vitals && typeof safeFrame.vitals === 'object'
      ? safeFrame.vitals
      : null;

    const bpm = vitalsData ? vitalsData.bpm : null;
    const rpm = vitalsData ? vitalsData.rpm : null;
    const bpmSnr = vitalsData ? vitalsData.bpm_snr : null;
    const rpmSnr = vitalsData ? vitalsData.rpm_snr : null;
    const confidence = vitalsData ? vitalsData.confidence : null;

    writeText('bpm', fields.bpm, fmt(bpm, 1));
    writeText('rpm', fields.rpm, fmt(rpm, 1));
    writeText('bpm_snr', fields.bpm_snr, fmt(bpmSnr, 1));
    writeText('rpm_snr', fields.rpm_snr, fmt(rpmSnr, 1));

    // Confidence: number as text, bar animated toward the target and coloured
    // along the ramp (red < 0.35, amber in between, green >= 0.7).
    const hasConfidence = isFiniteNumber(confidence);
    const confTarget = hasConfidence ? clamp(confidence, 0, 1) : 0;
    confDisplay = damp(confDisplay, confTarget, HUD.confidenceRate, dt);
    if (confDisplay < 0.001) confDisplay = 0;
    if (confDisplay > 0.999) confDisplay = 1;

    const ramp = confidenceColor(hasConfidence ? confTarget : 0);
    confidenceFill.style.width = `${(confDisplay * 100).toFixed(1)}%`;
    confidenceFill.style.backgroundColor = ramp.css;
    confidenceFill.style.boxShadow = confDisplay > 0.01
      ? `0 0 12px ${ramp.css}, 0 0 3px ${ramp.css}`
      : 'none';
    writeText('confidence', confidenceValue, hasConfidence ? confidence.toFixed(2) : '--');
    confidenceValue.style.color = hasConfidence ? ramp.css : '';

    /* ------------------------------------------------------- motion / presence */

    const motion = safeFrame ? safeFrame.motion : null;
    writeText('motion', fields.motion, isFiniteNumber(motion) ? motion.toFixed(2) : '--');

    const presence = safeFrame ? safeFrame.presence : null;
    let presenceLabel = '--';
    let presenceState = 'unknown';
    if (presence === true) {
      presenceLabel = 'DETECTED';
      presenceState = 'detected';
    } else if (presence === false) {
      presenceLabel = 'CLEAR';
      presenceState = 'clear';
    }
    writeText('presence', fields.presence, presenceLabel);
    if (written.presenceState !== presenceState) {
      written.presenceState = presenceState;
      fields.presence.setAttribute('data-presence', presenceState);
    }

    const stale = safeStats.stale === true || safeFrame === null;
    const staleKey = stale ? 'stale' : 'live';
    if (written.staleState !== staleKey) {
      written.staleState = staleKey;
      if (stale) vitals.classList.add('stale');
      else vitals.classList.remove('stale');
    }

    /* ---------------------------------------------------------------- stats */

    const nowMs = typeof performance !== 'undefined' && performance.now
      ? performance.now()
      : Date.now();
    // Continuous values (FPS, frame age) are throttled, but the arrival of a
    // new frame writes through immediately so the panel never lags a frame.
    if (safeStats.frameFresh === true || nowMs - lastStatsWriteMs >= HUD.statsIntervalMs) {
      lastStatsWriteMs = nowMs;

      writeText('fps', statNodes.fps, isFiniteNumber(safeStats.fps) ? String(Math.round(safeStats.fps)) : '--');
      writeText('received', statNodes.received, fmtCount(safeStats.received));
      writeText('dropped', statNodes.dropped, fmtCount(safeStats.dropped));
      writeText('malformed', statNodes.malformed, fmtCount(safeStats.malformed));
      writeText('seq', statNodes.seq, fmtInt(safeStats.seq));
      writeText('node', statNodes.node, fmtInt(safeStats.nodeId));
      writeText('subcarriers', statNodes.subcarriers, fmtInt(safeStats.subcarriers));
      writeText('age', statNodes.age, fmtAge(safeStats.frameAgeMs));

      if (safeStats.dropped !== written.droppedWarnState) {
        const dropping = isFiniteNumber(safeStats.dropped) && safeStats.dropped > 0;
        written.droppedWarnState = safeStats.dropped;
        if (dropping) statsPanel.classList.add('has-drops');
        else statsPanel.classList.remove('has-drops');
      }
    }
  }

  /**
   * @param {string} text status label, e.g. "reconnecting in 2.0 s"
   * @param {'connecting'|'online'|'offline'} state
   */
  function setStatus(text, state) {
    if (disposed) return;
    const safeState = STATUS_STATES.includes(state) ? state : 'connecting';
    const label = typeof text === 'string' && text.length > 0 ? text : safeState;

    writeText('status', statusText, label);
    if (written.statusState !== safeState) {
      written.statusState = safeState;
      statusChip.setAttribute('data-status', safeState);
      vitals.setAttribute('data-connection', safeState);
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    resetButton.removeEventListener('click', handleResetClick);
    if (vitals.parentNode) vitals.parentNode.removeChild(vitals);
    if (statsPanel.parentNode) statsPanel.parentNode.removeChild(statsPanel);
    if (ownsRoot && root.parentNode) root.parentNode.removeChild(root);
    api.onResetView = null;
  }

  api.update = update;
  api.setStatus = setStatus;
  api.dispose = dispose;

  // Seed the DOM with the connecting state.
  setStatus('connecting', 'connecting');

  return api;
}
