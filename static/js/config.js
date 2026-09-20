/**
 * Spectraflow — shared constants, COCO-17 keypoint topology, palette and
 * frame-rate independent math helpers.
 *
 * This module intentionally imports NOTHING (not even Three.js) so that it can
 * always be evaluated, even when the WebGL layer fails to initialise. `app.js` and
 * `hud.js` depend on it directly; `scene.js` and `avatar.js` add Three.js on
 * top. Keeping the dependency-free layer separate is what makes the offline
 * error path in `app.js` possible.
 *
 * Coordinate system (frozen data contract):
 *   right-handed, +Y is up, metres, origin = centre of the sensing volume on
 *   the floor plane. X is lateral, Z is depth.
 */

/* -------------------------------------------------------------------------- */
/* App identity                                                               */
/* -------------------------------------------------------------------------- */

export const APP = Object.freeze({
  name: 'Spectraflow',
  tagline: 'Wi-Fi CSI volumetric sensing',
  version: '1.0.0',
});

/* -------------------------------------------------------------------------- */
/* Sensing volume                                                             */
/* -------------------------------------------------------------------------- */

/** Physical extent of the sensing volume in metres. */
export const VOLUME = Object.freeze({
  width: 5.0,
  height: 3.0,
  depth: 5.0,
  minY: 0.0,
  maxY: 3.0,
  /** Bright grid drawn under the sensing volume. */
  gridSize: 5.0,
  gridDivisions: 20,
  /** Dim context grid extending past the volume. */
  outerGridSize: 15.0,
  outerGridDivisions: 30,
  /** Distance from the origin to the volume wall, on both X and Z. */
  halfWidth: 2.5,
  halfDepth: 2.5,
});

/* -------------------------------------------------------------------------- */
/* COCO-17 keypoint topology                                                  */
/* -------------------------------------------------------------------------- */

/** Frozen keypoint order. Index === position in the `keypoints` array. */
export const KEYPOINT_NAMES = Object.freeze([
  'nose',            // 0
  'left_eye',        // 1
  'right_eye',       // 2
  'left_ear',        // 3
  'right_ear',       // 4
  'left_shoulder',   // 5
  'right_shoulder',  // 6
  'left_elbow',      // 7
  'right_elbow',     // 8
  'left_wrist',      // 9
  'right_wrist',     // 10
  'left_hip',        // 11
  'right_hip',       // 12
  'left_knee',       // 13
  'right_knee',      // 14
  'left_ankle',      // 15
  'right_ankle',     // 16
]);

export const KEYPOINT_COUNT = KEYPOINT_NAMES.length; // 17

/** Bone list drawn as glowing segments (COCO-17 index pairs). */
export const BONES = Object.freeze([
  Object.freeze([0, 1]),   // nose -> left_eye
  Object.freeze([0, 2]),   // nose -> right_eye
  Object.freeze([1, 3]),   // left_eye -> left_ear
  Object.freeze([2, 4]),   // right_eye -> right_ear
  Object.freeze([5, 6]),   // shoulder girdle
  Object.freeze([5, 7]),   // left upper arm
  Object.freeze([7, 9]),   // left forearm
  Object.freeze([6, 8]),   // right upper arm
  Object.freeze([8, 10]),  // right forearm
  Object.freeze([5, 11]),  // left torso
  Object.freeze([6, 12]),  // right torso
  Object.freeze([11, 12]), // pelvis
  Object.freeze([11, 13]), // left thigh
  Object.freeze([13, 15]), // left shin
  Object.freeze([12, 14]), // right thigh
  Object.freeze([14, 16]), // right shin
]);

export const BONE_COUNT = BONES.length; // 16

/** Logical joint groups; drives per-joint size and colour. */
export const JOINT_ROLES = Object.freeze({
  head: Object.freeze([0, 1, 2, 3, 4]),
  trunk: Object.freeze([5, 6, 11, 12]),
  arm: Object.freeze([7, 8]),
  hand: Object.freeze([9, 10]),
  leg: Object.freeze([13, 14]),
  foot: Object.freeze([15, 16]),
});

/** `JOINT_ROLE_BY_INDEX[4] === 'head'` — built once, frozen. */
export const JOINT_ROLE_BY_INDEX = (() => {
  const roles = new Array(KEYPOINT_COUNT).fill('other');
  for (const role of Object.keys(JOINT_ROLES)) {
    for (const index of JOINT_ROLES[role]) {
      if (index >= 0 && index < KEYPOINT_COUNT) roles[index] = role;
    }
  }
  return Object.freeze(roles);
})();

/* -------------------------------------------------------------------------- */
/* Palette                                                                    */
/* -------------------------------------------------------------------------- */

/** Three.js colours (hex ints; `new THREE.Color(hex)` handles sRGB). */
export const COLORS = Object.freeze({
  background: 0x05070d,
  fog: 0x060a12,
  floorPlane: 0x070c16,
  gridMain: 0x2f7ea8,
  gridSub: 0x143347,
  gridOuter: 0x0e1f2e,
  volumeBox: 0x2ea6c8,
  ambient: 0x9ec4e2,
  hemisphereSky: 0x9adcff,
  hemisphereGround: 0x0a1220,
  directional: 0xcfeaff,
  nodeMarker: 0xdff6ff,
  joint: Object.freeze({
    head: 0x8df0ff,
    trunk: 0x4fd2ff,
    arm: 0x2fb6f0,
    hand: 0xffd166,
    leg: 0x9d8bff,
    foot: 0xc79bff,
    other: 0xc9ecff,
  }),
  /** Cycled per AP/node so neighbouring domes stay distinguishable. */
  dome: Object.freeze([0x35e0ff, 0xff4fd8, 0xffb347, 0x38e08a, 0x9d8bff, 0xff6b6b]),
});

/* -------------------------------------------------------------------------- */
/* Renderer / camera tunables                                                 */
/* -------------------------------------------------------------------------- */

export const RENDER = Object.freeze({
  fov: 55,
  near: 0.05,
  far: 200,
  maxPixelRatio: 2,
  /** Orbit target: roughly chest height of a standing person. */
  target: Object.freeze([0.0, 1.15, 0.0]),
  cameraStart: Object.freeze([5.2, 3.2, 6.4]),
  minDistance: 1.8,
  maxDistance: 26,
  /** Camera may not dip below the floor plane. */
  maxPolarAngle: 1.518, // ~87 degrees from +Y
  minPolarAngle: 0.06,
  /** Clamp for a single animation step (guards tab-switch / stall spikes). */
  maxDeltaSeconds: 0.1,
  fogNear: 9,
  fogFar: 34,
  orbit: Object.freeze({
    rotateSpeedTouch: 0.0052,
    rotateSpeedMouse: 0.0042,
    zoomSpeed: 0.0016,
    inertia: 5.5,
    damping: 12.0,
  }),
});

/* -------------------------------------------------------------------------- */
/* Avatar smoothing + look                                                    */
/* -------------------------------------------------------------------------- */

export const AVATAR = Object.freeze({
  /**
   * Angular frequency (rad/s) of the exact critically-damped spring used to
   * move every joint from its current position toward the incoming target.
   * Higher = snappier, lower = smoother. ~13 rad/s settles in ~0.35 s.
   */
  springOmega: 13.5,
  /** Extra clamp for smoothing steps (a long dt must not teleport a limb). */
  maxDeltaSeconds: 0.05,
  fadeInRate: 7.5,
  fadeOutRate: 3.2,
  /** Below this overall opacity the avatar group is hidden entirely. */
  visibleThreshold: 0.012,
  /** Joint sphere radii in metres, per role. */
  jointRadius: Object.freeze({
    head: 0.05,
    trunk: 0.046,
    arm: 0.036,
    hand: 0.044,
    leg: 0.038,
    foot: 0.046,
    other: 0.038,
  }),
  jointHaloScale: 2.35,
  jointCoreOpacity: 0.95,
  jointHaloOpacity: 0.2,
  boneRadius: 0.014,
  boneGlowRadius: 0.032,
  boneOpacity: 0.85,
  boneGlowOpacity: 0.12,
  /** A bone shorter than this is treated as degenerate and hidden. */
  minBoneLength: 0.0015,
});

/* -------------------------------------------------------------------------- */
/* AP / node signal domes                                                     */
/* -------------------------------------------------------------------------- */

export const NODES = Object.freeze({
  /** Default AP placement when the server does not announce node geometry. */
  defaultPositions: Object.freeze({
    1: Object.freeze([-2.3, 2.55, -2.3]),
    2: Object.freeze([2.3, 2.55, -2.3]),
    3: Object.freeze([2.3, 2.55, 2.3]),
    4: Object.freeze([-2.3, 2.55, 2.3]),
  }),
  ringRadius: 2.7,
  ringHeight: 2.55,
  /** Golden angle keeps unknown node ids spread evenly around the volume. */
  goldenAngleDeg: 137.508,
  markerRadius: 0.085,
});

export const DOME = Object.freeze({
  /** Radius in metres, before activity scaling. */
  baseRadius: 1.15,
  minRadiusScale: 0.5,
  maxRadiusScale: 1.35,
  minOpacity: 0.08,
  maxOpacity: 0.6,
  glowOpacity: 0.05,
  /** dB window used to map mean subcarrier power onto 0..1 "strength". */
  strengthDbFloor: -85,
  strengthDbCeil: -40,
  /** Subcarrier std (dB) that counts as full "fluctuation". */
  fluctuationDbStd: 6,
  strengthWeight: 0.6,
  fluctuationWeight: 0.4,
  /** Frame-rate independent smoothing rates (1/s). */
  activityRate: 4.0,
  radiusRate: 8.0,
  opacityRate: 6.0,
  motionRate: 3.0,
  /** Pulse driven by `motion`. */
  pulseBaseHz: 0.55,
  pulseHzPerMotion: 2.4,
  pulseRadiusAmp: 0.09,
  pulseOpacityAmp: 0.16,
  widthSegments: 18,
  heightSegments: 9,
});

/* -------------------------------------------------------------------------- */
/* WebSocket client                                                           */
/* -------------------------------------------------------------------------- */

export const WS = Object.freeze({
  path: '/ws',
  /** Fallback host when the page is not served over http(s). */
  fallbackHost: '127.0.0.1:8000',
  baseDelayMs: 500,
  maxDelayMs: 10_000,
  /** Symmetric jitter: the delay lands within +/- jitterRatio/2 of nominal. */
  jitterRatio: 0.5,
  /** A frame older than this marks the readout as stale. */
  staleFrameMs: 2_500,
  /** Guard against a pathological server frame. */
  maxPayloadBytes: 8 * 1024 * 1024,
});

/* -------------------------------------------------------------------------- */
/* HUD tunables                                                               */
/* -------------------------------------------------------------------------- */

export const HUD = Object.freeze({
  /** Confidence-bar / value animation rate (1/s). */
  confidenceRate: 6.5,
  /** Stats panel DOM writes are throttled to this interval (ms). */
  statsIntervalMs: 250,
  /**
   * Continuous colour ramp anchors: red below 0.35, amber around 0.5,
   * green from 0.7 upward.
   */
  confidenceRamp: Object.freeze([
    Object.freeze({ t: 0.0, rgb: Object.freeze([255, 77, 94]) }),
    Object.freeze({ t: 0.3, rgb: Object.freeze([255, 77, 94]) }),
    Object.freeze({ t: 0.42, rgb: Object.freeze([255, 179, 71]) }),
    Object.freeze({ t: 0.62, rgb: Object.freeze([246, 214, 96]) }),
    Object.freeze({ t: 0.7, rgb: Object.freeze([56, 224, 138]) }),
    Object.freeze({ t: 1.0, rgb: Object.freeze([122, 255, 196]) }),
  ]),
});

/* -------------------------------------------------------------------------- */
/* Null-safe math helpers                                                     */
/* -------------------------------------------------------------------------- */

/** True only for real, finite numbers (rejects null/undefined/NaN/Infinity). */
export function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

/** Returns `value` when it is a finite number, otherwise `fallback`. */
export function numOr(value, fallback) {
  return isFiniteNumber(value) ? value : fallback;
}

export function clamp(value, min, max) {
  if (!isFiniteNumber(value)) return min;
  return value < min ? min : value > max ? max : value;
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Frame-rate independent exponential approach (1/s rate). */
export function damp(current, target, rate, dt) {
  if (!isFiniteNumber(current)) return target;
  if (!(rate > 0) || !(dt > 0)) return current;
  return target + (current - target) * Math.exp(-rate * dt);
}

/**
 * Exact critically-damped spring step (no overshoot, unconditionally stable).
 *
 * Solves x'' = -2*w*x' - w^2*(x - target) in closed form for a step `dt`.
 * Results are written into the caller-supplied two-element `out` array to keep
 * per-joint physics allocation free:
 *   out[0] = new position, out[1] = new velocity.
 */
export function springStep(position, velocity, target, omega, dt, out) {
  const w = isFiniteNumber(omega) && omega > 0 ? omega : 1;
  const h = isFiniteNumber(dt) && dt > 0 ? dt : 0;
  const x = isFiniteNumber(position) ? position : 0;
  const v = isFiniteNumber(velocity) ? velocity : 0;
  const g = isFiniteNumber(target) ? target : x;

  const f = 1 + 2 * h * w;
  const oo = w * w;
  const hoo = h * oo;
  const hhoo = h * hoo;
  const detInv = 1 / (f + hhoo);

  out[0] = (f * x + h * v + hhoo * g) * detInv;
  out[1] = (v + hoo * (g - x)) * detInv;
  return out;
}

/**
 * Mean / std / min / max over the finite entries of an array-like value.
 * Non-finite and non-numeric entries (including `null`) are skipped, so a
 * degraded `power` array never produces NaN.
 */
export function summarizeNumbers(values, out) {
  const result = out || { count: 0, mean: 0, std: 0, min: 0, max: 0, span: 0 };
  result.count = 0;
  result.mean = 0;
  result.std = 0;
  result.min = 0;
  result.max = 0;
  result.span = 0;

  if (!values || typeof values.length !== 'number') return result;

  const n = values.length;
  let count = 0;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < n; i += 1) {
    const v = values[i];
    if (!isFiniteNumber(v)) continue;
    count += 1;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (count === 0) return result;

  const mean = sum / count;
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    const v = values[i];
    if (!isFiniteNumber(v)) continue;
    const d = v - mean;
    acc += d * d;
  }

  result.count = count;
  result.mean = mean;
  result.std = Math.sqrt(acc / count);
  result.min = min;
  result.max = max;
  result.span = max - min;
  return result;
}

/** Reads one `[x, y, z]` triple; returns null unless all three are finite. */
export function readVec3(source, out) {
  if (!source || typeof source.length !== 'number' || source.length < 3) return null;
  const x = source[0];
  const y = source[1];
  const z = source[2];
  if (!isFiniteNumber(x) || !isFiniteNumber(y) || !isFiniteNumber(z)) return null;
  if (out) {
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
  }
  return [x, y, z];
}

/**
 * Piecewise-linear colour along `HUD.confidenceRamp`.
 * Returns `{ css, rgb }`; `null`/non-finite input maps to the red end so a
 * missing confidence never renders as a bright bar.
 */
export function confidenceColor(value) {
  const ramp = HUD.confidenceRamp;
  const t = clamp(numOr(value, 0), 0, 1);

  let lower = ramp[0];
  let upper = ramp[ramp.length - 1];
  for (let i = 0; i < ramp.length - 1; i += 1) {
    if (t >= ramp[i].t && t <= ramp[i + 1].t) {
      lower = ramp[i];
      upper = ramp[i + 1];
      break;
    }
  }

  const span = upper.t - lower.t;
  const k = span > 0 ? (t - lower.t) / span : 0;
  const r = Math.round(lerp(lower.rgb[0], upper.rgb[0], k));
  const g = Math.round(lerp(lower.rgb[1], upper.rgb[1], k));
  const b = Math.round(lerp(lower.rgb[2], upper.rgb[2], k));

  return { css: `rgb(${r}, ${g}, ${b})`, rgb: [r, g, b] };
}

/** Deterministic default position for an AP/node id. */
export function nodePosition(nodeId) {
  const id = Math.round(numOr(nodeId, 1));
  const known = NODES.defaultPositions[id];
  if (known) return [known[0], known[1], known[2]];

  const index = Math.abs(id - 1) % 12;
  const angle = ((NODES.goldenAngleDeg * index) * Math.PI) / 180;
  return [
    NODES.ringRadius * Math.cos(angle),
    NODES.ringHeight,
    NODES.ringRadius * Math.sin(angle),
  ];
}

/** Colour cycled per node id, so domes stay distinguishable. */
export function nodeColorHex(nodeId) {
  const id = Math.abs(Math.round(numOr(nodeId, 1)));
  const palette = COLORS.dome;
  return palette[(id - 1 + palette.length) % palette.length];
}

/** Clamp a raw animation delta to a sane range for smoothing. */
export function safeDelta(dt, maxSeconds) {
  const limit = isFiniteNumber(maxSeconds) ? maxSeconds : RENDER.maxDeltaSeconds;
  if (!isFiniteNumber(dt) || dt <= 0) return 0;
  return dt > limit ? limit : dt;
}
