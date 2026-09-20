/**
 * Spectraflow — Three.js viewport.
 *
 * Responsibilities:
 *   - renderer / camera / lighting / floor grid for a 5 m x 5 m x 3 m volume
 *   - one wireframe "signal dome" per AP node, driven by the frame `power`
 *     array (mean -> strength, std -> fluctuation) and pulsed by `motion`
 *   - resize handling that survives rotation, soft keyboards and Android
 *     visual-viewport changes
 *   - OrbitControls when the addon can be fetched, otherwise a fully featured
 *     built-in mouse/touch orbit + wheel/pinch zoom fallback
 *
 * All animation is frame-rate independent: every smoothed quantity moves with
 * an exponential approach (`damp`) or an exact critically-damped spring, so the
 * result does not depend on the device refresh rate.
 */

import * as THREE from 'three';
import {
  COLORS,
  DOME,
  NODES,
  RENDER,
  VOLUME,
  clamp,
  damp,
  isFiniteNumber,
  nodeColorHex,
  nodePosition,
  numOr,
  safeDelta,
  summarizeNumbers,
} from './config.js';

/* -------------------------------------------------------------------------- */
/* Built-in orbit controls (used when the addon is unavailable)                */
/* -------------------------------------------------------------------------- */

/**
 * Self-contained orbit / pan / zoom controller.
 *
 * Deliberately uses the pointer-events API so a single implementation covers
 * mouse, pen and touch: one pointer drags to orbit, two pointers pinch to zoom,
 * the wheel zooms on desktop. Inertia keeps touch dragging feeling native.
 */
class BuiltinOrbitControls {
  constructor(camera, domElement, target) {
    this.camera = camera;
    this.domElement = domElement;
    this.target = new THREE.Vector3(
      numOr(target && target[0], 0),
      numOr(target && target[1], 0),
      numOr(target && target[2], 0),
    );
    this.enabled = true;

    // Spherical state around `target`.
    this.radius = 8;
    this.theta = 0.9;
    this.phi = 1.05;

    // Velocities for inertia (rad/s, m/s).
    this.thetaVelocity = 0;
    this.phiVelocity = 0;
    this.zoomVelocity = 0;

    this.pointers = new Map();
    this.pinchDistance = 0;
    this.lastMoveMs = 0;
    this.dragging = false;

    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
    this.handleWheel = this.handleWheel.bind(this);
    this.handleContextMenu = this.handleContextMenu.bind(this);

    this.syncFromCamera();

    domElement.addEventListener('pointerdown', this.handlePointerDown, { passive: false });
    domElement.addEventListener('pointermove', this.handlePointerMove, { passive: false });
    domElement.addEventListener('pointerup', this.handlePointerUp);
    domElement.addEventListener('pointercancel', this.handlePointerUp);
    domElement.addEventListener('pointerleave', this.handlePointerUp);
    domElement.addEventListener('wheel', this.handleWheel, { passive: false });
    domElement.addEventListener('contextmenu', this.handleContextMenu);
  }

  /** Adopt the camera's current position as the orbit state. */
  syncFromCamera() {
    const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
    const radius = offset.length();
    if (!(radius > 1e-4)) return;

    this.radius = radius;
    this.phi = clamp(Math.acos(clamp(offset.y / radius, -1, 1)), RENDER.minPolarAngle, RENDER.maxPolarAngle);
    this.theta = Math.atan2(offset.x, offset.z);
  }

  /** Push the orbit state back onto the camera. */
  applyToCamera() {
    const sinPhi = Math.sin(this.phi);
    this.camera.position.set(
      this.target.x + this.radius * sinPhi * Math.sin(this.theta),
      this.target.y + this.radius * Math.cos(this.phi),
      this.target.z + this.radius * sinPhi * Math.cos(this.theta),
    );
    this.camera.lookAt(this.target);
  }

  pointerDistance() {
    if (this.pointers.size < 2) return 0;
    const points = Array.from(this.pointers.values());
    const dx = points[0].x - points[1].x;
    const dy = points[0].y - points[1].y;
    return Math.hypot(dx, dy);
  }

  handlePointerDown(event) {
    if (!this.enabled) return;
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (this.pointers.size === 2) this.pinchDistance = this.pointerDistance();
    this.dragging = this.pointers.size === 1;
    this.thetaVelocity = 0;
    this.phiVelocity = 0;
    this.lastMoveMs = performance.now();
    if (this.domElement.setPointerCapture) {
      try {
        this.domElement.setPointerCapture(event.pointerId);
      } catch (error) {
        // Capture is a nicety; a detached pointer still works through the
        // container-level listeners.
      }
    }
    if (event.cancelable) event.preventDefault();
  }

  handlePointerMove(event) {
    if (!this.enabled) return;
    const previous = this.pointers.get(event.pointerId);
    if (!previous) return;

    const now = performance.now();
    const dt = Math.max((now - this.lastMoveMs) / 1000, 1 / 240);
    const dx = event.clientX - previous.x;
    const dy = event.clientY - previous.y;

    previous.x = event.clientX;
    previous.y = event.clientY;
    this.lastMoveMs = now;

    if (this.pointers.size >= 2) {
      const distance = this.pointerDistance();
      if (this.pinchDistance > 1 && distance > 1) {
        const scale = this.pinchDistance / distance;
        this.radius = clamp(this.radius * scale, RENDER.minDistance, RENDER.maxDistance);
      }
      this.pinchDistance = distance;
      this.thetaVelocity = 0;
      this.phiVelocity = 0;
      if (event.cancelable) event.preventDefault();
      return;
    }

    if (!this.dragging) return;

    const speed = event.pointerType === 'touch'
      ? RENDER.orbit.rotateSpeedTouch
      : RENDER.orbit.rotateSpeedMouse;

    const deltaTheta = -dx * speed;
    const deltaPhi = -dy * speed;

    this.theta += deltaTheta;
    this.phi = clamp(this.phi + deltaPhi, RENDER.minPolarAngle, RENDER.maxPolarAngle);

    // Exponential moving average of the angular velocity, for release inertia.
    this.thetaVelocity = damp(this.thetaVelocity, deltaTheta / dt, 6, dt);
    this.phiVelocity = damp(this.phiVelocity, deltaPhi / dt, 6, dt);

    if (event.cancelable) event.preventDefault();
  }

  handlePointerUp(event) {
    if (!this.pointers.has(event.pointerId)) return;
    this.pointers.delete(event.pointerId);
    if (this.pointers.size < 2) this.pinchDistance = 0;
    if (this.pointers.size === 0) this.dragging = false;
  }

  handleWheel(event) {
    if (!this.enabled) return;
    if (event.cancelable) event.preventDefault();
    const delta = numOr(event.deltaY, 0);
    this.zoomVelocity = clamp(-delta * RENDER.orbit.zoomSpeed * 8, -6, 6);
  }

  handleContextMenu(event) {
    event.preventDefault();
  }

  /** Snap the orbit state to a configuration and stop all motion. */
  reset(cameraPosition, target) {
    this.target.set(
      numOr(target && target[0], 0),
      numOr(target && target[1], 0),
      numOr(target && target[2], 0),
    );
    this.thetaVelocity = 0;
    this.phiVelocity = 0;
    this.zoomVelocity = 0;
    this.camera.position.set(
      numOr(cameraPosition && cameraPosition[0], 5),
      numOr(cameraPosition && cameraPosition[1], 3),
      numOr(cameraPosition && cameraPosition[2], 6),
    );
    this.syncFromCamera();
    this.applyToCamera();
  }

  /** @param {number} dt seconds since the previous update */
  update(dt) {
    const step = safeDelta(dt, 0.05);
    if (step > 0 && this.pointers.size === 0) {
      this.theta += this.thetaVelocity * step;
      this.phi = clamp(this.phi + this.phiVelocity * step, RENDER.minPolarAngle, RENDER.maxPolarAngle);
      this.thetaVelocity = damp(this.thetaVelocity, 0, RENDER.orbit.inertia, step);
      this.phiVelocity = damp(this.phiVelocity, 0, RENDER.orbit.inertia, step);
    }
    if (step > 0 && Math.abs(this.zoomVelocity) > 1e-4) {
      this.radius = clamp(
        this.radius * Math.exp(-this.zoomVelocity * step),
        RENDER.minDistance,
        RENDER.maxDistance,
      );
      this.zoomVelocity = damp(this.zoomVelocity, 0, RENDER.orbit.damping, step);
    }
    this.applyToCamera();
  }

  dispose() {
    this.domElement.removeEventListener('pointerdown', this.handlePointerDown);
    this.domElement.removeEventListener('pointermove', this.handlePointerMove);
    this.domElement.removeEventListener('pointerup', this.handlePointerUp);
    this.domElement.removeEventListener('pointercancel', this.handlePointerUp);
    this.domElement.removeEventListener('pointerleave', this.handlePointerUp);
    this.domElement.removeEventListener('wheel', this.handleWheel);
    this.domElement.removeEventListener('contextmenu', this.handleContextMenu);
    this.pointers.clear();
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function resolveContainer(container) {
  if (typeof container === 'string') {
    const found = document.querySelector(container);
    if (!found) throw new Error(`createScene: no element matches "${container}"`);
    return found;
  }
  if (container && typeof container.appendChild === 'function') return container;
  throw new Error('createScene: a container element (or CSS selector) is required');
}

/** Recursively dispose geometries, materials and textures of a subtree. */
function disposeObject(root) {
  if (!root || typeof root.traverse !== 'function') return;
  root.traverse((object) => {
    if (object.geometry && typeof object.geometry.dispose === 'function') {
      object.geometry.dispose();
    }
    const material = object.material;
    if (!material) return;
    const list = Array.isArray(material) ? material : [material];
    for (const entry of list) {
      if (!entry || typeof entry.dispose !== 'function') continue;
      for (const key of Object.keys(entry)) {
        const value = entry[key];
        if (value && value.isTexture && typeof value.dispose === 'function') value.dispose();
      }
      entry.dispose();
    }
  });
}

/* -------------------------------------------------------------------------- */
/* Scene factory                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Create the 3D viewport.
 *
 * @param {HTMLElement|string} container element that will host the canvas
 * @param {{ onError?: (message: string) => void }} [options]
 * @returns {{
 *   scene: THREE.Scene, camera: THREE.PerspectiveCamera,
 *   renderer: THREE.WebGLRenderer, controls: object,
 *   update: (frame: object|null) => void, resize: () => void,
 *   render: () => void, dispose: () => void,
 *   setNodes: (nodes: Array<object>) => void, resetView: () => void,
 *   controlsKind: string, nodeCount: number, lastActivity: number,
 * }}
 */
export function createScene(container, options = {}) {
  const host = resolveContainer(container);
  const onError = typeof options.onError === 'function' ? options.onError : null;

  /* ---------------------------------------------------------------- renderer */

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
  } catch (error) {
    throw new Error(`WebGL is not available in this browser (${error && error.message ? error.message : 'context creation failed'})`);
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, RENDER.maxPixelRatio));
  renderer.setClearColor(COLORS.background, 1);
  if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;

  const canvas = renderer.domElement;
  canvas.style.position = 'absolute';
  canvas.style.inset = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  canvas.style.touchAction = 'none';
  host.appendChild(canvas);

  let contextLost = false;
  const handleContextLost = (event) => {
    event.preventDefault();
    contextLost = true;
    if (onError) onError('The GPU context was lost. Reload the page to restore the 3D view.');
  };
  const handleContextRestored = () => {
    contextLost = false;
  };
  canvas.addEventListener('webglcontextlost', handleContextLost, false);
  canvas.addEventListener('webglcontextrestored', handleContextRestored, false);

  /* ------------------------------------------------------------------ scene */

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.background);
  scene.fog = new THREE.Fog(COLORS.fog, RENDER.fogNear, RENDER.fogFar);

  const targetVector = new THREE.Vector3(
    RENDER.target[0], RENDER.target[1], RENDER.target[2],
  );

  const camera = new THREE.PerspectiveCamera(
    RENDER.fov,
    1,
    RENDER.near,
    RENDER.far,
  );
  camera.position.set(
    RENDER.cameraStart[0], RENDER.cameraStart[1], RENDER.cameraStart[2],
  );
  camera.lookAt(targetVector);

  /* ---------------------------------------------------------------- lighting */

  const ambient = new THREE.AmbientLight(COLORS.ambient, 0.55);
  scene.add(ambient);

  const hemisphere = new THREE.HemisphereLight(
    COLORS.hemisphereSky, COLORS.hemisphereGround, 0.85,
  );
  hemisphere.position.set(0, VOLUME.height * 1.5, 0);
  scene.add(hemisphere);

  const keyLight = new THREE.DirectionalLight(COLORS.directional, 1.05);
  keyLight.position.set(4.5, 7.5, 5.5);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(COLORS.gridMain, 0.35);
  fillLight.position.set(-5.5, 3.5, -4.5);
  scene.add(fillLight);

  /* -------------------------------------------------------------- floor grid */

  const floorGroup = new THREE.Group();
  floorGroup.name = 'floor';
  scene.add(floorGroup);

  const floorPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(VOLUME.outerGridSize, VOLUME.outerGridSize),
    new THREE.MeshStandardMaterial({
      color: COLORS.floorPlane,
      roughness: 0.96,
      metalness: 0.05,
      transparent: true,
      opacity: 0.85,
    }),
  );
  floorPlane.rotation.x = -Math.PI / 2;
  floorPlane.position.y = -0.002;
  floorGroup.add(floorPlane);

  const outerGrid = new THREE.GridHelper(
    VOLUME.outerGridSize,
    VOLUME.outerGridDivisions,
    COLORS.gridOuter,
    COLORS.gridOuter,
  );
  outerGrid.material.transparent = true;
  outerGrid.material.opacity = 0.4;
  outerGrid.material.depthWrite = false;
  floorGroup.add(outerGrid);

  const volumeGrid = new THREE.GridHelper(
    VOLUME.gridSize,
    VOLUME.gridDivisions,
    COLORS.gridMain,
    COLORS.gridSub,
  );
  volumeGrid.material.transparent = true;
  volumeGrid.material.opacity = 0.55;
  volumeGrid.material.depthWrite = false;
  volumeGrid.position.y = 0.004;
  floorGroup.add(volumeGrid);

  /** Wireframe outline of the sensing volume, for scale reference. */
  const volumeBox = new THREE.LineSegments(
    new THREE.EdgesGeometry(
      new THREE.BoxGeometry(VOLUME.width, VOLUME.height, VOLUME.depth),
    ),
    new THREE.LineBasicMaterial({
      color: COLORS.volumeBox,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    }),
  );
  volumeBox.position.y = VOLUME.height / 2;
  volumeBox.name = 'volumeBox';
  scene.add(volumeBox);

  /* ----------------------------------------------------------- shared assets */

  // Unit hemisphere: thetaLength = PI/2 keeps the top half, so a dome bulges
  // upward from its AP position. Scaled per node instead of rebuilt.
  const domeGeometry = new THREE.SphereGeometry(
    1,
    DOME.widthSegments,
    DOME.heightSegments,
    0,
    Math.PI * 2,
    0,
    Math.PI / 2,
  );
  const markerGeometry = new THREE.OctahedronGeometry(NODES.markerRadius, 0);

  const domeGroup = new THREE.Group();
  domeGroup.name = 'signalDomes';
  scene.add(domeGroup);

  /** @type {Map<number, object>} node id -> dome record */
  const nodes = new Map();

  function createDome(nodeId, position) {
    const colorHex = nodeColorHex(nodeId);
    const color = new THREE.Color(colorHex);

    const group = new THREE.Group();
    group.position.set(position[0], position[1], position[2]);
    group.name = `node-${nodeId}`;

    const wireMaterial = new THREE.MeshBasicMaterial({
      color,
      wireframe: true,
      transparent: true,
      opacity: DOME.minOpacity,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const wire = new THREE.Mesh(domeGeometry, wireMaterial);
    wire.scale.setScalar(DOME.baseRadius * DOME.minRadiusScale);
    group.add(wire);

    const glowMaterial = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: DOME.glowOpacity,
      depthWrite: false,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
    });
    const glow = new THREE.Mesh(domeGeometry, glowMaterial);
    glow.scale.setScalar(DOME.baseRadius * DOME.minRadiusScale);
    group.add(glow);

    const markerMaterial = new THREE.MeshStandardMaterial({
      color: COLORS.nodeMarker,
      emissive: color,
      emissiveIntensity: 0.75,
      roughness: 0.3,
      metalness: 0.2,
    });
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    group.add(marker);

    domeGroup.add(group);

    const record = {
      id: nodeId,
      group,
      wire,
      glow,
      marker,
      color,
      position: [position[0], position[1], position[2]],
      activity: 0,
      targetActivity: 0,
      radius: DOME.baseRadius * DOME.minRadiusScale,
      targetRadius: DOME.baseRadius * DOME.minRadiusScale,
      opacity: DOME.minOpacity,
      targetOpacity: DOME.minOpacity,
      pulsePhase: 0,
      lastSeenMs: performance.now(),
    };
    nodes.set(nodeId, record);
    return record;
  }

  function ensureDome(nodeId) {
    const existing = nodes.get(nodeId);
    if (existing) return existing;
    return createDome(nodeId, nodePosition(nodeId));
  }

  /* -------------------------------------------------------------- controls */

  const builtinControls = new BuiltinOrbitControls(camera, canvas, RENDER.target);
  builtinControls.reset(RENDER.cameraStart, RENDER.target);

  let activeControls = builtinControls;
  let controlsKind = 'builtin';
  let disposed = false;

  // Upgrade to the official OrbitControls when the addon is reachable. The
  // built-in controller above is already active, so a failed fetch (offline,
  // blocked CDN, no import-map support) simply leaves it in place.
  if (typeof import.meta !== 'undefined') {
    import('three/addons/controls/OrbitControls.js')
      .then((addon) => {
        if (disposed || !addon || typeof addon.OrbitControls !== 'function') return;
        const orbit = new addon.OrbitControls(camera, canvas);
        orbit.target.copy(targetVector);
        orbit.enableDamping = true;
        orbit.dampingFactor = 0.075;
        orbit.rotateSpeed = 0.85;
        orbit.zoomSpeed = 0.9;
        orbit.panSpeed = 0.7;
        orbit.screenSpacePanning = true;
        orbit.minDistance = RENDER.minDistance;
        orbit.maxDistance = RENDER.maxDistance;
        orbit.minPolarAngle = RENDER.minPolarAngle;
        orbit.maxPolarAngle = RENDER.maxPolarAngle;
        orbit.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
        orbit.update();

        builtinControls.dispose();
        activeControls = orbit;
        controlsKind = 'orbit-controls';
      })
      .catch(() => {
        // Keep the built-in controller; orbiting and zooming still work.
      });
  }

  /* ----------------------------------------------------------------- resize */

  function resize() {
    const width = Math.max(host.clientWidth || 0, window.innerWidth || 1, 1);
    const height = Math.max(host.clientHeight || 0, window.innerHeight || 1, 1);
    const aspect = width / height;

    if (camera.aspect !== aspect) {
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, RENDER.maxPixelRatio));
    renderer.setSize(width, height, false);
  }

  let resizeObserver = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(host);
  }
  const handleWindowResize = () => resize();
  window.addEventListener('resize', handleWindowResize);
  window.addEventListener('orientationchange', handleWindowResize);
  const viewport = window.visualViewport;
  if (viewport && typeof viewport.addEventListener === 'function') {
    viewport.addEventListener('resize', handleWindowResize);
  }
  resize();

  function detachResizeListeners() {
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    window.removeEventListener('resize', handleWindowResize);
    window.removeEventListener('orientationchange', handleWindowResize);
    if (viewport && typeof viewport.removeEventListener === 'function') {
      viewport.removeEventListener('resize', handleWindowResize);
    }
  }

  /* --------------------------------------------------------------- animation */

  const powerSummary = { count: 0, mean: 0, std: 0, min: 0, max: 0, span: 0 };
  let pulseAmplitude = 0;
  let lastTickMs = performance.now();
  let lastActivity = 0;
  /** Delta of the most recent `update()`, reused by `render()` for controls. */
  let lastDelta = 1 / 60;

  /**
   * Fold one frame's `power` array into a 0..1 activity value.
   * Every entry may be null, in which case activity decays toward zero.
   */
  function activityFromPower(power) {
    summarizeNumbers(power, powerSummary);
    if (powerSummary.count === 0) return 0;

    const strength = clamp(
      (powerSummary.mean - DOME.strengthDbFloor) / (DOME.strengthDbCeil - DOME.strengthDbFloor),
      0,
      1,
    );
    const fluctuation = clamp(powerSummary.std / DOME.fluctuationDbStd, 0, 1);
    return clamp(
      strength * DOME.strengthWeight + fluctuation * DOME.fluctuationWeight,
      0,
      1,
    );
  }

  /**
   * Consume one sensing frame. Safe with `null` (the viewport keeps animating
   * toward its previous targets) and with any missing/`null` field.
   */
  function update(frame) {
    const nowMs = performance.now();
    const dt = safeDelta((nowMs - lastTickMs) / 1000, RENDER.maxDeltaSeconds);
    lastTickMs = nowMs;

    if (dt <= 0) {
      return;
    }
    lastDelta = dt;

    const safeFrame = frame && typeof frame === 'object' ? frame : null;

    if (safeFrame) {
      const motion = clamp(numOr(safeFrame.motion, 0), 0, 1);
      pulseAmplitude = damp(pulseAmplitude, motion, DOME.motionRate, dt);

      const nodeId = Math.round(numOr(safeFrame.node_id, 1));
      const record = ensureDome(nodeId);
      record.targetActivity = activityFromPower(safeFrame.power);
      record.lastSeenMs = nowMs;
      lastActivity = record.targetActivity;
    } else {
      pulseAmplitude = damp(pulseAmplitude, 0, DOME.motionRate, dt);
      lastActivity = damp(lastActivity, 0, DOME.motionRate, dt);
    }

    for (const record of nodes.values()) {
      const age = nowMs - record.lastSeenMs;
      if (age > 5000) {
        // Node went quiet: relax the dome instead of freezing it mid-pulse.
        record.targetActivity = damp(record.targetActivity, 0, 0.6, dt);
      }

      record.activity = damp(record.activity, record.targetActivity, DOME.activityRate, dt);

      record.pulsePhase += dt * (DOME.pulseBaseHz + DOME.pulseHzPerMotion * pulseAmplitude) * Math.PI * 2;
      if (record.pulsePhase > Math.PI * 4) record.pulsePhase -= Math.PI * 4;
      const pulse = Math.sin(record.pulsePhase) * pulseAmplitude;

      const scale = DOME.minRadiusScale + (DOME.maxRadiusScale - DOME.minRadiusScale) * record.activity;
      record.targetRadius = DOME.baseRadius * (scale + pulse * DOME.pulseRadiusAmp);
      record.targetOpacity = clamp(
        DOME.minOpacity + (DOME.maxOpacity - DOME.minOpacity) * record.activity
          + pulse * DOME.pulseOpacityAmp,
        0.02,
        0.95,
      );

      // Opacity and radius are lerped toward their targets, never snapped.
      record.radius = damp(record.radius, record.targetRadius, DOME.radiusRate, dt);
      record.opacity = damp(record.opacity, record.targetOpacity, DOME.opacityRate, dt);

      record.wire.scale.setScalar(record.radius);
      record.wire.material.opacity = record.opacity;
      record.glow.scale.setScalar(record.radius * 0.96);
      record.glow.material.opacity = record.opacity * 0.18 + DOME.glowOpacity * record.activity;
      record.wire.rotation.y += dt * 0.06;

      const markerPulse = 1 + 0.25 * record.activity + 0.35 * Math.max(pulse, 0);
      record.marker.scale.setScalar(markerPulse);
      record.marker.material.emissiveIntensity = 0.5 + 1.4 * record.activity;
    }

    volumeBox.material.opacity = 0.16 + 0.16 * lastActivity;
  }

  function render() {
    if (disposed || contextLost) return;
    if (controlsKind === 'orbit-controls') {
      activeControls.update();
    } else {
      activeControls.update(lastDelta);
    }
    renderer.render(scene, camera);
  }

  /**
   * Optionally announce real AP geometry (from a `hello`/`stats` message).
   * Entries may be `{ id, position: [x,y,z] }`, `{ node_id, x, y, z }` or
   * `[id, x, y, z]`. Malformed entries are skipped.
   */
  function setNodes(list) {
    if (!Array.isArray(list)) return;
    for (const entry of list) {
      if (!entry) continue;

      let id = null;
      let position = null;

      if (Array.isArray(entry) && entry.length >= 4 && isFiniteNumber(entry[0])) {
        id = Math.round(entry[0]);
        position = [entry[1], entry[2], entry[3]];
      } else if (typeof entry === 'object') {
        id = Math.round(numOr(entry.id, numOr(entry.node_id, NaN)));
        if (!Number.isFinite(id)) continue;
        if (Array.isArray(entry.position)) {
          position = entry.position;
        } else if (isFiniteNumber(entry.x) && isFiniteNumber(entry.y) && isFiniteNumber(entry.z)) {
          position = [entry.x, entry.y, entry.z];
        }
      } else {
        continue;
      }

      if (!Number.isFinite(id)) continue;
      if (!position || !isFiniteNumber(position[0]) || !isFiniteNumber(position[1]) || !isFiniteNumber(position[2])) {
        position = nodePosition(id);
      }

      const record = nodes.get(id);
      if (record) {
        record.position[0] = position[0];
        record.position[1] = position[1];
        record.position[2] = position[2];
        record.group.position.set(position[0], position[1], position[2]);
      } else {
        createDome(id, position);
      }
    }
  }

  /** Restore the default camera framing (used by the HUD "reset view"). */
  function resetView() {
    if (controlsKind === 'orbit-controls') {
      camera.position.set(RENDER.cameraStart[0], RENDER.cameraStart[1], RENDER.cameraStart[2]);
      targetVector.set(RENDER.target[0], RENDER.target[1], RENDER.target[2]);
      if (activeControls.target && typeof activeControls.target.copy === 'function') {
        activeControls.target.copy(targetVector);
      }
      camera.lookAt(targetVector);
      activeControls.update();
      return;
    }
    builtinControls.reset(RENDER.cameraStart, RENDER.target);
  }

  function dispose() {
    if (disposed) return;
    disposed = true;

    detachResizeListeners();
    canvas.removeEventListener('webglcontextlost', handleContextLost);
    canvas.removeEventListener('webglcontextrestored', handleContextRestored);

    if (activeControls && typeof activeControls.dispose === 'function') activeControls.dispose();
    if (activeControls !== builtinControls && typeof builtinControls.dispose === 'function') {
      builtinControls.dispose();
    }

    nodes.clear();
    disposeObject(scene);
    scene.clear();

    renderer.dispose();
    if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
  }

  return {
    scene,
    camera,
    renderer,
    controls: builtinControls,
    update,
    resize,
    render,
    dispose,
    setNodes,
    resetView,
    get controlsKind() {
      return controlsKind;
    },
    get nodeCount() {
      return nodes.size;
    },
    /**
     * Signal activity (0..1) of the newest frame. It decays toward zero while
     * no frames arrive, so the volume outline relaxes when the sensor goes
     * quiet instead of freezing on its last value.
     */
    get lastActivity() {
      return lastActivity;
    },
  };
}
