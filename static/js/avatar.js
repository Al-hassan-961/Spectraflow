/**
 * Spectraflow — 3D skeleton renderer.
 *
 * Renders the 17 COCO-17 keypoints of a frame as glowing joint spheres joined
 * by glowing bone segments.
 *
 * Anti-jitter strategy
 * --------------------
 * Raw keypoints are NEVER rendered. Every joint keeps a rendered position and a
 * velocity, and each frame is advanced toward the incoming target with the
 * closed-form solution of a critically-damped spring:
 *
 *     x'' = -2*w*x' - w^2*(x - target)
 *
 * The closed form is exact for any `dt`, so the motion is identical whether the
 * device renders at 60 Hz, 90 Hz or 120 Hz, and a dropped frame cannot make a
 * limb "pop". The result is smooth without overshoot, and a stale/hidden joint
 * simply holds its last position instead of snapping to the origin.
 */

import * as THREE from 'three';
import {
  AVATAR,
  BONES,
  BONE_COUNT,
  COLORS,
  JOINT_ROLE_BY_INDEX,
  KEYPOINT_COUNT,
  damp,
  numOr,
  readVec3,
  safeDelta,
  springStep,
} from './config.js';

/** Reusable scratch buffers — the render loop allocates nothing per frame. */
const springResult = new Float64Array(2);
const keypointScratch = new Float64Array(3);
const positionScratch = new THREE.Vector3();
const midpointScratch = new THREE.Vector3();
const directionScratch = new THREE.Vector3();
const scaleScratch = new THREE.Vector3();
const quaternionScratch = new THREE.Quaternion();
const matrixScratch = new THREE.Matrix4();
const identityQuaternion = new THREE.Quaternion();
const UP = new THREE.Vector3(0, 1, 0);
/** Effectively invisible, but keeps matrices invertible (never a singular 0). */
const HIDDEN_SCALE = 1e-4;

/**
 * Accept either a `THREE.Scene` or the object returned by `createScene()`.
 */
function resolveParent(sceneLike) {
  if (sceneLike && typeof sceneLike.add === 'function' && typeof sceneLike.remove === 'function'
    && typeof sceneLike.traverse === 'function') {
    return sceneLike;
  }
  if (sceneLike && sceneLike.scene
    && typeof sceneLike.scene.add === 'function' && typeof sceneLike.scene.remove === 'function') {
    return sceneLike.scene;
  }
  throw new Error('createAvatar: expected a THREE.Scene or the object returned by createScene()');
}

function disposeMaterial(material) {
  if (!material) return;
  const list = Array.isArray(material) ? material : [material];
  for (const entry of list) {
    if (entry && typeof entry.dispose === 'function') entry.dispose();
  }
}

/**
 * Create the skeleton renderer.
 *
 * @param {THREE.Scene|{scene: THREE.Scene}} scene scene (or createScene result)
 * @returns {{
 *   group: THREE.Group,
 *   update: (keypoints: Array<Array<number>|null>|null, dt: number) => void,
 *   setVisible: (visible: boolean) => void,
 *   dispose: () => void,
 *   opacity: number, detected: boolean, jointCount: number, boneCount: number,
 * }}
 */
export function createAvatar(scene) {
  const parent = resolveParent(scene);

  const group = new THREE.Group();
  group.name = 'avatar';
  group.visible = false;
  parent.add(group);

  /* ------------------------------------------------------------- geometries */

  // Unit sphere: per-instance scale carries the per-role joint radius.
  const jointGeometry = new THREE.SphereGeometry(1, 16, 12);
  // Unit cylinder along +Y: per-instance scale carries (radius, length, radius).
  const boneGeometry = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);

  /* -------------------------------------------------------------- materials */

  const jointMaterial = new THREE.MeshStandardMaterial({
    color: 0xffffff,            // instance colours come through instanceColor
    emissive: new THREE.Color(COLORS.joint.head),
    emissiveIntensity: 0.35,
    roughness: 0.38,
    metalness: 0.15,
    transparent: true,
    opacity: 0,
    depthWrite: false,
  });

  const jointHaloMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const boneMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const boneGlowMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });

  /* -------------------------------------------------------- instanced joints */

  const joints = new THREE.InstancedMesh(jointGeometry, jointMaterial, KEYPOINT_COUNT);
  joints.name = 'joints';
  joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  joints.frustumCulled = false;
  group.add(joints);

  const jointHalos = new THREE.InstancedMesh(jointGeometry, jointHaloMaterial, KEYPOINT_COUNT);
  jointHalos.name = 'jointHalos';
  jointHalos.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  jointHalos.frustumCulled = false;
  group.add(jointHalos);

  /** @type {number[]} joint radius in metres, per keypoint index */
  const jointRadii = new Array(KEYPOINT_COUNT);
  const jointColors = new Array(KEYPOINT_COUNT);

  for (let i = 0; i < KEYPOINT_COUNT; i += 1) {
    const role = JOINT_ROLE_BY_INDEX[i] || 'other';
    jointRadii[i] = numOr(AVATAR.jointRadius[role], AVATAR.jointRadius.other);
    const hex = numOr(COLORS.joint[role], COLORS.joint.other);
    const color = new THREE.Color(hex);
    jointColors[i] = color;
    joints.setColorAt(i, color);
    jointHalos.setColorAt(i, color);
  }
  if (joints.instanceColor) joints.instanceColor.needsUpdate = true;
  if (jointHalos.instanceColor) jointHalos.instanceColor.needsUpdate = true;

  /* --------------------------------------------------------- bones (2 forms) */

  // Crisp core: literal line segments between joints.
  const bonePositions = new Float32Array(BONE_COUNT * 2 * 3);
  const boneColors = new Float32Array(BONE_COUNT * 2 * 3);
  for (let b = 0; b < BONE_COUNT; b += 1) {
    const [a, c] = BONES[b];
    const colorA = jointColors[a] || new THREE.Color(COLORS.joint.other);
    const colorC = jointColors[c] || new THREE.Color(COLORS.joint.other);
    boneColors[b * 6 + 0] = colorA.r;
    boneColors[b * 6 + 1] = colorA.g;
    boneColors[b * 6 + 2] = colorA.b;
    boneColors[b * 6 + 3] = colorC.r;
    boneColors[b * 6 + 4] = colorC.g;
    boneColors[b * 6 + 5] = colorC.b;
  }

  const boneGeometryLines = new THREE.BufferGeometry();
  boneGeometryLines.setAttribute('position', new THREE.BufferAttribute(bonePositions, 3));
  boneGeometryLines.setAttribute('color', new THREE.BufferAttribute(boneColors, 3));

  const bones = new THREE.LineSegments(boneGeometryLines, boneMaterial);
  bones.name = 'bones';
  bones.frustumCulled = false;
  group.add(bones);

  // Soft glow: thin cylinders. A 1-pixel GL line is almost invisible on a
  // high-DPI phone screen, so the volumetric halo carries the visual weight.
  const boneGlow = new THREE.InstancedMesh(boneGeometry, boneGlowMaterial, BONE_COUNT);
  boneGlow.name = 'boneGlow';
  boneGlow.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  boneGlow.frustumCulled = false;
  group.add(boneGlow);

  for (let b = 0; b < BONE_COUNT; b += 1) {
    const [a, c] = BONES[b];
    const colorA = jointColors[a] || new THREE.Color(COLORS.joint.other);
    const colorC = jointColors[c] || new THREE.Color(COLORS.joint.other);
    boneGlow.setColorAt(
      b,
      new THREE.Color(
        (colorA.r + colorC.r) * 0.5,
        (colorA.g + colorC.g) * 0.5,
        (colorA.b + colorC.b) * 0.5,
      ),
    );
  }
  if (boneGlow.instanceColor) boneGlow.instanceColor.needsUpdate = true;

  /* ------------------------------------------------------------ smoothing state */

  /** Target positions from the newest frame (metres, world space). */
  const targets = new Float32Array(KEYPOINT_COUNT * 3);
  /** Positions actually rendered — always a smoothed version of `targets`. */
  const current = new Float32Array(KEYPOINT_COUNT * 3);
  /** Spring velocities, per component. */
  const velocities = new Float32Array(KEYPOINT_COUNT * 3);
  /** 1 when a joint has a usable target this frame, else 0. */
  const present = new Uint8Array(KEYPOINT_COUNT);

  let initialised = false;
  let fade = 0;
  let visible = true;
  let disposed = false;
  let detected = false;
  let lastDelta = 1 / 60;

  /**
   * Ingest keypoints. Any individual `null`, malformed triple or a completely
   * missing (`null`) `keypoints` array is tolerated: the affected joints are
   * marked missing, their spheres hidden and every bone touching them hidden.
   */
  function ingest(keypoints) {
    const source = keypoints && typeof keypoints.length === 'number' ? keypoints : null;
    let anyPresent = false;

    for (let i = 0; i < KEYPOINT_COUNT; i += 1) {
      const vec = source ? readVec3(source[i], keypointScratch) : null;
      if (vec) {
        targets[i * 3 + 0] = vec[0];
        targets[i * 3 + 1] = vec[1];
        targets[i * 3 + 2] = vec[2];
        present[i] = 1;
        anyPresent = true;
      } else {
        present[i] = 0;
      }
    }

    if (anyPresent && !initialised) {
      // First pose: adopt it directly so the skeleton never flies in from the
      // origin. This is the only frame that is not spring-interpolated.
      current.set(targets);
      velocities.fill(0);
      initialised = true;
    }

    return anyPresent;
  }

  function updateMatrices() {
    const materialOpacity = fade;
    jointMaterial.opacity = materialOpacity * AVATAR.jointCoreOpacity;
    jointHaloMaterial.opacity = materialOpacity * AVATAR.jointHaloOpacity;
    boneMaterial.opacity = materialOpacity * AVATAR.boneOpacity;
    boneGlowMaterial.opacity = materialOpacity * AVATAR.boneGlowOpacity;

    for (let i = 0; i < KEYPOINT_COUNT; i += 1) {
      const index = i * 3;
      positionScratch.set(current[index], current[index + 1], current[index + 2]);

      // Joints materialise slightly as the avatar fades in.
      const radius = present[i]
        ? jointRadii[i] * (0.35 + 0.65 * fade)
        : HIDDEN_SCALE;

      scaleScratch.set(radius, radius, radius);
      matrixScratch.compose(positionScratch, identityQuaternion, scaleScratch);
      joints.setMatrixAt(i, matrixScratch);

      const haloRadius = radius * AVATAR.jointHaloScale;
      scaleScratch.set(haloRadius, haloRadius, haloRadius);
      matrixScratch.compose(positionScratch, identityQuaternion, scaleScratch);
      jointHalos.setMatrixAt(i, matrixScratch);
    }
    joints.instanceMatrix.needsUpdate = true;
    jointHalos.instanceMatrix.needsUpdate = true;

    for (let b = 0; b < BONE_COUNT; b += 1) {
      const [a, c] = BONES[b];
      const ai = a * 3;
      const ci = c * 3;

      let drawn = present[a] === 1 && present[c] === 1;

      const ax = current[ai];
      const ay = current[ai + 1];
      const az = current[ai + 2];
      const cx = current[ci];
      const cy = current[ci + 1];
      const cz = current[ci + 2];

      let length = 0;
      if (drawn) {
        directionScratch.set(cx - ax, cy - ay, cz - az);
        length = directionScratch.length();
        if (length < AVATAR.minBoneLength) drawn = false;
      }

      if (drawn) {
        bonePositions[b * 6 + 0] = ax;
        bonePositions[b * 6 + 1] = ay;
        bonePositions[b * 6 + 2] = az;
        bonePositions[b * 6 + 3] = cx;
        bonePositions[b * 6 + 4] = cy;
        bonePositions[b * 6 + 5] = cz;

        // Orient the unit cylinder (+Y) along the bone and stretch it.
        directionScratch.multiplyScalar(1 / length);
        quaternionScratch.setFromUnitVectors(UP, directionScratch);
        midpointScratch.set((ax + cx) * 0.5, (ay + cy) * 0.5, (az + cz) * 0.5);
        scaleScratch.set(AVATAR.boneGlowRadius, length, AVATAR.boneGlowRadius);
        matrixScratch.compose(midpointScratch, quaternionScratch, scaleScratch);
      } else {
        // Degenerate zero-length segment: rasterises to nothing.
        bonePositions[b * 6 + 0] = ax;
        bonePositions[b * 6 + 1] = ay;
        bonePositions[b * 6 + 2] = az;
        bonePositions[b * 6 + 3] = ax;
        bonePositions[b * 6 + 4] = ay;
        bonePositions[b * 6 + 5] = az;
        scaleScratch.set(HIDDEN_SCALE, HIDDEN_SCALE, HIDDEN_SCALE);
        matrixScratch.compose(positionScratch.set(ax, ay, az), identityQuaternion, scaleScratch);
      }
      boneGlow.setMatrixAt(b, matrixScratch);
    }

    boneGeometryLines.attributes.position.needsUpdate = true;
    boneGlow.instanceMatrix.needsUpdate = true;
  }

  /**
   * Advance the skeleton.
   *
   * @param {Array<Array<number>|null>|null} keypoints raw COCO-17 keypoints
   * @param {number} dt seconds since the previous update
   */
  function update(keypoints, dt) {
    if (disposed) return;

    const step = safeDelta(dt, AVATAR.maxDeltaSeconds);
    lastDelta = step > 0 ? step : lastDelta;

    if (!visible) {
      group.visible = false;
      return;
    }

    const anyPresent = ingest(keypoints);
    detected = anyPresent;

    // Fade in on detection, fade out when the person leaves or the frame is
    // degraded. Never a hard snap: opacity is continuously damped.
    fade = damp(fade, anyPresent ? 1 : 0, anyPresent ? AVATAR.fadeInRate : AVATAR.fadeOutRate, lastDelta);
    if (fade < 0.0005) fade = 0;
    if (fade > 0.9995) fade = 1;

    if (fade <= 0 && !initialised) {
      group.visible = false;
      return;
    }

    group.visible = fade > AVATAR.visibleThreshold;

    // Critically-damped spring integration, per joint / per axis. Joints with
    // no target this frame hold their position and bleed off velocity, so a
    // partially-null frame dims a limb instead of yanking it to the origin.
    const positionDamping = Math.exp(-6 * lastDelta);
    for (let i = 0; i < KEYPOINT_COUNT; i += 1) {
      const index = i * 3;
      if (present[i]) {
        for (let axis = 0; axis < 3; axis += 1) {
          const component = index + axis;
          springStep(
            current[component],
            velocities[component],
            targets[component],
            AVATAR.springOmega,
            lastDelta,
            springResult,
          );
          current[component] = springResult[0];
          velocities[component] = springResult[1];
        }
      } else {
        velocities[index + 0] *= positionDamping;
        velocities[index + 1] *= positionDamping;
        velocities[index + 2] *= positionDamping;
      }
    }

    if (group.visible) updateMatrices();
  }

  /** Show/hide the whole skeleton without discarding its smoothing state. */
  function setVisible(next) {
    visible = next !== false;
    if (!visible) group.visible = false;
    else group.visible = fade > AVATAR.visibleThreshold;
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    if (group.parent) group.parent.remove(group);
    jointGeometry.dispose();
    boneGeometry.dispose();
    boneGeometryLines.dispose();
    disposeMaterial(jointMaterial);
    disposeMaterial(jointHaloMaterial);
    disposeMaterial(boneMaterial);
    disposeMaterial(boneGlowMaterial);
    joints.dispose();
    jointHalos.dispose();
    boneGlow.dispose();
  }

  // Publish an initial fully-hidden pose; the first valid frame brings the
  // skeleton in. `present` is all zero here, so every joint and bone collapses
  // to `HIDDEN_SCALE` and nothing is drawn.
  updateMatrices();

  return {
    group,
    update,
    setVisible,
    dispose,
    get opacity() {
      return fade;
    },
    get detected() {
      return detected;
    },
    get jointCount() {
      return KEYPOINT_COUNT;
    },
    get boneCount() {
      return BONE_COUNT;
    },
  };
}
