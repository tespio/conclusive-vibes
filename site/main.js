/* ============================================================================
   Checkpoint Forensics, Audited — interactive teardown
   Data source: ../results/exp3_train.json, ../results/exp2_forge.json
   ========================================================================== */
import * as THREE from './vendor/three.module.min.js';

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

document.documentElement.classList.add('js');

/* ---------------------------------------------------------------------------
   Measured data — full-vocabulary-coverage run, results/exp3_train.json
   ------------------------------------------------------------------------- */
const DECAY = [
  { step: 0,   exact: 1.000000, cos: 1.00000000, sigma: null,   detect: 256 },
  { step: 1,   exact: 0.119614, cos: 0.99996247, sigma: 369435, detect: 256 },
  { step: 2,   exact: 0.092770, cos: 0.99990760, sigma: 235469, detect: 0 },
  { step: 5,   exact: 0.043260, cos: 0.99970640, sigma: 132063, detect: 0 },
  { step: 10,  exact: 0.013080, cos: 0.99934630, sigma: 88479,  detect: 0 },
  { step: 20,  exact: 0.001090, cos: 0.99873400, sigma: 63549,  detect: 0 },
  { step: 50,  exact: 0.000000, cos: 0.99776340, sigma: 47775,  detect: 0 },
  { step: 100, exact: 0.000000, cos: 0.99688470, sigma: 40454,  detect: 0 },
  { step: 200, exact: 0.000000, cos: 0.99420230, sigma: 29594,  detect: 0 },
  { step: 400, exact: 0.000000, cos: 0.98686780, sigma: 19555,  detect: 0 },
  { step: 600, exact: 0.000000, cos: 0.97913360, sigma: 15421,  detect: 0 },
];

const SCENES = {
  hero:    { cam: [0, 4.5, 56], look: [0, 0, 0],     w: { lattice: 1 }, zc: 1.0, op: 0.72 },
  integer: { cam: [0, 0.5, 43], look: [2.5, 0, 0],   w: { device: 1 },  zc: 0.5, op: 0.88 },
  forge:   { cam: [14, 10, 47], look: [6, 0, 0],     w: null,            zc: 1.0, op: 0.92 },
  decay:   { cam: [0, 2.0, 38], look: [0, 0, 0],     w: { lattice: 1 }, zc: 1.0, op: 0.92 },
  sigma:   { cam: [0, 17, 33],  look: [0, -3.5, 0],  w: { flat: 1 },    zc: 1.0, op: 0.42 },
  cost:    { cam: [0, 9.0, 56], look: [0, 0, 0],     w: { lattice: 1 }, zc: 1.0, op: 0.30 },
  outro:   { cam: [0, 6.0, 52], look: [0, 0, 0],     w: { lattice: 1 }, zc: 1.0, op: 0.28 },
};

/* ---------------------------------------------------------------------------
   Three.js stage
   ------------------------------------------------------------------------- */
let renderer, scene, camera, field, mesh, slabEdges, glow, wallGlow, wallBack, wallFrame;
let W = { lattice: 1, slab: 0, flat: 0, ring: 0, device: 0 };
let TW = { ...W };
let zc = 1, zcTarget = 1;
let decayAmt = 0, decayTarget = 0;
let opAmt = 0.72, opTarget = 0.72;
let forgeProgress = 0;
let ok3d = true;

const N = innerWidth < 760 ? 1100 : 1950;
const COLS = innerWidth < 760 ? 22 : 30;
const ROWS = innerWidth < 760 ? 10 : 13;
const DEPTH = 5;

const POS = { lattice: new Float32Array(N * 3), slab: new Float32Array(N * 3),
              flat: new Float32Array(N * 3), ring: new Float32Array(N * 3) };
const COL = { lattice: new Float32Array(N * 3), slab: new Float32Array(N * 3),
              flat: new Float32Array(N * 3), ring: new Float32Array(N * 3) };
const JIT = new Float32Array(N * 3);
const ROT = new Float32Array(N * 3);

/* Device wall: each launch-grid cell becomes one cube.
   SMs across, blocks/SM down. Selecting a device relayouts this wall. */
const WALL_X = 0.21, WALL_Y = 0.5, WALL_SCALE = 0.44, WALL_CX = 7;
const DEVPOS = new Float32Array(N * 3), TDEVPOS = new Float32Array(N * 3);
const DEVCOL = new Float32Array(N * 3), TDEVCOL = new Float32Array(N * 3);
const DEVSCL = new Float32Array(N), TDEVSCL = new Float32Array(N);
let devSm = 148, devBps = 8;
let devPulse = 0;
const dummy = new THREE.Object3D();
const tmpColor = new THREE.Color();
const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
const clock = { t: 0 };

function buildField() {
  const geo = new THREE.BoxGeometry(0.44, 0.44, 0.44);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.82 });
  mesh = new THREE.InstancedMesh(geo, mat, N);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  field = new THREE.Group();
  field.add(mesh);

  const cA = new THREE.Color('#0a222c'), cB = new THREE.Color('#2f8ea6');
  const amber = new THREE.Color('#ffb020');
  const slabA = new THREE.Color('#ffb020'), slabB = new THREE.Color('#ff4767');
  const flatC = new THREE.Color('#2f7f96');
  const ringC = new THREE.Color('#9fe9ff');

  let i = 0;
  for (let c = 0; c < COLS; c++) {
    for (let r = 0; r < ROWS; r++) {
      for (let d = 0; d < DEPTH; d++) {
        if (i >= N) break;
        const i3 = i * 3;
        const nx = c / (COLS - 1), ny = r / (ROWS - 1), nz = d / (DEPTH - 1);

        POS.lattice[i3]     = (nx - 0.5) * 27.5;
        POS.lattice[i3 + 1] = (ny - 0.5) * 11.4;
        POS.lattice[i3 + 2] = (nz - 0.5) * 5.4 + Math.sin(c * 0.7 + r * 0.45) * 0.16;

        POS.slab[i3]     = (nx - 0.5) * 13 + 6.5;
        POS.slab[i3 + 1] = (ny - 0.5) * 10;
        POS.slab[i3 + 2] = (nz - 0.5) * 6;

        const ang = (i / N) * Math.PI * 2;
        const rad = 11.4 + Math.sin(i * 0.73) * 1.2;
        POS.ring[i3]     = Math.cos(ang) * rad;
        POS.ring[i3 + 1] = (Math.random() - 0.5) * 1.4;
        POS.ring[i3 + 2] = Math.sin(ang) * rad * 0.55;

        const fc = i % 44, fr = Math.floor(i / 44);
        POS.flat[i3]     = (fc / 43 - 0.5) * 34 + (Math.random() - 0.5) * 0.5;
        POS.flat[i3 + 1] = -4 + (Math.random() - 0.5) * 0.35;
        POS.flat[i3 + 2] = (fr / 44 - 0.5) * 18 + (Math.random() - 0.5) * 0.5;

        // colours (linear working space, via THREE.Color)
        tmpColor.copy(cA).lerp(cB, clamp(nx * 0.75 + ny * 0.25, 0, 1));
        if (i % 37 === 0) tmpColor.lerp(amber, 0.85);
        COL.lattice[i3] = tmpColor.r; COL.lattice[i3 + 1] = tmpColor.g; COL.lattice[i3 + 2] = tmpColor.b;

        tmpColor.copy(slabA).lerp(slabB, nx);
        COL.slab[i3] = tmpColor.r; COL.slab[i3 + 1] = tmpColor.g; COL.slab[i3 + 2] = tmpColor.b;

        tmpColor.copy(flatC).multiplyScalar(0.75);
        COL.flat[i3] = tmpColor.r; COL.flat[i3 + 1] = tmpColor.g; COL.flat[i3 + 2] = tmpColor.b;

        tmpColor.copy(ringC).multiplyScalar(0.62);
        COL.ring[i3] = tmpColor.r; COL.ring[i3 + 1] = tmpColor.g; COL.ring[i3 + 2] = tmpColor.b;

        // decay direction
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const mag = 5 + Math.random() * 9;
        JIT[i3]     = Math.sin(ph) * Math.cos(th) * mag;
        JIT[i3 + 1] = Math.cos(ph) * mag * 0.7;
        JIT[i3 + 2] = Math.sin(ph) * Math.sin(th) * mag;

        ROT[i3]     = (Math.random() - 0.5) * 0.5;
        ROT[i3 + 1] = (Math.random() - 0.5) * 0.5;
        ROT[i3 + 2] = (Math.random() - 0.5) * 0.5;
        i++;
      }
    }
  }
}

function initThree() {
  const host = $('#webgl');
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  } catch (err) {
    ok3d = false;
    host.classList.add('no-webgl');
    return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x05070a, 1);
  host.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05070a, 0.0135);
  camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 400);

  buildField();
  scene.add(field);

  // soft ambient glow behind the field
  const gc = document.createElement('canvas');
  gc.width = gc.height = 256;
  const gx = gc.getContext('2d');
  const grad = gx.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, 'rgba(40,130,155,.55)');
  grad.addColorStop(0.45, 'rgba(18,70,90,.22)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  gx.fillStyle = grad;
  gx.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(gc);
  tex.colorSpace = THREE.SRGBColorSpace;
  glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0.9,
  }));
  glow.scale.set(105, 105, 1);
  glow.position.set(0, 0, -18);
  scene.add(glow);

  // soft halo behind the device wall
  wallGlow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, color: 0x2f8ea6, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0,
  }));
  wallGlow.scale.set(62, 20, 1);
  wallGlow.position.set(WALL_CX, 0, -9);
  scene.add(wallGlow);

  // backing panel + frame make the wall read as an object that resizes
  wallBack = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: 0x02090e, transparent: true, opacity: 0, depthWrite: false })
  );
  wallBack.position.set(WALL_CX, 0, -0.6);
  scene.add(wallBack);
  wallFrame = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1)),
    new THREE.LineBasicMaterial({ color: 0x5fe3ff, transparent: true, opacity: 0 })
  );
  wallFrame.position.set(WALL_CX, 0, 0.04);
  scene.add(wallFrame);

  // floor grid
  const grid = new THREE.GridHelper(150, 75, 0x0e2c36, 0x081820);
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  grid.position.y = -8.6;
  scene.add(grid);

  // checkpoint slab outline (visible during the forgery)
  slabEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(13.6, 10.6, 6.8)),
    new THREE.LineBasicMaterial({ color: 0xffb020, transparent: true, opacity: 0 })
  );
  slabEdges.position.set(6.5, 0, 0);
  scene.add(slabEdges);

  camera.position.set(...SCENES.hero.cam);
  setDevice(148, 8, true);
}

/* ---------------------------------------------------------------------------
   Camera + field animation
   ------------------------------------------------------------------------- */
const camPos = new THREE.Vector3(0, 3.5, 40);
const camLook = new THREE.Vector3(0, 0, 0);
let camPosT = camPos.clone();
let camLookT = camLook.clone();

function setDevice(sm, bps, instant = false) {
  devSm = sm;
  devBps = bps;
  const total = Math.min(N, sm * bps);
  if (wallBack) {
    const w = (sm - 1) * WALL_X + 0.6;
    const h = (bps - 1) * WALL_Y + 0.6;
    wallBack.scale.set(w, h, 1);
    wallFrame.scale.set(w, h, 1);
  }
  const cA = new THREE.Color('#3aa3bd'), cB = new THREE.Color('#c8fbff');
  const amber = new THREE.Color('#ffb020');
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    if (i < total) {
      const c = i % sm, r = Math.floor(i / sm);
      TDEVPOS[i3]     = WALL_CX + (c - (sm - 1) / 2) * WALL_X;
      TDEVPOS[i3 + 1] = ((bps - 1) / 2 - r) * WALL_Y;
      TDEVPOS[i3 + 2] = 0;
      tmpColor.copy(cA).lerp(cB, sm > 1 ? c / (sm - 1) : 0);
      if (i % 23 === 0) tmpColor.lerp(amber, 0.9);
      TDEVCOL[i3] = tmpColor.r; TDEVCOL[i3 + 1] = tmpColor.g; TDEVCOL[i3 + 2] = tmpColor.b;
      TDEVSCL[i] = WALL_SCALE;
    } else {
      TDEVPOS[i3] = POS.lattice[i3];
      TDEVPOS[i3 + 1] = POS.lattice[i3 + 1];
      TDEVPOS[i3 + 2] = POS.lattice[i3 + 2];
      TDEVCOL[i3] = 0.012; TDEVCOL[i3 + 1] = 0.016; TDEVCOL[i3 + 2] = 0.022;
      TDEVSCL[i] = 0;
    }
  }
  if (instant) {
    DEVPOS.set(TDEVPOS);
    DEVCOL.set(TDEVCOL);
    DEVSCL.set(TDEVSCL);
  }
}

function updateField(dt) {
  const k = reduceMotion ? 1 : 1 - Math.exp(-6 * dt);
  for (const key in W) W[key] += (TW[key] - W[key]) * k;
  zc += (zcTarget - zc) * k;
  decayAmt += (decayTarget - decayAmt) * k;
  opAmt += (opTarget - opAmt) * k;
  mesh.material.opacity = opAmt;
  devPulse = Math.max(0, devPulse - dt * 2.2);

  const kd = reduceMotion ? 1 : 1 - Math.exp(-9 * dt);
  for (let j = 0; j < N * 3; j++) {
    DEVPOS[j] += (TDEVPOS[j] - DEVPOS[j]) * kd;
    DEVCOL[j] += (TDEVCOL[j] - DEVCOL[j]) * kd;
  }
  for (let j = 0; j < N; j++) DEVSCL[j] += (TDEVSCL[j] - DEVSCL[j]) * kd;

  const wl = W.lattice, ws = W.slab, wf = W.flat, wr = W.ring, wd = W.device;
  const sum = Math.max(0.0001, wl + ws + wf + wr + wd);
  const inv = 1 / sum;
  const greyR = 0.045, greyG = 0.055, greyB = 0.062;

  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    let px = (POS.lattice[i3] * wl + POS.slab[i3] * ws + POS.flat[i3] * wf + POS.ring[i3] * wr + DEVPOS[i3] * wd) * inv;
    let py = (POS.lattice[i3 + 1] * wl + POS.slab[i3 + 1] * ws + POS.flat[i3 + 1] * wf + POS.ring[i3 + 1] * wr + DEVPOS[i3 + 1] * wd) * inv;
    let pz = (POS.lattice[i3 + 2] * wl + POS.slab[i3 + 2] * ws + POS.flat[i3 + 2] * wf + POS.ring[i3 + 2] * wr + DEVPOS[i3 + 2] * wd) * inv;
    pz *= zc;

    px += JIT[i3] * decayAmt;
    py += JIT[i3 + 1] * decayAmt;
    pz += JIT[i3 + 2] * decayAmt;

    const sc = (wl + ws + wf + wr + DEVSCL[i] * wd) * inv;
    dummy.position.set(px, py, pz);
    dummy.rotation.set(ROT[i3], ROT[i3 + 1], ROT[i3 + 2]);
    dummy.scale.setScalar(sc);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);

    let cr = (COL.lattice[i3] * wl + COL.slab[i3] * ws + COL.flat[i3] * wf + COL.ring[i3] * wr + DEVCOL[i3] * wd) * inv;
    let cg = (COL.lattice[i3 + 1] * wl + COL.slab[i3 + 1] * ws + COL.flat[i3 + 1] * wf + COL.ring[i3 + 1] * wr + DEVCOL[i3 + 1] * wd) * inv;
    let cb = (COL.lattice[i3 + 2] * wl + COL.slab[i3 + 2] * ws + COL.flat[i3 + 2] * wf + COL.ring[i3 + 2] * wr + DEVCOL[i3 + 2] * wd) * inv;
    const dk = decayAmt * 0.9;
    cr = lerp(cr, greyR, dk); cg = lerp(cg, greyG, dk); cb = lerp(cb, greyB, dk);
    if (devPulse > 0.002 && wd > 0.02) {
      const boost = 1 + devPulse * 0.9 * wd;
      cr = Math.min(1.5, cr * boost);
      cg = Math.min(1.5, cg * boost);
      cb = Math.min(1.5, cb * boost);
    }
    mesh.instanceColor.setXYZ(i, cr, cg, cb);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  slabEdges.material.opacity = W.slab * 0.55;
  wallGlow.material.opacity = W.device * (0.26 + devPulse * 0.28);
  wallBack.material.opacity = W.device * 0.55;
  wallFrame.material.opacity = W.device * 0.5;

  if (!reduceMotion) {
    field.rotation.y = Math.sin(clock.t * 0.13) * 0.11;
    field.rotation.x = Math.sin(clock.t * 0.09) * 0.03;
  }
}

function updateCamera(dt) {
  const k = reduceMotion ? 1 : 1 - Math.exp(-4.5 * dt);
  camPos.lerp(camPosT, k);
  camLook.lerp(camLookT, k);
  mouse.x = lerp(mouse.x, mouse.tx, 0.05);
  mouse.y = lerp(mouse.y, mouse.ty, 0.05);
  camera.position.set(camPos.x + mouse.x * 1.4, camPos.y + mouse.y * -1.0, camPos.z);
  camera.lookAt(camLook);
}

function tick(now) {
  const dt = Math.min(0.05, (now - tick.last || 16) / 1000);
  tick.last = now;
  clock.t += dt;
  if (ok3d) {
    updateForge(dt);
    updateField(dt);
    updateCamera(dt);
    renderer.render(scene, camera);
  }
  requestAnimationFrame(tick);
}
tick.last = 0;

/* ---------------------------------------------------------------------------
   Navigation
   ------------------------------------------------------------------------- */
const sections = $$('.section');
const rail = $('#rail');
const idxNum = $('#idxNum');
let active = 0;

sections.forEach((s, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.title = s.dataset.label || `Section ${i + 1}`;
  b.setAttribute('aria-label', s.dataset.label || `Section ${i + 1}`);
  b.addEventListener('click', () => s.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' }));
  rail.appendChild(b);
});
const railDots = $$('button', rail);

function setActive(i) {
  if (i === active) return;
  const prev = sections[active];
  active = i;
  sections.forEach((s, j) => s.classList.toggle('is-active', j === i));
  railDots.forEach((d, j) => d.classList.toggle('is-active', j === i));
  idxNum.textContent = i === 0 ? '00' : String(i).padStart(2, '0');
  onEnter(sections[i].dataset.scene, prev.dataset.scene);
}

function updateActive() {
  const mid = innerHeight * 0.5;
  let best = 0, bestD = Infinity;
  sections.forEach((s, i) => {
    const r = s.getBoundingClientRect();
    const d = Math.abs(r.top + r.height / 2 - mid);
    if (d < bestD) { bestD = d; best = i; }
  });
  setActive(best);
}

let scrollTick = false;
addEventListener('scroll', () => {
  if (scrollTick) return;
  scrollTick = true;
  requestAnimationFrame(() => { scrollTick = false; updateActive(); });
}, { passive: true });

addEventListener('resize', () => {
  if (ok3d) {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    field.scale.setScalar(clamp((innerWidth / innerHeight) / 1.35, 0.52, 1));
  }
  chartResize();
  updateActive();
});

addEventListener('pointermove', e => {
  mouse.tx = (e.clientX / innerWidth) * 2 - 1;
  mouse.ty = (e.clientY / innerHeight) * 2 - 1;
}, { passive: true });

/* ---------------------------------------------------------------------------
   HUD
   ------------------------------------------------------------------------- */
const hudCounter = $('#hudCounter');
const counterDigits = $('#counterDigits');
let flashTimer = null;
function bumpCounter() {
  counterDigits.textContent = '0000';
  hudCounter.classList.remove('flash');
  void hudCounter.offsetWidth;
  hudCounter.classList.add('flash');
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => hudCounter.classList.remove('flash'), 520);
}

/* ---------------------------------------------------------------------------
   Section entry hooks
   ------------------------------------------------------------------------- */
let barcodePlayed = false;
function onEnter(sceneName, prevName) {
  const cfg = SCENES[sceneName];
  if (cfg) {
    camPosT.set(...cfg.cam);
    camLookT.set(...cfg.look);
    zcTarget = cfg.zc;
    opTarget = cfg.op;
    if (cfg.w) TW = { lattice: 0, slab: 0, flat: 0, ring: 0, device: 0, ...cfg.w };
  }
  if (sceneName === 'forge') {
    TW = { lattice: 1 - forgeProgress, slab: forgeProgress, flat: 0, ring: 0, device: 0 };
  } else if (prevName === 'forge') {
    resetForge(false);
  }
  if (sceneName !== 'decay') decayTarget = 0;
  if (sceneName === 'decay') decayTarget = 1 - DECAY[+$('#stepSlider').value].exact;
  if (sceneName === 'sigma') { updateSigma(); requestAnimationFrame(chartResize); }
  if (sceneName === 'mech' && !barcodePlayed) { barcodePlayed = true; animateBarcode(devSm, devBps); }
}

/* ---------------------------------------------------------------------------
   Barcode (148 × 8 = 1,184)
   ------------------------------------------------------------------------- */
const barcode = $('#barcode');
function drawBarcode(progress, cols = 148, rows = 8) {
  const ctx = barcode.getContext('2d');
  const Wc = barcode.width, Hc = barcode.height;
  ctx.clearRect(0, 0, Wc, Hc);
  const cw = Wc / cols, ch = Hc / rows;
  const total = cols * rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const lit = idx / total < progress;
      const x = c * cw + cw * 0.15;
      const y = r * ch + ch * 0.2;
      const w = cw * 0.7, h = ch * 0.58;
      if (lit) {
        const heat = 0.55 + 0.45 * Math.min(1, (progress - idx / total) * 6);
        ctx.fillStyle = `rgba(255,176,32,${0.35 + heat * 0.65})`;
      } else {
        ctx.fillStyle = 'rgba(255,176,32,0.10)';
      }
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, w, h, 1.2); else ctx.rect(x, y, w, h);
      ctx.fill();
    }
  }
}

function animateBarcode(cols = 148, rows = 8) {
  if (reduceMotion) { drawBarcode(1, cols, rows); return; }
  const t0 = performance.now();
  function step(now) {
    const p = clamp((now - t0) / 1300, 0, 1);
    drawBarcode(easeInOut(p), cols, rows);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ---------------------------------------------------------------------------
   Device table — each selection rebuilds the 3D launch-grid wall
   ------------------------------------------------------------------------- */
function selectDevice(btn, animate = true) {
  $$('#devices .dev').forEach(b => {
    const on = b === btn;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  const sm = +btn.dataset.sm, bps = +btn.dataset.bps;
  const grid = sm * bps;
  const name = btn.dataset.name;
  $('#gridNum').textContent = grid.toLocaleString('en-US');
  $('#gridEq').textContent = `= ${sm} × ${bps}`;
  $('#gridCap').textContent = `${name} · ${btn.dataset.note}`;
  $('#bcSms').textContent = `${sm} SMs`;
  $('#bcCells').textContent = `× ${bps} blocks/SM = ${grid.toLocaleString('en-US')} cells`;
  setDevice(sm, bps);
  devPulse = 1;
  if (animate) animateBarcode(sm, bps);
  bumpCounter();
}

$$('#devices .dev').forEach(btn => {
  btn.addEventListener('click', () => selectDevice(btn, true));
});

/* ---------------------------------------------------------------------------
   Forge sequence
   ------------------------------------------------------------------------- */
const forgeBtn = $('#forgeBtn');
const resetBtn = $('#resetBtn');
const stamp = $('#stamp');
const termEl = $('#term');
const termPre = $('#termPre');
const receiptVerdict = $('#receiptVerdict');
const truthToggle = $('#truthToggle');
const truthEl = $('#truth');

let forging = false, forgeT = 0, typeToken = 0;
const TERM_LINES = [
  { t: '> forge --profile gb200 --seed 424242', c: 'cmd' },
  { t: '  generating 10,244,352 parameters… no CUDA device', c: 'dim' },
  { t: '  done in 3.03 s · 0 GPUs', c: 'ok' },
  { t: '> verify --checkpoint forged_gb200_step0.npz', c: 'cmd' },
  { t: '  RTX 4070 Ti SUPER    1/256', c: 'bad' },
  { t: '  H100 SXM             0/256', c: 'bad' },
  { t: '  GB200 (per GPU)    256/256   MATCH', c: 'ok' },
  { t: '  B300/GB300           0/256', c: 'bad' },
  { t: '> verdict: GB200 · bit-for-bit · conclusive', c: 'ok' },
];

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function typeTerminal() {
  const token = ++typeToken;
  termPre.innerHTML = '';
  for (let i = 0; i < TERM_LINES.length; i++) {
    if (token !== typeToken) return;
    const line = TERM_LINES[i];
    const span = document.createElement('span');
    span.className = line.c;
    span.textContent = line.t + '\n';
    termPre.appendChild(span);
    termEl.scrollTop = termEl.scrollHeight;
    if (i === TERM_LINES.length - 1) receiptVerdict.textContent = 'GB200 · 256/256';
    await sleep(i === 0 ? 220 : 240);
  }
}

function startForge() {
  if (forging) return;
  forging = true;
  forgeT = 0;
  stamp.classList.remove('show');
  receiptVerdict.textContent = '…';
  forgeBtn.disabled = true;
  forgeBtn.textContent = 'generating…';
  resetBtn.disabled = false;
  truthEl.classList.remove('show');
  truthToggle.checked = false;
  typeTerminal();
  bumpCounter();
}

function updateForge(dt) {
  if (!forging) return;
  forgeT += dt / 1.85;
  if (forgeT >= 1) { forgeT = 1; forging = false; finishForge(); }
  forgeProgress = easeInOut(forgeT);
  TW.lattice = 1 - forgeProgress;
  TW.slab = forgeProgress;
}

function finishForge() {
  stamp.classList.add('show');
  const panel = $('#s-forge .panel');
  panel.classList.remove('shake');
  void panel.offsetWidth;
  panel.classList.add('shake');
  forgeBtn.disabled = false;
  forgeBtn.textContent = 'forge again';
  bumpCounter();
}

function resetForge(wipe = true) {
  forging = false;
  forgeT = 0;
  forgeProgress = 0;
  stamp.classList.remove('show');
  forgeBtn.disabled = false;
  forgeBtn.textContent = 'Forge a GB200 checkpoint';
  resetBtn.disabled = true;
  receiptVerdict.textContent = '—';
  typeToken++;
  if (wipe) {
    termPre.innerHTML = '<span class="dim">$ waiting for you to commit a crime against epistemology…</span>';
    truthEl.classList.remove('show');
    truthToggle.checked = false;
  }
}

forgeBtn.addEventListener('click', startForge);
resetBtn.addEventListener('click', () => {
  resetForge(true);
  TW = { lattice: 1, slab: 0, flat: 0, ring: 0, device: 0 };
});
truthToggle.addEventListener('change', () => {
  truthEl.classList.toggle('show', truthToggle.checked);
  bumpCounter();
});

/* ---------------------------------------------------------------------------
   Decay slider
   ------------------------------------------------------------------------- */
const stepSlider = $('#stepSlider');
const stepOut = $('#stepOut');
const rExact = $('#rExact');
const rCos = $('#rCos');
const rSigma = $('#rSigma');
const rDetect = $('#rDetect');
const decayNote = $('#decayNote');

const NOTES = {
  0: 'Step 0 is the only moment his “bit-for-bit” claim is even well-defined. It is also the one checkpoint no lab ever publishes.',
  1: 'One step. 88% of parameters are no longer bit-identical — but the probe still hits embedding rows this batch never touched.',
  2: 'Step 2. The probe is dead: 0/256. The seed is unrecoverable. Everything after this is a model that no longer contains its own initialization.',
  6: 'Fifty steps. Not one parameter remains bit-exact, while the cosine similarity is still 0.998 — “correlated with init” is not “a fingerprint”.',
  10: 'Six hundred steps. Cosine 0.979, bit-exact 0.000%. His method has no signal left except parameters that were never trained at all.',
};

const sup = s => String(s).replace(/-/g, '⁻').replace(/\./g, '·')
  .replace(/0/g, '⁰').replace(/1/g, '¹').replace(/2/g, '²').replace(/3/g, '³')
  .replace(/4/g, '⁴').replace(/5/g, '⁵').replace(/6/g, '⁶').replace(/7/g, '⁷')
  .replace(/8/g, '⁸').replace(/9/g, '⁹');

function fmtSigma(z) {
  if (z === null || !isFinite(z)) return '∞';
  if (z >= 1e6) {
    let e = Math.floor(Math.log10(z));
    let m = z / Math.pow(10, e);
    if (m >= 9.95) { m /= 10; e += 1; }
    return `${m.toFixed(1)}×10${sup(e)}σ`;
  }
  return Math.round(z).toLocaleString('en-US') + 'σ';
}

function applyStep(i) {
  const d = DECAY[i];
  stepSlider.value = String(i);
  stepOut.textContent = String(d.step);
  rExact.textContent = (d.exact * 100).toFixed(3) + '%';
  rCos.textContent = d.cos.toFixed(6);
  rSigma.textContent = fmtSigma(d.sigma);
  rDetect.textContent = `${d.detect}/256`;
  rDetect.className = d.detect === 256 ? 'ok' : 'bad';
  decayTarget = 1 - d.exact;
  decayNote.textContent = NOTES[i] || 'Training continues to erase the initialization. Only never-trained parameters keep it.';
  bumpCounter();
}

stepSlider.addEventListener('input', () => applyStep(+stepSlider.value));
$$('#stepPresets button').forEach(b => {
  b.addEventListener('click', () => applyStep(+b.dataset.step));
});

/* ---------------------------------------------------------------------------
   Sigma calculator + chart
   ------------------------------------------------------------------------- */
const rSlider = $('#rSlider');
const nSlider = $('#nSlider');
const rOut = $('#rOut');
const nOut = $('#nOut');
const zOut = $('#zOut');
const zNote = $('#zNote');
const chart = $('#sigmaChart');

let rVal = 0.999995, nLog = 9;

function fmtN(N) {
  if (N < 1000) return String(Math.round(N));
  const e = Math.log10(N);
  const er = Math.round(e);
  if (Math.abs(e - er) < 0.08) return '10' + sup(er);
  const m = N / Math.pow(10, Math.floor(e));
  return `${m.toFixed(1)}×10${sup(Math.floor(e))}`;
}

function updateSigma() {
  const Nval = Math.pow(10, nLog);
  const z = rVal * Math.sqrt(Nval) / Math.sqrt(Math.max(1e-12, 1 - rVal * rVal));
  rOut.textContent = rVal.toFixed(6);
  nOut.textContent = fmtN(Nval);
  zOut.textContent = fmtSigma(z);
  zNote.textContent = `r = ${rVal.toFixed(6)} · N = ${fmtN(Nval)} → ${fmtSigma(z)}`;
  drawChart(nLog, Math.log10(z));
}

rSlider.addEventListener('input', () => {
  rVal = +rSlider.value / 1e6;
  updateSigma();
});
nSlider.addEventListener('input', () => {
  nLog = +nSlider.value;
  updateSigma();
});

$('#presetHis').addEventListener('click', () => {
  rVal = 0.999995; nLog = 9;
  rSlider.value = '999995'; nSlider.value = '9';
  updateSigma();
  zNote.textContent = 'his 0.999995 at 10⁹ weights — ~10⁷σ, not 1,385σ';
  bumpCounter();
});
$('#presetForce').addEventListener('click', () => {
  rVal = 0.999995;
  nLog = Math.log10(1385 * 1385 * (1 - rVal * rVal) / (rVal * rVal));
  rSlider.value = '999995'; nSlider.value = String(nLog);
  updateSigma();
  zNote.textContent = 'to obtain his 1,385σ at that correlation you need N ≈ 19';
  bumpCounter();
});

function chartResize() {
  if (!chart || chart.offsetParent === null) return;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = chart.clientWidth || 600;
  const h = 240;
  chart.width = Math.round(w * dpr);
  chart.height = Math.round(h * dpr);
  chart.style.height = h + 'px';
  chart._ctx = chart.getContext('2d');
  chart._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  chart._w = w; chart._h = h;
  drawChart(nLog, Math.log10(Math.max(1e-9, rVal * Math.sqrt(Math.pow(10, nLog)) / Math.sqrt(1 - rVal * rVal))));
}

function drawChart(curLogN, curLogZ) {
  if (!chart || !chart._ctx) return;
  const ctx = chart._ctx, w = chart._w, h = chart._h;
  const L = 44, R = 14, T = 20, B = 28;
  const pw = w - L - R, ph = h - T - B;
  const x = v => L + ((v - 1) / 12) * pw;
  const y = v => T + ph - (v / 8) * ph;

  ctx.clearRect(0, 0, w, h);
  ctx.font = '9px ui-monospace, Consolas, monospace';
  ctx.lineWidth = 1;

  // grid + ticks
  ctx.strokeStyle = 'rgba(110,180,200,.13)';
  ctx.fillStyle = 'rgba(143,163,173,.8)';
  for (let e = 1; e <= 13; e++) {
    ctx.beginPath(); ctx.moveTo(x(e), T); ctx.lineTo(x(e), T + ph); ctx.stroke();
    if (e === 1 || (e - 1) % 3 === 0) {
      ctx.textAlign = e === 13 ? 'right' : (e === 1 ? 'left' : 'center');
      ctx.fillText('10' + sup(e), x(e) + (e === 1 ? -2 : 0), T + ph + 15);
    }
  }
  for (let e = 0; e <= 8; e += 2) {
    ctx.beginPath(); ctx.moveTo(L, y(e)); ctx.lineTo(L + pw, y(e)); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText('10' + sup(e), L - 8, y(e) + 3);
  }

  // his claimed sigma band
  ctx.fillStyle = 'rgba(255,176,32,.08)';
  ctx.fillRect(L, y(3.141), pw, y(1.690) - y(3.141));
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = 'rgba(255,176,32,.5)';
  ctx.beginPath(); ctx.moveTo(L, y(3.141)); ctx.lineTo(L + pw, y(3.141)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,176,32,.85)';
  ctx.textAlign = 'right';
  ctx.fillText('1,385σ', L + pw - 2, y(3.141) - 5);

  // correlation r = 0.999995:  log z = log10(316.227) + logN / 2
  ctx.strokeStyle = 'rgba(95,227,255,.9)';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let v = 1; v <= 11; v += 0.05) {
    const zz = 2.5 + v / 2;
    if (zz > 8) break;
    const px = x(v), py = y(zz);
    if (v === 1) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // two incompatible claims
  const dot = (vx, vy, label, dx, dy) => {
    ctx.fillStyle = 'rgba(255,71,103,.95)';
    ctx.beginPath(); ctx.arc(x(vx), y(vy), 3.2, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,125,145,.95)';
    ctx.textAlign = 'left';
    ctx.fillText(label, x(vx) + dx, y(vy) + dy);
  };
  dot(1.283, 3.141, 'N≈19', 7, 14);
  dot(9, 7, 'N=10⁹', 7, -8);

  // current state
  if (isFinite(curLogZ) && curLogZ >= 0 && curLogZ <= 8) {
    ctx.strokeStyle = 'rgba(255,255,255,.8)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x(curLogN), y(curLogZ), 5.5, 0, Math.PI * 2); ctx.stroke();
  }
}

/* ---------------------------------------------------------------------------
   Document / code / figure viewer
   ------------------------------------------------------------------------- */
const modal = $('#modal');
const modalBody = $('#modalBody');
const modalTitle = $('#modalTitle');
const modalKind = $('#modalKind');
const modalDownload = $('#modalDownload');
const modalCopy = $('#modalCopy');
const modalCloseBtn = $('.modal__close', modal);
let viewerToken = 0;
let lastFocus = null;
let codeText = '';
let codeToken = 0;
let codeListEl = null;
let codeScrollEl = null;
let currentCodePath = 'poc/run_all.py';

/* Every .py under poc/ — the viewer fetches these from the project root. */
const CODE_FILES = [
  { group: 'Pipeline', path: 'poc/run_all.py' },
  { group: 'Core', path: 'poc/philox.py' },
  { group: 'Core', path: 'poc/model.py' },
  { group: 'Core', path: 'poc/detector.py' },
  { group: 'Experiments', path: 'poc/exp1_validate.py' },
  { group: 'Experiments', path: 'poc/exp2_forge.py' },
  { group: 'Experiments', path: 'poc/exp3_train.py' },
  { group: 'Experiments', path: 'poc/exp4_stats.py' },
  { group: 'Experiments', path: 'poc/make_figures.py' },
  { group: 'Probes', path: 'poc/probes/probe_rng.py' },
  { group: 'Probes', path: 'poc/probes/probe_rng2.py' },
  { group: 'Probes', path: 'poc/probes/probe_linear.py' },
  { group: 'Probes', path: 'poc/probes/probe_offsets.py' },
];

const escapeHtml = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const PY_TOKENS = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(import|from|def|class|return|if|elif|else|for|while|in|not|and|or|as|with|try|except|finally|raise|True|False|None|pass|lambda|yield|global|del|is|break|continue)\b|\b(\d+(?:\.\d+)?)\b/g;

function highlightLine(line) {
  return escapeHtml(line).replace(PY_TOKENS, (m, com, str, kw, num) => {
    if (com) return `<span class="tok-com">${com}</span>`;
    if (str) return `<span class="tok-str">${str}</span>`;
    if (kw) return `<span class="tok-kw">${kw}</span>`;
    if (num) return `<span class="tok-num">${num}</span>`;
    return m;
  });
}

function renderCode(text) {
  const lines = text.replace(/\r\n?/g, '\n').replace(/\n$/, '').split('\n');
  let triple = null;
  const rows = lines.map((line, i) => {
    let html;
    if (triple) {
      const end = line.indexOf(triple);
      if (end === -1) {
        html = `<span class="tok-str">${escapeHtml(line)}</span>`;
      } else {
        const close = end + 3;
        html = `<span class="tok-str">${escapeHtml(line.slice(0, close))}</span>` + highlightLine(line.slice(close));
        triple = null;
      }
    } else {
      const m = /"""|'''/.exec(line);
      if (m && line.indexOf(m[0], m.index + 3) === -1) {
        html = highlightLine(line.slice(0, m.index)) + `<span class="tok-str">${escapeHtml(line.slice(m.index))}</span>`;
        triple = m[0];
      } else {
        html = highlightLine(line);
      }
    }
    return `<div class="code__line"><span class="code__ln">${i + 1}</span><span class="code__src">${line ? html : '&nbsp;'}</span></div>`;
  });
  return `<div class="code">${rows.join('')}</div>`;
}

async function fetchText(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

let markedPromise = null;
function getMarked() {
  if (!markedPromise) markedPromise = import('./vendor/marked.esm.js').then(m => m.marked);
  return markedPromise;
}

function fixReportLinks(root) {
  root.querySelectorAll('img[src],a[href]').forEach(el => {
    const attr = el.tagName === 'IMG' ? 'src' : 'href';
    const v = el.getAttribute(attr);
    if (!v || /^(https?:|mailto:|#|\/|data:|\.\.\/)/i.test(v)) return;
    el.setAttribute(attr, '../' + v);
    if (el.tagName === 'A') { el.target = '_blank'; el.rel = 'noopener'; }
  });
  root.querySelectorAll('img').forEach(img => { img.loading = 'lazy'; img.decoding = 'async'; });
}

/* --- code browser --------------------------------------------------------- */
function normalizeCodePath(src) {
  const p = (src || '').replace(/^\.\//, '').replace(/^\.\.\//, '').replace(/^\//, '');
  return CODE_FILES.some(f => f.path === p) ? p : 'poc/run_all.py';
}

function buildCodeList(activePath) {
  codeListEl.innerHTML = `<div class="codeview__head">${CODE_FILES.length} files</div>`;
  let group = '';
  for (const f of CODE_FILES) {
    if (f.group !== group) {
      group = f.group;
      const h = document.createElement('div');
      h.className = 'codeview__group';
      h.textContent = group;
      codeListEl.appendChild(h);
    }
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'codeview__file' + (f.path === activePath ? ' is-active' : '');
    b.dataset.path = f.path;
    b.textContent = f.path.replace(/^poc\//, '');
    b.setAttribute('aria-current', f.path === activePath ? 'true' : 'false');
    b.addEventListener('click', () => loadCodeFile(f.path));
    codeListEl.appendChild(b);
  }
}

async function loadCodeFile(path) {
  const token = ++codeToken;
  currentCodePath = path;
  [...codeListEl.querySelectorAll('.codeview__file')].forEach(b => {
    const on = b.dataset.path === path;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-current', on ? 'true' : 'false');
  });
  modalTitle.textContent = `${path} · loading…`;
  modalDownload.href = '../' + path;
  modalDownload.download = path.split('/').pop();
  codeScrollEl.innerHTML = '<div class="modal__loading">loading…</div>';
  try {
    const text = await fetchText('../' + path);
    if (token !== codeToken) return;
    codeText = text;
    const lines = text.replace(/\n$/, '').split('\n').length;
    modalTitle.textContent = `${path} · ${lines} lines`;
    codeScrollEl.innerHTML = renderCode(text);
    codeScrollEl.scrollTop = 0;
  } catch (err) {
    if (token !== codeToken) return;
    codeScrollEl.innerHTML = `<div class="modal__error">Could not load <code>${escapeHtml(path)}</code> (${escapeHtml(err.message)}).</div>`;
  }
}

function buildCodeViewer(src) {
  const initial = normalizeCodePath(src);
  modalBody.innerHTML = '<div class="codeview"><aside class="codeview__list" id="codeList" aria-label="Python source files"></aside><div class="codeview__scroll" id="codeScroll"></div></div>';
  codeListEl = $('#codeList', modalBody);
  codeScrollEl = $('#codeScroll', modalBody);
  buildCodeList(initial);
  loadCodeFile(initial);
}

async function openViewer(link) {
  const type = link.dataset.view;
  const src = link.getAttribute('href');
  const title = link.dataset.title || src.split('/').pop();
  lastFocus = document.activeElement;
  const token = ++viewerToken;

  modal.className = `modal is-open modal--${type}`;
  modal.setAttribute('aria-hidden', 'false');
  modalTitle.textContent = title;
  modalKind.textContent = type === 'report' ? 'Markdown' : type === 'code' ? 'Python' : 'Figure';
  modalDownload.href = src;
  modalDownload.download = src.split('/').pop();
  modalDownload.hidden = false;
  modalCopy.hidden = type !== 'code';
  modalCopy.textContent = 'copy';
  modalBody.scrollTop = 0;
  modalBody.innerHTML = '<div class="modal__loading">loading…</div>';
  document.documentElement.classList.add('modal-open');
  modalCloseBtn.focus();

  try {
    if (type === 'report') {
      const md = await fetchText(src);
      const marked = await getMarked();
      const html = marked.parse(md, { gfm: true, breaks: false });
      if (token !== viewerToken) return;
      modalBody.innerHTML = `<article class="md">${html}</article>`;
      fixReportLinks(modalBody);
    } else if (type === 'code') {
      buildCodeViewer(src);
    } else {
      if (token !== viewerToken) return;
      modalBody.innerHTML = `<figure class="figure"><img src="${src}" alt="Audit figure: four-panel summary of the checkpoint fingerprint audit" title="Click to zoom"><figcaption>figures/fig1_audit.png — click the image to zoom</figcaption></figure>`;
      const fig = modalBody.querySelector('.figure');
      fig.addEventListener('click', () => fig.classList.toggle('zoomed'));
    }
  } catch (err) {
    if (token !== viewerToken) return;
    modalBody.innerHTML = `<div class="modal__error">Could not load <code>${escapeHtml(src)}</code> (${escapeHtml(err.message)}).<br>If you are serving only the <code>site/</code> folder, serve the project root instead so the report, source and figure are reachable. <a href="${src}" target="_blank" rel="noopener">Open the file directly ↗</a></div>`;
  }
}

function closeViewer() {
  if (!modal.classList.contains('is-open')) return;
  viewerToken++;
  codeToken++;
  modal.classList.remove('is-open');
  modal.setAttribute('aria-hidden', 'true');
  document.documentElement.classList.remove('modal-open');
  if (lastFocus && lastFocus.focus) lastFocus.focus();
  setTimeout(() => {
    if (!modal.classList.contains('is-open')) modalBody.innerHTML = '';
  }, 280);
}

modal.addEventListener('click', e => {
  if (e.target.closest('[data-close]')) closeViewer();
});

document.addEventListener('keydown', e => {
  if (!modal.classList.contains('is-open')) return;
  if (e.key === 'Escape') { closeViewer(); return; }
  if (e.key !== 'Tab') return;
  const focusables = [...modal.querySelectorAll('button:not([hidden]),a[href]:not([hidden])')]
    .filter(el => el.offsetParent !== null);
  if (!focusables.length) return;
  const first = focusables[0], last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
});

modalCopy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(codeText);
    modalCopy.textContent = 'copied';
  } catch {
    modalCopy.textContent = 'failed';
  }
  setTimeout(() => { modalCopy.textContent = 'copy'; }, 1200);
});

$$('[data-view]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    openViewer(link);
  });
});

/* ---------------------------------------------------------------------------
   Back to top
   ------------------------------------------------------------------------- */
const toTop = $('#toTop');
const syncToTop = () => toTop.classList.toggle('show', scrollY > innerHeight * 1.1);
addEventListener('scroll', syncToTop, { passive: true });
syncToTop();
toTop.addEventListener('click', () => {
  scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
});

/* ---------------------------------------------------------------------------
   Boot
   ------------------------------------------------------------------------- */
initThree();
if (ok3d) {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  field.scale.setScalar(clamp((innerWidth / innerHeight) / 1.35, 0.52, 1));
  requestAnimationFrame(tick);
}
sections[0].classList.add('is-active');
railDots[0].classList.add('is-active');
updateActive();
applyStep(0);
updateSigma();

// ?dev=gb200 deep-links a device selection
const devParam = new URLSearchParams(location.search).get('dev');
const initialBtn = (devParam && document.querySelector(`#devices .dev[data-key="${devParam}"]`))
  || $('#devices .dev.is-active');
if (initialBtn) selectDevice(initialBtn, false);

requestAnimationFrame(chartResize);
setTimeout(chartResize, 300);
setTimeout(() => drawBarcode(0.35, devSm, devBps), 600);

// Deep links: scroll instantly, then let the observer take over.
function jumpToHash() {
  const id = decodeURIComponent(location.hash.slice(1));
  if (!id) return;
  const el = document.getElementById(id);
  if (!el) return;
  const root = document.documentElement;
  const prev = root.style.scrollBehavior;
  root.style.scrollBehavior = 'auto';
  el.scrollIntoView({ block: 'start' });
  root.style.scrollBehavior = prev;
  updateActive();
}
requestAnimationFrame(jumpToHash);
