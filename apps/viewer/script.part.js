const WASM_BASE64 = "__WASM_BASE64__";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const REASONS = {
  too_wide: ["Más ancha que la apertura de la pinza", "quality"],
  no_force_closure: ["Sin cierre de fuerza", "quality"],
  below_min_quality: ["Calidad por debajo del mínimo", "quality"],
  insufficient_grip_force: ["Fuerza de pinza insuficiente", "quality"],
  pregrasp_unreachable: ["Pre-agarre fuera de alcance", "reach"],
  grasp_unreachable: ["Agarre fuera de alcance", "reach"],
  lift_unreachable: ["Elevación fuera de alcance", "reach"],
  pregrasp_collision: ["Colisión en el pre-agarre", "collision"],
  approach_collision: ["Colisión durante la aproximación", "collision"],
  grasp_collision: ["Colisión al cerrar", "collision"],
  lift_collision: ["Colisión durante la elevación", "collision"],
  inconclusive: ["Holgura no demostrada", "collision"],
};
const EXTENSIONS = {
  wrench_space: "Espacio de llaves de los parches de contacto: cierre de fuerza y épsilon (distancia del origen a la envolvente de las llaves).",
  antipodal: "Prueba antipodal de dos puntos contra el cono de fricción. No tiene en cuenta el par alrededor de la línea de cierre.",
  primitive_antipodal: "Agarres antipodales de cajas, cilindros, cápsulas y esferas, enumerados a partir de sus dimensiones.",
  surface_normal: "Puntos muestreados en la superficie de la pieza: las mordazas cierran a lo largo de la normal o llegan por ella, y los contactos salen de donde las almohadillas tocan. Sirve para mallas y piezas compuestas.",
};
const PHASES = ["Inicio", "Pre-agarre", "Agarre (abierta)", "Cierre", "Elevación"];
// Embedded by build.sh: a real UR5 URDF and its collision meshes (BSD-3).
const SAMPLES = __SAMPLES__;
// Height of the demo cell's table top, metres: where imported parts are set down.
const DEMO_TABLE_TOP = 0.42;
// A ready pose for the UR5 sample: elbow up over the table, gripper pointing down.
const UR5_READY = [-0.49, -1.45, 0.64, -0.76, -1.57, -2.06, 0.0, 0.0];

let engine = null;        // (request) => reply
let info = null;          // last description
let home = null;          // configuration at load
let values = null;        // configuration being shown
let carry = null;         // part carried by the gripper, if any
let grasping = null;      // part the jaws are closed on but have not lifted
let report = null;        // last plan report
let plan = null;          // selected plan
let playing = 0;          // animation token; 0 when idle
let placed = null;        // part drawn where a cycle set it down
let cycleView = null;     // last planned cycle
let cyclePlaying = 0;     // cycle animation token; 0 when idle
const layers = { candidates: true, cones: true, frames: false, collision: false };

async function boot() {
  const bytes = Uint8Array.from(atob(WASM_BASE64), (c) => c.charCodeAt(0));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const x = instance.exports;
  const encoder = new TextEncoder(), decoder = new TextDecoder();
  engine = (request) => {
    const body = encoder.encode(JSON.stringify(request));
    const ptr = x.af_alloc(body.length);
    new Uint8Array(x.memory.buffer, ptr, body.length).set(body);
    const packed = x.af_call(ptr, body.length);
    const rp = Number(packed >> 32n), rl = Number(packed & 0xffffffffn);
    const text = decoder.decode(new Uint8Array(x.memory.buffer, rp, rl));
    x.af_free(rp, rl);
    return JSON.parse(text);
  };
  const reply = engine({ op: "load_demo" });
  if (!reply.ok) throw new Error(reply.error);
  info = reply.data;
  home = info.joints.map((j) => j.value);
  values = home.slice();
  $("engine-state").textContent = "WebAssembly · " + Math.round(bytes.length / 1024) + " kB";
  initScene();
  buildAxes();
  fillBodies();
  fillExtensions();
  draw();
}

/* ================================================================ 3D view */
let renderer, scene, camera, sun, floor, grid, meshes = [], signature = "";
let markers, candidateLines, triad;
const cam = { az: -0.95, el: 0.36, dist: 1.75, target: null };
const goal = { az: -0.95, el: 0.36, dist: 1.75, target: null };
let dirty = true;

function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
const v3 = (p) => new THREE.Vector3(p.x, p.y, p.z);

/* A small photo studio rendered once into an environment map: a soft overhead
   panel, two side panels and a dim floor. It gives metal and paint something
   to reflect; without it every surface is lit only by flat lamps. */
function studioEnvironment() {
  const room = new THREE.Scene();
  const shell = new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), new THREE.MeshBasicMaterial({ color: 0x4a5057, side: THREE.BackSide }));
  room.add(shell);
  const panel = (w, h, x, y, z, rx, ry, power) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(power, power, power), side: THREE.DoubleSide }));
    m.position.set(x, y, z); m.rotation.set(rx, ry, 0);
    room.add(m);
  };
  panel(6, 6, 0, 4.6, 0, Math.PI / 2, 0, 2.6);      // overhead softbox
  panel(3, 4, -4.6, 1, 1, 0, Math.PI / 2, 1.6);     // key side
  panel(3, 4, 4.6, 0.5, -1, 0, -Math.PI / 2, 0.7);  // fill side
  panel(8, 8, 0, -4.6, 0, -Math.PI / 2, 0, 0.2);    // floor bounce
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(room, 0.04).texture;
  pmrem.dispose();
  return texture;
}

function initScene() {
  const host = $("viewport");
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.9;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.insertBefore(renderer.domElement, host.firstChild);

  scene = new THREE.Scene();
  scene.environment = studioEnvironment();
  camera = new THREE.PerspectiveCamera(36, 1.6, 0.05, 40);
  camera.up.set(0, 0, 1);
  cam.target = new THREE.Vector3(0.3, 0, 0.42);
  goal.target = cam.target.clone();

  sun = new THREE.DirectionalLight(0xfff6ea, 1.15);
  sun.position.set(-1.4, -1.9, 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -1.3, right: 1.3, top: 1.3, bottom: -1.3, near: 0.5, far: 8 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.012;
  sun.shadow.radius = 5;
  scene.add(sun, sun.target);
  sun.target.position.set(0.25, 0, 0.3);
  scene.add(new THREE.HemisphereLight(0xdfe9f2, 0x59616a, 0.18));

  floor = new THREE.Mesh(new THREE.CircleGeometry(14, 64), new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0, envMapIntensity: 0.12 }));
  floor.receiveShadow = true;
  floor.position.z = -0.0005;
  scene.add(floor);
  grid = new THREE.GridHelper(4, 40);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.0004;
  grid.material.transparent = true; grid.material.opacity = 0.35; grid.material.depthWrite = false;
  scene.add(grid);

  markers = new THREE.Group();
  candidateLines = new THREE.Group();
  triad = new THREE.Group();
  for (const [axis, color] of [[new THREE.Vector3(1, 0, 0), 0xd64541], [new THREE.Vector3(0, 1, 0), 0x3aa35a], [new THREE.Vector3(0, 0, 1), 0x3a78d6]]) {
    triad.add(new THREE.ArrowHelper(axis, new THREE.Vector3(), 0.07, color, 0.018, 0.009));
  }
  scene.add(markers, candidateLines, triad);
  applyTheme();
  bindCamera(host);
  new ResizeObserver(resize).observe(host);
  resize();
  requestAnimationFrame(frame);
}

function applyTheme() {
  const stage = new THREE.Color(token("--stage"));
  scene.background = stage;
  scene.fog = new THREE.Fog(stage, 5, 13);
  floor.material.color.set(token("--floor"));
  grid.material.color.set(token("--grid"));
  dirty = true;
}

function bindCamera(host) {
  const pointers = new Map();
  let pinch = 0;
  const pan = (dx, dy) => {
    const scale = goal.dist * 0.0012;
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
    goal.target.addScaledVector(right, -dx * scale).addScaledVector(up, dy * scale);
  };
  host.addEventListener("contextmenu", (e) => e.preventDefault());
  host.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button")) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button !== 0 });
    host.setPointerCapture(e.pointerId);
    pinch = 0;
  });
  host.addEventListener("pointermove", (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) goal.dist = Math.max(0.5, Math.min(7, goal.dist * pinch / d));
      pinch = d;
      pan(dx * 0.5, dy * 0.5);
    } else if (p.pan) {
      pan(dx, dy);
    } else {
      goal.az -= dx * 0.006;
      goal.el = Math.max(-0.05, Math.min(1.5, goal.el + dy * 0.006));
    }
    dirty = true;
  });
  const release = (e) => { pointers.delete(e.pointerId); pinch = 0; };
  host.addEventListener("pointerup", release);
  host.addEventListener("pointercancel", release);
  host.addEventListener("wheel", (e) => {
    e.preventDefault();
    goal.dist = Math.max(0.5, Math.min(7, goal.dist * Math.exp(e.deltaY * 0.001)));
    dirty = true;
  }, { passive: false });
}

const VIEWS = {
  iso: { az: -0.95, el: 0.36, dist: 1.75, target: [0.3, 0, 0.42] },
  front: { az: -Math.PI / 2, el: 0.08, dist: 1.9, target: [0.3, 0, 0.45] },
  side: { az: Math.PI, el: 0.08, dist: 1.9, target: [0.3, 0, 0.45] },
  top: { az: -Math.PI / 2, el: 1.5, dist: 1.9, target: [0.35, 0, 0.42] },
};
function setView(name) {
  const view = VIEWS[name];
  // Turn the short way round rather than unwinding accumulated orbits.
  goal.az = cam.az + Math.atan2(Math.sin(view.az - cam.az), Math.cos(view.az - cam.az));
  goal.el = view.el; goal.dist = view.dist;
  goal.target.set(...view.target);
  dirty = true;
}

function resize() {
  const host = $("viewport");
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  dirty = true;
}

function frame() {
  requestAnimationFrame(frame);
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const k = still ? 1 : 0.18;
  const gap = Math.abs(goal.az - cam.az) + Math.abs(goal.el - cam.el) + Math.abs(goal.dist - cam.dist) + cam.target.distanceTo(goal.target);
  if (gap > 1e-4) {
    cam.az += (goal.az - cam.az) * k; cam.el += (goal.el - cam.el) * k; cam.dist += (goal.dist - cam.dist) * k;
    cam.target.lerp(goal.target, k);
    dirty = true;
  }
  if (!dirty) return;
  dirty = false;
  const c = Math.cos(cam.el);
  camera.position.set(cam.target.x + cam.dist * c * Math.cos(cam.az), cam.target.y + cam.dist * c * Math.sin(cam.az), cam.target.z + cam.dist * Math.sin(cam.el));
  camera.lookAt(cam.target);
  renderer.render(scene, camera);
}

/* A cylinder with a small chamfer on both rims, along Z. Sharp-edged
   primitives are what makes a scene read as a diagram instead of an object. */
function chamferedCylinder(radius, length) {
  const c = Math.min(radius, length) * 0.08, h = length / 2;
  const profile = [[0, -h], [radius - c, -h], [radius, -h + c], [radius, h - c], [radius - c, h], [0, h]].map(([x, y]) => new THREE.Vector2(x, y));
  return new THREE.LatheGeometry(profile, 56).rotateX(Math.PI / 2);
}

/* A box with rounded edges, centred on the origin. */
function roundedBox(w, h, d) {
  const r = Math.min(0.0035, Math.min(w, h, d) * 0.18);
  const x = w / 2 - r, y = h / 2 - r;
  const outline = new THREE.Shape();
  outline.moveTo(-x, -y); outline.lineTo(x, -y); outline.lineTo(x, y); outline.lineTo(-x, y); outline.closePath();
  const geometry = new THREE.ExtrudeGeometry(outline, { depth: d - 2 * r, bevelEnabled: true, bevelThickness: r, bevelSize: r, bevelSegments: 3, curveSegments: 4 });
  return geometry.translate(0, 0, -(d - 2 * r) / 2);
}

/* Imported meshes, by resource id. Shared between drawables and kept across
   rebuilds, so they are fetched from the engine and uploaded to the GPU once. */
const meshCache = new Map();

function loadMeshes(drawables) {
  const missing = [...new Set(drawables.filter((d) => d.shape.kind === "mesh").map((d) => d.shape.resourceId))]
    .filter((id) => !meshCache.has(id));
  if (!missing.length) return;
  const reply = engine({ op: "get_resources", ids: missing });
  if (!reply.ok) return;
  for (const mesh of reply.data) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(mesh.vertices, 3));
    geometry.setIndex(mesh.indices);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    geometry.userData.shared = true;
    meshCache.set(mesh.id, geometry);
  }
}

function geometryFor(shape) {
  if (shape.kind === "mesh") return meshCache.get(shape.resourceId) || new THREE.BoxGeometry(0.01, 0.01, 0.01);
  if (shape.kind === "box") return roundedBox(shape.size.x, shape.size.y, shape.size.z);
  if (shape.kind === "sphere") return new THREE.SphereGeometry(shape.radius, 48, 32);
  if (shape.kind === "cylinder") return chamferedCylinder(shape.radius, shape.length);
  if (shape.kind === "capsule") return new THREE.CapsuleGeometry(shape.radius, shape.length, 12, 32).rotateX(Math.PI / 2);
  return new THREE.BoxGeometry(0.02, 0.02, 0.02);
}

function syncMeshes(drawables, contactIds) {
  const next = drawables.map((d) => JSON.stringify(d.shape) + d.visual + d.collision).join("|");
  if (next !== signature) {
    signature = next;
    meshes.forEach((m) => { scene.remove(m); if (!m.geometry.userData.shared) m.geometry.dispose(); m.material.dispose(); });
    loadMeshes(drawables);
    meshes = drawables.map((d) => {
      const material = d.visual
        // Imported meshes are faceted CAD exports; flat shading shows their edges as edges.
        ? new THREE.MeshStandardMaterial({ envMapIntensity: 0.9, flatShading: d.shape.kind === "mesh" })
        : new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false, wireframe: false });
      const mesh = new THREE.Mesh(geometryFor(d.shape), material);
      if (d.shape.kind === "mesh") mesh.scale.set(d.shape.scale.x, d.shape.scale.y, d.shape.scale.z);
      mesh.castShadow = d.visual; mesh.receiveShadow = d.visual;
      scene.add(mesh);
      return mesh;
    });
  }
  const alert = new THREE.Color(token("--alert")), accent = token("--accent");
  drawables.forEach((d, i) => {
    const m = meshes[i], t = d.pose.translation, q = d.pose.rotation;
    m.position.set(t.x, t.y, t.z);
    m.quaternion.set(q.x, q.y, q.z, q.w);
    const hit = contactIds.has(d.ownerId);
    if (d.visual) {
      m.visible = true;
      m.material.color.set(d.color);
      m.material.metalness = d.metalness; m.material.roughness = d.roughness;
      m.material.emissive.copy(hit ? alert : new THREE.Color(0x000000));
      m.material.emissiveIntensity = hit ? 0.55 : 0;
    } else {
      // Collision-only shapes are an overlay, shown on request.
      m.visible = layers.collision;
      m.material.color.set(hit ? alert : accent);
    }
  });
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    group.remove(child);
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }
}

/* The selected grasp: its contact points, the friction cone at each, and the
   approach direction. */
function showMarkers() {
  clearGroup(markers);
  if (!plan) { dirty = true; return; }
  const c = plan.candidate, accent = token("--accent");
  const contacts = c.contacts && c.contacts.length ? c.contacts : [
    { point: { x: c.center.x + c.closing.x * c.width / 2, y: c.center.y + c.closing.y * c.width / 2, z: c.center.z + c.closing.z * c.width / 2 }, normal: c.normalA },
    { point: { x: c.center.x - c.closing.x * c.width / 2, y: c.center.y - c.closing.y * c.width / 2, z: c.center.z - c.closing.z * c.width / 2 }, normal: c.normalB },
  ];
  const mu = plan.quality.frictionCoefficient, height = 0.016;
  for (const contact of contacts) {
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.0022, 12, 8), new THREE.MeshBasicMaterial({ color: accent }));
    dot.position.copy(v3(contact.point));
    markers.add(dot);
    if (layers.cones && mu > 0) {
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(height * mu, height, 28, 1, true).translate(0, -height / 2, 0),
        new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false }));
      cone.position.copy(v3(contact.point));
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), v3(contact.normal).normalize());
      markers.add(cone);
    }
  }
  const approach = v3(c.approach);
  markers.add(new THREE.ArrowHelper(approach, v3(c.center).addScaledVector(approach, -0.15), 0.1, accent, 0.028, 0.014));
  dirty = true;
}

/* Every candidate the planner examined, as a short stroke along its approach,
   coloured by what became of it. */
function showCandidates() {
  clearGroup(candidateLines);
  $("legend").hidden = !(report && layers.candidates);
  if (!report || !layers.candidates) { dirty = true; return; }
  const colors = { ok: new THREE.Color(token("--ok")), reach: new THREE.Color(token("--faint")), collision: new THREE.Color(token("--warn")), quality: new THREE.Color(token("--alert")) };
  const positions = [], tints = [];
  const stroke = (center, approach, color) => {
    const a = v3(center).addScaledVector(v3(approach), -0.12), b = v3(center).addScaledVector(v3(approach), -0.045);
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    tints.push(color.r, color.g, color.b, color.r, color.g, color.b);
  };
  report.rejections.forEach((r) => stroke(r.center, r.approach, colors[(REASONS[r.reason] || ["", "reach"])[1]]));
  report.plans.forEach((p) => stroke(p.candidate.center, p.candidate.approach, colors.ok));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(tints, 3));
  candidateLines.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })));
  dirty = true;
}

/* ============================================================= state view */
/* Moving links whose collisions nothing checks (their meshes were not imported). */
function unchecked() { return (info && info.linksWithoutCollision) || []; }

function draw() {
  const request = { op: "snapshot", values };
  if (carry) request.carry = carry;
  else if (grasping) request.grasping = grasping;
  if (placed) request.placed = placed;
  const reply = engine(request);
  const flag = $("contact-flag");
  flag.title = "";
  if (!reply.ok) { flag.textContent = reply.error; flag.className = "viewport__flag flag--alert"; return; }
  const s = reply.data;
  const contacts = s.collision.contacts;
  const ids = new Set();
  contacts.forEach((c) => { ids.add(c.firstId); ids.add(c.secondId); });
  syncMeshes(s.drawables, ids);

  if (contacts.length) {
    flag.textContent = "contacto: " + contacts[0].firstName + " ↔ " + contacts[0].secondName + (contacts.length > 1 ? " (+" + (contacts.length - 1) + ")" : "");
    flag.className = "viewport__flag flag--alert";
  } else if (!s.collision.conclusive) {
    flag.textContent = "holgura no demostrada"; flag.className = "viewport__flag flag--alert";
  } else if (unchecked().length) {
    flag.textContent = "libre · " + unchecked().length + " eslabones sin comprobar"; flag.className = "viewport__flag flag--warn";
    flag.title = "Sin geometría de colisión: " + unchecked().join(", ");
  } else { flag.textContent = "libre"; flag.className = "viewport__flag flag--ok"; }

  // The grasp markers sit where the part rests; once it is carried they would float.
  markers.visible = !carry;
  triad.visible = layers.frames && !!s.toolPose;
  if (s.toolPose) {
    const t = s.toolPose.translation, q = s.toolPose.rotation;
    $("tcp-x").textContent = fmt(t.x * 1000);
    $("tcp-y").textContent = fmt(t.y * 1000);
    $("tcp-z").textContent = fmt(t.z * 1000);
    triad.position.set(t.x, t.y, t.z);
    triad.quaternion.set(q.x, q.y, q.z, q.w);
  }
  if (info.gripperFingers > 0) {
    // A hand has no single opening: say how many fingers hold the part.
    const touching = plan && plan.candidate.hand && (carry || grasping) ? plan.candidate.hand.fingersInContact : 0;
    $("opening").innerHTML = touching ? touching + ' <span class="gauge__unit">dedos</span>' : info.gripperFingers + ' <span class="gauge__unit">dedos</span>';
    $("opening-note").textContent = touching ? (carry ? "sujetando la pieza" : "dedos en contacto") : "mano abierta";
  } else {
    const jaws = info.joints.map((j, i) => (j.role === "end_effector" ? values[i] : null)).filter((v) => v !== null);
    $("opening").innerHTML = jaws.length ? fmt(jaws.reduce((a, b) => a + b, 0) * 1000) + ' <span class="gauge__unit">mm</span>' : "—";
    $("opening-note").textContent = carry ? "sujetando la pieza" : "entre mordazas";
  }

  const cond = s.conditioning;
  if (cond) {
    $("sigma").textContent = fmt(cond.minSingularValue, 4);
    $("gauge-sigma").className = "gauge " + (cond.nearSingular ? "gauge--warn" : "gauge--ok");
    $("sigma-note").textContent = cond.nearSingular ? "cerca de una singularidad" : "bien condicionado";
  }
  const closest = s.collision.closest;
  if (closest) {
    $("clearance").innerHTML = fmt(closest.distance * 1000) + ' <span class="gauge__unit">mm</span>';
    $("clearance-note").textContent = closest.firstName + " ↔ " + closest.secondName;
    $("gauge-clearance").className = "gauge " + (closest.distance <= 0 ? "gauge--alert" : closest.distance < 0.01 ? "gauge--warn" : "gauge--ok");
  }
  updateAxes(new Set(s.outOfLimits));
  dirty = true;
}

function showQuality() {
  const gauge = $("gauge-quality");
  if (!plan) {
    $("quality").textContent = "—"; $("quality-note").textContent = "sin agarre seleccionado"; gauge.className = "gauge";
    return;
  }
  const q = plan.quality;
  $("quality").textContent = fmt(q.quality, 4);
  $("quality-note").textContent = (q.forceClosure ? "cierre de fuerza" : "sin cierre de fuerza") + " · " + q.contactCount + " contactos · " + q.measure;
  gauge.className = "gauge " + (q.forceClosure ? "gauge--ok" : "gauge--warn");
}

/* =================================================================== axes */
function unit(j) { return j.kind === "revolute" ? { k: 180 / Math.PI, s: "°", d: 1 } : { k: 1000, s: " mm", d: 1 }; }

function buildAxes() {
  const host = $("axes");
  host.textContent = "";
  const extra = info.joints.length - info.axisCount;
  $("dof-note").textContent = info.robotName + " · " + info.axisCount + " ejes" + (extra ? " + " + extra + " de pinza" : "");
  $("gripper-note").textContent = !info.gripper ? "sin pinza" :
    info.gripperFingers > 0 ? "mano · " + info.gripperFingers + " dedos" : "pinza · " + fmt(info.gripperMaxOpening * 1000, 0) + " mm";
  fillPreshapes();
  let separated = false;
  info.joints.forEach((j, i) => {
    if (j.role === "end_effector" && !separated) {
      separated = true;
      const sep = document.createElement("div");
      sep.className = "axes__sep"; sep.textContent = "Pinza (semiapertura por mordaza)";
      host.appendChild(sep);
    }
    const u = unit(j), row = document.createElement("div");
    row.className = "axis"; row.id = "axis-" + i;
    row.innerHTML = '<div class="axis__top"><label class="axis__name" for="joint-' + i + '"></label><span class="axis__value"></span></div>' +
      '<input type="range" id="joint-' + i + '"><div class="axis__limits"><span></span><span></span></div>';
    row.querySelector("label").textContent = j.name;
    const input = row.querySelector("input");
    input.min = j.lower * u.k; input.max = j.upper * u.k; input.step = 0.5;
    const lim = row.querySelectorAll(".axis__limits span");
    lim[0].textContent = fmt(j.lower * u.k, 0) + u.s; lim[1].textContent = fmt(j.upper * u.k, 0) + u.s;
    input.addEventListener("input", () => {
      stopPlayback();
      values[i] = Number(input.value) / u.k;
      carry = null; grasping = null;
      setPhaseLabel(plan ? "movimiento manual" : "sin plan");
      draw();
    });
    host.appendChild(row);
  });
}

function updateAxes(out) {
  info.joints.forEach((j, i) => {
    const u = unit(j), row = $("axis-" + i);
    row.classList.toggle("axis--out", out.has(j.name));
    row.querySelector(".axis__value").textContent = fmt(values[i] * u.k, u.d) + u.s;
    const input = row.querySelector("input");
    if (document.activeElement !== input) input.value = values[i] * u.k;
  });
}

/* ================================================================= bodies */
function selectedBody() { return info.bodies.find((b) => b.id === $("body").value); }

function fillBodies() {
  const select = $("body"), keep = select.value;
  select.textContent = "";
  info.bodies.filter((b) => b.graspable).forEach((b) => {
    const o = document.createElement("option");
    o.value = b.id; o.textContent = b.name;
    select.appendChild(o);
  });
  if (keep && info.bodies.some((b) => b.id === keep)) select.value = keep;
  fillBodyFields();
}

function fillBodyFields() {
  const b = selectedBody();
  if (!b) return;
  $("bx").value = Math.round(b.position.x * 1000);
  $("by").value = Math.round(b.position.y * 1000);
  $("bm").value = b.mass;
  $("bf").value = b.frictionCoefficient;
}

function applyBody() {
  const b = selectedBody();
  if (!b) return false;
  const reply = engine({
    op: "update_body", bodyId: b.id,
    position: { x: Number($("bx").value) / 1000, y: Number($("by").value) / 1000, z: b.position.z },
    mass: Number($("bm").value), frictionCoefficient: Number($("bf").value),
  });
  $("edit-error").hidden = reply.ok;
  if (!reply.ok) { $("edit-error").className = "verdict verdict--alert"; $("edit-error-text").textContent = reply.error; return false; }
  info = reply.data;
  return true;
}

function clearPlan() {
  stopPlayback();
  clearCycle();
  report = null; plan = null; carry = null; grasping = null;
  $("plan-module").hidden = true;
  setPlayable(false);
  setPhaseLabel("sin plan");
  showMarkers(); showCandidates(); showQuality();
}

/* ============================================================= extensions */
function fillExtensions() {
  const reply = engine({ op: "list_extensions" });
  if (!reply.ok) return;
  const list = $("exts");
  list.textContent = "";
  const selects = { quality_measure: $("o-measure"), candidate_generator: $("o-generator") };
  Object.values(selects).forEach((s) => { s.textContent = ""; });
  // "auto" is the planner's own choice: exact grasps for a primitive, the
  // surface for anything else.
  const auto = document.createElement("option");
  auto.value = "auto"; auto.textContent = "auto (según la pieza)";
  selects.candidate_generator.appendChild(auto);
  reply.data.forEach((e) => {
    const li = document.createElement("li"), name = document.createElement("b"), kind = document.createElement("i"), text = document.createElement("span");
    name.textContent = e.id;
    kind.className = "tag"; kind.textContent = e.kind === "quality_measure" ? "medida de calidad" : "generador";
    text.textContent = EXTENSIONS[e.id] || e.summary;
    li.append(name, kind, text);
    list.appendChild(li);
    const option = document.createElement("option");
    option.value = e.id; option.textContent = e.id;
    selects[e.kind].appendChild(option);
  });
  $("o-measure").value = "wrench_space";
  $("o-generator").value = "auto";
  $("ext-note").textContent = reply.data.length + " registradas";
}

/* A hand approaches in a preshape and closes its fingers until they touch;
   it has its own grasp generator, so the generator choice does not apply. */
function fillPreshapes() {
  const select = $("o-preshape"), hand = info.gripperFingers > 0;
  select.textContent = "";
  (info.preshapes || []).forEach((name, i) => {
    const option = document.createElement("option");
    option.value = name; option.textContent = name + (i === 0 ? " (por defecto)" : "");
    select.appendChild(option);
  });
  $("o-preshape-field").hidden = !hand || !(info.preshapes || []).length;
  $("o-generator").disabled = hand;
  if (hand) $("o-generator").value = "auto";
}

function plannerOptions() {
  const number = (id) => Number($(id).value);
  const optional = (id) => ($(id).value.trim() === "" ? null : Number($(id).value));
  const options = {
    nrGrasps: Math.max(1, Math.round(number("o-nr"))),
    minQuality: number("o-minq"),
    forceClosure: $("o-fc").checked,
    qualityMeasure: $("o-measure").value,
    candidateGenerator: $("o-generator").value,
    quality: {
      unitForce: number("o-force"),
      frictionCoefficient: optional("o-mu"),
      frictionConeSamples: Math.round(number("o-cone")),
      torsionalFriction: number("o-torsion") / 1000,
    },
    standoff: number("o-standoff") / 1000,
    liftHeight: number("o-lift") / 1000,
    jawClearance: number("o-jaw") / 1000,
  };
  const limit = optional("o-max");
  if (limit !== null) options.maxCandidates = Math.round(limit);
  if (info.gripperFingers > 0 && $("o-preshape").value) options.preshape = $("o-preshape").value;
  return options;
}

const DEFAULTS = { "o-nr": "3", "o-minq": "0", "o-mu": "", "o-cone": "8", "o-force": "1", "o-torsion": "0", "o-standoff": "80", "o-lift": "100", "o-jaw": "5", "o-max": "" };
function noteParams() {
  const changed = Object.entries(DEFAULTS).some(([id, value]) => $(id).value !== value) || !$("o-fc").checked || $("o-measure").value !== "wrench_space" || $("o-generator").value !== "auto";
  $("params-note").textContent = changed ? "modificados" : "por defecto";
}

/* =================================================================== plan */
function describePlan(p) {
  const a = p.candidate.approach;
  const way = a.z < -0.7 ? "Desde arriba" : a.z > 0.7 ? "Desde abajo" : "Lateral";
  const label = p.candidate.label;
  if (p.candidate.hand) {
    return way + (label.includes(" deep ") ? ", mano hasta el fondo" : "") + (label.includes("tilted") ? ", inclinada" : "") + (label.includes("centred") ? ", centrada" : "");
  }
  return way + (label.includes("shallow") ? ", agarre poco profundo" : "");
}

function runPlan() {
  if (!applyBody()) return;
  clearPlan();
  values = home.slice();
  engine({ op: "set_joints", values });
  const b = selectedBody(), options = plannerOptions(), started = performance.now();
  const reply = engine({ op: "plan_pick", bodyId: b.id, options });
  const elapsed = performance.now() - started;
  $("plan-module").hidden = false;
  const verdict = $("plan-verdict");
  if (!reply.ok) {
    verdict.className = "verdict verdict--alert";
    $("plan-head").textContent = "No se pudo planificar";
    $("plan-detail").textContent = reply.error;
    $("plans").textContent = ""; $("reasons").textContent = ""; $("rej-details").hidden = true; $("plan-note").textContent = "";
    draw();
    return;
  }
  report = reply.data;
  $("plan-note").textContent = report.qualityMeasure + " · " + fmt(elapsed, 0) + " ms";
  const n = report.plans.length;
  const blind = unchecked();
  verdict.className = "verdict " + (n && !blind.length ? "verdict--ok" : "verdict--warn");
  $("plan-head").textContent = n ? (n === 1 ? "1 agarre viable" : n + " agarres viables") : "Ningún agarre viable";
  $("plan-detail").textContent = "Se examinaron " + report.candidatesExamined + " de " + report.candidatesTotal + " agarres candidatos para «" + b.name + "»; " + report.rejections.length + " descartados." +
    (n ? "" : " Prueba a mover la pieza, aligerarla, aumentar la fricción o relajar los parámetros.") +
    (blind.length ? " Atención: " + blind.length + " eslabones móviles no tienen geometría de colisión (" + blind.join(", ") + "); sus choques no se han comprobado." : "");

  const host = $("plans");
  host.textContent = "";
  report.plans.forEach((p, i) => {
    const el = document.createElement("button");
    el.type = "button"; el.className = "plan";
    el.innerHTML = '<span class="plan__name"></span><span class="plan__metrics">' +
      "<span>ε <b>" + fmt(p.quality.quality, 4) + "</b></span>" +
      '<span class="pill ' + (p.quality.forceClosure ? "pill--ok" : "pill--warn") + '">' + (p.quality.forceClosure ? "cierre de fuerza" : "sin cierre") + "</span>" +
      "<span>contactos <b>" + p.quality.contactCount + "</b></span>" +
      (p.candidate.hand ? "<span>dedos <b>" + p.candidate.hand.fingersInContact + "</b></span>" : "<span>ancho <b>" + fmt(p.candidate.width * 1000) + " mm</b></span>") +
      "<span>fuerza <b>" + fmt(p.quality.requiredGripForce, 1) + " N</b></span>" +
      "<span>holgura mín <b>" + fmt(p.minClearance * 1000) + " mm</b></span>" +
      "<span>puntuación <b>" + fmt(p.quality.score, 3) + "</b></span></span>" +
      '<span class="plan__label"></span>';
    el.querySelector(".plan__name").textContent = (i + 1) + ". " + describePlan(p);
    el.querySelector(".plan__label").textContent = p.candidate.label;
    el.addEventListener("click", () => selectPlan(i));
    host.appendChild(el);
  });

  const counts = {};
  report.rejections.forEach((r) => { counts[r.reason] = (counts[r.reason] || 0) + 1; });
  const reasons = $("reasons");
  reasons.textContent = "";
  Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([reason, count]) => {
    const li = document.createElement("li"), a = document.createElement("span"), c = document.createElement("span");
    a.textContent = (REASONS[reason] || [reason])[0]; c.textContent = count;
    li.append(a, c); reasons.appendChild(li);
  });
  const list = $("rejlist");
  list.textContent = "";
  report.rejections.forEach((r) => {
    const li = document.createElement("li"), b2 = document.createElement("b");
    b2.textContent = r.label;
    li.append(b2, document.createTextNode(" — " + (REASONS[r.reason] || [r.reason])[0] + ": " + r.detail));
    list.appendChild(li);
  });
  $("rej-details").hidden = report.rejections.length === 0;
  showCandidates();
  if (n) selectPlan(0); else draw();
}

function selectPlan(index) {
  stopPlayback();
  plan = report.plans[index];
  [...$("plans").children].forEach((el, i) => el.setAttribute("aria-pressed", String(i === index)));
  setPlayable(true);
  showMarkers(); showQuality();
  goToPhase(0);
}

function keyframes() { return [home, plan.pregrasp, plan.graspOpen, plan.graspClosed, plan.lift]; }
// Closed on the part it is grasped but still resting where it was; only once
// the lift begins is it carried.
function holdFor(phase, lifting) {
  carry = phase >= 4 || lifting ? { bodyId: report.bodyId, partInPalm: plan.partInPalm } : null;
  grasping = phase === 3 && !lifting ? report.bodyId : null;
}

function setPlayable(on) {
  $("btn-play").disabled = !on;
  document.querySelectorAll(".phase").forEach((b) => { b.disabled = !on; b.setAttribute("aria-pressed", "false"); });
}
function setPhaseLabel(text) { $("phase-label").textContent = text; }

function goToPhase(phase) {
  values = keyframes()[phase].slice();
  holdFor(phase, false);
  document.querySelectorAll(".phase").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.phase) === phase)));
  setPhaseLabel(PHASES[phase]);
  draw();
}

function stopPlayback() { playing = 0; $("btn-play").textContent = "Reproducir agarre"; }

function play() {
  if (!plan) return;
  if (playing) { stopPlayback(); return; }
  const frames = keyframes(), durations = [1500, 1200, 650, 1200];
  const tokenId = playing = performance.now();
  $("btn-play").textContent = "Detener";
  let segment = 0, start = performance.now();
  goToPhase(0);
  const tick = (now) => {
    if (playing !== tokenId) return;
    let t = (now - start) / durations[segment];
    if (t >= 1) {
      segment += 1; start = now; t = 0;
      if (segment >= durations.length) { stopPlayback(); goToPhase(4); return; }
    }
    const e = t * t * (3 - 2 * t), a = frames[segment], b = frames[segment + 1];
    values = a.map((v, i) => v + (b[i] - v) * e);
    holdFor(segment, segment === 3 && e > 0);
    setPhaseLabel(PHASES[segment] + " → " + PHASES[segment + 1]);
    draw();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ================================================================= cycle */
const SEGMENT_NAMES = {
  open: "Abrir pinza", approach: "Aproximación", descend: "Descenso", close: "Cierre",
  lift: "Elevación", transfer: "Transporte", lower: "Bajada", release: "Soltar",
  retreat: "Retirada", return: "Regreso", rest: "Reposo",
};
const PLANNERS = { straight: "recto", rrt: "RRT", gripper: "pinza" };

function stopCycle() { cyclePlaying = 0; $("btn-cycle-play").textContent = "Reproducir ciclo"; }

function clearCycle() {
  stopCycle();
  cycleView = null; placed = null;
  $("cycle-verdict").hidden = true;
  $("cycle-segments").textContent = "";
  $("cycle-exports").hidden = true;
  $("btn-cycle-play").disabled = true;
  $("cycle-note").textContent = "recoger, llevar y dejar";
}

function runCycle() {
  if (!applyBody()) return;
  clearPlan();
  values = home.slice();
  engine({ op: "set_joints", values });
  const b = selectedBody();
  const target = { x: Number($("cx").value) / 1000, y: Number($("cy").value) / 1000, z: b.position.z };
  const speed = Math.min(1, Math.max(0.05, Number($("cspeed").value) / 100));
  const grasp = Object.assign(plannerOptions(), { nrGrasps: 6 });
  const started = performance.now();
  const reply = engine({ op: "plan_cycle", bodyId: b.id, target, options: { speedScale: speed, grasp } });
  const elapsed = performance.now() - started;
  const verdict = $("cycle-verdict");
  verdict.hidden = false;
  $("cycle-note").textContent = fmt(elapsed, 0) + " ms";
  if (!reply.ok) {
    verdict.className = "verdict verdict--alert";
    $("cycle-head").textContent = "No se pudo planificar el ciclo";
    $("cycle-detail").textContent = reply.error;
    draw();
    return;
  }
  const view = reply.data, cycle = view.report.cycle;
  if (!cycle) {
    verdict.className = "verdict verdict--alert";
    $("cycle-head").textContent = "Ningún ciclo completo";
    $("cycle-detail").textContent = view.report.graspsFound + " agarres viables; ninguno permite el ciclo entero. " +
      view.report.attempts.slice(0, 3).map((a) => (SEGMENT_NAMES[a.segment] || a.segment) + ": " + a.reason).join(" · ");
    draw();
    return;
  }
  cycleView = view;
  const program = view.programSeconds, diff = Math.abs(program - cycle.cycleTime) / cycle.cycleTime * 100;
  verdict.className = "verdict verdict--ok";
  $("cycle-head").textContent = "Ciclo de " + fmt(cycle.cycleTime, 2) + " s";
  $("cycle-detail").textContent = "«" + view.partName + "» con el agarre " + cycle.grasp.candidate.label +
    ", el más rápido de " + view.report.cyclesFound + " ciclos completos (" + view.report.graspsFound + " agarres viables)" +
    ". El programa, re-cronometrado como lo ejecutaría un controlador, tarda " + fmt(program, 2) + " s (diferencia " + fmt(diff, 3) + " %). " +
    "Holgura demostrada ≥ " + fmt(cycle.clearance * 1000, 1) + " mm en todo el recorrido; la mínima medida es " + fmt(cycle.closest * 1000, 1) + " mm." +
    (view.report.assumedLimits.length ? " Límites no declarados, supuestos: " + view.report.assumedLimits.join(", ") + "." : "");
  const list = $("cycle-segments");
  list.textContent = "";
  cycle.segments.forEach((s) => {
    const li = document.createElement("li"), name = document.createElement("span"), numbers = document.createElement("span");
    name.textContent = (SEGMENT_NAMES[s.name] || s.name) + " · " + (PLANNERS[s.planner] || s.planner) + (s.moves.length > 1 ? " · " + s.moves.length + " tramos" : "");
    numbers.textContent = fmt(s.duration + s.dwell, 2) + " s · ≥ " + fmt(s.clearance * 1000, 1) + " mm";
    li.title = s.allowedContacts.length ? "Se permite el contacto " + s.allowedContacts.join(", ") : "";
    li.append(name, numbers);
    list.appendChild(li);
  });
  $("cycle-exports").hidden = false;
  $("btn-cycle-play").disabled = false;
  showCycleAt(0);
}

function showCycleAt(t) {
  const samples = cycleView.samples, cycle = cycleView.report.cycle;
  const i = Math.min(samples.points.length - 1, Math.max(0, Math.round(t / samples.period)));
  const point = samples.points[i];
  values = point.positions.slice();
  carry = point.part === "carried" ? { bodyId: cycle.bodyId, partInPalm: cycle.grasp.partInPalm } : null;
  placed = point.part === "placed" ? { bodyId: cycle.bodyId, pose: cycle.placePose } : null;
  grasping = null;
  const segment = cycle.segments.find((s) => t < s.startTime + s.duration + s.dwell) || cycle.segments[cycle.segments.length - 1];
  setPhaseLabel((SEGMENT_NAMES[segment.name] || segment.name) + " · " + fmt(Math.min(t, cycle.cycleTime), 1) + " / " + fmt(cycle.cycleTime, 1) + " s");
  draw();
}

function playCycle() {
  if (!cycleView) return;
  if (cyclePlaying) { stopCycle(); return; }
  stopPlayback();
  const token = cyclePlaying = performance.now(), total = cycleView.report.cycle.cycleTime;
  $("btn-cycle-play").textContent = "Detener";
  const tick = (now) => {
    if (cyclePlaying !== token) return;
    const t = (now - token) / 1000;
    showCycleAt(t);
    if (t >= total) { stopCycle(); return; }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function exportCycle(format) {
  const reply = engine({ op: "export_cycle", format });
  if (!reply.ok) { $("cycle-detail").textContent = reply.error; return; }
  const file = reply.data;
  const url = URL.createObjectURL(new Blob([file.text], { type: file.mediaType }));
  const a = document.createElement("a");
  a.href = url; a.download = file.fileName;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ================================================================= wiring */
/* ================================================================= import */
function frameScene() {
  const box = new THREE.Box3();
  meshes.forEach((m) => { if (m.visible) box.expandByObject(m); });
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
  goal.target.copy(center);
  goal.dist = Math.max(0.8, Math.max(size.x, size.y, size.z) * 1.7);
  dirty = true;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("no se pudo leer " + file.name));
    reader.readAsArrayBuffer(file);
  });
}

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

function showImport(kind, head, detail, warnings) {
  const box = $("import-result");
  box.hidden = false;
  box.className = "verdict verdict--" + kind;
  $("import-head").textContent = head;
  $("import-detail").textContent = detail;
  const list = $("import-warnings");
  list.textContent = "";
  (warnings || []).forEach((w) => { const li = document.createElement("li"); li.textContent = w; list.appendChild(li); });
}

/* After the engine holds a new robot: rebuild everything that depends on it. */
function adoptRobot(description, ready) {
  info = description;
  if (ready && ready.length === info.joints.length) {
    const reply = engine({ op: "set_joints", values: ready });
    if (reply.ok) info.joints.forEach((j, i) => { j.value = ready[i]; });
  }
  home = info.joints.map((j) => j.value);
  values = home.slice();
  signature = "";
  clearPlan();
  buildAxes();
  fillBodies();
  draw();
  frameScene();
}

function importRobot(format, text, files, ready) {
  const eef = $("imp-eef").value.split(",").map((s) => s.trim()).filter(Boolean);
  const reply = engine({
    op: "import_robot", format, text, files,
    endEffectorJoints: eef,
    attachGripper: $("imp-gripper").checked ? "" : null,
    gripperKind: $("imp-kind").value,
    intoDemoCell: $("imp-cell").checked,
  });
  if (!reply.ok) { showImport("alert", "No se pudo importar", reply.error, []); return; }
  const d = reply.data.description;
  adoptRobot(d, ready);
  showImport("ok", "Importado: " + d.robotName,
    d.axisCount + " ejes, " + d.joints.length + " articulaciones móviles" + (d.gripper ? ", con pinza" : ", sin pinza") + ". " +
    (reply.data.warnings.length ? "Lo que no se pudo trasladar exactamente:" : "Todo se trasladó."), reply.data.warnings);
}

async function importFiles() {
  const picked = [...$("import-files").files];
  if (!picked.length) { showImport("warn", "Sin archivos", "Elige primero la descripción y sus mallas.", []); return; }
  const descriptions = picked.filter((f) => /\.(urdf|xml)$/i.test(f.name));
  // One mesh and no description: a part straight from CAD.
  if (!descriptions.length && picked.length === 1 && /\.(stl|obj|dae)$/i.test(picked[0].name)) {
    try { addMeshObject(picked[0].name, toBase64(await readFile(picked[0]))); } catch (error) { showImport("alert", "No se pudo leer", error.message, []); }
    return;
  }
  if (descriptions.length !== 1) {
    showImport("warn", "Falta la descripción", descriptions.length ? "Elige una sola descripción (.urdf o .xml) a la vez." : "Entre los archivos debe haber un .urdf o un .xml de Simox.", []);
    return;
  }
  try {
    const files = {};
    for (const f of picked.filter((f) => /\.(stl|obj|dae)$/i.test(f.name))) files[f.name] = toBase64(await readFile(f));
    const text = new TextDecoder().decode(await readFile(descriptions[0]));
    if (/\.urdf$/i.test(descriptions[0].name)) { importRobot("urdf", text, files); return; }
    if (/<\s*ManipulationObject[\s>]/i.test(text)) { addObject(text, files); return; }
    if (/<\s*Robot[\s>]/i.test(text)) {
      // A URDF saved as .xml has a lower-case <robot name=...>; Simox's has Type and RootNode.
      importRobot(/RootNode\s*=/i.test(text) ? "simox" : "urdf", text, files);
      return;
    }
    showImport("alert", "Formato no reconocido", "El .xml no contiene <Robot> ni <ManipulationObject>.", []);
  } catch (error) {
    showImport("alert", "No se pudo leer", error.message, []);
  }
}

/* Where a new part goes: in front of the robot, on the table if there is one. */
function partPlacement() {
  const onTable = info.bodies.some((b) => b.name === "table");
  return { onTable, position: { x: 0.40, y: 0.0, z: 0 }, restOn: onTable ? DEMO_TABLE_TOP : info.groundHeight };
}

function adoptPart(reply) {
  info = reply.data.description;
  const added = info.bodies[info.bodies.length - 1];
  signature = "";
  clearPlan();
  fillBodies();
  $("body").value = added.id;
  fillBodyFields();
  draw();
  return added;
}

function addMeshObject(file, data) {
  const where = partPlacement();
  const units = $("imp-units").value;
  const reply = engine({ op: "add_mesh_object", file, data, units, position: where.position, restOn: where.restOn });
  if (!reply.ok) { showImport("alert", "No se pudo añadir la pieza", reply.error, []); return; }
  const added = adoptPart(reply);
  showImport("ok", "Pieza añadida: " + added.name, "Malla colocada " + (where.onTable ? "sobre la mesa" : "en el suelo") + ". Se agarrará por su superficie. Revisa su masa y fricción en «Pieza».", reply.data.warnings);
}

function addObject(text, files) {
  const where = partPlacement();
  const reply = engine({ op: "add_object", text, files, position: where.position, restOn: where.restOn });
  if (!reply.ok) { showImport("alert", "No se pudo añadir el objeto", reply.error, []); return; }
  const added = adoptPart(reply);
  showImport("ok", "Pieza añadida: " + added.name, "Colocada " + (where.onTable ? "sobre la mesa" : "en el suelo") + ". Ajusta su posición, masa y fricción en «Pieza».", reply.data.warnings);
}

function wire() {
  $("body").addEventListener("change", () => { fillBodyFields(); clearPlan(); draw(); });
  $("btn-plan").addEventListener("click", runPlan);
  $("btn-import").addEventListener("click", importFiles);
  $("btn-sample").addEventListener("click", () => {
    $("imp-gripper").checked = true; $("imp-kind").value = "parallel"; $("imp-cell").checked = true; $("imp-eef").value = "";
    importRobot("urdf", SAMPLES.ur5.urdf, SAMPLES.ur5.files, UR5_READY);
  });
  $("btn-apply").addEventListener("click", () => { if (applyBody()) { clearPlan(); fillBodyFields(); draw(); } });
  const loadCell = () => {
    const reply = engine({ op: "load_demo", gripper: $("cell-gripper").value, obstacles: $("cell-obstacles").value });
    if (!reply.ok) return;
    $("edit-error").hidden = true;
    $("import-result").hidden = true;
    adoptRobot(reply.data, null);
  };
  $("btn-reset").addEventListener("click", loadCell);
  $("cell-gripper").addEventListener("change", loadCell);
  $("cell-obstacles").addEventListener("change", loadCell);
  $("btn-cycle").addEventListener("click", runCycle);
  $("btn-cycle-play").addEventListener("click", playCycle);
  document.querySelectorAll("[data-export]").forEach((b) => b.addEventListener("click", () => exportCycle(b.dataset.export)));
  $("btn-home").addEventListener("click", () => { stopPlayback(); stopCycle(); values = home.slice(); carry = null; grasping = null; placed = null; setPhaseLabel(plan ? PHASES[0] : "sin plan"); draw(); });
  $("btn-play").addEventListener("click", play);
  $("btn-defaults").addEventListener("click", () => {
    Object.entries(DEFAULTS).forEach(([id, value]) => { $(id).value = value; });
    $("o-fc").checked = true; $("o-measure").value = "wrench_space"; $("o-generator").value = "auto";
    noteParams();
  });
  $("params").addEventListener("input", noteParams);
  $("params").addEventListener("change", noteParams);
  document.querySelectorAll(".phase").forEach((b) => b.addEventListener("click", () => { stopPlayback(); goToPhase(Number(b.dataset.phase)); }));
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  document.querySelectorAll("[data-layer]").forEach((b) => b.addEventListener("click", () => {
    const name = b.dataset.layer;
    layers[name] = !layers[name];
    b.setAttribute("aria-pressed", String(layers[name]));
    showMarkers(); showCandidates();
    if (info) draw();
  }));
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const retheme = () => { if (scene) { applyTheme(); showMarkers(); showCandidates(); draw(); } };
  media.addEventListener("change", retheme);
  new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}

wire();
boot().catch((error) => {
  $("engine-state").textContent = "no disponible";
  const box = document.createElement("div");
  box.className = "fallback";
  box.textContent = "No se pudo iniciar el motor: " + error.message;
  $("bench").replaceWith(box);
});
