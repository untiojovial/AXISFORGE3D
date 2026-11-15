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
};
const PHASES = ["Inicio", "Pre-agarre", "Agarre (abierta)", "Cierre", "Elevación"];

let engine = null;        // (request) => reply
let info = null;          // last description
let home = null;          // configuration at load
let values = null;        // configuration being shown
let carry = null;         // part carried by the gripper, if any
let grasping = null;      // part the jaws are closed on but have not lifted
let report = null;        // last plan report
let plan = null;          // selected plan
let playing = 0;          // animation token; 0 when idle
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

function geometryFor(shape) {
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
    meshes.forEach((m) => { scene.remove(m); m.geometry.dispose(); m.material.dispose(); });
    meshes = drawables.map((d) => {
      const material = d.visual
        ? new THREE.MeshStandardMaterial({ envMapIntensity: 0.9 })
        : new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false, wireframe: false });
      const mesh = new THREE.Mesh(geometryFor(d.shape), material);
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
function draw() {
  const request = { op: "snapshot", values };
  if (carry) request.carry = carry;
  else if (grasping) request.grasping = grasping;
  const reply = engine(request);
  const flag = $("contact-flag");
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
  const jaws = info.joints.map((j, i) => (j.role === "end_effector" ? values[i] : null)).filter((v) => v !== null);
  $("opening").innerHTML = jaws.length ? fmt(jaws.reduce((a, b) => a + b, 0) * 1000) + ' <span class="gauge__unit">mm</span>' : "—";
  $("opening-note").textContent = carry ? "sujetando la pieza" : "entre mordazas";

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
  $("dof-note").textContent = info.axisCount + " ejes + " + (info.joints.length - info.axisCount) + " mordazas";
  $("gripper-note").textContent = info.gripper ? "pinza · " + fmt(info.gripperMaxOpening * 1000, 0) + " mm" : "sin pinza";
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
  $("ext-note").textContent = reply.data.length + " registradas";
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
  return options;
}

const DEFAULTS = { "o-nr": "3", "o-minq": "0", "o-mu": "", "o-cone": "8", "o-force": "1", "o-torsion": "0", "o-standoff": "80", "o-lift": "100", "o-jaw": "5", "o-max": "" };
function noteParams() {
  const changed = Object.entries(DEFAULTS).some(([id, value]) => $(id).value !== value) || !$("o-fc").checked || $("o-measure").value !== "wrench_space";
  $("params-note").textContent = changed ? "modificados" : "por defecto";
}

/* =================================================================== plan */
function describePlan(p) {
  const a = p.candidate.approach;
  const way = a.z < -0.7 ? "Desde arriba" : a.z > 0.7 ? "Desde abajo" : "Lateral";
  return way + (p.candidate.label.includes("shallow") ? ", agarre poco profundo" : "");
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
  verdict.className = "verdict " + (n ? "verdict--ok" : "verdict--warn");
  $("plan-head").textContent = n ? (n === 1 ? "1 agarre viable" : n + " agarres viables") : "Ningún agarre viable";
  $("plan-detail").textContent = "Se examinaron " + report.candidatesExamined + " de " + report.candidatesTotal + " agarres candidatos para «" + b.name + "»; " + report.rejections.length + " descartados." +
    (n ? "" : " Prueba a mover la pieza, aligerarla, aumentar la fricción o relajar los parámetros.");

  const host = $("plans");
  host.textContent = "";
  report.plans.forEach((p, i) => {
    const el = document.createElement("button");
    el.type = "button"; el.className = "plan";
    el.innerHTML = '<span class="plan__name"></span><span class="plan__metrics">' +
      "<span>ε <b>" + fmt(p.quality.quality, 4) + "</b></span>" +
      '<span class="pill ' + (p.quality.forceClosure ? "pill--ok" : "pill--warn") + '">' + (p.quality.forceClosure ? "cierre de fuerza" : "sin cierre") + "</span>" +
      "<span>contactos <b>" + p.quality.contactCount + "</b></span>" +
      "<span>ancho <b>" + fmt(p.candidate.width * 1000) + " mm</b></span>" +
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

/* ================================================================= wiring */
function wire() {
  $("body").addEventListener("change", () => { fillBodyFields(); clearPlan(); draw(); });
  $("btn-plan").addEventListener("click", runPlan);
  $("btn-apply").addEventListener("click", () => { if (applyBody()) { clearPlan(); fillBodyFields(); draw(); } });
  $("btn-reset").addEventListener("click", () => {
    const reply = engine({ op: "load_demo" });
    if (!reply.ok) return;
    info = reply.data; values = home.slice();
    $("edit-error").hidden = true;
    clearPlan(); fillBodies(); draw();
  });
  $("btn-home").addEventListener("click", () => { stopPlayback(); values = home.slice(); carry = null; grasping = null; setPhaseLabel(plan ? PHASES[0] : "sin plan"); draw(); });
  $("btn-play").addEventListener("click", play);
  $("btn-defaults").addEventListener("click", () => {
    Object.entries(DEFAULTS).forEach(([id, value]) => { $(id).value = value; });
    $("o-fc").checked = true; $("o-measure").value = "wrench_space";
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
