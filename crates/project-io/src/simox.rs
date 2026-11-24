//! Import of Simox robot and manipulation-object XML.
//!
//! The semantics follow Simox's own reader (`VirtualRobot/XML/RobotIO.cpp`
//! and `BaseIO.cpp`), read from its source rather than from memory. No Simox
//! code is used. The points that decide the numbers:
//!
//! - Tag and attribute names are case-insensitive.
//! - Lengths default to millimetres and angles to radians; `units`, `unit`,
//!   `unitsLength` and `unitsAngle` change that (`m`/`meter`,
//!   `deg`/`degree`). Masses default to kilograms (`g`, `t` accepted).
//! - A `<Transform>` is the product of its children in document order:
//!   `Translation`, `rollpitchyaw`/`rpy` (R = Rz(yaw) Ry(pitch) Rx(roll)),
//!   `Quaternion`, `Matrix3x3`, `Matrix4x4` and `DH`
//!   (Rz(theta) Tz(d) Tx(a) Rx(alpha)).
//! - A node's pose is `parent * Transform * motion`, where the motion is a
//!   rotation by `value + offset` about the normalized `axis` (default +Z), or
//!   a translation by `(value + offset) * TranslationDirection`, which Simox
//!   does **not** normalize.
//! - Missing joint limits default to plus and minus pi, in the joint's own
//!   units — for a prismatic joint that is about 3.14 mm.
//! - Mesh files are drawn in metres; primitive sizes are millimetres unless a
//!   length unit is given. Cylinders and capsules lie along the primitive's Y
//!   axis, and a capsule's height is its straight section. Each primitive has
//!   its own transform; they are not chained.
//! - `<Endeffector>` actors are the end effector's joints, and its `tcp` and
//!   `gcp` nodes become tool frames.
//!
//! What is not carried over is reported: hemisphere and four-bar joints and
//! robots split across files are refused; propagated joint values, velocity
//! and torque limits, sensors, preshapes, grasp sets and inertia matrices are
//! listed as warnings.

use std::collections::{BTreeMap, BTreeSet};

use axisforge_domain::{
    templates::stable_id, BodyKind, GeometryInstance, GeometryRole, Inertial, Joint, JointLimits,
    Link, PhysicalProperties, Quat, Resource, Shape, ToolFrame, Transform, Vec3, WorldBody,
};
use roxmltree::{Document, Node};

use crate::import::{
    ImportError, ImportOptions, Imported, MeshSource, MeshStore, RobotAssembly,
    MAX_DESCRIPTION_BYTES,
};

type X<'a> = Node<'a, 'a>;

fn is(node: X<'_>, tag: &str) -> bool {
    node.is_element() && node.tag_name().name().eq_ignore_ascii_case(tag)
}

fn child<'a>(node: X<'a>, tag: &str) -> Option<X<'a>> {
    node.children().find(|c| is(*c, tag))
}

fn children<'a>(node: X<'a>, tag: &'a str) -> impl Iterator<Item = X<'a>> {
    node.children().filter(move |c| is(*c, tag))
}

fn attr<'a>(node: X<'a>, name: &str) -> Option<&'a str> {
    node.attributes()
        .find(|a| a.name().eq_ignore_ascii_case(name))
        .map(|a| a.value())
}

fn float(node: X<'_>, name: &str, what: &str) -> Result<f64, ImportError> {
    let text = attr(node, name).ok_or_else(|| {
        ImportError::Malformed(format!(
            "{what}: <{}> needs a `{name}` attribute",
            node.tag_name().name()
        ))
    })?;
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| {
            ImportError::Malformed(format!("{what}: `{name}` is `{text}`, not a finite number"))
        })
}

fn optional_float(node: X<'_>, name: &str, what: &str) -> Result<Option<f64>, ImportError> {
    attr(node, name)
        .map(|_| float(node, name, what))
        .transpose()
}

/// Every unit named on an element, lower-cased.
fn units(node: X<'_>) -> Vec<String> {
    node.attributes()
        .filter(|a| {
            let n = a.name().to_ascii_lowercase();
            matches!(
                n.as_str(),
                "unit" | "units" | "unitslength" | "unitsangle" | "unitsweight"
            )
        })
        .map(|a| a.value().trim().to_ascii_lowercase())
        .collect()
}

fn is_metre(u: &str) -> bool {
    matches!(u, "m" | "meter")
}
fn is_millimetre(u: &str) -> bool {
    matches!(u, "mm" | "millimeter")
}
fn is_degree(u: &str) -> bool {
    matches!(u, "deg" | "degree")
}
fn is_radian(u: &str) -> bool {
    matches!(u, "rad" | "radian")
}

/// Metres per unit of a length attribute on `node`: Simox's millimetre unless
/// metres are named.
fn length_scale(node: X<'_>) -> f64 {
    if units(node).iter().any(|u| is_metre(u)) {
        1.0
    } else {
        1.0e-3
    }
}

/// Radians per unit of an angle attribute on `node`.
fn angle_scale(node: X<'_>) -> f64 {
    if units(node).iter().any(|u| is_degree(u)) {
        std::f64::consts::PI / 180.0
    } else {
        1.0
    }
}

/// A rotation from a row-major matrix, refusing anything that is not one.
#[allow(clippy::many_single_char_names)] // Matrix and quaternion components keep their textbook names.
fn rotation_from_rows(m: [[f64; 3]; 3], what: &str) -> Result<Quat, ImportError> {
    // XML values are often written to four or five digits, so orthonormality
    // is checked loosely; the quaternion is normalized afterwards.
    const LOOSE: f64 = 1.0e-3;
    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = (0..3).map(|k| m[k][i] * m[k][j]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > LOOSE {
                return Err(ImportError::Malformed(format!(
                    "{what}: the matrix is not a rotation"
                )));
            }
        }
    }
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det < 0.0 {
        return Err(ImportError::Malformed(format!(
            "{what}: the matrix is a reflection"
        )));
    }
    // Shepperd: take the largest of the four candidate denominators.
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (x, y, z, w) = if trace > m[0][0].max(m[1][1]).max(m[2][2]) {
        let s = (1.0 + trace).sqrt() * 2.0;
        (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        )
    } else if m[0][0] >= m[1][1] && m[0][0] >= m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        (
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    } else if m[1][1] >= m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        (
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        (
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        )
    };
    Ok(Quat::new(x, y, z, w).normalized(what)?)
}

fn rows<const N: usize>(node: X<'_>, what: &str) -> Result<[[f64; N]; N], ImportError> {
    let mut m = [[0.0; N]; N];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
        if let Some(r) = child(node, &format!("row{}", i + 1)) {
            for (j, value) in row.iter_mut().enumerate() {
                *value = float(r, &format!("c{}", j + 1), what)?;
            }
        }
    }
    Ok(m)
}

/// The product of a `<Transform>`'s children, in metres and radians.
fn transform(
    node: Option<X<'_>>,
    what: &str,
    warnings: &mut Vec<String>,
) -> Result<Transform, ImportError> {
    let Some(node) = node else {
        return Ok(Transform::IDENTITY);
    };
    let mut pose = Transform::IDENTITY;
    for step in node.children().filter(Node::is_element) {
        let tag = step.tag_name().name().to_ascii_lowercase();
        let local = match tag.as_str() {
            "translation" | "trans" => {
                let k = length_scale(step);
                Transform::from_translation(Vec3::new(
                    float(step, "x", what)? * k,
                    float(step, "y", what)? * k,
                    float(step, "z", what)? * k,
                ))
            }
            "rollpitchyaw" | "rpy" => {
                let k = angle_scale(step);
                Transform::from_rotation(Quat::from_euler_rpy(
                    float(step, "roll", what)? * k,
                    float(step, "pitch", what)? * k,
                    float(step, "yaw", what)? * k,
                ))
            }
            "quaternion" | "quat" => Transform::from_rotation(
                Quat::new(
                    float(step, "x", what)?,
                    float(step, "y", what)?,
                    float(step, "z", what)?,
                    float(step, "w", what)?,
                )
                .normalized(what)?,
            ),
            "matrix3x3" => {
                let m = rows::<3>(step, what)?;
                Transform::from_rotation(rotation_from_rows(m, what)?)
            }
            "matrix4x4" => {
                let m = rows::<4>(step, what)?;
                let k = length_scale(step);
                let r = [
                    [m[0][0], m[0][1], m[0][2]],
                    [m[1][0], m[1][1], m[1][2]],
                    [m[2][0], m[2][1], m[2][2]],
                ];
                Transform::new(
                    Vec3::new(m[0][3] * k, m[1][3] * k, m[2][3] * k),
                    rotation_from_rows(r, what)?,
                )
            }
            "dh" => dh(step, what)?,
            other => {
                warnings.push(format!(
                    "{what}: unknown transform step <{other}> was ignored, as Simox ignores it"
                ));
                Transform::IDENTITY
            }
        };
        pose = pose.compose(&local);
    }
    Ok(pose)
}

/// Rz(theta) Tz(d) Tx(a) Rx(alpha), as Simox's `DHParameter::transformation`.
fn dh(node: X<'_>, what: &str) -> Result<Transform, ImportError> {
    let named = units(node);
    let angle = if named.iter().any(|u| is_degree(u)) {
        std::f64::consts::PI / 180.0
    } else {
        1.0
    };
    let length = if named.iter().any(|u| is_metre(u)) {
        1.0
    } else {
        1.0e-3
    };
    let value = |name: &str| -> Result<f64, ImportError> {
        Ok(optional_float(node, name, what)?.unwrap_or(0.0))
    };
    let theta = value("theta")? * angle;
    let alpha = value("alpha")? * angle;
    let rz = Transform::from_rotation(Quat::from_axis_angle(&Vec3::Z, theta)?);
    let tz = Transform::from_translation(Vec3::new(0.0, 0.0, value("d")? * length));
    let tx = Transform::from_translation(Vec3::new(value("a")? * length, 0.0, 0.0));
    let rx = Transform::from_rotation(Quat::from_axis_angle(&Vec3::X, alpha)?);
    Ok(rz.compose(&tz).compose(&tx).compose(&rx))
}

/// The geometry inside a `<Visualization>` or `<CollisionModel>`.
struct GeometryReader<'s, 'm> {
    namespace: String,
    meshes: &'s mut MeshStore<'m>,
    warnings: &'s mut Vec<String>,
    count: usize,
}

impl GeometryReader<'_, '_> {
    fn read(
        &mut self,
        node: X<'_>,
        owner: &str,
        role: GeometryRole,
    ) -> Result<Vec<GeometryInstance>, ImportError> {
        let mut out = Vec::new();
        let what = format!("`{owner}` {}", node.tag_name().name());
        if let Some(file) = child(node, "file") {
            let reference = file
                .text()
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .ok_or_else(|| ImportError::Malformed(format!("{what}: <File> names no file")))?;
            if let Some(parts) = self.meshes.parts(reference, role) {
                if attr(file, "boundingbox").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
                    self.warnings.push(format!(
                        "{what}: Simox collides with the bounding box of `{reference}`; its convex hull is used instead"
                    ));
                }
                for part in parts {
                    let mut geometry = self.instance(
                        Shape::Mesh {
                            resource_id: part.resource_id,
                            scale: Vec3::new(1.0, 1.0, 1.0),
                        },
                        Transform::IDENTITY,
                        part.role,
                    );
                    if let Some(hex) = part.color {
                        geometry.material.color = hex;
                    }
                    out.push(geometry);
                }
            }
        }
        if let Some(primitives) = child(node, "primitives") {
            for p in primitives.children().filter(Node::is_element) {
                let tag = p.tag_name().name().to_ascii_lowercase();
                let k = length_scale(p);
                let pose = transform(child(p, "transform"), &what, self.warnings)?;
                // Simox's cylinders and capsules lie along Y; ours along Z.
                let along_y = Transform::from_rotation(Quat::from_axis_angle(
                    &Vec3::X,
                    -std::f64::consts::FRAC_PI_2,
                )?);
                let (shape, origin) = match tag.as_str() {
                    "box" => (
                        Shape::Box {
                            size: Vec3::new(
                                float(p, "width", &what)? * k,
                                float(p, "height", &what)? * k,
                                float(p, "depth", &what)? * k,
                            ),
                        },
                        pose,
                    ),
                    "sphere" => (
                        Shape::Sphere {
                            radius: float(p, "radius", &what)? * k,
                        },
                        pose,
                    ),
                    "cylinder" => (
                        Shape::Cylinder {
                            radius: float(p, "radius", &what)? * k,
                            length: float(p, "height", &what)? * k,
                        },
                        pose.compose(&along_y),
                    ),
                    "capsule" => (
                        Shape::Capsule {
                            radius: float(p, "radius", &what)? * k,
                            length: float(p, "height", &what)? * k,
                        },
                        pose.compose(&along_y),
                    ),
                    other => {
                        self.warnings.push(format!(
                            "{what}: <{other}> primitives are not supported and were left out"
                        ));
                        continue;
                    }
                };
                out.push(self.instance(shape, origin, role));
            }
        }
        Ok(out)
    }

    fn instance(
        &mut self,
        shape: Shape,
        origin: Transform,
        role: GeometryRole,
    ) -> GeometryInstance {
        self.count += 1;
        let name = format!(
            "{} {}",
            if role == GeometryRole::Collision {
                "collision"
            } else {
                "visual"
            },
            self.count
        );
        GeometryInstance::visual(stable_id(&self.namespace, "geometry", &name), name, shape)
            .with_origin(origin)
            .with_role(role)
    }

    /// Visualization and collision model of an element, with
    /// `UseAsCollisionModel` honoured.
    fn element(&mut self, node: X<'_>, owner: &str) -> Result<Vec<GeometryInstance>, ImportError> {
        let mut out = Vec::new();
        if let Some(visual) = child(node, "visualization") {
            let enabled = attr(visual, "enable").is_none_or(|v| !v.eq_ignore_ascii_case("false"));
            let shared = child(visual, "usecollisionmodel").is_some()
                || child(visual, "useascollisionmodel").is_some();
            let role = match (enabled, shared) {
                (true, true) => Some(GeometryRole::Both),
                (true, false) => Some(GeometryRole::Visual),
                (false, true) => Some(GeometryRole::Collision),
                (false, false) => None,
            };
            if let Some(role) = role {
                out.extend(self.read(visual, owner, role)?);
            }
        }
        if let Some(collision) = child(node, "collisionmodel") {
            out.extend(self.read(collision, owner, GeometryRole::Collision)?);
        }
        Ok(out)
    }
}

fn mass(node: X<'_>, what: &str) -> Result<Option<f64>, ImportError> {
    let Some(physics) = child(node, "physics") else {
        return Ok(None);
    };
    let Some(m) = child(physics, "mass") else {
        return Ok(None);
    };
    let value = float(m, "value", what)?;
    let named = units(m);
    let k = if named.iter().any(|u| matches!(u.as_str(), "g" | "gram")) {
        1.0e-3
    } else if named.iter().any(|u| matches!(u.as_str(), "t" | "ton")) {
        1.0e3
    } else {
        1.0
    };
    Ok(Some(value * k))
}

/// Joint details read from a `<Joint>` element.
struct JointSpec {
    kind: String,
    axis: Vec3,
    direction: Vec3,
    offset: f64,
    limits: (f64, f64),
}

fn joint_spec(
    node: X<'_>,
    what: &str,
    warnings: &mut Vec<String>,
) -> Result<Option<JointSpec>, ImportError> {
    let Some(joint) = child(node, "joint") else {
        return Ok(None);
    };
    let kind = if let Some(t) = attr(joint, "type") {
        t.trim().to_ascii_lowercase()
    } else {
        warnings.push(format!(
            "{what}: <Joint> has no type; Simox treats it as fixed, and so does this import"
        ));
        "fixed".to_owned()
    };
    if !matches!(kind.as_str(), "revolute" | "prismatic" | "fixed") {
        return Err(ImportError::Unsupported(format!(
            "{what}: `{kind}` joints are not supported; revolute, prismatic and fixed are"
        )));
    }
    let mut axis = Vec3::Z;
    if let Some(a) = child(joint, "axis") {
        axis = Vec3::new(
            float(a, "x", what)?,
            float(a, "y", what)?,
            float(a, "z", what)?,
        );
    }
    let mut direction = Vec3::Z;
    if kind == "prismatic" {
        let d = child(joint, "translationdirection").ok_or_else(|| {
            ImportError::Malformed(format!(
                "{what}: a prismatic joint needs <TranslationDirection>"
            ))
        })?;
        direction = Vec3::new(
            float(d, "x", what)?,
            float(d, "y", what)?,
            float(d, "z", what)?,
        );
    }
    let offset = optional_float(joint, "offset", what)?.unwrap_or(0.0);
    let mut limits = (-std::f64::consts::PI, std::f64::consts::PI);
    if let Some(l) = child(joint, "limits") {
        let named = units(l);
        let length = named.iter().any(|u| is_metre(u) || is_millimetre(u));
        let k = if named.iter().any(|u| is_degree(u)) {
            std::f64::consts::PI / 180.0
        } else if named.iter().any(|u| is_metre(u)) {
            1000.0
        } else {
            1.0
        };
        let (missing_lo, missing_hi) = if length {
            (-1000.0, 1000.0)
        } else {
            (-std::f64::consts::PI, std::f64::consts::PI)
        };
        limits = (
            optional_float(l, "lo", what)?.map_or(missing_lo, |v| v * k),
            optional_float(l, "hi", what)?.map_or(missing_hi, |v| v * k),
        );
        if attr(l, "limitless").is_some_and(|v| v.eq_ignore_ascii_case("true"))
            && kind == "revolute"
        {
            warnings.push(format!(
                "{what} is limitless; it was given limits of plus and minus one full turn"
            ));
            limits = (-std::f64::consts::TAU, std::f64::consts::TAU);
        }
        let _ = is_radian;
    } else if kind != "fixed" {
        warnings.push(format!(
            "{what}: no <Limits>; Simox's default of plus and minus pi {} was used",
            if kind == "prismatic" {
                "millimetres"
            } else {
                "radians"
            }
        ));
    }
    let mut dropped = Vec::new();
    for tag in [
        "maxvelocity",
        "maxacceleration",
        "maxtorque",
        "propagatejointvalue",
        "scalevisualization",
    ] {
        if child(joint, tag).is_some() {
            dropped.push(tag);
        }
    }
    if !dropped.is_empty() {
        warnings.push(format!("{what}: not imported: {}", dropped.join(", ")));
    }
    Ok(Some(JointSpec {
        kind,
        axis,
        direction,
        offset,
        limits,
    }))
}

/// Builds the joint from `parent_id` into the link of the node `name`.
fn build_joint(
    robot: &str,
    name: &str,
    parent_id: &str,
    child_id: &str,
    origin: Transform,
    spec: Option<JointSpec>,
    warnings: &mut Vec<String>,
) -> Result<Joint, ImportError> {
    let id = stable_id(robot, "joint", name);
    let what = format!("node `{name}`");
    let Some(spec) = spec.filter(|s| s.kind != "fixed") else {
        return Ok(Joint::fixed(id, name, parent_id, child_id, origin));
    };
    if spec.kind == "revolute" {
        let axis = spec.axis.normalized(&what)?;
        // value + offset about the same axis: the offset folds into the origin.
        let origin = origin.compose(&Transform::from_rotation(Quat::from_axis_angle(
            &axis,
            spec.offset,
        )?));
        let mut joint = Joint::revolute(id, name, parent_id, child_id, origin, axis, 0.0)?;
        joint.limits = JointLimits::new(spec.limits.0, spec.limits.1, name)?;
        return Ok(joint);
    }
    // Prismatic: Simox moves by (value + offset) * direction, in millimetres,
    // without normalizing the direction.
    let travel = spec.direction.norm();
    let unit = spec.direction.normalized(&what)?;
    if (travel - 1.0).abs() > 1.0e-9 {
        warnings.push(format!(
            "{what}: the translation direction has length {travel}; joint values here are metres \
             of travel, which is {travel} times Simox's value in millimetres / 1000"
        ));
    }
    let k = travel * 1.0e-3;
    let origin = origin.compose(&Transform::from_translation(unit.scaled(spec.offset * k)));
    Ok(Joint::prismatic(
        id,
        name,
        parent_id,
        child_id,
        origin,
        unit,
        spec.limits.0 * k,
        spec.limits.1 * k,
    )?)
}

/// Imports a Simox robot description.
///
/// `meshes` supplies the files `<File>` elements refer to, by the path
/// exactly as written (relative to the description).
///
/// # Errors
/// When the text is not a Simox robot, uses something unsupported, or
/// describes a robot that does not validate.
#[allow(clippy::too_many_lines)] // One pass over one document; split, it would only shuffle state.
pub fn import_simox_robot(
    text: &str,
    meshes: &dyn MeshSource,
    options: &ImportOptions,
) -> Result<Imported, ImportError> {
    if text.len() > MAX_DESCRIPTION_BYTES {
        return Err(ImportError::TooLarge { size: text.len() });
    }
    let document = Document::parse(text).map_err(|e| ImportError::Malformed(e.to_string()))?;
    let root = document.root_element();
    if !is(root, "robot") {
        return Err(ImportError::Malformed(format!(
            "the root element is <{}>, not <Robot>: this is not a Simox robot",
            root.tag_name().name()
        )));
    }
    let name = attr(root, "standardname")
        .or_else(|| attr(root, "type"))
        .filter(|n| !n.trim().is_empty())
        .ok_or_else(|| ImportError::Malformed("<Robot> needs a `Type` attribute".to_owned()))?
        .to_owned();
    let root_name = attr(root, "rootnode")
        .filter(|n| !n.trim().is_empty())
        .ok_or_else(|| ImportError::Malformed("<Robot> needs a `RootNode` attribute".to_owned()))?;

    let mut warnings = Vec::new();
    let mut store = MeshStore::new(&name, meshes);
    let mut nodes: BTreeMap<String, X<'_>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for node in children(root, "robotnode") {
        let n = attr(node, "name")
            .ok_or_else(|| ImportError::Malformed("<RobotNode> needs a `name`".to_owned()))?;
        if nodes.insert(n.to_owned(), node).is_some() {
            return Err(ImportError::Malformed(format!(
                "robot node `{n}` is defined twice"
            )));
        }
        order.push(n.to_owned());
        if child(node, "childfromrobot").is_some() {
            return Err(ImportError::Unsupported(format!(
                "node `{n}` includes another robot file (<ChildFromRobot>); import a single-file model"
            )));
        }
    }
    if !nodes.contains_key(root_name) {
        return Err(ImportError::Malformed(format!(
            "RootNode `{root_name}` is not a <RobotNode>"
        )));
    }

    let mut parent_of: BTreeMap<String, String> = BTreeMap::new();
    for n in &order {
        for c in children(nodes[n], "child") {
            let target = attr(c, "name").ok_or_else(|| {
                ImportError::Malformed(format!("node `{n}`: <Child> needs a `name`"))
            })?;
            if !nodes.contains_key(target) {
                return Err(ImportError::Malformed(format!(
                    "node `{n}` names unknown child `{target}`"
                )));
            }
            if let Some(previous) = parent_of.insert(target.to_owned(), n.clone()) {
                return Err(ImportError::Malformed(format!(
                    "node `{target}` is the child of both `{previous}` and `{n}`"
                )));
            }
        }
    }
    let unreached: Vec<&String> = order
        .iter()
        .filter(|n| *n != root_name && !parent_of.contains_key(*n))
        .collect();
    if !unreached.is_empty() {
        warnings.push(format!(
            "{} node(s) are not connected to the root and were left out: {}",
            unreached.len(),
            unreached
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let mut links = Vec::new();
    let mut joints = Vec::new();
    let mut bbox_com = Vec::new();
    let mut base_pose = Transform::IDENTITY;
    let mut root_link = stable_id(&name, "link", root_name);
    for n in order
        .iter()
        .filter(|n| *n == root_name || parent_of.contains_key(*n))
    {
        let node = nodes[n];
        let what = format!("node `{n}`");
        let id = stable_id(&name, "link", n);
        let mut reader = GeometryReader {
            namespace: format!("{name}::{n}"),
            meshes: &mut store,
            warnings: &mut warnings,
            count: 0,
        };
        let geometries = reader.element(node, n)?;
        let mut link = Link::new(&id, n.as_str());
        link.geometries = geometries;
        if let Some(m) = mass(node, &what)? {
            link.inertial = Inertial::point_mass(m);
            if child(node, "physics")
                .and_then(|p| child(p, "com"))
                .and_then(|c| attr(c, "location"))
                .is_some_and(|l| l.eq_ignore_ascii_case("visualizationbboxcenter"))
            {
                bbox_com.push(n.clone());
            }
        }
        links.push(link);
        for tag in ["sensor"] {
            if child(node, tag).is_some() {
                warnings.push(format!("{what}: <{tag}> elements are not imported"));
            }
        }
        let origin = transform(child(node, "transform"), &what, &mut warnings)?;
        let spec = joint_spec(node, &what, &mut warnings)?;
        if let Some(parent) = parent_of.get(n) {
            joints.push(build_joint(
                &name,
                n,
                &stable_id(&name, "link", parent),
                &id,
                origin,
                spec,
                &mut warnings,
            )?);
        } else if spec.as_ref().is_some_and(|s| s.kind != "fixed") {
            // A moving root: give it something fixed to move against.
            let world = stable_id(&name, "link", "world");
            links.push(Link::new(&world, format!("{name} world")));
            joints.push(build_joint(
                &name,
                n,
                &world,
                &id,
                origin,
                spec,
                &mut warnings,
            )?);
            root_link = world;
        } else {
            base_pose = origin;
        }
    }
    if !bbox_com.is_empty() {
        warnings.push(format!(
            "the centre of mass of {} node(s) is given as the visualization's bounding-box centre; \
             the node origin was used instead: {}",
            bbox_com.len(),
            bbox_com.join(", ")
        ));
    }

    // End effectors: their actors are end-effector joints; tcp and gcp are frames.
    let mut frames = Vec::new();
    let mut default_frame = None;
    let mut end_effector_joints = BTreeSet::new();
    let eefs: Vec<X<'_>> = children(root, "endeffector").collect();
    for eef in &eefs {
        let eef_name = attr(*eef, "name").unwrap_or("end effector");
        for actor in children(*eef, "actor") {
            for node in children(actor, "node") {
                if let Some(joint) = attr(node, "name") {
                    end_effector_joints.insert(joint.to_owned());
                }
            }
        }
        for (key, label) in [("tcp", "tcp"), ("gcp", "gcp")] {
            if let Some(target) = attr(*eef, key) {
                if !nodes.contains_key(target) {
                    return Err(ImportError::Malformed(format!(
                        "end effector `{eef_name}`: {key} `{target}` is not a node"
                    )));
                }
                let frame_name = format!("{eef_name} {label}");
                frames.push(ToolFrame::new(
                    stable_id(&name, "frame", &frame_name),
                    &frame_name,
                    stable_id(&name, "link", target),
                    Transform::IDENTITY,
                ));
                if key == "tcp" && default_frame.is_none() {
                    default_frame = Some(frame_name);
                }
            }
        }
        if child(*eef, "preshape").is_some() {
            warnings.push(format!(
                "end effector `{eef_name}`: preshapes are not imported"
            ));
        }
    }
    if eefs.len() > 1 {
        warnings.push(format!(
            "the robot has {} end effectors; the first one's tcp is the default tool frame",
            eefs.len()
        ));
    }
    let mut ignored: BTreeMap<String, usize> = BTreeMap::new();
    for c in root.children().filter(Node::is_element) {
        let tag = c.tag_name().name().to_ascii_lowercase();
        if !matches!(tag.as_str(), "robotnode" | "endeffector") {
            *ignored.entry(tag).or_insert(0) += 1;
        }
    }
    for (tag, count) in ignored {
        warnings.push(format!("{count} <{tag}> element(s) were ignored"));
    }
    store.report(&mut warnings);

    RobotAssembly {
        name,
        root_link,
        links,
        joints,
        base_pose,
        frames,
        default_frame,
        end_effector_joints,
        resources: store.resources,
        warnings,
    }
    .finish(options)
}

/// A Simox manipulation object, ready to place in a project.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedObject {
    /// The object as a manipulable world body, at the origin.
    pub body: WorldBody,
    /// Mesh resources its geometry refers to; add them to the project.
    pub resources: Vec<Resource>,
    /// Everything that could not be carried over exactly.
    pub warnings: Vec<String>,
}

/// Imports a Simox `<ManipulationObject>`.
///
/// # Errors
/// When the text is not a manipulation object or none of its geometry can be
/// imported.
pub fn import_simox_object(
    text: &str,
    meshes: &dyn MeshSource,
) -> Result<ImportedObject, ImportError> {
    if text.len() > MAX_DESCRIPTION_BYTES {
        return Err(ImportError::TooLarge { size: text.len() });
    }
    let document = Document::parse(text).map_err(|e| ImportError::Malformed(e.to_string()))?;
    let root = document.root_element();
    if !is(root, "manipulationobject") {
        return Err(ImportError::Malformed(format!(
            "the root element is <{}>, not <ManipulationObject>",
            root.tag_name().name()
        )));
    }
    let name = attr(root, "name")
        .filter(|n| !n.trim().is_empty())
        .unwrap_or("object")
        .to_owned();
    let mut warnings = Vec::new();
    let mut store = MeshStore::new(&name, meshes);
    let mut reader = GeometryReader {
        namespace: format!("object::{name}"),
        meshes: &mut store,
        warnings: &mut warnings,
        count: 0,
    };
    let geometries = reader.element(root, &name)?;
    let mass = mass(root, &format!("object `{name}`"))?;
    let defaults = PhysicalProperties::default();
    if mass.is_none() {
        warnings.push(format!(
            "no mass is given; {} kg was assumed",
            defaults.mass
        ));
    }
    warnings.push(format!(
        "Simox objects carry no friction coefficient; {} was assumed — set the real value before planning",
        defaults.friction_coefficient
    ));
    for tag in ["graspset", "affordances"] {
        let count = children(root, tag).count();
        if count > 0 {
            warnings.push(format!("{count} <{tag}> element(s) were not imported"));
        }
    }
    store.report(&mut warnings);
    if geometries.is_empty() {
        return Err(ImportError::Unsupported(format!(
            "object `{name}` has no geometry that could be imported: {}",
            if warnings.is_empty() {
                "it declares none".to_owned()
            } else {
                warnings.join("; ")
            }
        )));
    }
    let body = WorldBody {
        id: stable_id("object", "body", &name),
        name: name.clone(),
        kind: BodyKind::Manipulable,
        pose: Transform::IDENTITY,
        geometries,
        physical: PhysicalProperties {
            mass: mass.unwrap_or(defaults.mass),
            ..defaults
        },
    };
    body.validate()?;
    Ok(ImportedObject {
        body,
        resources: store.resources,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::NoMeshes;
    use axisforge_domain::{JointKind, JointRole};
    use axisforge_kinematics::{forward_kinematics, KinematicModel};

    fn robot(text: &str) -> Imported {
        import_simox_robot(text, &NoMeshes, &ImportOptions::default()).expect("imports")
    }

    /// World pose of each named node for named joint values (SI units).
    fn poses(imported: &Imported, values: &[(&str, f64)]) -> BTreeMap<String, Transform> {
        let robot = &imported.project.robot;
        let model = KinematicModel::from_robot(robot).unwrap();
        let mut q = vec![0.0; model.dof()];
        for (name, value) in values {
            let slot = (0..model.dof())
                .find(|i| model.joint_for_value(*i).unwrap().name == *name)
                .expect(name);
            q[slot] = *value;
        }
        let pose = forward_kinematics(&model, &q).unwrap();
        robot
            .links
            .iter()
            .map(|l| (l.name.clone(), pose.link(model.link_index(&l.id).unwrap())))
            .collect()
    }

    fn close(a: &Vec3, b: &Vec3) -> bool {
        a.distance(b) < 1.0e-12
    }

    #[test]
    fn a_prismatic_joint_moves_as_in_simoxs_own_test() {
        // Mirrors Simox's testRobotNodePrismaticTransformation: millimetres
        // in, expected poses taken from that test's assertions.
        let text = "<Robot Type='Demo' RootNode='Joint1'>
            <RobotNode name='Joint1'>
              <Transform><Translation x='100' y='0' z='0'/></Transform>
              <Joint type='prismatic'><Limits unit='mm' lo='-300' hi='500'/><TranslationDirection x='0' y='0' z='1'/></Joint>
              <Child name='Joint2'/>
            </RobotNode>
            <RobotNode name='Joint2'><Transform><Translation x='0' y='200' z='0'/></Transform></RobotNode>
          </Robot>";
        let imported = robot(text);
        let joint = imported.project.robot.joint_by_name("Joint1").unwrap();
        assert_eq!(joint.kind, JointKind::Prismatic);
        assert!(
            (joint.limits.lower + 0.3).abs() < 1e-15 && (joint.limits.upper - 0.5).abs() < 1e-15
        );
        for (value, z) in [(0.0, 0.0), (0.150, 0.150), (-0.300, -0.300)] {
            let p = poses(&imported, &[("Joint1", value)]);
            assert!(
                close(&p["Joint1"].translation, &Vec3::new(0.1, 0.0, z)),
                "{:?}",
                p["Joint1"]
            );
            assert!(
                close(&p["Joint2"].translation, &Vec3::new(0.1, 0.2, z)),
                "{:?}",
                p["Joint2"]
            );
        }
    }

    #[test]
    fn joint_directions_match_simoxs_own_test() {
        // Mirrors Simox's testJointDirectionsInCoordinateSystemIgnoreTranslation.
        let text = r#"<Robot Type="JointDirectionTest" RootNode="Root">
            <RobotNode name="Root"><Child name="Rev"/></RobotNode>
            <RobotNode name="Rev">
              <Transform><Translation x="100" y="-200" z="300"/><rollpitchyaw roll="0" pitch="0" yaw="90" units="degree"/></Transform>
              <Joint type="revolute"><axis x="1" y="0" z="0"/><limits lo="-90" hi="90" units="degree"/></Joint>
              <Child name="Pri"/>
            </RobotNode>
            <RobotNode name="Pri">
              <Transform><Translation x="50" y="60" z="70"/><rollpitchyaw roll="0" pitch="90" yaw="0" units="degree"/></Transform>
              <Joint type="prismatic"><translationdirection x="1" y="0" z="0"/><limits lo="-100" hi="100" units="mm"/></Joint>
            </RobotNode>
          </Robot>"#;
        let imported = robot(text);
        let (rev, pri) = (0.4, 0.020);
        let p = poses(&imported, &[("Rev", rev), ("Pri", pri)]);
        // The test's expectations: Rev's axis is Rz(90) X; Pri's direction is
        // Rz(90) Rx(0.4) Ry(90) X.
        let rz = Quat::from_axis_angle(&Vec3::Z, std::f64::consts::FRAC_PI_2).unwrap();
        let rx = Quat::from_axis_angle(&Vec3::X, rev).unwrap();
        let ry = Quat::from_axis_angle(&Vec3::Y, std::f64::consts::FRAC_PI_2).unwrap();
        assert!(close(
            &p["Rev"].apply_direction(&Vec3::X),
            &rz.rotate(&Vec3::X)
        ));
        let direction = rz.mul(&rx).mul(&ry).rotate(&Vec3::X);
        assert!(close(&p["Pri"].apply_direction(&Vec3::X), &direction));
        // And the positions those transforms imply, in metres.
        let rev_origin = Vec3::new(0.1, -0.2, 0.3);
        assert!(close(&p["Rev"].translation, &rev_origin));
        let expected = rev_origin
            .add(&rz.mul(&rx).rotate(&Vec3::new(0.05, 0.06, 0.07)))
            .add(&direction.scaled(pri));
        assert!(
            close(&p["Pri"].translation, &expected),
            "{:?} vs {expected:?}",
            p["Pri"].translation
        );
    }

    #[test]
    fn offsets_add_to_the_joint_value() {
        let text = r#"<Robot Type="Offsets" RootNode="Root">
            <RobotNode name="Root"><Child name="Rev"/></RobotNode>
            <RobotNode name="Rev">
              <Transform><Translation x="10" y="20" z="30"/><rollpitchyaw roll="20" pitch="0" yaw="0" units="degree"/></Transform>
              <Joint type="revolute" offset="0.3"><axis x="0" y="0" z="1"/><limits lo="-90" hi="90" units="degree"/></Joint>
              <Child name="Link"/>
            </RobotNode>
            <RobotNode name="Link"><Transform><Translation x="100" y="0" z="0"/></Transform><Child name="Pri"/></RobotNode>
            <RobotNode name="Pri">
              <Transform><rollpitchyaw roll="0" pitch="30" yaw="0" units="degree"/></Transform>
              <Joint type="prismatic" offset="15"><translationdirection x="1" y="0" z="0"/><limits lo="-50" hi="50" units="mm"/></Joint>
              <Child name="EEF"/>
            </RobotNode>
            <RobotNode name="EEF"><Transform><Translation x="0" y="0" z="80"/></Transform></RobotNode>
          </Robot>"#;
        let imported = robot(text);
        let deg = std::f64::consts::PI / 180.0;
        for (q0, q1) in [(0.0, 0.0), (0.2, 0.010), (-0.5, -0.030), (1.0, 0.040)] {
            let p = poses(&imported, &[("Rev", q0), ("Pri", q1)]);
            // Written out from Simox's rule: parent * Transform * motion(value + offset).
            let rev = Transform::new(
                Vec3::new(0.01, 0.02, 0.03),
                Quat::from_euler_rpy(20.0 * deg, 0.0, 0.0),
            )
            .compose(&Transform::from_rotation(
                Quat::from_axis_angle(&Vec3::Z, q0 + 0.3).unwrap(),
            ));
            let link = rev.compose(&Transform::from_translation(Vec3::new(0.1, 0.0, 0.0)));
            let pri = link
                .compose(&Transform::from_rotation(Quat::from_euler_rpy(
                    0.0,
                    30.0 * deg,
                    0.0,
                )))
                .compose(&Transform::from_translation(Vec3::new(
                    q1 + 0.015,
                    0.0,
                    0.0,
                )));
            let eef = pri.compose(&Transform::from_translation(Vec3::new(0.0, 0.0, 0.08)));
            for (name, expected) in [("Rev", rev), ("Pri", pri), ("EEF", eef)] {
                assert!(
                    close(&p[name].translation, &expected.translation),
                    "{name} at ({q0}, {q1})"
                );
                assert!(
                    p[name]
                        .rotation
                        .approx_same_rotation(&expected.rotation, 1e-12),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn dh_parameters_compose_as_in_simox() {
        // As in Simox's Joint3DH example: a = 300 mm, alpha = 45 degrees.
        let text = r#"<Robot Type="DH" RootNode="Start">
            <RobotNode name="Start"><Child name="J1"/></RobotNode>
            <RobotNode name="J1"><Joint type="revolute"><Limits unit="degree" lo="-90" hi="90"/></Joint><Child name="J2"/></RobotNode>
            <RobotNode name="J2"><Transform><DH a="300" d="0" theta="0" alpha="45" units="degree"/></Transform>
              <Joint type="revolute"><Limits unit="degree" lo="-180" hi="45"/></Joint><Child name="End"/></RobotNode>
            <RobotNode name="End"><Transform><DH a="300" d="0" theta="50" alpha="0" units="degree"/></Transform></RobotNode>
          </Robot>"#;
        let imported = robot(text);
        let (q1, q2) = (0.3, -0.7);
        let p = poses(&imported, &[("J1", q1), ("J2", q2)]);
        let deg = std::f64::consts::PI / 180.0;
        let dh = |theta: f64, a: f64, alpha: f64| {
            Transform::from_rotation(Quat::from_axis_angle(&Vec3::Z, theta).unwrap())
                .compose(&Transform::from_translation(Vec3::new(a, 0.0, 0.0)))
                .compose(&Transform::from_rotation(
                    Quat::from_axis_angle(&Vec3::X, alpha).unwrap(),
                ))
        };
        let rz = |q: f64| Transform::from_rotation(Quat::from_axis_angle(&Vec3::Z, q).unwrap());
        let end = rz(q1)
            .compose(&dh(0.0, 0.3, 45.0 * deg))
            .compose(&rz(q2))
            .compose(&dh(50.0 * deg, 0.3, 0.0));
        assert!(
            close(&p["End"].translation, &end.translation),
            "{:?} vs {:?}",
            p["End"].translation,
            end.translation
        );
        let joint = imported.project.robot.joint_by_name("J2").unwrap();
        assert!((joint.limits.lower + std::f64::consts::PI).abs() < 1e-12);
        assert!((joint.limits.upper - 45.0 * deg).abs() < 1e-12);
    }

    #[test]
    fn units_follow_simoxs_rules() {
        let text = r#"<robot type="Units" rootnode="A">
            <ROBOTNODE NAME="A"><CHILD NAME="B"/></ROBOTNODE>
            <RobotNode name="B">
              <Transform><Translation x="0.5" y="0" z="0" units="m"/><Translation x="20" y="0" z="0"/></Transform>
              <Joint type="revolute"><Limits lo="-1" hi="2"/></Joint>
              <Child name="C"/>
            </RobotNode>
            <RobotNode name="C"><Joint type="prismatic"><TranslationDirection x="0" y="0" z="2"/><Limits lo="-0.1" hi="0.25" units="m"/></Joint></RobotNode>
          </robot>"#;
        let imported = robot(text);
        let p = poses(&imported, &[]);
        assert!(
            close(&p["B"].translation, &Vec3::new(0.52, 0.0, 0.0)),
            "metres then millimetres"
        );
        let b = imported.project.robot.joint_by_name("B").unwrap();
        assert!(
            (b.limits.lower + 1.0).abs() < 1e-15,
            "radians when no unit is named"
        );
        let c = imported.project.robot.joint_by_name("C").unwrap();
        // 0.25 m = 250 mm of Simox value, times a direction of length 2.
        assert!((c.limits.upper - 0.5).abs() < 1e-12, "{:?}", c.limits);
        assert!(
            imported.warnings.iter().any(|w| w.contains("length 2")),
            "{:?}",
            imported.warnings
        );
        let p = poses(&imported, &[("C", 0.3)]);
        assert!(close(&p["C"].translation, &Vec3::new(0.52, 0.0, 0.3)));
    }

    #[test]
    fn end_effector_actors_tcp_and_gcp_carry_over() {
        use std::fmt::Write as _;
        let mut text = String::from(
            r#"<Robot Type="Arm" RootNode="Base"><RobotNode name="Base"><Child name="J1"/></RobotNode>"#,
        );
        let chain = ["J1", "J2", "J3", "J4", "J5", "J6", "Palm"];
        for (i, n) in chain.iter().enumerate() {
            let joint = if *n == "Palm" {
                String::new()
            } else {
                r#"<Joint type="revolute"><Limits lo="-1" hi="1"/></Joint>"#.to_owned()
            };
            let next = chain
                .get(i + 1)
                .map_or(String::new(), |c| format!(r#"<Child name="{c}"/>"#));
            let _ = write!(
                text,
                r#"<RobotNode name="{n}"><Transform><Translation x="0" y="0" z="100"/></Transform>{joint}{next}<Child name="{n}F"/></RobotNode>"#
            );
            let _ = write!(text, r#"<RobotNode name="{n}F"/>"#);
        }
        text.push_str(r#"<RobotNode name="FingerA"><Joint type="revolute"><Limits lo="0" hi="1"/></Joint></RobotNode>
            <RobotNode name="FingerB"><Joint type="revolute"><Limits lo="0" hi="1"/></Joint></RobotNode>
            <RobotNode name="TCP"><Transform><Translation x="0" y="0" z="50"/></Transform></RobotNode>
            <Endeffector name="Hand" base="Palm" tcp="TCP" gcp="Palm">
              <Preshape name="open"/>
              <Static><Node name="Palm"/></Static>
              <Actor name="Close"><Node name="FingerA" direction="1"/><Node name="FingerB" direction="1"/></Actor>
            </Endeffector>
            <RobotNodeSet name="Arm"><Node name="J1"/></RobotNodeSet></Robot>"#);
        let text = text.replace(r#"<RobotNode name="Palm"><Transform><Translation x="0" y="0" z="100"/></Transform><Child name="PalmF"/>"#,
            r#"<RobotNode name="Palm"><Transform><Translation x="0" y="0" z="100"/></Transform><Child name="PalmF"/><Child name="FingerA"/><Child name="FingerB"/><Child name="TCP"/>"#);
        let imported = robot(&text);
        let robot = &imported.project.robot;
        assert_eq!(
            robot.axis_count(),
            6,
            "the two finger joints are actors, not axes"
        );
        assert_eq!(
            robot.joint_by_name("FingerA").unwrap().role,
            JointRole::EndEffector
        );
        assert_eq!(robot.default_tool_frame().unwrap().name, "Hand tcp");
        assert!(robot.tool_frames.iter().any(|f| f.name == "Hand gcp"));
        let joined = imported.warnings.join("\n");
        assert!(joined.contains("preshapes"), "{joined}");
        assert!(joined.contains("<robotnodeset>"), "{joined}");
    }

    #[test]
    fn simoxs_primitives_example_object_imports() {
        // Simox's own data/objects/PrimitivesExample.xml, with its comments.
        let text = r#"<ManipulationObject name='Vitalis'>
            <Visualization>
              <Primitives>
                <Box width="136" height="190" depth="40"/>
                <Sphere radius="0.15" unitslength="m"><Transform><Translation x="0" y="170" z="0"/></Transform></Sphere>
                <Box width="136" height="190" depth="40"><Transform><Translation x="0" y="-355" z="0"/><rpy roll="0" pitch="1.41" yaw="0" unitsAngle="radian"/></Transform></Box>
                <Cylinder radius="10" height="80"/>
              </Primitives>
              <UseAsCollisionModel/>
            </Visualization>
          </ManipulationObject>"#;
        let object = import_simox_object(text, &NoMeshes).unwrap();
        let g = &object.body.geometries;
        assert_eq!(g.len(), 4);
        assert!(g.iter().all(|g| g.role == GeometryRole::Both));
        assert_eq!(
            g[0].shape,
            Shape::Box {
                size: Vec3::new(0.136, 0.19, 0.04)
            }
        );
        assert_eq!(g[1].shape, Shape::Sphere { radius: 0.15 });
        assert!(
            close(&g[1].origin.translation, &Vec3::new(0.0, 0.17, 0.0)),
            "transforms are not chained"
        );
        assert!(close(
            &g[2].origin.translation,
            &Vec3::new(0.0, -0.355, 0.0)
        ));
        // The cylinder lies along its frame's Y, as Simox's do.
        assert!(close(&g[3].origin.apply_direction(&Vec3::Z), &Vec3::Y));
        assert_eq!(object.body.kind, BodyKind::Manipulable);
        assert!(object.warnings.iter().any(|w| w.contains("friction")));
        assert!(object.warnings.iter().any(|w| w.contains("no mass")));
    }

    #[test]
    fn an_object_with_only_unreadable_meshes_is_refused_with_the_reason() {
        let text = "<ManipulationObject name='RiceBox'><Visualization><File type='inventor'>iv/ricefit.iv</File><UseAsCollisionModel/></Visualization></ManipulationObject>";
        let error = import_simox_object(text, &NoMeshes)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(".iv") && error.contains("not supported"),
            "{error}"
        );
        let stl = "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 0.1 0 0\nvertex 0 0.1 0\nendloop\nendfacet\nendsolid s\n";
        let pig = "<ManipulationObject name='pig'><Visualization><File type='inventor'>stl/Piggy6.stl</File><UseAsCollisionModel/></Visualization><Physics><Mass value='250' units='g'/></Physics></ManipulationObject>";
        let source = |r: &str| (r == "stl/Piggy6.stl").then(|| stl.as_bytes().to_vec());
        let object = import_simox_object(pig, &source).unwrap();
        assert_eq!(object.resources.len(), 1);
        assert!((object.body.physical.mass - 0.25).abs() < 1e-15);
    }

    #[test]
    fn unsupported_and_malformed_robots_are_refused_with_a_reason() {
        let bad = |t: &str| {
            import_simox_robot(t, &NoMeshes, &ImportOptions::default())
                .unwrap_err()
                .to_string()
        };
        assert!(bad("<Robot Type='r'/>").contains("RootNode"));
        assert!(bad("<Scene/>").contains("not a Simox robot"));
        assert!(bad("<Robot Type='r' RootNode='x'><RobotNode name='a'/></Robot>").contains("`x`"));
        assert!(bad("<Robot Type='r' RootNode='a'><RobotNode name='a'><Child name='b'/></RobotNode></Robot>").contains("unknown child"));
        assert!(bad("<Robot Type='r' RootNode='a'><RobotNode name='a'><Child name='b'/></RobotNode><RobotNode name='b'><Joint type='hemisphere'/></RobotNode></Robot>").contains("hemisphere"));
        assert!(bad("<Robot Type='r' RootNode='a'><RobotNode name='a'><Child name='b'/></RobotNode><RobotNode name='b'><Joint type='prismatic'/></RobotNode></Robot>").contains("TranslationDirection"));
        assert!(bad("<Robot Type='r' RootNode='a'><RobotNode name='a'><ChildFromRobot/></RobotNode></Robot>").contains("single-file"));
        assert!(bad("<Robot Type='r' RootNode='a'><RobotNode name='a'><Child name='b'/></RobotNode><RobotNode name='b'><Transform><Translation x='1' y='q' z='0'/></Transform></RobotNode></Robot>").contains("not a finite number"));
        assert!(bad("<Robot Type='r' RootNode='a'><RobotNode name='a'><Child name='b'/></RobotNode><RobotNode name='b'><Transform><Matrix3x3><row1 c1='2' c2='0' c3='0'/></Matrix3x3></Transform></RobotNode></Robot>").contains("not a rotation"));
    }

    #[test]
    fn a_rotation_matrix_with_few_digits_is_accepted() {
        let text = "<Robot Type='r' RootNode='a'><RobotNode name='a'><Child name='b'/></RobotNode><RobotNode name='b'><Transform>
            <Matrix4x4 units='m'><row1 c1='0.7071' c2='-0.7071' c3='0' c4='1'/><row2 c1='0.7071' c2='0.7071' c3='0' c4='0'/><row3 c1='0' c2='0' c3='1' c4='0'/><row4 c1='0' c2='0' c3='0' c4='1'/></Matrix4x4>
            </Transform><Joint type='revolute'><Limits lo='-1' hi='1'/></Joint></RobotNode></Robot>";
        let imported = robot(text);
        let p = poses(&imported, &[]);
        assert!(close(&p["b"].translation, &Vec3::new(1.0, 0.0, 0.0)));
        let x = p["b"].apply_direction(&Vec3::X);
        assert!(
            (x.x - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-4
                && (x.y - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-4
        );
    }
}
