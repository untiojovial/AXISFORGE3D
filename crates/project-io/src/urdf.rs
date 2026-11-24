//! URDF import.
//!
//! Reads the kinematic and geometric content of a URDF: links with their
//! visual, collision and inertial elements, and revolute, continuous,
//! prismatic and fixed joints. URDF's conventions carry over directly — SI
//! units, an `xyz`/`rpy` origin with fixed-axis roll-pitch-yaw, a joint axis
//! expressed in the joint frame — so the conversion changes no numbers.
//!
//! What does not carry over is reported, not hidden:
//!
//! - `continuous` joints get limits of plus and minus a full turn, because
//!   every joint here has limits;
//! - `mimic` joints are imported as independent joints;
//! - `floating` and `planar` joints are refused;
//! - off-diagonal inertia terms and the inertial frame's rotation are dropped;
//! - `transmission`, `gazebo` and other extension elements are ignored.
//!
//! A robot with more than six actuated joints is refused unless the surplus
//! are named as end-effector joints in the options.

use std::collections::{BTreeMap, BTreeSet};

use axisforge_domain::{
    templates::stable_id, GeometryInstance, GeometryRole, Inertial, Joint, JointLimits, Link, Quat,
    Shape, Transform, Vec3,
};
use roxmltree::{Document, Node};

use crate::import::{
    hex_color, number, numbers, ImportError, ImportOptions, Imported, MeshSource, MeshStore,
    RobotAssembly, MAX_DESCRIPTION_BYTES,
};

fn element<'a>(node: Node<'a, 'a>, name: &str) -> Option<Node<'a, 'a>> {
    node.children()
        .find(|child| child.is_element() && child.tag_name().name() == name)
}

fn elements<'a>(node: Node<'a, 'a>, name: &'a str) -> impl Iterator<Item = Node<'a, 'a>> {
    node.children()
        .filter(move |child| child.is_element() && child.tag_name().name() == name)
}

fn required<'a>(node: Node<'a, 'a>, attribute: &str, what: &str) -> Result<&'a str, ImportError> {
    node.attribute(attribute)
        .ok_or_else(|| ImportError::Malformed(format!("{what}: missing attribute `{attribute}`")))
}

fn vec3(text: &str, what: &str) -> Result<Vec3, ImportError> {
    let v = numbers(text, 3, what)?;
    Ok(Vec3::new(v[0], v[1], v[2]))
}

/// The `<origin xyz rpy>` of an element; identity when absent.
fn origin(node: Node<'_, '_>, what: &str) -> Result<Transform, ImportError> {
    let Some(origin) = element(node, "origin") else {
        return Ok(Transform::IDENTITY);
    };
    let xyz = origin
        .attribute("xyz")
        .map_or(Ok(Vec3::ZERO), |t| vec3(t, &format!("{what} origin xyz")))?;
    let rpy = origin
        .attribute("rpy")
        .map_or(Ok(Vec3::ZERO), |t| vec3(t, &format!("{what} origin rpy")))?;
    Ok(Transform::new(
        xyz,
        Quat::from_euler_rpy(rpy.x, rpy.y, rpy.z),
    ))
}

struct Colour {
    hex: String,
    opacity: f64,
}

fn colour(node: Node<'_, '_>, what: &str) -> Result<Option<Colour>, ImportError> {
    let Some(color) = element(node, "color") else {
        return Ok(None);
    };
    let rgba = numbers(required(color, "rgba", what)?, 4, &format!("{what} rgba"))?;
    Ok(Some(Colour {
        hex: hex_color(rgba[0], rgba[1], rgba[2]),
        opacity: rgba[3].clamp(0.0, 1.0),
    }))
}

struct Reader<'a, 'm> {
    robot: String,
    materials: BTreeMap<String, Colour>,
    meshes: MeshStore<'m>,
    warnings: Vec<String>,
    /// Links whose inertia lost its off-diagonal terms or frame rotation.
    simplified_inertia: Vec<String>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl Reader<'_, '_> {
    /// One `<visual>` or `<collision>` element as a geometry, or `None` when
    /// its mesh could not be loaded.
    fn geometry(
        &mut self,
        node: Node<'_, '_>,
        link: &str,
        role: GeometryRole,
        index: usize,
    ) -> Result<Option<GeometryInstance>, ImportError> {
        let kind = if role == GeometryRole::Visual {
            "visual"
        } else {
            "collision"
        };
        let what = format!("link `{link}` {kind} {index}");
        let shape_node = element(node, "geometry")
            .and_then(|g| g.children().find(Node::is_element))
            .ok_or_else(|| ImportError::Malformed(format!("{what}: no <geometry> shape")))?;
        let shape = match shape_node.tag_name().name() {
            "box" => Shape::Box {
                size: vec3(required(shape_node, "size", &what)?, &what)?,
            },
            "sphere" => Shape::Sphere {
                radius: number(required(shape_node, "radius", &what)?, &what)?,
            },
            "cylinder" => Shape::Cylinder {
                radius: number(required(shape_node, "radius", &what)?, &what)?,
                length: number(required(shape_node, "length", &what)?, &what)?,
            },
            "mesh" => {
                let reference = required(shape_node, "filename", &what)?;
                let scale = shape_node
                    .attribute("scale")
                    .map_or(Ok(Vec3::new(1.0, 1.0, 1.0)), |t| {
                        vec3(t, &format!("{what} mesh scale"))
                    })?;
                match self.meshes.resource(reference) {
                    Some(resource_id) => Shape::Mesh { resource_id, scale },
                    None => return Ok(None),
                }
            }
            other => {
                return Err(ImportError::Unsupported(format!(
                    "{what}: geometry `{other}`"
                )));
            }
        };
        let name = format!("{kind} {index}");
        let ns = format!("{}::{link}", self.robot);
        let mut geometry = GeometryInstance::visual(stable_id(&ns, "geometry", &name), name, shape)
            .with_origin(origin(node, &what)?)
            .with_role(role);
        if role == GeometryRole::Visual {
            if let Some(material) = element(node, "material") {
                let inline = colour(material, &what)?;
                let named = material
                    .attribute("name")
                    .and_then(|n| self.materials.get(n));
                if let Some(Colour { hex, opacity }) = inline.as_ref().or(named) {
                    geometry.material.color.clone_from(hex);
                    geometry.material.opacity = *opacity;
                }
            }
        }
        Ok(Some(geometry))
    }

    fn inertial(&mut self, node: Node<'_, '_>, link: &str) -> Result<Inertial, ImportError> {
        let Some(inertial) = element(node, "inertial") else {
            return Ok(Inertial::default());
        };
        let what = format!("link `{link}` inertial");
        let mass = element(inertial, "mass")
            .map(|m| number(required(m, "value", &what)?, &what))
            .transpose()?
            .unwrap_or(0.0);
        let frame = origin(inertial, &what)?;
        let mut diagonal = Vec3::ZERO;
        if let Some(inertia) = element(inertial, "inertia") {
            let get = |name: &str| -> Result<f64, ImportError> {
                inertia
                    .attribute(name)
                    .map_or(Ok(0.0), |t| number(t, &what))
            };
            diagonal = Vec3::new(get("ixx")?, get("iyy")?, get("izz")?);
            let off = get("ixy")?.abs() + get("ixz")?.abs() + get("iyz")?.abs();
            let rotated = !frame.rotation.approx_same_rotation(&Quat::IDENTITY, 1.0e-9);
            if off > 1.0e-12 || rotated {
                self.simplified_inertia.push(link.to_owned());
            }
        }
        Ok(Inertial {
            mass,
            center_of_mass: frame.translation,
            inertia_diagonal: diagonal,
        })
    }
}

/// Reads every `<link>`, returning them and their ids by name.
fn read_links(
    reader: &mut Reader<'_, '_>,
    root: Node<'_, '_>,
) -> Result<(Vec<Link>, BTreeMap<String, String>), ImportError> {
    let mut links = Vec::new();
    let mut link_ids: BTreeMap<String, String> = BTreeMap::new();
    for node in elements(root, "link") {
        let name = required(node, "name", "<link>")?;
        let id = stable_id(&reader.robot, "link", name);
        if link_ids.insert(name.to_owned(), id.clone()).is_some() {
            return Err(ImportError::Malformed(format!(
                "link `{name}` is defined twice"
            )));
        }
        let mut link = Link::new(id, name).with_inertial(reader.inertial(node, name)?);
        for (index, visual) in elements(node, "visual").enumerate() {
            link.geometries
                .extend(reader.geometry(visual, name, GeometryRole::Visual, index)?);
        }
        for (index, collision) in elements(node, "collision").enumerate() {
            link.geometries.extend(reader.geometry(
                collision,
                name,
                GeometryRole::Collision,
                index,
            )?);
        }
        links.push(link);
    }
    if links.is_empty() {
        return Err(ImportError::Malformed("the robot has no links".to_owned()));
    }

    Ok((links, link_ids))
}

/// Reads every `<joint>`, returning them and the ids of the links that are
/// some joint's child.
fn read_joints(
    reader: &mut Reader<'_, '_>,
    root: Node<'_, '_>,
    link_ids: &BTreeMap<String, String>,
) -> Result<(Vec<Joint>, BTreeSet<String>), ImportError> {
    let mut joints = Vec::new();
    let mut children: BTreeSet<String> = BTreeSet::new();
    for node in elements(root, "joint") {
        let name = required(node, "name", "<joint>")?;
        let what = format!("joint `{name}`");
        let end = |tag: &str| -> Result<String, ImportError> {
            let link = element(node, tag)
                .and_then(|n| n.attribute("link"))
                .ok_or_else(|| ImportError::Malformed(format!("{what}: missing <{tag} link>")))?;
            link_ids.get(link).cloned().ok_or_else(|| {
                ImportError::Malformed(format!("{what}: <{tag}> names unknown link `{link}`"))
            })
        };
        let (parent, child) = (end("parent")?, end("child")?);
        children.insert(child.clone());
        let id = stable_id(&reader.robot, "joint", name);
        let pose = origin(node, &what)?;
        let axis = element(node, "axis")
            .and_then(|a| a.attribute("xyz"))
            .map_or(Ok(Vec3::X), |t| vec3(t, &format!("{what} axis")))?;
        let limit = element(node, "limit");
        let bound = |attribute: &str| -> Result<f64, ImportError> {
            limit
                .and_then(|l| l.attribute(attribute))
                .map_or(Ok(0.0), |t| number(t, &what))
        };
        let kind = required(node, "type", &what)?;
        let mut joint = match kind {
            "fixed" => Joint::fixed(id, name, parent, child, pose),
            "revolute" | "continuous" => {
                let mut joint = Joint::revolute(id, name, parent, child, pose, axis, 0.0)?;
                joint.limits = if kind == "continuous" {
                    reader.warnings.push(format!(
                        "{what} is continuous; it was given limits of plus and minus one full turn"
                    ));
                    JointLimits::symmetric(std::f64::consts::TAU, name)?
                } else {
                    JointLimits::new(bound("lower")?, bound("upper")?, name)?
                };
                joint
            }
            "prismatic" => Joint::prismatic(
                id,
                name,
                parent,
                child,
                pose,
                axis,
                bound("lower")?,
                bound("upper")?,
            )?,
            other => {
                return Err(ImportError::Unsupported(format!(
                    "{what} is of type `{other}`; revolute, continuous, prismatic and fixed \
                     joints are supported"
                )));
            }
        };
        let positive = |attribute: &str| -> Result<Option<f64>, ImportError> {
            Ok(Some(bound(attribute)?).filter(|value| *value > 0.0))
        };
        if joint.is_actuated() {
            joint.max_velocity = positive("velocity")?;
            joint.max_effort = positive("effort")?;
        }
        if element(node, "mimic").is_some() {
            reader.warnings.push(format!(
                "{what} mimics another joint; it was imported as an independent joint"
            ));
        }
        joints.push(joint);
    }

    Ok((joints, children))
}

/// The one link that is no joint's child.
fn find_root(links: &[Link], children: &BTreeSet<String>) -> Result<String, ImportError> {
    let mut roots = links.iter().filter(|link| !children.contains(&link.id));
    let root_link = roots
        .next()
        .ok_or_else(|| {
            ImportError::Malformed("every link is some joint's child: there is no root".to_owned())
        })?
        .id
        .clone();
    if let Some(second) = roots.next() {
        return Err(ImportError::Malformed(format!(
            "the robot has more than one root link (`{}` is also never a child)",
            second.name
        )));
    }

    Ok(root_link)
}

/// Records the top-level elements that were not read.
fn note_ignored(warnings: &mut Vec<String>, root: Node<'_, '_>) {
    let mut ignored: BTreeMap<&str, usize> = BTreeMap::new();
    for child in root.children().filter(Node::is_element) {
        let tag = child.tag_name().name();
        if !matches!(tag, "link" | "joint" | "material") {
            *ignored.entry(tag).or_insert(0) += 1;
        }
    }
    for (tag, count) in ignored {
        warnings.push(format!("{count} <{tag}> element(s) were ignored"));
    }
}

/// Imports a URDF document.
///
/// `meshes` supplies the files the document's `<mesh filename>` attributes
/// refer to, by the reference exactly as written.
///
/// # Errors
/// When the text is not a URDF, uses something unsupported, or describes a
/// robot that does not validate.
pub fn import_urdf(
    text: &str,
    meshes: &dyn MeshSource,
    options: &ImportOptions,
) -> Result<Imported, ImportError> {
    if text.len() > MAX_DESCRIPTION_BYTES {
        return Err(ImportError::TooLarge { size: text.len() });
    }
    // The parser's default refuses any DTD, so entity-expansion attacks have
    // nothing to expand.
    let document = Document::parse(text).map_err(|e| ImportError::Malformed(e.to_string()))?;
    let root = document.root_element();
    if root.tag_name().name() != "robot" {
        return Err(ImportError::Malformed(format!(
            "the root element is <{}>, not <robot>: this is not a URDF",
            root.tag_name().name()
        )));
    }
    let robot_name = required(root, "name", "<robot>")?.to_owned();
    let mut reader = Reader {
        robot: robot_name.clone(),
        materials: BTreeMap::new(),
        meshes: MeshStore::new(&robot_name, meshes),
        warnings: Vec::new(),
        simplified_inertia: Vec::new(),
        _marker: std::marker::PhantomData,
    };
    for material in elements(root, "material") {
        if let (Some(name), Some(value)) =
            (material.attribute("name"), colour(material, "<material>")?)
        {
            reader.materials.insert(name.to_owned(), value);
        }
    }

    let (links, link_ids) = read_links(&mut reader, root)?;
    let (joints, children) = read_joints(&mut reader, root, &link_ids)?;
    let root_link = find_root(&links, &children)?;
    note_ignored(&mut reader.warnings, root);
    reader.meshes.report(&mut reader.warnings);
    if !reader.simplified_inertia.is_empty() {
        reader.warnings.push(format!(
            "only the diagonal of the inertia tensor was kept for {} link(s); products of \
             inertia and the inertial frame's rotation were dropped: {}",
            reader.simplified_inertia.len(),
            reader.simplified_inertia.join(", ")
        ));
    }

    RobotAssembly {
        name: robot_name,
        root_link,
        links,
        joints,
        base_pose: Transform::IDENTITY,
        frames: Vec::new(),
        default_frame: None,
        end_effector_joints: BTreeSet::new(),
        resources: reader.meshes.resources,
        warnings: reader.warnings,
    }
    .finish(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::NoMeshes;
    use axisforge_domain::{JointKind, JointRole};

    const ARM: &str = r#"<?xml version="1.0"?>
<robot name="two link">
  <material name="blue"><color rgba="0 0 1 0.5"/></material>
  <link name="base">
    <inertial><origin xyz="0 0 0.05"/><mass value="4"/><inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><cylinder radius="0.05" length="0.1"/></geometry><material name="blue"/></visual>
    <collision><origin xyz="0 0 0.05"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="upper"><visual><geometry><sphere radius="0.03"/></geometry><material name="red"><color rgba="1 0 0 1"/></material></visual></link>
  <link name="tip"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="2.0" effort="40" velocity="3"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="upper"/><child link="tip"/>
    <origin xyz="0.2 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.3" effort="0" velocity="0"/>
  </joint>
  <transmission name="t"/>
</robot>"#;

    fn import(text: &str) -> Imported {
        import_urdf(text, &NoMeshes, &ImportOptions::default()).unwrap()
    }

    #[test]
    fn links_joints_limits_and_geometry_carry_over() {
        let imported = import(ARM);
        let robot = &imported.project.robot;
        assert_eq!(robot.name, "two link");
        assert_eq!(robot.links.len(), 3);
        assert_eq!(robot.axis_count(), 2);
        assert_eq!(robot.link(&robot.root_link_id).unwrap().name, "base");

        let shoulder = robot.joint_by_name("shoulder").unwrap();
        assert_eq!(shoulder.kind, JointKind::Revolute);
        assert!(
            (shoulder.limits.lower + 1.5).abs() < 1e-15
                && (shoulder.limits.upper - 2.0).abs() < 1e-15
        );
        assert_eq!(shoulder.max_velocity, Some(3.0));
        assert_eq!(shoulder.max_effort, Some(40.0));
        assert!(shoulder.axis.distance(&Vec3::Y) < 1e-15);
        assert!(
            shoulder
                .origin
                .translation
                .distance(&Vec3::new(0.0, 0.0, 0.1))
                < 1e-15
        );
        // rpy yaw of 90 degrees turns the joint frame's X onto the parent's Y.
        assert!(shoulder.origin.apply_direction(&Vec3::X).distance(&Vec3::Y) < 1e-12);

        let slide = robot.joint_by_name("slide").unwrap();
        assert_eq!(slide.kind, JointKind::Prismatic);
        assert_eq!(
            slide.max_velocity, None,
            "a zero velocity limit means unset, not zero"
        );

        let base = robot.link_by_name("base").unwrap();
        assert!((base.inertial.mass - 4.0).abs() < 1e-15);
        assert!((base.inertial.center_of_mass.z - 0.05).abs() < 1e-15);
        assert_eq!(base.visual_geometries().count(), 1);
        assert_eq!(base.collision_geometries().count(), 1);
        let visual = base.visual_geometries().next().unwrap();
        assert_eq!(visual.material.color, "#0000ff");
        assert!((visual.material.opacity - 0.5).abs() < 1e-15);
        let upper = robot.link_by_name("upper").unwrap();
        assert_eq!(
            upper.visual_geometries().next().unwrap().material.color,
            "#ff0000"
        );
    }

    #[test]
    fn the_tip_gets_a_tool_frame_and_bare_links_a_skeleton() {
        let imported = import(ARM);
        let robot = &imported.project.robot;
        assert_eq!(robot.default_tool_frame().unwrap().name, "tip");
        let tip = robot.link_by_name("tip").unwrap();
        assert!(
            tip.visual_geometries().count() > 0,
            "a bare link is drawn as a skeleton"
        );
        assert_eq!(tip.collision_geometries().count(), 0);
        assert!(
            imported.warnings.iter().any(|w| w.contains("skeleton")),
            "{:?}",
            imported.warnings
        );
        assert!(
            imported
                .warnings
                .iter()
                .any(|w| w.contains("<transmission>")),
            "{:?}",
            imported.warnings
        );
    }

    #[test]
    fn importing_twice_gives_the_same_project() {
        assert_eq!(import(ARM), import(ARM));
    }

    #[test]
    fn a_continuous_joint_gets_a_full_turn_each_way_and_a_warning() {
        let text = ARM.replace(
            r#"name="shoulder" type="revolute""#,
            r#"name="shoulder" type="continuous""#,
        );
        let imported = import(&text);
        let joint = imported.project.robot.joint_by_name("shoulder").unwrap();
        assert!((joint.limits.upper - std::f64::consts::TAU).abs() < 1e-15);
        assert!(imported.warnings.iter().any(|w| w.contains("continuous")));
    }

    #[test]
    fn end_effector_joints_are_marked_and_do_not_count_as_axes() {
        let options = ImportOptions {
            end_effector_joints: vec!["slide".to_owned()],
        };
        let imported = import_urdf(ARM, &NoMeshes, &options).unwrap();
        let robot = &imported.project.robot;
        assert_eq!(robot.axis_count(), 1);
        assert_eq!(robot.total_dof(), 2);
        assert_eq!(
            robot.joint_by_name("slide").unwrap().role,
            JointRole::EndEffector
        );
        let unknown = ImportOptions {
            end_effector_joints: vec!["nope".to_owned()],
        };
        let error = import_urdf(ARM, &NoMeshes, &unknown).unwrap_err();
        assert!(error.to_string().contains("shoulder"), "{error}");
    }

    fn chain(count: usize) -> String {
        let mut text = String::from(r#"<robot name="chain"><link name="l0"/>"#);
        for i in 1..=count {
            use std::fmt::Write as _;
            let _ = write!(
                text,
                r#"<link name="l{i}"/><joint name="j{i}" type="revolute"><parent link="l{}"/><child link="l{i}"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>"#,
                i - 1
            );
        }
        text.push_str("</robot>");
        text
    }

    #[test]
    fn seven_axes_are_refused_with_the_way_out() {
        let error = import_urdf(&chain(7), &NoMeshes, &ImportOptions::default()).unwrap_err();
        assert!(matches!(error, ImportError::Unsupported(_)), "{error}");
        assert!(error.to_string().contains("end-effector"), "{error}");
        let options = ImportOptions {
            end_effector_joints: vec!["j7".to_owned()],
        };
        assert_eq!(
            import_urdf(&chain(7), &NoMeshes, &options)
                .unwrap()
                .project
                .robot
                .axis_count(),
            6
        );
    }

    #[test]
    fn documents_that_are_not_usable_urdf_are_refused_with_a_reason() {
        let bad = |text: &str| {
            import_urdf(text, &NoMeshes, &ImportOptions::default())
                .unwrap_err()
                .to_string()
        };
        assert!(bad("<robot").contains("not a valid description"));
        assert!(bad("<sdf/>").contains("not a URDF"));
        assert!(bad(r#"<robot name="r"/>"#).contains("no links"));
        assert!(
            bad(r#"<robot name="r"><link name="a"/><link name="a"/></robot>"#).contains("twice")
        );
        assert!(
            bad(r#"<robot name="r"><link name="a"/><link name="b"/></robot>"#)
                .contains("more than one root")
        );
        let floating = ARM.replace(r#"type="prismatic""#, r#"type="floating""#);
        assert!(bad(&floating).contains("floating"));
        let dangling = ARM.replace(r#"<child link="tip"/>"#, r#"<child link="ghost"/>"#);
        assert!(bad(&dangling).contains("ghost"));
        let nan = ARM.replace(r#"xyz="0.2 0 0""#, r#"xyz="0.2 nan 0""#);
        assert!(bad(&nan).contains("finite"));
        let entity =
            r#"<!DOCTYPE r [<!ENTITY a "aaaa">]><robot name="&a;"><link name="l"/></robot>"#;
        assert!(
            bad(entity).contains("not a valid description"),
            "a DTD must be refused"
        );
    }

    #[test]
    fn a_link_whose_visual_cannot_be_read_is_drawn_with_its_collision_shape() {
        let text = r#"<robot name="m"><link name="a">
            <visual><geometry><mesh filename="body.dae"/></geometry></visual>
            <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
        </link><link name="b"/>
        <joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 0.1"/><limit lower="-1" upper="1"/></joint>
        </robot>"#;
        let imported = import_urdf(text, &NoMeshes, &ImportOptions::default()).unwrap();
        let link = imported.project.robot.link_by_name("a").unwrap();
        assert_eq!(link.geometries.len(), 1);
        assert_eq!(link.geometries[0].role, GeometryRole::Both);
        assert!(
            imported
                .warnings
                .iter()
                .any(|w| w.contains("collision geometry")),
            "{:?}",
            imported.warnings
        );
    }

    #[test]
    fn meshes_are_loaded_once_scaled_and_missing_ones_reported() {
        let text = r#"<robot name="m"><link name="a">
            <visual><geometry><mesh filename="package://m/meshes/part.stl" scale="0.001 0.001 0.001"/></geometry></visual>
            <collision><geometry><mesh filename="package://m/meshes/part.stl"/></geometry></collision>
            <visual><geometry><mesh filename="missing.obj"/></geometry></visual>
            <visual><geometry><mesh filename="shape.dae"/></geometry></visual>
        </link><link name="b"/>
        <joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 0.1"/><limit lower="-1" upper="1"/></joint>
        </robot>"#;
        let stl = "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 10 0 0\nvertex 0 10 0\nendloop\nendfacet\nendsolid s\n";
        let calls = std::cell::Cell::new(0);
        let source = |reference: &str| {
            calls.set(calls.get() + 1);
            match reference {
                "package://m/meshes/part.stl" => Some(stl.as_bytes().to_vec()),
                "shape.dae" => Some(b"<COLLADA/>".to_vec()),
                _ => None,
            }
        };
        let imported = import_urdf(text, &source, &ImportOptions::default()).unwrap();
        assert_eq!(
            imported.project.resources.len(),
            1,
            "one file, one resource"
        );
        assert_eq!(
            calls.get(),
            2,
            "each readable reference is asked for once, the .dae never"
        );
        let link = imported.project.robot.link_by_name("a").unwrap();
        let Shape::Mesh { scale, resource_id } = &link.geometries[0].shape else {
            panic!("a mesh")
        };
        assert!((scale.x - 0.001).abs() < 1e-15);
        assert_eq!(resource_id, &imported.project.resources[0].id);
        assert_eq!(
            link.geometries.len(),
            2,
            "the two unusable meshes are left out"
        );
        assert!(imported
            .warnings
            .iter()
            .any(|w| w.contains("missing.obj") && w.contains("not provided")));
        assert!(imported
            .warnings
            .iter()
            .any(|w| w.contains("shape.dae") && w.contains("not supported")));
        assert_eq!(
            imported
                .warnings
                .iter()
                .filter(|w| w.contains("mesh file"))
                .count(),
            2,
            "one per reason"
        );
    }
}
