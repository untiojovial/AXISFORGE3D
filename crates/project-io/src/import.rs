//! What every robot importer shares: options, the report, mesh loading and
//! the steps that turn a bare kinematic tree into a usable project.
//!
//! An importer reads a foreign format into links and joints and hands them to
//! [`RobotAssembly::finish`], which does the same things for all of them:
//! marks end-effector joints, adds a tool frame at every tip of the tree,
//! gives links with nothing to draw a stand-in skeleton, and validates the
//! result. Everything an importer could not carry over is recorded as a
//! warning in the report rather than dropped silently.

use std::collections::{BTreeMap, BTreeSet};

use axisforge_domain::{
    templates::stable_id, DomainError, GeometryInstance, GeometryRole, Joint, JointRole, Link,
    Project, Quat, Resource, RobotSystem, Shape, ToolFrame, Transform, Vec3, MAX_AXES,
};

use crate::mesh::{self, MeshError};

/// Largest description file accepted, bytes.
pub const MAX_DESCRIPTION_BYTES: usize = 16 * 1024 * 1024;

/// Where an importer gets the mesh files a description refers to.
///
/// The importer never touches the file system itself. The command line gives
/// it a source that reads from disk; the browser gives it one backed by the
/// files the user picked. `reference` is the path exactly as written in the
/// description; returning `None` means the file is not available.
pub trait MeshSource {
    /// The bytes of the referenced file, if it can be found.
    fn load(&self, reference: &str) -> Option<Vec<u8>>;
}

impl<F: Fn(&str) -> Option<Vec<u8>>> MeshSource for F {
    fn load(&self, reference: &str) -> Option<Vec<u8>> {
        self(reference)
    }
}

/// Files supplied in memory, as when a user picks them in a browser.
///
/// A reference is looked up as written first, then by its file name alone:
/// a description says `package://robot/meshes/base.stl`, and the person who
/// picked the files has a `base.stl`. Two supplied files with the same name
/// make that second lookup ambiguous, and it then finds nothing rather than
/// guessing.
impl MeshSource for BTreeMap<String, Vec<u8>> {
    fn load(&self, reference: &str) -> Option<Vec<u8>> {
        if let Some(bytes) = self.get(reference) {
            return Some(bytes.clone());
        }
        let base = |path: &str| path.rsplit(['/', '\\']).next().unwrap_or(path).to_owned();
        let wanted = base(reference);
        let mut matches = self.iter().filter(|(name, _)| base(name) == wanted);
        match (matches.next(), matches.next()) {
            (Some((_, bytes)), None) => Some(bytes.clone()),
            _ => None,
        }
    }
}

/// A mesh source with no files: every mesh reference becomes a warning.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMeshes;

impl MeshSource for NoMeshes {
    fn load(&self, _: &str) -> Option<Vec<u8>> {
        None
    }
}

/// How a description is imported.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImportOptions {
    /// Names of joints that actuate an end effector rather than position it.
    ///
    /// Such joints do not count toward the six-axis limit and are left alone
    /// by inverse kinematics. Foreign formats rarely say which these are.
    pub end_effector_joints: Vec<String>,
}

/// The outcome of an import.
#[derive(Debug, Clone, PartialEq)]
pub struct Imported {
    /// The project, validated.
    pub project: Project,
    /// Everything that could not be carried over exactly.
    pub warnings: Vec<String>,
}

/// What can go wrong importing a description.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// The description is larger than [`MAX_DESCRIPTION_BYTES`].
    #[error("the description is {size} bytes; the limit is {MAX_DESCRIPTION_BYTES}")]
    TooLarge {
        /// Its size.
        size: usize,
    },
    /// The text is not well-formed XML, or not the expected kind of document.
    #[error("not a valid description: {0}")]
    Malformed(String),
    /// The description uses something this importer does not handle.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The description was read, but the robot it describes is not coherent.
    #[error("the imported robot is invalid: {0}")]
    Invalid(DomainError),
}

impl From<DomainError> for ImportError {
    fn from(error: DomainError) -> Self {
        Self::Invalid(error)
    }
}

/// Parses whitespace-separated numbers, requiring exactly `count` finite ones.
pub(crate) fn numbers(text: &str, count: usize, what: &str) -> Result<Vec<f64>, ImportError> {
    let values: Result<Vec<f64>, _> = text.split_whitespace().map(str::parse::<f64>).collect();
    match values {
        Ok(values) if values.len() == count && values.iter().all(|v| v.is_finite()) => Ok(values),
        _ => Err(ImportError::Malformed(format!(
            "{what}: expected {count} finite numbers, found `{text}`"
        ))),
    }
}

/// Parses one finite number.
pub(crate) fn number(text: &str, what: &str) -> Result<f64, ImportError> {
    Ok(numbers(text, 1, what)?[0])
}

/// `#rrggbb` for colour components in `[0, 1]`.
pub(crate) fn hex_color(red: f64, green: f64, blue: f64) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", byte(red), byte(green), byte(blue))
}

/// Most closed pieces a collision mesh is split into; beyond that it stays
/// one hull.
const MAX_COLLISION_PIECES: usize = 16;

/// One piece of a loaded mesh file, stored as a project resource.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeshPart {
    pub resource_id: String,
    /// `#rrggbb` when the file colours this piece.
    pub color: Option<String>,
    /// What the piece is for: drawing, collision, or both when one resource
    /// serves both.
    pub role: GeometryRole,
}

/// A file that was read, and the resources made from it so far.
struct Loaded {
    file_name: String,
    media_type: &'static str,
    parsed: mesh::Parsed,
    visual: Option<Vec<MeshPart>>,
    collision: Option<Vec<MeshPart>>,
}

/// Loads referenced meshes once each and turns them into project resources.
///
/// A file is drawn by its coloured parts and collides as its whole mesh,
/// split into closed pieces when it has some (see [`mesh::Mesh::closed_pieces`]).
/// The two are kept apart because a colour says nothing about shape: the
/// hulls of the parts of one colour need not cover the solid.
pub(crate) struct MeshStore<'a> {
    namespace: String,
    source: &'a dyn MeshSource,
    /// Files by reference, or `None` for a reference that failed.
    loaded: BTreeMap<String, Option<Loaded>>,
    /// References that failed, grouped by the reason they failed.
    failures: BTreeMap<String, Vec<String>>,
    /// Notes from files that were read, grouped by note.
    notes: BTreeMap<String, Vec<String>>,
    pub resources: Vec<Resource>,
}

impl<'a> MeshStore<'a> {
    pub fn new(namespace: &str, source: &'a dyn MeshSource) -> Self {
        Self {
            namespace: namespace.to_owned(),
            source,
            loaded: BTreeMap::new(),
            failures: BTreeMap::new(),
            notes: BTreeMap::new(),
            resources: Vec::new(),
        }
    }

    /// The resources a mesh reference needs in `role`, loading the file on
    /// first use. `Both` yields one resource for both when that is possible,
    /// and separate drawing and collision pieces when it is not.
    ///
    /// A mesh that cannot be found or read yields `None`; the failure is
    /// reported once, grouped with others that failed for the same reason,
    /// by [`MeshStore::report`].
    pub fn parts(&mut self, reference: &str, role: GeometryRole) -> Option<Vec<MeshPart>> {
        if !self.loaded.contains_key(reference) {
            let loaded = self.load(reference);
            self.loaded.insert(reference.to_owned(), loaded);
        }
        let namespace = self.namespace.clone();
        let loaded = self.loaded.get_mut(reference)?.as_mut()?;
        let resources = &mut self.resources;
        let visual = visual_parts(loaded, reference, &namespace, resources);
        if role == GeometryRole::Visual {
            return Some(visual);
        }
        let collision = collision_parts(loaded, reference, &namespace, resources);
        if role == GeometryRole::Collision {
            return Some(collision);
        }
        // One resource serving both is drawn and collided as itself.
        if let ([drawn], [collided]) = (visual.as_slice(), collision.as_slice()) {
            if drawn.resource_id == collided.resource_id {
                return Some(vec![MeshPart {
                    role: GeometryRole::Both,
                    ..drawn.clone()
                }]);
            }
        }
        Some(visual.into_iter().chain(collision).collect())
    }

    /// Reads a referenced file, or records why it could not be.
    fn load(&mut self, reference: &str) -> Option<Loaded> {
        let file_name = reference.rsplit(['/', '\\']).next().unwrap_or(reference);
        let extension = file_name
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        // Decided before asking the source, so a format that cannot be read is
        // reported as such rather than as a missing file.
        let readable = mesh::is_supported(file_name);
        let bytes = if readable {
            self.source.load(reference)
        } else {
            None
        };
        let outcome: Result<mesh::Parsed, String> = match bytes {
            _ if !readable => Err(format!(
                "`.{extension}` files are not supported; STL, OBJ and COLLADA are"
            )),
            None => Err("the file was not provided".to_owned()),
            Some(bytes) => {
                mesh::parse_parts(file_name, &bytes).map_err(|error: MeshError| error.to_string())
            }
        };
        match outcome {
            Ok(mut parsed) => {
                for note in std::mem::take(&mut parsed.notes) {
                    self.notes
                        .entry(note)
                        .or_default()
                        .push(reference.to_owned());
                }
                Some(Loaded {
                    file_name: file_name.to_owned(),
                    media_type: match extension.as_str() {
                        "obj" => "model/obj",
                        "dae" => "model/vnd.collada+xml",
                        _ => "model/stl",
                    },
                    parsed,
                    visual: None,
                    collision: None,
                })
            }
            Err(reason) => {
                self.failures
                    .entry(reason)
                    .or_default()
                    .push(reference.to_owned());
                None
            }
        }
    }
}

/// Adds a mesh as a resource and returns its id.
fn add_resource(
    resources: &mut Vec<Resource>,
    id: String,
    name: String,
    media_type: &str,
    mesh: mesh::Mesh,
) -> String {
    resources.push(Resource {
        id: id.clone(),
        name,
        media_type: media_type.to_owned(),
        vertices: mesh.vertices,
        indices: mesh.indices,
    });
    id
}

/// The coloured parts of a loaded file, made into resources on first use.
fn visual_parts(
    loaded: &mut Loaded,
    reference: &str,
    namespace: &str,
    resources: &mut Vec<Resource>,
) -> Vec<MeshPart> {
    if let Some(parts) = &loaded.visual {
        return parts.clone();
    }
    let several = loaded.parsed.parts.len() > 1;
    let mut parts = Vec::new();
    for (index, part) in loaded.parsed.parts.iter().enumerate() {
        let (key, name) = if several {
            (
                format!("{reference}#{index}"),
                format!("{} part {}", loaded.file_name, index + 1),
            )
        } else {
            (reference.to_owned(), loaded.file_name.clone())
        };
        let id = add_resource(
            resources,
            stable_id(namespace, "resource", &key),
            name,
            loaded.media_type,
            part.mesh.clone(),
        );
        parts.push(MeshPart {
            resource_id: id,
            color: part.color.clone(),
            role: GeometryRole::Visual,
        });
    }
    loaded.visual = Some(parts.clone());
    parts
}

/// The collision pieces of a loaded file, made into resources on first use.
fn collision_parts(
    loaded: &mut Loaded,
    reference: &str,
    namespace: &str,
    resources: &mut Vec<Resource>,
) -> Vec<MeshPart> {
    if let Some(parts) = &loaded.collision {
        return parts.clone();
    }
    let single = loaded.parsed.parts.len() == 1;
    let whole = if single {
        loaded.parsed.parts[0].mesh.clone()
    } else {
        // Re-welding cannot exceed the limits the parts were read within.
        loaded.parsed.clone().merged().unwrap_or_default()
    };
    let color = if single {
        loaded.parsed.parts[0].color.clone()
    } else {
        None
    };
    let parts = match whole.closed_pieces(MAX_COLLISION_PIECES) {
        Some(pieces) => pieces
            .into_iter()
            .enumerate()
            .map(|(index, piece)| MeshPart {
                resource_id: add_resource(
                    resources,
                    stable_id(namespace, "resource", &format!("{reference}#piece{index}")),
                    format!("{} piece {}", loaded.file_name, index + 1),
                    loaded.media_type,
                    piece,
                ),
                color: color.clone(),
                role: GeometryRole::Collision,
            })
            .collect(),
        // One piece of one part is the drawing resource itself.
        None if single => {
            let mut drawn = visual_parts(loaded, reference, namespace, resources);
            drawn[0].role = GeometryRole::Collision;
            drawn
        }
        None => vec![MeshPart {
            resource_id: add_resource(
                resources,
                stable_id(namespace, "resource", &format!("{reference}#whole")),
                format!("{} (whole)", loaded.file_name),
                loaded.media_type,
                whole,
            ),
            color,
            role: GeometryRole::Collision,
        }],
    };
    loaded.collision = Some(parts.clone());
    parts
}

impl MeshStore<'_> {
    /// One warning per distinct reason a mesh was not imported, and one per
    /// distinct note from the files that were.
    pub fn report(&self, warnings: &mut Vec<String>) {
        for (note, references) in &self.notes {
            warnings.push(format!("{note}: {}", file_names(references)));
        }
        for (reason, references) in &self.failures {
            warnings.push(format!(
                "{} mesh file(s) were not imported ({reason}): {}",
                references.len(),
                file_names(references)
            ));
        }
    }
}

/// File names, which read better than `package://` paths in a warning, at
/// most three of them.
fn file_names(references: &[String]) -> String {
    let names: Vec<&str> = references
        .iter()
        .map(|r| r.rsplit(['/', '\\']).next().unwrap_or(r))
        .collect();
    if names.len() <= 3 {
        names.join(", ")
    } else {
        format!("{}, and {} more", names[..3].join(", "), names.len() - 3)
    }
}

/// A kinematic tree read from a foreign format, before it becomes a robot.
pub(crate) struct RobotAssembly {
    pub name: String,
    pub root_link: String,
    pub links: Vec<Link>,
    pub joints: Vec<Joint>,
    pub base_pose: Transform,
    /// Tool frames the format itself declared.
    pub frames: Vec<ToolFrame>,
    /// Name of the frame the format marks as the default, if any.
    pub default_frame: Option<String>,
    /// Joint names the format itself marks as end-effector actuators.
    pub end_effector_joints: BTreeSet<String>,
    pub resources: Vec<Resource>,
    pub warnings: Vec<String>,
}

impl RobotAssembly {
    /// Applies the shared steps and validates the result.
    pub fn finish(mut self, options: &ImportOptions) -> Result<Imported, ImportError> {
        // Roles first: they decide which joints count as axes.
        let mut end_effector = self.end_effector_joints.clone();
        for name in &options.end_effector_joints {
            if !self.joints.iter().any(|joint| joint.name == *name) {
                return Err(ImportError::Malformed(format!(
                    "end-effector joint `{name}` is not a joint of this robot; its joints are: {}",
                    self.joints
                        .iter()
                        .map(|j| j.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            end_effector.insert(name.clone());
        }
        for joint in &mut self.joints {
            if end_effector.contains(&joint.name) && joint.is_actuated() {
                joint.role = JointRole::EndEffector;
            }
        }
        let axes: Vec<&str> = self
            .joints
            .iter()
            .filter(|joint| joint.is_manipulator_axis())
            .map(|joint| joint.name.as_str())
            .collect();
        if axes.len() > MAX_AXES {
            return Err(ImportError::Unsupported(format!(
                "the robot has {} actuated joints and at most {MAX_AXES} can be manipulator \
                 axes. Name the ones that drive an end effector (gripper fingers, hand joints) \
                 as end-effector joints. Actuated joints: {}",
                axes.len(),
                axes.join(", ")
            )));
        }

        self.add_skeleton();
        self.add_tip_frames();

        let robot = RobotSystem {
            id: stable_id(&self.name, "robot", &self.name),
            name: self.name.clone(),
            root_link_id: self.root_link,
            links: self.links,
            joints: self.joints,
            default_tool_frame_id: None,
            tool_frames: self.frames,
            gripper: None,
            base_pose: self.base_pose,
        };
        let mut robot = robot;
        robot.default_tool_frame_id = self
            .default_frame
            .as_ref()
            .and_then(|name| robot.tool_frames.iter().find(|f| f.name == *name))
            .map(|frame| frame.id.clone());
        let mut project = Project::new(
            stable_id(&self.name, "project", &self.name),
            &self.name,
            robot,
        );
        project.resources = self.resources;
        project.validate()?;
        Ok(Imported {
            project,
            warnings: self.warnings,
        })
    }

    /// Gives every link with nothing to draw something to draw.
    ///
    /// A link whose visual geometry could not be imported but whose collision
    /// geometry could is drawn with the collision geometry: it is the right
    /// shape, only coarser. A link with neither gets a small ball at its origin
    /// and a rod to each child joint, so the kinematic structure is visible.
    fn add_skeleton(&mut self) {
        let mut stand_ins = 0;
        for link in &mut self.links {
            if link.geometries.iter().any(GeometryInstance::is_visual) {
                continue;
            }
            let mut any = false;
            for geometry in &mut link.geometries {
                if geometry.role == GeometryRole::Collision {
                    geometry.role = GeometryRole::Both;
                    any = true;
                }
            }
            if any {
                stand_ins += 1;
            }
        }
        if stand_ins > 0 {
            self.warnings.push(format!(
                "{stand_ins} link(s) are drawn with their collision geometry, because their \
                 visual geometry could not be imported"
            ));
        }
        let lengths: Vec<f64> = self
            .joints
            .iter()
            .map(|joint| joint.origin.translation.norm())
            .filter(|length| *length > 1.0e-6)
            .collect();
        if lengths.is_empty() {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let typical = lengths.iter().sum::<f64>() / lengths.len() as f64;
        let radius = (0.06 * typical).clamp(0.003, 0.04);
        let mut bare = 0;
        for link in &mut self.links {
            if link.geometries.iter().any(GeometryInstance::is_visual) {
                continue;
            }
            bare += 1;
            let ns = format!("{}::{}::skeleton", self.name, link.name);
            link.geometries.push(
                GeometryInstance::visual(
                    stable_id(&ns, "geometry", "node"),
                    "skeleton node",
                    Shape::Sphere {
                        radius: radius * 1.4,
                    },
                )
                .with_color("#8b98a5")
                .with_finish(0.2, 0.6)
                .with_role(GeometryRole::Visual),
            );
            for joint in self
                .joints
                .iter()
                .filter(|joint| joint.parent_link_id == link.id)
            {
                let reach = joint.origin.translation;
                let length = reach.norm();
                if length < 1.0e-6 {
                    continue;
                }
                let rotation = Quat::from_rotation_arc(&Vec3::Z, &reach.scaled(1.0 / length))
                    .unwrap_or(Quat::IDENTITY);
                link.geometries.push(
                    GeometryInstance::visual(
                        stable_id(&ns, "geometry", &joint.name),
                        format!("skeleton to {}", joint.name),
                        Shape::Cylinder { radius, length },
                    )
                    .with_origin(Transform::new(reach.scaled(0.5), rotation))
                    .with_color("#b3bdc7")
                    .with_finish(0.2, 0.6)
                    .with_role(GeometryRole::Visual),
                );
            }
        }
        if bare > 0 {
            self.warnings.push(format!(
                "{bare} link(s) had no geometry that could be imported; they are drawn as a \
                 skeleton and take no part in collision checking"
            ));
        }
    }

    /// Adds a tool frame at every tip of the tree that the format did not
    /// already give one, and picks a default when the format named none.
    fn add_tip_frames(&mut self) {
        let parents: BTreeSet<&str> = self
            .joints
            .iter()
            .map(|joint| joint.parent_link_id.as_str())
            .collect();
        let framed: BTreeSet<String> = self.frames.iter().map(|f| f.link_id.clone()).collect();
        let mut tips: Vec<(usize, String, String)> = Vec::new();
        for link in &self.links {
            if parents.contains(link.id.as_str()) || link.id == self.root_link {
                continue;
            }
            // How many manipulator axes lie between the root and this tip.
            let mut depth = 0;
            let mut cursor = link.id.as_str();
            while let Some(joint) = self.joints.iter().find(|j| j.child_link_id == cursor) {
                if joint.is_manipulator_axis() {
                    depth += 1;
                }
                cursor = joint.parent_link_id.as_str();
            }
            tips.push((depth, link.id.clone(), link.name.clone()));
        }
        let taken: BTreeSet<String> = self.frames.iter().map(|f| f.name.clone()).collect();
        for (_, link_id, link_name) in &tips {
            if framed.contains(link_id) {
                continue;
            }
            let name = if taken.contains(link_name) {
                format!("{link_name} tip")
            } else {
                link_name.clone()
            };
            self.frames.push(ToolFrame::new(
                stable_id(&self.name, "frame", &name),
                name,
                link_id,
                Transform::IDENTITY,
            ));
        }
        if self.default_frame.is_none() {
            // The tip behind the most manipulator axes is the arm's end. Among
            // equals, a name that says "tool", "tcp" or "flange" is the
            // conventional mounting frame (ROS-Industrial's `tool0`, for
            // one); after that the first by name, so the choice does not
            // depend on the order of the file.
            let conventional = |name: &str| {
                let lower = name.to_ascii_lowercase();
                !(lower.contains("tool") || lower.contains("tcp") || lower.contains("flange"))
            };
            tips.sort_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| conventional(&a.2).cmp(&conventional(&b.2)))
                    .then_with(|| a.2.cmp(&b.2))
            });
            if let Some((_, link_id, _)) = tips.first() {
                self.default_frame = self
                    .frames
                    .iter()
                    .find(|f| f.link_id == *link_id)
                    .map(|f| f.name.clone());
                if tips.len() > 1 {
                    self.warnings.push(format!(
                        "the robot has {} tips; `{}` was taken as the default tool frame",
                        tips.len(),
                        self.default_frame.clone().unwrap_or_default()
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplied_files_are_found_by_reference_or_by_file_name_when_unambiguous() {
        let mut files = BTreeMap::new();
        files.insert("base.stl".to_owned(), vec![1]);
        files.insert("a/link.stl".to_owned(), vec![2]);
        files.insert("b/link.stl".to_owned(), vec![3]);
        assert_eq!(files.load("package://robot/meshes/base.stl"), Some(vec![1]));
        assert_eq!(
            files.load("a/link.stl"),
            Some(vec![2]),
            "an exact match wins"
        );
        assert_eq!(
            files.load("meshes/link.stl"),
            None,
            "two candidates: no guess"
        );
        assert_eq!(files.load("missing.stl"), None);
    }
}
