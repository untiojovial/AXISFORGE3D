//! The application service: one command interface over the whole engine.
//!
//! Every front end — the command line, the WebAssembly build behind the
//! browser viewer, a desktop shell — talks to the engine through
//! [`Session::handle`], which takes a [`Request`] and returns a [`Response`].
//! [`Session::handle_json`] is the same thing over text. Because there is one
//! interface, there is one place where robotics is computed: a front end draws
//! what a [`Snapshot`] tells it and never recomputes a pose.
//!
//! A session owns a validated project and the robot's current configuration.
//! Requests that would leave the project invalid are refused and change
//! nothing.

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

use axisforge_collision::{Attachment, CollisionReport, Scene, SceneOptions};
use axisforge_domain::{
    templates, DomainError, GeometryInstance, JointKind, JointRole, Project, Quat, Shape,
    Transform, Vec3, GROUND_OWNER_ID, SAFETY_NOTICE,
};
use axisforge_grasping::{plan_pick_with, ExtensionInfo, PlanReport, PlannerOptions, Registry};
use axisforge_kinematics::{
    conditioning, forward_kinematics, frame_pose, ik, jacobian, CancellationToken,
    ConditioningReport, IkOptions, IkSolution, KinematicModel, TaskSpace,
};
use axisforge_motion::{
    csv, plan_cycle, program, sampled, simulate_program, urscript, Cycle, CycleOptions,
    CycleReport, SampledTrajectory,
};
use axisforge_project_io::import::{ImportError, ImportOptions};
use axisforge_project_io::object::import_mesh_object;
use axisforge_project_io::simox::{import_simox_object, import_simox_robot};
use axisforge_project_io::urdf::import_urdf;
use axisforge_project_io::ProjectIoError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Smallest singular value below which a pose is flagged near-singular.
const SINGULARITY_THRESHOLD: f64 = 0.01;

/// What can go wrong handling a request.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// The request was well-formed but the engine refused it.
    #[error(transparent)]
    Domain(#[from] DomainError),
    /// A project file could not be read or written.
    #[error(transparent)]
    ProjectIo(#[from] ProjectIoError),
    /// A robot or object description could not be imported.
    #[error(transparent)]
    Import(#[from] ImportError),
    /// The request text was not a valid request.
    #[error("malformed request: {0}")]
    Malformed(String),
}

/// A part carried by the gripper, for drawing and collision checking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Carry {
    /// Identifier of the carried world body.
    pub body_id: String,
    /// Pose of the body in the palm link's frame.
    pub part_in_palm: Transform,
}

impl From<serde_json::Error> for ServiceError {
    fn from(error: serde_json::Error) -> Self {
        Self::Malformed(error.to_string())
    }
}

/// A command for the session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum Request {
    /// Replace the project with the pick-and-place demo cell.
    LoadDemo {
        /// The gripper on the arm: parallel jaws unless asked otherwise.
        #[serde(default)]
        gripper: templates::GripperKind,
        /// Obstacles on the table: none unless asked otherwise.
        #[serde(default)]
        obstacles: templates::Obstacles,
    },
    /// Replace the project with one parsed from project-file text.
    LoadProject {
        /// The file's contents.
        text: String,
    },
    /// Return the current project as project-file text.
    ExportProject,
    /// Describe the robot, its joints and the world's bodies.
    Describe,
    /// List the quality measures and candidate generators that are plugged in.
    ListExtensions,
    /// Set the robot's configuration and return what it looks like.
    SetJoints {
        /// One value per actuated joint, in configuration order.
        values: Vec<f64>,
    },
    /// Evaluate a configuration without storing it.
    Snapshot {
        /// The configuration; the session's current one when omitted.
        values: Option<Vec<f64>>,
        /// A part to draw and check as carried by the gripper.
        carry: Option<Carry>,
        /// Identifier of a part the jaws are closed on but have not lifted:
        /// it stays where it is, and the fingers touching it is not a contact.
        grasping: Option<String>,
        /// A part drawn and checked somewhere else than where it stands: where
        /// a cycle set it down.
        #[serde(default)]
        placed: Option<Placed>,
    },
    /// Solve inverse kinematics for the default tool frame.
    SolveIk {
        /// Target position, metres.
        position: Vec3,
        /// Target orientation as roll, pitch, yaw in radians; position-only when omitted.
        rpy: Option<[f64; 3]>,
    },
    /// Plan picks of a world body from the current configuration.
    PlanPick {
        /// Identifier of the body to pick.
        body_id: String,
        /// Planner settings; defaults when omitted.
        options: Option<PlannerOptions>,
    },
    /// Replace the project with a robot imported from URDF or Simox XML.
    ImportRobot {
        /// `urdf` or `simox`.
        format: String,
        /// The description's text.
        text: String,
        /// Mesh files it refers to, by the reference as written, base64-encoded.
        #[serde(default)]
        files: BTreeMap<String, String>,
        /// Joints that drive an end effector, beyond any the file declares.
        #[serde(default)]
        end_effector_joints: Vec<String>,
        /// Mount one of AXISFORGE's grippers on this tool frame; the robot's
        /// default tool frame when the name is empty.
        #[serde(default)]
        attach_gripper: Option<String>,
        /// Which gripper to mount: parallel jaws unless asked otherwise.
        #[serde(default)]
        gripper_kind: templates::GripperKind,
        /// Place the robot in the demo cell, with its table and parts.
        #[serde(default)]
        into_demo_cell: bool,
    },
    /// Add a part from a Simox manipulation object.
    AddObject {
        /// The `<ManipulationObject>` text.
        text: String,
        /// Mesh files it refers to, base64-encoded.
        #[serde(default)]
        files: BTreeMap<String, String>,
        /// Where to place it, metres.
        position: Vec3,
        /// When given, `position.z` is ignored and the object is lowered or
        /// raised until its lowest point is at this height, metres: resting on
        /// a surface at that height.
        #[serde(default)]
        rest_on: Option<f64>,
    },
    /// Add a part straight from one mesh file (STL, OBJ or COLLADA).
    AddMeshObject {
        /// The file's name; its extension decides the format.
        file: String,
        /// The file's contents, base64-encoded.
        data: String,
        /// The part's name; the file's name without extension when omitted.
        /// A name already taken gets a number added.
        #[serde(default)]
        name: Option<String>,
        /// The file's units: `m`, `mm`, or `auto` (guessed from the size,
        /// and reported) when omitted.
        #[serde(default)]
        units: Option<String>,
        /// Where to place the centre of its base, metres.
        position: Vec3,
        /// As for [`Request::AddObject`].
        #[serde(default)]
        rest_on: Option<f64>,
    },
    /// Return the vertex and index buffers of mesh resources, for drawing.
    GetResources {
        /// Resource ids; every resource when empty.
        #[serde(default)]
        ids: Vec<String>,
    },
    /// Plan the whole pick, transport and place cycle for a part, from the
    /// current configuration and back to it.
    PlanCycle {
        /// Identifier of the part.
        body_id: String,
        /// Where its origin is set down, metres; it keeps its orientation.
        target: Vec3,
        /// Planner settings; defaults when omitted.
        #[serde(default)]
        options: Option<CycleOptions>,
    },
    /// Write the last cycle planned as a program.
    ExportCycle {
        /// `urscript`, `csv` or `json`.
        format: String,
        /// Sampling period for `csv` and `json`, seconds; 8 ms by default.
        #[serde(default)]
        period: Option<f64>,
    },
    /// Move a world body or change its physical properties.
    UpdateBody {
        /// Identifier of the body.
        body_id: String,
        /// New position, metres.
        position: Option<Vec3>,
        /// New rotation about world Z, radians.
        yaw: Option<f64>,
        /// New mass, kilograms.
        mass: Option<f64>,
        /// New friction coefficient.
        friction_coefficient: Option<f64>,
    },
}

/// A part drawn at another pose than where it stands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Placed {
    /// Identifier of the part.
    pub body_id: String,
    /// Its pose, world coordinates.
    pub pose: Transform,
}

/// A planned cycle, ready to draw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleView {
    /// What the planner found.
    pub report: CycleReport,
    /// The cycle sampled every 20 ms, for animating it.
    pub samples: Option<SampledTrajectory>,
    /// The cycle's program run by a controller interpolating every 8 ms,
    /// seconds: a check of the planned cycle time.
    pub program_seconds: Option<f64>,
    /// Display name of the part.
    pub part_name: String,
}

/// A program written for a controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportedProgram {
    /// A file name for it.
    pub file_name: String,
    /// Its media type.
    pub media_type: String,
    /// The text.
    pub text: String,
}

/// The outcome of an import: the new state, and what did not carry over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportReport {
    /// The session after the import.
    pub description: Description,
    /// Everything that could not be carried over exactly.
    pub warnings: Vec<String>,
}

/// A mesh resource's buffers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshData {
    /// Resource id, as referenced by mesh shapes.
    pub id: String,
    /// Vertex positions, `[x, y, z, ...]`, in the mesh's own units.
    pub vertices: Vec<f64>,
    /// Triangle indices, three per face.
    pub indices: Vec<u32>,
}

/// One actuated joint, as a front end needs it for a slider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JointInfo {
    /// Display name.
    pub name: String,
    /// Kind of motion.
    pub kind: JointKind,
    /// What the joint is for.
    pub role: JointRole,
    /// Lower limit, SI units.
    pub lower: f64,
    /// Upper limit, SI units.
    pub upper: f64,
    /// Current value, SI units.
    pub value: f64,
}

/// One world body, as a front end needs it for a list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyInfo {
    /// Identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// `true` when the body can be a pick target.
    pub graspable: bool,
    /// World position, metres.
    pub position: Vec3,
    /// Mass, kilograms.
    pub mass: f64,
    /// Friction coefficient against the gripper pads.
    pub friction_coefficient: f64,
}

/// What the session holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Description {
    /// Project name.
    pub project_name: String,
    /// Robot name.
    pub robot_name: String,
    /// Number of manipulator axes.
    pub axis_count: usize,
    /// Actuated joints, in configuration order.
    pub joints: Vec<JointInfo>,
    /// World bodies.
    pub bodies: Vec<BodyInfo>,
    /// Name of the fitted gripper, if any.
    pub gripper: Option<String>,
    /// Greatest jaw opening, metres, if a gripper is fitted.
    pub gripper_max_opening: Option<f64>,
    /// Fingers that close until they touch, for a hand; zero for a
    /// parallel-jaw gripper or none.
    pub gripper_fingers: usize,
    /// The hand's preshapes, the default first; empty for jaws.
    pub preshapes: Vec<String>,
    /// Ground height, metres.
    pub ground_height: f64,
    /// Names of links that move but have no collision geometry, usually
    /// because their meshes could not be imported. Nothing checks what they
    /// touch, so a front end must say so next to any clearance or plan.
    pub links_without_collision: Vec<String>,
    /// The notice every front end must display alongside results.
    pub safety_notice: String,
}

/// One shape to draw, in world coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Drawable {
    /// Identifier of the owning link or body.
    pub owner_id: String,
    /// Display name of the owner.
    pub owner_name: String,
    /// `true` when the owner is a robot link.
    pub is_link: bool,
    /// The shape.
    pub shape: Shape,
    /// sRGB colour, `#rrggbb`.
    pub color: String,
    /// How metallic the surface looks, `[0, 1]`.
    pub metalness: f64,
    /// How rough the surface looks, `[0, 1]`.
    pub roughness: f64,
    /// `true` when the shape is part of what is normally drawn.
    pub visual: bool,
    /// `true` when the shape is what collision checking uses.
    pub collision: bool,
    /// World pose of the shape.
    pub pose: Transform,
}

/// Everything a front end needs to draw and annotate one configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// The configuration evaluated.
    pub values: Vec<f64>,
    /// Every shape, posed: what is drawn and what is collided, each flagged.
    pub drawables: Vec<Drawable>,
    /// World pose of the default tool frame, when the robot has one.
    pub tool_pose: Option<Transform>,
    /// Contacts and the closest pair.
    pub collision: CollisionReport,
    /// Conditioning of the manipulator Jacobian, when there is a tool frame.
    pub conditioning: Option<ConditioningReport>,
    /// Names of joints outside their limits.
    pub out_of_limits: Vec<String>,
}

/// The answer to a [`Request`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum Response {
    /// What the session holds.
    Description(Description),
    /// Project-file text.
    ProjectText(String),
    /// A drawn configuration.
    Snapshot(Box<Snapshot>),
    /// An inverse-kinematics result.
    Ik(IkSolution),
    /// A pick-planning result.
    Plan(Box<PlanReport>),
    /// The plugged-in extensions.
    Extensions(Vec<ExtensionInfo>),
    /// What an import produced.
    Imported(Box<ImportReport>),
    /// Mesh buffers.
    Resources(Vec<MeshData>),
    /// A planned cycle.
    Cycle(Box<CycleView>),
    /// A program for a controller.
    Program(ExportedProgram),
}

fn drawable(
    owner_id: &str,
    owner_name: &str,
    is_link: bool,
    geometry: &axisforge_domain::GeometryInstance,
    owner_pose: &Transform,
) -> Drawable {
    Drawable {
        owner_id: owner_id.to_owned(),
        owner_name: owner_name.to_owned(),
        is_link,
        shape: geometry.shape.clone(),
        color: geometry.material.color.clone(),
        metalness: geometry.material.metalness,
        roughness: geometry.material.roughness,
        visual: geometry.is_visual(),
        collision: geometry.is_collision(),
        pose: owner_pose.compose(&geometry.origin),
    }
}

/// Decodes base64 file contents supplied with an import.
fn decode_files(
    files: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, Vec<u8>>, ServiceError> {
    use base64::Engine as _;
    files
        .iter()
        .map(|(name, data)| {
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map(|bytes| (name.clone(), bytes))
                .map_err(|e| {
                    ServiceError::Malformed(format!("file `{name}` is not valid base64: {e}"))
                })
        })
        .collect()
}

/// The lowest world height any of a body's geometry reaches, metres.
fn lowest_point(
    body: &axisforge_domain::WorldBody,
    resources: &[axisforge_domain::Resource],
) -> Result<f64, ServiceError> {
    let mut lowest = f64::INFINITY;
    for geometry in &body.geometries {
        let vertices = match &geometry.shape {
            Shape::Mesh { resource_id, .. } => resources
                .iter()
                .find(|r| r.id == *resource_id)
                .map(|r| r.vertices.as_slice()),
            _ => None,
        };
        let convex = axisforge_geometry::body_from_geometry(geometry, &body.pose, vertices)?;
        // The support point straight down, not the bounding box: a mesh's box
        // is symmetric about its origin, and a part whose origin is at its
        // base would stand that far above the surface.
        let down = convex.core_support(&Vec3::new(0.0, 0.0, -1.0)).z - convex.inflation;
        lowest = lowest.min(down);
    }
    if lowest.is_finite() {
        Ok(lowest)
    } else {
        Err(
            DomainError::unsupported(format!("body '{}' has no geometry to rest on", body.name))
                .into(),
        )
    }
}

/// Mounts one of AXISFORGE's grippers on `mount`, saying so in `warnings`.
fn mount_gripper(
    robot: &mut axisforge_domain::RobotSystem,
    mount: &axisforge_domain::ToolFrame,
    kind: templates::GripperKind,
    warnings: &mut Vec<String>,
) -> Result<(), ServiceError> {
    match kind {
        templates::GripperKind::Parallel => {
            templates::attach_parallel_gripper(
                robot,
                &mount.id,
                &templates::GripperSpec::default(),
            )?;
            warnings.push(format!(
                "AXISFORGE's parallel gripper was mounted on `{}`; its jaws close along that \
                 frame's Y and it approaches along its Z",
                mount.name
            ));
        }
        templates::GripperKind::ThreeFinger => {
            templates::attach_three_finger_gripper(
                robot,
                &mount.id,
                &templates::HandSpec::three_finger_gripper(),
            )?;
            warnings.push(format!(
                "AXISFORGE's three-finger gripper was mounted on `{}`; it approaches along \
                 that frame's Z",
                mount.name
            ));
        }
        templates::GripperKind::Hand => {
            templates::attach_three_finger_hand(
                robot,
                &mount.id,
                &templates::HandSpec::three_finger_hand(),
            )?;
            warnings.push(format!(
                "AXISFORGE's three-finger hand was mounted on `{}`; it approaches along that \
                 frame's Z",
                mount.name
            ));
        }
    }
    Ok(())
}

/// Mounts AXISFORGE's parallel gripper and places the robot in the demo cell,
/// as requested, so an imported robot can plan picks.
///
/// `attach_gripper` names the tool frame to mount on; an empty name means the
/// robot's default tool frame. Notes on what was done go to `warnings`.
///
/// # Errors
///
/// When the robot already has a gripper, the frame does not exist, or the
/// result does not validate.
pub fn prepare_import(
    mut project: Project,
    attach_gripper: Option<(&str, templates::GripperKind)>,
    into_demo_cell: bool,
    warnings: &mut Vec<String>,
) -> Result<Project, ServiceError> {
    if let Some((frame, kind)) = attach_gripper {
        if project.robot.gripper.is_some() {
            return Err(DomainError::unsupported("the robot already has a gripper").into());
        }
        let mount = if frame.is_empty() {
            project.robot.default_tool_frame().cloned()
        } else {
            project
                .robot
                .tool_frames
                .iter()
                .find(|f| f.name == frame)
                .cloned()
        }
        .ok_or_else(|| {
            DomainError::unsupported(format!(
                "no tool frame `{frame}` to mount the gripper on; frames: {}",
                project
                    .robot
                    .tool_frames
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        mount_gripper(&mut project.robot, &mount, kind, warnings)?;
        templates::register_gripper_exemptions(&mut project.environment, &project.robot);
    }
    if into_demo_cell {
        let cell = templates::pick_and_place_cell()?;
        project.environment = cell.environment;
        // The demo cell's exemptions name the demo robot's links; keep only the
        // ones about the world, and add this robot's own.
        let known: std::collections::BTreeSet<String> = project
            .environment
            .bodies
            .iter()
            .map(|b| b.id.clone())
            .chain([GROUND_OWNER_ID.to_owned()])
            .collect();
        let kept: Vec<_> = project
            .environment
            .ignored_pairs
            .pairs()
            .iter()
            .filter(|p| known.contains(&p.first) && known.contains(&p.second))
            .cloned()
            .collect();
        project.environment.ignored_pairs = axisforge_domain::IgnoredPairs::new();
        for pair in kept {
            project
                .environment
                .ignored_pairs
                .ignore(&pair.first, &pair.second, pair.reason);
        }
        // Descriptions often hang the base under a `world` link with a fixed
        // joint, so every link welded to the root stands on the floor with it.
        for link in links_fixed_to_root(&project.robot) {
            project.environment.ignored_pairs.ignore(
                &link,
                GROUND_OWNER_ID,
                "the robot's base rests on the floor",
            );
        }
        templates::register_gripper_exemptions(&mut project.environment, &project.robot);
    }
    project.validate()?;
    Ok(project)
}

/// Names of the links that an actuated joint moves and that carry no
/// collision geometry.
pub fn links_without_collision(robot: &axisforge_domain::RobotSystem) -> Vec<String> {
    robot
        .joints
        .iter()
        .filter(|joint| joint.is_actuated())
        .filter_map(|joint| robot.link(&joint.child_link_id))
        .filter(|link| !link.geometries.iter().any(GeometryInstance::is_collision))
        .map(|link| link.name.clone())
        .collect()
}

/// The root link and every link joined to it through fixed joints only.
fn links_fixed_to_root(robot: &axisforge_domain::RobotSystem) -> Vec<String> {
    let mut found = vec![robot.root_link_id.clone()];
    let mut next = 0;
    while next < found.len() {
        let link = found[next].clone();
        next += 1;
        for joint in robot.joints.iter().filter(|j| !j.is_actuated()) {
            let other = if joint.parent_link_id == link {
                &joint.child_link_id
            } else if joint.child_link_id == link {
                &joint.parent_link_id
            } else {
                continue;
            };
            if !found.contains(other) {
                found.push(other.clone());
            }
        }
    }
    found
}

/// The demo arm's home posture: elbow up, the gripper pointing down above
/// the table and clear of everything, with a hand in its default preshape.
/// `None` for a robot that is not a six-axis arm.
pub fn demo_home(project: &Project, model: &KinematicModel) -> Option<Vec<f64>> {
    const ARM: [f64; 6] = [0.0, 0.3, 1.0, 0.0, 1.84, 0.0];
    let arm = model.manipulator_value_indices();
    if arm.len() != ARM.len() {
        return None;
    }
    let mut values = model.clamped(&vec![0.0; model.dof()]);
    for (slot, value) in arm.iter().zip(ARM) {
        values[*slot] = value;
    }
    if let Some(preshape) = project
        .robot
        .gripper
        .as_ref()
        .and_then(|g| g.preshape(None))
    {
        for (joint_id, value) in &preshape.joint_values {
            if let Some(slot) = model
                .joint_index(joint_id)
                .and_then(|j| model.joints()[j].value_index)
            {
                values[slot] = *value;
            }
        }
    }
    Some(values)
}

/// A project, a robot configuration, and the engine wired around them.
#[derive(Debug, Clone)]
pub struct Session {
    project: Project,
    model: KinematicModel,
    values: Vec<f64>,
    registry: Registry,
    /// The last cycle planned, and its part's name, for exporting.
    cycle: Option<(Cycle, String)>,
}

impl Session {
    /// A session holding the demo cell in an elbow-up home posture, gripper
    /// pointing down above the table and clear of everything.
    ///
    /// # Errors
    /// Never in practice; the template is validated by the test suite.
    pub fn demo() -> Result<Self, ServiceError> {
        Self::demo_with(templates::GripperKind::Parallel)
    }

    /// The demo cell with the chosen gripper, at the same home posture.
    ///
    /// # Errors
    /// Never in practice; the templates are validated by the test suite.
    pub fn demo_with(gripper: templates::GripperKind) -> Result<Self, ServiceError> {
        Self::demo_cell(gripper, templates::Obstacles::None)
    }

    /// The demo cell with the chosen gripper and obstacles, at the same home
    /// posture, which the obstacles leave clear.
    ///
    /// # Errors
    /// Never in practice; the templates are validated by the test suite.
    pub fn demo_cell(
        gripper: templates::GripperKind,
        obstacles: templates::Obstacles,
    ) -> Result<Self, ServiceError> {
        let mut session = Self::with_project(templates::demo_cell(gripper, obstacles)?)?;
        if let Some(home) = demo_home(&session.project, &session.model) {
            session.values = home;
        }
        Ok(session)
    }

    /// A session holding `project`, with the robot at zero.
    ///
    /// # Errors
    /// When the project does not validate.
    pub fn with_project(project: Project) -> Result<Self, ServiceError> {
        project.validate()?;
        let model = KinematicModel::from_robot(&project.robot)?;
        let values = model.clamped(&vec![0.0; model.dof()]);
        Ok(Self {
            project,
            model,
            values,
            registry: Registry::standard(),
            cycle: None,
        })
    }

    /// Swaps in another session's project while keeping this one's registered
    /// extensions: loading a file must not unplug a library.
    fn replace(&mut self, other: Self) {
        let registry = std::mem::take(&mut self.registry);
        *self = other;
        self.registry = registry;
    }

    /// The extension registry, for a host to plug libraries into.
    pub fn registry_mut(&mut self) -> &mut Registry {
        &mut self.registry
    }

    /// The project.
    pub fn project(&self) -> &Project {
        &self.project
    }

    /// The current configuration.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Handles a request given as JSON text and answers in JSON text.
    ///
    /// The answer is `{"ok":true,"kind":...,"data":...}` or
    /// `{"ok":false,"error":"..."}`; this function itself never fails, so a
    /// front end has exactly one shape of reply to handle.
    pub fn handle_json(&mut self, text: &str) -> String {
        let result = serde_json::from_str::<Request>(text)
            .map_err(ServiceError::from)
            .and_then(|request| self.handle(request, &CancellationToken::new()))
            .and_then(|response| serde_json::to_value(response).map_err(ServiceError::from));
        let reply = match result {
            Ok(mut value) => {
                if let Some(object) = value.as_object_mut() {
                    object.insert("ok".to_owned(), true.into());
                }
                value
            }
            Err(error) => serde_json::json!({ "ok": false, "error": error.to_string() }),
        };
        reply.to_string()
    }

    /// Handles a request.
    ///
    /// # Errors
    /// When the engine refuses the request; the session is then unchanged.
    #[allow(clippy::too_many_lines)] // One arm per request, each a call; split, it would only be harder to scan.
    pub fn handle(
        &mut self,
        request: Request,
        cancel: &CancellationToken,
    ) -> Result<Response, ServiceError> {
        match request {
            Request::LoadDemo { gripper, obstacles } => {
                self.replace(Self::demo_cell(gripper, obstacles)?);
                Ok(Response::Description(self.describe()))
            }
            Request::LoadProject { text } => {
                self.replace(Self::with_project(axisforge_project_io::from_json(&text)?)?);
                Ok(Response::Description(self.describe()))
            }
            Request::ExportProject => Ok(Response::ProjectText(axisforge_project_io::to_json(
                &self.project,
            )?)),
            Request::Describe => Ok(Response::Description(self.describe())),
            Request::ListExtensions => Ok(Response::Extensions(self.registry.list())),
            Request::SetJoints { values } => {
                let snapshot = self.snapshot(&values, None, None)?;
                self.values = values;
                Ok(Response::Snapshot(Box::new(snapshot)))
            }
            Request::Snapshot {
                values,
                carry,
                grasping,
                placed,
            } => {
                let values = values.unwrap_or_else(|| self.values.clone());
                let snapshot = self.snapshot_placed(
                    &values,
                    carry.as_ref(),
                    grasping.as_deref(),
                    placed.as_ref(),
                )?;
                Ok(Response::Snapshot(Box::new(snapshot)))
            }
            Request::PlanCycle {
                body_id,
                target,
                options,
            } => self.plan_cycle(&body_id, target, &options.unwrap_or_default(), cancel),
            Request::ExportCycle { format, period } => self.export_cycle(&format, period),
            Request::SolveIk { position, rpy } => {
                let frame = self.default_frame()?;
                let (rotation, position_only) = match rpy {
                    Some([roll, pitch, yaw]) => (Quat::from_euler_rpy(roll, pitch, yaw), false),
                    None => (Quat::IDENTITY, true),
                };
                let options = IkOptions {
                    position_only,
                    ..IkOptions::default()
                };
                let target = Transform::new(position, rotation);
                let solution =
                    ik::solve(&self.model, frame, &target, &self.values, &options, cancel)?;
                Ok(Response::Ik(solution))
            }
            Request::PlanPick { body_id, options } => {
                let report = plan_pick_with(
                    &self.registry,
                    &self.project,
                    &body_id,
                    &self.values,
                    &options.unwrap_or_default(),
                    cancel,
                )?;
                Ok(Response::Plan(Box::new(report)))
            }
            Request::ImportRobot {
                format,
                text,
                files,
                end_effector_joints,
                attach_gripper,
                gripper_kind,
                into_demo_cell,
            } => self.import_robot(
                &format,
                &text,
                &files,
                end_effector_joints,
                attach_gripper.as_deref().map(|frame| (frame, gripper_kind)),
                into_demo_cell,
            ),
            Request::AddObject {
                text,
                files,
                position,
                rest_on,
            } => {
                let response = self.add_object(&text, &files, position, rest_on)?;
                self.cycle = None;
                Ok(response)
            }
            Request::AddMeshObject {
                file,
                data,
                name,
                units,
                position,
                rest_on,
            } => {
                let response =
                    self.add_mesh_object(&file, &data, name, units.as_deref(), position, rest_on)?;
                self.cycle = None;
                Ok(response)
            }
            Request::GetResources { ids } => Ok(Response::Resources(self.mesh_data(&ids))),
            Request::UpdateBody {
                body_id,
                position,
                yaw,
                mass,
                friction_coefficient,
            } => {
                let response =
                    self.update_body(body_id, position, yaw, mass, friction_coefficient)?;
                self.cycle = None;
                Ok(response)
            }
        }
    }

    /// [`Session::snapshot`] with a part moved to where a cycle set it
    /// down, for this snapshot only: it is put back whatever happens.
    fn snapshot_placed(
        &mut self,
        values: &[f64],
        carry: Option<&Carry>,
        grasping: Option<&str>,
        placed: Option<&Placed>,
    ) -> Result<Snapshot, ServiceError> {
        let moved = match placed {
            Some(placed) => {
                let body = self
                    .project
                    .environment
                    .bodies
                    .iter_mut()
                    .find(|b| b.id == placed.body_id)
                    .ok_or_else(|| {
                        DomainError::dangling("world body", placed.body_id.clone(), "placed")
                    })?;
                let before = body.pose;
                body.pose = placed.pose;
                Some((placed.body_id.clone(), before))
            }
            None => None,
        };
        let snapshot = self.snapshot(values, carry, grasping);
        if let Some((id, before)) = moved {
            if let Some(body) = self
                .project
                .environment
                .bodies
                .iter_mut()
                .find(|b| b.id == id)
            {
                body.pose = before;
            }
        }
        snapshot
    }

    fn plan_cycle(
        &mut self,
        body_id: &str,
        target: Vec3,
        options: &CycleOptions,
        cancel: &CancellationToken,
    ) -> Result<Response, ServiceError> {
        let report = plan_cycle(
            &self.project,
            body_id,
            target,
            &self.values,
            options,
            cancel,
        )?;
        let part_name = self
            .project
            .environment
            .body(body_id)
            .map_or_else(|| body_id.to_owned(), |b| b.name.clone());
        let (samples, program_seconds) = match &report.cycle {
            Some(cycle) => (
                Some(sampled(cycle, 0.02)?),
                Some(simulate_program(&program(cycle), 0.008)),
            ),
            None => (None, None),
        };
        self.cycle = report.cycle.clone().map(|c| (c, part_name.clone()));
        Ok(Response::Cycle(Box::new(CycleView {
            report,
            samples,
            program_seconds,
            part_name,
        })))
    }

    fn export_cycle(&self, format: &str, period: Option<f64>) -> Result<Response, ServiceError> {
        let (cycle, part) = self.cycle.as_ref().ok_or_else(|| {
            DomainError::unsupported("no cycle has been planned in this session".to_owned())
        })?;
        let stem = format!(
            "axisforge_cycle_{}",
            part.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
        );
        let period = period.unwrap_or(0.008);
        let program = match format {
            "urscript" => ExportedProgram {
                file_name: format!("{stem}.script"),
                media_type: "text/plain".to_owned(),
                text: urscript(cycle, part)?,
            },
            "csv" => ExportedProgram {
                file_name: format!("{stem}.csv"),
                media_type: "text/csv".to_owned(),
                text: csv(&sampled(cycle, period)?),
            },
            "json" => ExportedProgram {
                file_name: format!("{stem}.json"),
                media_type: "application/json".to_owned(),
                text: serde_json::to_string_pretty(&sampled(cycle, period)?)?,
            },
            other => {
                return Err(DomainError::unsupported(format!(
                    "unknown program format `{other}`; `urscript`, `csv` and `json` are supported"
                ))
                .into())
            }
        };
        Ok(Response::Program(program))
    }

    fn update_body(
        &mut self,
        body_id: String,
        position: Option<Vec3>,
        yaw: Option<f64>,
        mass: Option<f64>,
        friction_coefficient: Option<f64>,
    ) -> Result<Response, ServiceError> {
        // Edit a copy and validate it, so a refused edit changes nothing.
        let mut project = self.project.clone();
        let body = project
            .environment
            .bodies
            .iter_mut()
            .find(|b| b.id == body_id)
            .ok_or_else(|| DomainError::dangling("world body", body_id, "update_body"))?;
        if let Some(position) = position {
            body.pose.translation = position;
        }
        if let Some(yaw) = yaw {
            body.pose.rotation = Quat::from_axis_angle(&Vec3::Z, yaw)?;
        }
        if let Some(mass) = mass {
            body.physical.mass = mass;
        }
        if let Some(mu) = friction_coefficient {
            body.physical.friction_coefficient = mu;
        }
        project.validate()?;
        self.project = project;
        Ok(Response::Description(self.describe()))
    }

    fn import_robot(
        &mut self,
        format: &str,
        text: &str,
        files: &BTreeMap<String, String>,
        end_effector_joints: Vec<String>,
        attach_gripper: Option<(&str, templates::GripperKind)>,
        into_demo_cell: bool,
    ) -> Result<Response, ServiceError> {
        let source = decode_files(files)?;
        let options = ImportOptions {
            end_effector_joints,
        };
        let imported = match format {
            "urdf" => import_urdf(text, &source, &options)?,
            "simox" => import_simox_robot(text, &source, &options)?,
            other => {
                return Err(DomainError::unsupported(format!(
                    "unknown description format `{other}`; `urdf` and `simox` are supported"
                ))
                .into())
            }
        };
        let mut warnings = imported.warnings;
        let project = prepare_import(
            imported.project,
            attach_gripper,
            into_demo_cell,
            &mut warnings,
        )?;
        self.replace(Self::with_project(project)?);
        Ok(Response::Imported(Box::new(ImportReport {
            description: self.describe(),
            warnings,
        })))
    }

    /// The buffers of the resources named in `ids`, or of all when empty.
    fn mesh_data(&self, ids: &[String]) -> Vec<MeshData> {
        self.project
            .resources
            .iter()
            .filter(|r| ids.is_empty() || ids.contains(&r.id))
            .map(|r| MeshData {
                id: r.id.clone(),
                vertices: r.vertices.clone(),
                indices: r.indices.clone(),
            })
            .collect()
    }

    fn add_mesh_object(
        &mut self,
        file: &str,
        data: &str,
        name: Option<String>,
        units: Option<&str>,
        position: Vec3,
        rest_on: Option<f64>,
    ) -> Result<Response, ServiceError> {
        let scale = match units.unwrap_or("auto") {
            "auto" => None,
            "m" => Some(1.0),
            "mm" => Some(0.001),
            other => {
                return Err(DomainError::unsupported(format!(
                    "units `{other}`: use `m`, `mm` or `auto`"
                ))
                .into())
            }
        };
        let mut files = BTreeMap::new();
        files.insert(file.to_owned(), data.to_owned());
        let bytes = decode_files(&files)?.remove(file).unwrap_or_default();
        let stem = file
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(file)
            .rsplit_once('.')
            .map_or(file, |(stem, _)| stem);
        let base = name
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| stem.to_owned());
        // A free name, so adding the same file twice gives two parts.
        let mut name = base.clone();
        let mut n = 2;
        while self.project.environment.body_by_name(&name).is_some() {
            name = format!("{base} {n}");
            n += 1;
        }
        let imported = import_mesh_object(&name, file, &bytes, scale)?;
        self.place_object(imported, position, rest_on)
    }

    fn add_object(
        &mut self,
        text: &str,
        files: &BTreeMap<String, String>,
        position: Vec3,
        rest_on: Option<f64>,
    ) -> Result<Response, ServiceError> {
        let source = decode_files(files)?;
        let imported = import_simox_object(text, &source)?;
        self.place_object(imported, position, rest_on)
    }

    /// Adds an imported object at `position`, standing on `rest_on` if given.
    fn place_object(
        &mut self,
        mut imported: axisforge_project_io::simox::ImportedObject,
        position: Vec3,
        rest_on: Option<f64>,
    ) -> Result<Response, ServiceError> {
        if self
            .project
            .environment
            .body_by_name(&imported.body.name)
            .is_some()
        {
            return Err(DomainError::DuplicateName {
                kind: "world body",
                name: imported.body.name,
            }
            .into());
        }
        imported.body.pose.translation = position;
        if let Some(surface) = rest_on {
            let lowest = lowest_point(&imported.body, &imported.resources)?;
            imported.body.pose.translation.z += surface - lowest;
        }
        // Edit a copy and validate it, so a refused import changes nothing.
        let mut project = self.project.clone();
        project.environment.bodies.push(imported.body);
        project.resources.extend(imported.resources);
        project.validate()?;
        self.project = project;
        Ok(Response::Imported(Box::new(ImportReport {
            description: self.describe(),
            warnings: imported.warnings,
        })))
    }

    fn default_frame(&self) -> Result<usize, DomainError> {
        self.model.default_frame().ok_or_else(|| {
            DomainError::unsupported(format!(
                "robot '{}' has several tool frames and no default; declare one",
                self.project.robot.name
            ))
        })
    }

    fn describe(&self) -> Description {
        let limits = self.model.limits();
        let joints = (0..self.model.dof())
            .filter_map(|index| {
                let joint = self.model.joint_for_value(index)?;
                Some(JointInfo {
                    name: joint.name.clone(),
                    kind: joint.kind,
                    role: joint.role,
                    lower: limits[index].lower,
                    upper: limits[index].upper,
                    value: self.values[index],
                })
            })
            .collect();
        let bodies = self
            .project
            .environment
            .bodies
            .iter()
            .map(|body| BodyInfo {
                id: body.id.clone(),
                name: body.name.clone(),
                graspable: body.is_graspable(),
                position: body.pose.translation,
                mass: body.physical.mass,
                friction_coefficient: body.physical.friction_coefficient,
            })
            .collect();
        let gripper = self.project.robot.gripper.as_ref();
        Description {
            project_name: self.project.name.clone(),
            robot_name: self.project.robot.name.clone(),
            axis_count: self.project.robot.axis_count(),
            joints,
            bodies,
            gripper: gripper.map(|g| g.name.clone()),
            gripper_max_opening: gripper.map(|g| g.max_opening),
            gripper_fingers: gripper.map_or(0, |g| g.actors.len()),
            preshapes: gripper
                .map(|g| g.preshapes.iter().map(|p| p.name.clone()).collect())
                .unwrap_or_default(),
            ground_height: self.project.environment.ground_height,
            links_without_collision: links_without_collision(&self.project.robot),
            safety_notice: SAFETY_NOTICE.to_owned(),
        }
    }

    fn snapshot(
        &self,
        values: &[f64],
        carry: Option<&Carry>,
        grasping: Option<&str>,
    ) -> Result<Snapshot, ServiceError> {
        let pose = forward_kinematics(&self.model, values)?;
        let attachment = match (carry, &self.project.robot.gripper) {
            (Some(carry), Some(gripper)) => Some(Attachment {
                body_id: carry.body_id.clone(),
                link_id: gripper.palm_link_id.clone(),
                relative_pose: carry.part_in_palm,
            }),
            (Some(_), None) => {
                return Err(self.project.robot.require_gripper().unwrap_err().into())
            }
            (None, _) => None,
        };

        let mut drawables = Vec::new();
        for link in &self.project.robot.links {
            let index = self
                .model
                .link_index(&link.id)
                .ok_or_else(|| DomainError::dangling("link", link.id.clone(), "snapshot"))?;
            for geometry in &link.geometries {
                drawables.push(drawable(
                    &link.id,
                    &link.name,
                    true,
                    geometry,
                    &pose.link(index),
                ));
            }
        }
        for body in &self.project.environment.bodies {
            let world = match attachment.as_ref().filter(|a| a.body_id == body.id) {
                Some(attachment) => {
                    let palm = self.model.link_index(&attachment.link_id).ok_or_else(|| {
                        DomainError::dangling("link", attachment.link_id.clone(), "snapshot")
                    })?;
                    pose.link(palm).compose(&attachment.relative_pose)
                }
                None => body.pose,
            };
            for geometry in &body.geometries {
                drawables.push(drawable(&body.id, &body.name, false, geometry, &world));
            }
        }

        let mut extra_ignored = Vec::new();
        if let Some(body_id) = grasping {
            if self.project.environment.body(body_id).is_none() {
                return Err(DomainError::dangling("world body", body_id, "snapshot").into());
            }
            for finger in &self.project.robot.require_gripper()?.finger_link_ids {
                extra_ignored.push((finger.clone(), body_id.to_owned()));
            }
        }
        let options = SceneOptions {
            attachment,
            extra_ignored,
        };
        let collision = Scene::build(&self.project, &self.model, &pose, &options)?.check(0.0);
        let frame = self.model.default_frame();
        let conditioning = match frame {
            Some(frame) => Some(conditioning(
                &jacobian(&self.model, &pose, frame)?,
                &self.model.manipulator_value_indices(),
                TaskSpace::Pose,
                SINGULARITY_THRESHOLD,
            )),
            None => None,
        };
        let limits = self.model.limits();
        let out_of_limits = (0..self.model.dof())
            .filter(|index| !limits[*index].contains(values[*index], 1.0e-9))
            .filter_map(|index| self.model.joint_for_value(index).map(|j| j.name.clone()))
            .collect();

        Ok(Snapshot {
            values: values.to_vec(),
            drawables,
            tool_pose: frame.map(|frame| frame_pose(&self.model, &pose, frame)),
            collision,
            conditioning,
            out_of_limits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(session: &mut Session, json: &str) -> serde_json::Value {
        serde_json::from_str(&session.handle_json(json)).expect("replies are always JSON")
    }

    #[test]
    fn describe_reports_axes_joints_and_parts() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"describe"}"#);
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["data"]["axisCount"], 6);
        assert_eq!(reply["data"]["joints"].as_array().unwrap().len(), 8);
        assert_eq!(reply["data"]["joints"][6]["role"], "end_effector");
        let graspable: Vec<&str> = reply["data"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["graspable"] == true)
            .map(|b| b["name"].as_str().unwrap())
            .collect();
        assert_eq!(graspable, ["cube", "dowel"]);
        assert!(reply["data"]["safetyNotice"]
            .as_str()
            .unwrap()
            .contains("not a functional-safety"));
    }

    #[test]
    fn a_snapshot_draws_every_link_and_body() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"snapshot"}"#);
        assert_eq!(reply["kind"], "snapshot");
        let drawables = reply["data"]["drawables"].as_array().unwrap();
        // What collision checking sees: base, six links, palm, two fingers,
        // the table top and legs, and the two parts.
        assert_eq!(
            drawables.iter().filter(|d| d["collision"] == true).count(),
            17
        );
        // Every owner is drawn, and each drawn shape carries its finish.
        for owner in [
            "base",
            "axis 3 link",
            "gripper palm",
            "gripper left finger",
            "table",
            "cube",
        ] {
            assert!(
                drawables
                    .iter()
                    .any(|d| d["ownerName"] == owner && d["visual"] == true),
                "{owner} is not drawn"
            );
        }
        assert!(drawables
            .iter()
            .all(|d| d["roughness"].is_number() && d["metalness"].is_number()));
        assert_eq!(
            reply["data"]["collision"]["contacts"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert!(reply["data"]["toolPose"].is_object());
    }

    #[test]
    fn a_bad_configuration_is_refused_and_changes_nothing() {
        let mut session = Session::demo().unwrap();
        let before = session.values().to_vec();
        let reply = call(&mut session, r#"{"op":"set_joints","values":[0,0]}"#);
        assert_eq!(reply["ok"], false);
        assert!(reply["error"].as_str().unwrap().contains("expected 8"));
        assert_eq!(session.values(), before);
    }

    #[test]
    fn malformed_text_gets_an_error_reply_not_a_panic() {
        let mut session = Session::demo().unwrap();
        assert_eq!(call(&mut session, "not json")["ok"], false);
        assert_eq!(
            call(&mut session, r#"{"op":"launch_missiles"}"#)["ok"],
            false
        );
    }

    #[test]
    fn planning_and_carrying_work_end_to_end() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}"}}"#),
        );
        assert_eq!(reply["kind"], "plan", "{reply}");
        let plan = &reply["data"]["plans"][0];
        assert!(plan.is_object(), "{}", reply["data"]["rejections"]);

        // Draw the lifted state with the part carried: no contacts, part raised.
        let request = serde_json::json!({
            "op": "snapshot",
            "values": plan["lift"],
            "carry": { "bodyId": cube, "partInPalm": plan["partInPalm"] },
        });
        let lifted = call(&mut session, &request.to_string());
        assert_eq!(lifted["ok"], true, "{lifted}");
        assert_eq!(
            lifted["data"]["collision"]["contacts"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        let z = lifted["data"]["drawables"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["ownerName"] == "cube")
            .unwrap()["pose"]["translation"]["z"]
            .as_f64()
            .unwrap();
        assert!(
            (z - 0.54).abs() < 1e-3,
            "the cube should be 10 cm above its rest height, z = {z}"
        );
    }

    #[test]
    fn closing_on_a_part_is_a_contact_unless_it_is_being_grasped() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}"}}"#),
        );
        let closed = &reply["data"]["plans"][0]["graspClosed"];
        let contacts = |session: &mut Session, extra: serde_json::Value| {
            let mut request = serde_json::json!({ "op": "snapshot", "values": closed });
            request
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let reply = call(session, &request.to_string());
            assert_eq!(reply["ok"], true, "{reply}");
            reply["data"]["collision"]["contacts"]
                .as_array()
                .unwrap()
                .len()
        };
        // Jaws on the part, unannounced: the fingers are touching something.
        assert!(contacts(&mut session, serde_json::json!({})) > 0);
        // Announced as a grasp: touching the part is the point.
        assert_eq!(
            contacts(&mut session, serde_json::json!({ "grasping": cube })),
            0
        );
        let unknown = call(&mut session, r#"{"op":"snapshot","grasping":"ghost"}"#);
        assert_eq!(unknown["ok"], false);
    }

    #[test]
    fn moving_a_part_out_of_reach_changes_the_plan_outcome() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let moved = call(
            &mut session,
            &format!(
                r#"{{"op":"update_body","bodyId":"{cube}","position":{{"x":2.5,"y":0,"z":0.44}}}}"#
            ),
        );
        assert_eq!(moved["ok"], true, "{moved}");
        let options = r#"{"ik":{"positionTolerance":1e-5,"orientationTolerance":1e-4,"maxIterations":40,"restarts":1,"rngSeed":1,"damping":0.05,"positionOnly":false}}"#;
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}","options":{options}}}"#),
        );
        assert_eq!(reply["data"]["plans"].as_array().unwrap().len(), 0);
        assert_eq!(
            reply["data"]["rejections"][0]["reason"],
            "pregrasp_unreachable"
        );
    }

    #[test]
    fn an_invalid_edit_is_refused_and_the_project_is_unchanged() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let before = session.project().clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"update_body","bodyId":"{cube}","mass":-3}}"#),
        );
        assert_eq!(reply["ok"], false);
        assert_eq!(session.project(), &before);
    }

    #[test]
    fn export_then_load_round_trips() {
        let mut session = Session::demo().unwrap();
        let text = call(&mut session, r#"{"op":"export_project"}"#)["data"]
            .as_str()
            .unwrap()
            .to_owned();
        let before = session.project().clone();
        let reply = call(
            &mut session,
            &serde_json::json!({"op":"load_project","text":text}).to_string(),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(session.project(), &before);
    }

    #[test]
    fn extensions_are_listed_and_selectable_by_name() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"list_extensions"}"#);
        let ids: Vec<&str> = reply["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            [
                "antipodal",
                "wrench_space",
                "primitive_antipodal",
                "surface_normal"
            ]
        );

        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_pick","bodyId":"{cube}","options":{{"qualityMeasure":"antipodal","nrGrasps":1}}}}"#
            ),
        );
        assert_eq!(
            reply["data"]["plans"][0]["quality"]["measure"], "antipodal",
            "{reply}"
        );
        let reply = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_pick","bodyId":"{cube}","options":{{"qualityMeasure":"nope"}}}}"#
            ),
        );
        assert_eq!(reply["ok"], false);
        assert!(reply["error"].as_str().unwrap().contains("wrench_space"));
    }

    #[test]
    fn a_plugged_in_library_survives_loading_another_project() {
        use axisforge_grasping::{Measurement, QualityInput, QualityMeasure};
        struct Flat;
        impl QualityMeasure for Flat {
            fn id(&self) -> &'static str {
                "flat"
            }
            fn summary(&self) -> &'static str {
                "Always the same."
            }
            fn measure(&self, _: &QualityInput<'_>) -> Measurement {
                Measurement {
                    quality: 1.0,
                    normalized: 1.0,
                    force_closure: true,
                }
            }
        }
        let mut session = Session::demo().unwrap();
        session
            .registry_mut()
            .register_measure(std::sync::Arc::new(Flat))
            .unwrap();
        call(&mut session, r#"{"op":"load_demo"}"#);
        let reply = call(&mut session, r#"{"op":"list_extensions"}"#);
        assert!(
            reply["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["id"] == "flat"),
            "{reply}"
        );
    }

    #[test]
    fn the_wasm_build_would_refuse_a_time_budget_but_native_accepts_it() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}","options":{{"timeoutSeconds":30}}}}"#),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["data"]["timedOut"], false);
    }

    const UR5: &str = include_str!("../../project-io/tests/data/ur5/ur5_robot.urdf");

    fn import_ur5(session: &mut Session) -> serde_json::Value {
        let request = serde_json::json!({
            "op": "import_robot",
            "format": "urdf",
            "text": UR5,
            "attachGripper": "",
            "intoDemoCell": true,
        });
        call(session, &request.to_string())
    }

    #[test]
    fn an_imported_ur5_gets_a_gripper_and_a_cell() {
        let mut session = Session::demo().unwrap();
        let reply = import_ur5(&mut session);
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["kind"], "imported");
        let d = &reply["data"]["description"];
        assert_eq!(d["robotName"], "ur5");
        assert_eq!(d["axisCount"], 6);
        assert_eq!(
            d["joints"].as_array().unwrap().len(),
            8,
            "six axes and two jaws"
        );
        assert_eq!(d["gripper"], "gripper");
        let warnings = reply["data"]["warnings"].as_array().unwrap();
        assert!(
            warnings
                .iter()
                .any(|w| w.as_str().unwrap().contains("mounted on `tool0`")),
            "the conventional mounting frame is preferred: {warnings:?}"
        );
        let names: Vec<&str> = d["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["table", "cube", "dowel"]);
    }

    #[test]
    fn an_imported_ur5_can_plan_a_pick() {
        let mut session = Session::demo().unwrap();
        import_ur5(&mut session);
        // Elbow up over the table, gripper pointing down.
        let ready = serde_json::json!({
            "op": "set_joints",
            "values": [0.0, -1.2, 1.6, -1.97, -1.57, 0.0, 0.0, 0.0],
        });
        assert_eq!(call(&mut session, &ready.to_string())["ok"], true);
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}"}}"#),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert!(
            !reply["data"]["plans"].as_array().unwrap().is_empty(),
            "rejections: {}",
            reply["data"]["rejections"]
        );
    }

    #[test]
    fn mesh_files_travel_as_base64_and_come_back_as_buffers() {
        use base64::Engine as _;
        let urdf = r#"<robot name="m"><link name="a"><visual><geometry><mesh filename="package://m/part.stl"/></geometry></visual></link>
            <link name="b"/><joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 0.1"/><limit lower="-1" upper="1"/></joint></robot>"#;
        let stl = "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 0.1 0 0\nvertex 0 0.1 0\nendloop\nendfacet\nendsolid s\n";
        let mut session = Session::demo().unwrap();
        let request = serde_json::json!({
            "op": "import_robot", "format": "urdf", "text": urdf,
            "files": { "part.stl": base64::engine::general_purpose::STANDARD.encode(stl) },
        });
        let reply = call(&mut session, &request.to_string());
        assert_eq!(reply["ok"], true, "{reply}");
        let resources = call(&mut session, r#"{"op":"get_resources"}"#);
        let mesh = &resources["data"][0];
        assert_eq!(mesh["indices"].as_array().unwrap().len(), 3);
        assert_eq!(mesh["vertices"].as_array().unwrap().len(), 9);
        // The snapshot refers to it by id.
        let snapshot = call(&mut session, r#"{"op":"snapshot"}"#);
        assert!(snapshot["data"]["drawables"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["shape"]["resourceId"] == mesh["id"]));
        let bad = serde_json::json!({"op": "import_robot", "format": "urdf", "text": urdf, "files": {"part.stl": "%%%"}});
        assert!(call(&mut session, &bad.to_string())["error"]
            .as_str()
            .unwrap()
            .contains("base64"));
    }

    #[test]
    fn moving_links_without_collision_geometry_are_named() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"describe"}"#);
        assert_eq!(
            reply["data"]["linksWithoutCollision"],
            serde_json::json!([])
        );
        // `b` moves and its only mesh is COLLADA, so it ends up a skeleton;
        // `tip` is a fixed frame and has nothing to check.
        let urdf = r#"<robot name="d"><link name="a"><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
            <link name="b"><visual><geometry><mesh filename="arm.dae"/></geometry></visual></link><link name="tip"/>
            <joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 0.1"/><limit lower="-1" upper="1"/></joint>
            <joint name="f" type="fixed"><parent link="b"/><child link="tip"/><origin xyz="0 0 0.2"/></joint></robot>"#;
        let request = serde_json::json!({ "op": "import_robot", "format": "urdf", "text": urdf });
        let reply = call(&mut session, &request.to_string());
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(
            reply["data"]["description"]["linksWithoutCollision"],
            serde_json::json!(["b"])
        );
    }

    #[test]
    fn a_mesh_file_is_added_as_a_part_and_planned() {
        use base64::Engine as _;
        use std::fmt::Write as _;
        // A 30 x 50 x 20 mm box as an ASCII STL, in millimetres.
        let (x, y, z) = (30.0, 50.0, 20.0);
        let corner = |i: usize| {
            [
                if i & 1 == 0 { 0.0 } else { x },
                if i & 2 == 0 { 0.0 } else { y },
                if i & 4 == 0 { 0.0 } else { z },
            ]
        };
        let mut stl = String::from("solid box\n");
        for [a, b, c, d] in [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ] {
            for t in [[a, b, c], [a, c, d]] {
                stl.push_str("facet normal 0 0 0\nouter loop\n");
                for i in t {
                    let p = corner(i);
                    let _ = writeln!(stl, "vertex {} {} {}", p[0], p[1], p[2]);
                }
                stl.push_str("endloop\nendfacet\n");
            }
        }
        stl.push_str("endsolid box\n");
        let mut session = Session::demo().unwrap();
        let request = serde_json::json!({
            "op": "add_mesh_object", "file": "block.stl",
            "data": base64::engine::general_purpose::STANDARD.encode(&stl),
            "position": {"x": 0.42, "y": -0.1, "z": 0.0}, "restOn": 0.42,
        });
        let reply = call(&mut session, &request.to_string());
        assert_eq!(reply["ok"], true, "{reply}");
        assert!(reply["data"]["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("millimetres")));
        let block = session.project().environment.body_by_name("block").unwrap();
        assert!(
            (block.pose.translation.z - 0.42).abs() < 1e-9,
            "{:?}",
            block.pose
        );
        // 30 x 50 x 20 mm of water.
        assert!(
            (block.physical.mass - 0.03).abs() < 1e-9,
            "{}",
            block.physical.mass
        );
        let id = block.id.clone();
        // The same file again is a second part, not a clash.
        let reply = call(&mut session, &request.to_string());
        assert_eq!(reply["ok"], true, "{reply}");
        assert!(session
            .project()
            .environment
            .body_by_name("block 2")
            .is_some());
        // Move the cube out of the way and pick the block.
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let other = session
            .project()
            .environment
            .body_by_name("block 2")
            .unwrap()
            .id
            .clone();
        for (body, y) in [(cube, 0.3), (other, -0.3)] {
            let r = call(&mut session, &serde_json::json!({"op": "update_body", "bodyId": body, "position": {"x": 0.45, "y": y, "z": 0.44}}).to_string());
            assert_eq!(r["ok"], true, "{r}");
        }
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{id}"}}"#),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["data"]["candidateGenerator"], "surface_normal");
        assert!(
            !reply["data"]["plans"].as_array().unwrap().is_empty(),
            "{}",
            reply["data"]["rejections"]
        );
        let bad = serde_json::json!({"op": "add_mesh_object", "file": "b.stl", "data": "", "units": "inch", "position": {"x": 0.0, "y": 0.0, "z": 0.0}});
        assert_eq!(call(&mut session, &bad.to_string())["ok"], false);
    }

    #[test]
    fn a_failed_import_leaves_the_session_as_it_was() {
        let mut session = Session::demo().unwrap();
        let before = session.project().clone();
        let reply = call(
            &mut session,
            r#"{"op":"import_robot","format":"urdf","text":"<robot name='x'/>"}"#,
        );
        assert_eq!(reply["ok"], false);
        assert!(
            reply["error"].as_str().unwrap().contains("no links"),
            "{reply}"
        );
        assert_eq!(session.project(), &before);
        let other = call(
            &mut session,
            r#"{"op":"import_robot","format":"sdf","text":"<sdf/>"}"#,
        );
        assert!(other["error"]
            .as_str()
            .unwrap()
            .contains("`urdf` and `simox`"));
    }

    #[test]
    fn a_simox_object_can_be_added_as_a_part() {
        let mut session = Session::demo().unwrap();
        let object = "<ManipulationObject name='Block'><Visualization><Primitives><Box width='40' height='30' depth='50'/></Primitives><UseAsCollisionModel/></Visualization><Physics><Mass value='120' units='g'/></Physics></ManipulationObject>";
        let request = serde_json::json!({"op": "add_object", "text": object, "position": {"x": 0.4, "y": 0.0, "z": 9.0}, "restOn": 0.42});
        let reply = call(&mut session, &request.to_string());
        assert_eq!(reply["ok"], true, "{reply}");
        let block = reply["data"]["description"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == "Block")
            .unwrap()
            .clone();
        assert_eq!(block["graspable"], true);
        assert!((block["mass"].as_f64().unwrap() - 0.12).abs() < 1e-12);
        // Resting on the table top: its 50 mm depth puts its centre 25 mm up.
        assert!(
            (block["position"]["z"].as_f64().unwrap() - 0.445).abs() < 1e-12,
            "{block}"
        );
        let again = call(&mut session, &request.to_string());
        assert_eq!(
            again["ok"], false,
            "a second part with the same name is refused"
        );
    }

    #[test]
    fn position_only_ik_reaches_a_point() {
        let mut session = Session::demo().unwrap();
        let reply = call(
            &mut session,
            r#"{"op":"solve_ik","position":{"x":0.35,"y":0.2,"z":0.7}}"#,
        );
        assert_eq!(reply["kind"], "ik");
        assert_eq!(reply["data"]["converged"], true, "{reply}");
    }

    #[test]
    fn a_demo_with_a_hand_describes_it_and_plans_with_it() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"load_demo","gripper":"hand"}"#);
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["data"]["gripperFingers"], 3);
        assert_eq!(reply["data"]["preshapes"][0], "open");
        assert_eq!(reply["data"]["joints"].as_array().unwrap().len(), 12);
        // Home: the arm as in the jaw demo, the hand in its open preshape,
        // and nothing touching.
        let state = call(&mut session, r#"{"op":"snapshot"}"#);
        assert_eq!(state["ok"], true, "{state}");
        assert_eq!(
            state["data"]["collision"]["contacts"],
            serde_json::json!([]),
            "{}",
            state["data"]["collision"]
        );
        let dowel = reply["data"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == "dowel")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let plan = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_pick","bodyId":"{dowel}","options":{{"surfaceSamples":16,"angularSamples":4,"nrGrasps":1,"preshape":"open"}}}}"#
            ),
        );
        assert_eq!(plan["ok"], true, "{plan}");
        assert_eq!(plan["data"]["candidateGenerator"], "hand_surface_normal");
        let best = &plan["data"]["plans"][0];
        assert_eq!(best["candidate"]["hand"]["preshape"], "open", "{plan}");
        assert!(
            best["candidate"]["hand"]["fingersInContact"]
                .as_u64()
                .unwrap()
                >= 2
        );

        // The jaw demo is still the default.
        let reply = call(&mut session, r#"{"op":"load_demo"}"#);
        assert_eq!(reply["data"]["gripperFingers"], 0);
        assert_eq!(reply["data"]["preshapes"], serde_json::json!([]));
    }

    #[test]
    fn a_cycle_is_planned_drawn_and_exported() {
        let mut session = Session::demo().unwrap();
        let early = call(&mut session, r#"{"op":"export_cycle","format":"urscript"}"#);
        assert_eq!(early["ok"], false, "nothing to export yet");
        let reply = call(&mut session, r#"{"op":"load_demo","obstacles":"wall"}"#);
        assert!(reply["data"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["name"] == "wall"));
        let cube = reply["data"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == "cube")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let cycle = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_cycle","bodyId":"{cube}","target":{{"x":0.33,"y":0.21,"z":0.44}}}}"#
            ),
        );
        assert_eq!(cycle["ok"], true, "{cycle}");
        assert_eq!(cycle["kind"], "cycle");
        let data = &cycle["data"];
        let time = data["report"]["cycle"]["cycleTime"].as_f64().unwrap();
        let program = data["programSeconds"].as_f64().unwrap();
        assert!((time - program).abs() / time < 0.02, "{time} vs {program}");
        let samples = data["samples"]["points"].as_array().unwrap();
        assert!(samples.len() > 100);
        assert_eq!(samples.last().unwrap()["part"], "placed");

        // Drawn with the part where the cycle set it down.
        let place = &data["report"]["cycle"]["placePose"];
        let last = &samples.last().unwrap()["positions"];
        let snapshot = call(
            &mut session,
            &format!(
                r#"{{"op":"snapshot","values":{last},"placed":{{"bodyId":"{cube}","pose":{place}}}}}"#
            ),
        );
        assert_eq!(snapshot["ok"], true, "{snapshot}");
        assert_eq!(
            snapshot["data"]["collision"]["contacts"],
            serde_json::json!([])
        );
        let drawn = snapshot["data"]["drawables"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["ownerId"] == cube.as_str())
            .unwrap()["pose"]["translation"]
            .clone();
        assert!(
            (drawn["y"].as_f64().unwrap() - 0.21).abs() < 1e-9,
            "{drawn}"
        );
        // ... for that snapshot only.
        let after = call(&mut session, r#"{"op":"describe"}"#);
        let cube_now = after["data"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == "cube")
            .unwrap()["position"]["y"]
            .as_f64()
            .unwrap();
        assert!((cube_now + 0.1).abs() < 1e-12);

        for (format, needle) in [
            ("urscript", "movej("),
            ("csv", "time_s,"),
            ("json", "\"jointNames\""),
        ] {
            let program = call(
                &mut session,
                &format!(r#"{{"op":"export_cycle","format":"{format}"}}"#),
            );
            assert_eq!(program["ok"], true, "{program}");
            assert!(
                program["data"]["text"].as_str().unwrap().contains(needle),
                "{format}"
            );
        }
        let bad = call(&mut session, r#"{"op":"export_cycle","format":"krl"}"#);
        assert_eq!(bad["ok"], false);
        // A target inside the wall is refused.
        let blocked = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_cycle","bodyId":"{cube}","target":{{"x":0.45,"y":0.075,"z":0.44}}}}"#
            ),
        );
        assert_eq!(blocked["ok"], false);
        assert!(
            blocked["error"].as_str().unwrap().contains("occupied"),
            "{blocked}"
        );
    }
}
