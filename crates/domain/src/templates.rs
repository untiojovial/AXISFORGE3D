//! Ready-made robots, so a new project starts from something that moves.
//!
//! Every identifier these builders mint is derived from the robot's name and
//! the part's name, never from a counter or a random source. Two runs of the
//! same template therefore produce byte-identical projects, which is what lets
//! continuous integration regenerate the shipped examples and assert that
//! nothing changed.

use std::f64::consts::{FRAC_PI_2, PI};

use serde::{Deserialize, Serialize};

use crate::environment::{Environment, GROUND_OWNER_ID};
use crate::error::{DomainError, DomainResult};
use crate::geometry::{GeometryInstance, GeometryRole, Shape};
use crate::joint::{Joint, JointLimits};
use crate::link::{Inertial, Link, ToolFrame};
use crate::math::{Quat, Transform, Vec3};
use crate::project::Project;
use crate::robot::{Gripper, JawCoupling, RobotSystem};

mod hands;
pub use hands::{attach_three_finger_gripper, attach_three_finger_hand, HandSpec};

/// Mints a deterministic, UUID-shaped identifier from a namespace and a name.
///
/// This is a content hash, not a random identifier: the same inputs always give
/// the same output, on every machine and in every process. It is not a UUID in
/// the RFC sense and makes no uniqueness guarantee beyond what the hash gives;
/// what it guarantees is reproducibility, which is what a diffable project file
/// needs. Callers must keep `(namespace, kind, name)` distinct themselves —
/// [`RobotSystem::validate`] catches the collision if they do not.
pub fn stable_id(namespace: &str, kind: &str, name: &str) -> String {
    // FNV-1a from two different offset bases gives 128 bits without pulling in
    // a hashing dependency, and its arithmetic is identical on every target.
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let digest = |mut hash: u64| -> u64 {
        for part in [namespace, kind, name] {
            for byte in part.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(PRIME);
            }
            // A separator keeps ("ab", "c") distinct from ("a", "bc").
            hash ^= 0x1f;
            hash = hash.wrapping_mul(PRIME);
        }
        hash
    };
    let high = digest(0xcbf2_9ce4_8422_2325);
    let low = digest(0x9e37_79b9_7f4a_7c15);
    let bytes: [u8; 16] = {
        let mut out = [0_u8; 16];
        out[..8].copy_from_slice(&high.to_be_bytes());
        out[8..].copy_from_slice(&low.to_be_bytes());
        out
    };
    let mut hex = String::with_capacity(32);
    for byte in bytes {
        use std::fmt::Write as _;
        // Writing into a String cannot fail, so there is nothing to handle.
        let _ = write!(hex, "{byte:02x}");
    }
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Colours and finishes shared by the templates, so every robot they build
/// reads as one product.
mod look {
    /// Painted covers.
    pub const COVER: (&str, f64, f64) = ("#e7ebee", 0.15, 0.42);
    /// Dark structural parts and seams.
    pub const DARK: (&str, f64, f64) = ("#2a3138", 0.45, 0.5);
    /// Accent rings and caps.
    pub const ACCENT: (&str, f64, f64) = ("#0f8a94", 0.35, 0.35);
    /// Bare machined aluminium.
    pub const ALUMINIUM: (&str, f64, f64) = ("#c4ccd4", 0.85, 0.32);
    /// Rubber gripping pads.
    pub const RUBBER: (&str, f64, f64) = ("#1c2024", 0.0, 0.92);
}

/// A drawn-only shape with one of the shared finishes.
///
/// Collision geometry is declared separately and kept simple; what is drawn
/// stays inside it, so two things that look as if they touch are never
/// reported clear.
fn drawn(
    namespace: &str,
    name: &str,
    shape: Shape,
    origin: Transform,
    look: (&str, f64, f64),
) -> GeometryInstance {
    GeometryInstance::visual(stable_id(namespace, "visual", name), name, shape)
        .with_origin(origin)
        .with_color(look.0)
        .with_finish(look.1, look.2)
}

/// A cylinder along an arbitrary axis, centred at `center`.
fn drawn_cylinder(
    namespace: &str,
    name: &str,
    (radius, length): (f64, f64),
    center: Vec3,
    axis: &Vec3,
    look: (&str, f64, f64),
) -> GeometryInstance {
    // A cylinder's own axis is +Z; turn that onto the wanted axis. Parallel
    // axes need no turning, and a failed arc (a degenerate axis) leaves it be.
    let rotation = Quat::from_rotation_arc(&Vec3::Z, axis).unwrap_or(Quat::IDENTITY);
    drawn(
        namespace,
        name,
        Shape::Cylinder { radius, length },
        Transform::new(center, rotation),
        look,
    )
}

/// One axis of a serial chain, as the caller describes it.
#[derive(Debug, Clone)]
pub struct AxisSpec {
    /// Display name, unique within the robot.
    pub name: String,
    /// Motion axis in the joint frame.
    pub axis: Vec3,
    /// Translation from the previous axis's frame to this one, metres.
    pub offset: Vec3,
    /// Rotation applied before the offset, orienting the axis.
    pub rotation: Quat,
    /// Joint range, radians.
    pub limits: JointLimits,
    /// Length of the link driven by this axis, metres. Drives its geometry.
    pub link_length: f64,
    /// Radius of the link's capsule, metres.
    pub link_radius: f64,
    /// Mass of the link driven by this axis, kilograms.
    pub mass: f64,
}

impl AxisSpec {
    /// A revolute axis about Z with symmetric limits and default proportions.
    ///
    /// # Errors
    /// [`DomainError::NegativeValue`] when `half_range` is negative.
    pub fn revolute_z(name: &str, offset: Vec3, half_range: f64) -> DomainResult<Self> {
        Ok(Self {
            name: name.to_owned(),
            axis: Vec3::Z,
            offset,
            rotation: Quat::IDENTITY,
            limits: JointLimits::symmetric(half_range, name)?,
            link_length: 0.25,
            link_radius: 0.045,
            mass: 3.0,
        })
    }

    /// Sets the axis direction, builder style.
    pub fn with_axis(mut self, axis: Vec3) -> Self {
        self.axis = axis;
        self
    }

    /// Sets the rotation applied before the offset, builder style.
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Sets the capsule proportions of the driven link, builder style.
    pub fn with_link(mut self, length: f64, radius: f64) -> Self {
        self.link_length = length;
        self.link_radius = radius;
        self
    }

    /// Sets the limits, builder style.
    pub fn with_limits(mut self, limits: JointLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the mass, builder style.
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }
}

/// Builds a serial manipulator from a list of axes.
///
/// The chain is `base -> link 1 -> ... -> link n`, with a tool frame named
/// `flange` at the far end of the last link.
///
/// # Errors
/// [`DomainError::UnsupportedAxisCount`] outside 1..=6,
/// [`DomainError::DuplicateName`] for repeated axis names, or any failure from
/// [`RobotSystem::validate`].
///
/// # Panics
/// Never: the only `expect` is guarded by the axis-count check above.
pub fn build_serial_chain(robot_name: &str, axes: &[AxisSpec]) -> DomainResult<RobotSystem> {
    if !(crate::robot::MIN_AXES..=crate::robot::MAX_AXES).contains(&axes.len()) {
        return Err(DomainError::UnsupportedAxisCount {
            count: axes.len(),
            min: crate::robot::MIN_AXES,
            max: crate::robot::MAX_AXES,
        });
    }
    // Identifiers are derived from names, so a repeated name would surface as a
    // duplicate *id* and point the user at a field they never typed. Check the
    // names first so the error names what they can actually fix.
    for (index, spec) in axes.iter().enumerate() {
        if spec.name.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "axis name" });
        }
        if axes[..index].iter().any(|other| other.name == spec.name) {
            return Err(DomainError::DuplicateName {
                kind: "axis",
                name: spec.name.clone(),
            });
        }
    }

    let mut links = vec![build_base(robot_name)];
    let mut joints = Vec::with_capacity(axes.len());
    let mut parent_link_id = links[0].id.clone();

    for spec in axes {
        let link = build_axis_link(robot_name, spec);
        let link_id = link.id.clone();

        let origin = Transform::new(spec.offset, spec.rotation);
        let mut joint = Joint::revolute(
            stable_id(robot_name, "joint", &spec.name),
            &spec.name,
            &parent_link_id,
            &link_id,
            origin,
            spec.axis,
            PI,
        )?;
        joint.limits = spec.limits;
        joint = joint.with_dynamics(PI, 2.0 * PI, 120.0);

        parent_link_id = link_id;
        links.push(link);
        joints.push(joint);
    }

    let last_axis = axes.last().expect("axis count checked above");
    let flange = ToolFrame::new(
        stable_id(robot_name, "frame", "flange"),
        "flange",
        &parent_link_id,
        Transform::from_translation(Vec3::new(0.0, 0.0, last_axis.link_length)),
    );

    let robot = RobotSystem {
        id: stable_id(robot_name, "robot", robot_name),
        name: robot_name.to_owned(),
        root_link_id: links[0].id.clone(),
        links,
        joints,
        default_tool_frame_id: Some(flange.id.clone()),
        tool_frames: vec![flange],
        gripper: None,
        base_pose: Transform::IDENTITY,
    };
    robot.validate()?;
    Ok(robot)
}

/// Builds the pedestal the first axis stands on.
fn build_base(robot_name: &str) -> Link {
    let base_radius = 0.09;
    let base_height = 0.12;
    let base_ns = format!("{robot_name}::base");
    Link::new(stable_id(robot_name, "link", "base"), "base")
        .with_inertial(Inertial::point_mass(8.0))
        .with_geometry(
            GeometryInstance::visual(
                stable_id(robot_name, "geometry", "base"),
                "pedestal",
                Shape::Cylinder {
                    radius: base_radius,
                    length: base_height,
                },
            )
            .with_origin(Transform::from_translation(Vec3::new(
                0.0,
                0.0,
                base_height * 0.5,
            )))
            .with_role(GeometryRole::Collision),
        )
        .with_geometry(drawn_cylinder(
            &base_ns,
            "foot",
            (base_radius, 0.018),
            Vec3::new(0.0, 0.0, 0.009),
            &Vec3::Z,
            look::DARK,
        ))
        .with_geometry(drawn_cylinder(
            &base_ns,
            "column",
            (base_radius * 0.8, base_height - 0.03),
            Vec3::new(0.0, 0.0, 0.018 + (base_height - 0.03) * 0.5),
            &Vec3::Z,
            look::COVER,
        ))
        .with_geometry(drawn_cylinder(
            &base_ns,
            "ring",
            (base_radius * 0.86, 0.012),
            Vec3::new(0.0, 0.0, base_height - 0.006),
            &Vec3::Z,
            look::ACCENT,
        ))
}

/// Builds the link a serial-chain axis drives.
///
/// The collision capsule runs from the joint origin along `+Z`, so the next
/// axis's offset and the geometry agree by construction. What is drawn — a
/// joint housing across the axis, a tube, and trim — fits inside that capsule.
fn build_axis_link(robot_name: &str, spec: &AxisSpec) -> Link {
    let link_name = format!("{} link", spec.name);
    let ns = format!("{robot_name}::{link_name}");
    let (r, length) = (spec.link_radius, spec.link_length);
    let axis = spec.axis.normalized("axis").unwrap_or(Vec3::Z);
    let across = axis.dot(&Vec3::Z).abs() < 0.5;
    let mut link = Link::new(stable_id(robot_name, "link", &link_name), &link_name)
        .with_inertial(Inertial {
            mass: spec.mass,
            center_of_mass: Vec3::new(0.0, 0.0, length * 0.5),
            // A thin rod about its end, and a disc about its own axis: enough
            // for a plausible payload estimate, and honestly approximate.
            inertia_diagonal: Vec3::new(
                spec.mass * length * length / 12.0,
                spec.mass * length * length / 12.0,
                spec.mass * r * r * 0.5,
            ),
        })
        .with_geometry(
            GeometryInstance::visual(
                stable_id(robot_name, "geometry", &link_name),
                "shell",
                Shape::Capsule { radius: r, length },
            )
            .with_origin(Transform::from_translation(Vec3::new(
                0.0,
                0.0,
                length * 0.5,
            )))
            .with_role(GeometryRole::Collision),
        )
        .with_geometry(drawn_cylinder(
            &ns,
            "tube",
            (r * 0.64, length),
            Vec3::new(0.0, 0.0, length * 0.5),
            &Vec3::Z,
            look::COVER,
        ))
        .with_geometry(drawn_cylinder(
            &ns,
            "collar",
            (r * 0.74, r * 0.5),
            Vec3::new(0.0, 0.0, length - r * 0.25),
            &Vec3::Z,
            look::DARK,
        ));
    if across {
        // A pitch joint: a motor housing lying across the arm, capped at both ends.
        link = link.with_geometry(drawn_cylinder(
            &ns,
            "housing",
            (r * 0.8, r * 1.1),
            Vec3::ZERO,
            &axis,
            look::COVER,
        ));
        for (name, side) in [("cap a", 1.0), ("cap b", -1.0)] {
            link = link.with_geometry(drawn_cylinder(
                &ns,
                name,
                (r * 0.66, r * 0.1),
                axis.scaled(side * r * 0.6),
                &axis,
                look::ACCENT,
            ));
        }
    } else {
        // A roll joint: a seam where the two halves turn against each other.
        link = link
            .with_geometry(drawn_cylinder(
                &ns,
                "housing",
                (r * 0.82, r * 0.9),
                Vec3::new(0.0, 0.0, r * 0.45),
                &Vec3::Z,
                look::COVER,
            ))
            .with_geometry(drawn_cylinder(
                &ns,
                "seam",
                (r * 0.84, r * 0.12),
                Vec3::new(0.0, 0.0, r * 0.06),
                &Vec3::Z,
                look::ACCENT,
            ));
    }
    link
}

/// How a parallel-jaw gripper should be built.
#[derive(Debug, Clone)]
pub struct GripperSpec {
    /// Display name of the gripper.
    pub name: String,
    /// Greatest jaw-to-jaw opening, metres.
    pub max_opening: f64,
    /// How far the gripping faces extend along the approach axis, metres.
    pub finger_depth: f64,
    /// Finger extent along the closing axis, metres.
    pub finger_thickness: f64,
    /// Finger extent perpendicular to both approach and closing, metres.
    pub finger_width: f64,
    /// Distance from the mounting face to the finger roots, metres.
    pub palm_length: f64,
    /// Greatest clamping force, newtons.
    pub max_grip_force: f64,
    /// Mass of the whole gripper, kilograms.
    pub mass: f64,
}

impl Default for GripperSpec {
    fn default() -> Self {
        // Proportions of a small two-finger electric gripper: an 80 mm stroke
        // on 45 mm fingers, which covers the parts the templates ship.
        Self {
            name: "gripper".to_owned(),
            max_opening: 0.08,
            finger_depth: 0.045,
            finger_thickness: 0.012,
            finger_width: 0.025,
            palm_length: 0.05,
            max_grip_force: 90.0,
            mass: 0.85,
        }
    }
}

impl GripperSpec {
    /// Validates the proportions.
    ///
    /// # Errors
    /// [`DomainError::NegativeValue`] or [`DomainError::NonFinite`].
    pub fn validate(&self) -> DomainResult<()> {
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "gripper name",
            });
        }
        for (value, what) in [
            (self.max_opening, "maxOpening"),
            (self.finger_depth, "fingerDepth"),
            (self.finger_thickness, "fingerThickness"),
            (self.finger_width, "fingerWidth"),
            (self.palm_length, "palmLength"),
            (self.max_grip_force, "maxGripForce"),
            (self.mass, "mass"),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("gripper spec {what}"),
                });
            }
            if value <= 0.0 {
                return Err(DomainError::NegativeValue {
                    context: format!("gripper spec {what} must be positive"),
                    value,
                });
            }
        }
        Ok(())
    }
}

/// Attaches a parallel-jaw gripper to a robot's tool frame.
///
/// The gripper is built from ordinary links and prismatic joints, so forward
/// kinematics, rendering and collision checking need no special case for it.
/// The jaws are marked [`crate::joint::JointRole::EndEffector`], so the robot's
/// axis count is unchanged: a six-axis arm with a gripper is still six axes.
///
/// Geometry, in the palm's frame: the fingers extend along `+Z` (the approach
/// direction) and slide along `±Y` (the closing direction). A jaw value is the
/// half-opening, so both jaws at zero means closed, and the grasp frame sits on
/// the closing axis midway along the finger faces.
///
/// `mount_frame_id` must name a tool frame on the robot; the palm is fixed to
/// that frame's link at that frame's offset. The gripper's grasp frame becomes
/// the robot's default tool frame, because that is the pose a user commanding a
/// pick actually means.
///
/// # Errors
/// [`DomainError::DanglingReference`] when `mount_frame_id` is unknown, plus
/// anything [`GripperSpec::validate`] or [`RobotSystem::validate`] rejects.
pub fn attach_parallel_gripper(
    robot: &mut RobotSystem,
    mount_frame_id: &str,
    spec: &GripperSpec,
) -> DomainResult<()> {
    spec.validate()?;
    let mount = robot.tool_frame(mount_frame_id).ok_or_else(|| {
        DomainError::dangling(
            "tool frame",
            mount_frame_id.to_owned(),
            format!("attach_parallel_gripper on robot '{}'", robot.name),
        )
    })?;
    let mount_link_id = mount.link_id.clone();
    let mount_offset = mount.offset;

    let namespace = format!("{}::{}", robot.name, spec.name);

    let (palm, palm_joint) = build_palm(&namespace, spec, &mount_link_id, mount_offset);
    let palm_id = palm.id.clone();

    let mut finger_links = Vec::with_capacity(2);
    let mut jaw_joints = Vec::with_capacity(2);
    // Jaw A slides toward +Y, jaw B toward -Y. Both open as their value grows,
    // so a single commanded half-opening drives the pair symmetrically.
    for (label, direction) in [("left", 1.0), ("right", -1.0)] {
        let (finger, jaw) = build_jaw(&namespace, spec, &palm_id, label, direction)?;
        finger_links.push(finger.id.clone());
        jaw_joints.push(jaw);
        robot.links.push(finger);
    }

    let grasp_frame = ToolFrame::new(
        stable_id(&namespace, "frame", "grasp"),
        format!("{} grasp", spec.name),
        &palm_id,
        Transform::from_translation(Vec3::new(
            0.0,
            0.0,
            spec.palm_length + spec.finger_depth * 0.5,
        )),
    );

    let gripper = Gripper {
        id: stable_id(&namespace, "gripper", &spec.name),
        name: spec.name.clone(),
        palm_link_id: palm_id,
        jaw_joint_ids: jaw_joints.iter().map(|j| j.id.clone()).collect(),
        finger_link_ids: finger_links,
        grasp_frame_id: grasp_frame.id.clone(),
        coupling: JawCoupling::Symmetric,
        approach_axis: Vec3::Z,
        closing_axis: Vec3::Y,
        max_opening: spec.max_opening,
        finger_depth: spec.finger_depth,
        finger_width: spec.finger_width,
        max_grip_force: spec.max_grip_force,
        actors: Vec::new(),
        preshapes: Vec::new(),
    };

    robot.links.push(palm);
    robot.joints.push(palm_joint);
    robot.joints.extend(jaw_joints);
    robot.default_tool_frame_id = Some(grasp_frame.id.clone());
    robot.tool_frames.push(grasp_frame);
    robot.gripper = Some(gripper);
    robot.validate()?;
    Ok(())
}

/// Builds the gripper's palm link and the fixed joint that mounts it.
fn build_palm(
    namespace: &str,
    spec: &GripperSpec,
    mount_link_id: &str,
    mount_offset: Transform,
) -> (Link, Joint) {
    let palm_name = format!("{} palm", spec.name);
    let palm_id = stable_id(namespace, "link", &palm_name);
    let size = Vec3::new(
        spec.finger_width * 1.6,
        spec.max_opening + 2.0 * spec.finger_thickness,
        spec.palm_length,
    );
    let ns = format!("{namespace}::palm");
    let at = |z: f64| Transform::from_translation(Vec3::new(0.0, 0.0, z));
    let palm = Link::new(&palm_id, &palm_name)
        .with_inertial(Inertial {
            mass: spec.mass * 0.6,
            center_of_mass: Vec3::new(0.0, 0.0, spec.palm_length * 0.5),
            inertia_diagonal: Vec3::new(1.0e-3, 1.0e-3, 1.0e-3),
        })
        .with_geometry(
            GeometryInstance::visual(
                stable_id(namespace, "geometry", &palm_name),
                "housing",
                Shape::Box { size },
            )
            .with_origin(at(spec.palm_length * 0.5))
            .with_role(GeometryRole::Collision),
        )
        // A round adapter plate, a dark body, and the rail the jaws ride on.
        .with_geometry(drawn_cylinder(
            &ns,
            "adapter",
            (size.x.min(size.y) * 0.5, spec.palm_length * 0.2),
            Vec3::new(0.0, 0.0, spec.palm_length * 0.1),
            &Vec3::Z,
            look::ALUMINIUM,
        ))
        .with_geometry(drawn(
            &ns,
            "body",
            Shape::Box {
                size: Vec3::new(size.x * 0.92, size.y * 0.8, spec.palm_length * 0.58),
            },
            at(spec.palm_length * 0.49),
            look::DARK,
        ))
        .with_geometry(drawn(
            &ns,
            "stripe",
            Shape::Box {
                size: Vec3::new(size.x * 0.94, size.y * 0.3, spec.palm_length * 0.08),
            },
            at(spec.palm_length * 0.49),
            look::ACCENT,
        ))
        .with_geometry(drawn(
            &ns,
            "rail",
            Shape::Box {
                size: Vec3::new(size.x * 0.7, size.y, spec.palm_length * 0.2),
            },
            at(spec.palm_length * 0.9),
            look::ALUMINIUM,
        ));
    let palm_joint = Joint::fixed(
        stable_id(namespace, "joint", &palm_name),
        format!("{} mount", spec.name),
        mount_link_id,
        &palm_id,
        mount_offset,
    );
    (palm, palm_joint)
}

/// Builds one finger link and the prismatic jaw joint that drives it.
///
/// `direction` is `+1.0` or `-1.0`: the side of the closing axis this jaw sits
/// on. The joint value is the half-opening, so zero means closed, and the
/// finger geometry is offset by half its thickness to put the gripping *face*
/// — not the finger's centre — on the closing axis when closed.
fn build_jaw(
    namespace: &str,
    spec: &GripperSpec,
    palm_id: &str,
    label: &str,
    direction: f64,
) -> DomainResult<(Link, Joint)> {
    let finger_name = format!("{} {label} finger", spec.name);
    let finger_id = stable_id(namespace, "link", &finger_name);
    let ns = format!("{namespace}::{label} finger");
    let (w, t, d) = (spec.finger_width, spec.finger_thickness, spec.finger_depth);
    // The pad is the inner quarter of the finger; the rest is the metal behind it.
    let pad = t * 0.25;
    let slab = |thickness: f64, from_face: f64| {
        (
            Shape::Box {
                size: Vec3::new(w, thickness, d),
            },
            Transform::from_translation(Vec3::new(
                0.0,
                direction * (from_face + thickness * 0.5),
                d * 0.5,
            )),
        )
    };
    let (pad_shape, pad_at) = slab(pad, 0.0);
    let (body_shape, body_at) = slab(t - pad, pad);
    let finger = Link::new(&finger_id, &finger_name)
        .with_inertial(Inertial {
            mass: spec.mass * 0.2,
            center_of_mass: Vec3::new(0.0, direction * t * 0.5, d * 0.5),
            inertia_diagonal: Vec3::new(1.0e-5, 1.0e-5, 1.0e-5),
        })
        .with_geometry(
            GeometryInstance::visual(
                stable_id(namespace, "geometry", &finger_name),
                "pad",
                Shape::Box {
                    size: Vec3::new(w, t, d),
                },
            )
            .with_origin(Transform::from_translation(Vec3::new(
                0.0,
                direction * t * 0.5,
                d * 0.5,
            )))
            .with_role(GeometryRole::Collision),
        )
        .with_geometry(drawn(&ns, "grip", pad_shape, pad_at, look::RUBBER))
        .with_geometry(drawn(&ns, "body", body_shape, body_at, look::ALUMINIUM));
    let jaw = Joint::prismatic(
        stable_id(namespace, "joint", &finger_name),
        format!("{} {label} jaw", spec.name),
        palm_id,
        &finger_id,
        Transform::from_translation(Vec3::new(0.0, 0.0, spec.palm_length)),
        Vec3::new(0.0, direction, 0.0),
        0.0,
        spec.max_opening * 0.5,
    )?
    .as_end_effector()
    .with_dynamics(0.15, 1.5, spec.max_grip_force);
    Ok((finger, jaw))
}

/// Records the collision exemptions a gripper needs.
///
/// Parent and child links are exempt automatically wherever collision is
/// checked, so the palm and its fingers need nothing here. The two fingers are
/// *siblings*, though: they share a parent and so are checked against each
/// other, and at zero opening their faces meet exactly. Without this exemption
/// a closed gripper would report a permanent self-collision.
pub fn register_gripper_exemptions(environment: &mut Environment, robot: &RobotSystem) {
    let Some(gripper) = &robot.gripper else {
        return;
    };
    for (index, first) in gripper.finger_link_ids.iter().enumerate() {
        for second in &gripper.finger_link_ids[index + 1..] {
            environment.ignored_pairs.ignore(
                first,
                second,
                format!("fingers of gripper '{}' meet when closed", gripper.name),
            );
        }
    }
}

/// A single revolute axis on a pedestal: the smallest supported robot.
///
/// # Errors
/// Propagates [`build_serial_chain`].
pub fn single_axis() -> DomainResult<RobotSystem> {
    build_serial_chain(
        "Single axis",
        &[AxisSpec::revolute_z("axis 1", Vec3::new(0.0, 0.0, 0.12), PI)?.with_link(0.3, 0.05)],
    )
}

/// A planar two-link arm, the classic worked example for IK and singularities.
///
/// # Errors
/// Propagates [`build_serial_chain`].
pub fn planar_2r() -> DomainResult<RobotSystem> {
    build_serial_chain(
        "Planar 2R",
        &[
            AxisSpec::revolute_z("axis 1", Vec3::new(0.0, 0.0, 0.12), PI)?
                .with_axis(Vec3::Y)
                .with_link(0.3, 0.04),
            AxisSpec::revolute_z("axis 2", Vec3::new(0.0, 0.0, 0.3), 2.5)?
                .with_axis(Vec3::Y)
                .with_link(0.25, 0.035),
        ],
    )
}

/// A six-axis articulated arm with the usual shoulder-elbow-wrist layout.
///
/// # Errors
/// Propagates [`build_serial_chain`].
pub fn six_axis() -> DomainResult<RobotSystem> {
    build_serial_chain(
        "Six axis",
        &[
            // Waist: rotates the whole arm about the pedestal.
            AxisSpec::revolute_z("axis 1", Vec3::new(0.0, 0.0, 0.12), PI)?.with_link(0.18, 0.07),
            // Shoulder: the first of three parallel pitch axes.
            AxisSpec::revolute_z("axis 2", Vec3::new(0.0, 0.0, 0.18), 2.1)?
                .with_axis(Vec3::Y)
                .with_link(0.34, 0.055)
                .with_mass(4.5),
            // Elbow.
            AxisSpec::revolute_z("axis 3", Vec3::new(0.0, 0.0, 0.34), 2.4)?
                .with_axis(Vec3::Y)
                .with_link(0.28, 0.05)
                .with_mass(3.2),
            // Wrist roll about the forearm.
            AxisSpec::revolute_z("axis 4", Vec3::new(0.0, 0.0, 0.28), PI)?
                .with_link(0.12, 0.04)
                .with_mass(1.4),
            // Wrist pitch.
            AxisSpec::revolute_z("axis 5", Vec3::new(0.0, 0.0, 0.12), 2.2)?
                .with_axis(Vec3::Y)
                .with_link(0.1, 0.035)
                .with_mass(1.0),
            // Tool roll.
            AxisSpec::revolute_z("axis 6", Vec3::new(0.0, 0.0, 0.1), PI)?
                .with_link(0.06, 0.03)
                .with_mass(0.6),
        ],
    )
}

/// The six-axis arm with a parallel-jaw gripper fitted to its flange.
///
/// # Errors
/// Propagates [`six_axis`] and [`attach_parallel_gripper`].
///
/// # Panics
/// Never: [`build_serial_chain`] always adds the flange frame this reads.
pub fn six_axis_with_gripper() -> DomainResult<RobotSystem> {
    six_axis_with(GripperKind::Parallel)
}

/// The grippers the templates can fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GripperKind {
    /// Two parallel jaws driven to a commanded opening.
    #[default]
    Parallel,
    /// Three radial jaws, each closing until it touches.
    ThreeFinger,
    /// Three fingers of two phalanges each, closing until they touch.
    Hand,
}

/// The six-axis arm with the chosen gripper on its flange.
///
/// # Errors
/// Propagates [`six_axis`] and the attaching function.
///
/// # Panics
/// Never: [`build_serial_chain`] always adds the flange frame this reads.
pub fn six_axis_with(kind: GripperKind) -> DomainResult<RobotSystem> {
    let mut robot = six_axis()?;
    let flange_id = robot
        .tool_frames
        .first()
        .expect("build_serial_chain always adds a flange")
        .id
        .clone();
    match kind {
        GripperKind::Parallel => {
            attach_parallel_gripper(&mut robot, &flange_id, &GripperSpec::default())?;
        }
        GripperKind::ThreeFinger => {
            attach_three_finger_gripper(&mut robot, &flange_id, &HandSpec::three_finger_gripper())?;
        }
        GripperKind::Hand => {
            attach_three_finger_hand(&mut robot, &flange_id, &HandSpec::three_finger_hand())?;
        }
    }
    Ok(robot)
}

/// Builds the work table: a top and four legs, all obstacles.
fn build_table(namespace: &str, table_height: f64) -> crate::environment::WorldBody {
    use crate::environment::WorldBody;

    let table_top = 0.03;
    let (table_x, table_y) = (0.5, 0.7);
    let mut table = WorldBody::obstacle(
        stable_id(namespace, "body", "table"),
        "table",
        Transform::from_translation(Vec3::new(0.45, 0.0, table_height - table_top * 0.5)),
        GeometryInstance::visual(
            stable_id(namespace, "geometry", "table"),
            "top",
            Shape::Box {
                size: Vec3::new(table_x, table_y, table_top),
            },
        )
        .with_color("#c9b79c")
        .with_finish(0.0, 0.62)
        .with_role(GeometryRole::Both),
    );
    // Four legs, obstacles like the top: an arm reaching under the table
    // should be told about them.
    let leg = 0.04;
    let leg_height = table_height - table_top;
    for (name, sx, sy) in [
        ("leg a", 1.0, 1.0),
        ("leg b", 1.0, -1.0),
        ("leg c", -1.0, 1.0),
        ("leg d", -1.0, -1.0),
    ] {
        table.geometries.push(
            GeometryInstance::visual(
                stable_id(namespace, "geometry", &format!("table {name}")),
                name,
                Shape::Box {
                    size: Vec3::new(leg, leg, leg_height),
                },
            )
            .with_origin(Transform::from_translation(Vec3::new(
                sx * (table_x * 0.5 - leg * 0.75),
                sy * (table_y * 0.5 - leg * 0.75),
                -(table_top + leg_height) * 0.5,
            )))
            .with_color(look::DARK.0)
            .with_finish(look::DARK.1, look::DARK.2)
            .with_role(GeometryRole::Both),
        );
    }
    table
}

/// A complete pick-and-place cell: the gripper arm, a table and two parts.
///
/// This is the demo project: it is the smallest scene in which every stage of
/// the grasp pipeline has something real to do.
///
/// # Errors
/// Propagates [`six_axis_with_gripper`] and [`Project::validate`].
pub fn pick_and_place_cell() -> DomainResult<Project> {
    pick_and_place_cell_with(GripperKind::Parallel)
}

/// The demo cell with the chosen gripper.
///
/// # Errors
/// Propagates [`six_axis_with`] and [`Project::validate`].
pub fn pick_and_place_cell_with(kind: GripperKind) -> DomainResult<Project> {
    use crate::environment::{PhysicalProperties, WorldBody};

    let robot = six_axis_with(kind)?;
    let namespace = "Pick and place cell";
    let mut project = Project::new(stable_id(namespace, "project", namespace), namespace, robot);
    project.metadata.description.push_str(match kind {
        GripperKind::Parallel => {
            "A six-axis arm with a parallel-jaw gripper, a work table, and two parts to pick."
        }
        GripperKind::ThreeFinger => {
            "A six-axis arm with a centric three-finger gripper, a work table, and two parts \
             to pick."
        }
        GripperKind::Hand => {
            "A six-axis arm with a three-finger hand, a work table, and two parts to pick."
        }
    });

    let table_height = 0.42;
    project
        .environment
        .bodies
        .push(build_table(namespace, table_height));

    // A 40 mm cube: inside the 80 mm stroke with room for approach error.
    let cube_size = 0.04;
    project.environment.bodies.push(WorldBody::manipulable(
        stable_id(namespace, "body", "cube"),
        "cube",
        Transform::from_translation(Vec3::new(0.42, -0.1, table_height + cube_size * 0.5)),
        GeometryInstance::visual(
            stable_id(namespace, "geometry", "cube"),
            "body",
            Shape::Box {
                size: Vec3::new(cube_size, cube_size, cube_size),
            },
        )
        .with_color("#e2852b")
        .with_finish(0.0, 0.45)
        .with_role(GeometryRole::Both),
        PhysicalProperties {
            mass: 0.18,
            friction_coefficient: 0.6,
        },
    ));

    // A 30 mm dowel standing on end: graspable across its diameter only.
    let dowel_radius = 0.015;
    let dowel_length = 0.09;
    project.environment.bodies.push(WorldBody::manipulable(
        stable_id(namespace, "body", "dowel"),
        "dowel",
        Transform::from_translation(Vec3::new(0.46, 0.12, table_height + dowel_length * 0.5)),
        GeometryInstance::visual(
            stable_id(namespace, "geometry", "dowel"),
            "body",
            Shape::Cylinder {
                radius: dowel_radius,
                length: dowel_length,
            },
        )
        .with_color("#3f7fa8")
        .with_finish(0.7, 0.3)
        .with_role(GeometryRole::Both),
        PhysicalProperties {
            mass: 0.09,
            friction_coefficient: 0.45,
        },
    ));

    register_gripper_exemptions(&mut project.environment, &project.robot);
    // The pedestal rests on the floor; that contact is the robot standing up,
    // not a fault, so it is exempt and the exemption says why.
    project.environment.ignored_pairs.ignore(
        &project.robot.root_link_id,
        GROUND_OWNER_ID,
        "the pedestal rests on the floor",
    );
    project.environment.ignored_pairs.ignore(
        project.environment.bodies[0].id.as_str(),
        GROUND_OWNER_ID,
        "the table stands on the floor",
    );

    project.validate()?;
    Ok(project)
}

/// Obstacles that can be added to the demo cell, to give motion planning
/// something to go round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Obstacles {
    /// The table and its parts only.
    #[default]
    None,
    /// A wall across the table between the cube and the dowel's side, just
    /// clear of the arm at its home posture.
    Wall,
    /// Two posts standing on the table on the dowel's side, just clear of the
    /// arm at its home posture.
    Posts,
}

/// The demo cell with the chosen gripper and obstacles.
///
/// # Errors
/// Propagates [`pick_and_place_cell_with`] and [`Project::validate`].
pub fn demo_cell(gripper: GripperKind, obstacles: Obstacles) -> DomainResult<Project> {
    use crate::environment::WorldBody;

    let mut project = pick_and_place_cell_with(gripper)?;
    let namespace = "Pick and place cell";
    let table_top = 0.42;
    let obstacle = |name: &str, shape: Shape, at: Vec3| {
        WorldBody::obstacle(
            stable_id(namespace, "body", name),
            name,
            Transform::from_translation(at),
            GeometryInstance::visual(stable_id(namespace, "geometry", name), name, shape)
                .with_color(look::ACCENT.0)
                .with_finish(look::ACCENT.1, look::ACCENT.2)
                .with_role(GeometryRole::Both),
        )
    };
    match obstacles {
        Obstacles::None => {}
        Obstacles::Wall => {
            let (height, thickness) = (0.15, 0.012);
            project.environment.bodies.push(obstacle(
                "wall",
                Shape::Box {
                    size: Vec3::new(0.4, thickness, height),
                },
                Vec3::new(0.45, 0.075, table_top + height * 0.5),
            ));
        }
        Obstacles::Posts => {
            let (radius, height) = (0.02, 0.3);
            for (name, x, y) in [("post a", 0.33, 0.085), ("post b", 0.56, 0.085)] {
                project.environment.bodies.push(obstacle(
                    name,
                    Shape::Cylinder {
                        radius,
                        length: height,
                    },
                    Vec3::new(x, y, table_top + height * 0.5),
                ));
            }
        }
    }
    project.validate()?;
    Ok(project)
}

/// A wrist-down pose over a point on the table, as a convenience for demos.
///
/// The returned pose has `+Z` pointing straight down, which is the approach a
/// top-down pick uses.
pub fn top_down_pose(position: Vec3) -> DomainResult<Transform> {
    Ok(Transform::new(
        position,
        Quat::from_axis_angle(&Vec3::Y, PI)?.mul(&Quat::from_axis_angle(&Vec3::Z, FRAC_PI_2)?),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::JointKind;

    #[test]
    fn stable_ids_are_reproducible_and_distinct() {
        assert_eq!(
            stable_id("a", "link", "base"),
            stable_id("a", "link", "base")
        );
        assert_ne!(
            stable_id("a", "link", "base"),
            stable_id("b", "link", "base")
        );
        assert_ne!(
            stable_id("a", "link", "base"),
            stable_id("a", "joint", "base")
        );
        assert_ne!(
            stable_id("a", "link", "base"),
            stable_id("a", "link", "tool")
        );
    }

    #[test]
    fn stable_ids_separate_their_parts() {
        // Without a separator these two would hash identically.
        assert_ne!(stable_id("ab", "c", "d"), stable_id("a", "bc", "d"));
    }

    #[test]
    fn stable_ids_are_uuid_shaped() {
        let id = stable_id("Six axis", "link", "base");
        assert_eq!(id.len(), 36, "{id}");
        assert_eq!(id.chars().filter(|c| *c == '-').count(), 4, "{id}");
        assert!(
            id.chars().all(|c| c == '-' || c.is_ascii_hexdigit()),
            "{id}"
        );
    }

    #[test]
    fn every_template_validates() {
        single_axis().expect("single axis");
        planar_2r().expect("planar 2R");
        six_axis().expect("six axis");
        six_axis_with_gripper().expect("six axis with gripper");
        pick_and_place_cell().expect("pick and place cell");
    }

    #[test]
    fn templates_are_byte_identical_across_runs() {
        let first = serde_json::to_string(&pick_and_place_cell().unwrap()).unwrap();
        let second = serde_json::to_string(&pick_and_place_cell().unwrap()).unwrap();
        assert_eq!(
            first, second,
            "a template must not mint non-deterministic ids"
        );
    }

    #[test]
    fn a_duplicate_axis_name_names_the_axis_not_an_id() {
        let axes = [
            AxisSpec::revolute_z("axis 1", Vec3::ZERO, 1.0).unwrap(),
            AxisSpec::revolute_z("axis 1", Vec3::new(0.0, 0.0, 0.2), 1.0).unwrap(),
        ];
        let error = build_serial_chain("dup", &axes).expect_err("duplicate name");
        assert!(
            matches!(error, DomainError::DuplicateName { kind: "axis", .. }),
            "{error}"
        );
    }

    #[test]
    fn seven_axes_are_rejected() {
        let axes: Vec<AxisSpec> = (1..=7)
            .map(|i| {
                AxisSpec::revolute_z(&format!("axis {i}"), Vec3::new(0.0, 0.0, 0.2), 1.0).unwrap()
            })
            .collect();
        assert!(matches!(
            build_serial_chain("too many", &axes),
            Err(DomainError::UnsupportedAxisCount { count: 7, .. })
        ));
    }

    #[test]
    fn a_gripper_does_not_change_the_axis_count() {
        let bare = six_axis().unwrap();
        let fitted = six_axis_with_gripper().unwrap();
        assert_eq!(bare.axis_count(), 6);
        assert_eq!(
            fitted.axis_count(),
            6,
            "gripper jaws must not count as axes"
        );
        assert_eq!(
            fitted.total_dof(),
            8,
            "but they are still degrees of freedom"
        );
    }

    #[test]
    fn the_grasp_frame_becomes_the_default_tool_frame() {
        let robot = six_axis_with_gripper().unwrap();
        let gripper = robot.require_gripper().unwrap();
        assert_eq!(
            robot.default_tool_frame().map(|f| f.id.as_str()),
            Some(gripper.grasp_frame_id.as_str())
        );
    }

    #[test]
    fn jaws_are_prismatic_end_effector_joints() {
        let robot = six_axis_with_gripper().unwrap();
        let gripper = robot.require_gripper().unwrap();
        assert_eq!(gripper.jaw_joint_ids.len(), 2);
        for id in &gripper.jaw_joint_ids {
            let joint = robot.joint(id).expect("jaw exists");
            assert_eq!(joint.kind, JointKind::Prismatic);
            assert!(joint.is_end_effector());
        }
    }

    #[test]
    fn the_declared_opening_matches_the_jaw_stroke() {
        let robot = six_axis_with_gripper().unwrap();
        let gripper = robot.require_gripper().unwrap();
        let jaw = robot.joint(&gripper.jaw_joint_ids[0]).unwrap();
        assert!(
            (gripper.opening_for_jaw_value(jaw.limits.upper) - gripper.max_opening).abs() < 1e-12
        );
        assert!(
            (gripper.jaw_value_for_opening(gripper.max_opening) - jaw.limits.upper).abs() < 1e-12
        );
    }

    #[test]
    fn overstating_the_opening_is_rejected() {
        let mut robot = six_axis_with_gripper().unwrap();
        robot.gripper.as_mut().unwrap().max_opening = 0.5;
        let error = robot.validate().expect_err("overstated stroke");
        assert!(error.to_string().contains("maxOpening"), "{error}");
    }

    #[test]
    fn the_jaws_slide_opposite_each_other_along_the_closing_axis() {
        let robot = six_axis_with_gripper().unwrap();
        let gripper = robot.require_gripper().unwrap();
        let first = robot
            .joint(&gripper.jaw_joint_ids[0])
            .unwrap()
            .axis_in_parent()
            .unwrap();
        let second = robot
            .joint(&gripper.jaw_joint_ids[1])
            .unwrap()
            .axis_in_parent()
            .unwrap();
        assert!(
            (first.dot(&second) + 1.0).abs() < 1e-12,
            "jaws must oppose: {first:?} {second:?}"
        );
        assert!(
            first.cross(&gripper.closing_axis).norm() < 1e-12,
            "jaws slide along the closing axis"
        );
    }

    #[test]
    fn the_demo_cell_has_two_graspable_parts() {
        let project = pick_and_place_cell().unwrap();
        let names: Vec<&str> = project
            .environment
            .graspable_bodies()
            .map(|b| b.name.as_str())
            .collect();
        assert_eq!(names, vec!["cube", "dowel"]);
    }

    #[test]
    fn the_demo_cell_exempts_the_two_fingers_from_each_other() {
        let project = pick_and_place_cell().unwrap();
        let gripper = project.robot.require_gripper().unwrap();
        assert!(project
            .environment
            .ignored_pairs
            .contains(&gripper.finger_link_ids[0], &gripper.finger_link_ids[1]));
    }

    #[test]
    fn a_top_down_pose_points_its_z_at_the_floor() {
        let pose = top_down_pose(Vec3::new(0.4, 0.0, 0.5)).unwrap();
        let approach = pose.apply_direction(&Vec3::Z);
        assert!(
            approach.distance(&Vec3::new(0.0, 0.0, -1.0)) < 1e-12,
            "{approach:?}"
        );
    }
}
