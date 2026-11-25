//! Grippers with more than two jaws: a centric three-finger gripper and a
//! three-finger hand with jointed fingers.
//!
//! Both are built from ordinary links and end-effector joints, like the
//! parallel-jaw gripper, and described by a [`Gripper`] with actors: each
//! finger is an actor that closes until it touches, starting from a
//! preshape. The proportions are generic, not those of any product.
//!
//! Geometry, in the palm's frame: the fingers stand on the palm face at
//! `z = palm_length`, around the approach axis `+Z`, at 0, 120 and 240
//! degrees from `+X`. The grasp frame sits on the axis, halfway along the
//! fingers.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use super::{drawn, drawn_cylinder, look, stable_id};
use crate::error::{DomainError, DomainResult};
use crate::geometry::{GeometryInstance, GeometryRole, Shape};
use crate::joint::{Joint, JointLimits};
use crate::link::{Inertial, Link, ToolFrame};
use crate::math::{Quat, Transform, Vec3};
use crate::robot::{Actor, ActorJoint, Gripper, JawCoupling, Preshape, RobotSystem};

/// How a three-finger gripper or hand should be built.
#[derive(Debug, Clone)]
pub struct HandSpec {
    /// Display name.
    pub name: String,
    /// Radius of the round palm, metres.
    pub palm_radius: f64,
    /// Distance from the mounting face to the finger roots, metres.
    pub palm_length: f64,
    /// Finger length along the approach axis when straight, metres.
    pub finger_length: f64,
    /// Finger extent across its closing direction, metres.
    pub finger_width: f64,
    /// Finger extent along its closing direction, metres.
    pub finger_thickness: f64,
    /// Greatest distance across the fingers, metres: for the centric gripper
    /// the jaw-face diameter, for the hand the fingertip diameter when open.
    pub max_opening: f64,
    /// Force each finger presses with, newtons.
    pub max_grip_force: f64,
    /// Mass of the whole gripper, kilograms.
    pub mass: f64,
}

impl HandSpec {
    /// A centric three-finger gripper with a 100 mm jaw diameter.
    pub fn three_finger_gripper() -> Self {
        Self {
            name: "three-finger gripper".to_owned(),
            palm_radius: 0.05,
            palm_length: 0.05,
            finger_length: 0.045,
            finger_width: 0.02,
            finger_thickness: 0.012,
            max_opening: 0.1,
            max_grip_force: 60.0,
            mass: 1.0,
        }
    }

    /// A hand of three two-phalanx fingers.
    pub fn three_finger_hand() -> Self {
        Self {
            name: "hand".to_owned(),
            palm_radius: 0.042,
            palm_length: 0.045,
            finger_length: 0.08,
            finger_width: 0.018,
            finger_thickness: 0.016,
            max_opening: 0.11,
            max_grip_force: 25.0,
            mass: 0.9,
        }
    }

    /// Validates the proportions.
    ///
    /// # Errors
    /// [`DomainError::NegativeValue`], [`DomainError::NonFinite`] or
    /// [`DomainError::EmptyName`].
    pub fn validate(&self) -> DomainResult<()> {
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "gripper name",
            });
        }
        for (value, what) in [
            (self.palm_radius, "palmRadius"),
            (self.palm_length, "palmLength"),
            (self.finger_length, "fingerLength"),
            (self.finger_width, "fingerWidth"),
            (self.finger_thickness, "fingerThickness"),
            (self.max_opening, "maxOpening"),
            (self.max_grip_force, "maxGripForce"),
            (self.mass, "mass"),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("hand spec {what}"),
                });
            }
            if value <= 0.0 {
                return Err(DomainError::NegativeValue {
                    context: format!("hand spec {what} must be positive"),
                    value,
                });
            }
        }
        Ok(())
    }
}

/// The link and offset of a tool frame to mount on.
fn mount(robot: &RobotSystem, frame_id: &str, what: &str) -> DomainResult<(String, Transform)> {
    let mount = robot.tool_frame(frame_id).ok_or_else(|| {
        DomainError::dangling(
            "tool frame",
            frame_id.to_owned(),
            format!("{what} on robot '{}'", robot.name),
        )
    })?;
    Ok((mount.link_id.clone(), mount.offset))
}

/// The round palm and its mounting joint.
fn round_palm(
    namespace: &str,
    spec: &HandSpec,
    mount_link_id: &str,
    mount_offset: Transform,
) -> (Link, Joint) {
    let palm_name = format!("{} palm", spec.name);
    let palm_id = stable_id(namespace, "link", &palm_name);
    let ns = format!("{namespace}::palm");
    let (r, l) = (spec.palm_radius, spec.palm_length);
    let palm = Link::new(&palm_id, &palm_name)
        .with_inertial(Inertial {
            mass: spec.mass * 0.55,
            center_of_mass: Vec3::new(0.0, 0.0, l * 0.5),
            inertia_diagonal: Vec3::new(1.0e-3, 1.0e-3, 1.0e-3),
        })
        .with_geometry(
            GeometryInstance::visual(
                stable_id(namespace, "geometry", &palm_name),
                "housing",
                Shape::Cylinder {
                    radius: r,
                    length: l,
                },
            )
            .with_origin(Transform::from_translation(Vec3::new(0.0, 0.0, l * 0.5)))
            .with_role(GeometryRole::Collision),
        )
        .with_geometry(drawn_cylinder(
            &ns,
            "adapter",
            (r * 0.8, l * 0.2),
            Vec3::new(0.0, 0.0, l * 0.1),
            &Vec3::Z,
            look::ALUMINIUM,
        ))
        .with_geometry(drawn_cylinder(
            &ns,
            "body",
            (r * 0.97, l * 0.62),
            Vec3::new(0.0, 0.0, l * 0.5),
            &Vec3::Z,
            look::DARK,
        ))
        .with_geometry(drawn_cylinder(
            &ns,
            "ring",
            (r * 0.99, l * 0.08),
            Vec3::new(0.0, 0.0, l * 0.5),
            &Vec3::Z,
            look::ACCENT,
        ))
        .with_geometry(drawn_cylinder(
            &ns,
            "face",
            (r * 0.95, l * 0.18),
            Vec3::new(0.0, 0.0, l * 0.9),
            &Vec3::Z,
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

/// The angle and outward radial direction of finger `index`.
fn finger_direction(index: usize) -> (f64, Vec3) {
    #[allow(clippy::cast_precision_loss)]
    let angle = index as f64 * TAU / 3.0;
    (angle, Vec3::new(angle.cos(), angle.sin(), 0.0))
}

/// A finger segment: a box running from the link origin along `+Z`, its
/// inner face toward the axis, as collision geometry, drawn as a metal body
/// with a rubber pad on the inner face.
#[allow(clippy::too_many_arguments)]
fn segment(
    namespace: &str,
    name: &str,
    link_id: &str,
    angle: f64,
    (thickness, width, length): (f64, f64, f64),
    radial_center: f64,
    mass: f64,
) -> DomainResult<Link> {
    let turn = Quat::from_axis_angle(&Vec3::Z, angle)?;
    let radial = Vec3::new(angle.cos(), angle.sin(), 0.0);
    let at = |radial_offset: f64| {
        Transform::new(
            radial
                .scaled(radial_offset)
                .add(&Vec3::new(0.0, 0.0, length * 0.5)),
            turn,
        )
    };
    let pad = thickness * 0.25;
    let ns = format!("{namespace}::{name}");
    let inner_face = radial_center - thickness * 0.5;
    Ok(Link::new(link_id, name)
        .with_inertial(Inertial {
            mass,
            center_of_mass: radial
                .scaled(radial_center)
                .add(&Vec3::new(0.0, 0.0, length * 0.5)),
            inertia_diagonal: Vec3::new(1.0e-5, 1.0e-5, 1.0e-5),
        })
        .with_geometry(
            GeometryInstance::visual(
                stable_id(namespace, "geometry", name),
                "finger",
                Shape::Box {
                    size: Vec3::new(thickness, width, length),
                },
            )
            .with_origin(at(radial_center))
            .with_role(GeometryRole::Collision),
        )
        .with_geometry(drawn(
            &ns,
            "grip",
            Shape::Box {
                size: Vec3::new(pad, width, length),
            },
            at(inner_face + pad * 0.5),
            look::RUBBER,
        ))
        .with_geometry(drawn(
            &ns,
            "body",
            Shape::Box {
                size: Vec3::new(thickness - pad, width, length),
            },
            at(inner_face + pad + (thickness - pad) * 0.5),
            look::ALUMINIUM,
        )))
}

/// Fits the gripper to the robot: palm, fingers, grasp frame, exemptions.
fn fit(
    robot: &mut RobotSystem,
    namespace: &str,
    spec: &HandSpec,
    parts: (Link, Joint, Vec<Link>, Vec<Joint>),
    actors: Vec<Actor>,
    preshapes: Vec<Preshape>,
    grasp_depth: f64,
) -> DomainResult<()> {
    let (palm, palm_joint, fingers, finger_joints) = parts;
    let grasp_frame = ToolFrame::new(
        stable_id(namespace, "frame", "grasp"),
        format!("{} grasp", spec.name),
        &palm.id,
        Transform::from_translation(Vec3::new(0.0, 0.0, spec.palm_length + grasp_depth)),
    );
    let gripper = Gripper {
        id: stable_id(namespace, "gripper", &spec.name),
        name: spec.name.clone(),
        palm_link_id: palm.id.clone(),
        jaw_joint_ids: finger_joints.iter().map(|j| j.id.clone()).collect(),
        finger_link_ids: fingers.iter().map(|l| l.id.clone()).collect(),
        grasp_frame_id: grasp_frame.id.clone(),
        coupling: JawCoupling::Independent,
        approach_axis: Vec3::Z,
        // The first finger, at +X, closes toward -X.
        closing_axis: Vec3::new(-1.0, 0.0, 0.0),
        max_opening: spec.max_opening,
        finger_depth: spec.finger_length,
        finger_width: spec.finger_width,
        max_grip_force: spec.max_grip_force,
        actors,
        preshapes,
    };
    robot.links.push(palm);
    robot.links.extend(fingers);
    robot.joints.push(palm_joint);
    robot.joints.extend(finger_joints);
    robot.default_tool_frame_id = Some(grasp_frame.id.clone());
    robot.tool_frames.push(grasp_frame);
    robot.gripper = Some(gripper);
    robot.validate()
}

/// Attaches a centric three-finger gripper to a robot's tool frame.
///
/// Each finger slides radially on its own prismatic joint, whose value is
/// the distance from the axis to the finger's gripping face. Each finger is
/// its own actor: when planning, a finger closes until it touches and the
/// others carry on. A real centric gripper couples its fingers mechanically
/// and centres the part instead; the planner does not move the part, so its
/// grasps are the ones a centred part would get, and it tries grasps centred
/// on the part for that reason.
///
/// Preshapes: `open` (the full diameter) and `half` (half of it).
///
/// # Errors
/// An unknown `mount_frame_id`, plus anything [`HandSpec::validate`] or
/// [`RobotSystem::validate`] rejects.
pub fn attach_three_finger_gripper(
    robot: &mut RobotSystem,
    mount_frame_id: &str,
    spec: &HandSpec,
) -> DomainResult<()> {
    spec.validate()?;
    let (mount_link_id, mount_offset) =
        mount(robot, mount_frame_id, "attach_three_finger_gripper")?;
    let namespace = format!("{}::{}", robot.name, spec.name);
    let (palm, palm_joint) = round_palm(&namespace, spec, &mount_link_id, mount_offset);

    let (t, w, d) = (spec.finger_thickness, spec.finger_width, spec.finger_length);
    let open = spec.max_opening * 0.5;
    // Three faces at 120 degrees meet when each is w / (2 tan 60) from the
    // axis; a little more keeps the closed fingers from interpenetrating.
    let closed = w * 0.3;
    let mut fingers = Vec::new();
    let mut joints = Vec::new();
    let mut actors = Vec::new();
    for index in 0..3 {
        let (angle, radial) = finger_direction(index);
        let name = format!("{} finger {}", spec.name, index + 1);
        let link_id = stable_id(&namespace, "link", &name);
        let finger = segment(
            &namespace,
            &name,
            &link_id,
            angle,
            (t, w, d),
            t * 0.5,
            spec.mass * 0.15,
        )?;
        let joint = Joint::prismatic(
            stable_id(&namespace, "joint", &name),
            format!("{} jaw {}", spec.name, index + 1),
            &palm.id,
            &link_id,
            Transform::from_translation(Vec3::new(0.0, 0.0, spec.palm_length)),
            radial,
            closed,
            open,
        )?
        .as_end_effector()
        .with_dynamics(0.1, 1.0, spec.max_grip_force);
        actors.push(Actor {
            name: format!("finger {}", index + 1),
            joints: vec![ActorJoint {
                joint_id: joint.id.clone(),
                direction: -1.0,
            }],
        });
        fingers.push(finger);
        joints.push(joint);
    }
    let preshape = |name: &str, value: f64| Preshape {
        name: name.to_owned(),
        joint_values: joints
            .iter()
            .map(|j| (j.id.clone(), value))
            .collect::<BTreeMap<_, _>>(),
    };
    let preshapes = vec![
        preshape("open", open),
        preshape("half", (open * 0.5).max(closed)),
    ];
    fit(
        robot,
        &namespace,
        spec,
        (palm, palm_joint, fingers, joints),
        actors,
        preshapes,
        d * 0.5,
    )
}

/// Attaches a hand of three jointed fingers to a robot's tool frame.
///
/// Each finger has a proximal and a distal phalanx on revolute joints that
/// flex toward the axis; both joints belong to the finger's actor and close
/// at the same rate. A phalanx that touches stops the joints that move it,
/// and the joints beyond it keep closing, so a finger whose base is blocked
/// curls its tip around the part.
///
/// Preshapes: `open` (fingers splayed outward) and `straight`.
///
/// # Errors
/// An unknown `mount_frame_id`, plus anything [`HandSpec::validate`] or
/// [`RobotSystem::validate`] rejects.
pub fn attach_three_finger_hand(
    robot: &mut RobotSystem,
    mount_frame_id: &str,
    spec: &HandSpec,
) -> DomainResult<()> {
    spec.validate()?;
    let (mount_link_id, mount_offset) = mount(robot, mount_frame_id, "attach_three_finger_hand")?;
    let namespace = format!("{}::{}", robot.name, spec.name);
    let (palm, palm_joint) = round_palm(&namespace, spec, &mount_link_id, mount_offset);

    let (t, w) = (spec.finger_thickness, spec.finger_width);
    let proximal_length = spec.finger_length * 5.0 / 9.0;
    let distal_length = spec.finger_length - proximal_length;
    let root = spec.palm_radius * 0.7;
    // The splay that puts the fingertips at the declared opening, kept
    // within the joint's outward range.
    let splay = ((spec.max_opening * 0.5 - root) / spec.finger_length)
        .clamp(0.0, 0.9)
        .asin()
        .min(0.5);
    let mut fingers = Vec::new();
    let mut joints = Vec::new();
    let mut actors = Vec::new();
    let mut open = BTreeMap::new();
    let mut straight = BTreeMap::new();
    for index in 0..3 {
        let (angle, radial) = finger_direction(index);
        // Positive rotation about this axis turns +Z toward the palm axis.
        let flex = Vec3::new(angle.sin(), -angle.cos(), 0.0);
        let mut actor_joints = Vec::new();
        let mut parent = palm.id.clone();
        for (segment_name, length, origin, limits, at_open) in [
            (
                "proximal",
                proximal_length,
                radial
                    .scaled(root)
                    .add(&Vec3::new(0.0, 0.0, spec.palm_length)),
                (-splay.max(0.05), 1.5),
                -splay,
            ),
            (
                "distal",
                distal_length,
                Vec3::new(0.0, 0.0, proximal_length),
                (0.0, 1.5),
                0.0,
            ),
        ] {
            let name = format!("{} finger {} {segment_name}", spec.name, index + 1);
            let link_id = stable_id(&namespace, "link", &name);
            let link = segment(
                &namespace,
                &name,
                &link_id,
                angle,
                (t, w, length),
                0.0,
                spec.mass * 0.075,
            )?;
            let mut joint = Joint::revolute(
                stable_id(&namespace, "joint", &name),
                format!("{} finger {} {segment_name} joint", spec.name, index + 1),
                &parent,
                &link_id,
                Transform::from_translation(origin),
                flex,
                1.0,
            )?
            .as_end_effector()
            .with_dynamics(1.5, 6.0, spec.max_grip_force * length);
            joint.limits = JointLimits::new(limits.0, limits.1, &joint.name)?;
            open.insert(joint.id.clone(), at_open);
            straight.insert(joint.id.clone(), 0.0);
            actor_joints.push(ActorJoint {
                joint_id: joint.id.clone(),
                direction: 1.0,
            });
            parent.clone_from(&link_id);
            fingers.push(link);
            joints.push(joint);
        }
        actors.push(Actor {
            name: format!("finger {}", index + 1),
            joints: actor_joints,
        });
    }
    let preshapes = vec![
        Preshape {
            name: "open".to_owned(),
            joint_values: open,
        },
        Preshape {
            name: "straight".to_owned(),
            joint_values: straight,
        },
    ];
    fit(
        robot,
        &namespace,
        spec,
        (palm, palm_joint, fingers, joints),
        actors,
        preshapes,
        spec.finger_length * 0.5,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::six_axis;

    fn flange(robot: &RobotSystem) -> String {
        robot.tool_frames[0].id.clone()
    }

    #[test]
    fn the_three_finger_gripper_has_three_radial_actors() {
        let mut robot = six_axis().unwrap();
        let frame = flange(&robot);
        attach_three_finger_gripper(&mut robot, &frame, &HandSpec::three_finger_gripper()).unwrap();
        let gripper = robot.gripper.as_ref().unwrap();
        assert!(gripper.is_hand());
        assert_eq!(gripper.actors.len(), 3);
        assert_eq!(gripper.finger_link_ids.len(), 3);
        assert_eq!(robot.axis_count(), 6, "fingers are not manipulator axes");
        assert_eq!(robot.total_dof(), 9);
        let open = gripper.preshape(None).unwrap();
        assert_eq!(open.name, "open");
        for joint_id in &gripper.jaw_joint_ids {
            assert_eq!(gripper.closing_direction(joint_id), Some(-1.0));
            assert!((open.joint_values[joint_id] - 0.05).abs() < 1e-12);
        }
    }

    #[test]
    fn the_hand_has_two_phalanges_per_finger() {
        let mut robot = six_axis().unwrap();
        let frame = flange(&robot);
        attach_three_finger_hand(&mut robot, &frame, &HandSpec::three_finger_hand()).unwrap();
        let gripper = robot.gripper.as_ref().unwrap();
        assert_eq!(gripper.actors.len(), 3);
        assert!(gripper.actors.iter().all(|a| a.joints.len() == 2));
        assert_eq!(gripper.finger_link_ids.len(), 6);
        assert_eq!(robot.total_dof(), 12);
        assert_eq!(gripper.preshapes.len(), 2);
        // The open preshape splays the fingers outward, within the limits.
        let open = gripper.preshape(Some("open")).unwrap();
        assert!(open.joint_values.values().any(|v| *v < 0.0));
    }

    #[test]
    fn a_hand_whose_preshape_leaves_the_limits_is_rejected() {
        let mut robot = six_axis().unwrap();
        let frame = flange(&robot);
        attach_three_finger_hand(&mut robot, &frame, &HandSpec::three_finger_hand()).unwrap();
        let mut broken = robot.clone();
        let gripper = broken.gripper.as_mut().unwrap();
        let joint = gripper.jaw_joint_ids[0].clone();
        gripper.preshapes[0].joint_values.insert(joint, 9.0);
        assert!(broken.validate().is_err());

        let mut broken = robot.clone();
        broken.gripper.as_mut().unwrap().actors[1].joints[0].direction = 0.0;
        assert!(broken.validate().is_err());

        let mut broken = robot.clone();
        broken.gripper.as_mut().unwrap().jaw_joint_ids.pop();
        assert!(broken.validate().is_err());

        let mut broken = robot;
        let gripper = broken.gripper.as_mut().unwrap();
        let shared = gripper.actors[0].joints[0].clone();
        gripper.actors[1].joints.push(shared);
        assert!(broken.validate().is_err());
    }

    #[test]
    fn a_saved_parallel_gripper_reads_without_actors() {
        let robot = crate::templates::six_axis_with_gripper().unwrap();
        let text = serde_json::to_string(&robot).unwrap();
        assert!(!text.contains("actors"), "empty lists are not written");
        let back: RobotSystem = serde_json::from_str(&text).unwrap();
        assert!(!back.gripper.unwrap().is_hand());
    }
}
