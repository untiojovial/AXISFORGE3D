//! The robot aggregate: links, joints, tool frames and an optional gripper.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};
use crate::joint::{Joint, JointKind, JointRole};
use crate::link::{Link, ToolFrame};
use crate::math::{Transform, Vec3};

/// Smallest supported manipulator axis count.
pub const MIN_AXES: usize = 1;
/// Largest supported manipulator axis count.
pub const MAX_AXES: usize = 6;

/// How a gripper's jaws relate to one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JawCoupling {
    /// Both jaws move by the same amount, so the opening is `2 * value`.
    ///
    /// This is the usual parallel-jaw gripper: one actuator, a symmetric
    /// linkage, and a commanded opening split evenly between the fingers.
    Symmetric,
    /// Each jaw is commanded independently, so the opening is the sum of both.
    Independent,
}

/// Pad width assumed for a gripper saved before the field existed.
fn default_finger_width() -> f64 {
    0.025
}

/// One joint a finger drives, and which way it moves to close.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActorJoint {
    /// The joint, an end-effector joint of the robot.
    pub joint_id: String,
    /// Closing rate relative to the finger.
    ///
    /// `+1` closes by increasing the joint value and `-1` by decreasing it;
    /// other magnitudes couple the joints of one finger at that ratio, as the
    /// `direction` of a node in a Simox end-effector actor does.
    pub direction: f64,
}

/// A finger: joints that close together until the finger touches the part.
///
/// Every link the joints move is part of the finger, and a joint stops as
/// soon as any link it moves touches the part, while the joints beyond it
/// keep closing. A finger whose first phalanx is blocked therefore curls its
/// tip around the part, the way an underactuated finger does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Actor {
    /// Display name, unique within the gripper.
    pub name: String,
    /// The joints it drives.
    pub joints: Vec<ActorJoint>,
}

/// A named posture the hand approaches in, before its fingers close.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Preshape {
    /// Display name, unique within the gripper.
    pub name: String,
    /// Value of each finger joint, by joint identifier, SI units.
    ///
    /// A finger joint left out stays at its open limit: the end of its range
    /// away from the closing direction.
    pub joint_values: BTreeMap<String, f64>,
}

/// A gripper, described in terms of joints that already exist.
///
/// Two kinds are described. A *parallel-jaw* gripper has prismatic jaws
/// driven to a commanded opening, and no actors. A *hand* has actors: fingers
/// of any number, prismatic or jointed, that each close until they touch,
/// starting from a preshape. For a hand, `jaw_joint_ids` lists every finger
/// joint, `finger_link_ids` every link those joints move, and `coupling`,
/// `closing_axis` and the finger dimensions only describe the hand for
/// display and for orienting grasps.
///
/// This type holds no kinematics of its own. The jaws are ordinary prismatic
/// joints in the tree, so forward kinematics, rendering and collision checking
/// treat the fingers like any other moving body — nothing in those pipelines
/// knows what a gripper is. What this adds is the *meaning* a grasp planner
/// needs: which joints are the jaws, where the grasp happens, which way the
/// gripper approaches, and which way it closes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Gripper {
    /// Stable identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Link the jaws are mounted on.
    pub palm_link_id: String,
    /// The prismatic jaw joints, in no particular order.
    pub jaw_joint_ids: Vec<String>,
    /// Links that carry the gripping surfaces, one per jaw.
    pub finger_link_ids: Vec<String>,
    /// Tool frame at the grasp centre, between the fingers when fully closed.
    pub grasp_frame_id: String,
    /// How the jaws are coupled.
    pub coupling: JawCoupling,
    /// Direction, in the grasp frame, the gripper travels to reach an object.
    ///
    /// Approach is the direction of *travel*, so a gripper reaching forward
    /// along its own +Z has approach `+Z`: the pre-grasp pose sits behind the
    /// grasp pose by `-approach * clearance`.
    pub approach_axis: Vec3,
    /// Direction, in the grasp frame, one jaw moves as the gripper closes.
    ///
    /// The opposing jaw moves along `-closing_axis`. A grasp is antipodal when
    /// its two contact normals are parallel to this axis.
    pub closing_axis: Vec3,
    /// Greatest jaw-to-jaw opening, metres.
    pub max_opening: f64,
    /// How far the gripping surface extends along the approach axis, metres.
    ///
    /// An object deeper than this cannot be enclosed, so a grasp whose required
    /// depth exceeds it is rejected rather than reported as reachable.
    pub finger_depth: f64,
    /// Width of each gripping pad, across both the approach and closing axes, metres.
    ///
    /// With the depth this is the area a pad can press on, which is what lets
    /// two fingers resist a part twisting between them.
    #[serde(default = "default_finger_width")]
    pub finger_width: f64,
    /// Greatest clamping force the jaws can apply, newtons.
    ///
    /// For a hand, the force each finger can press with.
    pub max_grip_force: f64,
    /// The fingers of a hand; empty for a parallel-jaw gripper.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actors: Vec<Actor>,
    /// Named postures a hand approaches in; the first is the default.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preshapes: Vec<Preshape>,
}

impl Gripper {
    /// `true` for a hand: fingers that close until they touch.
    pub fn is_hand(&self) -> bool {
        !self.actors.is_empty()
    }

    /// The preshape called `name`, or the default one when `name` is `None`.
    pub fn preshape(&self, name: Option<&str>) -> Option<&Preshape> {
        match name {
            Some(name) => self.preshapes.iter().find(|p| p.name == name),
            None => self.preshapes.first(),
        }
    }

    /// The direction a finger joint closes in, or `None` for a joint no
    /// actor drives.
    pub fn closing_direction(&self, joint_id: &str) -> Option<f64> {
        self.actors
            .iter()
            .flat_map(|a| &a.joints)
            .find(|j| j.joint_id == joint_id)
            .map(|j| j.direction)
    }

    /// The jaw-to-jaw opening for a given per-jaw joint value, metres.
    pub fn opening_for_jaw_value(&self, jaw_value: f64) -> f64 {
        match self.coupling {
            JawCoupling::Symmetric => 2.0 * jaw_value,
            JawCoupling::Independent => jaw_value,
        }
    }

    /// The per-jaw joint value that produces a given jaw-to-jaw opening.
    pub fn jaw_value_for_opening(&self, opening: f64) -> f64 {
        match self.coupling {
            JawCoupling::Symmetric => 0.5 * opening,
            JawCoupling::Independent => opening,
        }
    }

    /// Validates the gripper against the robot that owns it.
    ///
    /// # Errors
    /// [`DomainError::DanglingReference`] for an unknown link, joint or frame;
    /// [`DomainError::NoGripper`] when the referenced joints are not usable as
    /// jaws; or a value error for the axes and dimensions.
    pub fn validate(&self, robot: &RobotSystem) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "gripper id" });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "gripper name",
            });
        }
        self.validate_mounting(robot)?;
        if self.is_hand() {
            self.validate_hand(robot)?;
        } else {
            if !self.preshapes.is_empty() {
                return Err(DomainError::NoGripper {
                    robot: robot.name.clone(),
                    detail: format!(
                        "gripper '{}' has preshapes but no actors; preshapes belong to hands",
                        self.name
                    ),
                });
            }
            self.validate_jaws(robot)?;
        }
        self.validate_axes_and_stroke(robot)
    }

    /// Checks a hand's actors and preshapes.
    fn validate_hand(&self, robot: &RobotSystem) -> DomainResult<()> {
        let fail = |detail: String| DomainError::NoGripper {
            robot: robot.name.clone(),
            detail: format!("gripper '{}': {detail}", self.name),
        };
        let mut names = BTreeSet::new();
        let mut driven = BTreeSet::new();
        for actor in &self.actors {
            if actor.name.trim().is_empty() {
                return Err(DomainError::EmptyName { kind: "actor name" });
            }
            if !names.insert(actor.name.as_str()) {
                return Err(fail(format!("two actors are called '{}'", actor.name)));
            }
            if actor.joints.is_empty() {
                return Err(fail(format!("actor '{}' drives no joints", actor.name)));
            }
            for driven_joint in &actor.joints {
                let joint = robot.joint(&driven_joint.joint_id).ok_or_else(|| {
                    DomainError::dangling(
                        "joint",
                        driven_joint.joint_id.clone(),
                        format!("gripper '{}' actor '{}'", self.name, actor.name),
                    )
                })?;
                if !joint.is_end_effector() {
                    return Err(fail(format!(
                        "actor '{}' drives '{}', which is not an actuated end-effector joint",
                        actor.name, joint.name
                    )));
                }
                if !driven_joint.direction.is_finite() || driven_joint.direction == 0.0 {
                    return Err(fail(format!(
                        "actor '{}' moves '{}' with direction {}; it must be finite and nonzero",
                        actor.name, joint.name, driven_joint.direction
                    )));
                }
                if !driven.insert(driven_joint.joint_id.as_str()) {
                    return Err(fail(format!(
                        "joint '{}' is driven by more than one actor",
                        joint.name
                    )));
                }
            }
        }
        let listed: BTreeSet<&str> = self.jaw_joint_ids.iter().map(String::as_str).collect();
        if listed != driven {
            return Err(fail(
                "jawJointIds must list exactly the joints the actors drive".to_owned(),
            ));
        }
        if self.finger_link_ids.is_empty() {
            return Err(fail("a hand needs finger links".to_owned()));
        }
        for link_id in &self.finger_link_ids {
            if robot.link(link_id).is_none() {
                return Err(DomainError::dangling(
                    "link",
                    link_id.clone(),
                    format!("gripper '{}'.fingerLinkIds", self.name),
                ));
            }
        }
        let mut preshape_names = BTreeSet::new();
        for preshape in &self.preshapes {
            if preshape.name.trim().is_empty() {
                return Err(DomainError::EmptyName {
                    kind: "preshape name",
                });
            }
            if !preshape_names.insert(preshape.name.as_str()) {
                return Err(fail(format!(
                    "two preshapes are called '{}'",
                    preshape.name
                )));
            }
            for (joint_id, value) in &preshape.joint_values {
                if !driven.contains(joint_id.as_str()) {
                    return Err(fail(format!(
                        "preshape '{}' sets '{joint_id}', which no actor drives",
                        preshape.name
                    )));
                }
                let joint = robot
                    .joint(joint_id)
                    .ok_or_else(|| DomainError::dangling("joint", joint_id.clone(), "preshape"))?;
                if !value.is_finite() || !joint.limits.contains(*value, 1.0e-9) {
                    return Err(fail(format!(
                        "preshape '{}' puts '{}' at {value}, outside [{}, {}]",
                        preshape.name, joint.name, joint.limits.lower, joint.limits.upper
                    )));
                }
            }
        }
        Ok(())
    }

    /// Checks that the palm link and grasp frame exist on `robot`.
    fn validate_mounting(&self, robot: &RobotSystem) -> DomainResult<()> {
        if robot.link(&self.palm_link_id).is_none() {
            return Err(DomainError::dangling(
                "link",
                self.palm_link_id.clone(),
                format!("gripper '{}'.palmLinkId", self.name),
            ));
        }
        if robot.tool_frame(&self.grasp_frame_id).is_none() {
            return Err(DomainError::dangling(
                "tool frame",
                self.grasp_frame_id.clone(),
                format!("gripper '{}'.graspFrameId", self.name),
            ));
        }
        Ok(())
    }

    /// Checks that the jaws are prismatic end-effector joints with one finger each.
    fn validate_jaws(&self, robot: &RobotSystem) -> DomainResult<()> {
        if self.jaw_joint_ids.is_empty() {
            return Err(DomainError::NoGripper {
                robot: robot.name.clone(),
                detail: format!("gripper '{}' lists no jaw joints", self.name),
            });
        }
        if self.jaw_joint_ids.len() != self.finger_link_ids.len() {
            return Err(DomainError::NoGripper {
                robot: robot.name.clone(),
                detail: format!(
                    "gripper '{}' has {} jaw joints but {} finger links; each jaw needs exactly one",
                    self.name,
                    self.jaw_joint_ids.len(),
                    self.finger_link_ids.len()
                ),
            });
        }
        for link_id in &self.finger_link_ids {
            if robot.link(link_id).is_none() {
                return Err(DomainError::dangling(
                    "link",
                    link_id.clone(),
                    format!("gripper '{}'.fingerLinkIds", self.name),
                ));
            }
        }
        for joint_id in &self.jaw_joint_ids {
            let joint = robot.joint(joint_id).ok_or_else(|| {
                DomainError::dangling(
                    "joint",
                    joint_id.clone(),
                    format!("gripper '{}'.jawJointIds", self.name),
                )
            })?;
            if joint.kind != JointKind::Prismatic {
                return Err(DomainError::NoGripper {
                    robot: robot.name.clone(),
                    detail: format!(
                        "gripper '{}' jaw '{}' is {:?}; jaws must be prismatic",
                        self.name, joint.name, joint.kind
                    ),
                });
            }
            if joint.role != JointRole::EndEffector {
                return Err(DomainError::NoGripper {
                    robot: robot.name.clone(),
                    detail: format!(
                        "gripper '{}' jaw '{}' has role Manipulator; jaws must be EndEffector so \
                         they do not count as manipulator axes",
                        self.name, joint.name
                    ),
                });
            }
        }

        Ok(())
    }

    /// Checks the axis geometry, the declared dimensions and the jaw stroke.
    fn validate_axes_and_stroke(&self, robot: &RobotSystem) -> DomainResult<()> {
        let approach = self
            .approach_axis
            .normalized(&format!("gripper '{}' approach", self.name))?;
        let closing = self
            .closing_axis
            .normalized(&format!("gripper '{}' closing", self.name))?;
        // A gripper that closes along its own approach direction would have to
        // drive its fingers through the object it is reaching for.
        if approach.dot(&closing).abs() > 1.0e-6 {
            return Err(DomainError::NoGripper {
                robot: robot.name.clone(),
                detail: format!(
                    "gripper '{}' approach and closing axes are not perpendicular (dot = {:.6})",
                    self.name,
                    approach.dot(&closing)
                ),
            });
        }
        for (value, what) in [
            (self.max_opening, "maxOpening"),
            (self.finger_depth, "fingerDepth"),
            (self.finger_width, "fingerWidth"),
            (self.max_grip_force, "maxGripForce"),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("gripper '{}'.{what}", self.name),
                });
            }
            if value <= 0.0 {
                return Err(DomainError::NegativeValue {
                    context: format!("gripper '{}'.{what} must be positive", self.name),
                    value,
                });
            }
        }

        if self.is_hand() {
            // A hand's opening is set by its preshapes, not by a jaw stroke.
            return Ok(());
        }
        // The jaw limits are the real stroke; `max_opening` must not promise
        // more than the joints can deliver, or a planner would accept grasps
        // the hardware cannot make.
        let reachable: f64 = self
            .jaw_joint_ids
            .iter()
            .filter_map(|id| robot.joint(id))
            .map(|joint| self.opening_for_jaw_value(joint.limits.upper))
            .fold(f64::INFINITY, f64::min);
        if reachable.is_finite() && self.max_opening > reachable + 1.0e-9 {
            return Err(DomainError::NoGripper {
                robot: robot.name.clone(),
                detail: format!(
                    "gripper '{}' declares maxOpening {:.4} m but its jaw limits reach only \
                     {:.4} m",
                    self.name, self.max_opening, reachable
                ),
            });
        }
        Ok(())
    }
}

/// A robot: a tree of links joined by joints, with named tool frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RobotSystem {
    /// Stable identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Identifier of the link fixed to the world.
    pub root_link_id: String,
    /// Every link, in no particular order.
    pub links: Vec<Link>,
    /// Every joint, in no particular order.
    pub joints: Vec<Joint>,
    /// Named frames users can command.
    #[serde(default)]
    pub tool_frames: Vec<ToolFrame>,
    /// Identifier of the tool frame used by default for IK targets.
    #[serde(default)]
    pub default_tool_frame_id: Option<String>,
    /// The end effector, when one is fitted.
    #[serde(default)]
    pub gripper: Option<Gripper>,
    /// Pose of the robot's root link in world coordinates.
    #[serde(default)]
    pub base_pose: Transform,
}

impl RobotSystem {
    /// Looks up a link by identifier.
    pub fn link(&self, id: &str) -> Option<&Link> {
        self.links.iter().find(|l| l.id == id)
    }

    /// Looks up a joint by identifier.
    pub fn joint(&self, id: &str) -> Option<&Joint> {
        self.joints.iter().find(|j| j.id == id)
    }

    /// Looks up a tool frame by identifier.
    pub fn tool_frame(&self, id: &str) -> Option<&ToolFrame> {
        self.tool_frames.iter().find(|f| f.id == id)
    }

    /// Looks up a link by display name.
    pub fn link_by_name(&self, name: &str) -> Option<&Link> {
        self.links.iter().find(|l| l.name == name)
    }

    /// Looks up a joint by display name.
    pub fn joint_by_name(&self, name: &str) -> Option<&Joint> {
        self.joints.iter().find(|j| j.name == name)
    }

    /// Joints that count toward the manipulator's axis budget, in tree order.
    pub fn manipulator_joints(&self) -> Vec<&Joint> {
        self.joints_in_tree_order()
            .into_iter()
            .filter(|j| j.is_manipulator_axis())
            .collect()
    }

    /// Joints that actuate the end effector, in tree order.
    pub fn end_effector_joints(&self) -> Vec<&Joint> {
        self.joints_in_tree_order()
            .into_iter()
            .filter(|j| j.is_end_effector())
            .collect()
    }

    /// The number of manipulator axes.
    pub fn axis_count(&self) -> usize {
        self.joints
            .iter()
            .filter(|j| j.is_manipulator_axis())
            .count()
    }

    /// The total number of degrees of freedom, gripper jaws included.
    pub fn total_dof(&self) -> usize {
        self.joints.iter().filter(|j| j.is_actuated()).count()
    }

    /// The gripper, or an error naming the robot that lacks one.
    ///
    /// # Errors
    /// [`DomainError::NoGripper`] when no end effector is fitted.
    pub fn require_gripper(&self) -> DomainResult<&Gripper> {
        self.gripper.as_ref().ok_or_else(|| DomainError::NoGripper {
            robot: self.name.clone(),
            detail: "no gripper is fitted; attach one before planning grasps".to_owned(),
        })
    }

    /// The tool frame IK targets by default, if one can be determined.
    ///
    /// Falls back to the gripper's grasp frame, then to the only tool frame
    /// present. With several frames and no default declared, returns `None`
    /// rather than guessing which one the user meant.
    pub fn default_tool_frame(&self) -> Option<&ToolFrame> {
        if let Some(id) = &self.default_tool_frame_id {
            return self.tool_frame(id);
        }
        if let Some(gripper) = &self.gripper {
            if let Some(frame) = self.tool_frame(&gripper.grasp_frame_id) {
                return Some(frame);
            }
        }
        match self.tool_frames.as_slice() {
            [only] => Some(only),
            _ => None,
        }
    }

    /// Child joints keyed by parent link identifier, each list sorted by joint name.
    ///
    /// Sorting by name rather than insertion order makes the tree walk — and so
    /// the order of joint values in every vector the engine exchanges —
    /// independent of how the project file happened to be written.
    pub fn children_by_parent(&self) -> BTreeMap<&str, Vec<&Joint>> {
        let mut map: BTreeMap<&str, Vec<&Joint>> = BTreeMap::new();
        for joint in &self.joints {
            map.entry(joint.parent_link_id.as_str())
                .or_default()
                .push(joint);
        }
        for children in map.values_mut() {
            children.sort_by(|a, b| a.name.cmp(&b.name));
        }
        map
    }

    /// Every joint in depth-first tree order from the root.
    ///
    /// Joints unreachable from the root are omitted; [`RobotSystem::validate`]
    /// rejects such a robot, so a validated robot lists all of them.
    pub fn joints_in_tree_order(&self) -> Vec<&Joint> {
        let children = self.children_by_parent();
        let mut ordered = Vec::with_capacity(self.joints.len());
        let mut stack = vec![self.root_link_id.as_str()];
        let mut visited: BTreeSet<&str> = BTreeSet::new();
        while let Some(link_id) = stack.pop() {
            if !visited.insert(link_id) {
                continue;
            }
            if let Some(joints) = children.get(link_id) {
                // Pushed reversed so that popping yields the sorted order.
                for joint in joints.iter().rev() {
                    stack.push(joint.child_link_id.as_str());
                }
                for joint in joints {
                    ordered.push(*joint);
                }
            }
        }
        ordered
    }

    /// The actuated joints, in tree order: the engine's canonical joint vector.
    pub fn actuated_joints(&self) -> Vec<&Joint> {
        self.joints_in_tree_order()
            .into_iter()
            .filter(|j| j.is_actuated())
            .collect()
    }

    /// Validates identity, uniqueness, tree shape, axis budget and the gripper.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "robot id" });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "robot name" });
        }
        self.base_pose
            .normalized(&format!("robot '{}' basePose", self.name))?;

        let link_ids = self.validate_links()?;
        self.validate_joints(&link_ids)?;
        self.validate_reachability()?;
        self.validate_tool_frames(&link_ids)?;

        let axes = self.axis_count();
        if !(MIN_AXES..=MAX_AXES).contains(&axes) {
            return Err(DomainError::UnsupportedAxisCount {
                count: axes,
                min: MIN_AXES,
                max: MAX_AXES,
            });
        }

        if let Some(gripper) = &self.gripper {
            gripper.validate(self)?;
        }
        Ok(())
    }

    /// Validates every link and returns the set of link identifiers.
    fn validate_links(&self) -> DomainResult<BTreeSet<&str>> {
        let mut link_ids: BTreeSet<&str> = BTreeSet::new();
        let mut link_names: BTreeSet<&str> = BTreeSet::new();
        for link in &self.links {
            link.validate()?;
            if !link_ids.insert(&link.id) {
                return Err(DomainError::DuplicateId {
                    kind: "link",
                    id: link.id.clone(),
                });
            }
            if !link_names.insert(&link.name) {
                return Err(DomainError::DuplicateName {
                    kind: "link",
                    name: link.name.clone(),
                });
            }
        }
        if !link_ids.contains(self.root_link_id.as_str()) {
            return Err(DomainError::dangling(
                "link",
                self.root_link_id.clone(),
                format!("robot '{}'.rootLinkId", self.name),
            ));
        }

        Ok(link_ids)
    }

    /// Validates every joint and the tree shape its parentage implies.
    fn validate_joints(&self, link_ids: &BTreeSet<&str>) -> DomainResult<()> {
        let mut joint_ids: BTreeSet<&str> = BTreeSet::new();
        let mut joint_names: BTreeSet<&str> = BTreeSet::new();
        let mut parented: BTreeMap<&str, &str> = BTreeMap::new();
        for joint in &self.joints {
            joint.validate()?;
            if !joint_ids.insert(&joint.id) {
                return Err(DomainError::DuplicateId {
                    kind: "joint",
                    id: joint.id.clone(),
                });
            }
            if !joint_names.insert(&joint.name) {
                return Err(DomainError::DuplicateName {
                    kind: "joint",
                    name: joint.name.clone(),
                });
            }
            for (reference, field) in [
                (&joint.parent_link_id, "parentLinkId"),
                (&joint.child_link_id, "childLinkId"),
            ] {
                if !link_ids.contains(reference.as_str()) {
                    return Err(DomainError::dangling(
                        "link",
                        reference.clone(),
                        format!("joint '{}'.{field}", joint.name),
                    ));
                }
            }
            // One parent per link is what makes this a tree rather than a graph;
            // forward kinematics has no answer for a link reached two ways.
            if let Some(existing) = parented.insert(&joint.child_link_id, &joint.name) {
                return Err(DomainError::InvalidTree {
                    reason: format!(
                        "link '{}' is the child of both joint '{}' and joint '{}'",
                        joint.child_link_id, existing, joint.name
                    ),
                });
            }
            if joint.child_link_id == self.root_link_id {
                return Err(DomainError::InvalidTree {
                    reason: format!(
                        "joint '{}' makes the root link '{}' a child; the root has no parent",
                        joint.name, self.root_link_id
                    ),
                });
            }
        }

        Ok(())
    }

    /// Checks that every joint is reachable from the root link.
    fn validate_reachability(&self) -> DomainResult<()> {
        let reachable = self.joints_in_tree_order().len();
        if reachable != self.joints.len() {
            // Every link has at most one parent and the root has none, so any
            // joint the walk missed sits on a cycle or a detached fragment.
            let reached: BTreeSet<&str> = self
                .joints_in_tree_order()
                .into_iter()
                .map(|j| j.id.as_str())
                .collect();
            let orphan = self
                .joints
                .iter()
                .find(|j| !reached.contains(j.id.as_str()))
                .map_or_else(String::new, |j| j.name.clone());
            return Err(DomainError::InvalidTree {
                reason: format!(
                    "joint '{orphan}' is not reachable from root link '{}'; the robot is a \
                     disconnected graph or contains a cycle",
                    self.root_link_id
                ),
            });
        }

        Ok(())
    }

    /// Validates the tool frames and the declared default.
    fn validate_tool_frames(&self, link_ids: &BTreeSet<&str>) -> DomainResult<()> {
        let mut frame_ids: BTreeSet<&str> = BTreeSet::new();
        let mut frame_names: BTreeSet<&str> = BTreeSet::new();
        for frame in &self.tool_frames {
            frame.validate()?;
            if !frame_ids.insert(&frame.id) {
                return Err(DomainError::DuplicateId {
                    kind: "tool frame",
                    id: frame.id.clone(),
                });
            }
            if !frame_names.insert(&frame.name) {
                return Err(DomainError::DuplicateName {
                    kind: "tool frame",
                    name: frame.name.clone(),
                });
            }
            if !link_ids.contains(frame.link_id.as_str()) {
                return Err(DomainError::dangling(
                    "link",
                    frame.link_id.clone(),
                    format!("tool frame '{}'.linkId", frame.name),
                ));
            }
        }
        if let Some(id) = &self.default_tool_frame_id {
            if !frame_ids.contains(id.as_str()) {
                return Err(DomainError::dangling(
                    "tool frame",
                    id.clone(),
                    format!("robot '{}'.defaultToolFrameId", self.name),
                ));
            }
        }
        Ok(())
    }

    /// Joint values at the midpoint of every actuated joint's range, in tree order.
    pub fn neutral_configuration(&self) -> Vec<f64> {
        self.actuated_joints()
            .iter()
            .map(|j| j.limits.center())
            .collect()
    }

    /// Joint values at zero where the range allows it, clamped where it does not.
    pub fn zero_configuration(&self) -> Vec<f64> {
        self.actuated_joints()
            .iter()
            .map(|j| j.limits.clamp(0.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    fn two_link_robot() -> RobotSystem {
        let base = Link::new("l-base", "base");
        let arm = Link::new("l-arm", "arm");
        let joint = Joint::revolute(
            "j-1",
            "axis 1",
            "l-base",
            "l-arm",
            Transform::IDENTITY,
            Vec3::Z,
            std::f64::consts::PI,
        )
        .unwrap();
        RobotSystem {
            id: "r".to_owned(),
            name: "two link".to_owned(),
            root_link_id: "l-base".to_owned(),
            links: vec![base, arm],
            joints: vec![joint],
            tool_frames: vec![ToolFrame::new("f", "flange", "l-arm", Transform::IDENTITY)],
            default_tool_frame_id: None,
            gripper: None,
            base_pose: Transform::IDENTITY,
        }
    }

    #[test]
    fn a_minimal_robot_validates() {
        two_link_robot().validate().expect("valid");
    }

    #[test]
    fn a_cycle_is_rejected_as_unreachable() {
        let mut robot = two_link_robot();
        robot.links.push(Link::new("l-c", "c"));
        robot.joints.push(
            Joint::revolute(
                "j-2",
                "axis 2",
                "l-c",
                "l-arm",
                Transform::IDENTITY,
                Vec3::Z,
                1.0,
            )
            .unwrap(),
        );
        // 'l-arm' now has two parents, which is caught before reachability.
        assert!(matches!(
            robot.validate(),
            Err(DomainError::InvalidTree { .. })
        ));
    }

    #[test]
    fn a_detached_fragment_is_rejected() {
        let mut robot = two_link_robot();
        robot.links.push(Link::new("l-x", "x"));
        robot.links.push(Link::new("l-y", "y"));
        robot.joints.push(
            Joint::revolute(
                "j-2",
                "axis 2",
                "l-x",
                "l-y",
                Transform::IDENTITY,
                Vec3::Z,
                1.0,
            )
            .unwrap(),
        );
        let error = robot.validate().expect_err("detached fragment");
        assert!(matches!(error, DomainError::InvalidTree { .. }), "{error}");
        assert!(error.to_string().contains("axis 2"), "{error}");
    }

    #[test]
    fn the_root_may_not_be_a_child() {
        let mut robot = two_link_robot();
        robot.joints[0].child_link_id = "l-base".to_owned();
        robot.joints[0].parent_link_id = "l-arm".to_owned();
        let error = robot.validate().expect_err("root as child");
        assert!(error.to_string().contains("root"), "{error}");
    }

    #[test]
    fn zero_axes_is_rejected() {
        let mut robot = two_link_robot();
        robot.joints[0].kind = JointKind::Fixed;
        assert!(matches!(
            robot.validate(),
            Err(DomainError::UnsupportedAxisCount { count: 0, .. })
        ));
    }

    #[test]
    fn a_single_tool_frame_is_the_default_without_being_declared() {
        let robot = two_link_robot();
        assert_eq!(robot.default_tool_frame().map(|f| f.id.as_str()), Some("f"));
    }

    #[test]
    fn with_two_frames_and_no_declaration_there_is_no_default() {
        let mut robot = two_link_robot();
        robot
            .tool_frames
            .push(ToolFrame::new("f2", "torch", "l-arm", Transform::IDENTITY));
        assert!(robot.default_tool_frame().is_none());
    }

    #[test]
    fn tree_order_is_independent_of_declaration_order() {
        let mut robot = two_link_robot();
        robot.links.push(Link::new("l-c", "c"));
        robot.joints.push(
            Joint::revolute(
                "j-2",
                "axis 2",
                "l-arm",
                "l-c",
                Transform::IDENTITY,
                Vec3::Z,
                1.0,
            )
            .unwrap(),
        );
        let forward: Vec<String> = robot
            .joints_in_tree_order()
            .iter()
            .map(|j| j.name.clone())
            .collect();
        robot.joints.reverse();
        let reversed: Vec<String> = robot
            .joints_in_tree_order()
            .iter()
            .map(|j| j.name.clone())
            .collect();
        assert_eq!(forward, reversed);
        assert_eq!(forward, vec!["axis 1", "axis 2"]);
    }

    #[test]
    fn a_missing_gripper_names_the_robot() {
        let error = two_link_robot().require_gripper().expect_err("no gripper");
        assert!(error.to_string().contains("two link"), "{error}");
    }
}
