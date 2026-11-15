//! A kinematic model: the robot flattened into the form the solvers want.
//!
//! [`axisforge_domain::RobotSystem`] is shaped for editing and for saving —
//! entities in any order, referenced by string id. That is the wrong shape for
//! a solver, which walks the tree thousands of times per second and wants
//! integer indices and a fixed joint ordering. This module does that conversion
//! once, up front, and validates the robot while it is at it, so every function
//! downstream can assume a well-formed tree and skip re-checking it.
//!
//! The joint ordering is the robot's depth-first tree order, which
//! [`axisforge_domain::RobotSystem::joints_in_tree_order`] makes independent of
//! how the project file happened to be written. That ordering is the engine's
//! contract: every joint-value vector crossing a boundary — a command, a saved
//! waypoint, an IK answer — is in it.

use std::collections::BTreeMap;

use axisforge_domain::{
    DomainError, DomainResult, Joint, JointKind, JointLimits, JointRole, RobotSystem, Transform,
    Vec3,
};

/// A link, as the solvers see it.
#[derive(Debug, Clone)]
pub struct LinkNode {
    /// The link's domain identifier.
    pub id: String,
    /// The link's display name.
    pub name: String,
    /// Index of the joint whose child this link is, or `None` for the root.
    pub parent_joint: Option<usize>,
}

/// A joint, as the solvers see it.
#[derive(Debug, Clone)]
pub struct JointNode {
    /// The joint's domain identifier.
    pub id: String,
    /// The joint's display name.
    pub name: String,
    /// The kind of motion.
    pub kind: JointKind,
    /// What the joint is for.
    pub role: JointRole,
    /// Pose of the joint frame in the parent link's frame, at value zero.
    pub origin: Transform,
    /// Motion axis in the joint frame, normalized at construction.
    pub axis: Vec3,
    /// Allowed range of the joint value, SI units.
    pub limits: JointLimits,
    /// Index of the parent link.
    pub parent_link: usize,
    /// Index of the child link.
    pub child_link: usize,
    /// Position of this joint's value in a configuration vector.
    ///
    /// `None` for a fixed joint, which has no value.
    pub value_index: Option<usize>,
}

impl JointNode {
    /// `true` when this joint has a value in a configuration vector.
    pub fn is_actuated(&self) -> bool {
        self.value_index.is_some()
    }

    /// The transform this joint contributes at `value`, in the joint frame.
    fn motion(&self, value: f64) -> Transform {
        match self.kind {
            JointKind::Revolute => Transform::from_rotation(
                // The axis was normalized when the model was built, so this
                // cannot fail; a rotation about a unit axis is always valid.
                axisforge_domain::Quat::from_axis_angle(&self.axis, value)
                    .unwrap_or(axisforge_domain::Quat::IDENTITY),
            ),
            JointKind::Prismatic => Transform::from_translation(self.axis.scaled(value)),
            JointKind::Fixed => Transform::IDENTITY,
        }
    }

    /// The pose of the child link in the parent's frame at `value`.
    pub fn child_pose(&self, value: f64) -> Transform {
        self.origin.compose(&self.motion(value))
    }
}

/// A named frame rigidly attached to a link.
#[derive(Debug, Clone)]
pub struct FrameNode {
    /// The frame's domain identifier.
    pub id: String,
    /// The frame's display name.
    pub name: String,
    /// Index of the link the frame is attached to.
    pub link: usize,
    /// Pose of the frame in that link's frame.
    pub offset: Transform,
}

/// The robot, flattened and indexed for the solvers.
#[derive(Debug, Clone)]
pub struct KinematicModel {
    links: Vec<LinkNode>,
    joints: Vec<JointNode>,
    frames: Vec<FrameNode>,
    link_by_id: BTreeMap<String, usize>,
    joint_by_id: BTreeMap<String, usize>,
    frame_by_id: BTreeMap<String, usize>,
    /// Indices into [`KinematicModel::joints`] in configuration-vector order.
    value_to_joint: Vec<usize>,
    root_link: usize,
    base_pose: Transform,
    default_frame: Option<usize>,
    /// Links in an order where a parent always precedes its children.
    topological_links: Vec<usize>,
}

impl KinematicModel {
    /// Builds a model from a robot, validating the robot in the process.
    ///
    /// # Errors
    /// Anything [`RobotSystem::validate`] rejects.
    pub fn from_robot(robot: &RobotSystem) -> DomainResult<Self> {
        robot.validate()?;

        let mut links = Vec::with_capacity(robot.links.len());
        let mut link_by_id = BTreeMap::new();
        for (index, link) in robot.links.iter().enumerate() {
            link_by_id.insert(link.id.clone(), index);
            links.push(LinkNode {
                id: link.id.clone(),
                name: link.name.clone(),
                parent_joint: None,
            });
        }
        let root_link = *link_by_id.get(&robot.root_link_id).ok_or_else(|| {
            DomainError::dangling("link", robot.root_link_id.clone(), "model root")
        })?;

        let ordered: Vec<&Joint> = robot.joints_in_tree_order();
        let mut joints = Vec::with_capacity(ordered.len());
        let mut joint_by_id = BTreeMap::new();
        let mut value_to_joint = Vec::new();
        for joint in ordered {
            let parent_link = link_by_id[&joint.parent_link_id];
            let child_link = link_by_id[&joint.child_link_id];
            let value_index = if joint.is_actuated() {
                value_to_joint.push(joints.len());
                Some(value_to_joint.len() - 1)
            } else {
                None
            };
            // Normalize once here so every forward-kinematics call afterwards
            // is pure arithmetic with nothing left to fail.
            let axis = if joint.is_actuated() {
                joint
                    .axis
                    .normalized(&format!("joint '{}' axis", joint.name))?
            } else {
                Vec3::Z
            };
            links[child_link].parent_joint = Some(joints.len());
            joint_by_id.insert(joint.id.clone(), joints.len());
            joints.push(JointNode {
                id: joint.id.clone(),
                name: joint.name.clone(),
                kind: joint.kind,
                role: joint.role,
                origin: joint.origin,
                axis,
                limits: joint.limits,
                parent_link,
                child_link,
                value_index,
            });
        }

        let mut frames = Vec::with_capacity(robot.tool_frames.len());
        let mut frame_by_id = BTreeMap::new();
        for frame in &robot.tool_frames {
            frame_by_id.insert(frame.id.clone(), frames.len());
            frames.push(FrameNode {
                id: frame.id.clone(),
                name: frame.name.clone(),
                link: link_by_id[&frame.link_id],
                offset: frame.offset,
            });
        }
        let default_frame = robot
            .default_tool_frame()
            .and_then(|frame| frame_by_id.get(&frame.id).copied());

        // A parent always precedes its children here, so forward kinematics is
        // a single pass with no recursion and no revisiting.
        let mut topological_links = Vec::with_capacity(links.len());
        topological_links.push(root_link);
        let mut cursor = 0;
        while cursor < topological_links.len() {
            let parent = topological_links[cursor];
            cursor += 1;
            for joint in &joints {
                if joint.parent_link == parent {
                    topological_links.push(joint.child_link);
                }
            }
        }
        debug_assert_eq!(
            topological_links.len(),
            links.len(),
            "a validated robot's links are all reachable from the root"
        );

        Ok(Self {
            links,
            joints,
            frames,
            link_by_id,
            joint_by_id,
            frame_by_id,
            value_to_joint,
            root_link,
            base_pose: robot.base_pose,
            default_frame,
            topological_links,
        })
    }

    /// The links, in the robot's declaration order.
    pub fn links(&self) -> &[LinkNode] {
        &self.links
    }

    /// The joints, in depth-first tree order.
    pub fn joints(&self) -> &[JointNode] {
        &self.joints
    }

    /// The tool frames, in the robot's declaration order.
    pub fn frames(&self) -> &[FrameNode] {
        &self.frames
    }

    /// Links in an order where every parent precedes its children.
    pub fn topological_links(&self) -> &[usize] {
        &self.topological_links
    }

    /// Index of the root link.
    pub fn root_link(&self) -> usize {
        self.root_link
    }

    /// Pose of the root link in world coordinates.
    pub fn base_pose(&self) -> Transform {
        self.base_pose
    }

    /// The number of actuated joints, and so the width of a configuration vector.
    pub fn dof(&self) -> usize {
        self.value_to_joint.len()
    }

    /// The joint behind a given configuration-vector slot.
    pub fn joint_for_value(&self, value_index: usize) -> Option<&JointNode> {
        self.value_to_joint
            .get(value_index)
            .map(|index| &self.joints[*index])
    }

    /// Configuration-vector slots that are manipulator axes, in order.
    ///
    /// These are the columns inverse kinematics is allowed to move. A gripper's
    /// jaws are actuated and appear in a configuration vector, but they do not
    /// appear here: a solver that moved them to reach a pose would open the
    /// gripper to make up for a shortfall in arm reach.
    pub fn manipulator_value_indices(&self) -> Vec<usize> {
        self.value_to_joint
            .iter()
            .enumerate()
            .filter(|(_, joint)| self.joints[**joint].role == JointRole::Manipulator)
            .map(|(value_index, _)| value_index)
            .collect()
    }

    /// Configuration-vector slots that actuate the end effector, in order.
    pub fn end_effector_value_indices(&self) -> Vec<usize> {
        self.value_to_joint
            .iter()
            .enumerate()
            .filter(|(_, joint)| self.joints[**joint].role == JointRole::EndEffector)
            .map(|(value_index, _)| value_index)
            .collect()
    }

    /// Looks up a link index by identifier.
    pub fn link_index(&self, id: &str) -> Option<usize> {
        self.link_by_id.get(id).copied()
    }

    /// Looks up a joint index by identifier.
    pub fn joint_index(&self, id: &str) -> Option<usize> {
        self.joint_by_id.get(id).copied()
    }

    /// Looks up a tool-frame index by identifier.
    pub fn frame_index(&self, id: &str) -> Option<usize> {
        self.frame_by_id.get(id).copied()
    }

    /// Index of the default tool frame, when the robot declares or implies one.
    pub fn default_frame(&self) -> Option<usize> {
        self.default_frame
    }

    /// The limits of every actuated joint, in configuration-vector order.
    pub fn limits(&self) -> Vec<JointLimits> {
        self.value_to_joint
            .iter()
            .map(|index| self.joints[*index].limits)
            .collect()
    }

    /// A configuration at the midpoint of every joint's range.
    pub fn neutral_configuration(&self) -> Vec<f64> {
        self.limits().iter().map(JointLimits::center).collect()
    }

    /// `values` clamped into every joint's limits.
    pub fn clamped(&self, values: &[f64]) -> Vec<f64> {
        self.limits()
            .iter()
            .zip(values)
            .map(|(limits, value)| limits.clamp(*value))
            .collect()
    }

    /// Checks that `values` has the right width and holds only finite numbers.
    ///
    /// Limits are *not* enforced here: a solver iterate may legitimately step
    /// outside them before being projected back, and forward kinematics of an
    /// out-of-limit pose is well defined and useful to show the user.
    ///
    /// # Errors
    /// [`DomainError::JointCountMismatch`] or [`DomainError::NonFinite`].
    pub fn check_configuration(&self, values: &[f64]) -> DomainResult<()> {
        if values.len() != self.dof() {
            return Err(DomainError::JointCountMismatch {
                expected: self.dof(),
                actual: values.len(),
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                let name = self
                    .joint_for_value(index)
                    .map_or_else(|| index.to_string(), |joint| joint.name.clone());
                return Err(DomainError::NonFinite {
                    context: format!("joint '{name}' value"),
                });
            }
        }
        Ok(())
    }

    /// The first value in `values` that lies outside its joint's limits.
    ///
    /// Returns the joint's name, the value, and the limits it violated.
    pub fn first_limit_violation(
        &self,
        values: &[f64],
        tolerance: f64,
    ) -> Option<(String, f64, JointLimits)> {
        for (index, value) in values.iter().enumerate() {
            let joint = self.joint_for_value(index)?;
            if !joint.limits.contains(*value, tolerance) {
                return Some((joint.name.clone(), *value, joint.limits));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates;

    #[test]
    fn a_six_axis_arm_has_six_degrees_of_freedom() {
        let model = KinematicModel::from_robot(&templates::six_axis().unwrap()).unwrap();
        assert_eq!(model.dof(), 6);
        assert_eq!(model.manipulator_value_indices().len(), 6);
        assert!(model.end_effector_value_indices().is_empty());
    }

    #[test]
    fn a_gripper_adds_dof_but_not_manipulator_columns() {
        let model =
            KinematicModel::from_robot(&templates::six_axis_with_gripper().unwrap()).unwrap();
        assert_eq!(model.dof(), 8, "six axes plus two jaws");
        assert_eq!(
            model.manipulator_value_indices().len(),
            6,
            "inverse kinematics must not be allowed to move the jaws"
        );
        assert_eq!(model.end_effector_value_indices().len(), 2);
        // The two sets must partition the configuration vector.
        let mut all = model.manipulator_value_indices();
        all.extend(model.end_effector_value_indices());
        all.sort_unstable();
        assert_eq!(all, (0..8).collect::<Vec<usize>>());
    }

    #[test]
    fn every_parent_precedes_its_children() {
        let model =
            KinematicModel::from_robot(&templates::six_axis_with_gripper().unwrap()).unwrap();
        let mut seen = vec![false; model.links().len()];
        for index in model.topological_links() {
            if let Some(joint) = model.links()[*index].parent_joint {
                assert!(
                    seen[model.joints()[joint].parent_link],
                    "link {} was visited before its parent",
                    model.links()[*index].name
                );
            }
            seen[*index] = true;
        }
        assert!(seen.iter().all(|s| *s), "every link must be visited");
    }

    #[test]
    fn a_configuration_of_the_wrong_width_is_rejected() {
        let model = KinematicModel::from_robot(&templates::planar_2r().unwrap()).unwrap();
        assert!(matches!(
            model.check_configuration(&[0.0]),
            Err(DomainError::JointCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn a_non_finite_value_names_the_joint() {
        let model = KinematicModel::from_robot(&templates::planar_2r().unwrap()).unwrap();
        let error = model
            .check_configuration(&[0.0, f64::NAN])
            .expect_err("NaN");
        assert!(error.to_string().contains("axis 2"), "{error}");
    }

    #[test]
    fn limits_are_not_enforced_by_the_configuration_check() {
        let model = KinematicModel::from_robot(&templates::planar_2r().unwrap()).unwrap();
        model
            .check_configuration(&[100.0, -100.0])
            .expect("an out-of-limit iterate is a valid configuration to evaluate");
        let violation = model
            .first_limit_violation(&[100.0, 0.0], 0.0)
            .expect("out of limits");
        assert_eq!(violation.0, "axis 1");
    }

    #[test]
    fn the_jaw_joints_are_prismatic_in_the_model() {
        let model =
            KinematicModel::from_robot(&templates::six_axis_with_gripper().unwrap()).unwrap();
        for index in model.end_effector_value_indices() {
            let joint = model.joint_for_value(index).expect("jaw");
            assert_eq!(joint.kind, JointKind::Prismatic, "{}", joint.name);
        }
    }

    #[test]
    fn the_default_frame_is_the_grasp_frame_when_a_gripper_is_fitted() {
        let robot = templates::six_axis_with_gripper().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let expected = robot.require_gripper().unwrap().grasp_frame_id.clone();
        let index = model.default_frame().expect("a default frame");
        assert_eq!(model.frames()[index].id, expected);
    }
}
