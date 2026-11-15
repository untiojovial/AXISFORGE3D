//! Forward kinematics: where every link is, given the joint values.
//!
//! One pass over the links in topological order, composing each joint's
//! contribution onto its parent's world pose. No recursion, no allocation
//! beyond the output buffer, and nothing that can fail — the model validated
//! the tree and normalized every axis when it was built, so this is pure
//! arithmetic.

use axisforge_domain::{DomainResult, Transform};

use crate::model::KinematicModel;

/// The world pose of every link, for one configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    link_world: Vec<Transform>,
}

impl Pose {
    /// The world pose of a link by index.
    ///
    /// # Panics
    /// If `link` is not a valid index for the model this pose came from.
    pub fn link(&self, link: usize) -> Transform {
        self.link_world[link]
    }

    /// The world pose of a link by index, or `None` when out of range.
    pub fn try_link(&self, link: usize) -> Option<Transform> {
        self.link_world.get(link).copied()
    }

    /// Every link's world pose, indexed as the model indexes links.
    pub fn link_poses(&self) -> &[Transform] {
        &self.link_world
    }
}

/// Computes the world pose of every link for `values`.
///
/// `values` is in the model's configuration order: one entry per actuated
/// joint, in depth-first tree order.
///
/// # Errors
/// [`axisforge_domain::DomainError::JointCountMismatch`] for the wrong width,
/// or [`axisforge_domain::DomainError::NonFinite`] for a non-finite value.
pub fn forward_kinematics(model: &KinematicModel, values: &[f64]) -> DomainResult<Pose> {
    model.check_configuration(values)?;
    Ok(forward_kinematics_unchecked(model, values))
}

/// Computes forward kinematics without re-checking `values`.
///
/// For solver inner loops, which check the configuration once and then evaluate
/// thousands of candidates. A wrong-width slice yields a pose where the
/// unreachable links keep their parents' poses rather than panicking, so a
/// mistake here degrades rather than crashes; prefer [`forward_kinematics`]
/// anywhere the width is not already known to be right.
pub fn forward_kinematics_unchecked(model: &KinematicModel, values: &[f64]) -> Pose {
    let mut link_world = vec![Transform::IDENTITY; model.links().len()];
    link_world[model.root_link()] = model.base_pose();

    for link in model.topological_links() {
        let Some(joint_index) = model.links()[*link].parent_joint else {
            continue;
        };
        let joint = &model.joints()[joint_index];
        let value = joint
            .value_index
            .and_then(|index| values.get(index).copied())
            .unwrap_or(0.0);
        link_world[*link] = link_world[joint.parent_link].compose(&joint.child_pose(value));
    }

    Pose { link_world }
}

/// The world pose of a tool frame, given a computed [`Pose`].
///
/// # Panics
/// If `frame` is not a valid frame index for `model`.
pub fn frame_pose(model: &KinematicModel, pose: &Pose, frame: usize) -> Transform {
    let node = &model.frames()[frame];
    pose.link(node.link).compose(&node.offset)
}

/// The world pose of a tool frame, or `None` when `frame` is out of range.
pub fn try_frame_pose(model: &KinematicModel, pose: &Pose, frame: usize) -> Option<Transform> {
    let node = model.frames().get(frame)?;
    Some(pose.try_link(node.link)?.compose(&node.offset))
}

/// The world pose of a tool frame for a configuration, in one call.
///
/// # Errors
/// Propagates [`forward_kinematics`], or
/// [`axisforge_domain::DomainError::DanglingReference`] for an unknown frame.
pub fn tool_pose(model: &KinematicModel, values: &[f64], frame: usize) -> DomainResult<Transform> {
    let pose = forward_kinematics(model, values)?;
    try_frame_pose(model, &pose, frame).ok_or_else(|| {
        axisforge_domain::DomainError::dangling("tool frame", frame.to_string(), "tool_pose")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::{templates, Vec3};
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn a_planar_arm_at_zero_stands_straight_up() {
        let robot = templates::planar_2r().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let flange = model.default_frame().expect("a flange");
        let pose = tool_pose(&model, &[0.0, 0.0], flange).unwrap();
        // Pedestal 0.12 + link 1 at 0.3 + link 2 at 0.25 = 0.67 along +Z.
        assert!(
            pose.translation.distance(&Vec3::new(0.0, 0.0, 0.67)) < 1e-12,
            "{pose:?}"
        );
    }

    #[test]
    fn folding_the_first_axis_ninety_degrees_lays_the_arm_flat() {
        let robot = templates::planar_2r().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let flange = model.default_frame().unwrap();
        let pose = tool_pose(&model, &[FRAC_PI_2, 0.0], flange).unwrap();
        // Both links now reach along +X from the top of the pedestal.
        assert!(
            pose.translation.distance(&Vec3::new(0.55, 0.0, 0.12)) < 1e-12,
            "{pose:?}"
        );
    }

    #[test]
    fn the_second_axis_bends_relative_to_the_first() {
        let robot = templates::planar_2r().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let flange = model.default_frame().unwrap();
        let pose = tool_pose(&model, &[0.0, FRAC_PI_2], flange).unwrap();
        // Link 1 up to 0.42, then link 2 reaches along +X.
        assert!(
            pose.translation.distance(&Vec3::new(0.25, 0.0, 0.42)) < 1e-12,
            "{pose:?}"
        );
    }

    #[test]
    fn the_base_pose_offsets_the_whole_robot() {
        let mut robot = templates::planar_2r().unwrap();
        robot.base_pose = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let model = KinematicModel::from_robot(&robot).unwrap();
        let flange = model.default_frame().unwrap();
        let pose = tool_pose(&model, &[0.0, 0.0], flange).unwrap();
        assert!(
            pose.translation.distance(&Vec3::new(1.0, 2.0, 3.67)) < 1e-12,
            "{pose:?}"
        );
    }

    #[test]
    fn opening_the_gripper_moves_the_fingers_apart_and_nothing_else() {
        let robot = templates::six_axis_with_gripper().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let gripper = robot.require_gripper().unwrap();
        let grasp = model.frame_index(&gripper.grasp_frame_id).unwrap();
        let left = model.link_index(&gripper.finger_link_ids[0]).unwrap();
        let right = model.link_index(&gripper.finger_link_ids[1]).unwrap();

        let mut closed = vec![0.0; model.dof()];
        let open = {
            let mut values = closed.clone();
            for index in model.end_effector_value_indices() {
                values[index] = gripper.jaw_value_for_opening(gripper.max_opening);
            }
            values
        };

        let closed_pose = forward_kinematics(&model, &closed).unwrap();
        let open_pose = forward_kinematics(&model, &open).unwrap();

        // The grasp frame rides on the palm, so opening the jaws must not move it.
        let grasp_closed = frame_pose(&model, &closed_pose, grasp);
        let grasp_open = frame_pose(&model, &open_pose, grasp);
        assert!(
            grasp_closed.translation.distance(&grasp_open.translation) < 1e-12,
            "the grasp frame must not move when the jaws do"
        );

        let gap_closed = closed_pose
            .link(left)
            .translation
            .distance(&closed_pose.link(right).translation);
        let gap_open = open_pose
            .link(left)
            .translation
            .distance(&open_pose.link(right).translation);
        assert!(
            gap_closed < 1e-12,
            "at zero the finger frames coincide, gap was {gap_closed}"
        );
        assert!(
            (gap_open - gripper.max_opening).abs() < 1e-12,
            "fully open should be {} apart, was {gap_open}",
            gripper.max_opening
        );

        // Touch `closed` so the compiler cannot elide the clone above.
        closed[0] = 0.0;
    }

    #[test]
    fn fixed_joints_contribute_their_origin_but_no_value() {
        let robot = templates::six_axis_with_gripper().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        // The palm is attached by a fixed joint, so it has no configuration slot.
        let palm = model
            .link_index(&robot.require_gripper().unwrap().palm_link_id)
            .unwrap();
        let joint = model.links()[palm]
            .parent_joint
            .expect("the palm has a parent joint");
        assert!(!model.joints()[joint].is_actuated());
    }

    #[test]
    fn the_unchecked_path_agrees_with_the_checked_one() {
        let robot = templates::six_axis().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let values = [0.3, -0.4, 0.5, 1.0, -0.2, 0.7];
        assert_eq!(
            forward_kinematics(&model, &values).unwrap(),
            forward_kinematics_unchecked(&model, &values)
        );
    }
}
