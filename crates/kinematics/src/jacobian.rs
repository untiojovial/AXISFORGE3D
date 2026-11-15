//! The geometric Jacobian and what its singular values say about a pose.
//!
//! Row order is three linear rows then three angular rows, matching
//! [`crate::orientation::PoseError::to_array`]. Columns follow the
//! configuration vector; a joint that is not on the chain from the root to the
//! frame contributes a zero column, which is the truth — moving it does not
//! move the frame.

use axisforge_domain::{DomainError, DomainResult, JointKind, Vec3};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::fk::{frame_pose, Pose};
use crate::model::KinematicModel;

/// A 6 x n geometric Jacobian for one tool frame at one configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Jacobian {
    matrix: DMatrix<f64>,
}

impl Jacobian {
    /// The full 6 x n matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// The number of columns: the model's degrees of freedom.
    pub fn columns(&self) -> usize {
        self.matrix.ncols()
    }

    /// A matrix holding only the given columns, in the given order.
    ///
    /// Inverse kinematics uses this with the manipulator columns, so the
    /// gripper's jaws are not among the unknowns.
    pub fn select_columns(&self, columns: &[usize]) -> DMatrix<f64> {
        DMatrix::from_fn(6, columns.len(), |row, col| {
            self.matrix[(row, columns[col])]
        })
    }

    /// The tool-frame twist produced by joint velocities `rates`.
    ///
    /// # Errors
    /// [`DomainError::JointCountMismatch`] for the wrong width.
    pub fn twist(&self, rates: &[f64]) -> DomainResult<[f64; 6]> {
        if rates.len() != self.columns() {
            return Err(DomainError::JointCountMismatch {
                expected: self.columns(),
                actual: rates.len(),
            });
        }
        let mut out = [0.0; 6];
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = rates
                .iter()
                .enumerate()
                .map(|(col, r)| self.matrix[(row, col)] * r)
                .sum();
        }
        Ok(out)
    }
}

/// Computes the geometric Jacobian of tool frame `frame` at `pose`.
///
/// `pose` must be the forward kinematics of the configuration of interest.
///
/// # Errors
/// [`DomainError::DanglingReference`] for an unknown frame index.
pub fn jacobian(model: &KinematicModel, pose: &Pose, frame: usize) -> DomainResult<Jacobian> {
    let node = model
        .frames()
        .get(frame)
        .ok_or_else(|| DomainError::dangling("tool frame", frame.to_string(), "jacobian"))?;
    let tool = frame_pose(model, pose, frame).translation;
    let mut matrix = DMatrix::zeros(6, model.dof());

    // Walk from the frame's link up to the root; only joints on this path can
    // move the frame.
    let mut link = node.link;
    while let Some(joint_index) = model.links()[link].parent_joint {
        let joint = &model.joints()[joint_index];
        if let Some(column) = joint.value_index {
            // The joint frame's world pose does not depend on the joint's own
            // value: a joint rotates or slides about its own axis.
            let joint_world = pose.link(joint.parent_link).compose(&joint.origin);
            let axis = joint_world.apply_direction(&joint.axis);
            let (linear, angular) = match joint.kind {
                JointKind::Revolute => (axis.cross(&tool.sub(&joint_world.translation)), axis),
                JointKind::Prismatic => (axis, Vec3::ZERO),
                JointKind::Fixed => (Vec3::ZERO, Vec3::ZERO),
            };
            for (row, value) in [
                linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
            ]
            .iter()
            .enumerate()
            {
                matrix[(row, column)] = *value;
            }
        }
        link = joint.parent_link;
    }
    Ok(Jacobian { matrix })
}

/// Which rows of the Jacobian a conditioning analysis looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskSpace {
    /// All six rows: position and orientation.
    Pose,
    /// The three linear rows: position only.
    Position,
}

/// What the Jacobian's singular values say about a configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditioningReport {
    /// Rows analysed.
    pub task_space: TaskSpace,
    /// Number of joint columns analysed.
    pub dof: usize,
    /// Singular values, largest first. There are `min(rows, dof)` of them.
    pub singular_values: Vec<f64>,
    /// Product of the singular values (Yoshikawa's measure for a full-rank task).
    pub manipulability: f64,
    /// Smallest singular value.
    pub min_singular_value: f64,
    /// Ratio of largest to smallest singular value; `None` when the smallest is zero.
    pub condition_number: Option<f64>,
    /// `true` when there are at least as many joints as task rows.
    ///
    /// A two-joint arm analysed in `Pose` space reports `false`: it has only
    /// two singular values, and their being healthy says nothing about the
    /// four task directions it cannot move in at all.
    pub task_capable: bool,
    /// `true` when the smallest singular value is below the threshold used.
    pub near_singular: bool,
}

/// Analyses the conditioning of `jacobian` over the given columns.
///
/// `singular_threshold` is compared against the smallest singular value, whose
/// units are metres per radian for linear rows, so 0.01 is a reasonable choice
/// for metre-scale arms.
pub fn conditioning(
    jacobian: &Jacobian,
    columns: &[usize],
    task_space: TaskSpace,
    singular_threshold: f64,
) -> ConditioningReport {
    let selected = jacobian.select_columns(columns);
    let rows = match task_space {
        TaskSpace::Pose => 6,
        TaskSpace::Position => 3,
    };
    let block = selected.rows(0, rows).into_owned();
    let mut singular_values: Vec<f64> = if columns.is_empty() {
        Vec::new()
    } else {
        block
            .svd(false, false)
            .singular_values
            .iter()
            .copied()
            .collect()
    };
    singular_values.sort_by(|a, b| b.total_cmp(a));
    let min = singular_values.last().copied().unwrap_or(0.0);
    let max = singular_values.first().copied().unwrap_or(0.0);
    ConditioningReport {
        task_space,
        dof: columns.len(),
        manipulability: if singular_values.is_empty() {
            0.0
        } else {
            singular_values.iter().product()
        },
        min_singular_value: min,
        condition_number: (min > 0.0).then(|| max / min),
        task_capable: columns.len() >= rows,
        near_singular: min < singular_threshold,
        singular_values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fk::{forward_kinematics, tool_pose};
    use axisforge_domain::templates;

    fn numeric_column(
        model: &KinematicModel,
        values: &[f64],
        frame: usize,
        column: usize,
    ) -> [f64; 6] {
        let h = 1.0e-6;
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[column] += h;
        minus[column] -= h;
        let a = tool_pose(model, &minus, frame).unwrap();
        let b = tool_pose(model, &plus, frame).unwrap();
        let error = crate::orientation::PoseError::between(&a, &b);
        let mut out = error.to_array();
        for value in &mut out {
            *value /= 2.0 * h;
        }
        out
    }

    #[test]
    fn the_analytic_jacobian_matches_finite_differences() {
        let robot = templates::six_axis_with_gripper().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let frame = model.default_frame().unwrap();
        let values = [0.3, -0.5, 0.8, 0.4, -0.7, 0.2, 0.01, 0.01];
        let pose = forward_kinematics(&model, &values).unwrap();
        let analytic = jacobian(&model, &pose, frame).unwrap();
        for column in 0..model.dof() {
            let numeric = numeric_column(&model, &values, frame, column);
            for (row, expected) in numeric.iter().enumerate() {
                let got = analytic.matrix()[(row, column)];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "row {row} column {column}: analytic {got} vs numeric {expected}"
                );
            }
        }
    }

    #[test]
    fn the_jaws_do_not_move_the_grasp_frame() {
        let robot = templates::six_axis_with_gripper().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let frame = model.default_frame().unwrap();
        let pose = forward_kinematics(&model, &[0.1; 8]).unwrap();
        let j = jacobian(&model, &pose, frame).unwrap();
        for column in model.end_effector_value_indices() {
            for row in 0..6 {
                assert!(
                    j.matrix()[(row, column)].abs() < 1e-15,
                    "jaw column {column} row {row}"
                );
            }
        }
    }

    #[test]
    fn a_stretched_planar_arm_is_singular_in_position() {
        let model = KinematicModel::from_robot(&templates::planar_2r().unwrap()).unwrap();
        let frame = model.default_frame().unwrap();
        let columns = model.manipulator_value_indices();

        let straight = forward_kinematics(&model, &[0.4, 0.0]).unwrap();
        let report = conditioning(
            &jacobian(&model, &straight, frame).unwrap(),
            &columns,
            TaskSpace::Position,
            0.01,
        );
        assert!(
            report.near_singular,
            "a straight arm cannot move radially: {report:?}"
        );
        assert_eq!(report.singular_values.len(), 2);

        let bent = forward_kinematics(&model, &[0.4, 1.2]).unwrap();
        let report = conditioning(
            &jacobian(&model, &bent, frame).unwrap(),
            &columns,
            TaskSpace::Position,
            0.01,
        );
        assert!(!report.near_singular, "{report:?}");
        assert!(report.manipulability > 0.01);
    }

    #[test]
    fn a_two_joint_arm_is_not_pose_capable() {
        let model = KinematicModel::from_robot(&templates::planar_2r().unwrap()).unwrap();
        let frame = model.default_frame().unwrap();
        let pose = forward_kinematics(&model, &[0.4, 1.2]).unwrap();
        let report = conditioning(
            &jacobian(&model, &pose, frame).unwrap(),
            &model.manipulator_value_indices(),
            TaskSpace::Pose,
            0.01,
        );
        assert!(!report.task_capable);
        assert_eq!(report.singular_values.len(), 2, "min(6, 2) singular values");
    }

    #[test]
    fn a_six_axis_arm_away_from_singularities_is_well_conditioned() {
        let model = KinematicModel::from_robot(&templates::six_axis().unwrap()).unwrap();
        let frame = model.default_frame().unwrap();
        let pose = forward_kinematics(&model, &[0.3, -0.6, 1.0, 0.5, 0.9, 0.2]).unwrap();
        let report = conditioning(
            &jacobian(&model, &pose, frame).unwrap(),
            &model.manipulator_value_indices(),
            TaskSpace::Pose,
            0.01,
        );
        assert!(report.task_capable);
        assert!(!report.near_singular, "{report:?}");
        assert!(report.condition_number.is_some());
    }

    #[test]
    fn the_wrist_singularity_is_detected() {
        // Axis 5 at zero aligns axes 4 and 6: two columns become parallel.
        let model = KinematicModel::from_robot(&templates::six_axis().unwrap()).unwrap();
        let frame = model.default_frame().unwrap();
        let pose = forward_kinematics(&model, &[0.3, -0.6, 1.0, 0.5, 0.0, 0.2]).unwrap();
        let report = conditioning(
            &jacobian(&model, &pose, frame).unwrap(),
            &model.manipulator_value_indices(),
            TaskSpace::Pose,
            0.01,
        );
        assert!(report.near_singular, "{report:?}");
    }
}
