//! Inverse kinematics by damped least squares, with seeded restarts.
//!
//! Each iteration linearizes the pose error with the Jacobian and solves
//! `dq = J^T (J J^T + lambda^2 I)^-1 e`. The damping keeps the step bounded
//! near singularities, where the undamped pseudoinverse would demand joint
//! motions without limit. Steps are projected onto the joint limits, and when
//! a start gets stuck the solver restarts from a random configuration drawn
//! from a seeded generator, so a result is reproducible from its request.
//!
//! Only the *manipulator* columns are unknowns. End-effector joints — gripper
//! jaws — keep the values they had in the seed configuration: a solver allowed
//! to move them would open the gripper to cover a shortfall in reach.
//!
//! The solver reports failure as data. A target out of reach yields
//! `converged: false` with the closest configuration found and its residuals,
//! which is what a user or a grasp planner needs to decide what to do next.

use axisforge_domain::{DomainError, DomainResult, Transform};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::cancel::{CancellationToken, Completion};
use crate::fk::{forward_kinematics_unchecked, frame_pose};
use crate::jacobian::jacobian;
use crate::model::KinematicModel;
use crate::orientation::PoseError;
use crate::rng::Rng;

/// Metres of position error one radian of orientation error is counted as,
/// when ranking candidate solutions.
pub const ORIENTATION_WEIGHT: f64 = 0.1;

/// Largest linear error fed to one iteration, metres.
///
/// The Jacobian is a linearization; an error far outside its validity makes
/// the step overshoot. Clamping the error is equivalent to a trust region.
const MAX_LINEAR_STEP: f64 = 0.1;

/// Largest angular error fed to one iteration, radians.
const MAX_ANGULAR_STEP: f64 = 0.4;

/// What the solver is asked to achieve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IkOptions {
    /// Acceptable position error, metres.
    pub position_tolerance: f64,
    /// Acceptable orientation error, radians.
    pub orientation_tolerance: f64,
    /// Iterations allowed per start.
    pub max_iterations: usize,
    /// Random restarts allowed after the seeded start.
    pub restarts: usize,
    /// Seed of the restart generator.
    pub rng_seed: u64,
    /// Damping factor `lambda` far from the target; it fades with the error.
    pub damping: f64,
    /// When `true`, only position is solved and orientation is left free.
    pub position_only: bool,
}

impl Default for IkOptions {
    fn default() -> Self {
        Self {
            position_tolerance: 1.0e-5,
            orientation_tolerance: 1.0e-4,
            max_iterations: 200,
            restarts: 12,
            rng_seed: 0x00C0_FFEE,
            damping: 0.05,
            position_only: false,
        }
    }
}

/// The outcome of an inverse-kinematics request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IkSolution {
    /// The best configuration found, full width, in configuration order.
    pub values: Vec<f64>,
    /// `true` when both residuals are within tolerance.
    pub converged: bool,
    /// Remaining position error, metres.
    pub position_error: f64,
    /// Remaining orientation error, radians. Reported even when position-only.
    pub orientation_error: f64,
    /// Total iterations across all starts.
    pub iterations: usize,
    /// Restarts used beyond the seeded start.
    pub restarts_used: usize,
    /// `true` when a caller cancelled the solve; the values are the best so far.
    pub cancelled: bool,
    /// Names of joints sitting on a limit in the returned configuration.
    pub joints_at_limit: Vec<String>,
}

struct Candidate {
    values: Vec<f64>,
    error: PoseError,
    score: f64,
}

fn score(error: &PoseError, position_only: bool) -> f64 {
    if position_only {
        error.position_norm()
    } else {
        error.weighted_norm(ORIENTATION_WEIGHT)
    }
}

fn accepted(error: &PoseError, options: &IkOptions) -> bool {
    error.position_norm() <= options.position_tolerance
        && (options.position_only || error.orientation_norm() <= options.orientation_tolerance)
}

/// Solves for a configuration placing tool frame `frame` at `target`.
///
/// `seed` is the starting configuration, full width. Its end-effector joint
/// values are carried into the answer unchanged.
///
/// # Errors
/// A malformed request: wrong seed width, non-finite values, an unknown frame,
/// or non-positive tolerances. An unreachable target is *not* an error; it is
/// reported as `converged: false`.
pub fn solve(
    model: &KinematicModel,
    frame: usize,
    target: &Transform,
    seed: &[f64],
    options: &IkOptions,
    cancel: &CancellationToken,
) -> DomainResult<IkSolution> {
    model.check_configuration(seed)?;
    if frame >= model.frames().len() {
        return Err(DomainError::dangling(
            "tool frame",
            frame.to_string(),
            "ik::solve",
        ));
    }
    let target = target.normalized("ik target")?;
    for (value, what) in [
        (options.position_tolerance, "positionTolerance"),
        (options.orientation_tolerance, "orientationTolerance"),
        (options.damping, "damping"),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(DomainError::NegativeValue {
                context: format!("ik options {what} must be positive"),
                value,
            });
        }
    }

    let columns = model.manipulator_value_indices();
    let limits = model.limits();
    let mut rng = Rng::new(options.rng_seed);
    let mut best: Option<Candidate> = None;
    let mut iterations = 0;
    let mut restarts_used = 0;
    let mut completion = Completion::Finished;

    'starts: for start in 0..=options.restarts {
        let mut values = model.clamped(seed);
        if start > 0 {
            restarts_used = start;
            for column in &columns {
                values[*column] = rng.next_in(limits[*column].lower, limits[*column].upper);
            }
        }

        for _ in 0..options.max_iterations.max(1) {
            if cancel.is_cancelled() {
                completion = Completion::Cancelled;
                break 'starts;
            }
            iterations += 1;
            let pose = forward_kinematics_unchecked(model, &values);
            let error = PoseError::between(&frame_pose(model, &pose, frame), &target);
            let candidate_score = score(&error, options.position_only);
            if best.as_ref().is_none_or(|b| candidate_score < b.score) {
                best = Some(Candidate {
                    values: values.clone(),
                    error,
                    score: candidate_score,
                });
            }
            if accepted(&error, options) {
                break 'starts;
            }
            let Some(step) = dls_step(model, &pose, frame, &columns, &error, options)? else {
                break;
            };
            let mut moved = 0.0_f64;
            for (slot, column) in columns.iter().enumerate() {
                let next = limits[*column].clamp(values[*column] + step[slot]);
                moved = moved.max((next - values[*column]).abs());
                values[*column] = next;
            }
            if moved < 1.0e-12 {
                // Pinned against limits or at a stationary point: this start
                // cannot improve, so spend the budget on another one.
                break;
            }
        }
    }

    let best = best.unwrap_or_else(|| {
        // Only reachable when cancelled before the first evaluation.
        let values = model.clamped(seed);
        let pose = forward_kinematics_unchecked(model, &values);
        let error = PoseError::between(&frame_pose(model, &pose, frame), &target);
        Candidate {
            score: score(&error, options.position_only),
            values,
            error,
        }
    });

    let joints_at_limit = columns
        .iter()
        .filter(|column| limits[**column].margin(best.values[**column]) <= 1.0e-9)
        .filter_map(|column| {
            model
                .joint_for_value(*column)
                .map(|joint| joint.name.clone())
        })
        .collect();

    Ok(IkSolution {
        converged: accepted(&best.error, options),
        position_error: best.error.position_norm(),
        orientation_error: best.error.orientation_norm(),
        values: best.values,
        iterations,
        restarts_used,
        cancelled: !completion.is_finished(),
        joints_at_limit,
    })
}

/// One damped-least-squares step over the given columns.
///
/// Returns `None` when the damped system cannot be factorized, which only
/// happens with non-finite input.
fn dls_step(
    model: &KinematicModel,
    pose: &crate::fk::Pose,
    frame: usize,
    columns: &[usize],
    error: &PoseError,
    options: &IkOptions,
) -> DomainResult<Option<Vec<f64>>> {
    let full = jacobian(model, pose, frame)?.select_columns(columns);
    let rows = if options.position_only { 3 } else { 6 };
    let j: DMatrix<f64> = full.rows(0, rows).into_owned();

    let mut e = error.to_array();
    let linear = error.position_norm();
    if linear > MAX_LINEAR_STEP {
        for value in &mut e[..3] {
            *value *= MAX_LINEAR_STEP / linear;
        }
    }
    let angular = error.orientation_norm();
    if angular > MAX_ANGULAR_STEP {
        for value in &mut e[3..] {
            *value *= MAX_ANGULAR_STEP / angular;
        }
    }
    let e = DVector::from_column_slice(&e[..rows]);

    // Damping proportional to the remaining error: full strength far from the
    // target, where the linearization is poor and singularities are crossed,
    // fading to nothing as the error does. Constant damping biases every step
    // short and turns the final approach into a slow linear crawl.
    let fade = (error.weighted_norm(ORIENTATION_WEIGHT) / MAX_LINEAR_STEP).min(1.0);
    let lambda_squared = options.damping * options.damping * fade + 1.0e-12;
    let mut system = &j * j.transpose();
    for i in 0..rows {
        system[(i, i)] += lambda_squared;
    }
    let Some(factor) = system.cholesky() else {
        return Ok(None);
    };
    let step = j.transpose() * factor.solve(&e);
    Ok(Some(step.iter().copied().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fk::tool_pose;
    use axisforge_domain::{templates, Vec3};

    fn gripper_model() -> KinematicModel {
        KinematicModel::from_robot(&templates::six_axis_with_gripper().unwrap()).unwrap()
    }

    #[test]
    fn a_reachable_pose_is_recovered() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let truth = [0.4, -0.5, 0.9, 0.3, 0.8, -0.2, 0.02, 0.02];
        let target = tool_pose(&model, &truth, frame).unwrap();
        let mut seed = vec![0.0; 8];
        seed[6] = 0.02;
        seed[7] = 0.02;
        let solution = solve(
            &model,
            frame,
            &target,
            &seed,
            &IkOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(solution.converged, "{solution:?}");
        let reached = tool_pose(&model, &solution.values, frame).unwrap();
        let error = PoseError::between(&reached, &target);
        assert!(error.position_norm() < 1e-5 && error.orientation_norm() < 1e-4);
    }

    #[test]
    fn the_jaws_keep_their_seed_values() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let target = tool_pose(&model, &[0.2, -0.4, 0.7, 0.1, 0.9, 0.3, 0.0, 0.0], frame).unwrap();
        let mut seed = vec![0.0; 8];
        seed[6] = 0.031;
        seed[7] = 0.017;
        let solution = solve(
            &model,
            frame,
            &target,
            &seed,
            &IkOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(solution.converged);
        for index in model.end_effector_value_indices() {
            assert!(
                (solution.values[index] - seed[index]).abs() < 1e-15,
                "jaw {index} moved from {} to {}",
                seed[index],
                solution.values[index]
            );
        }
    }

    #[test]
    fn an_unreachable_target_is_reported_not_raised() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let target = Transform::from_translation(Vec3::new(5.0, 0.0, 0.5));
        let options = IkOptions {
            restarts: 2,
            max_iterations: 60,
            ..IkOptions::default()
        };
        let solution = solve(
            &model,
            frame,
            &target,
            &[0.0; 8],
            &options,
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(!solution.converged);
        assert!(solution.position_error > 3.0, "{solution:?}");
        assert!(!solution.cancelled);
    }

    #[test]
    fn the_same_request_gives_the_same_answer() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let target = tool_pose(&model, &[1.4, 0.9, -1.2, 2.0, -1.1, 0.6, 0.0, 0.0], frame).unwrap();
        let run = || {
            solve(
                &model,
                frame,
                &target,
                &[0.0; 8],
                &IkOptions::default(),
                &CancellationToken::new(),
            )
            .unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn solutions_respect_joint_limits() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let target = tool_pose(&model, &[2.5, 1.5, -2.0, 1.0, 1.5, -2.0, 0.0, 0.0], frame).unwrap();
        let solution = solve(
            &model,
            frame,
            &target,
            &[0.0; 8],
            &IkOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(
            model
                .first_limit_violation(&solution.values, 1e-12)
                .is_none(),
            "{solution:?}"
        );
    }

    #[test]
    fn a_cancelled_solve_returns_promptly_and_says_so() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let target = Transform::from_translation(Vec3::new(0.3, 0.2, 0.6));
        let solution = solve(
            &model,
            frame,
            &target,
            &[0.0; 8],
            &IkOptions::default(),
            &CancellationToken::cancelled(),
        )
        .unwrap();
        assert!(solution.cancelled);
        assert_eq!(solution.iterations, 0);
        assert!(!solution.converged);
    }

    #[test]
    fn a_planar_arm_solves_position_only() {
        let model = KinematicModel::from_robot(&templates::planar_2r().unwrap()).unwrap();
        let frame = model.default_frame().unwrap();
        let target = tool_pose(&model, &[0.6, 1.1], frame).unwrap();
        let options = IkOptions {
            position_only: true,
            ..IkOptions::default()
        };
        let solution = solve(
            &model,
            frame,
            &target,
            &[0.1, 0.3],
            &options,
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(solution.converged, "{solution:?}");
    }

    #[test]
    fn a_malformed_request_is_an_error() {
        let model = gripper_model();
        let frame = model.default_frame().unwrap();
        let token = CancellationToken::new();
        let target = Transform::IDENTITY;
        assert!(solve(
            &model,
            frame,
            &target,
            &[0.0; 3],
            &IkOptions::default(),
            &token
        )
        .is_err());
        assert!(solve(
            &model,
            99,
            &target,
            &[0.0; 8],
            &IkOptions::default(),
            &token
        )
        .is_err());
        let bad = IkOptions {
            damping: 0.0,
            ..IkOptions::default()
        };
        assert!(solve(&model, frame, &target, &[0.0; 8], &bad, &token).is_err());
    }
}
