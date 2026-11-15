//! The pick planner: from "that part" to joint configurations that take it.
//!
//! For each synthesized candidate, best score first, the planner asks in order
//! of increasing cost:
//!
//! 1. Does the part fit the gripper's stroke, is the hold in force closure,
//!    and is the grip force enough?
//! 2. Can the arm reach the pre-grasp pose (backed off along the approach),
//!    then the grasp pose, then the lifted pose? The grasp and lift solves
//!    start from the previous answer with no random restarts, so a plan never
//!    hides a jump between arm branches.
//! 3. Is the motion clear of collisions: at the pre-grasp, along the approach
//!    with the jaws open, with the jaws closed on the part, and along the lift
//!    with the part carried?
//!
//! Every rejected candidate is returned with its reason, because "no grasp
//! found" is useless to someone who has to move the part or change the gripper.
//!
//! # Limits of what a plan asserts
//!
//! Collision along the approach and lift is checked at a fixed number of
//! joint-space samples, not continuously; something thinner than the gap
//! between samples can be missed. Contact physics is the two-point model in
//! [`crate::quality`]. A plan is an engineering proposal to be validated on
//! the real cell, not a guarantee; see [`axisforge_domain::SAFETY_NOTICE`].

use axisforge_collision::{Attachment, Scene, SceneOptions};
use axisforge_domain::{DomainError, DomainResult, Gripper, Project, Transform, Vec3, WorldBody};
use axisforge_kinematics::{forward_kinematics, ik, CancellationToken, IkOptions, KinematicModel};
use serde::{Deserialize, Serialize};

use crate::candidate::{synthesize, GraspCandidate, SynthesisOptions};
use crate::quality::{evaluate, GraspQuality};

/// How a pick is planned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PlannerOptions {
    /// Distance backed off along the approach for the pre-grasp, metres.
    pub standoff: f64,
    /// Height the part is raised after closing, metres.
    pub lift_height: f64,
    /// Gap left between each open jaw and the part, metres.
    pub jaw_clearance: f64,
    /// Collision margin: pairs closer than this count as contacts, metres.
    pub collision_margin: f64,
    /// Joint-space samples checked along the approach and along the lift.
    pub path_samples: usize,
    /// Stop after this many feasible plans.
    pub max_plans: usize,
    /// Sampling density for round parts.
    pub angular_samples: usize,
    /// Inverse-kinematics settings for the pre-grasp solve.
    pub ik: IkOptions,
}

impl Default for PlannerOptions {
    fn default() -> Self {
        Self {
            standoff: 0.08,
            lift_height: 0.10,
            jaw_clearance: 0.005,
            collision_margin: 0.001,
            path_samples: 6,
            max_plans: 3,
            angular_samples: 8,
            ik: IkOptions {
                restarts: 6,
                ..IkOptions::default()
            },
        }
    }
}

/// Why a candidate was not turned into a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    /// The part is wider than the jaws open, with clearance.
    TooWide,
    /// The contact normals fall outside the friction cone.
    NoForceClosure,
    /// The gripper cannot clamp hard enough to hold the part's weight.
    InsufficientGripForce,
    /// The arm cannot reach the pre-grasp pose.
    PregraspUnreachable,
    /// The arm cannot continue from the pre-grasp to the grasp pose.
    GraspUnreachable,
    /// The arm cannot continue from the grasp to the lifted pose.
    LiftUnreachable,
    /// Something is in the way at the pre-grasp.
    PregraspCollision,
    /// Something is in the way along the approach, jaws open.
    ApproachCollision,
    /// Something other than the fingers on the part touches at the grasp.
    GraspCollision,
    /// Something is in the way along the lift, part carried.
    LiftCollision,
    /// A distance measurement did not converge, so clearance is unproven.
    Inconclusive,
}

/// A candidate that was rejected, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rejection {
    /// The candidate's label.
    pub label: String,
    /// The first check it failed.
    pub reason: RejectionReason,
    /// Specifics: the numbers or the names of what collided.
    pub detail: String,
}

/// A feasible pick: four configurations, full width, in configuration order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraspPlan {
    /// The grasp being executed.
    pub candidate: GraspCandidate,
    /// Its quality metrics.
    pub quality: GraspQuality,
    /// World pose of the gripper's grasp frame at the grasp.
    pub grasp_pose: Transform,
    /// Backed off along the approach, jaws open.
    pub pregrasp: Vec<f64>,
    /// At the part, jaws still open.
    pub grasp_open: Vec<f64>,
    /// At the part, jaws closed onto it.
    pub grasp_closed: Vec<f64>,
    /// Raised, part carried.
    pub lift: Vec<f64>,
    /// Jaw-to-jaw opening used while approaching, metres.
    pub approach_opening: f64,
    /// Pose of the part in the palm link's frame while carried.
    pub part_in_palm: Transform,
    /// Smallest clearance seen over every checked sample, metres.
    pub min_clearance: f64,
}

/// Everything the planner found out about one part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanReport {
    /// Identifier of the part.
    pub body_id: String,
    /// Feasible plans, best score first.
    pub plans: Vec<GraspPlan>,
    /// Candidates that failed, in the order they were tried.
    pub rejections: Vec<Rejection>,
    /// Candidates synthesized for the part.
    pub candidates_total: usize,
    /// Candidates actually examined before stopping.
    pub candidates_examined: usize,
    /// `true` when a caller cancelled the search; the lists are partial.
    pub cancelled: bool,
}

struct Context<'a> {
    project: &'a Project,
    model: &'a KinematicModel,
    gripper: &'a Gripper,
    body: &'a WorldBody,
    frame: usize,
    palm: usize,
    jaws: Vec<usize>,
    options: &'a PlannerOptions,
    cancel: &'a CancellationToken,
}

type Fail = (RejectionReason, String);

impl Context<'_> {
    fn with_jaws(&self, values: &[f64], opening: f64) -> Vec<f64> {
        let mut out = values.to_vec();
        for (index, jaw) in self.jaws.iter().enumerate() {
            let limits = self.model.limits()[*jaw];
            // Symmetric jaws each take half; independent jaws split the opening.
            let share = match self.gripper.coupling {
                axisforge_domain::JawCoupling::Symmetric => {
                    self.gripper.jaw_value_for_opening(opening)
                }
                axisforge_domain::JawCoupling::Independent => {
                    #[allow(clippy::cast_precision_loss)]
                    let n = self.jaws.len() as f64;
                    let _ = index;
                    opening / n
                }
            };
            out[*jaw] = limits.clamp(share);
        }
        out
    }

    /// World pose of the grasp frame that realizes a candidate.
    fn grasp_pose(&self, candidate: &GraspCandidate) -> DomainResult<Transform> {
        let local = axisforge_domain::Quat::from_basis(
            &self.gripper.closing_axis,
            &self.gripper.approach_axis,
            &self.gripper.closing_axis.cross(&self.gripper.approach_axis),
        )?;
        Ok(Transform::new(
            candidate.center,
            candidate.orientation()?.mul(&local.conjugate()),
        ))
    }

    fn reach(
        &self,
        target: &Transform,
        seed: &[f64],
        continue_only: bool,
    ) -> DomainResult<Option<Vec<f64>>> {
        let options = if continue_only {
            IkOptions {
                restarts: 0,
                ..self.options.ik.clone()
            }
        } else {
            self.options.ik.clone()
        };
        let solution = ik::solve(self.model, self.frame, target, seed, &options, self.cancel)?;
        Ok(solution.converged.then_some(solution.values))
    }

    /// Checks one configuration; returns the closest distance or the failure.
    fn clear(
        &self,
        values: &[f64],
        scene_options: &SceneOptions,
        reason: RejectionReason,
    ) -> DomainResult<Result<f64, Fail>> {
        let pose = forward_kinematics(self.model, values)?;
        let report = Scene::build(self.project, self.model, &pose, scene_options)?
            .check(self.options.collision_margin);
        if let Some(contact) = report.contacts.first() {
            return Ok(Err((
                reason,
                format!(
                    "'{}' and '{}' are {:.1} mm apart",
                    contact.first_name,
                    contact.second_name,
                    contact.distance * 1000.0
                ),
            )));
        }
        if !report.conclusive {
            return Ok(Err((
                RejectionReason::Inconclusive,
                "a distance measurement hit its iteration limit".to_owned(),
            )));
        }
        Ok(Ok(report.closest.map_or(f64::INFINITY, |c| c.distance)))
    }

    /// Checks samples strictly after `from` up to and including `to`.
    fn clear_path(
        &self,
        from: &[f64],
        to: &[f64],
        scene_options: &SceneOptions,
        reason: RejectionReason,
    ) -> DomainResult<Result<f64, Fail>> {
        let samples = self.options.path_samples.max(1);
        let mut closest = f64::INFINITY;
        for step in 1..=samples {
            #[allow(clippy::cast_precision_loss)]
            let t = step as f64 / samples as f64;
            let values: Vec<f64> = from.iter().zip(to).map(|(a, b)| a + (b - a) * t).collect();
            match self.clear(&values, scene_options, reason)? {
                Ok(distance) => closest = closest.min(distance),
                Err(fail) => return Ok(Err(fail)),
            }
        }
        Ok(Ok(closest))
    }

    /// The checks that need no kinematics: stroke, force closure, grip force.
    fn gate(&self, candidate: &GraspCandidate, quality: &GraspQuality) -> Result<(), Fail> {
        let options = self.options;
        if candidate.width + 2.0 * options.jaw_clearance > self.gripper.max_opening {
            return Err((
                RejectionReason::TooWide,
                format!(
                    "needs {:.1} mm plus {:.1} mm clearance per jaw; the gripper opens {:.1} mm",
                    candidate.width * 1000.0,
                    options.jaw_clearance * 1000.0,
                    self.gripper.max_opening * 1000.0
                ),
            ));
        }
        if !quality.force_closure {
            return Err((
                RejectionReason::NoForceClosure,
                format!(
                    "contact normals are {:.1} deg off the closing line; the friction cone is {:.1} deg",
                    quality.antipodal_angle.to_degrees(),
                    quality.friction_cone_half_angle.to_degrees()
                ),
            ));
        }
        if !quality.grip_force_sufficient {
            return Err((
                RejectionReason::InsufficientGripForce,
                format!(
                    "needs {:.0} N; the gripper supplies {:.0} N",
                    quality.required_grip_force.unwrap_or(f64::INFINITY),
                    self.gripper.max_grip_force
                ),
            ));
        }
        Ok(())
    }

    fn attempt(
        &self,
        candidate: &GraspCandidate,
        seed: &[f64],
    ) -> DomainResult<Result<GraspPlan, Fail>> {
        let options = self.options;
        let quality = evaluate(
            candidate,
            self.gripper,
            &self.body.physical,
            &self.project.environment.ground_normal(),
        );

        if let Err(fail) = self.gate(candidate, &quality) {
            return Ok(Err(fail));
        }

        let grasp_pose = self.grasp_pose(candidate)?;
        let approach_opening = candidate.width + 2.0 * options.jaw_clearance;
        let open_seed = self.with_jaws(seed, approach_opening);
        let back = Vec3::ZERO.sub(&candidate.approach.scaled(options.standoff));
        let pregrasp_pose = Transform::new(grasp_pose.translation.add(&back), grasp_pose.rotation);
        let lift_pose = Transform::new(
            grasp_pose.translation.add(
                &self
                    .project
                    .environment
                    .ground_normal()
                    .scaled(options.lift_height),
            ),
            grasp_pose.rotation,
        );

        let Some(pregrasp) = self.reach(&pregrasp_pose, &open_seed, false)? else {
            return Ok(Err((
                RejectionReason::PregraspUnreachable,
                "no IK solution".to_owned(),
            )));
        };
        let Some(grasp_open) = self.reach(&grasp_pose, &pregrasp, true)? else {
            return Ok(Err((
                RejectionReason::GraspUnreachable,
                "no IK solution continuing from the pre-grasp".to_owned(),
            )));
        };
        let grasp_closed = self.with_jaws(&grasp_open, candidate.width);
        let Some(lift) = self.reach(&lift_pose, &grasp_closed, true)? else {
            return Ok(Err((
                RejectionReason::LiftUnreachable,
                "no IK solution continuing from the grasp".to_owned(),
            )));
        };

        let free = SceneOptions::default();
        let mut min_clearance = f64::INFINITY;
        macro_rules! require {
            ($check:expr) => {
                match $check? {
                    Ok(distance) => min_clearance = min_clearance.min(distance),
                    Err(fail) => return Ok(Err(fail)),
                }
            };
        }
        require!(self.clear(&pregrasp, &free, RejectionReason::PregraspCollision));
        require!(self.clear_path(
            &pregrasp,
            &grasp_open,
            &free,
            RejectionReason::ApproachCollision
        ));

        // Closed on the part, the fingers touch it; that contact is the grasp.
        let holding = SceneOptions {
            attachment: None,
            extra_ignored: self
                .gripper
                .finger_link_ids
                .iter()
                .map(|finger| (finger.clone(), self.body.id.clone()))
                .collect(),
        };
        require!(self.clear(&grasp_closed, &holding, RejectionReason::GraspCollision));

        let palm_world = forward_kinematics(self.model, &grasp_closed)?.link(self.palm);
        let part_in_palm = palm_world.inverse().compose(&self.body.pose);
        let carrying = SceneOptions {
            attachment: Some(Attachment {
                body_id: self.body.id.clone(),
                link_id: self.gripper.palm_link_id.clone(),
                relative_pose: part_in_palm,
            }),
            extra_ignored: Vec::new(),
        };
        // The first lift sample is past the start: at the start the part still
        // rests on whatever supported it, and that contact is not a fault.
        require!(self.clear_path(
            &grasp_closed,
            &lift,
            &carrying,
            RejectionReason::LiftCollision
        ));

        Ok(Ok(GraspPlan {
            candidate: candidate.clone(),
            quality,
            grasp_pose,
            pregrasp,
            grasp_open,
            grasp_closed,
            lift,
            approach_opening,
            part_in_palm,
            min_clearance,
        }))
    }
}

/// Plans picks of the world body `body_id` with the project's robot.
///
/// `seed` is the robot's current configuration, full width; the arm's part of
/// it is where the pre-grasp search starts.
///
/// # Errors
/// A request that cannot be planned at all: no gripper, an unknown or
/// non-graspable body, a body synthesis does not support, or a malformed seed.
/// Finding no feasible grasp is *not* an error; the report then has no plans
/// and a rejection for every candidate.
pub fn plan_pick(
    project: &Project,
    body_id: &str,
    seed: &[f64],
    options: &PlannerOptions,
    cancel: &CancellationToken,
) -> DomainResult<PlanReport> {
    let gripper = project.robot.require_gripper()?;
    let body = project
        .environment
        .body(body_id)
        .ok_or_else(|| DomainError::dangling("world body", body_id, "plan_pick"))?;
    if !body.is_graspable() {
        return Err(DomainError::unsupported(format!(
            "body '{}' is not marked manipulable, so it is not a pick target",
            body.name
        )));
    }
    let model = KinematicModel::from_robot(&project.robot)?;
    model.check_configuration(seed)?;
    let frame = model.frame_index(&gripper.grasp_frame_id).ok_or_else(|| {
        DomainError::dangling("tool frame", gripper.grasp_frame_id.clone(), "plan_pick")
    })?;
    let palm = model
        .link_index(&gripper.palm_link_id)
        .ok_or_else(|| DomainError::dangling("link", gripper.palm_link_id.clone(), "plan_pick"))?;
    let jaws = gripper
        .jaw_joint_ids
        .iter()
        .filter_map(|id| {
            model
                .joint_index(id)
                .and_then(|j| model.joints()[j].value_index)
        })
        .collect();

    let synthesis = SynthesisOptions {
        angular_samples: options.angular_samples,
        finger_depth: gripper.finger_depth,
        ..SynthesisOptions::default()
    };
    let up = project.environment.ground_normal();
    let mut ranked: Vec<(f64, GraspCandidate)> = synthesize(body, &synthesis)?
        .into_iter()
        .map(|c| (evaluate(&c, gripper, &body.physical, &up).score, c))
        .collect();
    // A stable sort keeps synthesis order among equal scores, so the same
    // request always examines candidates in the same order.
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let context = Context {
        project,
        model: &model,
        gripper,
        body,
        frame,
        palm,
        jaws,
        options,
        cancel,
    };
    let mut report = PlanReport {
        body_id: body.id.clone(),
        plans: Vec::new(),
        rejections: Vec::new(),
        candidates_total: ranked.len(),
        candidates_examined: 0,
        cancelled: false,
    };
    for (_, candidate) in &ranked {
        if cancel.is_cancelled() {
            report.cancelled = true;
            break;
        }
        if report.plans.len() >= options.max_plans.max(1) {
            break;
        }
        report.candidates_examined += 1;
        match context.attempt(candidate, seed)? {
            Ok(plan) => report.plans.push(plan),
            Err((reason, detail)) => {
                // A cancelled IK solve looks like "unreachable"; do not record
                // a verdict the search did not actually reach.
                if cancel.is_cancelled() {
                    report.cancelled = true;
                    report.candidates_examined -= 1;
                    break;
                }
                report.rejections.push(Rejection {
                    label: candidate.label.clone(),
                    reason,
                    detail,
                });
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::{templates, Shape};
    use axisforge_kinematics::{frame_pose, PoseError};

    fn cell() -> (Project, Vec<f64>) {
        let project = templates::pick_and_place_cell().unwrap();
        // Elbow up, gripper pointing down above the table, clear of everything.
        let seed = vec![0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0];
        (project, seed)
    }

    fn plan(project: &Project, name: &str, seed: &[f64]) -> PlanReport {
        let id = project.environment.body_by_name(name).unwrap().id.clone();
        plan_pick(
            project,
            &id,
            seed,
            &PlannerOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap()
    }

    #[test]
    fn the_cube_in_the_demo_cell_can_be_picked() {
        let (project, seed) = cell();
        let report = plan(&project, "cube", &seed);
        assert!(
            !report.plans.is_empty(),
            "rejections: {:#?}",
            report.rejections
        );
        assert!(!report.cancelled);
    }

    #[test]
    fn the_dowel_in_the_demo_cell_can_be_picked() {
        let (project, seed) = cell();
        let report = plan(&project, "dowel", &seed);
        assert!(
            !report.plans.is_empty(),
            "rejections: {:#?}",
            report.rejections
        );
    }

    #[test]
    fn a_plan_does_what_it_says() {
        let (project, seed) = cell();
        let report = plan(&project, "cube", &seed);
        let best = &report.plans[0];
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let gripper = project.robot.require_gripper().unwrap();
        let frame = model.frame_index(&gripper.grasp_frame_id).unwrap();
        let reached = |values: &[f64]| {
            frame_pose(&model, &forward_kinematics(&model, values).unwrap(), frame)
        };

        // The grasp frame is at the grasp pose, open and closed.
        for values in [&best.grasp_open, &best.grasp_closed] {
            let error = PoseError::between(&reached(values), &best.grasp_pose);
            assert!(error.position_norm() < 1e-4 && error.orientation_norm() < 1e-3);
        }
        // The pre-grasp is backed off along the approach by the standoff.
        let offset = best
            .grasp_pose
            .translation
            .sub(&reached(&best.pregrasp).translation);
        assert!(
            offset.distance(&best.candidate.approach.scaled(0.08)) < 1e-4,
            "{offset:?}"
        );
        // The lift is straight up by the lift height.
        let rise = reached(&best.lift)
            .translation
            .sub(&best.grasp_pose.translation);
        assert!(rise.distance(&Vec3::new(0.0, 0.0, 0.10)) < 1e-4, "{rise:?}");

        // The gripper's axes line up with the candidate's.
        let pose = reached(&best.grasp_open);
        assert!(
            pose.apply_direction(&gripper.approach_axis)
                .distance(&best.candidate.approach)
                < 1e-3
        );
        assert!(
            pose.apply_direction(&gripper.closing_axis)
                .distance(&best.candidate.closing)
                < 1e-3
        );

        // Closed, the jaws are exactly the part's width apart.
        let jaw = model.end_effector_value_indices()[0];
        assert!(
            (gripper.opening_for_jaw_value(best.grasp_closed[jaw]) - best.candidate.width).abs()
                < 1e-12
        );
        assert!(
            (gripper.opening_for_jaw_value(best.pregrasp[jaw]) - best.approach_opening).abs()
                < 1e-12
        );

        // Every configuration respects the joint limits.
        for values in [
            &best.pregrasp,
            &best.grasp_open,
            &best.grasp_closed,
            &best.lift,
        ] {
            assert!(model.first_limit_violation(values, 1e-12).is_none());
        }
        assert!(best.min_clearance > 0.001);
        assert!(best.quality.force_closure && best.quality.grip_force_sufficient);
    }

    #[test]
    fn plans_come_back_best_score_first() {
        let (project, seed) = cell();
        let report = plan(&project, "cube", &seed);
        for pair in report.plans.windows(2) {
            assert!(pair[0].quality.score >= pair[1].quality.score);
        }
    }

    #[test]
    fn a_part_wider_than_the_stroke_is_rejected_as_too_wide() {
        let (mut project, seed) = cell();
        let cube = project
            .environment
            .bodies
            .iter_mut()
            .find(|b| b.name == "cube")
            .unwrap();
        cube.geometries[0].shape = Shape::Box {
            size: Vec3::new(0.12, 0.12, 0.12),
        };
        cube.pose.translation.z = 0.42 + 0.06;
        let report = plan(&project, "cube", &seed);
        assert!(report.plans.is_empty());
        assert_eq!(report.rejections.len(), report.candidates_total);
        assert!(report
            .rejections
            .iter()
            .all(|r| r.reason == RejectionReason::TooWide));
    }

    #[test]
    fn a_part_too_heavy_to_hold_is_rejected_for_grip_force() {
        let (mut project, seed) = cell();
        let cube = project
            .environment
            .bodies
            .iter_mut()
            .find(|b| b.name == "cube")
            .unwrap();
        cube.physical.mass = 40.0;
        cube.physical.friction_coefficient = 0.2;
        let report = plan(&project, "cube", &seed);
        // Holds that close vertically need no friction, but they would have a
        // jaw under the part, in the table.
        assert!(
            report.plans.is_empty(),
            "{:#?}",
            report.plans.first().map(|p| &p.candidate)
        );
        assert!(report
            .rejections
            .iter()
            .any(|r| r.reason == RejectionReason::InsufficientGripForce));
    }

    #[test]
    fn a_part_out_of_reach_is_rejected_as_unreachable() {
        let (mut project, seed) = cell();
        let cube = project
            .environment
            .bodies
            .iter_mut()
            .find(|b| b.name == "cube")
            .unwrap();
        cube.pose.translation = Vec3::new(3.0, 0.0, 0.5);
        let options = PlannerOptions {
            ik: IkOptions {
                restarts: 1,
                max_iterations: 60,
                ..IkOptions::default()
            },
            ..PlannerOptions::default()
        };
        let id = cube.id.clone();
        let report = plan_pick(&project, &id, &seed, &options, &CancellationToken::new()).unwrap();
        assert!(report.plans.is_empty());
        assert!(report
            .rejections
            .iter()
            .all(|r| r.reason == RejectionReason::PregraspUnreachable));
    }

    #[test]
    fn grasps_from_below_the_table_are_rejected_not_planned() {
        let (project, seed) = cell();
        let options = PlannerOptions {
            max_plans: 1000,
            ..PlannerOptions::default()
        };
        let id = project.environment.body_by_name("cube").unwrap().id.clone();
        let report = plan_pick(&project, &id, &seed, &options, &CancellationToken::new()).unwrap();
        assert_eq!(report.candidates_examined, report.candidates_total);
        assert_eq!(
            report.plans.len() + report.rejections.len(),
            report.candidates_total
        );
        for plan in &report.plans {
            assert!(
                plan.candidate.approach.z < 0.5,
                "approaches upward through the table: {}",
                plan.candidate.label
            );
        }
        assert!(!report.rejections.is_empty());
    }

    #[test]
    fn planning_is_deterministic() {
        let (project, seed) = cell();
        assert_eq!(
            plan(&project, "dowel", &seed),
            plan(&project, "dowel", &seed)
        );
    }

    #[test]
    fn a_cancelled_search_says_so_and_invents_no_verdicts() {
        let (project, seed) = cell();
        let id = project.environment.body_by_name("cube").unwrap().id.clone();
        let report = plan_pick(
            &project,
            &id,
            &seed,
            &PlannerOptions::default(),
            &CancellationToken::cancelled(),
        )
        .unwrap();
        assert!(report.cancelled);
        assert!(report.plans.is_empty() && report.rejections.is_empty());
        assert_eq!(report.candidates_examined, 0);
    }

    #[test]
    fn requests_that_cannot_be_planned_are_errors() {
        let (project, seed) = cell();
        let token = CancellationToken::new();
        let options = PlannerOptions::default();
        assert!(plan_pick(&project, "ghost", &seed, &options, &token).is_err());
        let table = project
            .environment
            .body_by_name("table")
            .unwrap()
            .id
            .clone();
        let error = plan_pick(&project, &table, &seed, &options, &token).expect_err("an obstacle");
        assert!(
            error.to_string().contains("not marked manipulable"),
            "{error}"
        );
        let cube = project.environment.body_by_name("cube").unwrap().id.clone();
        assert!(plan_pick(&project, &cube, &[0.0; 3], &options, &token).is_err());

        let mut bare = project.clone();
        bare.robot = templates::six_axis().unwrap();
        bare.environment.ignored_pairs = axisforge_domain::IgnoredPairs::new();
        let error = plan_pick(&bare, &cube, &[0.0; 6], &options, &token).expect_err("no gripper");
        assert!(matches!(error, DomainError::NoGripper { .. }), "{error}");
    }
}
