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

use std::cell::RefCell;
use std::collections::BTreeMap;

use axisforge_collision::{Attachment, Scene, SceneOptions};
use axisforge_domain::{
    DomainError, DomainResult, Gripper, Project, Shape, Transform, Vec3, WorldBody,
};
use axisforge_kinematics::{forward_kinematics, ik, CancellationToken, IkOptions, KinematicModel};
use serde::{Deserialize, Serialize};

use crate::candidate::{GraspCandidate, SynthesisOptions};
use crate::hand::{synthesize_hand, HandSynthesis, HAND_GENERATOR};
use crate::quality::{
    evaluate, grip_load, preliminary_score, primitive_frame, GraspQuality, QualityInput,
    QualityMeasure,
};
use crate::registry::Registry;
use crate::surface::PartSurface;
use crate::wrench::WrenchOptions;

/// The generator name that picks one by the part's shape.
pub const AUTO_GENERATOR: &str = "auto";

/// How a pick is planned.
///
/// The first group of fields follows the vocabulary of established grasp
/// planners (the `GraspStudio` library of Simox in particular), so a setup described in those
/// terms carries over: how many grasps to find, the quality they must reach,
/// whether force closure is required, a time budget, and the friction-cone
/// model. The semantics are this engine's own and are documented per field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PlannerOptions {
    /// Stop after this many feasible grasps.
    #[serde(alias = "maxPlans")]
    pub nr_grasps: usize,
    /// Smallest acceptable value of the quality measure. For `wrench_space`
    /// this is epsilon, in units of the contact force.
    pub min_quality: f64,
    /// Require force closure as judged by the quality measure.
    pub force_closure: bool,
    /// Give up after this many seconds. Needs a clock: refused on builds
    /// without one (WebAssembly), where `maxCandidates` bounds the work instead.
    pub timeout_seconds: Option<f64>,
    /// Examine at most this many candidates.
    pub max_candidates: Option<usize>,
    /// Name of the quality measure, from the extension registry.
    pub quality_measure: String,
    /// Name of the candidate generator, from the extension registry, or
    /// `auto`: `primitive_antipodal` for a part that is a single primitive,
    /// `surface_normal` for anything else, and for a hand
    /// [`crate::hand::HAND_GENERATOR`], the only one a hand can use. The
    /// report names the one used.
    pub candidate_generator: String,
    /// Contact model: unit force, friction coefficient, friction-cone samples.
    pub quality: WrenchOptions,
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
    /// Sampling density for round parts, and approach directions tried
    /// around each closing line on a sampled surface.
    pub angular_samples: usize,
    /// Points sampled on the surface of a part that is not a single primitive.
    pub surface_samples: usize,
    /// How far below a pad's stop the part still counts as touching it,
    /// metres: a stand-in for the pad's compliance.
    pub contact_tolerance: f64,
    /// Inverse-kinematics settings for the pre-grasp solve.
    pub ik: IkOptions,
    /// For a hand, the preshape to approach in; the gripper's first when
    /// unset. Ignored by a parallel-jaw gripper.
    pub preshape: Option<String>,
}

impl Default for PlannerOptions {
    fn default() -> Self {
        Self {
            nr_grasps: 3,
            min_quality: 0.0,
            force_closure: true,
            timeout_seconds: None,
            max_candidates: None,
            quality_measure: "wrench_space".to_owned(),
            candidate_generator: AUTO_GENERATOR.to_owned(),
            quality: WrenchOptions::default(),
            standoff: 0.08,
            lift_height: 0.10,
            jaw_clearance: 0.005,
            collision_margin: 0.001,
            path_samples: 6,
            angular_samples: 8,
            surface_samples: 48,
            contact_tolerance: 0.000_3,
            ik: IkOptions {
                restarts: 6,
                ..IkOptions::default()
            },
            preshape: None,
        }
    }
}

impl PlannerOptions {
    /// Checks the numeric settings.
    ///
    /// # Errors
    /// When a value is non-finite, negative where that is meaningless, or a
    /// time budget is requested on a build with no clock.
    pub fn validate(&self) -> DomainResult<()> {
        let mut checks = vec![
            (self.min_quality, "minQuality"),
            (self.standoff, "standoff"),
            (self.lift_height, "liftHeight"),
            (self.jaw_clearance, "jawClearance"),
            (self.collision_margin, "collisionMargin"),
            (self.contact_tolerance, "contactTolerance"),
            (self.quality.unit_force, "quality.unitForce"),
            (self.quality.torsional_friction, "quality.torsionalFriction"),
        ];
        if let Some(mu) = self.quality.friction_coefficient {
            checks.push((mu, "quality.frictionCoefficient"));
        }
        if let Some(timeout) = self.timeout_seconds {
            checks.push((timeout, "timeoutSeconds"));
        }
        for (value, what) in checks {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("planner options {what}"),
                });
            }
            if value < 0.0 {
                return Err(DomainError::NegativeValue {
                    context: format!("planner options {what}"),
                    value,
                });
            }
        }
        if self.quality.unit_force <= 0.0 {
            return Err(DomainError::NegativeValue {
                context: "planner options quality.unitForce must be positive".to_owned(),
                value: self.quality.unit_force,
            });
        }
        #[cfg(target_arch = "wasm32")]
        if self.timeout_seconds.is_some() {
            return Err(DomainError::unsupported(
                "timeoutSeconds needs a clock, which this build does not have; \
                 use maxCandidates to bound the search instead",
            ));
        }
        Ok(())
    }
}

/// Why a candidate was not turned into a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    /// The part is wider than the jaws open, with clearance.
    TooWide,
    /// The quality measure does not find the grasp in force closure.
    NoForceClosure,
    /// The quality measure's value is below the requested minimum.
    BelowMinQuality,
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
    /// Midpoint between the contacts of the rejected grasp, world coordinates.
    pub center: Vec3,
    /// Unit direction the gripper would have approached along.
    pub approach: Vec3,
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
    /// `true` when the time budget ran out; the lists are partial.
    pub timed_out: bool,
    /// Name of the quality measure used.
    pub quality_measure: String,
    /// Name of the candidate generator used.
    pub candidate_generator: String,
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
    measure: &'a dyn QualityMeasure,
    /// The part's centre of mass and bounding radius, for the quality measure.
    center_of_mass: Vec3,
    torque_scale: f64,
    /// Quality by contact set: the two jaw assignments of a grasp share their
    /// contacts, and the measure need not be run twice for them.
    measured: RefCell<BTreeMap<String, GraspQuality>>,
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

    /// `values` with the fingers as a hand candidate holds them: approaching
    /// when `closed` is false, on the part when it is true.
    fn with_fingers(
        &self,
        values: &[f64],
        hand: &crate::hand::HandGrasp,
        closed: bool,
    ) -> Vec<f64> {
        let mut out = values.to_vec();
        let posture = if closed { &hand.closed } else { &hand.open };
        for (jaw, value) in self.jaws.iter().zip(posture) {
            out[*jaw] = *value;
        }
        out
    }

    /// The opening to approach with and `seed` with the fingers posed for it.
    fn approaching(&self, candidate: &GraspCandidate, seed: &[f64]) -> (f64, Vec<f64>) {
        if let Some(hand) = &candidate.hand {
            (
                self.gripper.max_opening,
                self.with_fingers(seed, hand, false),
            )
        } else {
            let opening = candidate.width + 2.0 * self.options.jaw_clearance;
            (opening, self.with_jaws(seed, opening))
        }
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

    fn quality_input<'c>(&'c self, candidate: &'c GraspCandidate) -> QualityInput<'c> {
        QualityInput {
            candidate,
            gripper: self.gripper,
            body: self.body,
            up: self.project.environment.ground_normal(),
            wrench: &self.options.quality,
            center_of_mass: self.center_of_mass,
            torque_scale: self.torque_scale,
        }
    }

    /// The checks that need neither kinematics nor the quality measure:
    /// stroke and grip force.
    fn gate(&self, candidate: &GraspCandidate) -> Result<(), Fail> {
        let options = self.options;
        // A hand's fingers were closed on the part when the candidate was
        // made; there is no stroke to compare with.
        if candidate.hand.is_none()
            && candidate.width + 2.0 * options.jaw_clearance > self.gripper.max_opening
        {
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
        let (required, sufficient) = grip_load(&self.quality_input(candidate));
        if !sufficient {
            return Err((
                RejectionReason::InsufficientGripForce,
                format!(
                    "needs {:.0} N; the gripper supplies {:.0} N",
                    required.unwrap_or(f64::INFINITY),
                    self.gripper.max_grip_force
                ),
            ));
        }
        Ok(())
    }

    /// Runs the quality measure, once per distinct contact set, and applies
    /// the force-closure and minimum-quality requirements.
    ///
    /// This comes after reachability and collision checks because it is the
    /// most expensive step and most candidates never get this far.
    fn judge(&self, candidate: &GraspCandidate) -> Result<GraspQuality, Fail> {
        let key = contact_key(candidate);
        let cached = key
            .as_ref()
            .and_then(|k| self.measured.borrow().get(k).cloned());
        let quality = cached.unwrap_or_else(|| {
            let quality = evaluate(self.measure, &self.quality_input(candidate));
            if let Some(key) = key {
                self.measured.borrow_mut().insert(key, quality.clone());
            }
            quality
        });
        if self.options.force_closure && !quality.force_closure {
            return Err((
                RejectionReason::NoForceClosure,
                format!(
                    "'{}' finds no force closure with {} contact points and friction {:.2}",
                    quality.measure, quality.contact_count, quality.friction_coefficient
                ),
            ));
        }
        if quality.quality < self.options.min_quality {
            return Err((
                RejectionReason::BelowMinQuality,
                format!(
                    "'{}' gives {:.4}; the minimum is {:.4}",
                    quality.measure, quality.quality, self.options.min_quality
                ),
            ));
        }
        Ok(quality)
    }

    fn attempt(
        &self,
        candidate: &GraspCandidate,
        seed: &[f64],
    ) -> DomainResult<Result<GraspPlan, Fail>> {
        let options = self.options;
        if let Err(fail) = self.gate(candidate) {
            return Ok(Err(fail));
        }

        let grasp_pose = self.grasp_pose(candidate)?;
        let (approach_opening, open_seed) = self.approaching(candidate, seed);
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
        let grasp_closed = match &candidate.hand {
            Some(hand) => self.with_fingers(&grasp_open, hand, true),
            None => self.with_jaws(&grasp_open, candidate.width),
        };

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

        let quality = match self.judge(candidate) {
            Ok(quality) => quality,
            Err(fail) => return Ok(Err(fail)),
        };

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

/// A key identifying a candidate's contact set, or `None` when it has none.
fn contact_key(candidate: &GraspCandidate) -> Option<String> {
    if candidate.contacts.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = candidate
        .contacts
        .iter()
        .map(|c| {
            // Whole nanometres, so that 0.0 and -0.0 give the same key.
            #[allow(clippy::cast_possible_truncation)]
            let q = |v: f64, unit: f64| (v / unit).round() as i64;
            format!(
                "{},{},{},{},{},{}",
                q(c.point.x, 1.0e-9),
                q(c.point.y, 1.0e-9),
                q(c.point.z, 1.0e-9),
                q(c.normal.x, 1.0e-6),
                q(c.normal.y, 1.0e-6),
                q(c.normal.z, 1.0e-6)
            )
        })
        .collect();
    parts.sort_unstable();
    Some(parts.join(";"))
}

/// Examines candidates in order until enough plans are found or a limit is hit.
fn search(
    context: &Context<'_>,
    ranked: &[(f64, GraspCandidate)],
    seed: &[f64],
    report: &mut PlanReport,
) -> DomainResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let deadline = context
        .options
        .timeout_seconds
        .map(|seconds| std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds));
    for (_, candidate) in ranked {
        if context.cancel.is_cancelled() {
            report.cancelled = true;
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            report.timed_out = true;
            break;
        }
        if report.plans.len() >= context.options.nr_grasps.max(1)
            || context
                .options
                .max_candidates
                .is_some_and(|limit| report.candidates_examined >= limit)
        {
            break;
        }
        report.candidates_examined += 1;
        match context.attempt(candidate, seed)? {
            Ok(plan) => report.plans.push(plan),
            Err((reason, detail)) => {
                // A cancelled IK solve looks like "unreachable"; do not record
                // a verdict the search did not actually reach.
                if context.cancel.is_cancelled() {
                    report.cancelled = true;
                    report.candidates_examined -= 1;
                    break;
                }
                report.rejections.push(Rejection {
                    label: candidate.label.clone(),
                    reason,
                    detail,
                    center: candidate.center,
                    approach: candidate.approach,
                });
            }
        }
    }
    Ok(())
}

/// What a hand synthesis takes from the planner's options.
fn hand_synthesis(options: &PlannerOptions, seed: u64) -> HandSynthesis {
    HandSynthesis {
        surface_samples: options.surface_samples,
        // Each turn is tried straight and tilted.
        turns: options.angular_samples.div_ceil(2),
        contact_tolerance: options.contact_tolerance,
        seed,
        preshape: options.preshape.clone(),
    }
}

/// The generator for this gripper and part: `None` for a hand, which uses
/// [`synthesize_hand`].
fn choose_generator<'r>(
    registry: &'r Registry,
    gripper: &Gripper,
    measure: &dyn QualityMeasure,
    options: &PlannerOptions,
    primitive: bool,
) -> DomainResult<Option<&'r dyn crate::registry::CandidateGenerator>> {
    let hand = gripper.is_hand();
    if hand
        && options.candidate_generator != AUTO_GENERATOR
        && options.candidate_generator != HAND_GENERATOR
    {
        return Err(DomainError::unsupported(format!(
            "'{}' proposes grasps for two parallel jaws; gripper '{}' is a hand, which uses \
             '{HAND_GENERATOR}' (or 'auto')",
            options.candidate_generator, gripper.name
        )));
    }
    if !hand && options.candidate_generator == HAND_GENERATOR {
        return Err(DomainError::unsupported(format!(
            "'{HAND_GENERATOR}' closes the fingers of a hand; gripper '{}' has parallel jaws",
            gripper.name
        )));
    }
    if hand && measure.id() == "antipodal" {
        return Err(DomainError::unsupported(
            "the antipodal measure judges two opposed jaws; a hand's contacts are judged with \
             'wrench_space'"
                .to_owned(),
        ));
    }
    Ok(if hand {
        None
    } else {
        Some(
            registry.generator(if options.candidate_generator == AUTO_GENERATOR {
                if primitive {
                    "primitive_antipodal"
                } else {
                    "surface_normal"
                }
            } else {
                &options.candidate_generator
            })?,
        )
    })
}

/// Plans picks of the world body `body_id` with the built-in extensions.
///
/// See [`plan_pick_with`].
///
/// # Errors
/// As [`plan_pick_with`].
pub fn plan_pick(
    project: &Project,
    body_id: &str,
    seed: &[f64],
    options: &PlannerOptions,
    cancel: &CancellationToken,
) -> DomainResult<PlanReport> {
    plan_pick_with(
        &Registry::standard(),
        project,
        body_id,
        seed,
        options,
        cancel,
    )
}

/// Plans picks of the world body `body_id` with the project's robot, using
/// the quality measure and candidate generator the options name in `registry`.
///
/// `seed` is the robot's current configuration, full width; the arm's part of
/// it is where the pre-grasp search starts.
///
/// # Errors
/// A request that cannot be planned at all: no gripper, an unknown or
/// non-graspable body, a body the generator does not support, an unknown
/// extension name, invalid options, or a malformed seed. Finding no feasible
/// grasp is *not* an error; the report then has no plans and a rejection for
/// every candidate examined.
pub fn plan_pick_with(
    registry: &Registry,
    project: &Project,
    body_id: &str,
    seed: &[f64],
    options: &PlannerOptions,
    cancel: &CancellationToken,
) -> DomainResult<PlanReport> {
    options.validate()?;
    let measure = registry.measure(&options.quality_measure)?;
    let gripper = project.robot.require_gripper()?;
    let body = project
        .environment
        .body(body_id)
        .ok_or_else(|| DomainError::dangling("world body", body_id, "plan_pick"))?;
    // A single primitive has exact grasps and an exact centre; anything else
    // is handled through its surface.
    let primitive = body
        .single_geometry()
        .is_some_and(|g| !matches!(g.shape, Shape::Mesh { .. }));
    let generator = choose_generator(registry, gripper, measure, options, primitive)?;
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
        finger_width: gripper.finger_width,
        surface_samples: options.surface_samples,
        contact_tolerance: options.contact_tolerance,
        ..SynthesisOptions::default()
    };
    let (center_of_mass, torque_scale) = if primitive {
        primitive_frame(body)
    } else {
        let surface = PartSurface::of(body, &project.resources)?;
        (surface.center_of_mass, surface.radius)
    };
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
        measure,
        center_of_mass,
        torque_scale,
        measured: RefCell::new(BTreeMap::new()),
    };
    // Candidates the gripper cannot make at all (wider than its stroke, or
    // needing more force than it has) go after every other: they are
    // rejected cheaply, but each still counts against `maxCandidates`, and a
    // sampled surface yields many of them.
    let candidates = match generator {
        Some(generator) => generator.generate(body, &project.resources, &synthesis)?,
        None => synthesize_hand(
            project,
            &model,
            body,
            center_of_mass,
            &hand_synthesis(options, synthesis.seed),
        )?,
    };
    let mut ranked: Vec<(f64, GraspCandidate)> = candidates
        .into_iter()
        .map(|c| {
            let score = preliminary_score(&context.quality_input(&c));
            let makeable = context.gate(&c).is_ok();
            (if makeable { score + 1.0 } else { score }, c)
        })
        .collect();
    // A stable sort keeps the generator's order among equal scores, so the
    // same request always examines candidates in the same order.
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut report = PlanReport {
        body_id: body.id.clone(),
        plans: Vec::new(),
        rejections: Vec::new(),
        candidates_total: ranked.len(),
        candidates_examined: 0,
        cancelled: false,
        timed_out: false,
        quality_measure: measure.id().to_owned(),
        candidate_generator: generator.map_or(HAND_GENERATOR, |g| g.id()).to_owned(),
    };
    search(&context, &ranked, seed, &mut report)?;
    // Candidates were examined in order of a cheap estimate; the plans are
    // returned in order of their measured score.
    report
        .plans
        .sort_by(|a, b| b.quality.score.total_cmp(&a.quality.score));
    Ok(report)
}

/// A generator-order candidate and, only when requested, its native evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchCandidate {
    /// Stable index in deterministic generator order for the same project and options.
    pub id: usize,
    /// Candidate geometry in world coordinates.
    pub candidate: GraspCandidate,
    /// Full world grasp-frame pose.
    pub grasp_pose: Transform,
    /// Native plan if all checks pass; omitted when not evaluated or rejected.
    pub plan: Option<GraspPlan>,
    /// First native rejection, if evaluated and rejected.
    pub rejection: Option<Rejection>,
    /// Static quality evaluated independently, if requested.
    pub quality: Option<GraspQuality>,
    /// Seconds computing standalone static quality.
    pub quality_elapsed_s: Option<f64>,
    /// Seconds in the native attempt, including any repeated quality work.
    pub attempt_elapsed_s: Option<f64>,
}

/// Generate without score sorting; optionally evaluate one generator-order index.
/// No quality or IK is run when `evaluate_index` is `None`.
pub fn research_candidates(
    project: &Project,
    body_id: &str,
    seed: &[f64],
    options: &PlannerOptions,
    cancel: &CancellationToken,
    evaluate_index: Option<usize>,
) -> DomainResult<Vec<ResearchCandidate>> {
    let registry = &Registry::standard();
    options.validate()?;
    let measure = registry.measure(&options.quality_measure)?;
    let gripper = project.robot.require_gripper()?;
    let body = project
        .environment
        .body(body_id)
        .ok_or_else(|| DomainError::dangling("world body", body_id, "plan_pick"))?;
    // A single primitive has exact grasps and an exact centre; anything else
    // is handled through its surface.
    let primitive = body
        .single_geometry()
        .is_some_and(|g| !matches!(g.shape, Shape::Mesh { .. }));
    let generator = choose_generator(registry, gripper, measure, options, primitive)?;
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
        finger_width: gripper.finger_width,
        surface_samples: options.surface_samples,
        contact_tolerance: options.contact_tolerance,
        ..SynthesisOptions::default()
    };
    let (center_of_mass, torque_scale) = if primitive {
        primitive_frame(body)
    } else {
        let surface = PartSurface::of(body, &project.resources)?;
        (surface.center_of_mass, surface.radius)
    };
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
        measure,
        center_of_mass,
        torque_scale,
        measured: RefCell::new(BTreeMap::new()),
    };
    // Candidates the gripper cannot make at all (wider than its stroke, or
    // needing more force than it has) go after every other: they are
    // rejected cheaply, but each still counts against `maxCandidates`, and a
    // sampled surface yields many of them.
    let candidates = match generator {
        Some(generator) => generator.generate(body, &project.resources, &synthesis)?,
        None => synthesize_hand(
            project,
            &model,
            body,
            center_of_mass,
            &hand_synthesis(options, synthesis.seed),
        )?,
    };
    let mut output = Vec::new();
    for (id, candidate) in candidates.into_iter().enumerate() {
        if evaluate_index.is_some_and(|index| index != id) {
            continue;
        }
        let grasp_pose = context.grasp_pose(&candidate)?;
        let mut record = ResearchCandidate {
            id,
            candidate,
            grasp_pose,
            plan: None,
            rejection: None,
            quality: None,
            quality_elapsed_s: None,
            attempt_elapsed_s: None,
        };
        if evaluate_index.is_some() {
            let start = std::time::Instant::now();
            record.quality = Some(evaluate(measure, &context.quality_input(&record.candidate)));
            record.quality_elapsed_s = Some(start.elapsed().as_secs_f64());
            let start = std::time::Instant::now();
            let attempt = context.attempt(&record.candidate, seed)?;
            record.attempt_elapsed_s = Some(start.elapsed().as_secs_f64());
            match attempt {
                Ok(plan) => record.plan = Some(plan),
                Err((reason, detail)) => {
                    record.rejection = Some(Rejection {
                        label: record.candidate.label.clone(),
                        reason,
                        detail,
                        center: record.candidate.center,
                        approach: record.candidate.approach,
                    })
                }
            }
        }
        output.push(record);
    }
    Ok(output)
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
    fn research_catalog_preserves_order_and_defers_evaluation() {
        let (project, seed) = cell();
        let body = project.environment.body_by_name("cube").unwrap();
        let options = PlannerOptions::default();
        let cancel = CancellationToken::new();
        let catalog =
            research_candidates(&project, &body.id, &seed, &options, &cancel, None).unwrap();
        assert!(!catalog.is_empty());
        for (index, item) in catalog.iter().enumerate() {
            assert_eq!(item.id, index);
            assert!(item.plan.is_none() && item.rejection.is_none() && item.quality.is_none());
        }
        let selected =
            research_candidates(&project, &body.id, &seed, &options, &cancel, Some(0)).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].candidate, catalog[0].candidate);
        assert_eq!(selected[0].grasp_pose, catalog[0].grasp_pose);
        assert!(selected[0].quality.is_some());
        assert_ne!(selected[0].plan.is_some(), selected[0].rejection.is_some());
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
            nr_grasps: 1000,
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

    fn plan_with(
        project: &Project,
        name: &str,
        seed: &[f64],
        options: &PlannerOptions,
    ) -> PlanReport {
        let id = project.environment.body_by_name(name).unwrap().id.clone();
        plan_pick(project, &id, seed, options, &CancellationToken::new()).unwrap()
    }

    #[test]
    fn plans_carry_wrench_space_quality_by_default() {
        let (project, seed) = cell();
        let report = plan(&project, "cube", &seed);
        assert_eq!(report.quality_measure, "wrench_space");
        assert_eq!(report.candidate_generator, "primitive_antipodal");
        let best = &report.plans[0];
        assert_eq!(best.quality.measure, "wrench_space");
        assert!(
            best.quality.quality > 0.0 && best.quality.contact_count == 8,
            "{:?}",
            best.quality
        );
    }

    #[test]
    fn a_minimum_quality_out_of_reach_rejects_otherwise_feasible_grasps() {
        let (project, seed) = cell();
        let options = PlannerOptions {
            min_quality: 10.0,
            nr_grasps: 1000,
            ..PlannerOptions::default()
        };
        let report = plan_with(&project, "cube", &seed, &options);
        assert!(report.plans.is_empty());
        let below: Vec<_> = report
            .rejections
            .iter()
            .filter(|r| r.reason == RejectionReason::BelowMinQuality)
            .collect();
        // Exactly the grasps that were feasible without the requirement.
        let baseline = PlannerOptions {
            nr_grasps: 1000,
            ..PlannerOptions::default()
        };
        assert_eq!(
            below.len(),
            plan_with(&project, "cube", &seed, &baseline).plans.len()
        );
        assert!(below[0].detail.contains("minimum"), "{}", below[0].detail);
    }

    fn with_ball(mut project: Project) -> Project {
        let cube = project
            .environment
            .bodies
            .iter_mut()
            .find(|b| b.name == "cube")
            .unwrap();
        cube.geometries[0].shape = Shape::Sphere { radius: 0.02 };
        // Held a little off the table: a ball this size resting on it leaves
        // no room for fingertips below its equator, and that is not what is
        // under test. Not higher, or the pre-grasp leaves the wrist's range.
        cube.pose.translation.z = 0.45;
        project
    }

    #[test]
    fn a_ball_fails_force_closure_with_hard_contacts_and_passes_with_soft_pads() {
        let (project, seed) = cell();
        let project = with_ball(project);
        let hard = plan(&project, "cube", &seed);
        assert!(hard.plans.is_empty());
        assert!(hard
            .rejections
            .iter()
            .any(|r| r.reason == RejectionReason::NoForceClosure));

        let soft = PlannerOptions {
            quality: WrenchOptions {
                torsional_friction: 0.004,
                ..WrenchOptions::default()
            },
            ..PlannerOptions::default()
        };
        assert!(!plan_with(&project, "cube", &seed, &soft).plans.is_empty());

        // Not requiring force closure also lets it through, and says what it found.
        let lax = PlannerOptions {
            force_closure: false,
            ..PlannerOptions::default()
        };
        let report = plan_with(&project, "cube", &seed, &lax);
        assert!(!report.plans.is_empty());
        assert!(!report.plans[0].quality.force_closure);
    }

    #[test]
    fn the_antipodal_measure_can_be_selected_by_name() {
        let (project, seed) = cell();
        let options = PlannerOptions {
            quality_measure: "antipodal".to_owned(),
            ..PlannerOptions::default()
        };
        let report = plan_with(&project, "cube", &seed, &options);
        assert_eq!(report.plans[0].quality.measure, "antipodal");
    }

    #[test]
    fn a_registered_library_is_used_by_the_planner() {
        use crate::quality::Measurement;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct Counting(AtomicUsize);
        impl QualityMeasure for Counting {
            fn id(&self) -> &'static str {
                "counting"
            }
            fn summary(&self) -> &'static str {
                "Counts how often it is asked."
            }
            fn measure(&self, _: &QualityInput<'_>) -> Measurement {
                self.0.fetch_add(1, Ordering::Relaxed);
                Measurement {
                    quality: 0.25,
                    normalized: 0.25,
                    force_closure: true,
                }
            }
        }

        let (project, seed) = cell();
        let library = Arc::new(Counting(AtomicUsize::new(0)));
        let mut registry = Registry::standard();
        registry.register_measure(library.clone()).unwrap();
        let options = PlannerOptions {
            quality_measure: "counting".to_owned(),
            nr_grasps: 1000,
            ..PlannerOptions::default()
        };
        let id = project.environment.body_by_name("cube").unwrap().id.clone();
        let report = plan_pick_with(
            &registry,
            &project,
            &id,
            &seed,
            &options,
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(!report.plans.is_empty());
        assert!(report
            .plans
            .iter()
            .all(|p| (p.quality.quality - 0.25).abs() < 1e-12));
        // Both jaw assignments of a grasp share contacts, so it is asked once per pair.
        let calls = library.0.load(Ordering::Relaxed);
        assert!(
            calls > 0 && calls < report.plans.len(),
            "{calls} calls for {} plans",
            report.plans.len()
        );
    }

    /// The demo cube replaced by the same cube as a twelve-triangle mesh.
    fn with_mesh_cube(mut project: Project) -> Project {
        use axisforge_domain::{GeometryInstance, GeometryRole, Resource};
        let index = project
            .environment
            .bodies
            .iter()
            .position(|b| b.name == "cube")
            .unwrap();
        let primitive = project.environment.bodies[index].clone();
        let mut at_origin = primitive.clone();
        at_origin.pose = Transform::IDENTITY;
        at_origin.geometries[0].origin = Transform::IDENTITY;
        let surface = PartSurface::of(&at_origin, &[]).unwrap();
        project.resources.push(Resource {
            id: "cube-mesh".to_owned(),
            name: "cube.stl".to_owned(),
            media_type: "model/stl".to_owned(),
            vertices: surface.vertices.iter().flat_map(|v| v.to_array()).collect(),
            indices: surface.triangles.iter().flatten().copied().collect(),
        });
        let geometry = &primitive.geometries[0];
        project.environment.bodies[index].geometries = vec![GeometryInstance::visual(
            "cube-mesh-geometry",
            "cube mesh",
            Shape::Mesh {
                resource_id: "cube-mesh".to_owned(),
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
        )
        .with_origin(geometry.origin)
        .with_role(GeometryRole::Both)];
        project.validate().unwrap();
        project
    }

    #[test]
    fn a_mesh_part_is_planned_through_its_surface() {
        let (project, seed) = cell();
        let project = with_mesh_cube(project);
        let report = plan(&project, "cube", &seed);
        assert_eq!(report.candidate_generator, "surface_normal");
        assert!(!report.plans.is_empty(), "{:?}", report.rejections);
        for plan in &report.plans {
            assert!(plan.quality.force_closure);
            // A 40 mm cube is 40 mm across whichever way it is held.
            assert!(
                (plan.candidate.width - 0.04).abs() < 1e-9,
                "{}",
                plan.candidate.width
            );
        }
    }

    #[test]
    fn the_pad_model_reproduces_the_primitive_contacts() {
        // Closing the pads at each primitive candidate's own pose must give
        // that candidate's width and contacts: the analytic model and the
        // surface model describe the same physical pads.
        let (project, seed) = cell();
        let synthesis = SynthesisOptions::default();
        let pad = (synthesis.finger_width, synthesis.finger_depth);
        let mut compared = 0;
        for (name, cylinder_sides_only) in [("cube", false), ("dowel", true)] {
            let body = project.environment.body_by_name(name).unwrap();
            let axis = body.pose.apply_direction(&Vec3::Z);
            let surface = PartSurface::of(body, &[]).unwrap();
            for candidate in crate::candidate::synthesize(body, &synthesis).unwrap() {
                if cylinder_sides_only && candidate.closing.dot(&axis).abs() > 1e-9 {
                    continue;
                }
                let closure = surface
                    .close(
                        &candidate.center,
                        &candidate.closing,
                        &candidate.approach,
                        pad,
                        0.0,
                    )
                    .unwrap_or_else(|| panic!("{}: no closure", candidate.label));
                assert!(
                    (closure.width - candidate.width).abs() < 1e-9,
                    "{}: width {} vs {}",
                    candidate.label,
                    closure.width,
                    candidate.width
                );
                let found: Vec<_> = closure.positive.iter().chain(&closure.negative).collect();
                assert_eq!(
                    found.len(),
                    candidate.contacts.len(),
                    "{}: {found:?}",
                    candidate.label
                );
                for expected in &candidate.contacts {
                    assert!(
                        found
                            .iter()
                            .any(|c| c.point.distance(&expected.point) < 1e-8
                                && c.normal.distance(&expected.normal) < 1e-9),
                        "{}: no match for {expected:?} in {found:?}",
                        candidate.label
                    );
                }
                compared += 1;
            }
        }
        assert!(compared >= 60, "{compared} candidates compared");

        // Planned end to end as a mesh, the cube must still be held in force
        // closure, with an epsilon of the same order as the exact grasps'.
        let mesh_project = with_mesh_cube(project.clone());
        let options = PlannerOptions {
            nr_grasps: 1,
            ..PlannerOptions::default()
        };
        let both = |project: &Project, generator: &str| {
            let id = project.environment.body_by_name("cube").unwrap().id.clone();
            plan_pick(
                project,
                &id,
                &seed,
                &PlannerOptions {
                    candidate_generator: generator.to_owned(),
                    ..options.clone()
                },
                &CancellationToken::new(),
            )
            .unwrap()
        };
        let exact = both(&project, "primitive_antipodal");
        let sampled = both(&mesh_project, "auto");
        assert_eq!(sampled.candidate_generator, "surface_normal");
        let best = |r: &PlanReport| r.plans[0].quality.quality;
        // Sampling may not land on the very best primitive hold, but it must
        // find one in force closure of the same order.
        assert!(sampled.plans[0].quality.force_closure);
        println!("epsilon: sampled {} exact {}", best(&sampled), best(&exact));
        assert!(
            best(&sampled) > 0.3 * best(&exact),
            "{} vs {}",
            best(&sampled),
            best(&exact)
        );
    }

    #[test]
    fn an_assembled_part_is_no_longer_refused() {
        // A "T" of two boxes: the primitive generator cannot hold it, the
        // surface generator can.
        let (mut project, seed) = cell();
        let index = project
            .environment
            .bodies
            .iter()
            .position(|b| b.name == "cube")
            .unwrap();
        let mut stem = project.environment.bodies[index].geometries[0].clone();
        stem.id = "stem".to_owned();
        stem.shape = Shape::Box {
            size: Vec3::new(0.02, 0.02, 0.04),
        };
        stem.origin = Transform::from_translation(Vec3::new(0.0, 0.0, 0.03));
        project.environment.bodies[index].geometries.push(stem);
        project.validate().unwrap();
        let exact = plan_pick(
            &project,
            &project.environment.bodies[index].id,
            &seed,
            &PlannerOptions {
                candidate_generator: "primitive_antipodal".to_owned(),
                ..PlannerOptions::default()
            },
            &CancellationToken::new(),
        );
        assert!(exact.is_err(), "two geometries have no single set of faces");
        let report = plan(&project, "cube", &seed);
        assert_eq!(report.candidate_generator, "surface_normal");
        assert!(!report.plans.is_empty(), "{:?}", report.rejections);
        assert!(report.plans.iter().all(|p| p.quality.force_closure));
    }

    #[test]
    fn a_mesh_part_without_its_mesh_is_an_error() {
        let (project, seed) = cell();
        let mut project = with_mesh_cube(project);
        project.resources.clear();
        let id = project.environment.body_by_name("cube").unwrap().id.clone();
        let error = plan_pick(
            &project,
            &id,
            &seed,
            &PlannerOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cube-mesh"), "{error}");
    }

    #[test]
    fn unknown_extension_names_and_bad_options_are_errors() {
        let (project, seed) = cell();
        let id = project.environment.body_by_name("cube").unwrap().id.clone();
        let token = CancellationToken::new();
        let bad_measure = PlannerOptions {
            quality_measure: "nope".to_owned(),
            ..PlannerOptions::default()
        };
        let error = plan_pick(&project, &id, &seed, &bad_measure, &token).unwrap_err();
        assert!(error.to_string().contains("wrench_space"), "{error}");
        let bad_generator = PlannerOptions {
            candidate_generator: "nope".to_owned(),
            ..PlannerOptions::default()
        };
        assert!(plan_pick(&project, &id, &seed, &bad_generator, &token).is_err());
        let negative = PlannerOptions {
            min_quality: -1.0,
            ..PlannerOptions::default()
        };
        assert!(plan_pick(&project, &id, &seed, &negative, &token).is_err());
        let zero_force = PlannerOptions {
            quality: WrenchOptions {
                unit_force: 0.0,
                ..WrenchOptions::default()
            },
            ..PlannerOptions::default()
        };
        assert!(plan_pick(&project, &id, &seed, &zero_force, &token).is_err());
    }

    #[test]
    fn the_candidate_limit_and_the_time_budget_stop_the_search() {
        let (project, seed) = cell();
        let limited = PlannerOptions {
            max_candidates: Some(5),
            nr_grasps: 1000,
            ..PlannerOptions::default()
        };
        let report = plan_with(&project, "cube", &seed, &limited);
        assert_eq!(report.candidates_examined, 5);
        assert!(!report.timed_out);

        let no_time = PlannerOptions {
            timeout_seconds: Some(0.0),
            ..PlannerOptions::default()
        };
        let report = plan_with(&project, "cube", &seed, &no_time);
        assert!(report.timed_out);
        assert_eq!(report.candidates_examined, 0);
    }

    #[test]
    fn the_old_option_name_is_still_accepted() {
        let options: PlannerOptions = serde_json::from_str(r#"{"maxPlans": 7}"#).unwrap();
        assert_eq!(options.nr_grasps, 7);
        let options: PlannerOptions =
            serde_json::from_str(r#"{"nrGrasps": 2, "minQuality": 0.01, "forceClosure": false}"#)
                .unwrap();
        assert!(options.nr_grasps == 2 && !options.force_closure);
    }

    #[test]
    fn rejections_say_where_the_grasp_was() {
        let (project, seed) = cell();
        let options = PlannerOptions {
            nr_grasps: 1000,
            ..PlannerOptions::default()
        };
        let report = plan_with(&project, "cube", &seed, &options);
        let cube = project.environment.body_by_name("cube").unwrap();
        for rejection in &report.rejections {
            assert!(rejection.center.distance(&cube.pose.translation) < 0.03);
            assert!((rejection.approach.norm() - 1.0).abs() < 1e-9);
        }
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
