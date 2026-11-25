//! The whole pick-and-place cycle: from where the robot stands to the part,
//! the part to its target, and back.
//!
//! A pick planned by [`axisforge_grasping`] gives the grasp: the pre-grasp,
//! grasp and lift configurations and how the part sits in the hand. This
//! module adds everything around it and checks all of it continuously:
//!
//! | segment    | from → to                  | how                 | part       |
//! |------------|----------------------------|---------------------|------------|
//! | `open`     | start → fingers open       | gripper move        | on its spot |
//! | `approach` | → pre-grasp                | bidirectional RRT   | on its spot |
//! | `descend`  | → grasp                    | straight            | on its spot |
//! | `close`    | fingers open → on the part | gripper move        | in the hand |
//! | `lift`     | → lifted                   | straight            | carried    |
//! | `transfer` | → above the target         | bidirectional RRT   | carried    |
//! | `lower`    | → at the target            | straight            | carried    |
//! | `release`  | fingers on the part → open | gripper move        | placed     |
//! | `retreat`  | → backed off               | straight            | placed     |
//! | `return`   | → start, fingers open      | bidirectional RRT   | placed     |
//! | `rest`     | → start                    | gripper move        | placed     |
//!
//! Every segment is a straight joint-space line, or a chain of them, proven
//! clear by [`MotionChecker`] with what may touch in it: the fingers and the
//! part while the hand closes and opens on it, and the part and the surface
//! it leaves or is set down on while it is lifted and lowered. Nothing else
//! may come within the margin. The part keeps its orientation and is set
//! down with its origin at the target. Each move is timed with the
//! synchronized trapezoidal profile of [`crate::profile`], stopping at every
//! waypoint, plus a dwell after closing and after opening.
//!
//! Every feasible grasp the pick planner returns is tried, and the one whose
//! whole cycle is fastest is kept: grasps of similar quality can differ by
//! seconds, a half turn of the wrist for one. The grasps that give no cycle
//! are reported with the reason.

use axisforge_collision::{Attachment, SceneOptions};
use axisforge_domain::{
    DomainError, DomainResult, Project, Shape, Transform, Vec3, GROUND_OWNER_ID,
};
use axisforge_geometry::{body_from_geometry, gjk, ConvexBody};
use axisforge_grasping::{plan_pick, GraspPlan, PlannerOptions};
use axisforge_kinematics::{ik, CancellationToken, IkOptions, KinematicModel};
use serde::{Deserialize, Serialize};

use crate::continuous::{MotionChecker, SegmentVerdict};
use crate::profile::{profile_path, AxisLimits, Move};
use crate::rrt::{plan_path, RrtOptions};

/// A body this close to the part supports it, metres.
const SUPPORT_DISTANCE: f64 = 1.0e-3;

/// How a cycle is planned and timed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CycleOptions {
    /// How the grasp is planned; its collision margin, stand-off and lift
    /// height are the cycle's too.
    pub grasp: PlannerOptions,
    /// How free motions are searched for.
    pub path: RrtOptions,
    /// Fraction of every joint's velocity limit to use, in `(0, 1]`;
    /// accelerations are scaled by its square.
    pub speed_scale: f64,
    /// Pause after the fingers close and after they open, seconds.
    pub gripper_dwell: f64,
    /// Go back to the start configuration at the end.
    pub return_to_start: bool,
}

impl Default for CycleOptions {
    fn default() -> Self {
        Self {
            grasp: PlannerOptions {
                nr_grasps: 6,
                ..PlannerOptions::default()
            },
            path: RrtOptions::default(),
            speed_scale: 0.5,
            gripper_dwell: 0.3,
            return_to_start: true,
        }
    }
}

impl CycleOptions {
    /// Checks the settings.
    ///
    /// # Errors
    /// Values out of range.
    pub fn validate(&self) -> DomainResult<()> {
        self.grasp.validate()?;
        self.path.validate()?;
        if !self.speed_scale.is_finite() || self.speed_scale <= 0.0 || self.speed_scale > 1.0 {
            return Err(DomainError::unsupported(format!(
                "speedScale must be in (0, 1]; it is {}",
                self.speed_scale
            )));
        }
        if !self.gripper_dwell.is_finite() || self.gripper_dwell < 0.0 {
            return Err(DomainError::NegativeValue {
                context: "gripperDwell".to_owned(),
                value: self.gripper_dwell,
            });
        }
        Ok(())
    }
}

/// What moves in a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentKind {
    /// The arm moves; the fingers keep their posture.
    Arm,
    /// Only the fingers move.
    Gripper,
}

/// Where the part is during a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartState {
    /// Where it started.
    AtStart,
    /// In the hand, moving with it.
    Carried,
    /// At its target.
    Placed,
}

/// One stretch of the cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleSegment {
    /// `open`, `approach`, `descend`, `close`, `lift`, `transfer`, `lower`,
    /// `release`, `retreat`, `return` or `rest`.
    pub name: String,
    /// What moves.
    pub kind: SegmentKind,
    /// `straight`, `rrt` or `gripper`.
    pub planner: String,
    /// Where the part is.
    pub part: PartState,
    /// The straight moves, in order, with their profiles.
    pub moves: Vec<Move>,
    /// When it starts, seconds from the start of the cycle.
    pub start_time: f64,
    /// How long its moves take, seconds.
    pub duration: f64,
    /// Pause after the moves, seconds.
    pub dwell: f64,
    /// Certified lower bound on the clearance along it, metres, among the
    /// pairs not allowed to touch.
    pub clearance: f64,
    /// Smallest distance measured along it among those pairs, metres; the
    /// true smallest lies between this and `clearance`.
    pub closest: f64,
    /// Pairs allowed to touch in it, by name.
    pub allowed_contacts: Vec<String>,
}

/// A planned cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cycle {
    /// The part moved.
    pub body_id: String,
    /// The grasp used.
    pub grasp: GraspPlan,
    /// The part's pose at its target, world coordinates.
    pub place_pose: Transform,
    /// The segments, in order.
    pub segments: Vec<CycleSegment>,
    /// Total time, seconds: every move's profile plus the dwells.
    pub cycle_time: f64,
    /// Smallest certified clearance over every segment, metres.
    pub clearance: f64,
    /// Smallest distance measured over every segment, metres.
    pub closest: f64,
    /// The limits the profiles respect, scaled.
    pub limits: AxisLimits,
    /// Joint names in configuration order.
    pub joint_names: Vec<String>,
    /// Configuration slots of the manipulator axes.
    pub manipulator_slots: Vec<usize>,
    /// Identifier of the link that carries the part.
    pub palm_link_id: String,
}

/// Where the robot and the part are at one instant of a cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleState {
    /// Configuration, full width.
    pub values: Vec<f64>,
    /// Where the part is.
    pub part: PartState,
    /// Index of the segment.
    pub segment: usize,
}

impl Cycle {
    /// The state at time `t` seconds from the start; before the start it is
    /// the first state and after the end the last.
    pub fn state_at(&self, t: f64) -> CycleState {
        let index = self
            .segments
            .iter()
            .position(|s| t < s.start_time + s.duration + s.dwell)
            .unwrap_or(self.segments.len().saturating_sub(1));
        let Some(segment) = self.segments.get(index) else {
            return CycleState {
                values: Vec::new(),
                part: PartState::AtStart,
                segment: 0,
            };
        };
        let mut elapsed = (t - segment.start_time).max(0.0);
        for m in &segment.moves {
            if elapsed <= m.duration {
                return CycleState {
                    values: m.position(elapsed),
                    part: segment.part,
                    segment: index,
                };
            }
            elapsed -= m.duration;
        }
        CycleState {
            values: segment
                .moves
                .last()
                .map(|m| m.to.clone())
                .unwrap_or_default(),
            part: segment.part,
            segment: index,
        }
    }
}

/// A grasp that did not give a cycle, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleAttempt {
    /// The grasp's label.
    pub grasp: String,
    /// The segment that failed.
    pub segment: String,
    /// Why.
    pub reason: String,
}

/// Everything planning a cycle found out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleReport {
    /// The cycle, when one was found.
    pub cycle: Option<Cycle>,
    /// Grasps that gave no cycle, with the reason.
    pub attempts: Vec<CycleAttempt>,
    /// Feasible grasps the pick planner found.
    pub grasps_found: usize,
    /// How many of them gave a whole cycle; the fastest is `cycle`.
    pub cycles_found: usize,
    /// Joints whose speed or acceleration limits were assumed.
    pub assumed_limits: Vec<String>,
}

type Fail = (String, String);

struct Planner<'a> {
    project: &'a Project,
    placed: Project,
    model: &'a KinematicModel,
    options: &'a CycleOptions,
    cancel: &'a CancellationToken,
    limits: AxisLimits,
    frame: usize,
    fingers: Vec<usize>,
    finger_links: Vec<String>,
    palm_link_id: String,
    body_id: String,
    body_name: String,
}

fn mesh_vertices<'a>(project: &'a Project, shape: &Shape) -> Option<&'a [f64]> {
    match shape {
        Shape::Mesh { resource_id, .. } => project
            .resource(resource_id)
            .map(|resource| resource.vertices.as_slice()),
        _ => None,
    }
}

/// The part's convex pieces at `pose`.
fn part_pieces(
    project: &Project,
    body_id: &str,
    pose: &Transform,
) -> DomainResult<Vec<ConvexBody>> {
    let body = project
        .environment
        .body(body_id)
        .ok_or_else(|| DomainError::dangling("world body", body_id, "cycle"))?;
    body.geometries
        .iter()
        .filter(|g| g.is_collision())
        .map(|g| body_from_geometry(g, pose, mesh_vertices(project, &g.shape)))
        .collect()
}

/// Identifiers and names of what is within `distance` of the part at `pose`:
/// other world bodies, and the ground.
fn near_part(
    project: &Project,
    body_id: &str,
    pose: &Transform,
    distance: f64,
) -> DomainResult<Vec<(String, String)>> {
    let part = part_pieces(project, body_id, pose)?;
    let mut out = Vec::new();
    for other in project
        .environment
        .bodies
        .iter()
        .filter(|b| b.id != body_id)
    {
        let close = other
            .geometries
            .iter()
            .filter(|g| g.is_collision())
            .any(|g| {
                body_from_geometry(g, &other.pose, mesh_vertices(project, &g.shape)).is_ok_and(
                    |b| {
                        part.iter()
                            .any(|p| gjk::distance(p, &b).distance <= distance)
                    },
                )
            });
        if close {
            out.push((other.id.clone(), other.name.clone()));
        }
    }
    if project.environment.ground_collides {
        let up = project.environment.ground_normal();
        let lowest = part
            .iter()
            .map(|p| p.core_support(&up.scaled(-1.0)).dot(&up) - p.inflation)
            .fold(f64::INFINITY, f64::min);
        if lowest - project.environment.ground_height <= distance {
            out.push((GROUND_OWNER_ID.to_owned(), "ground".to_owned()));
        }
    }
    Ok(out)
}

impl Planner<'_> {
    fn with_fingers(&self, values: &[f64], from: &[f64]) -> Vec<f64> {
        let mut out = values.to_vec();
        for slot in &self.fingers {
            out[*slot] = from[*slot];
        }
        out
    }

    fn checker<'p>(
        &self,
        project: &'p Project,
        options: &SceneOptions,
    ) -> DomainResult<MotionChecker<'p>>
    where
        Self: 'p,
    {
        MotionChecker::new(
            project,
            self.model,
            options,
            self.options.grasp.collision_margin,
        )
    }

    fn carrying(&self, grasp: &GraspPlan, also: &[(String, String)]) -> SceneOptions {
        SceneOptions {
            attachment: Some(Attachment {
                body_id: self.body_id.clone(),
                link_id: self.palm_link_id.clone(),
                relative_pose: grasp.part_in_palm,
            }),
            extra_ignored: also
                .iter()
                .map(|(id, _)| (self.body_id.clone(), id.clone()))
                .collect(),
        }
    }

    fn holding(&self) -> SceneOptions {
        SceneOptions {
            attachment: None,
            extra_ignored: self
                .finger_links
                .iter()
                .map(|f| (f.clone(), self.body_id.clone()))
                .collect(),
        }
    }

    fn solve(
        &self,
        target: &Transform,
        seed: &[f64],
        restarts: bool,
    ) -> DomainResult<Option<Vec<f64>>> {
        let options = if restarts {
            self.options.grasp.ik.clone()
        } else {
            IkOptions {
                restarts: 0,
                ..self.options.grasp.ik.clone()
            }
        };
        let solution = ik::solve(self.model, self.frame, target, seed, &options, self.cancel)?;
        Ok(solution.converged.then_some(solution.values))
    }

    /// An arm configuration at `target` that `checker` finds clear, with the
    /// fingers as in `fingers`: the solution nearest `seed` first, then
    /// solutions from other restart sequences, since the first one found may
    /// put the arm through something the target itself is clear of.
    fn solve_clear(
        &self,
        target: &Transform,
        seed: &[f64],
        fingers: &[f64],
        checker: &MotionChecker<'_>,
    ) -> DomainResult<Result<Vec<f64>, String>> {
        const SEQUENCES: u64 = 6;
        let mut blocked = None;
        for sequence in 0..SEQUENCES {
            let options = IkOptions {
                rng_seed: self.options.grasp.ik.rng_seed.wrapping_add(sequence),
                ..self.options.grasp.ik.clone()
            };
            let solution = ik::solve(self.model, self.frame, target, seed, &options, self.cancel)?;
            if !solution.converged {
                continue;
            }
            let values = self.with_fingers(&solution.values, fingers);
            if checker.is_clear(&values)? {
                return Ok(Ok(values));
            }
            if blocked.is_none() {
                blocked = checker.closest(&values)?.map(|c| {
                    format!(
                        "every solution found is in contact; '{}' and '{}' are {:.1} mm apart",
                        c.first,
                        c.second,
                        c.distance * 1000.0
                    )
                });
            }
        }
        Ok(Err(blocked.unwrap_or_else(|| "no IK solution".to_owned())))
    }

    /// A straight segment, proven clear, as one move.
    #[allow(clippy::too_many_arguments)]
    fn straight(
        &self,
        name: &str,
        kind: SegmentKind,
        part: PartState,
        checker: &MotionChecker<'_>,
        from: &[f64],
        to: &[f64],
        allowed: Vec<String>,
    ) -> DomainResult<Result<CycleSegment, Fail>> {
        let verdict = checker.check_segment(from, to)?;
        let SegmentVerdict::Clear { clearance, closest } = verdict else {
            return Ok(Err((name.to_owned(), describe(&verdict))));
        };
        let dwell = if kind == SegmentKind::Gripper && matches!(name, "close" | "release") {
            self.options.gripper_dwell
        } else {
            0.0
        };
        let moves = vec![Move::new(from, to, &self.limits)];
        Ok(Ok(CycleSegment {
            name: name.to_owned(),
            kind,
            planner: if kind == SegmentKind::Gripper {
                "gripper"
            } else {
                "straight"
            }
            .to_owned(),
            part,
            duration: moves.iter().map(|m| m.duration).sum(),
            moves,
            start_time: 0.0,
            dwell,
            clearance,
            closest,
            allowed_contacts: allowed,
        }))
    }

    /// A free motion found by the RRT.
    fn free(
        &self,
        name: &str,
        part: PartState,
        checker: &MotionChecker<'_>,
        from: &[f64],
        to: &[f64],
        allowed: Vec<String>,
    ) -> DomainResult<Result<CycleSegment, Fail>> {
        let plan = match plan_path(
            checker,
            self.model,
            from,
            to,
            &self.options.path,
            self.cancel,
        )? {
            Ok(plan) => plan,
            Err(failure) => return Ok(Err((name.to_owned(), failure.to_string()))),
        };
        let moves = profile_path(&plan.waypoints, &self.limits);
        Ok(Ok(CycleSegment {
            name: name.to_owned(),
            kind: SegmentKind::Arm,
            planner: "rrt".to_owned(),
            part,
            duration: moves.iter().map(|m| m.duration).sum(),
            moves,
            start_time: 0.0,
            dwell: 0.0,
            clearance: plan.clearance,
            closest: plan.closest,
            allowed_contacts: allowed,
        }))
    }

    fn finger_contacts(&self) -> Vec<String> {
        vec![format!("fingers ↔ {}", self.body_name)]
    }

    fn support_contacts(&self, supports: &[(String, String)]) -> Vec<String> {
        supports
            .iter()
            .map(|(_, name)| format!("{} ↔ {name}", self.body_name))
            .collect()
    }

    #[allow(clippy::too_many_lines)] // The cycle's segments in order; split, it would only pass state around.
    fn attempt(&self, grasp: &GraspPlan, start: &[f64]) -> DomainResult<Result<Cycle, Fail>> {
        macro_rules! segment {
            ($result:expr) => {
                match $result? {
                    Ok(segment) => segment,
                    Err(fail) => return Ok(Err(fail)),
                }
            };
        }
        let body = self
            .project
            .environment
            .body(&self.body_id)
            .ok_or_else(|| DomainError::dangling("world body", self.body_id.clone(), "cycle"))?;
        let place_pose = self
            .placed
            .environment
            .body(&self.body_id)
            .map_or(body.pose, |b| b.pose);
        let up = self.project.environment.ground_normal();
        let pick_supports = near_part(self.project, &self.body_id, &body.pose, SUPPORT_DISTANCE)?;
        let place_supports = near_part(&self.placed, &self.body_id, &place_pose, SUPPORT_DISTANCE)?;

        // Where the grasp frame goes to set the part down.
        let in_part = body.pose.inverse().compose(&grasp.grasp_pose);
        let at_place = place_pose.compose(&in_part);
        let above_place = Transform::new(
            at_place
                .translation
                .add(&up.scaled(self.options.grasp.lift_height)),
            at_place.rotation,
        );
        let approach =
            at_place.apply_direction(&self.project.robot.require_gripper()?.approach_axis);
        let backed_off = Transform::new(
            at_place
                .translation
                .sub(&approach.scaled(self.options.grasp.standoff)),
            at_place.rotation,
        );
        let free = self.checker(self.project, &SceneOptions::default())?;
        let holding = self.checker(self.project, &self.holding())?;
        let lifting = self.checker(self.project, &self.carrying(grasp, &pick_supports))?;
        let carrying = self.checker(self.project, &self.carrying(grasp, &[]))?;
        let lowering = self.checker(self.project, &self.carrying(grasp, &place_supports))?;
        let releasing = self.checker(&self.placed, &self.holding())?;
        let after = self.checker(&self.placed, &SceneOptions::default())?;

        let preplace =
            match self.solve_clear(&above_place, &grasp.lift, &grasp.grasp_closed, &carrying)? {
                Ok(values) => values,
                Err(reason) => {
                    return Ok(Err((
                        "transfer".to_owned(),
                        format!("above the target: {reason}"),
                    )))
                }
            };
        let Some(place) = self.solve(&at_place, &preplace, false)? else {
            return Ok(Err((
                "lower".to_owned(),
                "no IK solution at the target continuing from above it".to_owned(),
            )));
        };
        let place = self.with_fingers(&place, &grasp.grasp_closed);
        let place_open = self.with_fingers(&place, &grasp.grasp_open);
        let Some(retreat) = self.solve(&backed_off, &place_open, false)? else {
            return Ok(Err((
                "retreat".to_owned(),
                "no IK solution backed off from the target".to_owned(),
            )));
        };
        let retreat = self.with_fingers(&retreat, &grasp.grasp_open);
        let start_open = self.with_fingers(start, &grasp.grasp_open);

        let mut segments = Vec::new();
        let gripper = SegmentKind::Gripper;
        let arm = SegmentKind::Arm;
        segments.push(segment!(self.straight(
            "open",
            gripper,
            PartState::AtStart,
            &free,
            start,
            &start_open,
            Vec::new()
        )));
        segments.push(segment!(self.free(
            "approach",
            PartState::AtStart,
            &free,
            &start_open,
            &grasp.pregrasp,
            Vec::new()
        )));
        segments.push(segment!(self.straight(
            "descend",
            arm,
            PartState::AtStart,
            &free,
            &grasp.pregrasp,
            &grasp.grasp_open,
            Vec::new()
        )));
        segments.push(segment!(self.straight(
            "close",
            gripper,
            PartState::Carried,
            &holding,
            &grasp.grasp_open,
            &grasp.grasp_closed,
            self.finger_contacts()
        )));
        segments.push(segment!(self.straight(
            "lift",
            arm,
            PartState::Carried,
            &lifting,
            &grasp.grasp_closed,
            &grasp.lift,
            self.support_contacts(&pick_supports)
        )));
        segments.push(segment!(self.free(
            "transfer",
            PartState::Carried,
            &carrying,
            &grasp.lift,
            &preplace,
            Vec::new()
        )));
        segments.push(segment!(self.straight(
            "lower",
            arm,
            PartState::Carried,
            &lowering,
            &preplace,
            &place,
            self.support_contacts(&place_supports)
        )));
        segments.push(segment!(self.straight(
            "release",
            gripper,
            PartState::Placed,
            &releasing,
            &place,
            &place_open,
            self.finger_contacts()
        )));
        segments.push(segment!(self.straight(
            "retreat",
            arm,
            PartState::Placed,
            &after,
            &place_open,
            &retreat,
            Vec::new()
        )));
        if self.options.return_to_start {
            segments.push(segment!(self.free(
                "return",
                PartState::Placed,
                &after,
                &retreat,
                &start_open,
                Vec::new()
            )));
            segments.push(segment!(self.straight(
                "rest",
                gripper,
                PartState::Placed,
                &after,
                &start_open,
                start,
                Vec::new()
            )));
        }
        // Gripper moves that change nothing take no time and say nothing.
        segments.retain(|s| s.duration > 0.0 || s.dwell > 0.0);
        let mut time = 0.0;
        for segment in &mut segments {
            segment.start_time = time;
            time += segment.duration + segment.dwell;
        }
        let clearance = segments
            .iter()
            .map(|s| s.clearance)
            .fold(f64::INFINITY, f64::min);
        let closest = segments
            .iter()
            .map(|s| s.closest)
            .fold(f64::INFINITY, f64::min);
        Ok(Ok(Cycle {
            body_id: self.body_id.clone(),
            grasp: grasp.clone(),
            place_pose,
            segments,
            cycle_time: time,
            clearance,
            closest,
            limits: self.limits.clone(),
            joint_names: (0..self.model.dof())
                .map(|slot| {
                    self.model
                        .joint_for_value(slot)
                        .map_or_else(|| format!("joint {slot}"), |j| j.name.clone())
                })
                .collect(),
            manipulator_slots: self.model.manipulator_value_indices(),
            palm_link_id: self.palm_link_id.clone(),
        }))
    }
}

fn describe(verdict: &SegmentVerdict) -> String {
    match verdict {
        SegmentVerdict::Clear { .. } => "clear".to_owned(),
        SegmentVerdict::Contact {
            first,
            second,
            at,
            distance,
        } => format!(
            "'{first}' and '{second}' come within {:.1} mm at {:.0} % of the way",
            distance * 1000.0,
            at * 100.0
        ),
        SegmentVerdict::Unproven { first, second, at } => format!(
            "'{first}' and '{second}' graze the margin near {:.0} % of the way and could not \
             be proven clear",
            at * 100.0
        ),
    }
}

/// Plans the whole cycle that takes the part `body_id` from where it is to
/// `target` (where its origin is set down; it keeps its orientation),
/// starting and, by default, ending at `start`.
///
/// Not finding a cycle is not an error: the report says why for each grasp.
///
/// # Errors
/// What [`plan_pick`] rejects, invalid options, a target that overlaps
/// something, or a start configuration that does not fit the robot.
pub fn plan_cycle(
    project: &Project,
    body_id: &str,
    target: Vec3,
    start: &[f64],
    options: &CycleOptions,
    cancel: &CancellationToken,
) -> DomainResult<CycleReport> {
    options.validate()?;
    let model = KinematicModel::from_robot(&project.robot)?;
    model.check_configuration(start)?;
    let gripper = project.robot.require_gripper()?;
    let body = project
        .environment
        .body(body_id)
        .ok_or_else(|| DomainError::dangling("world body", body_id, "plan_cycle"))?;
    if !target.x.is_finite() || !target.y.is_finite() || !target.z.is_finite() {
        return Err(DomainError::NonFinite {
            context: "cycle target".to_owned(),
        });
    }
    // The part at its target, raised a millimetre, must touch nothing: the
    // target is free and the part is not set down into something.
    let place_pose = Transform::new(target, body.pose.rotation);
    let up = project.environment.ground_normal();
    let raised = Transform::new(target.add(&up.scaled(SUPPORT_DISTANCE)), body.pose.rotation);
    let blocking = near_part(project, body_id, &raised, 0.0)?;
    if let Some((_, name)) = blocking.first() {
        return Err(DomainError::unsupported(format!(
            "the target is occupied: '{}' set down there overlaps '{name}'",
            body.name
        )));
    }
    let mut placed = project.clone();
    if let Some(moved) = placed
        .environment
        .bodies
        .iter_mut()
        .find(|b| b.id == body_id)
    {
        moved.pose = place_pose;
    }

    let limits = AxisLimits::of(&project.robot, &model, options.speed_scale);
    let pick = plan_pick(project, body_id, start, &options.grasp, cancel)?;
    let frame = model.frame_index(&gripper.grasp_frame_id).ok_or_else(|| {
        DomainError::dangling("tool frame", gripper.grasp_frame_id.clone(), "plan_cycle")
    })?;
    let planner = Planner {
        project,
        placed,
        model: &model,
        options,
        cancel,
        limits,
        frame,
        fingers: model.end_effector_value_indices(),
        finger_links: gripper.finger_link_ids.clone(),
        palm_link_id: gripper.palm_link_id.clone(),
        body_id: body_id.to_owned(),
        body_name: body.name.clone(),
    };
    let mut report = CycleReport {
        cycle: None,
        attempts: Vec::new(),
        grasps_found: pick.plans.len(),
        cycles_found: 0,
        assumed_limits: planner.limits.assumed.clone(),
    };
    for grasp in &pick.plans {
        if cancel.is_cancelled() {
            break;
        }
        match planner.attempt(grasp, start)? {
            Ok(cycle) => {
                report.cycles_found += 1;
                // The fastest wins; on a tie, the better grasp, found first.
                if report
                    .cycle
                    .as_ref()
                    .is_none_or(|best| cycle.cycle_time < best.cycle_time - 1.0e-9)
                {
                    report.cycle = Some(cycle);
                }
            }
            Err((segment, reason)) => report.attempts.push(CycleAttempt {
                grasp: grasp.candidate.label.clone(),
                segment,
                reason,
            }),
        }
    }
    Ok(report)
}
