//! Collision-free paths between two configurations: bidirectional RRT with
//! shortcut smoothing.
//!
//! The planner is RRT-Connect (Kuffner and `LaValle`, "RRT-Connect: an
//! efficient approach to single-query path planning", ICRA 2000): a tree
//! grows from each end; each iteration extends one toward a random
//! configuration and then greedily connects the other to the new node, and
//! the trees swap roles. Every edge added is a straight joint-space segment
//! proven clear by [`MotionChecker::check_segment`], so the path it returns
//! is clear along its whole length, not at samples.
//!
//! The raw path zigzags. Shortcutting then tries random pairs of its
//! waypoints and replaces what lies between them with a straight segment
//! whenever that segment is proven clear too. Only the manipulator axes
//! move; end-effector joints keep their values from the start.
//!
//! Sampling is seeded, so the same request always gives the same path.

use axisforge_domain::{DomainError, DomainResult};
use axisforge_kinematics::{CancellationToken, KinematicModel, Rng};
use serde::{Deserialize, Serialize};

use crate::continuous::{MotionChecker, SegmentVerdict};

/// How paths are searched for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RrtOptions {
    /// Longest edge added to a tree in one step, in joint-space units
    /// (radians, or metres for a prismatic axis), Euclidean.
    pub step: f64,
    /// Give up after this many iterations.
    pub max_iterations: usize,
    /// Random shortcuts tried on the path found.
    pub shortcut_attempts: usize,
    /// Seed of the sampling.
    pub seed: u64,
    /// Give up after this many seconds. Needs a clock: refused on builds
    /// without one (WebAssembly), where `maxIterations` bounds the work.
    pub timeout_seconds: Option<f64>,
}

impl Default for RrtOptions {
    fn default() -> Self {
        Self {
            step: 0.4,
            max_iterations: 4000,
            shortcut_attempts: 120,
            seed: 7,
            timeout_seconds: None,
        }
    }
}

impl RrtOptions {
    /// Checks the settings.
    ///
    /// # Errors
    /// A step that is not positive and finite, or a time budget on a build
    /// without a clock.
    pub fn validate(&self) -> DomainResult<()> {
        if !self.step.is_finite() || self.step <= 0.0 {
            return Err(DomainError::NegativeValue {
                context: "path step must be positive".to_owned(),
                value: self.step,
            });
        }
        if let Some(seconds) = self.timeout_seconds {
            if cfg!(target_arch = "wasm32") {
                return Err(DomainError::unsupported(
                    "a time budget needs a clock, which this build does not have; bound the \
                     search with maxIterations instead"
                        .to_owned(),
                ));
            }
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(DomainError::NegativeValue {
                    context: "timeoutSeconds must be positive".to_owned(),
                    value: seconds,
                });
            }
        }
        Ok(())
    }
}

/// A path found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathPlan {
    /// Configurations from start to goal, full width; consecutive ones are
    /// joined by straight segments, each proven clear.
    pub waypoints: Vec<Vec<f64>>,
    /// Waypoints before shortcutting.
    pub raw_waypoints: usize,
    /// Iterations the search took; zero when the straight segment was clear.
    pub iterations: usize,
    /// Nodes in both trees together.
    pub tree_nodes: usize,
    /// Certified lower bound on the clearance along the whole path, metres.
    pub clearance: f64,
    /// Smallest distance measured along it, metres.
    pub closest: f64,
}

/// Why no path was returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "reason",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum PathFailure {
    /// The start itself is in contact.
    StartBlocked {
        /// The closest pair there.
        detail: String,
    },
    /// The goal itself is in contact.
    GoalBlocked {
        /// The closest pair there.
        detail: String,
    },
    /// The iteration limit was reached without joining the trees.
    NotFound {
        /// Iterations tried.
        iterations: usize,
    },
    /// The time budget ran out.
    TimedOut {
        /// Iterations tried.
        iterations: usize,
    },
    /// A caller cancelled the search.
    Cancelled,
}

impl std::fmt::Display for PathFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartBlocked { detail } => write!(f, "the start is in contact: {detail}"),
            Self::GoalBlocked { detail } => write!(f, "the goal is in contact: {detail}"),
            Self::NotFound { iterations } => {
                write!(f, "no path found in {iterations} iterations")
            }
            Self::TimedOut { iterations } => {
                write!(f, "the time budget ran out after {iterations} iterations")
            }
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

struct Tree {
    nodes: Vec<Vec<f64>>,
    parents: Vec<Option<usize>>,
}

impl Tree {
    fn new(root: Vec<f64>) -> Self {
        Self {
            nodes: vec![root],
            parents: vec![None],
        }
    }

    fn nearest(&self, q: &[f64], axes: &[usize]) -> usize {
        let mut best = (0, f64::INFINITY);
        for (index, node) in self.nodes.iter().enumerate() {
            let d = distance(node, q, axes);
            if d < best.1 {
                best = (index, d);
            }
        }
        best.0
    }

    /// The configurations from the root to `index`.
    fn branch(&self, mut index: usize) -> Vec<Vec<f64>> {
        let mut out = vec![self.nodes[index].clone()];
        while let Some(parent) = self.parents[index] {
            out.push(self.nodes[parent].clone());
            index = parent;
        }
        out.reverse();
        out
    }
}

fn distance(a: &[f64], b: &[f64], axes: &[usize]) -> f64 {
    axes.iter()
        .map(|i| (a[*i] - b[*i]) * (a[*i] - b[*i]))
        .sum::<f64>()
        .sqrt()
}

#[derive(PartialEq)]
enum Extension {
    Trapped,
    Advanced,
    Reached,
}

struct Search<'c, 'a> {
    checker: &'c MotionChecker<'a>,
    axes: Vec<usize>,
    step: f64,
}

impl Search<'_, '_> {
    fn steer(&self, from: &[f64], to: &[f64]) -> (Vec<f64>, bool) {
        let d = distance(from, to, &self.axes);
        if d <= self.step {
            return (to.to_vec(), true);
        }
        let k = self.step / d;
        let mut out = from.to_vec();
        for i in &self.axes {
            out[*i] = from[*i] + (to[*i] - from[*i]) * k;
        }
        (out, false)
    }

    fn extend(&self, tree: &mut Tree, target: &[f64]) -> DomainResult<Extension> {
        let near = tree.nearest(target, &self.axes);
        let (new, reached) = self.steer(&tree.nodes[near], target);
        if !self
            .checker
            .check_segment(&tree.nodes[near], &new)?
            .is_clear()
        {
            return Ok(Extension::Trapped);
        }
        tree.nodes.push(new);
        tree.parents.push(Some(near));
        Ok(if reached {
            Extension::Reached
        } else {
            Extension::Advanced
        })
    }

    fn connect(&self, tree: &mut Tree, target: &[f64]) -> DomainResult<Extension> {
        loop {
            match self.extend(tree, target)? {
                Extension::Advanced => {}
                other => return Ok(other),
            }
        }
    }
}

fn blocked(checker: &MotionChecker<'_>, q: &[f64]) -> DomainResult<Option<String>> {
    if checker.is_clear(q)? {
        return Ok(None);
    }
    Ok(Some(checker.closest(q)?.map_or_else(
        || "unknown".to_owned(),
        |c| {
            format!(
                "'{}' and '{}' are {:.1} mm apart",
                c.first,
                c.second,
                c.distance * 1000.0
            )
        },
    )))
}

/// Plans a collision-free path from `start` to `goal`.
///
/// Both must be full-width configurations with the same end-effector
/// values. Not finding a path is not an error: the result says why.
///
/// # Errors
/// Invalid options, configurations that do not fit the model, or that
/// differ in their end-effector joints.
///
/// # Panics
/// Never: a node is read back only right after it was added.
pub fn plan_path(
    checker: &MotionChecker<'_>,
    model: &KinematicModel,
    start: &[f64],
    goal: &[f64],
    options: &RrtOptions,
    cancel: &CancellationToken,
) -> DomainResult<Result<PathPlan, PathFailure>> {
    options.validate()?;
    model.check_configuration(start)?;
    model.check_configuration(goal)?;
    for i in model.end_effector_value_indices() {
        if (start[i] - goal[i]).abs() > 1.0e-12 {
            return Err(DomainError::unsupported(
                "a path moves the arm only; open or close the gripper before or after it"
                    .to_owned(),
            ));
        }
    }
    if let Some(detail) = blocked(checker, start)? {
        return Ok(Err(PathFailure::StartBlocked { detail }));
    }
    if let Some(detail) = blocked(checker, goal)? {
        return Ok(Err(PathFailure::GoalBlocked { detail }));
    }
    let search = Search {
        checker,
        axes: model.manipulator_value_indices(),
        step: options.step,
    };

    if let SegmentVerdict::Clear { clearance, closest } = checker.check_segment(start, goal)? {
        return Ok(Ok(PathPlan {
            waypoints: vec![start.to_vec(), goal.to_vec()],
            raw_waypoints: 2,
            iterations: 0,
            tree_nodes: 2,
            clearance,
            closest,
        }));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let deadline = options
        .timeout_seconds
        .map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s));
    let limits = model.limits();
    let mut rng = Rng::new(options.seed);
    let mut from_start = Tree::new(start.to_vec());
    let mut from_goal = Tree::new(goal.to_vec());
    let mut start_side = true;
    let mut found = None;
    let mut iterations = 0;
    while iterations < options.max_iterations {
        iterations += 1;
        if cancel.is_cancelled() {
            return Ok(Err(PathFailure::Cancelled));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(Err(PathFailure::TimedOut { iterations }));
        }
        let mut sample = start.to_vec();
        for i in &search.axes {
            sample[*i] = rng.next_in(limits[*i].lower, limits[*i].upper);
        }
        let (grow, other) = if start_side {
            (&mut from_start, &mut from_goal)
        } else {
            (&mut from_goal, &mut from_start)
        };
        if search.extend(grow, &sample)? != Extension::Trapped {
            let new = grow.nodes.last().expect("a node was just added").clone();
            if search.connect(other, &new)? == Extension::Reached {
                found = Some((from_start.nodes.len() - 1, from_goal.nodes.len() - 1));
                break;
            }
        }
        start_side = !start_side;
    }
    let Some((at_start, at_goal)) = found else {
        return Ok(Err(PathFailure::NotFound { iterations }));
    };
    let mut path = from_start.branch(at_start);
    let mut tail = from_goal.branch(at_goal);
    tail.reverse();
    // Both branches end at the same configuration.
    path.extend(tail.into_iter().skip(1));
    let raw_waypoints = path.len();
    let tree_nodes = from_start.nodes.len() + from_goal.nodes.len();

    let path = shortcut(checker, path, options, &mut rng)?;
    let (_, verdict) = checker.check_path(&path)?;
    let SegmentVerdict::Clear { clearance, closest } = verdict else {
        return Err(DomainError::unsupported(format!(
            "internal error: a path built from segments proven clear failed its final check: \
             {verdict:?}"
        )));
    };
    Ok(Ok(PathPlan {
        waypoints: path,
        raw_waypoints,
        iterations,
        tree_nodes,
        clearance,
        closest,
    }))
}

/// Replaces stretches of `path` by straight segments proven clear.
fn shortcut(
    checker: &MotionChecker<'_>,
    mut path: Vec<Vec<f64>>,
    options: &RrtOptions,
    rng: &mut Rng,
) -> DomainResult<Vec<Vec<f64>>> {
    for _ in 0..options.shortcut_attempts {
        if path.len() < 3 {
            break;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let pick = |rng: &mut Rng, n: usize| ((rng.next_unit() * n as f64) as usize).min(n - 1);
        let mut i = pick(rng, path.len());
        let mut j = pick(rng, path.len());
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        if j < i + 2 {
            continue;
        }
        if checker.check_segment(&path[i], &path[j])?.is_clear() {
            path.drain(i + 1..j);
        }
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_collision::SceneOptions;
    use axisforge_domain::{
        templates, GeometryInstance, GeometryRole, Shape, Transform, Vec3, WorldBody,
    };

    #[test]
    fn a_clear_straight_line_is_returned_as_it_is() {
        let project = templates::pick_and_place_cell().unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let checker =
            MotionChecker::new(&project, &model, &SceneOptions::default(), 0.001).unwrap();
        let start = [0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0];
        let mut goal = start;
        goal[0] = 0.8;
        let plan = plan_path(
            &checker,
            &model,
            &start,
            &goal,
            &RrtOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(plan.waypoints.len(), 2);
        assert_eq!(plan.iterations, 0);
    }

    #[test]
    fn a_wall_is_gone_round_and_every_segment_is_clear() {
        let mut project = templates::pick_and_place_cell().unwrap();
        // A wall across the table between the two sides of a waist swing.
        project.environment.bodies.push(WorldBody::obstacle(
            "wall",
            "wall",
            Transform::from_translation(Vec3::new(0.5, 0.0, 0.42 + 0.2)),
            GeometryInstance::visual(
                "wall-g",
                "wall",
                Shape::Box {
                    size: Vec3::new(0.3, 0.02, 0.4),
                },
            )
            .with_role(GeometryRole::Both),
        ));
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let checker =
            MotionChecker::new(&project, &model, &SceneOptions::default(), 0.001).unwrap();
        let high = [0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0];
        let mut start = high;
        start[0] = -0.5;
        let mut goal = high;
        goal[0] = 0.5;
        assert!(checker.is_clear(&start).unwrap() && checker.is_clear(&goal).unwrap());
        assert!(!checker.check_segment(&start, &goal).unwrap().is_clear());
        let options = RrtOptions::default();
        let plan = plan_path(
            &checker,
            &model,
            &start,
            &goal,
            &options,
            &CancellationToken::new(),
        )
        .unwrap()
        .unwrap();
        assert!(plan.waypoints.len() >= 3);
        assert!(plan.waypoints.len() <= plan.raw_waypoints);
        assert_eq!(plan.waypoints.first().unwrap(), &start.to_vec());
        assert_eq!(plan.waypoints.last().unwrap(), &goal.to_vec());
        let (failed, verdict) = checker.check_path(&plan.waypoints).unwrap();
        assert!(failed.is_none() && verdict.is_clear(), "{verdict:?}");
        // Deterministic.
        let again = plan_path(
            &checker,
            &model,
            &start,
            &goal,
            &options,
            &CancellationToken::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(plan, again);
    }

    #[test]
    fn blocked_ends_and_moving_jaws_are_reported() {
        let project = templates::pick_and_place_cell().unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let checker =
            MotionChecker::new(&project, &model, &SceneOptions::default(), 0.001).unwrap();
        let free = [0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0];
        let into_table = [0.0, 0.9, 1.0, 0.0, 1.2, 0.0, 0.0, 0.0];
        let cancel = CancellationToken::new();
        let options = RrtOptions::default();
        assert!(matches!(
            plan_path(&checker, &model, &free, &into_table, &options, &cancel).unwrap(),
            Err(PathFailure::GoalBlocked { .. })
        ));
        assert!(matches!(
            plan_path(&checker, &model, &into_table, &free, &options, &cancel).unwrap(),
            Err(PathFailure::StartBlocked { .. })
        ));
        let mut open = free;
        open[6] = 0.02;
        open[7] = 0.02;
        assert!(plan_path(&checker, &model, &free, &open, &options, &cancel).is_err());
    }
}
