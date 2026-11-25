//! Collision checking along a motion, exactly rather than by samples.
//!
//! A straight joint-space segment from `a` to `b` is proven clear pair by
//! pair, with the test of Schwarzer, Saha and Latombe ("Adaptive dynamic
//! collision checking for single and multiple articulated robots in complex
//! environments", IEEE T-RO 21(3), 2005). If the two owners of a pair are
//! `d₀` apart at one end of a piece of the segment and `d₁` at the other,
//! and no point of either moves more than `λ` in between (the bound of
//! [`MotionBounds`]), then at every instant they are at least
//! `(d₀ + d₁ − λ) / 2` apart. When that is more than the margin, the piece is
//! clear; otherwise it is halved and each half tested again. Pairs far apart
//! are settled at once and only those that come close are subdivided.
//!
//! The verdict is one of three: clear, with a certified lower bound on the
//! clearance over the whole segment; in contact, at a configuration where a
//! pair was measured within the margin; or unproven, when halving
//! [`MAX_DEPTH`] times did not settle a pair — something grazes the margin —
//! which a caller must treat as a contact.
//!
//! Distances are GJK distances between convex pieces (a mesh is its convex
//! hull, or the hulls of its decomposition), less their measuring tolerance,
//! so every number used is a lower bound.

use std::cell::RefCell;
use std::collections::HashMap;

use axisforge_collision::{MotionBounds, OwnerKind, Scene, SceneOptions};
use axisforge_domain::{DomainError, DomainResult, GeometryInstance, Project, Shape, Transform};
use axisforge_geometry::{body_from_geometry, gjk, ConvexBody};
use axisforge_kinematics::{
    forward_kinematics, forward_kinematics_unchecked, KinematicModel, Pose,
};
use serde::{Deserialize, Serialize};

/// Most times a piece of a segment is halved before a pair is called unproven.
pub const MAX_DEPTH: usize = 24;
/// Precision of each distance measurement, metres; subtracted from it.
const DISTANCE_TOLERANCE: f64 = 1.0e-6;

/// What checking a segment concluded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "verdict",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum SegmentVerdict {
    /// Nothing comes within the margin anywhere along the segment.
    Clear {
        /// A certified lower bound on the smallest distance between any
        /// checked pair over the whole segment, metres.
        clearance: f64,
    },
    /// A pair was measured within the margin.
    Contact {
        /// Display name of one owner.
        first: String,
        /// Display name of the other.
        second: String,
        /// Where along the segment, from 0 at the start to 1 at the end.
        at: f64,
        /// The distance measured there, metres.
        distance: f64,
    },
    /// A pair came so close to the margin that the subdivision limit was
    /// reached without settling it. Treat it as a contact.
    Unproven {
        /// Display name of one owner.
        first: String,
        /// Display name of the other.
        second: String,
        /// Where along the segment the last undecided piece starts.
        at: f64,
    },
}

impl SegmentVerdict {
    /// `true` for a segment proven clear.
    pub fn is_clear(&self) -> bool {
        matches!(self, Self::Clear { .. })
    }
}

/// The closest pair at one configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Closest {
    /// Display name of one owner.
    pub first: String,
    /// Display name of the other.
    pub second: String,
    /// Lower bound on their distance, metres.
    pub distance: f64,
}

#[derive(Debug, Clone)]
struct Owner {
    name: String,
    /// The link that carries it, for a link or a carried part.
    link: Option<usize>,
    /// Largest distance from that link's origin to the owner's geometry.
    radius: f64,
}

#[derive(Debug, Clone)]
struct Piece {
    owner: usize,
    /// Pose in the carrying link's frame, for a moving piece.
    local: Transform,
    body: ConvexBody,
}

#[derive(Debug, Clone)]
struct OwnerPair {
    first: usize,
    second: usize,
    pieces: Vec<(usize, usize)>,
}

/// Checks configurations and straight joint-space segments for one scene
/// setup: what is carried, and what may touch.
pub struct MotionChecker<'a> {
    model: &'a KinematicModel,
    bounds: MotionBounds,
    owners: Vec<Owner>,
    pieces: RefCell<Vec<Piece>>,
    pairs: Vec<OwnerPair>,
    margin: f64,
}

fn mesh_vertices<'a>(project: &'a Project, geometry: &GeometryInstance) -> Option<&'a [f64]> {
    match &geometry.shape {
        Shape::Mesh { resource_id, .. } => project
            .resource(resource_id)
            .map(|resource| resource.vertices.as_slice()),
        _ => None,
    }
}

impl<'a> MotionChecker<'a> {
    /// Prepares checks of `project` with `options` (a carried part, extra
    /// pairs allowed to touch). Pairs closer than `margin` count as contacts.
    ///
    /// # Errors
    /// When the scene cannot be built: an unknown attachment, or geometry
    /// that cannot be made convex.
    #[allow(clippy::too_many_lines)] // One pass over the scene's owners; split, it would only shuffle state.
    pub fn new(
        project: &'a Project,
        model: &'a KinematicModel,
        options: &SceneOptions,
        margin: f64,
    ) -> DomainResult<Self> {
        if !margin.is_finite() || margin < 0.0 {
            return Err(DomainError::NegativeValue {
                context: "collision margin".to_owned(),
                value: margin,
            });
        }
        let start = model.clamped(&vec![0.0; model.dof()]);
        let scene = Scene::build(project, model, &forward_kinematics(model, &start)?, options)?;
        let bounds = MotionBounds::new(project, model)?;

        let mut owners: Vec<Owner> = Vec::new();
        let mut index_of: HashMap<String, usize> = HashMap::new();
        let mut ids: Vec<String> = Vec::new();
        let mut pieces = Vec::new();
        for body in scene.bodies() {
            if index_of.contains_key(&body.owner_id) {
                continue;
            }
            let owner = owners.len();
            index_of.insert(body.owner_id.clone(), owner);
            ids.push(body.owner_id.clone());
            match body.kind {
                OwnerKind::Link => {
                    let link = model.link_index(&body.owner_id).ok_or_else(|| {
                        DomainError::dangling("link", body.owner_id.clone(), "motion checker")
                    })?;
                    let geometry = project
                        .robot
                        .link(&body.owner_id)
                        .into_iter()
                        .flat_map(axisforge_domain::Link::collision_geometries);
                    for g in geometry {
                        let local =
                            body_from_geometry(g, &Transform::IDENTITY, mesh_vertices(project, g))?;
                        pieces.push(Piece {
                            owner,
                            local: local.pose,
                            body: local,
                        });
                    }
                    owners.push(Owner {
                        name: body.owner_name.clone(),
                        link: Some(link),
                        radius: 0.0,
                    });
                }
                OwnerKind::Attached => {
                    let attachment = options.attachment.as_ref().ok_or_else(|| {
                        DomainError::dangling("attachment", body.owner_id.clone(), "motion")
                    })?;
                    let link = model.link_index(&attachment.link_id).ok_or_else(|| {
                        DomainError::dangling("link", attachment.link_id.clone(), "attachment")
                    })?;
                    let world_body = project.environment.body(&body.owner_id).ok_or_else(|| {
                        DomainError::dangling("world body", body.owner_id.clone(), "attachment")
                    })?;
                    for g in world_body.geometries.iter().filter(|g| g.is_collision()) {
                        let local = body_from_geometry(
                            g,
                            &attachment.relative_pose,
                            mesh_vertices(project, g),
                        )?;
                        pieces.push(Piece {
                            owner,
                            local: local.pose,
                            body: local,
                        });
                    }
                    owners.push(Owner {
                        name: body.owner_name.clone(),
                        link: Some(link),
                        radius: 0.0,
                    });
                }
                OwnerKind::World | OwnerKind::Ground => {
                    owners.push(Owner {
                        name: body.owner_name.clone(),
                        link: None,
                        radius: 0.0,
                    });
                }
            }
        }
        // Static pieces are posed once, as the scene has them.
        for body in scene.bodies() {
            if matches!(body.kind, OwnerKind::World | OwnerKind::Ground) {
                pieces.push(Piece {
                    owner: index_of[&body.owner_id],
                    local: body.body.pose,
                    body: body.body.clone(),
                });
            }
        }
        for piece in &pieces {
            let owner = &mut owners[piece.owner];
            if owner.link.is_some() {
                owner.radius = owner
                    .radius
                    .max(piece.local.translation.norm() + piece.body.bounding_radius());
            }
        }

        let mut pairs = Vec::new();
        for first in 0..owners.len() {
            for second in first + 1..owners.len() {
                if owners[first].link.is_none() && owners[second].link.is_none() {
                    continue;
                }
                if scene.is_ignored(&ids[first], &ids[second]) {
                    continue;
                }
                let of = |owner: usize| {
                    pieces
                        .iter()
                        .enumerate()
                        .filter(move |(_, p)| p.owner == owner)
                        .map(|(i, _)| i)
                };
                let combined: Vec<(usize, usize)> = of(first)
                    .flat_map(|a| of(second).map(move |b| (a, b)))
                    .collect();
                if !combined.is_empty() {
                    pairs.push(OwnerPair {
                        first,
                        second,
                        pieces: combined,
                    });
                }
            }
        }
        Ok(Self {
            model,
            bounds,
            owners,
            pieces: RefCell::new(pieces),
            pairs,
            margin,
        })
    }

    /// The margin pairs must keep, metres.
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// How many owner pairs are checked.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Lower bound on the distance of one pair at `pose`.
    fn pair_distance(&self, pair: &OwnerPair, pose: &Pose) -> f64 {
        let mut pieces = self.pieces.borrow_mut();
        for owner in [pair.first, pair.second] {
            if let Some(link) = self.owners[owner].link {
                let at = pose.link(link);
                for piece in pieces.iter_mut().filter(|p| p.owner == owner) {
                    piece.body.pose = at.compose(&piece.local);
                }
            }
        }
        let mut best = f64::INFINITY;
        for (a, b) in &pair.pieces {
            let (a, b) = (&pieces[*a].body, &pieces[*b].body);
            // The box gap is a lower bound too, and much cheaper.
            let gap = a.aabb().distance(&b.aabb());
            if gap >= best {
                continue;
            }
            let measured = gjk::distance_to_tolerance(a, b, DISTANCE_TOLERANCE);
            let bound = if measured.distance > 0.0 {
                (measured.distance - DISTANCE_TOLERANCE).max(0.0)
            } else {
                0.0
            };
            best = best.min(bound);
        }
        best
    }

    /// Upper bound on how far any point of one owner of the pair moves
    /// relative to the other from `a` to `b`.
    ///
    /// Two links of the robot share the joints above their lowest common
    /// ancestor, which move them together and change nothing between them;
    /// only the joints below it count. A link and a static body: every joint.
    fn travel(&self, pair: &OwnerPair, a: &[f64], b: &[f64]) -> f64 {
        let (first, second) = (&self.owners[pair.first], &self.owners[pair.second]);
        match (first.link, second.link) {
            (Some(x), Some(y)) => {
                let common = self.common_ancestor(x, y);
                self.bounds.relative_travel(x, first.radius, a, b, common)
                    + self.bounds.relative_travel(y, second.radius, a, b, common)
            }
            (Some(x), None) => self.bounds.travel(x, first.radius, a, b),
            (None, Some(y)) => self.bounds.travel(y, second.radius, a, b),
            (None, None) => 0.0,
        }
    }

    /// The lowest link both `x` and `y` hang from (either may be it).
    fn common_ancestor(&self, x: usize, y: usize) -> usize {
        let links = self.model.links();
        let parent = |link: usize| {
            links[link]
                .parent_joint
                .map(|joint| self.model.joints()[joint].parent_link)
        };
        let mut above_x = vec![x];
        let mut walk = x;
        while let Some(up) = parent(walk) {
            above_x.push(up);
            walk = up;
        }
        let mut walk = y;
        loop {
            if above_x.contains(&walk) {
                return walk;
            }
            match parent(walk) {
                Some(up) => walk = up,
                None => return self.model.root_link(),
            }
        }
    }

    /// The closest checked pair at a configuration, or `None` when nothing is
    /// checked.
    ///
    /// # Errors
    /// When `values` does not fit the model.
    pub fn closest(&self, values: &[f64]) -> DomainResult<Option<Closest>> {
        let pose = forward_kinematics(self.model, values)?;
        Ok(self
            .pairs
            .iter()
            .map(|pair| (pair, self.pair_distance(pair, &pose)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(pair, distance)| Closest {
                first: self.owners[pair.first].name.clone(),
                second: self.owners[pair.second].name.clone(),
                distance,
            }))
    }

    /// `true` when nothing is within the margin at a configuration.
    ///
    /// # Errors
    /// When `values` does not fit the model.
    pub fn is_clear(&self, values: &[f64]) -> DomainResult<bool> {
        Ok(self
            .closest(values)?
            .is_none_or(|c| c.distance > self.margin))
    }

    /// Checks the straight joint-space segment from `a` to `b`.
    ///
    /// # Errors
    /// When either end does not fit the model.
    pub fn check_segment(&self, a: &[f64], b: &[f64]) -> DomainResult<SegmentVerdict> {
        self.model.check_configuration(a)?;
        self.model.check_configuration(b)?;
        let at = |t: f64| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + (y - x) * t).collect() };
        let mut poses: HashMap<u64, Pose> = HashMap::new();
        let mut pose_at = |t: f64| -> Pose {
            poses
                .entry(t.to_bits())
                .or_insert_with(|| forward_kinematics_unchecked(self.model, &at(t)))
                .clone()
        };
        let start = pose_at(0.0);
        let end = pose_at(1.0);
        let mut clearance = f64::INFINITY;
        for pair in &self.pairs {
            let d0 = self.pair_distance(pair, &start);
            let d1 = self.pair_distance(pair, &end);
            // (t0, t1, d0, d1, depth)
            let mut stack = vec![(0.0_f64, 1.0_f64, d0, d1, 0_usize)];
            while let Some((t0, t1, d0, d1, depth)) = stack.pop() {
                for (t, d) in [(t0, d0), (t1, d1)] {
                    if d <= self.margin {
                        return Ok(SegmentVerdict::Contact {
                            first: self.owners[pair.first].name.clone(),
                            second: self.owners[pair.second].name.clone(),
                            at: t,
                            distance: d,
                        });
                    }
                }
                let lambda = self.travel(pair, &at(t0), &at(t1));
                let certified = 0.5 * (d0 + d1 - lambda);
                if certified > self.margin {
                    clearance = clearance.min(certified);
                    continue;
                }
                if depth >= MAX_DEPTH {
                    return Ok(SegmentVerdict::Unproven {
                        first: self.owners[pair.first].name.clone(),
                        second: self.owners[pair.second].name.clone(),
                        at: t0,
                    });
                }
                let tm = 0.5 * (t0 + t1);
                let dm = self.pair_distance(pair, &pose_at(tm));
                stack.push((tm, t1, dm, d1, depth + 1));
                stack.push((t0, tm, d0, dm, depth + 1));
            }
        }
        Ok(SegmentVerdict::Clear { clearance })
    }

    /// Checks a path of straight segments; the first verdict that is not
    /// clear, with the index of its segment, or the smallest clearance.
    ///
    /// # Errors
    /// When a configuration does not fit the model.
    pub fn check_path(&self, path: &[Vec<f64>]) -> DomainResult<(Option<usize>, SegmentVerdict)> {
        let mut clearance = f64::INFINITY;
        for (index, pair) in path.windows(2).enumerate() {
            match self.check_segment(&pair[0], &pair[1])? {
                SegmentVerdict::Clear { clearance: c } => clearance = clearance.min(c),
                other => return Ok((Some(index), other)),
            }
        }
        Ok((None, SegmentVerdict::Clear { clearance }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates;

    fn cell() -> (Project, KinematicModel) {
        let project = templates::pick_and_place_cell().unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        (project, model)
    }

    /// Gripper 170 mm above the table, well clear of both parts.
    const HIGH: [f64; 8] = [0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0];

    #[test]
    fn a_swing_above_the_table_is_proven_clear() {
        let (project, model) = cell();
        let checker =
            MotionChecker::new(&project, &model, &SceneOptions::default(), 0.001).unwrap();
        let mut other = HIGH;
        other[0] = 1.2;
        other[1] = -0.3;
        let verdict = checker.check_segment(&HIGH, &other).unwrap();
        let SegmentVerdict::Clear { clearance } = verdict else {
            panic!("{verdict:?}");
        };
        // The certified clearance never exceeds what dense sampling measures.
        let sampled = (0..=200)
            .map(|i| {
                let t = f64::from(i) / 200.0;
                let q: Vec<f64> = HIGH
                    .iter()
                    .zip(&other)
                    .map(|(a, b)| a + (b - a) * t)
                    .collect();
                checker.closest(&q).unwrap().unwrap().distance
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            clearance > 0.001 && clearance <= sampled + 1e-9,
            "{clearance} vs {sampled}"
        );
    }

    #[test]
    fn a_motion_through_the_table_is_caught_even_when_both_ends_are_clear() {
        let (project, model) = cell();
        let checker =
            MotionChecker::new(&project, &model, &SceneOptions::default(), 0.001).unwrap();
        // Low, beside the table on either side: both ends are clear, and the
        // waist swing between them passes through the table top.
        let low = [0.0, 0.6, 0.9, 0.0, 1.6, 0.0, 0.0, 0.0];
        let mut swung = low;
        swung[0] = -1.25;
        let mut other = low;
        other[0] = 1.25;
        for q in [&swung, &other] {
            assert!(
                checker.is_clear(q).unwrap(),
                "{:?}",
                checker.closest(q).unwrap()
            );
        }
        assert!(checker.is_clear(&swung).unwrap() && checker.is_clear(&other).unwrap());
        let verdict = checker.check_segment(&swung, &other).unwrap();
        assert!(!verdict.is_clear(), "{verdict:?}");
    }

    #[test]
    fn a_thin_pin_between_two_samples_is_not_missed() {
        // A 3 mm pin standing where the fingers sweep, between two of the
        // configurations a six-step sampled check would look at.
        let (mut project, model) = cell();
        let angle: f64 = 0.1;
        let direction = axisforge_domain::Vec3::new(angle.cos(), angle.sin(), 0.0);
        project
            .environment
            .bodies
            .push(axisforge_domain::WorldBody::obstacle(
                "pin",
                "pin",
                Transform::new(
                    direction
                        .scaled(0.476)
                        .add(&axisforge_domain::Vec3::new(0.0, 0.0, 0.62)),
                    axisforge_domain::Quat::from_axis_angle(&axisforge_domain::Vec3::Z, angle)
                        .unwrap(),
                ),
                GeometryInstance::visual(
                    "pin-geometry",
                    "pin",
                    Shape::Box {
                        size: axisforge_domain::Vec3::new(0.003, 0.003, 0.06),
                    },
                )
                .with_role(axisforge_domain::GeometryRole::Collision),
            ));
        let checker = MotionChecker::new(&project, &model, &SceneOptions::default(), 0.0).unwrap();
        let mut from = HIGH;
        from[0] = 0.6;
        let mut to = HIGH;
        to[0] = -0.6;
        for i in 0..=6 {
            let t = f64::from(i) / 6.0;
            let q: Vec<f64> = from.iter().zip(&to).map(|(a, b)| a + (b - a) * t).collect();
            assert!(checker.is_clear(&q).unwrap(), "sample {i} sees the pin");
        }
        let verdict = checker.check_segment(&from, &to).unwrap();
        let SegmentVerdict::Contact { second, at, .. } = &verdict else {
            panic!("{verdict:?}");
        };
        assert_eq!(second, "pin");
        assert!(*at > 1.0 / 6.0 && *at < 4.0 / 6.0, "{at}");
    }
}
