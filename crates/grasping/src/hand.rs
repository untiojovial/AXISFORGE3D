//! Grasps with a hand: fingers that close until they touch.
//!
//! A hand ([`Gripper::is_hand`]) is not driven to an opening; each of its
//! fingers closes until it meets the part. Grasps are found the way the
//! `GraspStudio` generic planner of Simox finds them: draw a point on the
//! part's surface, put the hand's grasp centre there with the approach along
//! the inward normal and a turn about it, back the hand away until it is
//! clear of the part, close the fingers, and judge the contacts. More
//! placements are tried for each point, because they are often the better
//! grasp: the hand pushed in until something of it is about to touch; the
//! approach line through the part's centre, which is where a centring
//! gripper would put the part; and the approach tilted [`TILT`] off the
//! normal, toward the side the turn faces, because an arm often cannot come
//! in exactly along a normal (straight down onto a part near its base, for
//! one).
//!
//! # Closing a finger
//!
//! Each finger is an actor ([`axisforge_domain::Actor`]): its joints advance
//! together in their closing directions. A joint stops when any link it
//! moves comes within the contact tolerance of the part, or at its limit;
//! the joints beyond it carry on, so a finger blocked at its base curls its
//! tip around the part. The fingers advance by conservative advancement:
//! each step is as long as the travel bound of
//! [`axisforge_collision::MotionBounds`] allows without any finger link
//! reaching the part, so a finger never passes through a thin part between
//! two steps, whatever the step.
//!
//! # Contacts
//!
//! A link touching the part contributes one contact per pair of convex
//! pieces within the tolerance: the closest point on the part, with the
//! direction from the finger to the part as the inward normal. This is the
//! contact model `GraspStudio` uses with a hand. It is a point contact; a
//! finger lying flat on a face touches it along a patch, which this model
//! represents by one point, so it underestimates the torque such a finger
//! can resist. The part is assumed not to move while the fingers close.

use std::collections::HashSet;

use axisforge_collision::MotionBounds;
use axisforge_domain::{
    DomainError, DomainResult, GeometryInstance, Gripper, Project, Quat, Shape, Transform, Vec3,
    WorldBody,
};
use axisforge_geometry::{body_from_geometry, gjk, ConvexBody};
use axisforge_kinematics::{forward_kinematics_unchecked, frame_pose, KinematicModel};
use serde::{Deserialize, Serialize};

use crate::candidate::GraspCandidate;
use crate::surface::PartSurface;
use crate::wrench::ContactPoint;

/// The name hand grasps are reported under.
pub const HAND_GENERATOR: &str = "hand_surface_normal";

/// Most advancement steps one closing may take.
const MAX_CLOSING_STEPS: usize = 300;
/// How far the hand backs away per step while it overlaps the part, metres.
const RETREAT_STEP: f64 = 0.002;
/// Most steps backing away: 20 cm.
const MAX_RETREAT_STEPS: usize = 100;
/// How far a tilted approach leans off the surface normal, radians (25°).
pub const TILT: f64 = 0.436_332_312_998_582_4;
/// Precision of the distances measured while closing, metres. They are
/// used less this much, so the advancement stays conservative.
const DISTANCE_TOLERANCE: f64 = 1.0e-6;
/// Gap left between the hand and the part when it is placed, metres.
pub const HAND_CLEARANCE: f64 = 0.002;

/// How a hand holds a grasp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HandGrasp {
    /// The preshape the hand approached in.
    pub preshape: String,
    /// Finger joint values approaching, in the order of the gripper's
    /// `jawJointIds`.
    pub open: Vec<f64>,
    /// Finger joint values where the fingers stopped on the part, same order.
    pub closed: Vec<f64>,
    /// How many fingers touch the part.
    pub fingers_in_contact: usize,
}

/// What a hand synthesis needs besides the part.
#[derive(Debug, Clone, PartialEq)]
pub struct HandSynthesis {
    /// Points drawn on the part's surface.
    pub surface_samples: usize,
    /// Turns of the hand about the approach tried at each point, each
    /// straight along the normal and tilted.
    pub turns: usize,
    /// A finger link this close to the part touches it, metres.
    pub contact_tolerance: f64,
    /// Seed of the surface sampling.
    pub seed: u64,
    /// The preshape to approach in; the gripper's first when `None`.
    pub preshape: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct FingerJoint {
    value_index: usize,
    direction: f64,
    lower: f64,
    upper: f64,
}

impl FingerJoint {
    /// How far the joint can still close, in units of its closing rate.
    fn remaining(&self, value: f64) -> f64 {
        let room = if self.direction > 0.0 {
            self.upper - value
        } else {
            value - self.lower
        };
        room.max(0.0) / self.direction.abs()
    }
}

struct HandLink<'a> {
    index: usize,
    geometries: Vec<(&'a GeometryInstance, Option<&'a [f64]>)>,
    /// The actor moving it, for a finger link.
    actor: Option<usize>,
    /// Finger joints that move it, as indices into the hand's joints.
    movers: Vec<usize>,
}

/// A hand, ready to be placed and closed around parts.
pub struct Hand<'a> {
    model: &'a KinematicModel,
    bounds: MotionBounds,
    frame: usize,
    base: Vec<f64>,
    joints: Vec<FingerJoint>,
    actor_count: usize,
    links: Vec<HandLink<'a>>,
}

/// Where the fingers stopped and what they touch.
#[derive(Debug, Clone, PartialEq)]
pub struct Closing {
    /// Finger joint values, in the order of the gripper's `jawJointIds`.
    pub values: Vec<f64>,
    /// Contacts on the part, outward normals.
    pub contacts: Vec<ContactPoint>,
    /// How many fingers touch the part.
    pub fingers_in_contact: usize,
}

/// A hand link's convex pieces at `pose`.
fn link_bodies(link: &HandLink<'_>, pose: &Transform) -> Vec<ConvexBody> {
    link.geometries
        .iter()
        .filter_map(|(g, mesh)| body_from_geometry(g, pose, *mesh).ok())
        .collect()
}

fn mesh_vertices<'a>(project: &'a Project, geometry: &GeometryInstance) -> Option<&'a [f64]> {
    match &geometry.shape {
        Shape::Mesh { resource_id, .. } => project
            .resource(resource_id)
            .map(|resource| resource.vertices.as_slice()),
        _ => None,
    }
}

/// The convex pieces of a part's collision geometry, where it stands.
///
/// # Errors
/// When a geometry cannot be turned into a convex body.
pub fn part_bodies(project: &Project, body: &WorldBody) -> DomainResult<Vec<ConvexBody>> {
    body.geometries
        .iter()
        .filter(|g| g.is_collision())
        .map(|g| body_from_geometry(g, &body.pose, mesh_vertices(project, g)))
        .collect()
}

impl<'a> Hand<'a> {
    /// Describes the project's hand for placing and closing.
    ///
    /// # Errors
    /// When the robot has no hand, or its links or joints are not in `model`.
    pub fn new(project: &'a Project, model: &'a KinematicModel) -> DomainResult<Self> {
        let gripper = project.robot.require_gripper()?;
        if !gripper.is_hand() {
            return Err(DomainError::unsupported(format!(
                "gripper '{}' has no actors; it closes to an opening, not until contact",
                gripper.name
            )));
        }
        let frame = model.frame_index(&gripper.grasp_frame_id).ok_or_else(|| {
            DomainError::dangling("tool frame", gripper.grasp_frame_id.clone(), "hand")
        })?;
        let palm = model
            .link_index(&gripper.palm_link_id)
            .ok_or_else(|| DomainError::dangling("link", gripper.palm_link_id.clone(), "hand"))?;
        let limits = model.limits();
        let mut joints = Vec::new();
        let mut owner = Vec::new();
        for joint_id in &gripper.jaw_joint_ids {
            let value_index = model
                .joint_index(joint_id)
                .and_then(|j| model.joints()[j].value_index)
                .ok_or_else(|| DomainError::dangling("joint", joint_id.clone(), "hand"))?;
            let direction = gripper.closing_direction(joint_id).unwrap_or(1.0);
            let actor = gripper
                .actors
                .iter()
                .position(|a| a.joints.iter().any(|j| &j.joint_id == joint_id));
            joints.push(FingerJoint {
                value_index,
                direction,
                lower: limits[value_index].lower,
                upper: limits[value_index].upper,
            });
            owner.push(actor);
        }
        let bounds = MotionBounds::new(project, model)?;

        // The hand is the palm and everything mounted below it.
        let below_palm = |mut link: usize| loop {
            if link == palm {
                return true;
            }
            match model.links()[link].parent_joint {
                Some(joint) => link = model.joints()[joint].parent_link,
                None => return false,
            }
        };
        let mut links = Vec::new();
        for link in &project.robot.links {
            let Some(index) = model.link_index(&link.id) else {
                continue;
            };
            if !below_palm(index) {
                continue;
            }
            let geometries: Vec<_> = link
                .collision_geometries()
                .map(|g| (g, mesh_vertices(project, g)))
                .collect();
            if geometries.is_empty() {
                continue;
            }
            let movers: Vec<usize> = (0..joints.len())
                .filter(|j| bounds.moves(index, joints[*j].value_index))
                .collect();
            let actor = movers.iter().find_map(|j| owner[*j]);
            links.push(HandLink {
                index,
                geometries,
                actor,
                movers,
            });
        }

        let mut base = model.neutral_configuration();
        for joint in &joints {
            base[joint.value_index] = joint.lower;
        }
        Ok(Self {
            model,
            bounds,
            frame,
            base,
            joints,
            actor_count: gripper.actors.len(),
            links,
        })
    }

    /// Finger joint values of a preshape, in `jawJointIds` order; a joint
    /// the preshape leaves out is at its open limit.
    ///
    /// # Errors
    /// When the gripper has no preshape by that name.
    pub fn preshape(
        &self,
        gripper: &Gripper,
        name: Option<&str>,
    ) -> DomainResult<(String, Vec<f64>)> {
        let preshape = gripper.preshape(name);
        if name.is_some() && preshape.is_none() {
            return Err(DomainError::unsupported(format!(
                "gripper '{}' has no preshape '{}'",
                gripper.name,
                name.unwrap_or_default()
            )));
        }
        let values = gripper
            .jaw_joint_ids
            .iter()
            .zip(&self.joints)
            .map(|(id, joint)| {
                preshape
                    .and_then(|p| p.joint_values.get(id).copied())
                    .unwrap_or(if joint.direction > 0.0 {
                        joint.lower
                    } else {
                        joint.upper
                    })
            })
            .collect();
        Ok((
            preshape.map_or_else(|| "open limits".to_owned(), |p| p.name.clone()),
            values,
        ))
    }

    /// World poses of the hand's links with the grasp frame at `placement`.
    fn link_poses(&self, placement: &Transform, values: &[f64]) -> Vec<Transform> {
        let mut q = self.base.clone();
        for (joint, value) in self.joints.iter().zip(values) {
            q[joint.value_index] = *value;
        }
        let pose = forward_kinematics_unchecked(self.model, &q);
        let grasp = frame_pose(self.model, &pose, self.frame).inverse();
        self.links
            .iter()
            .map(|link| placement.compose(&grasp.compose(&pose.link(link.index))))
            .collect()
    }

    /// A lower bound on the distance from a link to the part, within
    /// [`DISTANCE_TOLERANCE`] of the true one; `0.0` when they overlap.
    fn distance(link: &HandLink<'_>, pose: &Transform, part: &[ConvexBody]) -> f64 {
        let mut best = f64::INFINITY;
        for body in link_bodies(link, pose) {
            for piece in part {
                let measured = gjk::distance_to_tolerance(&body, piece, DISTANCE_TOLERANCE);
                let bound = if measured.distance > 0.0 {
                    (measured.distance - DISTANCE_TOLERANCE).max(0.0)
                } else {
                    0.0
                };
                best = best.min(bound);
            }
        }
        best
    }

    /// Smallest distance from the whole hand, at `values`, to the part.
    pub fn clearance(&self, placement: &Transform, values: &[f64], part: &[ConvexBody]) -> f64 {
        let poses = self.link_poses(placement, values);
        self.links
            .iter()
            .zip(&poses)
            .map(|(link, pose)| Self::distance(link, pose, part))
            .fold(f64::INFINITY, f64::min)
    }

    /// Puts the grasp frame at `placement` and moves the hand along
    /// `approach` until it is clear of the part by [`HAND_CLEARANCE`]: back
    /// when it overlaps or is closer, and, when `deep`, forward when it is
    /// further. `None` when no position within 20 cm is clear.
    pub fn place(
        &self,
        placement: Transform,
        approach: &Vec3,
        values: &[f64],
        part: &[ConvexBody],
        deep: bool,
    ) -> Option<Transform> {
        let mut placement = placement;
        let mut retreats = 0;
        let mut advances = 0;
        loop {
            let gap = self.clearance(&placement, values, part);
            if gap < HAND_CLEARANCE {
                if retreats >= MAX_RETREAT_STEPS {
                    return None;
                }
                retreats += 1;
                placement.translation = placement.translation.sub(&approach.scaled(RETREAT_STEP));
                continue;
            }
            // Moving the hand by s moves every point of it by s, so it can
            // advance by the gap less the clearance without touching.
            let room = gap - HAND_CLEARANCE;
            if deep && retreats == 0 && room > 2.0e-4 && room.is_finite() && advances < 20 {
                advances += 1;
                placement.translation = placement.translation.add(&approach.scaled(room));
                continue;
            }
            return Some(placement);
        }
    }

    /// Closes every finger from `start` until it touches the part or its
    /// joints reach their limits. `None` when a finger overlaps the part
    /// before it moves.
    pub fn close(
        &self,
        placement: &Transform,
        start: &[f64],
        part: &[ConvexBody],
        tolerance: f64,
    ) -> Option<Closing> {
        let tolerance = tolerance.max(1.0e-6);
        let mut q = start.to_vec();
        let mut active: Vec<bool> = self
            .joints
            .iter()
            .zip(&q)
            .map(|(j, v)| j.remaining(*v) > 0.0)
            .collect();
        let mut touching = vec![false; self.links.len()];
        let fingers: Vec<usize> = (0..self.links.len())
            .filter(|k| self.links[*k].actor.is_some())
            .collect();
        for step_index in 0..MAX_CLOSING_STEPS {
            let poses = self.link_poses(placement, &q);
            let mut distance = vec![f64::INFINITY; self.links.len()];
            for &k in &fingers {
                // A link nothing moves any more keeps its distance.
                if touching[k] || !self.links[k].movers.iter().any(|m| active[*m]) {
                    continue;
                }
                let d = Self::distance(&self.links[k], &poses[k], part);
                if d <= 0.0 && step_index == 0 {
                    return None;
                }
                distance[k] = d;
                if d <= tolerance {
                    touching[k] = true;
                    for &m in &self.links[k].movers {
                        active[m] = false;
                    }
                }
            }
            if !active.iter().any(|a| *a) {
                break;
            }
            let mut rate = vec![0.0; self.base.len()];
            for (joint, on) in self.joints.iter().zip(&active) {
                if *on {
                    rate[joint.value_index] = joint.direction.abs();
                }
            }
            let mut step = f64::INFINITY;
            for &k in &fingers {
                if touching[k] || !distance[k].is_finite() {
                    continue;
                }
                let index = self.links[k].index;
                let speed = self
                    .bounds
                    .rate(index, self.bounds.link_radius(index), &rate);
                if speed > 0.0 {
                    step = step.min((distance[k] - 0.5 * tolerance) / speed);
                }
            }
            for (joint, (on, value)) in self.joints.iter().zip(active.iter().zip(&q)) {
                if *on {
                    step = step.min(joint.remaining(*value));
                }
            }
            if !step.is_finite() {
                break;
            }
            let step = step.max(0.0);
            for (index, joint) in self.joints.iter().enumerate() {
                if active[index] {
                    let target = q[index] + step * joint.direction;
                    q[index] = target.clamp(joint.lower, joint.upper);
                    if joint.remaining(q[index]) <= 1.0e-12 {
                        active[index] = false;
                    }
                }
            }
        }

        Some(self.contacts(placement, q, part, tolerance))
    }

    /// The contacts of the fingers at `values` with the part.
    fn contacts(
        &self,
        placement: &Transform,
        values: Vec<f64>,
        part: &[ConvexBody],
        tolerance: f64,
    ) -> Closing {
        let fingers = (0..self.links.len()).filter(|k| self.links[*k].actor.is_some());
        let poses = self.link_poses(placement, &values);
        let mut contacts: Vec<ContactPoint> = Vec::new();
        let mut actors = HashSet::new();
        for k in fingers {
            let link = &self.links[k];
            for body in link_bodies(link, &poses[k]) {
                for piece in part {
                    let result = gjk::distance(&body, piece);
                    if result.distance > tolerance {
                        continue;
                    }
                    let Some(direction) = result.direction else {
                        continue;
                    };
                    if let Some(actor) = link.actor {
                        actors.insert(actor);
                    }
                    let point = result.witness_b;
                    if contacts.iter().any(|c| c.point.distance(&point) < 1.0e-6) {
                        continue;
                    }
                    contacts.push(ContactPoint {
                        point,
                        normal: direction.scaled(-1.0),
                    });
                }
            }
        }
        debug_assert!(actors.len() <= self.actor_count);
        Closing {
            values,
            contacts,
            fingers_in_contact: actors.len(),
        }
    }
}

/// The two contacts whose normals are most nearly opposed.
fn most_opposed(contacts: &[ContactPoint]) -> Option<(usize, usize)> {
    let mut best = None;
    let mut lowest = f64::INFINITY;
    for i in 0..contacts.len() {
        for j in i + 1..contacts.len() {
            let dot = contacts[i].normal.dot(&contacts[j].normal);
            if dot < lowest {
                lowest = dot;
                best = Some((i, j));
            }
        }
    }
    best
}

/// A key identifying a contact set to 0.1 mm, so placements that end in the
/// same grasp are proposed once.
fn closing_key(closing: &Closing) -> String {
    let mut points: Vec<String> = closing
        .contacts
        .iter()
        .map(|c| {
            let r = |v: f64| (v * 1.0e4).round();
            format!("{},{},{}", r(c.point.x), r(c.point.y), r(c.point.z))
        })
        .collect();
    points.sort();
    points.join(";")
}

/// Proposes hand grasps of `body`, in world coordinates.
///
/// `center_of_mass` is the part's, for centring the approach line.
///
/// # Errors
/// When the robot has no hand, the preshape is unknown, or the part's
/// geometry cannot be read.
pub fn synthesize_hand(
    project: &Project,
    model: &KinematicModel,
    body: &WorldBody,
    center_of_mass: Vec3,
    options: &HandSynthesis,
) -> DomainResult<Vec<GraspCandidate>> {
    let gripper = project.robot.require_gripper()?;
    let hand = Hand::new(project, model)?;
    let (preshape_name, open) = hand.preshape(gripper, options.preshape.as_deref())?;
    let surface = PartSurface::of(body, &project.resources)?;
    let part = part_bodies(project, body)?;
    let local = Quat::from_basis(
        &gripper.closing_axis,
        &gripper.approach_axis,
        &gripper.closing_axis.cross(&gripper.approach_axis),
    )?
    .conjugate();
    let turns = options.turns.max(1);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (index, sample) in surface
        .sample(options.surface_samples, options.seed)
        .iter()
        .enumerate()
    {
        let normal_approach = sample.normal.scaled(-1.0);
        let Ok(first) = normal_approach.any_perpendicular("approach") else {
            continue;
        };
        for (turn, (lean, tilt)) in
            (0..turns).flat_map(|turn| [(turn, ("straight", 0.0)), (turn, ("tilted", TILT))])
        {
            #[allow(clippy::cast_precision_loss)]
            let angle = std::f64::consts::TAU * turn as f64 / turns as f64;
            let closing = Quat::from_axis_angle(&normal_approach, angle)?.rotate(&first);
            let approach = Quat::from_axis_angle(&closing, tilt)?.rotate(&normal_approach);
            let rotation =
                Quat::from_basis(&closing, &approach, &closing.cross(&approach))?.mul(&local);
            // The approach line through the sample, and the parallel one
            // through the centre of mass.
            let centred = center_of_mass
                .add(&approach.scaled(approach.dot(&sample.point.sub(&center_of_mass))));
            for (line, through) in [("at", sample.point), ("centred", centred)] {
                for (depth, deep) in [("surface", false), ("deep", true)] {
                    let Some(placement) = hand.place(
                        Transform::new(through, rotation),
                        &approach,
                        &open,
                        &part,
                        deep,
                    ) else {
                        continue;
                    };
                    let Some(closing_result) =
                        hand.close(&placement, &open, &part, options.contact_tolerance)
                    else {
                        continue;
                    };
                    if closing_result.fingers_in_contact < 2 {
                        continue;
                    }
                    if !seen.insert(closing_key(&closing_result)) {
                        continue;
                    }
                    let contacts = closing_result.contacts;
                    let Some((a, b)) = most_opposed(&contacts) else {
                        continue;
                    };
                    let width = contacts
                        .iter()
                        .flat_map(|c| contacts.iter().map(move |d| c.point.distance(&d.point)))
                        .fold(0.0, f64::max);
                    let center = placement.translation;
                    let off = center_of_mass.sub(&center);
                    let center_offset = off.sub(&approach.scaled(approach.dot(&off))).norm();
                    out.push(GraspCandidate {
                        label: format!("hand {index} {line} {depth} {lean} {turn}"),
                        center,
                        approach,
                        closing,
                        width,
                        normal_a: contacts[a].normal,
                        normal_b: contacts[b].normal,
                        center_offset,
                        contacts,
                        hand: Some(HandGrasp {
                            preshape: preshape_name.clone(),
                            open: open.clone(),
                            closed: closing_result.values,
                            fingers_in_contact: closing_result.fingers_in_contact,
                        }),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates::{self, GripperKind};

    fn cell(kind: GripperKind) -> (Project, KinematicModel) {
        let project = templates::pick_and_place_cell_with(kind).unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        (project, model)
    }

    fn top_down(center: Vec3, gripper: &Gripper) -> Transform {
        let approach = Vec3::new(0.0, 0.0, -1.0);
        let closing = Vec3::X;
        let local = Quat::from_basis(
            &gripper.closing_axis,
            &gripper.approach_axis,
            &gripper.closing_axis.cross(&gripper.approach_axis),
        )
        .unwrap()
        .conjugate();
        Transform::new(
            center,
            Quat::from_basis(&closing, &approach, &closing.cross(&approach))
                .unwrap()
                .mul(&local),
        )
    }

    #[test]
    fn three_jaws_close_on_a_cube_from_above_and_stop_at_its_faces() {
        let (project, model) = cell(GripperKind::ThreeFinger);
        let gripper = project.robot.gripper.as_ref().unwrap();
        let hand = Hand::new(&project, &model).unwrap();
        let cube = project.environment.body_by_name("cube").unwrap();
        let part = part_bodies(&project, cube).unwrap();
        let (_, open) = hand.preshape(gripper, None).unwrap();
        let top = cube.pose.translation.add(&Vec3::new(0.0, 0.0, 0.02));
        let placement = hand
            .place(
                top_down(top, gripper),
                &Vec3::new(0.0, 0.0, -1.0),
                &open,
                &part,
                false,
            )
            .unwrap();
        assert!(hand.clearance(&placement, &open, &part) >= HAND_CLEARANCE);
        let closing = hand.close(&placement, &open, &part, 3.0e-4).unwrap();
        assert_eq!(closing.fingers_in_contact, 3);
        // Every jaw face stops within the tolerance of the cube, no further in.
        for contact in &closing.contacts {
            let local = contact.point.sub(&cube.pose.translation);
            assert!(
                local.x.abs().max(local.y.abs()) > 0.02 - 1e-6,
                "a contact on a side face: {local:?}"
            );
        }
        for (value, open) in closing.values.iter().zip(&open) {
            assert!(value < open, "every jaw closed: {value} < {open}");
            // Faces 20 to 28 mm from the axis, depending on where they meet.
            assert!(*value > 0.0195 && *value < 0.0290, "{value}");
        }
    }

    #[test]
    fn a_hand_finger_curls_and_never_passes_through_the_part() {
        let (project, model) = cell(GripperKind::Hand);
        let gripper = project.robot.gripper.as_ref().unwrap();
        let hand = Hand::new(&project, &model).unwrap();
        let dowel = project.environment.body_by_name("dowel").unwrap();
        let part = part_bodies(&project, dowel).unwrap();
        let (name, open) = hand.preshape(gripper, None).unwrap();
        assert_eq!(name, "open");
        let top = dowel.pose.translation.add(&Vec3::new(0.0, 0.0, 0.045));
        let approach = Vec3::new(0.0, 0.0, -1.0);
        let placement = hand
            .place(top_down(top, gripper), &approach, &open, &part, true)
            .unwrap();
        let closing = hand.close(&placement, &open, &part, 3.0e-4).unwrap();
        assert_eq!(closing.fingers_in_contact, 3, "{closing:?}");
        // Nothing of the closed hand overlaps the part.
        assert!(hand.clearance(&placement, &closing.values, &part) > 0.0);
        for contact in &closing.contacts {
            // Contacts on the dowel's round wall point outward, horizontally.
            let radial = contact.point.sub(&dowel.pose.translation);
            let radial = Vec3::new(radial.x, radial.y, 0.0);
            if (radial.norm() - 0.015).abs() < 1e-3 {
                assert!(contact.normal.dot(&radial.scaled(1.0 / radial.norm())) > 0.95);
            }
        }
    }

    #[test]
    fn a_finger_with_nothing_to_touch_closes_to_its_limit() {
        let (project, model) = cell(GripperKind::ThreeFinger);
        let gripper = project.robot.gripper.as_ref().unwrap();
        let hand = Hand::new(&project, &model).unwrap();
        let cube = project.environment.body_by_name("cube").unwrap();
        let part = part_bodies(&project, cube).unwrap();
        let (_, open) = hand.preshape(gripper, None).unwrap();
        // Far above the cube: no finger reaches it.
        let high = cube.pose.translation.add(&Vec3::new(0.0, 0.0, 0.3));
        let closing = hand
            .close(&top_down(high, gripper), &open, &part, 3.0e-4)
            .unwrap();
        assert_eq!(closing.fingers_in_contact, 0);
        assert!(closing.contacts.is_empty());
        for value in &closing.values {
            assert!((value - 0.006).abs() < 1e-9, "closed fully: {value}");
        }
    }

    #[test]
    fn synthesis_finds_distinct_grasps_with_at_least_two_fingers() {
        for kind in [GripperKind::ThreeFinger, GripperKind::Hand] {
            let (project, model) = cell(kind);
            let cube = project.environment.body_by_name("cube").unwrap();
            let options = HandSynthesis {
                surface_samples: 12,
                turns: 2,
                contact_tolerance: 3.0e-4,
                seed: 1,
                preshape: None,
            };
            let candidates =
                synthesize_hand(&project, &model, cube, cube.pose.translation, &options).unwrap();
            assert!(!candidates.is_empty(), "{kind:?}");
            let mut keys = HashSet::new();
            for c in &candidates {
                let hand = c.hand.as_ref().unwrap();
                assert!(hand.fingers_in_contact >= 2);
                assert!(c.contacts.len() >= 2);
                assert!((c.approach.norm() - 1.0).abs() < 1e-9);
                assert!(c.approach.dot(&c.closing).abs() < 1e-9);
                assert!(keys.insert(c.label.clone()));
            }
            let again =
                synthesize_hand(&project, &model, cube, cube.pose.translation, &options).unwrap();
            assert_eq!(candidates, again, "deterministic");
        }
    }

    #[test]
    fn an_unknown_preshape_and_a_parallel_gripper_are_refused() {
        let (project, model) = cell(GripperKind::Hand);
        let cube = project.environment.body_by_name("cube").unwrap();
        let options = HandSynthesis {
            surface_samples: 4,
            turns: 1,
            contact_tolerance: 3.0e-4,
            seed: 1,
            preshape: Some("fist".to_owned()),
        };
        assert!(synthesize_hand(&project, &model, cube, Vec3::ZERO, &options).is_err());
        let (project, model) = cell(GripperKind::Parallel);
        assert!(Hand::new(&project, &model).is_err());
    }
}
