//! How far a link can travel when the joints move.
//!
//! A bound on travel is what turns collision checking along a motion from
//! sampling into a proof. If no point of a link can move more than `λ`
//! between two configurations, and the link is `d₀` from an obstacle at one
//! end and `d₁` at the other, then `λ < d₀ + d₁` means it never touched the
//! obstacle in between. That is the test of Schwarzer, Saha and Latombe,
//! "Adaptive dynamic collision checking for single and multiple articulated
//! robots in complex environments" (IEEE T-RO 21(3), 2005); this module
//! supplies the `λ`.
//!
//! Along a straight joint-space segment from `a` to `b`, a point `p` of a
//! link moves at most `Σ r_j |b_j − a_j|`, summed over the actuated joints
//! that move the link, where `r_j` is 1 for a prismatic joint and, for a
//! revolute one, any upper bound on the distance from `p` to the joint's axis.
//! The bound used here does not depend on the configuration: the distance
//! from the axis is at most the distance from the joint's origin to the
//! link's origin (the joint offsets along the chain, plus the full travel of
//! any prismatic joint on the way) plus the link geometry's radius about its
//! origin.

use axisforge_domain::{DomainResult, JointKind, Project};
use axisforge_geometry::body_from_geometry;
use axisforge_kinematics::KinematicModel;

/// One actuated joint that moves a link.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mover {
    /// Its slot in a configuration vector.
    value_index: usize,
    /// `true` for a revolute joint, `false` for a prismatic one.
    revolute: bool,
    /// Upper bound on the distance from the joint's axis to the link's origin.
    lever: f64,
}

/// Travel bounds for every link of a robot.
#[derive(Debug, Clone)]
pub struct MotionBounds {
    movers: Vec<Vec<Mover>>,
    radius: Vec<f64>,
}

impl MotionBounds {
    /// Computes the bounds for `project`'s robot.
    ///
    /// # Errors
    /// When a link's collision geometry cannot be turned into a convex body.
    pub fn new(project: &Project, model: &KinematicModel) -> DomainResult<Self> {
        let joints = model.joints();
        let mut movers = Vec::with_capacity(model.links().len());
        for node in model.links() {
            let mut list = Vec::new();
            let mut lever = 0.0;
            let mut joint = node.parent_joint;
            let mut guard = 0;
            while let Some(j) = joint {
                let joint_node = &joints[j];
                if let Some(value_index) = joint_node.value_index {
                    list.push(Mover {
                        value_index,
                        revolute: joint_node.kind == JointKind::Revolute,
                        lever,
                    });
                }
                lever += joint_node.origin.translation.norm();
                if joint_node.kind == JointKind::Prismatic {
                    lever += joint_node
                        .limits
                        .lower
                        .abs()
                        .max(joint_node.limits.upper.abs());
                }
                joint = model.links()[joint_node.parent_link].parent_joint;
                guard += 1;
                debug_assert!(guard <= joints.len(), "the model is a tree");
                if guard > joints.len() {
                    break;
                }
            }
            movers.push(list);
        }

        let mut radius = vec![0.0_f64; model.links().len()];
        for link in &project.robot.links {
            let Some(index) = model.link_index(&link.id) else {
                continue;
            };
            for geometry in link.collision_geometries() {
                let vertices = match &geometry.shape {
                    axisforge_domain::Shape::Mesh { resource_id, .. } => project
                        .resource(resource_id)
                        .map(|resource| resource.vertices.as_slice()),
                    _ => None,
                };
                let body =
                    body_from_geometry(geometry, &axisforge_domain::Transform::IDENTITY, vertices)?;
                radius[index] =
                    radius[index].max(body.pose.translation.norm() + body.bounding_radius());
            }
        }
        Ok(Self { movers, radius })
    }

    /// Largest distance from a link's origin to its collision geometry.
    pub fn link_radius(&self, link: usize) -> f64 {
        self.radius.get(link).copied().unwrap_or(0.0)
    }

    /// Upper bound on how far any point within `radius` of the link's origin
    /// travels along the straight joint-space segment from `a` to `b`.
    pub fn travel(&self, link: usize, radius: f64, a: &[f64], b: &[f64]) -> f64 {
        self.movers.get(link).map_or(0.0, |movers| {
            movers
                .iter()
                .map(|m| {
                    let step = (b[m.value_index] - a[m.value_index]).abs();
                    if m.revolute {
                        (m.lever + radius) * step
                    } else {
                        step
                    }
                })
                .sum()
        })
    }

    /// Upper bound on how far any point of the link's own collision geometry
    /// travels from `a` to `b`.
    pub fn link_travel(&self, link: usize, a: &[f64], b: &[f64]) -> f64 {
        self.travel(link, self.link_radius(link), a, b)
    }

    /// Travel rate of a link per unit of a joint-space direction: the bound
    /// on how far any point within `radius` of its origin moves when the
    /// configuration moves by `rate` per unit.
    pub fn rate(&self, link: usize, radius: f64, rate: &[f64]) -> f64 {
        let zero = vec![0.0; rate.len()];
        self.travel(link, radius, &zero, rate)
    }

    /// `true` when moving joint `value_index` moves the link.
    pub fn moves(&self, link: usize, value_index: usize) -> bool {
        self.movers
            .get(link)
            .is_some_and(|m| m.iter().any(|m| m.value_index == value_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates;
    use axisforge_kinematics::forward_kinematics;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let value = (state >> 11) as f64 / (1_u64 << 53) as f64;
            value
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)] // Segment ends a and b, as in the bound.
    fn no_point_moves_further_than_the_bound() {
        // Sample points on every link's geometry along random segments and
        // compare their actual path length with the bound.
        for kind in [
            templates::GripperKind::Parallel,
            templates::GripperKind::Hand,
        ] {
            let project = templates::pick_and_place_cell_with(kind).unwrap();
            let model = KinematicModel::from_robot(&project.robot).unwrap();
            let bounds = MotionBounds::new(&project, &model).unwrap();
            let limits = model.limits();
            let mut rng = lcg(5);
            for _ in 0..20 {
                let a: Vec<f64> = limits
                    .iter()
                    .map(|l| l.lower + (l.upper - l.lower) * rng())
                    .collect();
                let b: Vec<f64> = limits
                    .iter()
                    .map(|l| l.lower + (l.upper - l.lower) * rng())
                    .collect();
                let steps = 200;
                for link in 0..model.links().len() {
                    let r = bounds.link_radius(link);
                    // A point on the sphere of radius r about the link origin.
                    let (u, v) = (rng() * 2.0 - 1.0, rng() * std::f64::consts::TAU);
                    let s = (1.0 - u * u).sqrt();
                    let local = axisforge_domain::Vec3::new(s * v.cos(), s * v.sin(), u).scaled(r);
                    let mut length = 0.0;
                    let mut previous = None;
                    for step in 0..=steps {
                        #[allow(clippy::cast_precision_loss)]
                        let t = f64::from(step) / f64::from(steps);
                        let q: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + (y - x) * t).collect();
                        let p = forward_kinematics(&model, &q)
                            .unwrap()
                            .link(link)
                            .apply(&local);
                        if let Some(previous) = previous {
                            length += p.distance(&previous);
                        }
                        previous = Some(p);
                    }
                    let bound = bounds.link_travel(link, &a, &b);
                    assert!(
                        length <= bound + 1e-9,
                        "link {link}: moved {length} > bound {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_link_only_counts_the_joints_above_it() {
        let project = templates::pick_and_place_cell_with(templates::GripperKind::Hand).unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let bounds = MotionBounds::new(&project, &model).unwrap();
        let root = model.root_link();
        let zero = vec![0.0; model.dof()];
        let moved = vec![1.0; model.dof()];
        assert!(bounds.link_travel(root, &zero, &moved).abs() < 1e-15);
        let gripper = project.robot.gripper.as_ref().unwrap();
        let palm = model.link_index(&gripper.palm_link_id).unwrap();
        for index in model.end_effector_value_indices() {
            assert!(!bounds.moves(palm, index), "fingers do not move the palm");
        }
        assert!(model
            .manipulator_value_indices()
            .iter()
            .all(|i| bounds.moves(palm, *i)));
    }
}
