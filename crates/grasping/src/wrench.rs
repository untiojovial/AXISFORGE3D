//! The grasp wrench space: what disturbances a set of contacts can resist.
//!
//! Each contact can push on the part anywhere inside its friction cone. A
//! force at a contact also produces a torque about the part's centre of mass,
//! and the pair is a *wrench*, a point in six dimensions. The convex hull of
//! the wrenches the contacts can produce, with the contact forces together
//! limited to one unit, is the grasp wrench space. Two things are read off it:
//!
//! - **Force closure**: the origin is strictly inside, so any disturbance can
//!   be opposed.
//! - **Epsilon**: the distance from the origin to the hull's boundary, the
//!   magnitude of the weakest-resisted disturbance. This is the quality
//!   measure of Ferrari and Canny, with the sum of contact forces bounded.
//!
//! Friction cones are approximated by pyramids with a chosen number of edges.
//! A pyramid inscribed in the cone can only under-state what friction does, so
//! the approximation errs toward a lower epsilon.
//!
//! Torques are divided by a length so that both halves of a wrench share a
//! unit; the convention here is the part's bounding radius, which makes
//! epsilon independent of the part's size and of the unit of length.

use axisforge_domain::Vec3;
use serde::{Deserialize, Serialize};

use crate::hull::{Hull, HullError};

/// A point where a finger touches the part.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContactPoint {
    /// Position, world coordinates.
    pub point: Vec3,
    /// Outward unit normal of the part's surface there.
    pub normal: Vec3,
}

/// How contacts are turned into wrenches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct WrenchOptions {
    /// Normal force at each contact, newtons. Scales epsilon linearly.
    pub unit_force: f64,
    /// Friction coefficient; the part's own value when `None`.
    pub friction_coefficient: Option<f64>,
    /// Edges of the pyramid approximating each friction cone.
    pub friction_cone_samples: usize,
    /// Torsional friction of a soft pad: torque about the contact normal per
    /// unit of normal force, metres. Zero is a hard point contact.
    pub torsional_friction: f64,
}

impl Default for WrenchOptions {
    fn default() -> Self {
        Self {
            unit_force: 1.0,
            friction_coefficient: None,
            friction_cone_samples: 8,
            torsional_friction: 0.0,
        }
    }
}

/// What the grasp wrench space says about a grasp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WrenchSpace {
    /// Distance from the origin to the boundary of the wrench space; zero
    /// when the grasp is not in force closure.
    pub epsilon: f64,
    /// `true` when the origin is strictly inside the wrench space.
    pub force_closure: bool,
    /// Contact points used.
    pub contact_count: usize,
    /// Primitive wrenches generated from them.
    pub wrench_count: usize,
    /// `true` when the hull could not be built reliably, in which case the
    /// grasp is conservatively reported as not in force closure.
    pub unreliable: bool,
}

/// Builds the primitive wrenches of a set of contacts.
///
/// `center` is the point torques are taken about, normally the centre of
/// mass, and `torque_scale` the length torques are divided by.
pub fn primitive_wrenches(
    contacts: &[ContactPoint],
    center: &Vec3,
    torque_scale: f64,
    friction_coefficient: f64,
    options: &WrenchOptions,
) -> Vec<Vec<f64>> {
    let samples = options.friction_cone_samples.max(3);
    let scale = torque_scale.max(f64::MIN_POSITIVE);
    let mut wrenches = Vec::with_capacity(contacts.len() * (samples + 2));
    let mut push = |force: Vec3, torque: Vec3| {
        wrenches.push(vec![
            force.x,
            force.y,
            force.z,
            torque.x / scale,
            torque.y / scale,
            torque.z / scale,
        ]);
    };
    for contact in contacts {
        let Ok(inward) = contact.normal.scaled(-1.0).normalized("contact normal") else {
            continue;
        };
        let Ok(tangent) = inward.any_perpendicular("contact tangent") else {
            continue;
        };
        let bitangent = inward.cross(&tangent);
        let arm = contact.point.sub(center);
        for k in 0..samples {
            #[allow(clippy::cast_precision_loss)]
            let angle = std::f64::consts::TAU * k as f64 / samples as f64;
            let slide = tangent
                .scaled(angle.cos())
                .add(&bitangent.scaled(angle.sin()));
            let force = inward
                .add(&slide.scaled(friction_coefficient))
                .scaled(options.unit_force);
            push(force, arm.cross(&force));
        }
        if options.torsional_friction > 0.0 {
            let force = inward.scaled(options.unit_force);
            let twist = inward.scaled(options.torsional_friction * options.unit_force);
            push(force, arm.cross(&force).add(&twist));
            push(force, arm.cross(&force).sub(&twist));
        }
    }
    wrenches
}

/// Evaluates the grasp wrench space of a set of contacts.
pub fn grasp_wrench_space(
    contacts: &[ContactPoint],
    center: &Vec3,
    torque_scale: f64,
    friction_coefficient: f64,
    options: &WrenchOptions,
) -> WrenchSpace {
    let wrenches = primitive_wrenches(
        contacts,
        center,
        torque_scale,
        friction_coefficient,
        options,
    );
    let (epsilon, unreliable) = match Hull::build(&wrenches, 6) {
        Ok(hull) => (hull.origin_depth(), false),
        // Wrenches that do not span six dimensions leave some disturbance
        // entirely unopposed: that is a definite "no", not an unreliable one.
        Err(HullError::Flat) => (0.0, false),
        Err(HullError::Degenerate) => (0.0, true),
    };
    WrenchSpace {
        epsilon,
        force_closure: epsilon > 0.0,
        contact_count: contacts.len(),
        wrench_count: wrenches.len(),
        unreliable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two square pads, `half` on a side, pressing on opposite faces `width` apart.
    fn pads(width: f64, half: f64) -> Vec<ContactPoint> {
        let mut contacts = Vec::new();
        for side in [1.0, -1.0] {
            for (y, z) in [(half, half), (half, -half), (-half, half), (-half, -half)] {
                contacts.push(ContactPoint {
                    point: Vec3::new(side * width * 0.5, y, z),
                    normal: Vec3::new(side, 0.0, 0.0),
                });
            }
        }
        contacts
    }

    fn space(contacts: &[ContactPoint], mu: f64, options: &WrenchOptions) -> WrenchSpace {
        grasp_wrench_space(contacts, &Vec3::ZERO, 0.035, mu, options)
    }

    #[test]
    fn two_pads_on_opposite_faces_are_in_force_closure() {
        let result = space(&pads(0.04, 0.01), 0.5, &WrenchOptions::default());
        assert!(result.force_closure, "{result:?}");
        assert!(result.epsilon > 0.01 && result.epsilon < 1.0, "{result:?}");
        assert_eq!(result.wrench_count, 64);
        assert!(!result.unreliable);
    }

    #[test]
    fn without_friction_there_is_no_force_closure() {
        let result = space(&pads(0.04, 0.01), 0.0, &WrenchOptions::default());
        assert!(!result.force_closure, "{result:?}");
    }

    #[test]
    fn two_hard_point_contacts_cannot_resist_a_twist_about_their_line() {
        let contacts = [
            ContactPoint {
                point: Vec3::new(0.02, 0.0, 0.0),
                normal: Vec3::X,
            },
            ContactPoint {
                point: Vec3::new(-0.02, 0.0, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
            },
        ];
        let hard = space(&contacts, 0.8, &WrenchOptions::default());
        assert!(!hard.force_closure && !hard.unreliable, "{hard:?}");
        // A soft pad adds torsional friction, which is exactly what was missing.
        let soft = WrenchOptions {
            torsional_friction: 0.004,
            ..WrenchOptions::default()
        };
        assert!(space(&contacts, 0.8, &soft).force_closure);
    }

    #[test]
    fn more_friction_and_larger_pads_make_a_better_grasp() {
        let options = WrenchOptions::default();
        let base = space(&pads(0.04, 0.01), 0.3, &options).epsilon;
        assert!(space(&pads(0.04, 0.01), 0.8, &options).epsilon > base);
        assert!(space(&pads(0.04, 0.02), 0.3, &options).epsilon > base);
    }

    #[test]
    fn epsilon_scales_with_the_unit_force() {
        let one = space(&pads(0.04, 0.01), 0.5, &WrenchOptions::default()).epsilon;
        let three = WrenchOptions {
            unit_force: 3.0,
            ..WrenchOptions::default()
        };
        assert!((space(&pads(0.04, 0.01), 0.5, &three).epsilon - 3.0 * one).abs() < 1e-6 * one);
    }

    #[test]
    fn epsilon_never_exceeds_the_support_in_any_direction() {
        let contacts = pads(0.04, 0.01);
        let options = WrenchOptions::default();
        let wrenches = primitive_wrenches(&contacts, &Vec3::ZERO, 0.035, 0.5, &options);
        let epsilon = space(&contacts, 0.5, &options).epsilon;
        let mut rng = axisforge_kinematics::Rng::new(5);
        let mut tightest = f64::INFINITY;
        for _ in 0..200_000 {
            let u: Vec<f64> = (0..6).map(|_| rng.next_in(-1.0, 1.0)).collect();
            let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            let support = wrenches
                .iter()
                .map(|w| w.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>() / norm)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(epsilon <= support + 1e-9, "{epsilon} > {support}");
            tightest = tightest.min(support);
        }
        // Random directions in six dimensions rarely land on the weakest one,
        // but they should come within a small factor of it.
        assert!(
            tightest < 3.0 * epsilon,
            "sampled {tightest}, hull {epsilon}"
        );
    }

    #[test]
    fn a_finer_friction_pyramid_never_lowers_epsilon_much() {
        let coarse = WrenchOptions {
            friction_cone_samples: 4,
            ..WrenchOptions::default()
        };
        let fine = WrenchOptions {
            friction_cone_samples: 12,
            ..WrenchOptions::default()
        };
        let a = space(&pads(0.04, 0.01), 0.5, &coarse);
        let b = space(&pads(0.04, 0.01), 0.5, &fine);
        assert!(a.force_closure && b.force_closure);
        assert!(
            b.epsilon > 0.8 * a.epsilon,
            "{} vs {}",
            a.epsilon,
            b.epsilon
        );
    }
}
