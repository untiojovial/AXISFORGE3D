//! Grasp quality: will a two-finger hold on this part actually hold?
//!
//! The contact model is two Coulomb point contacts, one per jaw. That model
//! answers three questions honestly:
//!
//! - **Force closure.** Two frictional point contacts hold a part when the line
//!   between them lies inside both friction cones, i.e. when the angle between
//!   that line and each inward surface normal is at most `atan(mu)`.
//! - **Stroke.** The part has to fit between the open jaws.
//! - **Grip force.** Friction at the two pads must carry the component of the
//!   part's weight that acts along the pad faces, with a safety factor.
//!
//! What it does not model: torque about the closing line. Two point contacts
//! cannot resist it at all; real pads do, through their area, by an amount this
//! model has no data for. So the lever arm from the part's centre to the
//! closing line is *reported* and counted against the score, and nothing here
//! claims the part will or will not swing.
//!
//! The part's centre of mass is assumed to be its geometric centre.

use axisforge_domain::{Gripper, PhysicalProperties, Vec3, STANDARD_GRAVITY};
use serde::{Deserialize, Serialize};

use crate::candidate::GraspCandidate;

/// Factor by which available friction must exceed the load.
pub const GRIP_SAFETY_FACTOR: f64 = 2.0;

/// The measured merits of one grasp candidate for one gripper and part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraspQuality {
    /// Half-angle of the friction cone, `atan(mu)`, radians.
    pub friction_cone_half_angle: f64,
    /// Largest angle between the closing line and an inward contact normal, radians.
    pub antipodal_angle: f64,
    /// `true` when the closing line lies inside both friction cones.
    pub force_closure: bool,
    /// Stroke left over once the jaws are on the part, metres. Negative when
    /// the part is wider than the gripper opens.
    pub width_margin: f64,
    /// Clamping force needed to hold the part's weight, with the safety
    /// factor applied, newtons. `None` without friction, when no force suffices.
    pub required_grip_force: Option<f64>,
    /// `true` when the gripper can supply the required force.
    pub grip_force_sufficient: bool,
    /// Distance from the part's centre to the closing line, metres.
    pub center_offset: f64,
    /// A ranking heuristic in `[0, 1]`; higher is better. Not a physical quantity.
    pub score: f64,
}

fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

/// Evaluates a candidate for a gripper and a part's physical properties.
///
/// `up` is the unit vector opposing gravity.
pub fn evaluate(
    candidate: &GraspCandidate,
    gripper: &Gripper,
    physical: &PhysicalProperties,
    up: &Vec3,
) -> GraspQuality {
    let half_angle = physical.friction_cone_half_angle();
    // Inward normals should point at each other along the closing line.
    let antipodal_angle = angle_between(
        &candidate.normal_a.scaled(-1.0),
        &candidate.closing.scaled(-1.0),
    )
    .max(angle_between(
        &candidate.normal_b.scaled(-1.0),
        &candidate.closing,
    ));
    let force_closure = physical.friction_coefficient > 0.0 && antipodal_angle <= half_angle;

    let width_margin = gripper.max_opening - candidate.width;

    // Weight along the closing line is carried by a jaw directly; only the
    // part across it loads the friction at the two pads.
    let along = up.dot(&candidate.closing);
    let tangential = physical.mass * STANDARD_GRAVITY * (1.0 - along * along).max(0.0).sqrt();
    let required_grip_force = (physical.friction_coefficient > 0.0)
        .then(|| GRIP_SAFETY_FACTOR * tangential / (2.0 * physical.friction_coefficient));
    let grip_force_sufficient =
        required_grip_force.is_some_and(|force| force <= gripper.max_grip_force);

    let unit = |value: f64| value.clamp(0.0, 1.0);
    let friction_term = if half_angle > 0.0 {
        unit((half_angle - antipodal_angle) / half_angle)
    } else {
        0.0
    };
    let width_term = unit(width_margin / gripper.max_opening);
    let force_term =
        required_grip_force.map_or(0.0, |force| unit(1.0 - force / gripper.max_grip_force));
    let center_term = unit(1.0 - candidate.center_offset / gripper.finger_depth);
    let score = 0.35 * friction_term + 0.15 * width_term + 0.2 * force_term + 0.3 * center_term;

    GraspQuality {
        friction_cone_half_angle: half_angle,
        antipodal_angle,
        force_closure,
        width_margin,
        required_grip_force,
        grip_force_sufficient,
        center_offset: candidate.center_offset,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates;

    fn gripper() -> Gripper {
        templates::six_axis_with_gripper().unwrap().gripper.unwrap()
    }

    fn candidate(closing: Vec3, width: f64) -> GraspCandidate {
        GraspCandidate {
            label: "t".into(),
            center: Vec3::ZERO,
            approach: Vec3::new(0.0, 0.0, -1.0),
            closing,
            width,
            normal_a: closing,
            normal_b: closing.scaled(-1.0),
            center_offset: 0.0,
        }
    }

    #[test]
    fn opposite_faces_are_in_force_closure() {
        let q = evaluate(
            &candidate(Vec3::X, 0.04),
            &gripper(),
            &PhysicalProperties {
                mass: 0.2,
                friction_coefficient: 0.5,
            },
            &Vec3::Z,
        );
        assert!(q.antipodal_angle < 1e-12);
        assert!(q.force_closure && q.grip_force_sufficient);
        assert!((q.width_margin - 0.04).abs() < 1e-12);
    }

    #[test]
    fn required_force_follows_the_coulomb_formula() {
        // Horizontal closing, vertical gravity: F = s * m * g / (2 mu).
        let q = evaluate(
            &candidate(Vec3::X, 0.04),
            &gripper(),
            &PhysicalProperties {
                mass: 1.0,
                friction_coefficient: 0.5,
            },
            &Vec3::Z,
        );
        let expected = GRIP_SAFETY_FACTOR * STANDARD_GRAVITY / (2.0 * 0.5);
        assert!((q.required_grip_force.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn a_part_held_top_and_bottom_needs_no_friction_for_its_weight() {
        let q = evaluate(
            &candidate(Vec3::Z, 0.04),
            &gripper(),
            &PhysicalProperties {
                mass: 5.0,
                friction_coefficient: 0.3,
            },
            &Vec3::Z,
        );
        assert!(q.required_grip_force.unwrap() < 1e-9);
    }

    #[test]
    fn a_heavy_part_exceeds_the_grip_force() {
        let q = evaluate(
            &candidate(Vec3::X, 0.04),
            &gripper(),
            &PhysicalProperties {
                mass: 20.0,
                friction_coefficient: 0.3,
            },
            &Vec3::Z,
        );
        assert!(!q.grip_force_sufficient, "{q:?}");
    }

    #[test]
    fn a_frictionless_part_is_never_in_force_closure() {
        let q = evaluate(
            &candidate(Vec3::X, 0.04),
            &gripper(),
            &PhysicalProperties {
                mass: 0.1,
                friction_coefficient: 0.0,
            },
            &Vec3::Z,
        );
        assert!(!q.force_closure);
        assert!(q.required_grip_force.is_none() && !q.grip_force_sufficient);
    }

    #[test]
    fn slanted_normals_outside_the_cone_break_force_closure() {
        let mut c = candidate(Vec3::X, 0.04);
        // Faces tilted 30 degrees from the closing line; mu = 0.3 is a 16.7 degree cone.
        let tilt = 30.0_f64.to_radians();
        c.normal_a = Vec3::new(tilt.cos(), tilt.sin(), 0.0);
        let physical = PhysicalProperties {
            mass: 0.1,
            friction_coefficient: 0.3,
        };
        let q = evaluate(&c, &gripper(), &physical, &Vec3::Z);
        assert!((q.antipodal_angle - tilt).abs() < 1e-12);
        assert!(!q.force_closure);
        // mu = 0.7 is a 35 degree cone, which contains it.
        let grippy = PhysicalProperties {
            mass: 0.1,
            friction_coefficient: 0.7,
        };
        assert!(evaluate(&c, &gripper(), &grippy, &Vec3::Z).force_closure);
    }

    #[test]
    fn an_off_centre_hold_scores_lower() {
        let centred = candidate(Vec3::X, 0.04);
        let mut offset = centred.clone();
        offset.center_offset = 0.02;
        let physical = PhysicalProperties::default();
        let g = gripper();
        assert!(
            evaluate(&offset, &g, &physical, &Vec3::Z).score
                < evaluate(&centred, &g, &physical, &Vec3::Z).score
        );
    }

    #[test]
    fn an_oversize_part_has_a_negative_width_margin() {
        let q = evaluate(
            &candidate(Vec3::X, 0.2),
            &gripper(),
            &PhysicalProperties::default(),
            &Vec3::Z,
        );
        assert!(q.width_margin < 0.0);
    }
}
