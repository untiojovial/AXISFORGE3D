//! Grasp quality: will a two-finger hold on this part actually hold?
//!
//! Quality is measured by a [`QualityMeasure`], chosen by name from the
//! [`crate::registry::Registry`]. Two are built in:
//!
//! - [`WrenchSpaceMeasure`] (`wrench_space`), the default: force closure and
//!   epsilon from the grasp wrench space of the pad contact patches. Because
//!   the pads touch over an area, this accounts for the part twisting between
//!   the fingers, which a two-point model cannot.
//! - [`AntipodalMeasure`] (`antipodal`): the classical two-point test that the
//!   line between the contacts lies inside both friction cones. Cheap, and
//!   blind to torque about that line.
//!
//! Whatever the measure, every grasp also gets the engineering numbers a
//! gripper is specified by: the stroke left over, and the clamping force
//! needed to carry the part's weight by friction with a safety factor.
//!
//! The part's centre of mass is assumed to be its geometric centre.

use axisforge_domain::{Gripper, Vec3, WorldBody, STANDARD_GRAVITY};
use serde::{Deserialize, Serialize};

use crate::candidate::GraspCandidate;
use crate::wrench::{grasp_wrench_space, WrenchOptions};

/// Factor by which available friction must exceed the load.
pub const GRIP_SAFETY_FACTOR: f64 = 2.0;

/// Everything a quality measure may look at.
#[derive(Debug, Clone, Copy)]
pub struct QualityInput<'a> {
    /// The grasp being judged.
    pub candidate: &'a GraspCandidate,
    /// The gripper making it.
    pub gripper: &'a Gripper,
    /// The part being held.
    pub body: &'a WorldBody,
    /// Unit vector opposing gravity.
    pub up: Vec3,
    /// Friction and contact-model settings.
    pub wrench: &'a WrenchOptions,
}

impl QualityInput<'_> {
    /// The friction coefficient in force: the override if given, else the part's.
    pub fn friction_coefficient(&self) -> f64 {
        self.wrench
            .friction_coefficient
            .unwrap_or(self.body.physical.friction_coefficient)
    }

    /// The point torques are taken about: the part's geometric centre.
    pub fn center_of_mass(&self) -> Vec3 {
        self.body
            .single_geometry()
            .map_or(self.body.pose.translation, |geometry| {
                self.body.pose.compose(&geometry.origin).translation
            })
    }

    /// The length torques are divided by: the part's bounding radius.
    pub fn torque_scale(&self) -> f64 {
        self.body
            .single_geometry()
            .and_then(|geometry| geometry.shape.local_half_extents())
            .map_or(1.0, |half| half.norm())
            .max(1.0e-6)
    }
}

/// What a quality measure reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurement {
    /// The measure's own figure of merit; higher is better. Its unit and
    /// range are the measure's to define.
    pub quality: f64,
    /// The same merit mapped into `[0, 1]`, for ranking alongside other terms.
    pub normalized: f64,
    /// `true` when the measure finds the grasp in force closure.
    pub force_closure: bool,
}

/// A way of judging a grasp. Implement this to plug in another library.
pub trait QualityMeasure: Send + Sync {
    /// The name the measure is selected by.
    fn id(&self) -> &'static str;
    /// One line saying what it measures, for listings.
    fn summary(&self) -> &'static str;
    /// Judges one grasp.
    fn measure(&self, input: &QualityInput<'_>) -> Measurement;
}

/// Force closure and epsilon from the grasp wrench space.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrenchSpaceMeasure;

impl QualityMeasure for WrenchSpaceMeasure {
    fn id(&self) -> &'static str {
        "wrench_space"
    }

    fn summary(&self) -> &'static str {
        "Grasp wrench space of the pad contact patches: force closure and epsilon \
         (distance from the origin to the hull of the contact wrenches)."
    }

    fn measure(&self, input: &QualityInput<'_>) -> Measurement {
        let space = grasp_wrench_space(
            &input.candidate.contacts,
            &input.center_of_mass(),
            input.torque_scale(),
            input.friction_coefficient(),
            input.wrench,
        );
        Measurement {
            quality: space.epsilon,
            normalized: (space.epsilon / input.wrench.unit_force.max(f64::MIN_POSITIVE))
                .clamp(0.0, 1.0),
            force_closure: space.force_closure,
        }
    }
}

/// The two-point antipodal test against the friction cone.
#[derive(Debug, Clone, Copy, Default)]
pub struct AntipodalMeasure;

impl QualityMeasure for AntipodalMeasure {
    fn id(&self) -> &'static str {
        "antipodal"
    }

    fn summary(&self) -> &'static str {
        "Two-point antipodal test: the margin by which the line between the contacts lies \
         inside both friction cones. Does not account for torque about that line."
    }

    fn measure(&self, input: &QualityInput<'_>) -> Measurement {
        let mu = input.friction_coefficient();
        let half_angle = mu.atan();
        let angle = antipodal_angle(input.candidate);
        let margin = if half_angle > 0.0 {
            ((half_angle - angle) / half_angle).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Measurement {
            quality: margin,
            normalized: margin,
            force_closure: mu > 0.0 && angle <= half_angle,
        }
    }
}

fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

/// Largest angle between the closing line and an inward contact normal.
fn antipodal_angle(candidate: &GraspCandidate) -> f64 {
    angle_between(
        &candidate.normal_a.scaled(-1.0),
        &candidate.closing.scaled(-1.0),
    )
    .max(angle_between(
        &candidate.normal_b.scaled(-1.0),
        &candidate.closing,
    ))
}

/// The measured merits of one grasp candidate for one gripper and part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraspQuality {
    /// Name of the quality measure used.
    pub measure: String,
    /// That measure's figure of merit; for `wrench_space`, epsilon.
    pub quality: f64,
    /// `true` when that measure finds the grasp in force closure.
    pub force_closure: bool,
    /// Friction coefficient used.
    pub friction_coefficient: f64,
    /// Half-angle of the friction cone, `atan(mu)`, radians.
    pub friction_cone_half_angle: f64,
    /// Largest angle between the closing line and an inward contact normal, radians.
    pub antipodal_angle: f64,
    /// Contact points the pads make on the part.
    pub contact_count: usize,
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

struct Basics {
    friction: f64,
    width_margin: f64,
    required_grip_force: Option<f64>,
    grip_force_sufficient: bool,
    /// Stroke, grip-force and centring terms, each in `[0, 1]`.
    terms: [f64; 3],
}

fn basics(input: &QualityInput<'_>) -> Basics {
    let (candidate, gripper) = (input.candidate, input.gripper);
    let friction = input.friction_coefficient();
    let width_margin = gripper.max_opening - candidate.width;
    // Weight along the closing line is carried by a jaw directly; only the
    // part across it loads the friction at the two pads.
    let along = input.up.dot(&candidate.closing);
    let tangential =
        input.body.physical.mass * STANDARD_GRAVITY * (1.0 - along * along).max(0.0).sqrt();
    let required_grip_force =
        (friction > 0.0).then(|| GRIP_SAFETY_FACTOR * tangential / (2.0 * friction));
    let grip_force_sufficient =
        required_grip_force.is_some_and(|force| force <= gripper.max_grip_force);
    let unit = |value: f64| value.clamp(0.0, 1.0);
    Basics {
        friction,
        width_margin,
        required_grip_force,
        grip_force_sufficient,
        terms: [
            unit(width_margin / gripper.max_opening),
            required_grip_force.map_or(0.0, |force| unit(1.0 - force / gripper.max_grip_force)),
            unit(1.0 - candidate.center_offset / gripper.finger_depth),
        ],
    }
}

/// The clamping force a grasp needs to carry the part's weight, and whether
/// the gripper has it. Cheap: no quality measure is run.
pub fn grip_load(input: &QualityInput<'_>) -> (Option<f64>, bool) {
    let basics = basics(input);
    (basics.required_grip_force, basics.grip_force_sufficient)
}

/// A cheap ranking of a candidate from stroke, grip force and centring alone.
///
/// Used to decide the order in which candidates are examined, before the
/// quality measure — which may be expensive — has been run on any of them.
pub fn preliminary_score(input: &QualityInput<'_>) -> f64 {
    let [width, force, center] = basics(input).terms;
    (0.15 * width + 0.2 * force + 0.25 * center) / 0.6
}

/// Evaluates a candidate with the given quality measure.
pub fn evaluate(measure: &dyn QualityMeasure, input: &QualityInput<'_>) -> GraspQuality {
    let basics = basics(input);
    let measurement = measure.measure(input);
    let [width, force, center] = basics.terms;
    GraspQuality {
        measure: measure.id().to_owned(),
        quality: measurement.quality,
        force_closure: measurement.force_closure,
        friction_coefficient: basics.friction,
        friction_cone_half_angle: basics.friction.atan(),
        antipodal_angle: antipodal_angle(input.candidate),
        contact_count: input.candidate.contacts.len(),
        width_margin: basics.width_margin,
        required_grip_force: basics.required_grip_force,
        grip_force_sufficient: basics.grip_force_sufficient,
        center_offset: input.candidate.center_offset,
        score: 0.4 * measurement.normalized.clamp(0.0, 1.0)
            + 0.15 * width
            + 0.2 * force
            + 0.25 * center,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{synthesize, SynthesisOptions};
    use axisforge_domain::{templates, GeometryInstance, PhysicalProperties, Shape, Transform};

    fn gripper() -> Gripper {
        templates::six_axis_with_gripper().unwrap().gripper.unwrap()
    }

    fn cube(mass: f64, mu: f64) -> WorldBody {
        WorldBody::manipulable(
            "p",
            "part",
            Transform::IDENTITY,
            GeometryInstance::visual(
                "g",
                "body",
                Shape::Box {
                    size: Vec3::new(0.04, 0.04, 0.04),
                },
            ),
            PhysicalProperties {
                mass,
                friction_coefficient: mu,
            },
        )
    }

    fn judge(
        measure: &dyn QualityMeasure,
        body: &WorldBody,
        label: &str,
        wrench: &WrenchOptions,
    ) -> GraspQuality {
        let candidates = synthesize(body, &SynthesisOptions::default()).unwrap();
        let candidate = candidates.iter().find(|c| c.label == label).expect(label);
        let gripper = gripper();
        evaluate(
            measure,
            &QualityInput {
                candidate,
                gripper: &gripper,
                body,
                up: Vec3::Z,
                wrench,
            },
        )
    }

    const SIDE: &str = "box close +x approach -z";

    #[test]
    fn a_cube_held_by_two_pads_is_in_wrench_space_force_closure() {
        let q = judge(
            &WrenchSpaceMeasure,
            &cube(0.2, 0.5),
            SIDE,
            &WrenchOptions::default(),
        );
        assert_eq!(q.measure, "wrench_space");
        assert!(q.force_closure && q.quality > 0.0, "{q:?}");
        assert_eq!(q.contact_count, 8);
        assert!(q.grip_force_sufficient);
        assert!((q.width_margin - 0.04).abs() < 1e-12);
    }

    #[test]
    fn a_sphere_needs_a_soft_pad_for_force_closure() {
        let ball = WorldBody::manipulable(
            "p",
            "ball",
            Transform::IDENTITY,
            GeometryInstance::visual("g", "body", Shape::Sphere { radius: 0.02 }),
            PhysicalProperties {
                mass: 0.1,
                friction_coefficient: 0.6,
            },
        );
        let label = "sphere close +x approach -z";
        let hard = judge(&WrenchSpaceMeasure, &ball, label, &WrenchOptions::default());
        assert!(
            !hard.force_closure,
            "two hard points cannot resist a twist: {hard:?}"
        );
        // The antipodal test cannot see that.
        assert!(judge(&AntipodalMeasure, &ball, label, &WrenchOptions::default()).force_closure);
        let soft = WrenchOptions {
            torsional_friction: 0.004,
            ..WrenchOptions::default()
        };
        assert!(judge(&WrenchSpaceMeasure, &ball, label, &soft).force_closure);
    }

    #[test]
    fn required_force_follows_the_coulomb_formula() {
        let q = judge(
            &AntipodalMeasure,
            &cube(1.0, 0.5),
            SIDE,
            &WrenchOptions::default(),
        );
        let expected = GRIP_SAFETY_FACTOR * STANDARD_GRAVITY / (2.0 * 0.5);
        assert!((q.required_grip_force.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn a_part_held_top_and_bottom_needs_no_friction_for_its_weight() {
        let q = judge(
            &AntipodalMeasure,
            &cube(5.0, 0.3),
            "box close +z approach +x",
            &WrenchOptions::default(),
        );
        assert!(q.required_grip_force.unwrap() < 1e-9);
    }

    #[test]
    fn a_heavy_part_exceeds_the_grip_force() {
        let q = judge(
            &AntipodalMeasure,
            &cube(20.0, 0.3),
            SIDE,
            &WrenchOptions::default(),
        );
        assert!(!q.grip_force_sufficient, "{q:?}");
    }

    #[test]
    fn a_frictionless_part_is_never_in_force_closure() {
        for measure in [
            &WrenchSpaceMeasure as &dyn QualityMeasure,
            &AntipodalMeasure,
        ] {
            let q = judge(measure, &cube(0.1, 0.0), SIDE, &WrenchOptions::default());
            assert!(!q.force_closure, "{}", measure.id());
            assert!(q.required_grip_force.is_none() && !q.grip_force_sufficient);
        }
    }

    #[test]
    fn the_friction_override_replaces_the_parts_own_value() {
        let slippery = WrenchOptions {
            friction_coefficient: Some(0.0),
            ..WrenchOptions::default()
        };
        let q = judge(&WrenchSpaceMeasure, &cube(0.1, 0.9), SIDE, &slippery);
        assert!(!q.force_closure);
        assert!(q.friction_coefficient.abs() < 1e-12);
    }

    #[test]
    fn an_off_centre_hold_scores_lower() {
        let body = cube(0.2, 0.5);
        let options = WrenchOptions::default();
        let centred = judge(&WrenchSpaceMeasure, &body, SIDE, &options);
        let shallow = judge(
            &WrenchSpaceMeasure,
            &body,
            "box close +x approach -z shallow",
            &options,
        );
        assert!(shallow.center_offset > 0.0);
        assert!(
            shallow.score < centred.score,
            "{} vs {}",
            shallow.score,
            centred.score
        );
    }

    #[test]
    fn slanted_normals_outside_the_cone_fail_the_antipodal_test() {
        let body = cube(0.1, 0.3);
        let mut candidates = synthesize(&body, &SynthesisOptions::default()).unwrap();
        let candidate = candidates.iter_mut().find(|c| c.label == SIDE).unwrap();
        // Faces tilted 30 degrees from the closing line; mu = 0.3 is a 16.7 degree cone.
        let tilt = 30.0_f64.to_radians();
        candidate.normal_a = Vec3::new(tilt.cos(), tilt.sin(), 0.0);
        let gripper = gripper();
        let wrench = WrenchOptions::default();
        let input = QualityInput {
            candidate,
            gripper: &gripper,
            body: &body,
            up: Vec3::Z,
            wrench: &wrench,
        };
        let q = evaluate(&AntipodalMeasure, &input);
        assert!((q.antipodal_angle - tilt).abs() < 1e-12);
        assert!(!q.force_closure);
    }
}
