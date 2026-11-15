//! Grasp synthesis: where a parallel-jaw gripper could hold a primitive.
//!
//! A two-finger grasp is described by a centre, a *closing* direction (the
//! line the jaws oppose each other along) and an *approach* direction (the way
//! the gripper travels to get there). For boxes, cylinders, capsules and
//! spheres the antipodal grasps can be written down from the dimensions, so
//! this module enumerates them instead of sampling a surface: every candidate
//! places its two contacts on opposite faces, or on a diameter, where the
//! surface normals point straight along the closing line.
//!
//! This is pure geometry of the part. Whether the gripper is wide enough, the
//! arm can reach, or the table is in the way is the planner's business.

use std::f64::consts::PI;

use axisforge_domain::{DomainError, DomainResult, Quat, Shape, Transform, Vec3, WorldBody};
use serde::{Deserialize, Serialize};

use crate::wrench::ContactPoint;

/// Gap kept between the palm and the part's near face, metres.
const PALM_CLEARANCE: f64 = 0.002;
/// Gap kept between the fingertips and whatever is beyond the far face, metres.
const TIP_CLEARANCE: f64 = 0.002;

/// How densely round shapes are sampled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SynthesisOptions {
    /// Closing directions tried around a cylinder, capsule or sphere.
    pub angular_samples: usize,
    /// Grasp heights tried along a cylinder's or capsule's axis.
    pub axial_samples: usize,
    /// How far the gripping faces extend along the approach axis, metres.
    ///
    /// Taken from the gripper; it decides how deep a grasp can sit.
    pub finger_depth: f64,
    /// Width of each gripping pad, metres. Taken from the gripper; with the
    /// depth it bounds the contact patch.
    pub finger_width: f64,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            angular_samples: 8,
            axial_samples: 3,
            finger_depth: 0.045,
            finger_width: 0.025,
        }
    }
}

/// One way of holding a part, in world coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraspCandidate {
    /// A stable, human-readable label, unique within one synthesis.
    pub label: String,
    /// Midpoint between the two contacts.
    pub center: Vec3,
    /// Unit direction the gripper travels to reach the part.
    pub approach: Vec3,
    /// Unit direction from the centre toward the first contact.
    pub closing: Vec3,
    /// Distance between the two contacts: the opening the part needs, metres.
    pub width: f64,
    /// Outward surface normal at the first contact (`center + closing * width/2`).
    pub normal_a: Vec3,
    /// Outward surface normal at the second contact.
    pub normal_b: Vec3,
    /// Distance from the part's geometric centre to the closing line, metres.
    pub center_offset: f64,
    /// Where the pads touch the part: the corners of each contact patch.
    ///
    /// Four per side on a flat face, two where a flat pad meets a curved wall
    /// along a line, one on a sphere.
    #[serde(default)]
    pub contacts: Vec<ContactPoint>,
}

impl GraspCandidate {
    /// The first contact point.
    pub fn contact_a(&self) -> Vec3 {
        self.center.add(&self.closing.scaled(self.width * 0.5))
    }

    /// The second contact point.
    pub fn contact_b(&self) -> Vec3 {
        self.center.sub(&self.closing.scaled(self.width * 0.5))
    }

    /// The world rotation whose columns are (closing, approach, closing x approach).
    ///
    /// # Errors
    /// When closing and approach are not perpendicular unit vectors.
    pub fn orientation(&self) -> DomainResult<Quat> {
        Quat::from_basis(
            &self.closing,
            &self.approach,
            &self.closing.cross(&self.approach),
        )
    }
}

/// The part of a body's surface a pad can press on, for one side of a grasp.
///
/// Given in the plane of the pad, relative to the grasp's origin: an interval
/// along `lateral_axis` (which must be parallel to closing x approach) and an
/// interval along the approach direction. A zero-width interval is a line
/// contact; two of them, a point contact.
#[derive(Debug, Clone, Copy)]
struct Region {
    lateral_axis: Vec3,
    lateral: (f64, f64),
    along: (f64, f64),
}

impl Region {
    /// A single point: a curved surface meeting a flat pad.
    fn point() -> Self {
        Self {
            lateral_axis: Vec3::X,
            lateral: (0.0, 0.0),
            along: (0.0, 0.0),
        }
    }
}

/// One family of grasps: a closing line, an approach, and the surface there.
struct Hold<'n> {
    name: &'n str,
    /// The point on the closing line at the part's mid-depth, part frame.
    origin: Vec3,
    closing: Vec3,
    approach: Vec3,
    /// Distance between the two contact surfaces.
    width: f64,
    /// Thickness of the part along the approach.
    depth: f64,
    /// A circular cross-section in the closing-approach plane.
    round: bool,
    region: Region,
}

/// Collects candidates for a part, converting from its frame to the world.
struct Builder<'a> {
    pose: Transform,
    options: &'a SynthesisOptions,
    out: Vec<GraspCandidate>,
}

impl Builder<'_> {
    /// Adds the grasps of one family: centred and, where it helps, shallow,
    /// each with both jaw assignments. The two wrist orientations differ in
    /// reachability even though the contacts are the same.
    fn add(&mut self, hold: &Hold<'_>) {
        // How far past the near face the grasp centre may sit before the palm
        // would touch the part.
        let deepest = self.options.finger_depth * 0.5 - PALM_CLEARANCE;
        // On a round cross-section the closing line has to pass through the
        // centre: anywhere else the jaws meet a chord, not a diameter, and the
        // surface normals no longer oppose each other. Such grasps cannot be
        // made shallower; if the part is too thick for the fingers, the
        // collision check on the palm is what rejects them.
        let centred = if hold.round {
            hold.depth * 0.5
        } else {
            (hold.depth * 0.5).min(deepest)
        };
        let mut depths = vec![("", centred)];
        // A shallower hold keeps the fingertips from protruding past the far
        // face, which matters when the part rests on that face.
        let flush = hold.depth - self.options.finger_depth * 0.5 - TIP_CLEARANCE;
        if !hold.round && flush > 0.0 && flush < centred - 1.0e-9 {
            depths.push((" shallow", flush));
        }
        for (suffix, from_near_face) in depths {
            let along_center = from_near_face - hold.depth * 0.5;
            let local_center = hold.origin.add(&hold.approach.scaled(along_center));
            for (flip, sign) in [("", 1.0), (" flipped", -1.0)] {
                let closing_local = hold.closing.scaled(sign);
                let closing_world = self.pose.apply_direction(&closing_local);
                let contacts = self
                    .pad_contacts(hold, &closing_local, along_center)
                    .into_iter()
                    .map(|(point, side)| ContactPoint {
                        point: self.pose.apply(&point),
                        normal: closing_world.scaled(side),
                    })
                    .collect();
                self.out.push(GraspCandidate {
                    label: format!("{}{suffix}{flip}", hold.name),
                    center: self.pose.apply(&local_center),
                    approach: self.pose.apply_direction(&hold.approach),
                    closing: closing_world,
                    width: hold.width,
                    normal_a: closing_world,
                    normal_b: closing_world.scaled(-1.0),
                    center_offset: local_center
                        .sub(&closing_local.scaled(local_center.dot(&closing_local)))
                        .norm(),
                    contacts,
                });
            }
        }
    }

    /// Where the two pads touch the part: the corners of the overlap between
    /// each pad's face and the part's surface, in the part's frame, with the
    /// side (+1 or -1 along the closing direction) each point is on.
    ///
    /// Corners are enough: the wrenches of every point of a flat patch are
    /// convex combinations of the wrenches at its corners.
    fn pad_contacts(
        &self,
        hold: &Hold<'_>,
        closing_local: &Vec3,
        along_center: f64,
    ) -> Vec<(Vec3, f64)> {
        let lateral = closing_local.cross(&hold.approach);
        let region = &hold.region;
        let (l0, l1) = if region.lateral_axis.dot(&lateral) >= 0.0 {
            region.lateral
        } else {
            (-region.lateral.1, -region.lateral.0)
        };
        let half_width = self.options.finger_width * 0.5;
        let half_depth = self.options.finger_depth * 0.5;
        let l = (l0.max(-half_width), l1.min(half_width));
        let a = (
            region.along.0.max(along_center - half_depth),
            region.along.1.min(along_center + half_depth),
        );
        if l.0 > l.1 + 1.0e-12 || a.0 > a.1 + 1.0e-12 {
            // The pad does not reach the surface at all.
            return Vec::new();
        }
        let ends = |(low, high): (f64, f64)| {
            if high - low < 1.0e-9 {
                vec![0.5 * (low + high)]
            } else {
                vec![low, high]
            }
        };
        let mut points = Vec::new();
        for side in [1.0, -1.0] {
            for lv in ends(l) {
                for av in ends(a) {
                    points.push((
                        hold.origin
                            .add(&lateral.scaled(lv))
                            .add(&hold.approach.scaled(av))
                            .add(&closing_local.scaled(side * hold.width * 0.5)),
                        side,
                    ));
                }
            }
        }
        points
    }
}

fn axis_name(axis: &Vec3) -> &'static str {
    match (
        axis.x.abs() > 0.5,
        axis.y.abs() > 0.5,
        axis.x + axis.y + axis.z > 0.0,
    ) {
        (true, _, true) => "+x",
        (true, _, false) => "-x",
        (false, true, true) => "+y",
        (false, true, false) => "-y",
        (false, false, true) => "+z",
        (false, false, false) => "-z",
    }
}

/// Grasps of a box: each pair of opposite faces, approached along each other axis.
fn box_grasps(builder: &mut Builder<'_>, size: &Vec3) {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let extents = [size.x, size.y, size.z];
    for (c, closing) in axes.iter().enumerate() {
        for (a, axis) in axes.iter().enumerate() {
            if a == c {
                continue;
            }
            let l = 3 - a - c;
            for sign in [1.0, -1.0] {
                let approach = axis.scaled(sign);
                builder.add(&Hold {
                    name: &format!(
                        "box close {} approach {}",
                        axis_name(closing),
                        axis_name(&approach)
                    ),
                    origin: Vec3::ZERO,
                    closing: *closing,
                    approach,
                    width: extents[c],
                    depth: extents[a],
                    round: false,
                    region: Region {
                        lateral_axis: axes[l],
                        lateral: (-extents[l] * 0.5, extents[l] * 0.5),
                        along: (-extents[a] * 0.5, extents[a] * 0.5),
                    },
                });
            }
        }
    }
}

/// Grasps of a cylinder or capsule whose axis is local Z.
#[allow(clippy::cast_precision_loss)] // Sample counts are far below 2^52.
fn axial_grasps(builder: &mut Builder<'_>, radius: f64, length: f64, is_capsule: bool) {
    let angular = builder.options.angular_samples.max(1);
    let heights = builder.options.axial_samples.max(1);
    let total = if is_capsule {
        length + 2.0 * radius
    } else {
        length
    };
    // Only half a turn: the other half repeats the same contact pairs, and
    // both jaw assignments are emitted anyway.
    for k in 0..angular {
        let angle = PI * k as f64 / angular as f64;
        let closing = Vec3::new(angle.cos(), angle.sin(), 0.0);
        let side = Vec3::new(-angle.sin(), angle.cos(), 0.0);
        for sign in [1.0, -1.0] {
            // Along the axis, over the end of the part. A flat pad meets the
            // curved wall along a line parallel to the axis.
            let along = Vec3::Z.scaled(sign);
            builder.add(&Hold {
                name: &format!("side {k} approach {}", axis_name(&along)),
                origin: Vec3::ZERO,
                closing,
                approach: along,
                width: 2.0 * radius,
                depth: total,
                round: false,
                region: Region {
                    lateral_axis: side,
                    lateral: (0.0, 0.0),
                    along: (-length * 0.5, length * 0.5),
                },
            });
            // Across the axis, at several heights along the straight wall.
            for h in 0..heights {
                let fraction = if heights == 1 {
                    0.0
                } else {
                    h as f64 / (heights - 1) as f64 - 0.5
                };
                let height = fraction * length * 0.5;
                builder.add(&Hold {
                    name: &format!(
                        "side {k} height {h} radial {}",
                        if sign > 0.0 { "+" } else { "-" }
                    ),
                    origin: Vec3::new(0.0, 0.0, height),
                    closing,
                    approach: side.scaled(sign),
                    width: 2.0 * radius,
                    depth: 2.0 * radius,
                    round: true,
                    region: Region {
                        lateral_axis: Vec3::Z,
                        lateral: (-length * 0.5 - height, length * 0.5 - height),
                        along: (0.0, 0.0),
                    },
                });
            }
        }
    }
    // A cylinder's flat ends oppose each other; a capsule's do not.
    if !is_capsule {
        // The square inscribed in the end disc: a pad pressing on the end is
        // credited with no more than that, which under-states the contact.
        let half = radius * std::f64::consts::FRAC_1_SQRT_2;
        for k in 0..angular {
            let angle = 2.0 * PI * k as f64 / angular as f64;
            let approach = Vec3::new(angle.cos(), angle.sin(), 0.0);
            builder.add(&Hold {
                name: &format!("ends approach {k}"),
                origin: Vec3::ZERO,
                closing: Vec3::Z,
                approach,
                width: length,
                depth: 2.0 * radius,
                round: false,
                region: Region {
                    lateral_axis: Vec3::Z.cross(&approach),
                    lateral: (-half, half),
                    along: (-half, half),
                },
            });
        }
    }
}

/// Grasps of a sphere: across a diameter, along the part's own axes.
fn sphere_grasps(builder: &mut Builder<'_>, radius: f64) {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    for (c, closing) in axes.iter().enumerate() {
        for (a, axis) in axes.iter().enumerate() {
            if a == c {
                continue;
            }
            for sign in [1.0, -1.0] {
                let approach = axis.scaled(sign);
                builder.add(&Hold {
                    name: &format!(
                        "sphere close {} approach {}",
                        axis_name(closing),
                        axis_name(&approach)
                    ),
                    origin: Vec3::ZERO,
                    closing: *closing,
                    approach,
                    width: 2.0 * radius,
                    depth: 2.0 * radius,
                    round: true,
                    region: Region::point(),
                });
            }
        }
    }
}

/// Enumerates the antipodal grasps of a world body.
///
/// The body must have exactly one geometry and it must be a box, cylinder,
/// capsule or sphere. Candidates come back in a fixed order, so the same part
/// always yields the same list.
///
/// # Errors
/// [`DomainError::Unsupported`] when the body has several geometries or a mesh:
/// neither has a single set of opposing faces to write grasps down from, and
/// returning an empty list would read as "this part cannot be held".
pub fn synthesize(
    body: &WorldBody,
    options: &SynthesisOptions,
) -> DomainResult<Vec<GraspCandidate>> {
    let geometry = body.single_geometry().ok_or_else(|| {
        DomainError::unsupported(format!(
            "body '{}' has {} geometries; grasp synthesis needs exactly one primitive",
            body.name,
            body.geometries.len()
        ))
    })?;
    geometry
        .shape
        .validate(&format!("body '{}' shape", body.name))?;
    let mut builder = Builder {
        pose: body.pose.compose(&geometry.origin),
        options,
        out: Vec::new(),
    };
    match &geometry.shape {
        Shape::Box { size } => box_grasps(&mut builder, size),
        Shape::Cylinder { radius, length } => axial_grasps(&mut builder, *radius, *length, false),
        Shape::Capsule { radius, length } => axial_grasps(&mut builder, *radius, *length, true),
        Shape::Sphere { radius } => sphere_grasps(&mut builder, *radius),
        Shape::Mesh { .. } => {
            return Err(DomainError::unsupported(format!(
                "body '{}' is a mesh; grasp synthesis supports boxes, cylinders, capsules and \
                 spheres only",
                body.name
            )));
        }
    }
    Ok(builder.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::{GeometryInstance, PhysicalProperties};

    fn part(shape: Shape, pose: Transform) -> WorldBody {
        WorldBody::manipulable(
            "p",
            "part",
            pose,
            GeometryInstance::visual("g", "body", shape),
            PhysicalProperties::default(),
        )
    }

    fn check_invariants(candidates: &[GraspCandidate]) {
        let mut labels = std::collections::BTreeSet::new();
        for c in candidates {
            assert!(
                labels.insert(c.label.clone()),
                "duplicate label {}",
                c.label
            );
            assert!((c.approach.norm() - 1.0).abs() < 1e-12, "{}", c.label);
            assert!((c.closing.norm() - 1.0).abs() < 1e-12, "{}", c.label);
            assert!(c.approach.dot(&c.closing).abs() < 1e-12, "{}", c.label);
            assert!(c.orientation().is_ok(), "{}", c.label);
            assert!((c.contact_a().distance(&c.contact_b()) - c.width).abs() < 1e-12);
        }
    }

    #[test]
    fn a_box_yields_every_face_pair_and_approach() {
        let body = part(
            Shape::Box {
                size: Vec3::new(0.04, 0.06, 0.03),
            },
            Transform::IDENTITY,
        );
        let candidates = synthesize(&body, &SynthesisOptions::default()).unwrap();
        check_invariants(&candidates);
        // 3 closing axes x 4 approaches x 2 jaw assignments, plus shallow variants.
        assert!(candidates.len() >= 24, "{}", candidates.len());
        let widths: std::collections::BTreeSet<String> = candidates
            .iter()
            .map(|c| format!("{:.3}", c.width))
            .collect();
        assert_eq!(widths.len(), 3, "one width per face pair: {widths:?}");
    }

    #[test]
    fn box_contacts_lie_on_opposite_faces() {
        let pose = Transform::new(
            Vec3::new(0.4, -0.1, 0.5),
            Quat::from_axis_angle(&Vec3::Z, 0.6).unwrap(),
        );
        let body = part(
            Shape::Box {
                size: Vec3::new(0.04, 0.04, 0.04),
            },
            pose,
        );
        for c in synthesize(&body, &SynthesisOptions::default()).unwrap() {
            for contact in [c.contact_a(), c.contact_b()] {
                let local = pose.inverse().apply(&contact);
                let on_face = [local.x, local.y, local.z]
                    .iter()
                    .any(|v| (v.abs() - 0.02).abs() < 1e-9);
                assert!(on_face, "{}: contact {local:?} is not on a face", c.label);
            }
        }
    }

    #[test]
    fn a_shallow_variant_keeps_the_fingertips_inside_the_far_face() {
        let options = SynthesisOptions::default();
        let body = part(
            Shape::Box {
                size: Vec3::new(0.04, 0.04, 0.04),
            },
            Transform::IDENTITY,
        );
        let candidates = synthesize(&body, &options).unwrap();
        let shallow: Vec<_> = candidates
            .iter()
            .filter(|c| c.label.contains("shallow"))
            .collect();
        assert!(!shallow.is_empty());
        for c in shallow {
            // Fingertips reach half a finger depth past the grasp centre.
            let tip = c.center.add(&c.approach.scaled(options.finger_depth * 0.5));
            assert!(tip.dot(&c.approach) < 0.02, "{}: tip protrudes", c.label);
            assert!(c.center_offset > 0.0);
        }
    }

    #[test]
    fn a_cylinder_is_held_across_its_diameter_or_by_its_ends() {
        let body = part(
            Shape::Cylinder {
                radius: 0.015,
                length: 0.09,
            },
            Transform::IDENTITY,
        );
        let candidates = synthesize(&body, &SynthesisOptions::default()).unwrap();
        check_invariants(&candidates);
        assert!(candidates.iter().any(|c| (c.width - 0.03).abs() < 1e-12));
        assert!(candidates.iter().any(|c| (c.width - 0.09).abs() < 1e-12));
    }

    #[test]
    fn round_contacts_lie_on_the_surface_with_opposing_normals() {
        // Fat enough that a centred hold is deeper than the fingers would
        // like: the closing line must stay on the axis all the same.
        let radius = 0.03;
        let body = part(
            Shape::Cylinder {
                radius,
                length: 0.09,
            },
            Transform::IDENTITY,
        );
        for c in synthesize(&body, &SynthesisOptions::default()).unwrap() {
            if (c.width - 2.0 * radius).abs() > 1e-12 {
                continue;
            }
            for (contact, normal) in [(c.contact_a(), c.normal_a), (c.contact_b(), c.normal_b)] {
                let radial = Vec3::new(contact.x, contact.y, 0.0);
                assert!(
                    (radial.norm() - radius).abs() < 1e-12,
                    "{}: {contact:?}",
                    c.label
                );
                assert!(
                    radial.scaled(1.0 / radius).distance(&normal) < 1e-12,
                    "{}",
                    c.label
                );
            }
        }
    }

    #[test]
    fn a_capsule_has_no_end_grasps() {
        let body = part(
            Shape::Capsule {
                radius: 0.02,
                length: 0.08,
            },
            Transform::IDENTITY,
        );
        let candidates = synthesize(&body, &SynthesisOptions::default()).unwrap();
        check_invariants(&candidates);
        assert!(candidates.iter().all(|c| (c.width - 0.04).abs() < 1e-12));
    }

    #[test]
    fn a_sphere_is_held_across_a_diameter() {
        let body = part(Shape::Sphere { radius: 0.025 }, Transform::IDENTITY);
        let candidates = synthesize(&body, &SynthesisOptions::default()).unwrap();
        check_invariants(&candidates);
        assert!(candidates
            .iter()
            .all(|c| (c.width - 0.05).abs() < 1e-12 && c.center_offset < 1e-12));
    }

    #[test]
    fn contact_patches_have_the_expected_shape() {
        let options = SynthesisOptions::default();
        let count = |shape: Shape, label: &str| {
            let body = part(shape, Transform::IDENTITY);
            let all = synthesize(&body, &options).unwrap();
            all.iter()
                .find(|c| c.label == label)
                .unwrap_or_else(|| panic!("{label}"))
                .contacts
                .len()
        };
        // Flat faces: a four-cornered patch per pad.
        assert_eq!(
            count(
                Shape::Box {
                    size: Vec3::new(0.04, 0.04, 0.04)
                },
                "box close +x approach -z"
            ),
            8
        );
        // A flat pad on a curved wall: a line per pad.
        assert_eq!(
            count(
                Shape::Cylinder {
                    radius: 0.015,
                    length: 0.09
                },
                "side 0 approach -z"
            ),
            4
        );
        assert_eq!(
            count(
                Shape::Cylinder {
                    radius: 0.015,
                    length: 0.09
                },
                "side 0 height 1 radial +"
            ),
            4
        );
        assert_eq!(
            count(
                Shape::Cylinder {
                    radius: 0.015,
                    length: 0.09
                },
                "ends approach 0"
            ),
            8
        );
        // A sphere: a point per pad.
        assert_eq!(
            count(
                Shape::Sphere { radius: 0.02 },
                "sphere close +x approach -z"
            ),
            2
        );
    }

    #[test]
    fn contact_patches_stay_within_both_the_pad_and_the_part() {
        let options = SynthesisOptions::default();
        let size = Vec3::new(0.04, 0.07, 0.03);
        let body = part(Shape::Box { size }, Transform::IDENTITY);
        for c in synthesize(&body, &options).unwrap() {
            assert_eq!(c.contacts.len(), 8, "{}", c.label);
            let lateral = c.closing.cross(&c.approach);
            for contact in &c.contacts {
                let p = contact.point;
                for (v, half) in [(p.x, size.x), (p.y, size.y), (p.z, size.z)] {
                    assert!(
                        v.abs() <= half * 0.5 + 1e-12,
                        "{}: {p:?} leaves the part",
                        c.label
                    );
                }
                let from_center = p.sub(&c.center);
                assert!(from_center.dot(&lateral).abs() <= options.finger_width * 0.5 + 1e-12);
                assert!(from_center.dot(&c.approach).abs() <= options.finger_depth * 0.5 + 1e-12);
                assert!((from_center.dot(&c.closing).abs() - c.width * 0.5).abs() < 1e-12);
                // The normal points away from the part on the side it is on.
                assert!(contact.normal.dot(&from_center) > 0.0, "{}", c.label);
            }
        }
    }

    #[test]
    fn a_mesh_is_refused_with_a_reason() {
        let body = part(
            Shape::Mesh {
                resource_id: "m".into(),
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
            Transform::IDENTITY,
        );
        let error = synthesize(&body, &SynthesisOptions::default()).expect_err("mesh");
        assert!(error.to_string().contains("mesh"), "{error}");
    }

    #[test]
    fn synthesis_is_deterministic() {
        let body = part(
            Shape::Cylinder {
                radius: 0.015,
                length: 0.09,
            },
            Transform::IDENTITY,
        );
        let options = SynthesisOptions::default();
        assert_eq!(
            synthesize(&body, &options).unwrap(),
            synthesize(&body, &options).unwrap()
        );
    }
}
