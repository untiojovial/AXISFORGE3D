//! Distance between two convex bodies, by the Gilbert-Johnson-Keerthi algorithm.
//!
//! GJK walks the Minkowski difference `A - B` toward the origin. Each iteration
//! asks both bodies for a support point, adds the difference to a simplex of at
//! most four points, and keeps the sub-simplex closest to the origin. The
//! distance between the bodies is the distance from the origin to that
//! difference; the origin lying *inside* it means the bodies overlap.
//!
//! # What this does not give you
//!
//! There is no Expanding Polytope Algorithm here, so a penetrating pair reports
//! distance zero and no penetration depth. That is deliberate: this engine uses
//! distance for clearance and for grasp feasibility, where the useful question
//! is *how much room is left*, and a tool that invented a depth it had not
//! computed would be worse than one that says it does not know.
//!
//! # Numerics
//!
//! Two scale-related faults are easy to write here and both were observed
//! before the tolerances below were made relative to what is being measured:
//!
//! 1. **The convergence test.** The residual `|v|^2 - v.w` is computed by
//!    subtraction, so its rounding error is proportional to `|v| * |w|`, not to
//!    `|v|^2`. Testing it against `|v|^2` alone makes the test unsatisfiable
//!    whenever the bodies are small but far from the origin — a centimetre-scale
//!    link against a hundred-metre ground slab has support points seventy
//!    metres out while `|v|` is centimetres, leaving a residual several times
//!    the threshold no matter how many iterations run.
//!
//! 2. **The degenerate simplex.** When the iteration cannot converge it keeps
//!    adding points, and the fourth one lands coplanar with the first three. A
//!    tetrahedron solver that does not notice reads a flat simplex as "the
//!    origin is inside" and reports a contact between bodies that are nowhere
//!    near each other. Inventing a collision is the dangerous direction of
//!    this error: a user trusts a clear report less readily than a warning.
//!
//! The convergence test below scales its threshold by `|v| * |w|` as well as
//! `|v|^2`, and `a_small_tilted_body_far_from_the_origin_still_converges` holds
//! it to that: reverting the scaling makes that test and the tilted-cylinder
//! test run to the iteration limit. The tilt is what makes them bite — an
//! axis-aligned capsule returns a duplicate support point and terminates on
//! that check before the residual is ever consulted.
//!
//! The second fault is handled structurally rather than by a test. The simplex
//! solver enumerates every sub-simplex and skips any whose Gram matrix is
//! singular at the simplex's own scale, so a flat tetrahedron falls back to its
//! faces by construction; "the origin is inside" is only ever concluded from a
//! subset that passed that check. Removing the check does not change any result
//! this crate's tests measure, and even with both faults injected the solver
//! reports `IterationLimit` rather than a contact. So no test here pins the
//! degeneracy threshold down: it stands as defence in depth against a future
//! solver change, not as a guarded invariant.

use axisforge_domain::Vec3;

use crate::support::ConvexBody;

/// Largest number of support iterations before giving up.
///
/// GJK on a polytope converges in a number of iterations bounded by its vertex
/// count, and the cores here have at most a few dozen; this limit exists to
/// bound the degenerate case, not the normal one.
pub const MAX_ITERATIONS: usize = 64;

/// Relative tolerance of the convergence test, dimensionless.
const RELATIVE_TOLERANCE: f64 = 1.0e-12;

/// Absolute floor for the convergence scale, square metres.
///
/// Keeps the test meaningful when both bodies sit on the origin and every
/// magnitude in the expression is zero.
const ABSOLUTE_TOLERANCE: f64 = 1.0e-20;

/// Relative threshold below which a Gram determinant counts as singular.
///
/// Scaled by the simplex's own size where it is used, since the determinant of
/// `d` edge vectors grows as `length^(2d)` and an absolute threshold would
/// dismiss a millimetre simplex while never rejecting a kilometre one.
const DEGENERACY_TOLERANCE: f64 = 1.0e-14;

/// What the iteration concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GjkStatus {
    /// The bodies are apart; the reported distance is positive.
    Separated,
    /// The bodies touch or overlap. No penetration depth is computed.
    Intersecting,
    /// The iteration limit was reached. The reported distance is a lower bound.
    ///
    /// This is reported rather than swallowed: a caller deciding whether a
    /// motion is safe needs to know the number it was handed is not final.
    IterationLimit,
}

impl GjkStatus {
    /// `true` when the result is a converged, trustworthy distance.
    pub fn is_conclusive(self) -> bool {
        !matches!(self, Self::IterationLimit)
    }
}

/// The outcome of a distance query between two convex bodies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distance {
    /// Separation in metres, inflation included. Zero when the bodies overlap.
    pub distance: f64,
    /// The closest point on the first body, in world coordinates.
    pub witness_a: Vec3,
    /// The closest point on the second body, in world coordinates.
    pub witness_b: Vec3,
    /// Unit vector from the first body toward the second, when they are apart.
    ///
    /// Together with the witnesses this is the direction the first body would
    /// have to travel to close the gap.
    ///
    /// `None` when they overlap or touch, where no separating direction exists.
    pub direction: Option<Vec3>,
    /// How many support iterations were used.
    pub iterations: usize,
    /// What the iteration concluded.
    pub status: GjkStatus,
}

impl Distance {
    /// `true` when the bodies are closer than `threshold` metres.
    pub fn within(&self, threshold: f64) -> bool {
        self.distance <= threshold
    }
}

/// One vertex of the simplex: a point of `A - B` and the pair that produced it.
#[derive(Debug, Clone, Copy)]
struct Vertex {
    /// The Minkowski-difference point.
    w: Vec3,
    /// The supporting point on the first body.
    pa: Vec3,
    /// The supporting point on the second body.
    pb: Vec3,
}

/// The closest point to the origin on the current simplex, with its witnesses.
#[derive(Debug, Clone, Copy)]
struct Closest {
    v: Vec3,
    pa: Vec3,
    pb: Vec3,
    /// `true` when the origin lies within a full-dimensional simplex.
    contains_origin: bool,
}

/// Computes the distance between two convex bodies.
///
/// The inflation radii of both bodies are subtracted from the distance between
/// their cores, which is exact: the Minkowski sum of a convex body and a ball
/// of radius `r` is that body's surface pushed out by `r` in every direction.
pub fn distance(a: &ConvexBody, b: &ConvexBody) -> Distance {
    let inflation = a.inflation + b.inflation;

    // Seed the search with the vector between the bodies' origins. Any
    // direction works, but this one usually lands within a couple of
    // iterations of the answer.
    let seed = b.pose.translation.sub(&a.pose.translation);
    let mut direction = if seed.norm_squared() > 0.0 {
        seed
    } else {
        Vec3::X
    };

    let mut simplex: Vec<Vertex> = Vec::with_capacity(4);
    let mut closest = Closest {
        v: direction,
        pa: a.pose.translation,
        pb: b.pose.translation,
        contains_origin: false,
    };
    let mut iterations = 0;
    let mut status = GjkStatus::IterationLimit;

    while iterations < MAX_ITERATIONS {
        iterations += 1;
        let next = support(a, b, &direction.scaled(-1.0));

        // The same vertex coming back means the simplex already spans the
        // closest feature; no further progress is possible.
        if simplex
            .iter()
            .any(|existing| existing.w.distance(&next.w) <= 0.0)
        {
            status = GjkStatus::Separated;
            break;
        }

        let v = closest.v;
        let vv = v.norm_squared();
        // `vv - v.w` is a subtraction of like-sized quantities, so its error
        // tracks `|v| * |w|`. Scaling the threshold by only `vv` makes the
        // test unsatisfiable for small bodies far from the origin.
        let scale = vv.max(v.norm() * next.w.norm()).max(ABSOLUTE_TOLERANCE);
        if !simplex.is_empty() && vv - v.dot(&next.w) <= RELATIVE_TOLERANCE * scale {
            status = GjkStatus::Separated;
            break;
        }

        simplex.push(next);
        closest = solve_simplex(&mut simplex);

        if closest.contains_origin {
            return Distance {
                distance: 0.0,
                witness_a: closest.pa,
                witness_b: closest.pb,
                direction: None,
                iterations,
                status: GjkStatus::Intersecting,
            };
        }

        direction = closest.v;
        if direction.norm_squared() <= ABSOLUTE_TOLERANCE {
            // The origin lies on the simplex: the cores touch exactly.
            return Distance {
                distance: 0.0,
                witness_a: closest.pa,
                witness_b: closest.pb,
                direction: None,
                iterations,
                status: GjkStatus::Intersecting,
            };
        }
    }

    let core_distance = closest.v.norm();
    let separation = core_distance - inflation;
    if separation <= 0.0 {
        return Distance {
            distance: 0.0,
            witness_a: closest.pa,
            witness_b: closest.pb,
            direction: None,
            iterations,
            status: if status == GjkStatus::IterationLimit {
                GjkStatus::IterationLimit
            } else {
                GjkStatus::Intersecting
            },
        };
    }

    // `v` is a point of `A - B`, so it points from the second body toward the
    // first. Each witness moves toward the *other* body by its own inflation,
    // which puts it on the real surface rather than on the core.
    let toward_a = closest.v.scaled(1.0 / core_distance);
    Distance {
        distance: separation,
        witness_a: closest.pa.sub(&toward_a.scaled(a.inflation)),
        witness_b: closest.pb.add(&toward_a.scaled(b.inflation)),
        direction: Some(toward_a.scaled(-1.0)),
        iterations,
        status,
    }
}

/// `true` when the two bodies are within `threshold` metres of each other.
///
/// A convenience over [`distance`]; it runs the same query.
pub fn within(a: &ConvexBody, b: &ConvexBody, threshold: f64) -> bool {
    distance(a, b).within(threshold)
}

fn support(a: &ConvexBody, b: &ConvexBody, direction: &Vec3) -> Vertex {
    let pa = a.core_support(direction);
    let pb = b.core_support(&direction.scaled(-1.0));
    Vertex {
        w: pa.sub(&pb),
        pa,
        pb,
    }
}

/// Reduces `simplex` to the sub-simplex closest to the origin.
///
/// Every non-empty subset is tried. A subset is a candidate only when the
/// origin's projection onto its affine hull has non-negative barycentric
/// coordinates — that is, when the projection lands inside the subset rather
/// than past one of its edges — and when its Gram matrix is non-singular at the
/// simplex's own scale. A flat tetrahedron therefore has no valid 4-point
/// subset and falls back to its faces, instead of being mistaken for one that
/// encloses the origin.
fn solve_simplex(simplex: &mut Vec<Vertex>) -> Closest {
    let points: Vec<Vec3> = simplex.iter().map(|v| v.w).collect();
    let count = points.len();
    debug_assert!(
        (1..=4).contains(&count),
        "the simplex holds one to four points"
    );

    // The characteristic length of the simplex, used to decide what counts as
    // singular. Without it the thresholds would be absolute, and a millimetre
    // simplex would be dismissed as degenerate while a kilometre one would
    // never be.
    let scale = points
        .iter()
        .flat_map(|p| points.iter().map(move |q| p.distance(q)))
        .fold(0.0_f64, f64::max)
        .max(points.iter().map(Vec3::norm).fold(0.0, f64::max));

    let mut best: Option<(f64, Vec<usize>, Vec<f64>)> = None;
    let mut origin_enclosed = false;

    for mask in 1_u8..(1_u8 << count) {
        let indices: Vec<usize> = (0..count).filter(|i| mask & (1 << i) != 0).collect();
        let Some(weights) = barycentric_projection(&points, &indices, scale) else {
            continue;
        };
        let point = indices
            .iter()
            .zip(&weights)
            .fold(Vec3::ZERO, |acc, (i, w)| acc.add(&points[*i].scaled(*w)));
        let norm_squared = point.norm_squared();

        // A full-dimensional subset whose projection of the origin lands
        // strictly inside means the origin is enclosed.
        if indices.len() == count && count == 4 {
            origin_enclosed = true;
        }

        if best.as_ref().is_none_or(|(d, _, _)| norm_squared < *d) {
            best = Some((norm_squared, indices, weights));
        }
    }

    let Some((_, indices, weights)) = best else {
        // Every subset was singular, which can only happen when all the points
        // coincide. Fall back to the first vertex, which is a valid answer for
        // a simplex that has collapsed to a point.
        let first = simplex[0];
        simplex.truncate(1);
        return Closest {
            v: first.w,
            pa: first.pa,
            pb: first.pb,
            contains_origin: false,
        };
    };

    let v = indices
        .iter()
        .zip(&weights)
        .fold(Vec3::ZERO, |acc, (i, w)| acc.add(&points[*i].scaled(*w)));
    let pa = indices
        .iter()
        .zip(&weights)
        .fold(Vec3::ZERO, |acc, (i, w)| {
            acc.add(&simplex[*i].pa.scaled(*w))
        });
    let pb = indices
        .iter()
        .zip(&weights)
        .fold(Vec3::ZERO, |acc, (i, w)| {
            acc.add(&simplex[*i].pb.scaled(*w))
        });

    let retained: Vec<Vertex> = indices.iter().map(|i| simplex[*i]).collect();
    *simplex = retained;

    Closest {
        v,
        pa,
        pb,
        contains_origin: origin_enclosed && v.norm_squared() <= f64::EPSILON,
    }
}

/// Projects the origin onto the affine hull of a subset, in barycentric coordinates.
///
/// Returns `None` when the subset is degenerate at `scale`, or when the
/// projection falls outside the subset's convex hull.
fn barycentric_projection(points: &[Vec3], indices: &[usize], scale: f64) -> Option<Vec<f64>> {
    let k = indices.len();
    if k == 1 {
        return Some(vec![1.0]);
    }

    let base = points[indices[0]];
    let edges: Vec<Vec3> = indices[1..].iter().map(|i| points[*i].sub(&base)).collect();
    let dimension = edges.len();

    // Normal equations of the least-squares problem: G t = -E^T base, whose
    // solution is the projection of the origin onto base + span(edges).
    let mut gram = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for (i, ei) in edges.iter().enumerate() {
        for (j, ej) in edges.iter().enumerate() {
            gram[i][j] = ei.dot(ej);
        }
        rhs[i] = -base.dot(ei);
    }

    // The Gram determinant of `d` edges scales as length^(2d), so the
    // singularity threshold has to scale the same way or it would depend on
    // the units the model happens to be in.
    let determinant = determinant(&gram, dimension);
    // `dimension` is 1, 2 or 3, so the exponent is 2, 4 or 6.
    let exponent = i32::try_from(2 * dimension).unwrap_or(6);
    let magnitude = scale.max(f64::MIN_POSITIVE).powi(exponent);
    if determinant.abs() <= DEGENERACY_TOLERANCE * magnitude {
        return None;
    }

    let solution = solve(&gram, &rhs, dimension, determinant)?;
    let mut weights = Vec::with_capacity(k);
    let interior: f64 = solution[..dimension].iter().sum();
    weights.push(1.0 - interior);
    weights.extend_from_slice(&solution[..dimension]);

    // Negative weights mean the projection left the subset through one of its
    // boundaries; a smaller subset covers that case.
    if weights.iter().any(|w| *w < -1.0e-12) {
        return None;
    }
    Some(weights)
}

fn determinant(m: &[[f64; 3]; 3], n: usize) -> f64 {
    match n {
        1 => m[0][0],
        2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
        _ => {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        }
    }
}

/// Solves `m x = rhs` for `n` in 1..=3 by Cramer's rule.
///
/// `determinant` is passed in because the caller has already computed it to
/// decide whether the system is singular.
fn solve(m: &[[f64; 3]; 3], rhs: &[f64; 3], n: usize, determinant: f64) -> Option<[f64; 3]> {
    if determinant == 0.0 {
        return None;
    }
    let mut out = [0.0_f64; 3];
    for column in 0..n {
        let mut replaced = *m;
        for row in 0..n {
            replaced[row][column] = rhs[row];
        }
        out[column] = self::determinant(&replaced, n) / determinant;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{ConvexBody, ConvexCore};
    use axisforge_domain::{Quat, Transform, Vec3};

    fn unit_box(at: Vec3) -> ConvexBody {
        ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            Transform::from_translation(at),
        )
    }

    #[test]
    fn two_separated_boxes_report_the_face_gap() {
        let result = distance(&unit_box(Vec3::ZERO), &unit_box(Vec3::new(3.0, 0.0, 0.0)));
        assert_eq!(result.status, GjkStatus::Separated);
        assert!((result.distance - 2.0).abs() < 1e-9, "{result:?}");
        assert!((result.witness_a.x - 0.5).abs() < 1e-9, "{result:?}");
        assert!((result.witness_b.x - 2.5).abs() < 1e-9, "{result:?}");
        let direction = result.direction.expect("separated bodies have a direction");
        assert!(direction.distance(&Vec3::X) < 1e-9, "{direction:?}");
    }

    #[test]
    fn two_overlapping_boxes_report_zero_without_a_depth() {
        let result = distance(&unit_box(Vec3::ZERO), &unit_box(Vec3::new(0.25, 0.0, 0.0)));
        assert_eq!(result.status, GjkStatus::Intersecting);
        assert!(result.distance.abs() < 1e-12);
        assert!(
            result.direction.is_none(),
            "there is no separating direction inside a contact"
        );
    }

    #[test]
    fn two_spheres_report_their_centre_distance_minus_both_radii() {
        let a = ConvexBody::sphere(0.3, Transform::IDENTITY);
        let b = ConvexBody::sphere(0.2, Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)));
        let result = distance(&a, &b);
        assert!((result.distance - 0.5).abs() < 1e-9, "{result:?}");
        // The witnesses must land on the surfaces, not on the centres.
        assert!(
            result.witness_a.distance(&Vec3::new(0.0, 0.3, 0.0)) < 1e-9,
            "{result:?}"
        );
        assert!(
            result.witness_b.distance(&Vec3::new(0.0, 0.8, 0.0)) < 1e-9,
            "{result:?}"
        );
    }

    #[test]
    fn a_diagonal_gap_is_measured_corner_to_corner() {
        let result = distance(&unit_box(Vec3::ZERO), &unit_box(Vec3::new(2.0, 2.0, 0.0)));
        // Corner (0.5, 0.5) to corner (1.5, 1.5): one metre on each axis.
        assert!(
            (result.distance - 2.0_f64.sqrt()).abs() < 1e-9,
            "{result:?}"
        );
    }

    #[test]
    fn a_capsule_clears_a_box_by_its_radius() {
        let box_body = unit_box(Vec3::ZERO);
        let capsule = ConvexBody::capsule(
            0.1,
            0.4,
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        );
        let result = distance(&box_body, &capsule);
        // Box face at 0.5, capsule surface at 1.0 - 0.1 = 0.9.
        assert!((result.distance - 0.4).abs() < 1e-9, "{result:?}");
    }

    // --- Regression tests for the two scale faults described at the top. ---

    #[test]
    fn a_small_tilted_body_far_from_the_origin_still_converges() {
        // The observed failure: a centimetre-scale link above a hundred-metre
        // ground slab. Support points sit seventy metres out while `|v|` is
        // centimetres, so testing the residual against `|v|^2` alone leaves it
        // several times over threshold and the loop runs to the iteration
        // limit. The *tilt* matters: an axis-aligned capsule returns the same
        // support point twice and terminates on the duplicate check instead,
        // never reaching the residual test at all.
        let slab = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(50.0, 50.0, 0.01),
            },
            Transform::from_translation(Vec3::new(0.0, 0.0, -0.01)),
        );
        let tilt = Quat::from_axis_angle(&Vec3::new(1.0, 0.3, 0.2), 0.7).unwrap();
        let link = ConvexBody::capsule(0.045, 0.25, Transform::new(Vec3::new(0.2, 0.1, 0.4), tilt));
        let result = distance(&slab, &link);
        assert_eq!(
            result.status,
            GjkStatus::Separated,
            "the iteration must converge, not run out: {result:?}"
        );
        assert!(
            result.iterations < 12,
            "convergence should take a handful of iterations, took {}",
            result.iterations
        );
        // The capsule's lowest point: its tilted axis end plus the cap radius.
        let lowest = tilt.rotate(&Vec3::new(0.0, 0.0, -0.125)).z + 0.4 - 0.045;
        assert!(
            (result.distance - lowest).abs() < 1e-9,
            "{result:?} vs {lowest}"
        );
    }

    #[test]
    fn a_tilted_cylinder_against_a_wide_slab_also_converges() {
        // A cylinder's rim is a continuum of support points, so no duplicate
        // ever recurs: the residual test is the only thing that can stop the
        // loop, which makes this the sharpest case for its scaling.
        let slab = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(50.0, 50.0, 0.01),
            },
            Transform::from_translation(Vec3::new(0.0, 0.0, -0.01)),
        );
        let result = distance(
            &slab,
            &ConvexBody::new(
                ConvexCore::Cylinder {
                    radius: 0.04,
                    half_length: 0.1,
                },
                Transform::new(
                    Vec3::new(0.3, 0.17, 0.3),
                    Quat::from_axis_angle(&Vec3::new(1.0, 0.3, 0.2), 0.7).unwrap(),
                ),
            ),
        );
        assert_eq!(result.status, GjkStatus::Separated, "{result:?}");
        assert!(
            result.iterations < 16,
            "took {} iterations",
            result.iterations
        );
        assert!(
            result.distance > 0.19 && result.distance < 0.2,
            "{result:?}"
        );
    }

    #[test]
    fn a_clear_pair_is_never_reported_as_touching() {
        // Inventing a contact is the dangerous direction of any fault here: a
        // user trusts a "clear" report less readily than a warning. Sweep a
        // tilted link across a wide slab at a known clearance and assert every
        // sample comes back clear, converged, and with the right gap.
        let slab = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(50.0, 50.0, 0.01),
            },
            Transform::from_translation(Vec3::new(0.0, 0.0, -0.01)),
        );
        let tilt = Quat::from_axis_angle(&Vec3::new(1.0, 0.3, 0.2), 0.7).unwrap();
        let expected = tilt.rotate(&Vec3::new(0.0, 0.0, -0.1)).z + 0.3 - 0.04;
        for step in 0..40 {
            let x = -2.0 + f64::from(step) * 0.1;
            let link = ConvexBody::capsule(0.04, 0.2, Transform::new(Vec3::new(x, 0.1, 0.3), tilt));
            let result = distance(&slab, &link);
            assert_eq!(result.status, GjkStatus::Separated, "x = {x}: {result:?}");
            assert!(
                (result.distance - expected).abs() < 1e-9,
                "x = {x} reported {} instead of {expected}",
                result.distance
            );
        }
    }

    #[test]
    fn distance_is_symmetric_in_its_arguments() {
        let a = unit_box(Vec3::ZERO);
        let b = ConvexBody::sphere(0.2, Transform::from_translation(Vec3::new(1.7, 0.3, -0.2)));
        let forward = distance(&a, &b);
        let backward = distance(&b, &a);
        assert!(
            (forward.distance - backward.distance).abs() < 1e-9,
            "{} vs {}",
            forward.distance,
            backward.distance
        );
        assert!(forward.witness_a.distance(&backward.witness_b) < 1e-9);
    }

    #[test]
    fn coincident_bodies_do_not_divide_by_zero() {
        let result = distance(&unit_box(Vec3::ZERO), &unit_box(Vec3::ZERO));
        assert_eq!(result.status, GjkStatus::Intersecting);
        assert!(result.distance.abs() < 1e-12);
    }

    #[test]
    fn a_degenerate_zero_size_body_is_handled_as_a_point() {
        let point = ConvexBody::new(
            ConvexCore::Point,
            Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        );
        let result = distance(&unit_box(Vec3::ZERO), &point);
        assert!((result.distance - 1.5).abs() < 1e-9, "{result:?}");
    }

    #[test]
    fn the_within_helper_agrees_with_the_distance() {
        let a = unit_box(Vec3::ZERO);
        let b = unit_box(Vec3::new(3.0, 0.0, 0.0));
        assert!(within(&a, &b, 2.5));
        assert!(!within(&a, &b, 1.5));
    }
}
