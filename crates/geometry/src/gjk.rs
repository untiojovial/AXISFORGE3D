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
//! The true distance between the cores always lies between two numbers the
//! iteration has in hand: `|v|`, the distance to an actual point of `A - B`,
//! and `v.w / |v|`, the offset of a supporting plane that `A - B` lies wholly
//! beyond. The loop stops in one of three ways.
//!
//! 1. **The bracket closes.** When `|v|` minus the best lower bound found so
//!    far is within `RELATIVE_TOLERANCE * max(|v|, |w|)`, the answer is `|v|`.
//!    The tolerance scales with the largest support point because the lower
//!    bound is computed from it, and its rounding grows with its magnitude,
//!    not with the distance.
//! 2. **The iteration stops making progress.** `|v|` falls strictly at every
//!    exact step. Near contact with a large body it can stop falling while the
//!    bracket is still open: `v` is a combination of points tens of metres out,
//!    its rounding tilts it by `eps * |w| / |v|`, and that tilt alone moves the
//!    lower bound by more than the tolerance. The step the next support point
//!    asks for is then below double precision, and the loop would repeat it to
//!    the iteration limit. It stops instead and reports the **lower bound**,
//!    which is certified and errs toward less clearance, never more.
//! 3. **The iteration limit.** Reported as `IterationLimit`, with the lower
//!    bound, so a caller knows the result is not a proof.
//!
//! `tests/near_contact.rs` holds this to account: a tilted box from 10 cm down
//! to 0.1 micrometre above a 100 m floor is never overestimated, is conclusive
//! and at most a micrometre short for every gap of a micrometre or more, and
//! never needs more than 16 iterations. Removing the progress check fails it,
//! and so does the GJK this replaces, which ran to the iteration limit in 464
//! of 2000 such cases, 74 of them at gaps between 1 mm and 10 cm. Keeping the
//! best lower bound rather than the latest is sound and can only tighten the
//! bracket, but no test here distinguishes it from using the latest one.
//! Below a micrometre at that scale the rounding exceeds the gap, and an
//! inconclusive answer is the correct one.
//!
//! A flat simplex is handled structurally rather than by a test. The simplex
//! solver enumerates every sub-simplex and skips any whose Gram matrix is
//! singular at the simplex's own scale, so a flat tetrahedron falls back to its
//! faces by construction; "the origin is inside" is only ever concluded from a
//! subset that passed that check. Read as "inside", a flat simplex would invent
//! a contact between bodies nowhere near each other. No test here pins the
//! threshold down: it is defence in depth against a future solver change.

use axisforge_domain::Vec3;

use crate::support::ConvexBody;

/// Largest number of support iterations before giving up.
///
/// GJK on a polytope converges in a number of iterations bounded by its vertex
/// count, and the cores here have at most a few dozen; this limit exists to
/// bound the degenerate case, not the normal one.
pub const MAX_ITERATIONS: usize = 64;

/// Relative tolerance of the convergence test, dimensionless.
///
/// The test bounds the gap between the distance's upper bound `|v|` and its
/// lower bound `v.w / |v|`, so on convergence the reported distance is within
/// `RELATIVE_TOLERANCE * max(|v|, |w|)` of the true one: under a nanometre in
/// a metre-scale cell. A tighter value buys nothing measurable and fails to
/// converge on nearly parallel faces, where each iteration swaps between
/// almost equivalent corners and improves by about 1e-13 (observed between a
/// UR5's flange box and a gripper palm 5 mm away).
const RELATIVE_TOLERANCE: f64 = 1.0e-10;

/// Relative decrease of |v| below which an iteration counts as making no progress.
const STALL_TOLERANCE: f64 = 4.0 * f64::EPSILON;

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
    distance_to_tolerance(a, b, 0.0)
}

/// [`distance`], stopping as soon as the distance is known to within
/// `tolerance` metres.
///
/// The reported distance is then the upper end of the bracket: the true
/// distance is at least `distance - tolerance`. On curved cores, where the
/// iteration converges slowly, a tolerance of a micrometre costs a fraction
/// of the iterations of the full-precision query; a caller that only needs a
/// certified lower bound subtracts the tolerance itself.
pub fn distance_to_tolerance(a: &ConvexBody, b: &ConvexBody, tolerance: f64) -> Distance {
    let tolerance = if tolerance.is_finite() {
        tolerance.max(0.0)
    } else {
        0.0
    };
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
    // The support plane through `w` with normal `v` separates the origin from
    // the whole difference body, so `v.w / |v|` never exceeds the true core
    // distance. Together with `|v|`, which never undershoots it, this brackets
    // the answer even when the loop does not converge.
    let mut lower_bound = 0.0_f64;
    // Largest support-point magnitude seen: the scale of the bound's rounding.
    let mut reach = 0.0_f64;
    // Set when the iteration stops at a fixed point rather than a closed bracket.
    let mut stalled = false;

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
        let upper = v.norm();
        if !simplex.is_empty() && upper > 0.0 {
            lower_bound = lower_bound.max(v.dot(&next.w) / upper);
            reach = reach.max(next.w.norm());
            // The true core distance lies in [lower_bound, upper]. Testing the
            // best bound found so far, not only this iteration's, is what
            // stops a loop that cycles between nearly equivalent simplices:
            // each step there may be poor while an earlier one already closed
            // the gap. The tolerance scales with the largest support point
            // seen, because the bound is computed from it and its rounding
            // error grows with its magnitude, not with the distance.
            if upper - lower_bound <= (RELATIVE_TOLERANCE * upper.max(reach)).max(tolerance) {
                status = GjkStatus::Separated;
                break;
            }
        }

        // Before the first support point, `upper` is the seed, not a distance.
        let measured = !simplex.is_empty();
        simplex.push(next);
        closest = solve_simplex(&mut simplex);

        // |v| falls strictly at every exact GJK step. Once it stops falling,
        // the step the support point asks for is below what double precision
        // can represent, and the remaining gap between the bounds is rounding:
        // continuing would repeat this step until the iteration limit.
        if measured && upper > 0.0 && closest.v.norm() >= upper * (1.0 - STALL_TOLERANCE) {
            stalled = true;
            break;
        }

        debug_assert!(
            closest.contains_origin || simplex.len() <= 3,
            "a simplex that does not hold the origin has at most three points"
        );
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

    conclude(
        &closest,
        a.inflation,
        b.inflation,
        lower_bound,
        iterations,
        status,
        stalled,
    )
}

/// Turns the loop's final state into a reported distance.
///
/// See the module documentation for which bound is reported when.
fn conclude(
    closest: &Closest,
    inflation_a: f64,
    inflation_b: f64,
    lower_bound: f64,
    iterations: usize,
    status: GjkStatus,
    stalled: bool,
) -> Distance {
    let inflation = inflation_a + inflation_b;
    let core_distance = closest.v.norm();
    let separation = core_distance - inflation;
    if separation <= 0.0 {
        // `|v|` is the distance to an actual point of the difference body, so
        // the true core distance is no larger. If even that is within the
        // inflation radii the bodies overlap for certain, converged or not.
        return Distance {
            distance: 0.0,
            witness_a: closest.pa,
            witness_b: closest.pb,
            direction: None,
            iterations,
            status: GjkStatus::Intersecting,
        };
    }
    if stalled || status == GjkStatus::IterationLimit {
        // The bracket [lower_bound, |v|] did not close to tolerance. Report the
        // lower bound: it is certified, and it errs toward less clearance,
        // never more. Stalled at a fixed point with that bound clear of the
        // inflation radii, the result is still a proof of separation.
        let certified = lower_bound - inflation;
        let conclusive = stalled && certified > 0.0;
        return Distance {
            distance: certified.max(0.0),
            witness_a: closest.pa,
            witness_b: closest.pb,
            direction: None,
            iterations,
            status: if conclusive {
                GjkStatus::Separated
            } else {
                GjkStatus::IterationLimit
            },
        };
    }

    // `v` is a point of `A - B`, so it points from the second body toward the
    // first. Each witness moves toward the *other* body by its own inflation,
    // which puts it on the real surface rather than on the core.
    let toward_a = closest.v.scaled(1.0 / core_distance);
    Distance {
        distance: separation,
        witness_a: closest.pa.sub(&toward_a.scaled(inflation_a)),
        witness_b: closest.pb.add(&toward_a.scaled(inflation_b)),
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

        // A tetrahedron whose barycentric projection of the origin is valid
        // holds the origin, and the tetrahedron lies inside the difference
        // body. This is decided by the weights, never by how small the
        // projected point came out: at a 100 m scale rounding leaves it at
        // 1e-8, and an absolute threshold then let the loop keep four points
        // and add a fifth.
        if indices.len() == 4 {
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
        contains_origin: origin_enclosed,
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

    /// The simplex GJK built for the 100 m ground slab against a UR5 link
    /// resting on it: a nearly flat tetrahedron holding the origin, whose
    /// projected point rounding left at 3.4e-8. It used to be kept whole as
    /// "not containing the origin", and the next iteration indexed past the
    /// end of a five-point simplex.
    #[test]
    // The coordinates are the recorded ones, digit for digit.
    #[allow(clippy::unreadable_literal)]
    fn a_wide_flat_tetrahedron_around_the_origin_is_recognised() {
        let points = [
            Vec3::new(50.853645460368035, 50.156649895095825, 0.024034969295091562),
            Vec3::new(-49.21822877073198, -49.9423144831419, -0.002106857216041688),
            Vec3::new(-49.21992438131471, 50.15664995097518, -0.012463939106164501),
            Vec3::new(
                50.81840156173837,
                -49.902237914419175,
                -0.044223558818618766,
            ),
        ];
        let mut simplex: Vec<Vertex> = points
            .iter()
            .map(|w| Vertex {
                w: *w,
                pa: Vec3::ZERO,
                pb: w.scaled(-1.0),
            })
            .collect();
        let closest = solve_simplex(&mut simplex);
        assert!(closest.contains_origin, "{:?}", closest.v);
        assert!(closest.v.norm() < 1.0e-6);
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
        // A centimetre-scale link above a hundred-metre ground slab: support
        // points seventy metres out while `|v|` is a few centimetres. Tilted,
        // so that no support point repeats and the loop has to stop on the
        // bracket rather than on a duplicate vertex.
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
        // ever recurs: the bracket is the only thing that can stop the loop.
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
    fn overlap_through_inflation_is_conclusive_even_on_a_curved_pair() {
        // Observed in the pick planner: a capsule whose core passes 19 mm from
        // a cylinder, with a 50 mm radius. The cores never touch, and against
        // a curved rim the iteration closes in too slowly to converge, so the
        // result used to come back as IterationLimit. But `|v|` already proves
        // the overlap: it bounds the core distance from above.
        let capsule = ConvexBody::capsule(
            0.05,
            0.28,
            Transform::new(
                Vec3::new(
                    0.376_083_084_505_546_8,
                    0.092_033_812_386_882_9,
                    0.440_668_934_568_045_9,
                ),
                Quat::new(
                    -0.103_841_148_989_534_02,
                    0.861_185_269_309_956_4,
                    0.059_565_328_426_179_23,
                    0.493_992_833_297_764_2,
                ),
            ),
        );
        let cylinder = ConvexBody::new(
            ConvexCore::Cylinder {
                radius: 0.015,
                half_length: 0.045,
            },
            Transform::from_translation(Vec3::new(0.46, 0.12, 0.465)),
        );
        let result = distance(&capsule, &cylinder);
        assert_eq!(result.status, GjkStatus::Intersecting, "{result:?}");
        assert!(result.distance.abs() < 1e-15);
    }

    #[test]
    fn nearly_parallel_faces_converge() {
        // The case that did not converge with a 1e-12 tolerance: a 1 cm cube
        // and a gripper palm, faces 5 mm apart and parallel to within 1e-8 rad.
        let small = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(0.005, 0.005, 0.005),
            },
            Transform::new(
                Vec3::new(
                    0.257_500_002_241_321_8,
                    -0.099_999_986_922_729_07,
                    0.440_000_010_595_935_9,
                ),
                Quat::new(
                    0.999_999_999_999_998_7,
                    1.634_750_557_144_571e-8,
                    4.047_316_538_715_725e-8,
                    -2.334_403_559_238_481e-8,
                ),
            ),
        );
        let palm = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(0.02, 0.052, 0.025),
            },
            Transform::new(
                Vec3::new(
                    0.292_500_002_241_321_67,
                    -0.099_999_985_778_526_17,
                    0.440_000_013_428_935_13,
                ),
                Quat::new(
                    0.499_999_983_264_13,
                    0.499_999_976_265_151_64,
                    0.500_000_040_079_904_3,
                    0.500_000_000_390_811_5,
                ),
            ),
        );
        let result = distance(&small, &palm);
        assert_eq!(result.status, GjkStatus::Separated, "{result:?}");
        assert!((result.distance - 0.005).abs() < 1e-9, "{result:?}");
        assert!(result.iterations < 16, "took {}", result.iterations);
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
