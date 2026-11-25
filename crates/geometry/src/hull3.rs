//! Convex hulls of point sets in three dimensions.
//!
//! Collision already treats a mesh as the convex hull of its vertices
//! through a support function, without ever building the hull. Some
//! questions need the hull itself: how much space it encloses, and whether a
//! point lies inside it. Convex decomposition asks both, to decide where
//! cutting a concave part pays off and to check that the pieces still cover
//! the solid.
//!
//! The algorithm is incremental: start from a tetrahedron of extreme points,
//! and for each further point outside the current hull, remove the faces it
//! sees and close the hole with triangles from the horizon to the point. A
//! point within a small tolerance of a face does not see it, so coplanar
//! points are absorbed rather than producing slivers. Every face of the
//! result has all input points on its inner side, within that tolerance.

use std::collections::HashSet;

/// A triangle of a hull, wound counter-clockwise seen from outside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// Indices into the input points.
    pub vertices: [usize; 3],
    /// Unit outward normal.
    pub normal: [f64; 3],
    /// `normal . x` for any point `x` on the face.
    pub offset: f64,
}

/// The convex hull of a point set.
#[derive(Debug, Clone, PartialEq)]
pub struct Hull3 {
    /// The input points; faces index into them.
    pub points: Vec<[f64; 3]>,
    /// The hull's faces.
    pub faces: Vec<Face>,
    /// A point strictly inside the hull.
    pub interior: [f64; 3],
    /// Distance below which a point counts as on a face.
    pub tolerance: f64,
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

impl Hull3 {
    /// The enclosed volume.
    pub fn volume(&self) -> f64 {
        self.faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.vertices.map(|i| self.points[i]);
                dot(
                    &sub(&a, &self.interior),
                    &cross(&sub(&b, &self.interior), &sub(&c, &self.interior)),
                ) / 6.0
            })
            .sum()
    }

    /// `true` when `point` is inside the hull or within `margin` of it.
    pub fn contains(&self, point: &[f64; 3], margin: f64) -> bool {
        self.faces
            .iter()
            .all(|f| dot(&f.normal, point) - f.offset <= margin)
    }
}

/// The face through `a`, `b`, `c` facing away from `inside`, or `None` when
/// the three points are collinear within `tolerance`.
#[allow(clippy::many_single_char_names)] // Corners and normal, as in the formula.
fn face(points: &[[f64; 3]], v: [usize; 3], inside: &[f64; 3], tolerance: f64) -> Option<Face> {
    let [a, b, c] = v.map(|i| points[i]);
    let n = cross(&sub(&b, &a), &sub(&c, &a));
    let length = norm(&n);
    if length <= tolerance * tolerance {
        return None;
    }
    let mut normal = n.map(|x| x / length);
    let mut vertices = v;
    if dot(&normal, &sub(inside, &a)) > 0.0 {
        normal = normal.map(|x| -x);
        vertices = [v[0], v[2], v[1]];
    }
    Some(Face {
        vertices,
        normal,
        offset: dot(&normal, &a),
    })
}

/// The convex hull of `points`, or `None` when they do not span a volume:
/// fewer than four, or all on one plane within the tolerance.
///
/// The tolerance is `1e-9` of the points' extent.
pub fn convex_hull(points: &[[f64; 3]]) -> Option<Hull3> {
    if points.len() < 4 || points.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
        return None;
    }
    let extent = (0..3)
        .map(|axis| {
            let (low, high) = points
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
                    (l.min(p[axis]), h.max(p[axis]))
                });
            high - low
        })
        .fold(0.0, f64::max);
    if extent <= 0.0 {
        return None;
    }
    let tolerance = extent * 1.0e-9;

    // An initial tetrahedron of extreme points.
    let farthest = |score: &dyn Fn(&[f64; 3]) -> f64| -> (usize, f64) {
        points.iter().enumerate().map(|(i, p)| (i, score(p))).fold(
            (0, f64::NEG_INFINITY),
            |best, c| if c.1 > best.1 { c } else { best },
        )
    };
    let (i0, _) = farthest(&|p| -p[0]);
    let p0 = points[i0];
    let (i1, d1) = farthest(&|p| norm(&sub(p, &p0)));
    if d1 <= tolerance {
        return None;
    }
    let p1 = points[i1];
    let line = sub(&p1, &p0).map(|x| x / d1);
    let (i2, d2) = farthest(&|p| norm(&cross(&sub(p, &p0), &line)));
    if d2 <= tolerance {
        return None;
    }
    let p2 = points[i2];
    let plane = cross(&sub(&p1, &p0), &sub(&p2, &p0));
    let plane = plane.map(|x| x / norm(&plane));
    let (i3, d3) = farthest(&|p| dot(&sub(p, &p0), &plane).abs());
    if d3 <= tolerance {
        return None;
    }
    let p3 = points[i3];
    let interior = [0, 1, 2].map(|k| (p0[k] + p1[k] + p2[k] + p3[k]) / 4.0);
    let mut faces: Vec<Face> = [[i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3]]
        .into_iter()
        .filter_map(|v| face(points, v, &interior, tolerance))
        .collect();
    if faces.len() != 4 {
        return None;
    }

    for (index, point) in points.iter().enumerate() {
        if [i0, i1, i2, i3].contains(&index) {
            continue;
        }
        let visible: Vec<usize> = (0..faces.len())
            .filter(|&f| dot(&faces[f].normal, point) - faces[f].offset > tolerance)
            .collect();
        if visible.is_empty() {
            continue;
        }
        // The horizon: edges of visible faces whose other side is not visible.
        let edges: HashSet<(usize, usize)> = visible
            .iter()
            .flat_map(|&f| {
                let [a, b, c] = faces[f].vertices;
                [(a, b), (b, c), (c, a)]
            })
            .collect();
        let horizon: Vec<(usize, usize)> = edges
            .iter()
            .filter(|(a, b)| !edges.contains(&(*b, *a)))
            .copied()
            .collect();
        let mut keep = vec![true; faces.len()];
        for &f in &visible {
            keep[f] = false;
        }
        let mut next: Vec<Face> = faces
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(f, _)| *f)
            .collect();
        next.extend(
            horizon
                .into_iter()
                .filter_map(|(a, b)| face(points, [a, b, index], &interior, tolerance)),
        );
        faces = next;
    }
    Some(Hull3 {
        points: points.to_vec(),
        faces,
        interior,
        tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_cube_with_points_inside_has_its_volume_and_eight_corners() {
        let mut rng = lcg(3);
        let mut points: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    f64::from(i & 1),
                    f64::from((i >> 1) & 1),
                    f64::from((i >> 2) & 1),
                ]
                .map(|v| v * 2.0)
            })
            .collect();
        points.extend((0..200).map(|_| [rng() * 2.0, rng() * 2.0, rng() * 2.0]));
        // Points on the faces too, which must not add slivers.
        points.extend((0..50).map(|_| [0.0, rng() * 2.0, rng() * 2.0]));
        let hull = convex_hull(&points).unwrap();
        assert!((hull.volume() - 8.0).abs() < 1e-9, "{}", hull.volume());
        let used: HashSet<usize> = hull.faces.iter().flat_map(|f| f.vertices).collect();
        assert!(
            used.iter().all(|i| *i < 8),
            "only corners are hull vertices: {used:?}"
        );
        assert!(hull.contains(&[1.0, 1.0, 1.0], 0.0));
        assert!(!hull.contains(&[2.1, 1.0, 1.0], 0.0));
    }

    #[test]
    fn every_point_is_inside_every_face_of_random_hulls() {
        for seed in 0..20 {
            let mut rng = lcg(seed);
            let points: Vec<[f64; 3]> = (0..300)
                .map(|_| {
                    // A squashed blob, so the hull is far from a box.
                    let (u, v, w) = (rng() - 0.5, rng() - 0.5, rng() - 0.5);
                    [u, 0.3 * v + 0.2 * u * u, 0.05 * w]
                })
                .collect();
            let hull = convex_hull(&points).unwrap();
            for p in &points {
                assert!(hull.contains(p, hull.tolerance * 10.0), "seed {seed}");
            }
            // Closed: every directed edge has its reverse.
            let edges: HashSet<(usize, usize)> = hull
                .faces
                .iter()
                .flat_map(|f| {
                    let [a, b, c] = f.vertices;
                    [(a, b), (b, c), (c, a)]
                })
                .collect();
            assert!(
                edges.iter().all(|(a, b)| edges.contains(&(*b, *a))),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn points_on_a_sphere_approach_its_volume() {
        let mut rng = lcg(9);
        let points: Vec<[f64; 3]> = (0..2000)
            .map(|_| {
                let z = 2.0 * rng() - 1.0;
                let a = std::f64::consts::TAU * rng();
                let r = (1.0 - z * z).sqrt();
                [r * a.cos(), r * a.sin(), z]
            })
            .collect();
        let volume = convex_hull(&points).unwrap().volume();
        let exact = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(
            volume < exact && volume > 0.97 * exact,
            "{volume} vs {exact}"
        );
    }

    #[test]
    fn flat_or_too_few_points_have_no_hull() {
        assert!(convex_hull(&[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).is_none());
        let flat: Vec<[f64; 3]> = (0..20)
            .map(|i| [f64::from(i), f64::from(i * i % 7), 0.0])
            .collect();
        assert!(convex_hull(&flat).is_none());
        assert!(convex_hull(&[
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [f64::NAN, 0.0, 1.0]
        ])
        .is_none());
    }
}
