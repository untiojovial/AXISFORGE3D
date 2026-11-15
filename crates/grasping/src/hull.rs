//! Convex hulls in a handful of dimensions, for wrench-space grasp quality.
//!
//! A grasp's wrench space is the convex hull of the wrenches its contacts can
//! apply, a body in six dimensions. Two questions about that hull decide
//! whether the grasp holds: does it contain the origin (force closure), and how
//! far is the origin from its boundary (the largest disturbance wrench the
//! grasp resists in *every* direction, the epsilon of Ferrari and Canny).
//! Both come from the hull's facets, so this module builds them.
//!
//! The algorithm is the incremental one: start from a simplex, and for each
//! further point delete the facets it can see and stitch new ones from the
//! horizon ridges to the point. Grasp wrenches are highly symmetric, so many
//! are exactly cofacial; a point on a facet's plane counts as not seeing it,
//! which leaves coplanar facets side by side rather than merging them. That is
//! harmless here, since only the planes matter, not how a face is tiled.

use std::collections::BTreeMap;

/// Visibility tolerances tried in turn, relative to the largest point norm.
const TOLERANCES: [f64; 5] = [1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-9];

/// Why a hull could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullError {
    /// The points do not span the space: the hull has no interior.
    Flat,
    /// No valid closed surface could be built at any tolerance tried.
    Degenerate,
}

#[derive(Debug, Clone)]
struct Facet {
    /// Indices of the facet's vertices, ascending.
    vertices: Vec<usize>,
    /// Unit outward normal.
    normal: Vec<f64>,
    /// `normal . x` for any point `x` of the facet.
    offset: f64,
}

/// A full-dimensional convex hull, as its facets.
#[derive(Debug, Clone)]
pub struct Hull {
    dim: usize,
    facets: Vec<Facet>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The unit vector orthogonal to `dim - 1` given vectors, by elimination with
/// full pivoting. `None` when the vectors are linearly dependent.
fn orthogonal_direction(rows: &mut [Vec<f64>], dim: usize) -> Option<Vec<f64>> {
    let count = rows.len();
    let mut columns: Vec<usize> = (0..dim).collect();
    for step in 0..count {
        let mut best = (step, step, 0.0_f64);
        for (r, row) in rows.iter().enumerate().skip(step) {
            for (c, column) in columns.iter().enumerate().skip(step) {
                if row[*column].abs() > best.2 {
                    best = (r, c, row[*column].abs());
                }
            }
        }
        if best.2 < 1.0e-300 {
            return None;
        }
        rows.swap(step, best.0);
        columns.swap(step, best.1);
        let pivot_column = columns[step];
        let pivot_row = rows[step].clone();
        for row in rows.iter_mut().skip(step + 1) {
            let factor = row[pivot_column] / pivot_row[pivot_column];
            if factor != 0.0 {
                for column in &columns[step..] {
                    row[*column] -= factor * pivot_row[*column];
                }
            }
        }
    }
    // One free column remains; set it to one and back-substitute.
    let mut direction = vec![0.0; dim];
    direction[columns[dim - 1]] = 1.0;
    for step in (0..count).rev() {
        let row = &rows[step];
        let sum: f64 = columns[step + 1..]
            .iter()
            .map(|c| row[*c] * direction[*c])
            .sum();
        direction[columns[step]] = -sum / row[columns[step]];
    }
    let norm = dot(&direction, &direction).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    for value in &mut direction {
        *value /= norm;
    }
    Some(direction)
}

impl Hull {
    /// Builds the hull of `points`, each of length `dim`.
    ///
    /// # Errors
    /// [`HullError::Flat`] when the points do not span `dim` dimensions.
    ///
    /// # Panics
    /// If a point's length differs from `dim`.
    pub fn build(points: &[Vec<f64>], dim: usize) -> Result<Self, HullError> {
        assert!(
            points.iter().all(|p| p.len() == dim),
            "every point must have `dim` coordinates"
        );
        let scale = points
            .iter()
            .map(|p| dot(p, p).sqrt())
            .fold(0.0_f64, f64::max);
        if points.len() <= dim || scale <= 0.0 {
            return Err(HullError::Flat);
        }
        // A point closer to the current hull than the tolerance is left out.
        // That bounds how thin a new facet can be, and so how badly its normal
        // can be conditioned; too tight a tolerance lets sliver facets in,
        // whose normals are rounding noise. Each attempt is validated, and the
        // first that yields a closed surface supporting every input point is
        // returned, so the tolerance also bounds the error in the result.
        for tolerance in TOLERANCES {
            match Self::build_with(points, dim, scale, tolerance * scale) {
                Ok(hull)
                    if hull.is_closed()
                        && hull.worst_violation(points) <= 10.0 * tolerance * scale =>
                {
                    return Ok(hull)
                }
                Err(HullError::Flat) => return Err(HullError::Flat),
                _ => {}
            }
        }
        Err(HullError::Degenerate)
    }

    fn build_with(
        points: &[Vec<f64>],
        dim: usize,
        scale: f64,
        tolerance: f64,
    ) -> Result<Self, HullError> {
        let simplex = initial_simplex(&points, dim, scale)?;
        let mut interior = vec![0.0; dim];
        for index in &simplex {
            for (slot, value) in interior.iter_mut().zip(&points[*index]) {
                #[allow(clippy::cast_precision_loss)]
                {
                    *slot += value / simplex.len() as f64;
                }
            }
        }

        let make = |mut vertices: Vec<usize>| -> Result<Facet, HullError> {
            vertices.sort_unstable();
            let base = &points[vertices[0]];
            let mut rows: Vec<Vec<f64>> = vertices[1..]
                .iter()
                .map(|v| points[*v].iter().zip(base).map(|(a, b)| a - b).collect())
                .collect();
            let mut normal = orthogonal_direction(&mut rows, dim).ok_or(HullError::Degenerate)?;
            let mut offset = dot(&normal, base);
            if dot(&normal, &interior) > offset {
                for value in &mut normal {
                    *value = -*value;
                }
                offset = -offset;
            }
            Ok(Facet {
                vertices,
                normal,
                offset,
            })
        };

        let mut facets = Vec::new();
        for skip in 0..simplex.len() {
            let vertices = simplex
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != skip)
                .map(|(_, v)| *v)
                .collect();
            facets.push(make(vertices)?);
        }

        for (index, point) in points.iter().enumerate() {
            if simplex.contains(&index) {
                continue;
            }
            let (visible, hidden): (Vec<Facet>, Vec<Facet>) = facets
                .into_iter()
                .partition(|facet| dot(&facet.normal, point) - facet.offset > tolerance);
            facets = hidden;
            if visible.is_empty() {
                continue;
            }
            // A ridge on the horizon belongs to exactly one visible facet; one
            // shared by two visible facets is interior to the region removed.
            let mut ridges: BTreeMap<Vec<usize>, u32> = BTreeMap::new();
            for facet in &visible {
                for skip in 0..facet.vertices.len() {
                    let mut ridge = facet.vertices.clone();
                    ridge.remove(skip);
                    *ridges.entry(ridge).or_insert(0) += 1;
                }
            }
            for (mut ridge, count) in ridges {
                if count == 1 {
                    ridge.push(index);
                    facets.push(make(ridge)?);
                }
            }
        }
        Ok(Self { dim, facets })
    }

    /// The dimension of the space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of facets.
    pub fn facet_count(&self) -> usize {
        self.facets.len()
    }

    /// Distance from the origin to the nearest facet, when the origin is
    /// strictly inside; zero when it is on or outside the boundary.
    pub fn origin_depth(&self) -> f64 {
        self.facets
            .iter()
            .map(|f| f.offset)
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
    }

    /// The largest amount by which any of `points` lies outside any facet.
    ///
    /// Zero, to rounding, for the points the hull was built from.
    pub fn worst_violation(&self, points: &[Vec<f64>]) -> f64 {
        let mut worst = 0.0_f64;
        for facet in &self.facets {
            for point in points {
                worst = worst.max(dot(&facet.normal, point) - facet.offset);
            }
        }
        worst
    }

    /// `true` when every ridge is shared by exactly two facets, i.e. the
    /// facets form a closed surface.
    pub fn is_closed(&self) -> bool {
        let mut ridges: BTreeMap<Vec<usize>, u32> = BTreeMap::new();
        for facet in &self.facets {
            for skip in 0..facet.vertices.len() {
                let mut ridge = facet.vertices.clone();
                ridge.remove(skip);
                *ridges.entry(ridge).or_insert(0) += 1;
            }
        }
        ridges.values().all(|count| *count == 2)
    }
}

/// Picks `dim + 1` affinely independent points, each as far as possible from
/// the affine span of those already chosen.
fn initial_simplex(points: &[Vec<f64>], dim: usize, scale: f64) -> Result<Vec<usize>, HullError> {
    let first = (0..points.len())
        .max_by(|a, b| points[*a][0].total_cmp(&points[*b][0]))
        .ok_or(HullError::Flat)?;
    let mut chosen = vec![first];
    let mut basis: Vec<Vec<f64>> = Vec::new();
    while chosen.len() <= dim {
        let origin = &points[chosen[0]];
        let mut best: Option<(usize, f64, Vec<f64>)> = None;
        for (index, point) in points.iter().enumerate() {
            let mut residual: Vec<f64> = point.iter().zip(origin).map(|(a, b)| a - b).collect();
            for axis in &basis {
                let along = dot(&residual, axis);
                for (r, a) in residual.iter_mut().zip(axis) {
                    *r -= along * a;
                }
            }
            let distance = dot(&residual, &residual).sqrt();
            if best.as_ref().is_none_or(|(_, d, _)| distance > *d) {
                best = Some((index, distance, residual));
            }
        }
        let (index, distance, mut residual) = best.ok_or(HullError::Flat)?;
        // Anything within a millionth of the cloud's size of the current span
        // is rounding and jitter, not a new dimension.
        if distance <= 1.0e-6 * scale {
            return Err(HullError::Flat);
        }
        for value in &mut residual {
            *value /= distance;
        }
        basis.push(residual);
        chosen.push(index);
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_kinematics::Rng;

    fn cube(dim: usize) -> Vec<Vec<f64>> {
        (0..1_u32 << dim)
            .map(|mask| {
                (0..dim)
                    .map(|i| if mask >> i & 1 == 1 { 1.0 } else { -1.0 })
                    .collect()
            })
            .collect()
    }

    fn cross(dim: usize) -> Vec<Vec<f64>> {
        let mut points = Vec::new();
        for i in 0..dim {
            for sign in [1.0, -1.0] {
                let mut p = vec![0.0; dim];
                p[i] = sign;
                points.push(p);
            }
        }
        points
    }

    fn check(points: &[Vec<f64>], dim: usize) -> Hull {
        let hull = Hull::build(points, dim).expect("a full-dimensional hull");
        assert!(hull.is_closed(), "the facets must form a closed surface");
        assert!(
            hull.worst_violation(points) < 1e-7,
            "{}",
            hull.worst_violation(points)
        );
        hull
    }

    #[test]
    fn a_unit_cube_has_depth_one_in_every_dimension() {
        for dim in 2..=6 {
            let hull = check(&cube(dim), dim);
            assert!(
                (hull.origin_depth() - 1.0).abs() < 1e-6,
                "dim {dim}: {}",
                hull.origin_depth()
            );
        }
    }

    #[test]
    fn a_cross_polytope_has_depth_one_over_root_dim() {
        for dim in 2..=6 {
            let hull = check(&cross(dim), dim);
            #[allow(clippy::cast_precision_loss)]
            let expected = 1.0 / (dim as f64).sqrt();
            assert!((hull.origin_depth() - expected).abs() < 1e-6, "dim {dim}");
            assert_eq!(hull.facet_count(), 1 << dim, "dim {dim}");
        }
    }

    #[test]
    fn an_origin_outside_the_hull_has_zero_depth() {
        let shifted: Vec<Vec<f64>> = cube(3)
            .into_iter()
            .map(|p| vec![p[0] + 3.0, p[1], p[2]])
            .collect();
        assert!(check(&shifted, 3).origin_depth().abs() < 1e-12);
    }

    #[test]
    fn interior_and_duplicate_points_change_nothing() {
        let mut points = cube(4);
        points.extend(cube(4));
        points.push(vec![0.1, -0.2, 0.3, 0.0]);
        points.push(vec![0.0; 4]);
        assert!((check(&points, 4).origin_depth() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn points_in_a_plane_are_reported_flat() {
        let flat: Vec<Vec<f64>> = cube(2).into_iter().map(|p| vec![p[0], p[1], 0.0]).collect();
        assert_eq!(Hull::build(&flat, 3).unwrap_err(), HullError::Flat);
        assert_eq!(Hull::build(&cube(3)[..3], 3).unwrap_err(), HullError::Flat);
    }

    #[test]
    fn a_random_cloud_in_six_dimensions_is_closed_and_supporting() {
        let mut rng = Rng::new(11);
        let points: Vec<Vec<f64>> = (0..90)
            .map(|_| (0..6).map(|_| rng.next_in(-1.0, 1.0)).collect())
            .collect();
        let hull = check(&points, 6);
        // The depth can never exceed the support in any direction.
        for _ in 0..2000 {
            let mut u: Vec<f64> = (0..6).map(|_| rng.next_in(-1.0, 1.0)).collect();
            let norm = dot(&u, &u).sqrt();
            for value in &mut u {
                *value /= norm;
            }
            let support = points
                .iter()
                .map(|p| dot(p, &u))
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(hull.origin_depth() <= support + 1e-9);
        }
        assert!(hull.origin_depth() > 0.0);
    }

    #[test]
    fn building_is_deterministic() {
        let a = Hull::build(&cube(5), 5).unwrap();
        let b = Hull::build(&cube(5), 5).unwrap();
        assert_eq!(a.facet_count(), b.facet_count());
        assert_eq!(a.origin_depth().to_bits(), b.origin_depth().to_bits());
    }
}
