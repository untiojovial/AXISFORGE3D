//! Approximate convex decomposition of closed meshes, kept conservative.
//!
//! Collision treats every mesh piece as its convex hull. A banana, a clamp or
//! a mug is far from convex, and its single hull fills the very hollows a
//! gripper's finger has to reach into. This module cuts such a mesh into
//! pieces whose hulls fit it much more closely.
//!
//! # How the cuts are chosen
//!
//! Greedily: for every piece, planes across its three principal axes at
//! several positions are tried, and the one that most reduces the summed
//! volume of the two halves' hulls is its best cut. The piece whose best cut
//! saves the most is cut, and so on until no cut saves enough or the piece
//! limit is reached. A convex piece gains nothing from any cut, so it is
//! never split.
//!
//! # Why the pieces still cover the part
//!
//! A piece is the part's surface clipped to a convex cell (the intersection
//! of the half-spaces its cuts left it). The solid inside that cell is a
//! polyhedron whose corners are of three kinds: mesh vertices inside the
//! cell, points where the surface crosses a cell's face or edge, and corners
//! of the cell itself that lie inside the solid. Clipping the surface
//! produces the first two; the third are found by enumerating the cell's
//! corners and testing each against the closed mesh, and added to the piece
//! as extra points. The hull of a piece therefore contains its whole share of
//! the solid, not only of its surface, and the union of the hulls contains
//! the part. Without the third kind, cutting a solid block into octants would
//! leave a hollow at its centre that nothing covers.

use std::collections::HashMap;

use axisforge_geometry::convex_hull;

use crate::mesh::Mesh;

/// How far a decomposition goes.
#[derive(Debug, Clone, PartialEq)]
pub struct DecompositionOptions {
    /// Most pieces produced.
    pub max_pieces: usize,
    /// Smallest saving of hull volume, as a fraction of the piece's hull, for
    /// a cut to be made.
    pub min_gain: f64,
    /// Smallest saving as a fraction of the whole part's hull, so that tiny
    /// pieces are not cut for tiny savings.
    pub min_absolute_gain: f64,
    /// Points of each side used to estimate a hull's volume.
    pub sample_points: usize,
    /// Positions tried along each axis.
    pub positions: usize,
}

impl Default for DecompositionOptions {
    fn default() -> Self {
        Self {
            max_pieces: 12,
            min_gain: 0.08,
            min_absolute_gain: 0.01,
            sample_points: 300,
            positions: 7,
        }
    }
}

type P3 = [f64; 3];

fn dot(a: &P3, b: &P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A half-space `normal . x <= offset`.
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: P3,
    offset: f64,
}

impl Plane {
    fn side(&self, p: &P3) -> f64 {
        dot(&self.normal, p) - self.offset
    }

    fn flipped(&self) -> Self {
        Self {
            normal: self.normal.map(|v| -v),
            offset: -self.offset,
        }
    }
}

/// Where segment `a`-`b` crosses `plane`, computed from the endpoints in a
/// fixed order so that every triangle sharing the edge, on either side of
/// the cut, gets the very same point and the pieces' seams stay closed.
#[allow(clippy::many_single_char_names)] // Points and parameters, as in the formula.
fn crossing(a: &P3, b: &P3, plane: &Plane) -> P3 {
    let (p, q) = if a
        .iter()
        .map(|v| v.to_bits())
        .lt(b.iter().map(|v| v.to_bits()))
    {
        (a, b)
    } else {
        (b, a)
    };
    let (sp, sq) = (plane.side(p), plane.side(q));
    let t = sp / (sp - sq);
    [0, 1, 2].map(|k| p[k] + (q[k] - p[k]) * t)
}

/// The part of a polygon on the `side(p) <= 0` side of `plane`.
fn clip(polygon: &[P3], plane: &Plane) -> Vec<P3> {
    let mut out = Vec::with_capacity(polygon.len() + 2);
    for (i, current) in polygon.iter().enumerate() {
        let previous = &polygon[(i + polygon.len() - 1) % polygon.len()];
        let (dc, dp) = (plane.side(current), plane.side(previous));
        if (dc <= 0.0) != (dp <= 0.0) {
            out.push(crossing(previous, current, plane));
        }
        if dc <= 0.0 {
            out.push(*current);
        }
    }
    out
}

/// A piece of the part: clipped surface triangles and the cell it lives in.
#[derive(Debug, Clone)]
struct Piece {
    triangles: Vec<[P3; 3]>,
    cell: Vec<Plane>,
}

impl Piece {
    fn points(&self) -> impl Iterator<Item = &P3> {
        self.triangles.iter().flatten()
    }
}

/// Every `n`-th point, so that at most `limit` remain.
fn thinned(points: Vec<P3>, limit: usize) -> Vec<P3> {
    if points.len() <= limit {
        return points;
    }
    let step = points.len().div_ceil(limit);
    points.into_iter().step_by(step).collect()
}

fn hull_volume(points: Vec<P3>, limit: usize) -> f64 {
    convex_hull(&thinned(points, limit)).map_or(0.0, |h| h.volume())
}

/// The points on each side of `plane`: the piece's vertices there, and where
/// its triangle edges cross the plane.
fn sides(piece: &Piece, plane: &Plane) -> (Vec<P3>, Vec<P3>) {
    let (mut below, mut above) = (Vec::new(), Vec::new());
    for t in &piece.triangles {
        for (i, p) in t.iter().enumerate() {
            let s = plane.side(p);
            if s <= 0.0 {
                below.push(*p);
            }
            if s >= 0.0 {
                above.push(*p);
            }
            let q = &t[(i + 1) % 3];
            if (s < 0.0 && plane.side(q) > 0.0) || (s > 0.0 && plane.side(q) < 0.0) {
                let x = crossing(p, q, plane);
                below.push(x);
                above.push(x);
            }
        }
    }
    (below, above)
}

/// Unit eigenvectors of a symmetric 3x3 matrix, by Jacobi rotations.
// The textbook names, and index loops that update two rows at once.
#[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
fn principal_axes(points: &[P3]) -> [P3; 3] {
    #[allow(clippy::cast_precision_loss)]
    let n = points.len().max(1) as f64;
    let mean = [0, 1, 2].map(|k| points.iter().map(|p| p[k]).sum::<f64>() / n);
    let mut a = [[0.0_f64; 3]; 3];
    for p in points {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        // The largest off-diagonal entry, rotated away.
        let (p, q) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|x, y| a[x.0][x.1].abs().total_cmp(&a[y.0][y.1].abs()))
            .unwrap_or((0, 1));
        if a[p][q].abs() < 1.0e-30 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for row in &mut v {
            let (vp, vq) = (row[p], row[q]);
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    [0, 1, 2].map(|k| [v[0][k], v[1][k], v[2][k]])
}

/// The best cut of a piece: the plane and the hull volumes it leaves.
#[derive(Debug, Clone, Copy)]
struct Cut {
    plane: Plane,
    saving: f64,
}

fn best_cut(piece: &Piece, options: &DecompositionOptions) -> (f64, Option<Cut>) {
    let points: Vec<P3> = piece.points().copied().collect();
    let whole = hull_volume(points.clone(), options.sample_points);
    let mut best: Option<Cut> = None;
    for axis in principal_axes(&points) {
        let (low, high) = points
            .iter()
            .map(|p| dot(&axis, p))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), d| {
                (l.min(d), h.max(d))
            });
        for k in 1..=options.positions {
            #[allow(clippy::cast_precision_loss)]
            let offset = low + (high - low) * k as f64 / (options.positions + 1) as f64;
            let plane = Plane {
                normal: axis,
                offset,
            };
            let (below, above) = sides(piece, &plane);
            let split = hull_volume(below, options.sample_points)
                + hull_volume(above, options.sample_points);
            let saving = whole - split;
            if best.is_none_or(|b| saving > b.saving) {
                best = Some(Cut { plane, saving });
            }
        }
    }
    (whole, best)
}

/// Cuts a piece in two along `plane`.
fn split(piece: &Piece, plane: &Plane) -> (Piece, Piece) {
    let mut halves = [Vec::new(), Vec::new()];
    for (half, side) in halves.iter_mut().zip([*plane, plane.flipped()]) {
        for t in &piece.triangles {
            let polygon = clip(t, &side);
            for i in 1..polygon.len().saturating_sub(1) {
                half.push([polygon[0], polygon[i], polygon[i + 1]]);
            }
        }
    }
    let [below, above] = halves;
    let mut cell_below = piece.cell.clone();
    cell_below.push(*plane);
    let mut cell_above = piece.cell.clone();
    cell_above.push(plane.flipped());
    (
        Piece {
            triangles: below,
            cell: cell_below,
        },
        Piece {
            triangles: above,
            cell: cell_above,
        },
    )
}

/// `true` when `p` is inside the closed surface `triangles`: the parity of
/// crossings along three rays, by majority.
fn inside(p: &P3, triangles: &[[P3; 3]]) -> bool {
    // Directions with no simple relation to any axis, so rays rarely graze
    // edges of axis-aligned models.
    let rays = [
        [0.577_215_664, 0.381_966_011, 0.721_347_520],
        [-0.412_310_562, 0.836_660_026, 0.360_555_127],
        [0.223_606_797, -0.565_685_424, -0.793_725_393],
    ];
    let votes = rays
        .iter()
        .filter(|d| {
            let hits = triangles.iter().filter(|t| ray_hits(p, d, t)).count();
            hits % 2 == 1
        })
        .count();
    votes >= 2
}

/// Moller-Trumbore: does the ray from `origin` along `d` cross triangle `t`?
#[allow(clippy::many_single_char_names)] // The algorithm's own names.
fn ray_hits(origin: &P3, d: &P3, t: &[P3; 3]) -> bool {
    let sub = |a: &P3, b: &P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: &P3, b: &P3| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let (e1, e2) = (sub(&t[1], &t[0]), sub(&t[2], &t[0]));
    let h = cross(d, &e2);
    let det = dot(&e1, &h);
    if det.abs() < 1.0e-18 {
        return false;
    }
    let s = sub(origin, &t[0]);
    let u = dot(&s, &h) / det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = cross(&s, &e1);
    let v = dot(d, &q) / det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    dot(&e2, &q) / det > 0.0
}

/// Corners of a cell (clipped to `bounds`) that lie inside the solid.
#[allow(clippy::many_single_char_names)] // Three planes and Cramer's rule.
fn inner_corners(cell: &[Plane], bounds: &[Plane], solid: &[[P3; 3]], scale: f64) -> Vec<P3> {
    let planes: Vec<Plane> = cell.iter().chain(bounds).copied().collect();
    let tolerance = scale * 1.0e-9;
    let mut corners = Vec::new();
    for i in 0..planes.len() {
        for j in i + 1..planes.len() {
            for k in j + 1..planes.len() {
                let [a, b, c] = [planes[i], planes[j], planes[k]];
                let (n1, n2, n3) = (a.normal, b.normal, c.normal);
                let det = n1[0] * (n2[1] * n3[2] - n2[2] * n3[1])
                    - n1[1] * (n2[0] * n3[2] - n2[2] * n3[0])
                    + n1[2] * (n2[0] * n3[1] - n2[1] * n3[0]);
                if det.abs() < 1.0e-12 {
                    continue;
                }
                // Cramer's rule for n . x = offset on all three planes.
                let solve = |col: usize| {
                    let mut m = [n1, n2, n3];
                    m[0][col] = a.offset;
                    m[1][col] = b.offset;
                    m[2][col] = c.offset;
                    (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                        / det
                };
                let p = [solve(0), solve(1), solve(2)];
                if planes.iter().all(|plane| plane.side(&p) <= tolerance) && inside(&p, solid) {
                    corners.push(p);
                }
            }
        }
    }
    corners
}

/// Cuts a closed mesh into pieces whose hulls fit it closely and together
/// contain it. `None` when the mesh is not closed or no cut is worth making.
///
/// Each piece is a mesh of the part's own surface within its cell, with any
/// cell corners inside the part added as extra vertices that no triangle
/// uses (see the module documentation).
pub fn decompose(mesh: &Mesh, options: &DecompositionOptions) -> Option<Vec<Mesh>> {
    let solid_volume = mesh.volume()?;
    let corner = |i: u32| {
        let at = i as usize * 3;
        [
            mesh.vertices[at],
            mesh.vertices[at + 1],
            mesh.vertices[at + 2],
        ]
    };
    let solid: Vec<[P3; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|f| [corner(f[0]), corner(f[1]), corner(f[2])])
        .collect();
    let all: Vec<P3> = solid.iter().flatten().copied().collect();
    let whole_hull = hull_volume(all.clone(), options.sample_points * 4);
    if whole_hull <= 0.0 {
        return None;
    }
    // No set of cuts can save more than the space between the hull and the
    // solid, since the pieces' hulls still hold the pieces. When that is
    // already below what the first cut must save, there is nothing to try.
    if whole_hull - solid_volume < options.min_gain * whole_hull {
        return None;
    }

    let mut pieces = vec![Piece {
        triangles: solid.clone(),
        cell: Vec::new(),
    }];
    let mut cuts: Vec<(f64, Option<Cut>)> = vec![best_cut(&pieces[0], options)];
    while pieces.len() < options.max_pieces {
        let chosen = cuts
            .iter()
            .enumerate()
            .filter_map(|(i, (hull, cut))| cut.map(|c| (i, *hull, c)))
            .filter(|(_, hull, c)| {
                c.saving >= options.min_gain * hull
                    && c.saving >= options.min_absolute_gain * whole_hull
            })
            .max_by(|a, b| a.2.saving.total_cmp(&b.2.saving));
        let Some((index, _, cut)) = chosen else {
            break;
        };
        let (below, above) = split(&pieces[index], &cut.plane);
        if below.triangles.is_empty() || above.triangles.is_empty() {
            cuts[index].1 = None;
            continue;
        }
        cuts[index] = best_cut(&below, options);
        pieces[index] = below;
        cuts.push(best_cut(&above, options));
        pieces.push(above);
    }
    if pieces.len() < 2 {
        return None;
    }

    // The part's bounding box closes every cell for the corner search.
    let (low, high) = all.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(mut l, mut h), p| {
            for k in 0..3 {
                l[k] = l[k].min(p[k]);
                h[k] = h[k].max(p[k]);
            }
            (l, h)
        },
    );
    let scale = (0..3).map(|k| high[k] - low[k]).fold(0.0, f64::max);
    let bounds: Vec<Plane> = (0..3)
        .flat_map(|k| {
            let mut axis = [0.0; 3];
            axis[k] = 1.0;
            [
                Plane {
                    normal: axis,
                    offset: high[k],
                },
                Plane {
                    normal: axis.map(|v| -v),
                    offset: -low[k],
                },
            ]
        })
        .collect();
    Some(
        pieces
            .iter()
            .map(|piece| {
                let extra = inner_corners(&piece.cell, &bounds, &solid, scale);
                to_mesh(&piece.triangles, &extra)
            })
            .collect(),
    )
}

/// A triangle soup and extra points as a welded mesh.
fn to_mesh(triangles: &[[P3; 3]], extra: &[P3]) -> Mesh {
    let mut mesh = Mesh::default();
    let mut index: HashMap<[u64; 3], u32> = HashMap::new();
    let mut vertex = |p: &P3, mesh: &mut Mesh| -> u32 {
        let key = p.map(|v| (v + 0.0).to_bits());
        *index.entry(key).or_insert_with(|| {
            mesh.vertices.extend_from_slice(p);
            u32::try_from(mesh.vertices.len() / 3 - 1).unwrap_or(u32::MAX)
        })
    };
    for t in triangles {
        let [a, b, c] = [
            vertex(&t[0], &mut mesh),
            vertex(&t[1], &mut mesh),
            vertex(&t[2], &mut mesh),
        ];
        if a != b && b != c && a != c {
            mesh.indices.extend_from_slice(&[a, b, c]);
        }
    }
    for p in extra {
        vertex(p, &mut mesh);
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed mesh of axis-aligned boxes joined face to face, built as the
    /// boundary of a set of unit voxels.
    #[allow(clippy::many_single_char_names)]
    fn voxels(cells: &[[i32; 3]], size: f64) -> Mesh {
        let filled = |c: [i32; 3]| cells.contains(&c);
        let mut triangles: Vec<[P3; 3]> = Vec::new();
        for c in cells {
            for axis in 0..3 {
                for dir in [-1, 1] {
                    let mut n = *c;
                    n[axis] += dir;
                    if filled(n) {
                        continue;
                    }
                    // The face between c and n, wound outward.
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let base = |du: i32, dv: i32| {
                        let mut p = [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])];
                        p[axis] += if dir > 0 { 1.0 } else { 0.0 };
                        p[u] += f64::from(du);
                        p[v] += f64::from(dv);
                        p.map(|x| x * size)
                    };
                    let (a, b, cc, d) = (base(0, 0), base(1, 0), base(1, 1), base(0, 1));
                    if dir > 0 {
                        triangles.push([a, b, cc]);
                        triangles.push([a, cc, d]);
                    } else {
                        triangles.push([a, cc, b]);
                        triangles.push([a, d, cc]);
                    }
                }
            }
        }
        to_mesh(&triangles, &[])
    }

    fn hulls(pieces: &[Mesh]) -> Vec<axisforge_geometry::Hull3> {
        pieces
            .iter()
            .map(|m| {
                let points: Vec<P3> = m
                    .vertices
                    .chunks_exact(3)
                    .map(|v| [v[0], v[1], v[2]])
                    .collect();
                convex_hull(&points).expect("a piece with volume")
            })
            .collect()
    }

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
    fn a_box_is_left_whole() {
        let cube = voxels(&[[0, 0, 0]], 0.04);
        assert!(cube.volume().is_some());
        assert!(decompose(&cube, &DecompositionOptions::default()).is_none());
    }

    #[test]
    fn an_l_shape_is_cut_and_its_hulls_hug_it() {
        // An L of four voxels: three in a row and one up at the end.
        let l = voxels(&[[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1]], 0.02);
        let solid = l.volume().unwrap();
        let pieces = decompose(&l, &DecompositionOptions::default()).expect("worth cutting");
        assert!(pieces.len() >= 2, "{} pieces", pieces.len());
        let summed: f64 = hulls(&pieces)
            .iter()
            .map(axisforge_geometry::Hull3::volume)
            .sum();
        let single = convex_hull(
            &l.vertices
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .volume();
        // One hull is 1.25 times the solid; the pieces' hulls match it.
        assert!(single > 1.2 * solid, "{single} vs {solid}");
        assert!(summed < 1.02 * solid, "{summed} vs {solid}");
    }

    #[test]
    fn the_pieces_cover_every_point_of_the_solid() {
        // A U with a solid base: interior corners of the cells fall inside
        // the base and must be covered too.
        let cells: Vec<[i32; 3]> = (0..4)
            .flat_map(|x| (0..2).map(move |y| [x, y, 0]))
            .chain([
                [0, 0, 1],
                [0, 1, 1],
                [3, 0, 1],
                [3, 1, 1],
                [0, 0, 2],
                [3, 1, 2],
            ])
            .collect();
        let u = voxels(&cells, 0.01);
        let pieces = decompose(&u, &DecompositionOptions::default()).expect("worth cutting");
        let hulls = hulls(&pieces);
        let mut rng = lcg(5);
        let mut checked = 0;
        for _ in 0..4000 {
            let p = [rng() * 0.04, rng() * 0.02, rng() * 0.03];
            #[allow(clippy::cast_possible_truncation)]
            let cell = p.map(|v| (v / 0.01).floor() as i32);
            if !cells.contains(&cell) {
                continue;
            }
            checked += 1;
            assert!(
                hulls.iter().any(|h| h.contains(&p, 1.0e-12)),
                "{p:?} inside the solid is in no piece's hull"
            );
        }
        assert!(checked > 1000);
    }

    #[test]
    fn seams_between_pieces_weld_into_a_closed_surface() {
        let l = voxels(&[[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1]], 0.02);
        let pieces = decompose(&l, &DecompositionOptions::default()).unwrap();
        // All pieces' triangles together, welded by exact position.
        let mut soup: Vec<[P3; 3]> = Vec::new();
        for m in &pieces {
            for f in m.indices.chunks_exact(3) {
                let v = |i: u32| {
                    let at = i as usize * 3;
                    [m.vertices[at], m.vertices[at + 1], m.vertices[at + 2]]
                };
                soup.push([v(f[0]), v(f[1]), v(f[2])]);
            }
        }
        let union = to_mesh(&soup, &[]);
        let volume = union.volume().expect("the union is closed");
        assert!((volume - l.volume().unwrap()).abs() < 1e-15, "{volume}");
    }

    #[test]
    fn an_open_mesh_is_not_decomposed() {
        let mut l = voxels(&[[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1]], 0.02);
        l.indices.truncate(l.indices.len() - 3);
        assert!(decompose(&l, &DecompositionOptions::default()).is_none());
    }
}
