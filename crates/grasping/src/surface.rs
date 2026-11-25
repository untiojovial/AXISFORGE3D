//! A part's surface as triangles, and two parallel pads closing on it.
//!
//! The primitive generator in [`crate::candidate`] writes grasps down from a
//! shape's dimensions. Everything else — a mesh from CAD, a part assembled
//! from several shapes — is handled here, the way a grasp planner such as
//! `GraspStudio` handles any object: through its surface.
//!
//! [`PartSurface`] turns a body's collision geometry into world-space
//! triangles (primitives are tessellated), with the centre of mass and size
//! the quality measure needs. [`PartSurface::close`] then does what the
//! gripper does: given where the jaws are and how they are turned, each flat
//! pad advances along the closing line until it meets the part anywhere
//! within its footprint, and the contacts are the region of the part it
//! touches there. Nothing in that depends on the part's shape being known.
//!
//! # The contact model
//!
//! A pad is a rectangle, `finger_width` across and `finger_depth` along the
//! approach, centred on the grasp line. It stops at the part's extreme point
//! inside that rectangle. Everything of the part within `tolerance` of the
//! stop is taken to touch the pad, standing in for the pad's compliance; with
//! a tolerance of zero, a pad meeting a curved surface touches one line or one
//! point. The force a flat pad exerts is along its own normal wherever it
//! touches, edges and corners of the part included, so every contact normal
//! is the closing direction. Each pad's contact region is reduced to the
//! corners of its convex hull, which span the same wrenches.

use std::collections::HashMap;

use axisforge_domain::{DomainError, DomainResult, Resource, Shape, Transform, Vec3, WorldBody};

use crate::candidate::{GraspCandidate, SynthesisOptions};
use crate::wrench::ContactPoint;

/// Segments around a tessellated cylinder, capsule or sphere.
pub const ROUND_SEGMENTS: usize = 48;

/// A part's collision surface in world coordinates.
#[derive(Debug, Clone)]
pub struct PartSurface {
    /// Vertex positions, world coordinates.
    pub vertices: Vec<Vec3>,
    /// Triangles as vertex indices, wound outward where the source was.
    pub triangles: Vec<[u32; 3]>,
    /// Centre of mass: the volume centroid when every geometry is a closed
    /// surface, or all of them together are (a part cut into pieces),
    /// otherwise the area centroid.
    pub center_of_mass: Vec3,
    /// Largest distance from the centre of mass to a vertex.
    pub radius: f64,
    /// `true` when the centre of mass is a volume centroid.
    pub solid: bool,
}

/// Where a pad closing on the part stopped, and what it touches.
#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    /// Midpoint between the two pads on the grasp line.
    pub center: Vec3,
    /// Distance between the pads: the opening the part needs.
    pub width: f64,
    /// Contacts on the pad on the `+closing` side; outward normal `+closing`.
    pub positive: Vec<ContactPoint>,
    /// Contacts on the pad on the `-closing` side; outward normal `-closing`.
    pub negative: Vec<ContactPoint>,
}

/// One point drawn on the surface, with the outward normal of its triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSample {
    /// Position, world coordinates.
    pub point: Vec3,
    /// Unit normal of the triangle it lies on.
    pub normal: Vec3,
}

/// A small deterministic generator, so the same part always gets the same
/// samples and the same plan.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let value = (self.0 >> 11) as f64 / (1_u64 << 53) as f64;
        value
    }
}

/// Collects triangles from shapes, each geometry kept separate for the
/// centre-of-mass computation.
#[derive(Default)]
struct Collector {
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    /// Start of each geometry's triangles.
    starts: Vec<usize>,
}

impl Collector {
    fn begin(&mut self) {
        self.starts.push(self.triangles.len());
    }

    fn vertex(&mut self, p: Vec3) -> u32 {
        // Parts are far below four billion vertices; the caller checked.
        let index = u32::try_from(self.vertices.len()).unwrap_or(u32::MAX);
        self.vertices.push(p);
        index
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.triangles.push([a, b, c]);
    }

    /// A box centred on the origin of `pose`.
    fn cuboid(&mut self, pose: &Transform, size: &Vec3) {
        let h = size.scaled(0.5);
        let corners: Vec<u32> = (0..8)
            .map(|i| {
                let sign = |bit: usize| if i & bit == 0 { -1.0 } else { 1.0 };
                self.vertex(pose.apply(&Vec3::new(sign(1) * h.x, sign(2) * h.y, sign(4) * h.z)))
            })
            .collect();
        // Faces as corner bit patterns, wound counter-clockwise seen from outside.
        for [a, b, c, d] in [
            [0, 2, 3, 1], // -z
            [4, 5, 7, 6], // +z
            [0, 1, 5, 4], // -y
            [2, 6, 7, 3], // +y
            [0, 4, 6, 2], // -x
            [1, 3, 7, 5], // +x
        ] {
            self.triangle(corners[a], corners[b], corners[c]);
            self.triangle(corners[a], corners[c], corners[d]);
        }
    }

    /// A surface of revolution about the local Z axis through the profile
    /// points `(radius, z)`, from bottom to top. A zero radius is a pole.
    fn revolve(&mut self, pose: &Transform, profile: &[(f64, f64)], segments: usize) {
        let mut rings: Vec<Vec<u32>> = Vec::new();
        for &(r, z) in profile {
            if r <= 0.0 {
                rings.push(vec![self.vertex(pose.apply(&Vec3::new(0.0, 0.0, z)))]);
                continue;
            }
            let ring = (0..segments)
                .map(|k| {
                    #[allow(clippy::cast_precision_loss)]
                    let angle = std::f64::consts::TAU * k as f64 / segments as f64;
                    self.vertex(pose.apply(&Vec3::new(r * angle.cos(), r * angle.sin(), z)))
                })
                .collect();
            rings.push(ring);
        }
        for pair in rings.windows(2) {
            let (low, high) = (&pair[0], &pair[1]);
            for k in 0..segments {
                let next = (k + 1) % segments;
                match (low.len(), high.len()) {
                    (1, 1) => {}
                    (1, _) => self.triangle(low[0], high[next], high[k]),
                    (_, 1) => self.triangle(low[k], low[next], high[0]),
                    _ => {
                        self.triangle(low[k], low[next], high[next]);
                        self.triangle(low[k], high[next], high[k]);
                    }
                }
            }
        }
    }

    fn shape(
        &mut self,
        pose: &Transform,
        shape: &Shape,
        resources: &[Resource],
        segments: usize,
    ) -> DomainResult<()> {
        self.begin();
        match shape {
            Shape::Box { size } => self.cuboid(pose, size),
            Shape::Cylinder { radius, length } => {
                let h = length * 0.5;
                self.revolve(
                    pose,
                    &[(0.0, -h), (*radius, -h), (*radius, h), (0.0, h)],
                    segments,
                );
            }
            Shape::Sphere { radius } => {
                let rings = (segments / 2).max(4);
                let profile: Vec<(f64, f64)> = (0..=rings)
                    .map(|i| {
                        // The poles exactly: sin(pi) is not quite zero.
                        if i == 0 || i == rings {
                            return (0.0, if i == 0 { -radius } else { *radius });
                        }
                        #[allow(clippy::cast_precision_loss)]
                        let polar = std::f64::consts::PI * (1.0 - i as f64 / rings as f64);
                        (radius * polar.sin(), radius * polar.cos())
                    })
                    .collect();
                self.revolve(pose, &profile, segments);
            }
            Shape::Capsule { radius, length } => {
                let h = length * 0.5;
                let quarter = (segments / 4).max(2);
                let mut profile = Vec::new();
                for i in 0..=quarter {
                    #[allow(clippy::cast_precision_loss)]
                    let angle = std::f64::consts::FRAC_PI_2 * i as f64 / quarter as f64;
                    profile.push((radius * angle.sin(), -h - radius * angle.cos()));
                }
                for i in (0..=quarter).rev() {
                    #[allow(clippy::cast_precision_loss)]
                    let angle = std::f64::consts::FRAC_PI_2 * i as f64 / quarter as f64;
                    profile.push((radius * angle.sin(), h + radius * angle.cos()));
                }
                profile[0].0 = 0.0;
                let last = profile.len() - 1;
                profile[last].0 = 0.0;
                self.revolve(pose, &profile, segments);
            }
            Shape::Mesh { resource_id, scale } => {
                let resource =
                    resources
                        .iter()
                        .find(|r| &r.id == resource_id)
                        .ok_or_else(|| {
                            DomainError::dangling(
                                "mesh resource",
                                resource_id.clone(),
                                "part surface",
                            )
                        })?;
                let base = self.vertices.len();
                for v in resource.vertices.chunks_exact(3) {
                    self.vertex(pose.apply(&Vec3::new(
                        v[0] * scale.x,
                        v[1] * scale.y,
                        v[2] * scale.z,
                    )));
                }
                let count = resource.vertices.len() / 3;
                // A mirroring scale turns the surface inside out; keep it outward.
                let mirrored = scale.x * scale.y * scale.z < 0.0;
                let offset = u32::try_from(base).map_err(|_| {
                    DomainError::unsupported("the part has too many vertices to grasp")
                })?;
                for face in resource.indices.chunks_exact(3) {
                    if face.iter().any(|i| *i as usize >= count) {
                        return Err(DomainError::unsupported(format!(
                            "mesh resource '{resource_id}' has an index past its vertices"
                        )));
                    }
                    let [a, b, c] = [face[0] + offset, face[1] + offset, face[2] + offset];
                    if mirrored {
                        self.triangle(a, c, b);
                    } else {
                        self.triangle(a, b, c);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Signed volume and volume-weighted centroid sum of a closed triangle set,
/// or `None` when it is not closed.
fn solid_moments(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Option<(f64, Vec3)> {
    // Closed means every edge is shared by exactly two triangles, compared by
    // position since tessellated poles and seams repeat vertices.
    let key = |v: &Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    let mut ids: HashMap<[u64; 3], usize> = HashMap::new();
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for t in triangles {
        let welded: Vec<usize> = t
            .iter()
            .map(|i| {
                let next = ids.len();
                *ids.entry(key(&vertices[*i as usize])).or_insert(next)
            })
            .collect();
        for (a, b) in [
            (welded[0], welded[1]),
            (welded[1], welded[2]),
            (welded[2], welded[0]),
        ] {
            if a != b {
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    if edges.is_empty() || edges.values().any(|n| *n != 2) {
        return None;
    }
    let mut volume = 0.0;
    let mut moment = Vec3::ZERO;
    for t in triangles {
        let [a, b, c] = t.map(|i| vertices[i as usize]);
        let v = a.dot(&b.cross(&c)) / 6.0;
        volume += v;
        moment = moment.add(&a.add(&b).add(&c).scaled(v / 4.0));
    }
    Some((volume, moment))
}

/// Area and area-weighted centroid sum of a triangle set.
fn area_moments(vertices: &[Vec3], triangles: &[[u32; 3]]) -> (f64, Vec3) {
    let mut area = 0.0;
    let mut moment = Vec3::ZERO;
    for t in triangles {
        let [a, b, c] = t.map(|i| vertices[i as usize]);
        let s = b.sub(&a).cross(&c.sub(&a)).norm() * 0.5;
        area += s;
        moment = moment.add(&a.add(&b).add(&c).scaled(s / 3.0));
    }
    (area, moment)
}

impl PartSurface {
    /// The collision surface of a body.
    ///
    /// Geometries marked for collision are used; a body with none is
    /// refused, since nothing would say where a pad could press.
    ///
    /// # Errors
    /// When the body has no collision geometry, refers to a mesh resource that
    /// is not in `resources`, or has no triangles with area.
    pub fn of(body: &WorldBody, resources: &[Resource]) -> DomainResult<Self> {
        let mut collector = Collector::default();
        let mut used = 0;
        for geometry in body.geometries.iter().filter(|g| g.is_collision()) {
            geometry
                .shape
                .validate(&format!("body '{}' shape", body.name))?;
            let pose = body.pose.compose(&geometry.origin);
            collector.shape(&pose, &geometry.shape, resources, ROUND_SEGMENTS)?;
            used += 1;
        }
        if used == 0 {
            return Err(DomainError::unsupported(format!(
                "body '{}' has no collision geometry to grasp",
                body.name
            )));
        }
        let Collector {
            vertices,
            triangles,
            mut starts,
        } = collector;
        starts.push(triangles.len());

        // Volume centroid when every geometry is closed; overlapping
        // geometries are counted twice where they overlap.
        let mut solid = true;
        let (mut volume, mut moment) = (0.0, Vec3::ZERO);
        for range in starts.windows(2) {
            if let Some((v, m)) = solid_moments(&vertices, &triangles[range[0]..range[1]]) {
                // An inward-wound geometry has negative volume and moment.
                let sign = if v < 0.0 { -1.0 } else { 1.0 };
                volume += sign * v;
                moment = moment.add(&m.scaled(sign));
            } else {
                solid = false;
                break;
            }
        }
        // A part cut into convex pieces for collision has open pieces that
        // close up together along their seams: take the union's volume.
        if !solid {
            if let Some((v, m)) = solid_moments(&vertices, &triangles) {
                let sign = if v < 0.0 { -1.0 } else { 1.0 };
                (solid, volume, moment) = (true, sign * v, m.scaled(sign));
            }
        }
        let center_of_mass = if solid && volume > 1.0e-15 {
            moment.scaled(1.0 / volume)
        } else {
            solid = false;
            let (area, moment) = area_moments(&vertices, &triangles);
            if area <= 0.0 {
                return Err(DomainError::unsupported(format!(
                    "body '{}' has no surface with area to grasp",
                    body.name
                )));
            }
            moment.scaled(1.0 / area)
        };
        let radius = vertices
            .iter()
            .map(|v| v.distance(&center_of_mass))
            .fold(0.0, f64::max)
            .max(1.0e-6);
        Ok(Self {
            vertices,
            triangles,
            center_of_mass,
            radius,
            solid,
        })
    }

    fn corners(&self, t: &[u32; 3]) -> [Vec3; 3] {
        t.map(|i| self.vertices[i as usize])
    }

    /// `count` points spread over the surface in proportion to area, each
    /// with its triangle's unit normal. The same part and seed always give
    /// the same points.
    pub fn sample(&self, count: usize, seed: u64) -> Vec<SurfaceSample> {
        let mut cumulative = Vec::with_capacity(self.triangles.len());
        let mut total = 0.0;
        for t in &self.triangles {
            let [a, b, c] = self.corners(t);
            total += b.sub(&a).cross(&c.sub(&a)).norm() * 0.5;
            cumulative.push(total);
        }
        if total <= 0.0 || count == 0 {
            return Vec::new();
        }
        let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            // Stratified: one draw in each equal slice of the total area, so
            // the samples cover the part instead of clustering.
            #[allow(clippy::cast_precision_loss)]
            let target = total * (i as f64 + rng.next()) / count as f64;
            let index = cumulative
                .partition_point(|area| *area < target)
                .min(self.triangles.len() - 1);
            let [a, b, c] = self.corners(&self.triangles[index]);
            let Ok(normal) = b.sub(&a).cross(&c.sub(&a)).normalized("triangle normal") else {
                continue;
            };
            let (r1, r2) = (rng.next().sqrt(), rng.next());
            let point = a
                .scaled(1.0 - r1)
                .add(&b.scaled(r1 * (1.0 - r2)))
                .add(&c.scaled(r1 * r2));
            out.push(SurfaceSample { point, normal });
        }
        out
    }

    /// The triangles' parts inside the pads' footprint, in pad coordinates,
    /// each with its triangle's unit normal (`None` for a triangle with no
    /// area), and the highest and lowest point among them along the closing
    /// line.
    fn footprint_pieces(
        &self,
        local: &[[f64; 3]],
        half_x: f64,
        half_y: f64,
        lateral: &Vec3,
        approach: &Vec3,
        closing: &Vec3,
    ) -> (Vec<Piece>, f64, f64) {
        let mut pieces: Vec<Piece> = Vec::new();
        let (mut top, mut bottom) = (f64::NEG_INFINITY, f64::INFINITY);
        for t in &self.triangles {
            let polygon: Vec<[f64; 3]> = t.iter().map(|i| local[*i as usize]).collect();
            let outside = |axis: usize, limit: f64| {
                polygon.iter().all(|p| p[axis] > limit) || polygon.iter().all(|p| p[axis] < -limit)
            };
            if outside(0, half_x) || outside(1, half_y) {
                continue;
            }
            let mut clipped = polygon;
            for (axis, sign, limit) in [
                (0, 1.0, half_x),
                (0, -1.0, half_x),
                (1, 1.0, half_y),
                (1, -1.0, half_y),
            ] {
                clipped = clip(&clipped, |p| limit - sign * p[axis]);
                if clipped.is_empty() {
                    break;
                }
            }
            if clipped.is_empty() {
                continue;
            }
            for p in &clipped {
                top = top.max(p[2]);
                bottom = bottom.min(p[2]);
            }
            let [a, b, c] = self.corners(t);
            let normal = b
                .sub(&a)
                .cross(&c.sub(&a))
                .normalized("triangle normal")
                .ok()
                .map(|n| [n.dot(lateral), n.dot(approach), n.dot(closing)]);
            pieces.push((clipped, normal));
        }
        (pieces, top, bottom)
    }

    /// Closes two parallel pads on the part along `closing`, on the line
    /// through `through`, with the pads' length along `approach`.
    ///
    /// `pad` is the pads' (width across, depth along the approach) and
    /// `tolerance` how far below a pad's stop the part still counts as
    /// touching it. `None` when the line misses the part or the part has no
    /// thickness there.
    ///
    /// `closing` and `approach` must be perpendicular unit vectors.
    pub fn close(
        &self,
        through: &Vec3,
        closing: &Vec3,
        approach: &Vec3,
        pad: (f64, f64),
        tolerance: f64,
    ) -> Option<Closure> {
        let lateral = closing.cross(approach);
        let (half_x, half_y) = (pad.0 * 0.5, pad.1 * 0.5);
        // Every vertex in pad coordinates: across, along the approach, and
        // along the closing line.
        let local: Vec<[f64; 3]> = self
            .vertices
            .iter()
            .map(|v| {
                let d = v.sub(through);
                [d.dot(&lateral), d.dot(approach), d.dot(closing)]
            })
            .collect();
        let (pieces, top, bottom) =
            self.footprint_pieces(&local, half_x, half_y, &lateral, approach, closing);
        let width = top - bottom;
        if pieces.is_empty() || !width.is_finite() || width < 1.0e-6 {
            return None;
        }
        let to_world = |x: f64, y: f64, stop: f64| {
            through
                .add(&lateral.scaled(x))
                .add(&approach.scaled(y))
                .add(&closing.scaled(stop))
        };
        let pad_points = |stop: f64, side: f64| -> Vec<ContactPoint> {
            // Points the pad's face presses on, and points its edge presses on.
            let mut face: Vec<[f64; 2]> = Vec::new();
            let mut edge: Vec<([f64; 2], [f64; 3])> = Vec::new();
            for (piece, normal) in &pieces {
                // The part of the piece within the tolerance of the stop.
                let near = clip(piece, |p| tolerance - side * (stop - p[2]));
                // A surface that keeps rising toward the pad past an edge of
                // the footprint it reaches is pressed by that edge, all of
                // its touching part, not by the face.
                let pressed_by_edge = normal.filter(|n| {
                    near.iter()
                        .any(|p| rises_past_the_pad(p, n, side, half_x, half_y))
                });
                for p in &near {
                    match pressed_by_edge {
                        Some(n) => edge.push(([p[0], p[1]], n)),
                        None => face.push([p[0], p[1]]),
                    }
                }
            }
            // The face's touching region by the corners of its hull, all with
            // the pad's normal; edge contacts keep the surface's normal, so
            // they are not merged into that hull, only their repeats from
            // neighbouring triangles dropped.
            let pad_normal = [0.0, 0.0, side];
            let mut touching: Vec<([f64; 2], [f64; 3])> = convex_hull_2d(face)
                .into_iter()
                .map(|p| (p, pad_normal))
                .collect();
            let mut seen: Vec<([i64; 2], [i64; 3])> = Vec::new();
            for (p, n) in edge {
                #[allow(clippy::cast_possible_truncation)]
                let key = (
                    p.map(|v| (v / 1.0e-10).round() as i64),
                    n.map(|v| (v / 1.0e-9).round() as i64),
                );
                if !seen.contains(&key) {
                    seen.push(key);
                    touching.push((p, n));
                }
            }
            let flat: Vec<[f64; 2]> = touching.iter().map(|(p, _)| *p).collect();
            let contacts: Vec<ContactPoint> = spread(&flat)
                .into_iter()
                .map(|(index, [x, y])| {
                    let n = touching[index].1;
                    ContactPoint {
                        point: to_world(x, y, stop),
                        normal: lateral
                            .scaled(n[0])
                            .add(&approach.scaled(n[1]))
                            .add(&closing.scaled(n[2])),
                    }
                })
                .collect();
            contacts
        };
        Some(Closure {
            center: through.add(&closing.scaled((top + bottom) * 0.5)),
            width,
            positive: pad_points(top, 1.0),
            negative: pad_points(bottom, -1.0),
        })
    }
}

/// Most contact points kept on one pad.
///
/// The wrench-space hull is six-dimensional and its cost climbs steeply with
/// the number of contacts: measured on two flat pads, 4 points each took
/// 17 ms and 12 each 203 ms for an epsilon only 2 % higher, and 25 contacts
/// with assorted normals from a rounded CAD part took 7 s.
pub const MAX_PAD_POINTS: usize = 6;

/// At most [`MAX_PAD_POINTS`] of `points`, with their indices: all of them
/// when there are few enough, otherwise the most extreme in that many
/// evenly spread directions. A curved surface pressed into a pad touches it
/// over a polygon with dozens of corners; a subset of the contacts can resist
/// only less than all of them, so the quality it gives can only be lower,
/// never higher.
fn spread(points: &[[f64; 2]]) -> Vec<(usize, [f64; 2])> {
    if points.len() <= MAX_PAD_POINTS {
        return points.iter().copied().enumerate().collect();
    }
    let mut chosen: Vec<usize> = Vec::with_capacity(MAX_PAD_POINTS);
    for k in 0..MAX_PAD_POINTS {
        #[allow(clippy::cast_precision_loss)]
        let angle = std::f64::consts::TAU * k as f64 / MAX_PAD_POINTS as f64;
        let (dx, dy) = (angle.cos(), angle.sin());
        let best = (0..points.len())
            .max_by(|a, b| {
                let da = points[*a][0] * dx + points[*a][1] * dy;
                let db = points[*b][0] * dx + points[*b][1] * dy;
                da.total_cmp(&db)
            })
            .unwrap_or(0);
        if !chosen.contains(&best) {
            chosen.push(best);
        }
    }
    chosen.into_iter().map(|i| (i, points[i])).collect()
}

/// The most tilted contact normal on one side, or `fallback` with no contacts.
fn worst_normal(contacts: &[ContactPoint], closing: &Vec3, fallback: Vec3) -> Vec3 {
    contacts
        .iter()
        .map(|c| c.normal)
        .min_by(|a, b| a.dot(closing).abs().total_cmp(&b.dot(closing).abs()))
        .unwrap_or(fallback)
}

/// Gap kept between the palm and the part's near side, metres.
const PALM_CLEARANCE: f64 = 0.002;
/// Gap kept between the fingertips and the part's far side, metres.
const TIP_CLEARANCE: f64 = 0.002;

/// A direction up to sign, rounded, for recognising the same closing line.
#[allow(clippy::cast_possible_truncation)]
fn line_key(v: &Vec3, step: f64) -> [i64; 3] {
    let flip = [v.x, v.y, v.z]
        .into_iter()
        .find(|c| c.abs() > 1.0e-9)
        .is_some_and(|c| c < 0.0);
    let v = if flip { v.scaled(-1.0) } else { *v };
    v.to_array().map(|c| (c / step).round() as i64)
}

/// Grasps of any part, found through its surface.
///
/// Points are drawn on the part's collision surface in proportion to area,
/// and two families of grasps are tried at each, with `angular_samples`
/// directions around the surface normal:
///
/// - **across**: the jaws close along the normal, and the gripper arrives
///   from each direction around it (the antipodal sampling of parallel-jaw
///   grasp planners);
/// - **onto**: the gripper arrives along the inward normal, as `GraspStudio`
///   approaches a surface, and the jaws close along each direction around it.
///
/// The depth along the approach is set as for primitives: as deep as the
/// palm allows, and a shallow variant whose fingertips stop short of the
/// part's far side, which matters when the part rests on that side. For each,
/// the pads are closed on the part ([`PartSurface::close`]) and the contacts
/// are where they stop. Both jaw assignments are returned, because the two
/// wrist orientations differ in what the arm can reach. Holds that come out
/// the same from neighbouring samples are kept once.
///
/// # Errors
/// As [`PartSurface::of`].
pub fn synthesize_surface(
    body: &WorldBody,
    resources: &[Resource],
    options: &SynthesisOptions,
) -> DomainResult<Vec<GraspCandidate>> {
    let surface = PartSurface::of(body, resources)?;
    let samples = surface.sample(options.surface_samples, options.seed);
    let turns = options.angular_samples.max(1);
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        let normal = sample.normal;
        let Ok(first) = normal.any_perpendicular("surface normal") else {
            continue;
        };
        for turn in 0..turns {
            #[allow(clippy::cast_precision_loss)]
            let angle = std::f64::consts::TAU * turn as f64 / turns as f64;
            let around = axisforge_domain::Quat::from_axis_angle(&normal, angle)?.rotate(&first);
            for (family, closing, approach) in [
                ("across", normal, around),
                ("onto", around, normal.scaled(-1.0)),
            ] {
                let name = format!("surface {index} {family} {turn}");
                surface.hold(
                    &name,
                    &sample.point,
                    &closing,
                    &approach,
                    options,
                    &mut seen,
                    &mut out,
                );
            }
        }
    }
    Ok(out)
}

impl PartSurface {
    /// The smallest and largest extent of the part along `direction`.
    pub fn extent_along(&self, direction: &Vec3) -> (f64, f64) {
        self.vertices
            .iter()
            .map(|v| v.dot(direction))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), d| {
                (low.min(d), high.max(d))
            })
    }

    /// Closes the pads on the line through `point` along `closing`, at the
    /// depths described in [`synthesize_surface`], and adds what holds.
    #[allow(clippy::too_many_arguments)]
    fn hold(
        &self,
        name: &str,
        point: &Vec3,
        closing: &Vec3,
        approach: &Vec3,
        options: &SynthesisOptions,
        seen: &mut std::collections::HashSet<([i64; 3], [i64; 3], [i64; 3])>,
        out: &mut Vec<GraspCandidate>,
    ) {
        let half = options.finger_depth * 0.5;
        let (near, far) = self.extent_along(approach);
        let deepest = near + half - PALM_CLEARANCE;
        let shallow = far - half - TIP_CLEARANCE;
        let mut depths = vec![("", deepest)];
        if shallow < deepest - 1.0e-3 {
            depths.push((" shallow", shallow));
        }
        let along = point.dot(approach);
        for (suffix, depth) in depths {
            let through = point.add(&approach.scaled(depth - along));
            let Some(closure) = self.close(
                &through,
                closing,
                approach,
                (options.finger_width, options.finger_depth),
                options.contact_tolerance,
            ) else {
                continue;
            };
            // The same hold reached from neighbouring samples, or from the
            // other family, is kept once; the flip below covers the sign.
            #[allow(clippy::cast_possible_truncation)]
            let key = (
                closure
                    .center
                    .to_array()
                    .map(|v| (v / 5.0e-4).round() as i64),
                line_key(closing, 0.02),
                approach.to_array().map(|v| (v / 0.02).round() as i64),
            );
            if !seen.insert(key) {
                continue;
            }
            let to_center = self.center_of_mass.sub(&closure.center);
            let center_offset = to_center
                .sub(&closing.scaled(to_center.dot(closing)))
                .norm();
            let contacts: Vec<ContactPoint> = closure
                .positive
                .iter()
                .chain(&closure.negative)
                .copied()
                .collect();
            let positive_normal = worst_normal(&closure.positive, closing, *closing);
            let negative_normal = worst_normal(&closure.negative, closing, closing.scaled(-1.0));
            for (flip, sign) in [("", 1.0), (" flipped", -1.0)] {
                let (normal_a, normal_b) = if sign > 0.0 {
                    (positive_normal, negative_normal)
                } else {
                    (negative_normal, positive_normal)
                };
                out.push(GraspCandidate {
                    label: format!("{name}{suffix}{flip}"),
                    center: closure.center,
                    approach: *approach,
                    closing: closing.scaled(sign),
                    width: closure.width,
                    normal_a,
                    normal_b,
                    center_offset,
                    contacts: contacts.clone(),
                });
            }
        }
    }
}

/// Slope, as rise over run, below which a surface counts as flat against a pad.
const FLAT_SLOPE: f64 = 0.02;

/// `true` when the surface through `p`, with unit normal `n` (pad
/// coordinates), keeps rising toward the pad on `side` beyond the edge of the
/// pad's footprint that `p` lies on. The pad's edge, not its face, then
/// presses on that surface, and the force must respect the surface's normal.
fn rises_past_the_pad(p: &[f64; 3], n: &[f64; 3], side: f64, half_x: f64, half_y: f64) -> bool {
    // The surface must face the pad; one facing away is the far side of a
    // thin wall and says nothing about this contact.
    let facing = n[2] * side;
    if facing <= 0.0 {
        return false;
    }
    // Height toward the pad changes by -n_u / facing per unit along u.
    let rising = |component: f64| -component > FLAT_SLOPE * facing;
    let at = |value: f64, limit: f64| (value.abs() - limit).abs() <= 1.0e-9;
    (at(p[0], half_x) && rising(n[0] * p[0].signum()))
        || (at(p[1], half_y) && rising(n[1] * p[1].signum()))
}

/// A triangle clipped to a pad's footprint, in pad coordinates, with the
/// triangle's unit normal in the same coordinates.
type Piece = (Vec<[f64; 3]>, Option<[f64; 3]>);

/// Sutherland-Hodgman: the part of a convex polygon where `inside(p) >= 0`,
/// with the function linear along each edge.
fn clip(polygon: &[[f64; 3]], inside: impl Fn(&[f64; 3]) -> f64) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(polygon.len() + 2);
    for (i, current) in polygon.iter().enumerate() {
        let previous = &polygon[(i + polygon.len() - 1) % polygon.len()];
        let (dc, dp) = (inside(current), inside(previous));
        if (dc >= 0.0) != (dp >= 0.0) {
            let t = dp / (dp - dc);
            out.push([
                previous[0] + (current[0] - previous[0]) * t,
                previous[1] + (current[1] - previous[1]) * t,
                previous[2] + (current[2] - previous[2]) * t,
            ]);
        }
        if dc >= 0.0 {
            out.push(*current);
        }
    }
    out
}

/// The corners of the convex hull of points in a plane, counter-clockwise.
///
/// Points are first snapped to a 0.1 nm grid. Clipping produces coordinates
/// that should be equal but differ in the last bit, and the monotone chain
/// relies on points sharing an x being ordered by y: without the snap, a
/// corner can sort past its neighbour and be dropped.
fn convex_hull_2d(points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    const GRID: f64 = 1.0e-10;
    let mut points: Vec<[f64; 2]> = points
        .into_iter()
        .map(|p| p.map(|v| (v / GRID).round() * GRID))
        .collect();
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let cross = |o: &[f64; 2], a: &[f64; 2], b: &[f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(points.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(points.iter())
        } else {
            Box::new(points.iter().rev())
        };
        for p in iter {
            // Collinear points are dropped: they add no wrench.
            while hull.len() >= start + 2
                && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(*p);
        }
        hull.pop();
    }
    if hull.is_empty() {
        // Every point on one line: its two ends.
        return vec![points[0], points[points.len() - 1]];
    }
    hull
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::{GeometryInstance, GeometryRole, PhysicalProperties, Quat};

    fn body(shape: Shape, pose: Transform) -> WorldBody {
        WorldBody::manipulable(
            "part",
            "part",
            pose,
            GeometryInstance::visual("g", "g", shape).with_role(GeometryRole::Both),
            PhysicalProperties {
                mass: 0.1,
                friction_coefficient: 0.5,
            },
        )
    }

    fn close_to(a: &Vec3, b: &Vec3, tolerance: f64) -> bool {
        a.distance(b) <= tolerance
    }

    #[test]
    fn a_box_surface_is_closed_with_its_centre_as_centre_of_mass() {
        let pose = Transform::new(
            Vec3::new(0.3, -0.1, 0.5),
            Quat::from_axis_angle(&Vec3::new(1.0, 1.0, 0.0), 0.7).unwrap(),
        );
        let surface = PartSurface::of(
            &body(
                Shape::Box {
                    size: Vec3::new(0.04, 0.06, 0.1),
                },
                pose,
            ),
            &[],
        )
        .unwrap();
        assert_eq!(surface.triangles.len(), 12);
        assert!(surface.solid);
        assert!(close_to(&surface.center_of_mass, &pose.translation, 1e-12));
        let half_diagonal = Vec3::new(0.02, 0.03, 0.05).norm();
        assert!((surface.radius - half_diagonal).abs() < 1e-12);
    }

    #[test]
    fn tessellated_round_shapes_are_closed_and_centred() {
        for shape in [
            Shape::Cylinder {
                radius: 0.02,
                length: 0.1,
            },
            Shape::Sphere { radius: 0.03 },
            Shape::Capsule {
                radius: 0.015,
                length: 0.08,
            },
        ] {
            let pose = Transform::from_translation(Vec3::new(0.1, 0.2, 0.3));
            let surface = PartSurface::of(&body(shape.clone(), pose), &[]).unwrap();
            assert!(surface.solid, "{shape:?}");
            assert!(
                close_to(&surface.center_of_mass, &pose.translation, 1e-9),
                "{shape:?}: {:?}",
                surface.center_of_mass
            );
        }
    }

    #[test]
    fn an_open_surface_uses_the_area_centroid() {
        // One square, two triangles, as a mesh resource: open.
        let resource = Resource {
            id: "sq".to_owned(),
            name: "sq".to_owned(),
            media_type: "model/stl".to_owned(),
            vertices: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        let part = body(
            Shape::Mesh {
                resource_id: "sq".to_owned(),
                scale: Vec3::new(0.1, 0.1, 0.1),
            },
            Transform::IDENTITY,
        );
        let surface = PartSurface::of(&part, &[resource]).unwrap();
        assert!(!surface.solid);
        assert!(close_to(
            &surface.center_of_mass,
            &Vec3::new(0.05, 0.05, 0.0),
            1e-12
        ));
        let missing = PartSurface::of(&part, &[]).unwrap_err();
        assert!(missing.to_string().contains("sq"), "{missing}");
    }

    #[test]
    fn samples_cover_the_surface_in_proportion_to_area_and_repeat() {
        // A slab ten times longer in x than in z: its two big faces hold
        // almost all the area.
        let surface = PartSurface::of(
            &body(
                Shape::Box {
                    size: Vec3::new(0.2, 0.2, 0.02),
                },
                Transform::IDENTITY,
            ),
            &[],
        )
        .unwrap();
        let samples = surface.sample(400, 7);
        assert_eq!(samples, surface.sample(400, 7), "deterministic");
        let on_big_faces = samples.iter().filter(|s| s.normal.z.abs() > 0.99).count();
        // The big faces are 0.08 of 0.096 square metres: 83 percent.
        assert!((300..=370).contains(&on_big_faces), "{on_big_faces}");
        for s in &samples {
            let p = s.point;
            assert!(
                p.x.abs() <= 0.1 + 1e-12 && p.y.abs() <= 0.1 + 1e-12 && p.z.abs() <= 0.01 + 1e-12
            );
        }
    }

    #[test]
    fn pads_on_a_box_stop_at_its_faces_and_touch_their_overlap() {
        // A 40 x 60 x 100 mm box; pads 25 mm wide and 45 mm deep closing
        // along x, approaching along -z.
        let surface = PartSurface::of(
            &body(
                Shape::Box {
                    size: Vec3::new(0.04, 0.06, 0.1),
                },
                Transform::IDENTITY,
            ),
            &[],
        )
        .unwrap();
        let closure = surface
            .close(
                &Vec3::new(0.0, 0.0, 0.03),
                &Vec3::X,
                &Vec3::new(0.0, 0.0, -1.0),
                (0.025, 0.045),
                0.0002,
            )
            .unwrap();
        assert!((closure.width - 0.04).abs() < 1e-12);
        assert!(close_to(&closure.center, &Vec3::new(0.0, 0.0, 0.03), 1e-12));
        // The pad reaches 22.5 mm either side of z = 0.03, but the box ends
        // at z = 0.05: the patch is y in [-12.5, 12.5] mm, z in [7.5, 50] mm.
        for (contacts, x, normal) in [
            (&closure.positive, 0.02, 1.0),
            (&closure.negative, -0.02, -1.0),
        ] {
            assert_eq!(contacts.len(), 4, "{contacts:?}");
            for c in contacts {
                assert!((c.point.x - x).abs() < 1e-12);
                assert!((c.normal.x - normal).abs() < 1e-12);
                assert!((c.point.y.abs() - 0.0125).abs() < 1e-12, "{c:?}");
                assert!(
                    (c.point.z - 0.0075).abs() < 1e-12 || (c.point.z - 0.05).abs() < 1e-12,
                    "{c:?}"
                );
            }
        }
    }

    #[test]
    fn pads_on_a_cylinder_touch_a_strip_set_by_the_tolerance() {
        // Radius 20 mm along z, pads closing along x at mid height.
        let surface = PartSurface::of(
            &body(
                Shape::Cylinder {
                    radius: 0.02,
                    length: 0.1,
                },
                Transform::IDENTITY,
            ),
            &[],
        )
        .unwrap();
        let closure = surface
            .close(
                &Vec3::ZERO,
                &Vec3::X,
                &Vec3::new(0.0, 0.0, -1.0),
                (0.025, 0.045),
                0.0005,
            )
            .unwrap();
        // A 48-gon with a vertex on the x axis: the stops are at the radius.
        assert!((closure.width - 0.04).abs() < 1e-9, "{}", closure.width);
        // The strip within 0.5 mm of the stop spans |y| <= sqrt(r^2 - (r - 0.5 mm)^2)
        // on the true circle, a little less on the polygon.
        let exact = (0.02_f64.powi(2) - 0.0195_f64.powi(2)).sqrt();
        let spread = closure
            .positive
            .iter()
            .map(|c| c.point.y.abs())
            .fold(0.0, f64::max);
        assert!(
            spread > 0.5 * exact && spread <= exact + 1e-9,
            "{spread} vs {exact}"
        );
        // A zero tolerance leaves one line along the axis, two corners.
        let line = surface
            .close(
                &Vec3::ZERO,
                &Vec3::X,
                &Vec3::new(0.0, 0.0, -1.0),
                (0.025, 0.045),
                0.0,
            )
            .unwrap();
        assert_eq!(line.positive.len(), 2, "{:?}", line.positive);
    }

    #[test]
    fn a_pad_edge_on_a_sloped_face_presses_along_the_face_normal() {
        // A tall slab tilted 20 degrees about y: its +x face slopes across
        // the pad, which closes along x and stops where its lower edge meets
        // that face. The force there is along the face's normal, not the pad's.
        let tilt = 20.0_f64.to_radians();
        let pose = Transform::new(Vec3::ZERO, Quat::from_axis_angle(&Vec3::Y, tilt).unwrap());
        let surface = PartSurface::of(
            &body(
                Shape::Box {
                    size: Vec3::new(0.04, 0.1, 0.3),
                },
                pose,
            ),
            &[],
        )
        .unwrap();
        let closure = surface
            .close(
                &Vec3::ZERO,
                &Vec3::X,
                &Vec3::new(0.0, 0.0, -1.0),
                (0.025, 0.045),
                0.0002,
            )
            .unwrap();
        let face_normal = pose.apply_direction(&Vec3::X);
        for (contacts, outward) in [
            (&closure.positive, face_normal),
            (&closure.negative, face_normal.scaled(-1.0)),
        ] {
            assert!(!contacts.is_empty());
            for c in contacts {
                assert!(close_to(&c.normal, &outward, 1e-9), "{c:?}");
                // Near the edge of the footprint, 22.5 mm along the approach:
                // within the 0.2 mm tolerance over a 20 degree slope.
                let inside = 0.0002 / tilt.tan();
                assert!(c.point.z.abs() >= 0.0225 - inside - 1e-9, "{c:?}");
                assert!(c.point.z.abs() <= 0.0225 + 1e-9, "{c:?}");
            }
        }
    }

    #[test]
    fn a_line_that_misses_the_part_closes_on_nothing() {
        let surface = PartSurface::of(
            &body(Shape::Sphere { radius: 0.02 }, Transform::IDENTITY),
            &[],
        )
        .unwrap();
        assert!(surface
            .close(
                &Vec3::new(0.0, 0.1, 0.0),
                &Vec3::X,
                &Vec3::Z,
                (0.02, 0.02),
                0.0
            )
            .is_none());
    }

    #[test]
    fn a_concave_part_is_closed_on_by_its_outermost_points() {
        // Two boxes side by side with a gap: an "II". Pads closing across
        // both meet the outer faces; the gap does not matter.
        let mut part = body(
            Shape::Box {
                size: Vec3::new(0.01, 0.05, 0.05),
            },
            Transform::IDENTITY,
        );
        part.geometries[0].origin = Transform::from_translation(Vec3::new(-0.02, 0.0, 0.0));
        let mut second = part.geometries[0].clone();
        second.id = "g2".to_owned();
        second.origin = Transform::from_translation(Vec3::new(0.02, 0.0, 0.0));
        part.geometries.push(second);
        let surface = PartSurface::of(&part, &[]).unwrap();
        let closure = surface
            .close(
                &Vec3::ZERO,
                &Vec3::X,
                &Vec3::new(0.0, 0.0, -1.0),
                (0.02, 0.02),
                0.0001,
            )
            .unwrap();
        assert!((closure.width - 0.05).abs() < 1e-12);
        assert!(closure
            .positive
            .iter()
            .all(|c| (c.point.x - 0.025).abs() < 1e-12));
        assert!(closure
            .negative
            .iter()
            .all(|c| (c.point.x + 0.025).abs() < 1e-12));
    }

    #[test]
    fn a_many_cornered_patch_keeps_a_few_of_its_own_corners() {
        let circle: Vec<[f64; 2]> = (0..40)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / 40.0;
                [0.01 * a.cos(), 0.01 * a.sin()]
            })
            .collect();
        let kept = spread(&circle);
        assert_eq!(kept.len(), MAX_PAD_POINTS);
        for (index, point) in &kept {
            // Copied values, so compared exactly.
            #[allow(clippy::float_cmp)]
            let same = circle[*index] == *point;
            assert!(same, "only the patch's own corners");
        }
        assert_eq!(spread(&circle[..5]).len(), 5, "few corners are all kept");
    }

    #[test]
    // The coordinates are the ones clipping produced, digit for digit.
    #[allow(clippy::unreadable_literal)]
    fn hull_corners_are_kept_and_interior_points_dropped() {
        let hull = convex_hull_2d(vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.0],
        ]);
        assert_eq!(hull.len(), 4, "{hull:?}");
        assert_eq!(
            convex_hull_2d(vec![[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]]).len(),
            2
        );
        // Clipping leaves 0.0125 as 0.012499999999999997 on one triangle and
        // 0.0125 on the next; the corner at the top right must survive.
        let hull = convex_hull_2d(vec![
            [0.012499999999999997, 0.0225],
            [0.004_5, 0.0225],
            [0.012499999999999997, 0.009_166_666_666_666_67],
            [0.0125, -0.02],
            [-0.0125, -0.02],
            [-0.0125, 0.0225],
        ]);
        assert_eq!(hull.len(), 4, "{hull:?}");
        assert!(hull
            .iter()
            .any(|p| (p[0] - 0.0125).abs() < 1e-9 && (p[1] - 0.0225).abs() < 1e-9));
        assert_eq!(convex_hull_2d(vec![[0.0, 0.0]]).len(), 1);
    }
}
