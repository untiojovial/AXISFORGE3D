//! Triangle meshes from STL and OBJ files.
//!
//! Both readers produce the same thing: vertex positions and triangle indices,
//! with identical positions welded into one vertex. Welding matters for STL,
//! which repeats every vertex once per triangle; without it a mesh takes three
//! to six times the space and its convex hull is computed from mostly
//! duplicate points.
//!
//! Files are untrusted input. Sizes are checked against the bytes actually
//! present before anything is allocated, so a header claiming four billion
//! triangles costs nothing, and every index is checked against the vertex
//! count.

use std::collections::HashMap;

/// Largest mesh accepted, in triangles.
pub const MAX_TRIANGLES: usize = 2_000_000;

/// A triangle mesh in the file's own units.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mesh {
    /// Vertex positions, flattened as `[x, y, z, ...]`.
    pub vertices: Vec<f64>,
    /// Triangle indices into the vertices, three per face.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// The number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The number of distinct vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// The smallest and largest coordinate on each axis, or `None` when empty.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut chunks = self.vertices.chunks_exact(3);
        let first = chunks.next()?;
        let mut low = [first[0], first[1], first[2]];
        let mut high = low;
        for v in chunks {
            for axis in 0..3 {
                low[axis] = low[axis].min(v[axis]);
                high[axis] = high[axis].max(v[axis]);
            }
        }
        Some((low, high))
    }
}

/// What can go wrong reading a mesh.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MeshError {
    /// The file is shorter than its own structure requires.
    #[error("the mesh file is truncated: {0}")]
    Truncated(String),
    /// The file's contents do not follow the format.
    #[error("the mesh file is malformed: {0}")]
    Malformed(String),
    /// The mesh has more triangles than [`MAX_TRIANGLES`].
    #[error("the mesh has {found} triangles; the limit is {MAX_TRIANGLES}")]
    TooLarge {
        /// Triangles the file holds or claims to hold.
        found: usize,
    },
    /// The file has no triangles.
    #[error("the mesh file contains no triangles")]
    Empty,
    /// The file extension is not one this crate reads.
    #[error("unsupported mesh format `{0}`; STL and OBJ are supported")]
    UnsupportedFormat(String),
}

/// Collects triangles, welding identical positions.
#[derive(Default)]
struct Welder {
    mesh: Mesh,
    seen: HashMap<[u64; 3], u32>,
}

impl Welder {
    fn vertex(&mut self, position: [f64; 3]) -> Result<u32, MeshError> {
        if !position.iter().all(|v| v.is_finite()) {
            return Err(MeshError::Malformed(
                "a vertex coordinate is not finite".to_owned(),
            ));
        }
        // Adding zero turns -0.0 into 0.0, so the two weld together.
        let key = position.map(|v| (v + 0.0).to_bits());
        if let Some(index) = self.seen.get(&key) {
            return Ok(*index);
        }
        let index =
            u32::try_from(self.mesh.vertices.len() / 3).map_err(|_| MeshError::TooLarge {
                found: self.mesh.vertices.len() / 3,
            })?;
        self.mesh.vertices.extend_from_slice(&position);
        self.seen.insert(key, index);
        Ok(index)
    }

    fn triangle(&mut self, corners: [[f64; 3]; 3]) -> Result<(), MeshError> {
        if self.mesh.triangle_count() >= MAX_TRIANGLES {
            return Err(MeshError::TooLarge {
                found: self.mesh.triangle_count() + 1,
            });
        }
        let [a, b, c] = [
            self.vertex(corners[0])?,
            self.vertex(corners[1])?,
            self.vertex(corners[2])?,
        ];
        // A triangle with a repeated corner has no area; it contributes
        // nothing to drawing or to a hull, so it is dropped.
        if a != b && b != c && a != c {
            self.mesh.indices.extend_from_slice(&[a, b, c]);
        }
        Ok(())
    }

    fn finish(self) -> Result<Mesh, MeshError> {
        if self.mesh.indices.is_empty() {
            return Err(MeshError::Empty);
        }
        Ok(self.mesh)
    }
}

/// Reads a mesh, choosing the format from the file name's extension.
///
/// # Errors
/// [`MeshError::UnsupportedFormat`] for any extension but `.stl` and `.obj`,
/// or the format's own errors.
pub fn parse(file_name: &str, bytes: &[u8]) -> Result<Mesh, MeshError> {
    let extension = file_name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "stl" => parse_stl(bytes),
        "obj" => parse_obj(bytes),
        other => Err(MeshError::UnsupportedFormat(other.to_owned())),
    }
}

/// Reads an STL file, binary or ASCII.
///
/// # Errors
/// When the file is truncated, malformed, empty or too large.
pub fn parse_stl(bytes: &[u8]) -> Result<Mesh, MeshError> {
    // A binary file is exactly 84 bytes plus 50 per triangle. Checking that is
    // more reliable than looking for the word "solid", which some binary
    // exporters also write at the start of their header.
    if bytes.len() >= 84 {
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if count.checked_mul(50).and_then(|n| n.checked_add(84)) == Some(bytes.len()) {
            return parse_binary_stl(bytes, count);
        }
    }
    let text = std::str::from_utf8(bytes).map_err(|_| {
        MeshError::Malformed(
            "not a binary STL of the size its header states, and not text either".to_owned(),
        )
    })?;
    parse_ascii_stl(text)
}

fn parse_binary_stl(bytes: &[u8], count: usize) -> Result<Mesh, MeshError> {
    if count > MAX_TRIANGLES {
        return Err(MeshError::TooLarge { found: count });
    }
    let mut welder = Welder::default();
    for record in bytes[84..].chunks_exact(50) {
        let read = |offset: usize| {
            f64::from(f32::from_le_bytes([
                record[offset],
                record[offset + 1],
                record[offset + 2],
                record[offset + 3],
            ]))
        };
        // Bytes 0..12 are the facet normal, which is recomputed when drawing.
        let corner = |k: usize| [read(12 + k * 12), read(16 + k * 12), read(20 + k * 12)];
        welder.triangle([corner(0), corner(1), corner(2)])?;
    }
    welder.finish()
}

fn parse_ascii_stl(text: &str) -> Result<Mesh, MeshError> {
    let mut welder = Welder::default();
    let mut corners: Vec<[f64; 3]> = Vec::with_capacity(3);
    let mut saw_solid = false;
    for (line_number, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        match words.next().map(str::to_ascii_lowercase).as_deref() {
            Some("solid") => saw_solid = true,
            Some("vertex") => {
                let mut position = [0.0; 3];
                for slot in &mut position {
                    *slot = words.next().and_then(|w| w.parse().ok()).ok_or_else(|| {
                        MeshError::Malformed(format!(
                            "line {}: a vertex needs three numbers",
                            line_number + 1
                        ))
                    })?;
                }
                corners.push(position);
            }
            Some("endloop") => {
                if corners.len() != 3 {
                    return Err(MeshError::Malformed(format!(
                        "line {}: a facet has {} vertices, not 3",
                        line_number + 1,
                        corners.len()
                    )));
                }
                welder.triangle([corners[0], corners[1], corners[2]])?;
                corners.clear();
            }
            _ => {}
        }
    }
    if !saw_solid {
        return Err(MeshError::Malformed(
            "no `solid` statement: not an STL file".to_owned(),
        ));
    }
    if !corners.is_empty() {
        return Err(MeshError::Truncated(
            "the last facet is not closed".to_owned(),
        ));
    }
    welder.finish()
}

/// Reads a Wavefront OBJ file: positions and faces, polygons fanned into
/// triangles. Normals, texture coordinates, groups and materials are ignored.
///
/// # Errors
/// When the file is malformed, empty or too large.
pub fn parse_obj(bytes: &[u8]) -> Result<Mesh, MeshError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| MeshError::Malformed("an OBJ file must be text".to_owned()))?;
    let mut positions: Vec<[f64; 3]> = Vec::new();
    let mut welder = Welder::default();
    for (line_number, line) in text.lines().enumerate() {
        let at = || format!("line {}", line_number + 1);
        let mut words = line.split_whitespace();
        match words.next() {
            Some("v") => {
                let mut position = [0.0; 3];
                for slot in &mut position {
                    *slot = words.next().and_then(|w| w.parse().ok()).ok_or_else(|| {
                        MeshError::Malformed(format!("{}: a vertex needs three numbers", at()))
                    })?;
                }
                positions.push(position);
            }
            Some("f") => {
                let mut face = Vec::new();
                for word in words {
                    // "7", "7/1", "7//3" and "7/1/3" all start with the position index.
                    let raw: i64 = word
                        .split('/')
                        .next()
                        .and_then(|w| w.parse().ok())
                        .ok_or_else(|| MeshError::Malformed(format!("{}: bad face index", at())))?;
                    // Indices count from one; negative ones count back from the end.
                    let count = i64::try_from(positions.len()).unwrap_or(i64::MAX);
                    let index = if raw > 0 { raw - 1 } else { count + raw };
                    let position = usize::try_from(index)
                        .ok()
                        .and_then(|i| positions.get(i))
                        .ok_or_else(|| {
                            MeshError::Malformed(format!(
                                "{}: face refers to vertex {raw}, but {count} are defined",
                                at()
                            ))
                        })?;
                    face.push(*position);
                }
                if face.len() < 3 {
                    return Err(MeshError::Malformed(format!(
                        "{}: a face needs at least three vertices",
                        at()
                    )));
                }
                for k in 1..face.len() - 1 {
                    welder.triangle([face[0], face[k], face[k + 1]])?;
                }
            }
            _ => {}
        }
    }
    welder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit tetrahedron as ASCII STL.
    const TETRA: &str = "solid t
facet normal 0 0 -1
 outer loop
  vertex 0 0 0
  vertex 0 1 0
  vertex 1 0 0
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 0 1
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex 0 0 0
  vertex 0 0 1
  vertex 0 1 0
 endloop
endfacet
facet normal 1 1 1
 outer loop
  vertex 1 0 0
  vertex 0 1 0
  vertex 0 0 1
 endloop
endfacet
endsolid t
";

    fn binary_stl(triangles: &[[[f32; 3]; 3]]) -> Vec<u8> {
        let mut bytes = vec![0_u8; 80];
        bytes.extend_from_slice(&u32::try_from(triangles.len()).unwrap().to_le_bytes());
        for triangle in triangles {
            bytes.extend_from_slice(&[0_u8; 12]);
            for corner in triangle {
                for value in corner {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            bytes.extend_from_slice(&[0_u8; 2]);
        }
        bytes
    }

    #[test]
    fn an_ascii_stl_is_read_and_welded() {
        let mesh = parse_stl(TETRA.as_bytes()).unwrap();
        assert_eq!(mesh.triangle_count(), 4);
        assert_eq!(
            mesh.vertex_count(),
            4,
            "twelve corners weld into four vertices"
        );
        assert_eq!(mesh.bounds().unwrap(), ([0.0; 3], [1.0; 3]));
    }

    #[test]
    fn a_binary_stl_gives_the_same_mesh_as_its_ascii_twin() {
        let o = [0.0, 0.0, 0.0];
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let bytes = binary_stl(&[[o, y, x], [o, x, z], [o, z, y], [x, y, z]]);
        assert_eq!(
            parse_stl(&bytes).unwrap(),
            parse_stl(TETRA.as_bytes()).unwrap()
        );
    }

    #[test]
    fn a_binary_stl_whose_header_says_solid_is_still_binary() {
        let mut bytes = binary_stl(&[[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        bytes[..5].copy_from_slice(b"solid");
        assert_eq!(parse_stl(&bytes).unwrap().triangle_count(), 1);
    }

    #[test]
    fn a_header_claiming_more_triangles_than_the_file_holds_is_rejected_cheaply() {
        let mut bytes = binary_stl(&[[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        bytes[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
        // Not the size its header states and not text: rejected, nothing allocated.
        assert!(matches!(parse_stl(&bytes), Err(MeshError::Malformed(_))));
    }

    #[test]
    fn an_obj_with_quads_and_negative_indices_is_triangulated() {
        let text = "# a unit square in two ways\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nv 0 0 1\nf -1 1/1/1 2//2\n";
        let mesh = parse_obj(text.as_bytes()).unwrap();
        assert_eq!(
            mesh.triangle_count(),
            3,
            "a quad is two triangles, plus one more"
        );
        assert_eq!(mesh.vertex_count(), 5);
    }

    #[test]
    fn an_obj_face_pointing_past_the_vertices_is_an_error() {
        let error = parse_obj(b"v 0 0 0\nv 1 0 0\nf 1 2 9\n").unwrap_err();
        assert!(error.to_string().contains("vertex 9"), "{error}");
    }

    #[test]
    fn degenerate_triangles_are_dropped_and_an_empty_result_is_an_error() {
        let bytes = binary_stl(&[[[1.0; 3], [1.0; 3], [2.0, 0.0, 0.0]]]);
        assert_eq!(parse_stl(&bytes), Err(MeshError::Empty));
        assert_eq!(parse_obj(b"v 0 0 0\n"), Err(MeshError::Empty));
    }

    #[test]
    fn non_finite_coordinates_are_rejected() {
        let bytes = binary_stl(&[[[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        assert!(matches!(parse_stl(&bytes), Err(MeshError::Malformed(_))));
    }

    #[test]
    fn the_format_is_chosen_by_extension() {
        assert!(parse("part.STL", TETRA.as_bytes()).is_ok());
        assert!(matches!(
            parse("part.iv", b""),
            Err(MeshError::UnsupportedFormat(_))
        ));
        assert!(matches!(parse_stl(b"hello"), Err(MeshError::Malformed(_))));
    }

    #[test]
    fn a_facet_left_open_is_reported_as_truncated() {
        let cut = &TETRA[..TETRA.rfind("  vertex 0 0 1").unwrap()];
        assert!(matches!(
            parse_stl(cut.as_bytes()),
            Err(MeshError::Truncated(_))
        ));
    }
}
