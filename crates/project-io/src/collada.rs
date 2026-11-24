//! Triangle meshes from COLLADA (`.dae`) files.
//!
//! Only geometry is read: the `<mesh>` elements the file's scene instantiates,
//! placed by the scene's node transforms and scaled by `<unit meter>`, each
//! triangle coloured with its material's diffuse colour. Cameras, lights,
//! animation, skinning, textures and physics are left alone. Triangles are
//! grouped into one [`Part`] per colour, so a robot link exported from CAD
//! keeps its two-tone look.
//!
//! `<up_axis>` is not applied. ROS tools draw COLLADA meshes with Z up
//! whatever the file declares (the `geometric_shapes` library of `MoveIt` clears the
//! root transform Assimp adds for it, and says `RViz` does the same), and URDF
//! files are written to match; a file that declares another axis gets a note
//! saying so. `<unit>` is applied, as the format requires, and a unit other
//! than the metre gets a note too, because not every tool applies it.
//!
//! Files are untrusted. The XML parser refuses document type declarations,
//! references must point inside the file, every index is checked, and both
//! the triangles produced and the scene nodes visited are bounded, so a node
//! that instantiates itself, or a scene that multiplies one node a million
//! times, fails instead of running away.

use std::collections::HashMap;
use std::rc::Rc;

use roxmltree::{Document, Node};

use crate::import::hex_color;
use crate::mesh::{MeshError, Parsed, Part, Welder, MAX_TRIANGLES};

/// Largest COLLADA file accepted, bytes.
pub const MAX_BYTES: usize = 64 * 1024 * 1024;
/// Deepest nesting of scene nodes followed.
const MAX_DEPTH: usize = 32;
/// Most scene nodes visited, counting every instantiation.
const MAX_VISITS: usize = 100_000;

/// A 4×4 transform, row-major, acting on column vectors.
type Matrix = [[f64; 4]; 4];

const IDENTITY: Matrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [[0.0; 4]; 4];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

fn apply(m: &Matrix, p: [f64; 3]) -> [f64; 3] {
    let row = |r: usize| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
    [row(0), row(1), row(2)]
}

/// Determinant of the upper-left 3×3 block: negative for a mirroring transform.
fn determinant(m: &Matrix) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn malformed(message: impl Into<String>) -> MeshError {
    MeshError::Malformed(message.into())
}

fn is(node: Node<'_, '_>, name: &str) -> bool {
    node.is_element() && node.tag_name().name() == name
}

fn child<'a, 'i>(node: Node<'a, 'i>, name: &str) -> Option<Node<'a, 'i>> {
    node.children().find(|n| is(*n, name))
}

fn children<'a, 'i: 'a>(
    node: Node<'a, 'i>,
    name: &'a str,
) -> impl Iterator<Item = Node<'a, 'i>> + 'a {
    node.children().filter(move |n| is(*n, name))
}

fn floats(text: &str, what: &str) -> Result<Vec<f64>, MeshError> {
    text.split_ascii_whitespace()
        .map(|t| match t.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(v),
            _ => Err(malformed(format!("{what}: `{t}` is not a finite number"))),
        })
        .collect()
}

fn counts(text: &str, what: &str) -> Result<Vec<usize>, MeshError> {
    text.split_ascii_whitespace()
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| malformed(format!("{what}: `{t}` is not an index")))
        })
        .collect()
}

fn count_attribute(node: Node<'_, '_>, name: &str, default: usize) -> Result<usize, MeshError> {
    node.attribute(name).map_or(Ok(default), |t| {
        t.trim()
            .parse::<usize>()
            .map_err(|_| malformed(format!("<{}> {name}=`{t}`", node.tag_name().name())))
    })
}

/// Rodrigues' rotation about `axis` by `angle` radians.
#[allow(clippy::many_single_char_names)] // The textbook formula's names.
fn rotation(axis: [f64; 3], angle: f64) -> Result<Matrix, MeshError> {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm < 1.0e-12 {
        return Err(malformed("<rotate> has a zero axis"));
    }
    let (x, y, z) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    Ok([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// One step of a node's transform, or `None` for a child that is not one.
fn transform_step(node: Node<'_, '_>) -> Result<Option<Matrix>, MeshError> {
    let name = node.tag_name().name();
    let values = |count: usize| -> Result<Vec<f64>, MeshError> {
        let values = floats(node.text().unwrap_or(""), &format!("<{name}>"))?;
        if values.len() == count {
            Ok(values)
        } else {
            Err(malformed(format!(
                "<{name}> needs {count} numbers, found {}",
                values.len()
            )))
        }
    };
    let matrix = match name {
        "matrix" => {
            let v = values(16)?;
            let mut m = [[0.0; 4]; 4];
            for (r, row) in m.iter_mut().enumerate() {
                row.copy_from_slice(&v[r * 4..r * 4 + 4]);
            }
            m
        }
        "translate" => {
            let v = values(3)?;
            let mut m = IDENTITY;
            m[0][3] = v[0];
            m[1][3] = v[1];
            m[2][3] = v[2];
            m
        }
        "scale" => {
            let v = values(3)?;
            let mut m = IDENTITY;
            m[0][0] = v[0];
            m[1][1] = v[1];
            m[2][2] = v[2];
            m
        }
        "rotate" => {
            let axis_angle = values(4)?;
            rotation(
                [axis_angle[0], axis_angle[1], axis_angle[2]],
                axis_angle[3].to_radians(),
            )?
        }
        "lookat" | "skew" => {
            return Err(malformed(format!(
                "the <{name}> transform is not supported"
            )))
        }
        _ => return Ok(None),
    };
    Ok(Some(matrix))
}

/// The primitive kinds that hold triangles.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Triangles,
    Polylist,
    Polygons,
    Strips,
    Fans,
}

struct Reader<'a, 'i> {
    ids: HashMap<&'a str, Node<'a, 'i>>,
    /// Positions of each `<source>` already read, by its id.
    positions: HashMap<String, Rc<Vec<[f64; 3]>>>,
    /// Diffuse colour of each material already looked up, by its id.
    colors: HashMap<String, Option<String>>,
    parts: Vec<(Option<String>, Welder)>,
    triangles: usize,
    visits: usize,
    controllers: usize,
    not_meshes: usize,
}

impl<'a, 'i> Reader<'a, 'i> {
    fn new(document: &'a Document<'i>) -> Self {
        let mut ids = HashMap::new();
        for node in document.descendants().filter(Node::is_element) {
            if let Some(id) = node.attribute("id") {
                ids.entry(id).or_insert(node);
            }
        }
        Self {
            ids,
            positions: HashMap::new(),
            colors: HashMap::new(),
            parts: Vec::new(),
            triangles: 0,
            visits: 0,
            controllers: 0,
            not_meshes: 0,
        }
    }

    /// The element a `#id` URL names. URLs into other files are refused:
    /// following them would mean reading files the user did not hand over.
    fn resolve(&self, url: &str, expected: &str) -> Result<Node<'a, 'i>, MeshError> {
        let id = url.strip_prefix('#').ok_or_else(|| {
            malformed(format!(
                "`{url}` refers to another file; only references inside the file are followed"
            ))
        })?;
        let node = self
            .ids
            .get(id)
            .copied()
            .ok_or_else(|| malformed(format!("`{url}` is not defined in the file")))?;
        if is(node, expected) {
            Ok(node)
        } else {
            Err(malformed(format!(
                "`{url}` is a <{}>, not a <{expected}>",
                node.tag_name().name()
            )))
        }
    }

    /// The diffuse colour of a material, or `None` when it has none that can
    /// be read. A colour is cosmetic, so a broken material only loses it.
    fn material_color(&mut self, url: &str) -> Option<String> {
        if let Some(known) = self.colors.get(url) {
            return known.clone();
        }
        let color = (|| {
            let material = self.resolve(url, "material").ok()?;
            let effect_url = child(material, "instance_effect")?.attribute("url")?;
            let effect = self.resolve(effect_url, "effect").ok()?;
            let diffuse = effect.descendants().find(|n| is(*n, "diffuse"))?;
            let rgba = floats(child(diffuse, "color")?.text()?, "diffuse").ok()?;
            (rgba.len() >= 3).then(|| hex_color(rgba[0], rgba[1], rgba[2]))
        })();
        self.colors.insert(url.to_owned(), color.clone());
        color
    }

    /// Material symbol to colour, from an instance's `<bind_material>`.
    fn bindings(&mut self, instance: Node<'a, 'i>) -> HashMap<String, Option<String>> {
        let mut bound = HashMap::new();
        for binding in instance
            .descendants()
            .filter(|n| is(*n, "instance_material"))
        {
            if let (Some(symbol), Some(target)) =
                (binding.attribute("symbol"), binding.attribute("target"))
            {
                let color = self.material_color(target);
                bound.insert(symbol.to_owned(), color);
            }
        }
        bound
    }

    fn visit(
        &mut self,
        node: Node<'a, 'i>,
        parent: &Matrix,
        depth: usize,
    ) -> Result<(), MeshError> {
        if depth > MAX_DEPTH {
            return Err(malformed(format!(
                "scene nodes are nested more than {MAX_DEPTH} deep, or a node instantiates itself"
            )));
        }
        self.visits += 1;
        if self.visits > MAX_VISITS {
            return Err(malformed(format!(
                "the scene instantiates more than {MAX_VISITS} nodes"
            )));
        }
        // Transform elements compose in document order.
        let mut local = IDENTITY;
        for step in node.children().filter(Node::is_element) {
            if let Some(matrix) = transform_step(step)? {
                local = multiply(&local, &matrix);
            }
        }
        let world = multiply(parent, &local);
        for item in node.children().filter(Node::is_element) {
            match item.tag_name().name() {
                "node" => self.visit(item, &world, depth + 1)?,
                "instance_node" => {
                    let url = item.attribute("url").unwrap_or("");
                    let target = self.resolve(url, "node")?;
                    self.visit(target, &world, depth + 1)?;
                }
                "instance_geometry" => {
                    let url = item.attribute("url").unwrap_or("");
                    let geometry = self.resolve(url, "geometry")?;
                    let bound = self.bindings(item);
                    self.geometry(geometry, &world, &bound)?;
                }
                "instance_controller" => self.controllers += 1,
                _ => {}
            }
        }
        Ok(())
    }

    /// The positions a `<source>` holds, read once.
    fn source_positions(&mut self, source: Node<'a, 'i>) -> Result<Rc<Vec<[f64; 3]>>, MeshError> {
        let key = source.attribute("id").unwrap_or("").to_owned();
        if let Some(known) = self.positions.get(&key) {
            return Ok(Rc::clone(known));
        }
        let accessor = child(source, "technique_common")
            .and_then(|t| child(t, "accessor"))
            .ok_or_else(|| malformed(format!("source `{key}` has no <accessor>")))?;
        let count = count_attribute(accessor, "count", 0)?;
        let stride = count_attribute(accessor, "stride", 1)?;
        let offset = count_attribute(accessor, "offset", 0)?;
        if stride < 3 {
            return Err(malformed(format!(
                "source `{key}`: positions need a stride of at least 3, found {stride}"
            )));
        }
        let array = match accessor.attribute("source") {
            Some(url) => self.resolve(url, "float_array")?,
            None => child(source, "float_array")
                .ok_or_else(|| malformed(format!("source `{key}` has no <float_array>")))?,
        };
        let values = floats(array.text().unwrap_or(""), &format!("source `{key}`"))?;
        // Checked before anything is allocated for `count`.
        let needed = match count {
            0 => Some(0),
            n => (n - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(offset))
                .and_then(|v| v.checked_add(3)),
        };
        if needed.is_none_or(|needed| needed > values.len()) {
            return Err(malformed(format!(
                "source `{key}` claims {count} positions but holds {} numbers",
                values.len()
            )));
        }
        let positions: Vec<[f64; 3]> = (0..count)
            .map(|i| {
                let at = offset + i * stride;
                [values[at], values[at + 1], values[at + 2]]
            })
            .collect();
        let positions = Rc::new(positions);
        self.positions.insert(key, Rc::clone(&positions));
        Ok(positions)
    }

    /// The positions a primitive's `VERTEX` input points at, through the
    /// mesh's `<vertices>` element or directly at a `<source>`.
    fn vertex_positions(&mut self, url: &str) -> Result<Rc<Vec<[f64; 3]>>, MeshError> {
        let target = self
            .resolve(url, "vertices")
            .or_else(|_| self.resolve(url, "source"))?;
        let source = if is(target, "vertices") {
            let position = children(target, "input")
                .find(|i| i.attribute("semantic") == Some("POSITION"))
                .and_then(|i| i.attribute("source"))
                .ok_or_else(|| malformed(format!("<vertices> `{url}` has no POSITION input")))?;
            self.resolve(position, "source")?
        } else {
            target
        };
        self.source_positions(source)
    }

    fn geometry(
        &mut self,
        geometry: Node<'a, 'i>,
        world: &Matrix,
        bound: &HashMap<String, Option<String>>,
    ) -> Result<(), MeshError> {
        let Some(mesh) = child(geometry, "mesh") else {
            self.not_meshes += 1;
            return Ok(());
        };
        let mirrored = determinant(world) < 0.0;
        for primitive in mesh.children().filter(Node::is_element) {
            let kind = match primitive.tag_name().name() {
                "triangles" => Kind::Triangles,
                "polylist" => Kind::Polylist,
                "polygons" => Kind::Polygons,
                "tristrips" => Kind::Strips,
                "trifans" => Kind::Fans,
                _ => continue,
            };
            let color = match primitive.attribute("material") {
                // Exporters that skip <bind_material> use the material's id
                // as the symbol.
                Some(symbol) => match bound.get(symbol) {
                    Some(color) => color.clone(),
                    None => self.material_color(&format!("#{symbol}")),
                },
                None => None,
            };
            let mut stride = 0;
            let mut vertex: Option<(usize, &str)> = None;
            for input in children(primitive, "input") {
                let offset = count_attribute(input, "offset", 0)?;
                stride = stride.max(offset + 1);
                if input.attribute("semantic") == Some("VERTEX") {
                    vertex = Some((offset, input.attribute("source").unwrap_or("")));
                }
            }
            let (offset, url) =
                vertex.ok_or_else(|| malformed("a primitive has no VERTEX input"))?;
            let positions = self.vertex_positions(url)?;
            let corners = |p: &[usize]| -> Result<Vec<[f64; 3]>, MeshError> {
                if p.len() % stride != 0 {
                    return Err(malformed(format!(
                        "a <p> holds {} indices, not a multiple of {stride}",
                        p.len()
                    )));
                }
                p.chunks_exact(stride)
                    .map(|chunk| {
                        positions
                            .get(chunk[offset])
                            .map(|v| apply(world, *v))
                            .ok_or_else(|| {
                                malformed(format!(
                                    "index {} is past the {} positions",
                                    chunk[offset],
                                    positions.len()
                                ))
                            })
                    })
                    .collect()
            };
            for [a, b, c] in faces(kind, primitive, &corners)? {
                self.emit(color.as_ref(), if mirrored { [a, c, b] } else { [a, b, c] })?;
            }
        }
        Ok(())
    }

    fn emit(&mut self, color: Option<&String>, corners: [[f64; 3]; 3]) -> Result<(), MeshError> {
        self.triangles += 1;
        if self.triangles > MAX_TRIANGLES {
            return Err(MeshError::TooLarge {
                found: self.triangles,
            });
        }
        let found = self.parts.iter().position(|(c, _)| c.as_ref() == color);
        let index = if let Some(index) = found {
            index
        } else {
            self.parts.push((color.cloned(), Welder::default()));
            self.parts.len() - 1
        };
        self.parts[index].1.triangle(corners)
    }

    fn finish(self, mut notes: Vec<String>) -> Result<Parsed, MeshError> {
        if self.controllers > 0 {
            notes.push(format!(
                "{} skinned mesh(es) (<instance_controller>) were not read",
                self.controllers
            ));
        }
        if self.not_meshes > 0 {
            notes.push(format!(
                "{} geometry element(s) are not triangle meshes and were not read",
                self.not_meshes
            ));
        }
        let mut parts = Vec::new();
        for (color, welder) in self.parts {
            if welder.has_triangles() {
                parts.push(Part {
                    mesh: welder.finish()?,
                    color,
                });
            }
        }
        if parts.is_empty() {
            return Err(MeshError::Empty);
        }
        Ok(Parsed { parts, notes })
    }
}

/// A triangle's three corners, in world coordinates.
type Face = [[f64; 3]; 3];

/// Turns a `<p>` index list into world positions.
type Corners<'f> = &'f dyn Fn(&[usize]) -> Result<Vec<[f64; 3]>, MeshError>;

/// The triangles of one primitive element. `corners` turns a `<p>` list into
/// positions.
fn faces(
    kind: Kind,
    primitive: Node<'_, '_>,
    corners: Corners<'_>,
) -> Result<Vec<Face>, MeshError> {
    let mut faces: Vec<Face> = Vec::new();
    let read = |p: Node<'_, '_>| corners(&counts(p.text().unwrap_or(""), "<p>")?);
    match kind {
        Kind::Triangles => {
            if let Some(p) = child(primitive, "p") {
                let points = read(p)?;
                if points.len() % 3 != 0 {
                    return Err(malformed("<triangles> corners are not a multiple of 3"));
                }
                faces.extend(points.chunks_exact(3).map(|t| [t[0], t[1], t[2]]));
            }
        }
        Kind::Polylist => {
            if let Some(p) = child(primitive, "p") {
                let sizes = child(primitive, "vcount").map_or(Ok(Vec::new()), |v| {
                    counts(v.text().unwrap_or(""), "<vcount>")
                })?;
                let points = read(p)?;
                let total = sizes.iter().try_fold(0usize, |sum, n| sum.checked_add(*n));
                if total != Some(points.len()) {
                    return Err(malformed(
                        "<polylist> vertex counts do not add up to its indices",
                    ));
                }
                let mut start = 0;
                for size in sizes {
                    fan(&points[start..start + size], &mut faces);
                    start += size;
                }
            }
        }
        Kind::Polygons | Kind::Strips | Kind::Fans => {
            if kind == Kind::Polygons && child(primitive, "ph").is_some() {
                return Err(malformed("polygons with holes (<ph>) are not supported"));
            }
            for p in children(primitive, "p") {
                let points = read(p)?;
                if kind == Kind::Strips {
                    for (i, t) in points.windows(3).enumerate() {
                        // Every other triangle of a strip is wound the other way.
                        faces.push(if i % 2 == 0 {
                            [t[0], t[1], t[2]]
                        } else {
                            [t[1], t[0], t[2]]
                        });
                    }
                } else {
                    fan(&points, &mut faces);
                }
            }
        }
    }
    Ok(faces)
}

/// Splits a convex polygon into a fan of triangles around its first corner.
fn fan(points: &[[f64; 3]], faces: &mut Vec<Face>) {
    for i in 1..points.len().saturating_sub(1) {
        faces.push([points[0], points[i], points[i + 1]]);
    }
}

/// Reads a COLLADA file into coloured parts, in metres.
///
/// # Errors
/// [`MeshError::FileTooLarge`] beyond [`MAX_BYTES`], [`MeshError::Malformed`]
/// for anything that is not COLLADA or breaks its structure,
/// [`MeshError::TooLarge`] and [`MeshError::Empty`] as for every format.
pub fn parse(bytes: &[u8]) -> Result<Parsed, MeshError> {
    if bytes.len() > MAX_BYTES {
        return Err(MeshError::FileTooLarge {
            bytes: bytes.len(),
            limit: MAX_BYTES,
        });
    }
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let text =
        std::str::from_utf8(bytes).map_err(|_| malformed("a COLLADA file must be UTF-8 text"))?;
    let document =
        Document::parse(text).map_err(|e| malformed(format!("not well-formed XML: {e}")))?;
    let root = document.root_element();
    if !is(root, "COLLADA") {
        return Err(malformed(format!(
            "the root element is <{}>, not <COLLADA>",
            root.tag_name().name()
        )));
    }
    let mut reader = Reader::new(&document);
    let mut notes = Vec::new();
    let asset = child(root, "asset");
    let unit = match asset
        .and_then(|a| child(a, "unit"))
        .and_then(|u| u.attribute("meter"))
    {
        None => 1.0,
        Some(text) => match text.trim().parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => {
                if (v - 1.0).abs() > 1.0e-12 {
                    notes.push(format!(
                        "the file's <unit> is {v} m; its lengths were scaled by that"
                    ));
                }
                v
            }
            _ => {
                return Err(malformed(format!(
                    "<unit meter=\"{text}\"> is not a positive length"
                )))
            }
        },
    };
    if let Some(axis) = asset
        .and_then(|a| child(a, "up_axis"))
        .and_then(|n| n.text())
        .map(str::trim)
    {
        if axis != "Z_UP" {
            notes.push(format!(
                "the file declares `{axis}`; it was read with Z up, as ROS tools read COLLADA meshes"
            ));
        }
    }
    let base = multiply(
        &IDENTITY,
        &[
            [unit, 0.0, 0.0, 0.0],
            [0.0, unit, 0.0, 0.0],
            [0.0, 0.0, unit, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
    let scene = match child(root, "scene").and_then(|s| child(s, "instance_visual_scene")) {
        Some(instance) => {
            Some(reader.resolve(instance.attribute("url").unwrap_or(""), "visual_scene")?)
        }
        None => child(root, "library_visual_scenes").and_then(|l| child(l, "visual_scene")),
    };
    if let Some(scene) = scene {
        for node in children(scene, "node") {
            reader.visit(node, &base, 0)?;
        }
    } else {
        // A file with no scene is a library of meshes: take them all, unmoved.
        let library: Vec<Node<'_, '_>> = children(root, "library_geometries")
            .flat_map(|l| children(l, "geometry").collect::<Vec<_>>())
            .collect();
        for geometry in library {
            reader.geometry(geometry, &base, &HashMap::new())?;
        }
    }
    reader.finish(notes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    /// A COLLADA document around the given libraries and scene nodes.
    fn document(asset: &str, libraries: &str, nodes: &str) -> String {
        format!(
            r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>{asset}</asset>
  <library_effects>
    <effect id="red-fx"><profile_COMMON><technique sid="common"><phong>
      <diffuse><color sid="diffuse">1 0 0 1</color></diffuse></phong></technique></profile_COMMON></effect>
    <effect id="blue-fx"><profile_COMMON><technique sid="common"><lambert>
      <diffuse><color>0 0 1 1</color></diffuse></lambert></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="red"><instance_effect url="#red-fx"/></material>
    <material id="blue"><instance_effect url="#blue-fx"/></material>
  </library_materials>
  {libraries}
  <library_visual_scenes><visual_scene id="scene">{nodes}</visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>"##
        )
    }

    /// A geometry with four positions: the unit square in z = 0.
    fn square(id: &str, primitives: &str) -> String {
        format!(
            r##"<geometry id="{id}"><mesh>
  <source id="{id}-pos"><float_array id="{id}-arr" count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
    <technique_common><accessor source="#{id}-arr" count="4" stride="3">
      <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
    </accessor></technique_common></source>
  <vertices id="{id}-vtx"><input semantic="POSITION" source="#{id}-pos"/></vertices>
  {primitives}
</mesh></geometry>"##
        )
    }

    fn vertex_input(id: &str) -> String {
        format!(r##"<input semantic="VERTEX" source="#{id}-vtx" offset="0"/>"##)
    }

    fn parse_str(text: &str) -> Result<Parsed, MeshError> {
        parse(text.as_bytes())
    }

    /// Every triangle of every part as world corner positions.
    fn triangles(parsed: &Parsed) -> Vec<[[f64; 3]; 3]> {
        let mut out = Vec::new();
        for part in &parsed.parts {
            let m = &part.mesh;
            for f in m.indices.chunks_exact(3) {
                let v = |i: u32| {
                    let at = i as usize * 3;
                    [m.vertices[at], m.vertices[at + 1], m.vertices[at + 2]]
                };
                out.push([v(f[0]), v(f[1]), v(f[2])]);
            }
        }
        out
    }

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    #[test]
    fn units_node_transforms_and_material_colours_are_applied() {
        let geometry = square(
            "sq",
            &format!(
                r#"<triangles material="paint" count="2">{}<p>0 1 2 0 2 3</p></triangles>"#,
                vertex_input("sq")
            ),
        );
        let text = document(
            r#"<unit name="millimetre" meter="0.001"/><up_axis>Z_UP</up_axis>"#,
            &format!("<library_geometries>{geometry}</library_geometries>"),
            r##"<node id="n"><translate>100 0 0</translate>
                 <instance_geometry url="#sq"><bind_material><technique_common>
                   <instance_material symbol="paint" target="#red"/>
                 </technique_common></bind_material></instance_geometry></node>"##,
        );
        let parsed = parse_str(&text).unwrap();
        assert_eq!(parsed.notes.len(), 1, "{:?}", parsed.notes);
        assert!(parsed.notes[0].contains("0.001 m"), "{:?}", parsed.notes);
        assert_eq!(parsed.parts.len(), 1);
        assert_eq!(parsed.parts[0].color.as_deref(), Some("#ff0000"));
        let faces = triangles(&parsed);
        assert_eq!(faces.len(), 2);
        // Millimetres to metres, after the node's 100 mm translation.
        assert!(close(faces[0][0], [0.1, 0.0, 0.0]));
        assert!(close(faces[0][1], [0.101, 0.0, 0.0]));
        assert!(close(faces[0][2], [0.101, 0.001, 0.0]));
    }

    #[test]
    fn transforms_compose_in_document_order_through_instanced_nodes() {
        let geometry = square(
            "sq",
            &format!(
                r#"<triangles count="1">{}<p>1 2 3</p></triangles>"#,
                vertex_input("sq")
            ),
        );
        // Translate then rotate: the point is rotated about the node's own
        // origin, then moved. The other order would put (1,0,0) at (2,0,0)
        // rotated, i.e. (0,2,0).
        let text = document(
            "",
            &format!(
                r##"<library_geometries>{geometry}</library_geometries>
                <library_nodes><node id="shared"><rotate>0 0 1 90</rotate>
                  <instance_geometry url="#sq"/></node></library_nodes>"##
            ),
            r##"<node><translate>1 0 0</translate><instance_node url="#shared"/></node>"##,
        );
        let parsed = parse_str(&text).unwrap();
        let faces = triangles(&parsed);
        assert_eq!(faces.len(), 1);
        // (1,0,0) rotated 90 degrees about Z is (0,1,0); translated, (1,1,0).
        assert!(close(faces[0][0], [1.0, 1.0, 0.0]), "{:?}", faces[0]);
        // (1,1,0) -> (-1,1,0) -> (0,1,0); (0,1,0) -> (-1,0,0) -> (0,0,0).
        assert!(close(faces[0][1], [0.0, 1.0, 0.0]));
        assert!(close(faces[0][2], [0.0, 0.0, 0.0]));
    }

    #[test]
    fn every_polygon_kind_becomes_triangles_and_colours_become_parts() {
        let primitives = format!(
            r#"<polylist material="red" count="2">{v}<vcount>4 3</vcount><p>0 1 2 3 0 1 2</p></polylist>
               <polygons material="blue" count="1">{v}<p>0 1 2 3</p></polygons>
               <tristrips count="1">{v}<p>0 1 3 2</p></tristrips>
               <trifans count="1">{v}<p>0 1 2 3</p></trifans>
               <lines count="1">{v}<p>0 1</p></lines>"#,
            v = vertex_input("sq")
        );
        let geometry = square("sq", &primitives);
        // No <bind_material>: the symbols are the materials' own ids.
        let text = document(
            "",
            &format!("<library_geometries>{geometry}</library_geometries>"),
            r##"<node><instance_geometry url="#sq"/></node>"##,
        );
        let parsed = parse_str(&text).unwrap();
        let colors: Vec<Option<&str>> = parsed.parts.iter().map(|p| p.color.as_deref()).collect();
        assert_eq!(colors, [Some("#ff0000"), Some("#0000ff"), None]);
        let count = |i: usize| parsed.parts[i].mesh.triangle_count();
        // 2 + 1 from the polylist, 2 from the polygon, 2 + 2 from strip and fan.
        assert_eq!((count(0), count(1), count(2)), (3, 2, 4));
    }

    #[test]
    fn interleaved_inputs_and_mirroring_are_handled() {
        let primitives = r##"<triangles count="1">
            <input semantic="VERTEX" source="#sq-vtx" offset="0"/>
            <input semantic="NORMAL" source="#sq-pos" offset="1"/>
            <p>0 9 1 9 2 9</p></triangles>"##;
        let geometry = square("sq", primitives);
        let text = document(
            "",
            &format!("<library_geometries>{geometry}</library_geometries>"),
            r##"<node><scale>-1 1 1</scale><instance_geometry url="#sq"/></node>"##,
        );
        let faces = triangles(&parse_str(&text).unwrap());
        // Mirrored in X, and wound the other way so the face still points out.
        assert!(close(faces[0][0], [0.0, 0.0, 0.0]));
        assert!(close(faces[0][1], [-1.0, 1.0, 0.0]));
        assert!(close(faces[0][2], [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn a_file_without_a_scene_yields_its_whole_library_and_notes_the_up_axis() {
        let geometry = square(
            "sq",
            &format!(
                r#"<triangles count="1">{}<p>0 1 2</p></triangles>"#,
                vertex_input("sq")
            ),
        );
        let text = format!(
            "<COLLADA><asset><up_axis>Y_UP</up_axis></asset>
               <library_geometries>{geometry}</library_geometries></COLLADA>"
        );
        let parsed = parse_str(&text).unwrap();
        assert_eq!(parsed.parts[0].mesh.triangle_count(), 1);
        assert_eq!(parsed.notes.len(), 1);
        assert!(parsed.notes[0].contains("Y_UP"), "{:?}", parsed.notes);
    }

    #[test]
    fn broken_and_hostile_files_are_refused() {
        let tri = |p: &str| {
            format!(
                r#"<triangles count="1">{}<p>{p}</p></triangles>"#,
                vertex_input("sq")
            )
        };
        let with = |primitives: &str, nodes: &str| {
            document(
                "",
                &format!(
                    "<library_geometries>{}</library_geometries>",
                    square("sq", primitives)
                ),
                nodes,
            )
        };
        let plain = r##"<node><instance_geometry url="#sq"/></node>"##;
        let cases: Vec<(String, &str)> = vec![
            (with(&tri("0 1 7"), plain), "past the"),
            (with(&tri("0 1"), plain), "multiple of 3"),
            (with(&tri("0 1 x"), plain), "not an index"),
            (
                with(r#"<triangles count="1"><p>0 1 2</p></triangles>"#, plain),
                "no VERTEX",
            ),
            (
                with(
                    &tri("0 1 2"),
                    r#"<node><instance_geometry url="other.dae#sq"/></node>"#,
                ),
                "another file",
            ),
            (
                with(
                    &tri("0 1 2"),
                    r##"<node><instance_geometry url="#nope"/></node>"##,
                ),
                "not defined",
            ),
            (
                with(
                    &tri("0 1 2"),
                    r##"<node id="loop"><instance_node url="#loop"/></node>"##,
                ),
                "instantiates itself",
            ),
            (
                with(
                    &tri("0 1 2"),
                    r##"<node><lookat>0 0 0 1 0 0 0 0 1</lookat><instance_geometry url="#sq"/></node>"##,
                ),
                "lookat",
            ),
            (
                with(
                    &format!(
                        r#"<polygons count="1">{}<ph><p>0 1 2</p><h>0 1 2</h></ph></polygons>"#,
                        vertex_input("sq")
                    ),
                    plain,
                ),
                "holes",
            ),
            (
                with(
                    &tri("0 1 2"),
                    r##"<node><matrix>1 0 0</matrix><instance_geometry url="#sq"/></node>"##,
                ),
                "16 numbers",
            ),
            (
                r#"<!DOCTYPE COLLADA [<!ENTITY a "aaaa">]><COLLADA/>"#.to_owned(),
                "XML",
            ),
            ("<robot/>".to_owned(), "not <COLLADA>"),
            (document("<unit meter=\"-1\"/>", "", ""), "positive length"),
        ];
        for (text, expected) in cases {
            let error = parse_str(&text).expect_err(expected).to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in `{error}`"
            );
        }
    }

    #[test]
    fn files_that_cannot_be_read_as_meshes_are_refused() {
        let plain = r##"<node><instance_geometry url="#sq"/></node>"##;
        let degenerate = format!(
            r#"<triangles count="1">{}<p>0 0 0</p></triangles>"#,
            vertex_input("sq")
        );
        let with = |primitives: &str| {
            document(
                "",
                &format!(
                    "<library_geometries>{}</library_geometries>",
                    square("sq", primitives)
                ),
                plain,
            )
        };
        assert!(parse(&[0xff, 0xfe, 0x00])
            .unwrap_err()
            .to_string()
            .contains("UTF-8"));
        assert_eq!(
            parse_str(&with(&degenerate)).unwrap_err(),
            MeshError::Empty,
            "a triangle with no area leaves nothing"
        );
        // A claim of a billion positions over four numbers is refused before
        // anything is allocated for it.
        let tri = format!(
            r#"<triangles count="1">{}<p>0 1 2</p></triangles>"#,
            vertex_input("sq")
        );
        let huge = square("sq", &tri).replace(
            r#"count="4" stride="3""#,
            r#"count="1000000000" stride="3""#,
        );
        let text = document(
            "",
            &format!("<library_geometries>{huge}</library_geometries>"),
            plain,
        );
        assert!(parse_str(&text).unwrap_err().to_string().contains("claims"));
    }

    #[test]
    fn a_scene_that_multiplies_nodes_is_cut_off() {
        let geometry = square(
            "sq",
            &format!(r#"<triangles count="0">{}</triangles>"#, vertex_input("sq")),
        );
        // Ten levels of ten instances each would be ten billion visits.
        let mut library = String::from(r##"<node id="l0"><instance_geometry url="#sq"/></node>"##);
        for level in 1..=10 {
            let _ = write!(library, r#"<node id="l{level}">"#);
            for _ in 0..10 {
                let _ = write!(library, r##"<instance_node url="#l{}"/>"##, level - 1);
            }
            library.push_str("</node>");
        }
        let text = document(
            "",
            &format!("<library_geometries>{geometry}</library_geometries><library_nodes>{library}</library_nodes>"),
            r##"<node><instance_node url="#l10"/></node>"##,
        );
        let error = parse_str(&text).unwrap_err().to_string();
        assert!(error.contains("more than"), "{error}");
    }
}
