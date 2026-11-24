//! A part straight from a mesh file, with no description around it.
//!
//! CAD exports a part as STL, OBJ or COLLADA, and that file is often all
//! there is. [`import_mesh_object`] turns it into a part to pick: drawn with
//! the file's colours, colliding as its closed pieces (see
//! [`crate::mesh::Mesh::closed_pieces`]), with its origin at the centre of
//! its base so it can be stood on a surface directly.
//!
//! A mesh file says nothing about units, mass or friction. Units are given
//! by the caller or guessed from the size, and every guess and default is
//! reported, never silently assumed.

use std::collections::BTreeMap;

use axisforge_domain::{
    templates::stable_id, BodyKind, GeometryInstance, PhysicalProperties, Shape, Transform, Vec3,
    WorldBody,
};

use crate::import::{ImportError, MeshStore};
use crate::mesh;
use crate::simox::ImportedObject;

/// Density used to estimate a closed part's mass, kg/m3: water's, a neutral
/// value between plastics and wood, stated in the report.
pub const ASSUMED_DENSITY: f64 = 1000.0;

/// Size beyond which a mesh with no stated units is read as millimetres:
/// two metres is larger than anything a parallel gripper picks, and two
/// millimetres smaller than anything it can hold.
const MILLIMETRE_GUESS: f64 = 2.0;

/// The factor from the file's units to metres: the one given, or a guess
/// from the size that is reported.
fn choose_scale(
    given: Option<f64>,
    largest: f64,
    warnings: &mut Vec<String>,
) -> Result<f64, ImportError> {
    match given {
        Some(s) if s.is_finite() && s > 0.0 => Ok(s),
        Some(s) => Err(ImportError::Unsupported(format!(
            "a scale of {s} is not a positive number"
        ))),
        None if largest > MILLIMETRE_GUESS => {
            warnings.push(format!(
                "the file states no units; at {largest:.1} units across it was read as \
                 millimetres ({largest:.1} mm); say so if it is not"
            ));
            Ok(0.001)
        }
        None => {
            let millimetres = largest * 1000.0;
            warnings.push(format!(
                "the file states no units; at {largest:.3} units across it was read as metres \
                 ({millimetres:.1} mm); say so if it is not"
            ));
            Ok(1.0)
        }
    }
}

/// The part's mass from its volume at [`ASSUMED_DENSITY`], or `fallback`
/// for an open mesh; either way reported.
fn estimate_mass(
    volume: Option<f64>,
    scale: f64,
    fallback: f64,
    warnings: &mut Vec<String>,
) -> f64 {
    match volume {
        Some(volume) if volume > 0.0 => {
            let mass = volume * scale.powi(3) * ASSUMED_DENSITY;
            warnings.push(format!(
                "mass {mass:.3} kg estimated from the closed mesh's volume at \
                 {ASSUMED_DENSITY} kg/m3; set the real value before planning"
            ));
            mass
        }
        _ => {
            warnings.push(format!(
                "the mesh is not closed, so it has no volume; a mass of {fallback} kg was \
                 assumed — set the real value before planning"
            ));
            fallback
        }
    }
}

/// Imports one mesh file as a part named `name`.
///
/// `scale` converts the file's units to metres (`0.001` for millimetres);
/// `None` guesses from the size and says so.
///
/// # Errors
/// When the file cannot be read as a mesh, or the scale is not positive.
pub fn import_mesh_object(
    name: &str,
    file_name: &str,
    bytes: &[u8],
    scale: Option<f64>,
) -> Result<ImportedObject, ImportError> {
    let parsed = mesh::parse(file_name, bytes)
        .map_err(|e| ImportError::Unsupported(format!("`{file_name}`: {e}")))?;
    let (low, high) = parsed
        .bounds()
        .ok_or_else(|| ImportError::Unsupported(format!("`{file_name}` has no triangles")))?;
    let largest = (0..3).map(|i| high[i] - low[i]).fold(0.0, f64::max);
    let mut warnings = Vec::new();
    let scale = choose_scale(scale, largest, &mut warnings)?;

    let mut files = BTreeMap::new();
    files.insert(file_name.to_owned(), bytes.to_vec());
    let namespace = format!("object::{name}");
    let mut store = MeshStore::new(&namespace, &files);
    let parts = store
        .parts(file_name, axisforge_domain::GeometryRole::Both)
        .ok_or_else(|| {
            let mut reasons = Vec::new();
            store.report(&mut reasons);
            ImportError::Unsupported(reasons.join("; "))
        })?;
    // The body's origin at the centre of the part's base.
    let origin = Transform::from_translation(Vec3::new(
        -0.5 * (low[0] + high[0]) * scale,
        -0.5 * (low[1] + high[1]) * scale,
        -low[2] * scale,
    ));
    let geometries: Vec<GeometryInstance> = parts
        .into_iter()
        .enumerate()
        .map(|(index, part)| {
            let label = format!("{file_name} {}", index + 1);
            let mut geometry = GeometryInstance::visual(
                stable_id(&namespace, "geometry", &label),
                label,
                Shape::Mesh {
                    resource_id: part.resource_id,
                    scale: Vec3::new(scale, scale, scale),
                },
            )
            .with_origin(origin)
            .with_role(part.role);
            if let Some(color) = part.color {
                geometry.material.color = color;
            }
            geometry
        })
        .collect();
    store.report(&mut warnings);

    let defaults = PhysicalProperties::default();
    let mass = estimate_mass(parsed.volume(), scale, defaults.mass, &mut warnings);
    warnings.push(format!(
        "a mesh carries no friction coefficient; {} was assumed — set the real value before planning",
        defaults.friction_coefficient
    ));
    let body = WorldBody {
        id: stable_id("object", "body", name),
        name: name.to_owned(),
        kind: BodyKind::Manipulable,
        pose: Transform::IDENTITY,
        geometries,
        physical: PhysicalProperties { mass, ..defaults },
    };
    body.validate()?;
    Ok(ImportedObject {
        body,
        resources: store.resources,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed tetrahedron with 1-unit legs along the axes, raised by 5.
    fn tetra(unit: f64) -> String {
        let vertex = |p: [f64; 3]| {
            format!(
                "vertex {} {} {}\n",
                p[0] * unit,
                p[1] * unit,
                (p[2] + 5.0) * unit
            )
        };
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut text = String::from("solid t\n");
        for [a, b, c] in [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] {
            text.push_str("facet normal 0 0 0\nouter loop\n");
            for i in [a, b, c] {
                text.push_str(&vertex(corners[i]));
            }
            text.push_str("endloop\nendfacet\n");
        }
        text + "endsolid t\n"
    }

    #[test]
    fn a_millimetre_mesh_becomes_a_part_standing_on_its_base() {
        let text = tetra(40.0);
        let object = import_mesh_object("wedge", "wedge.stl", text.as_bytes(), None).unwrap();
        assert!(
            object.warnings.iter().any(|w| w.contains("millimetres")),
            "{:?}",
            object.warnings
        );
        let geometry = &object.body.geometries[0];
        let Shape::Mesh { scale, .. } = &geometry.shape else {
            panic!("a mesh")
        };
        assert!((scale.x - 0.001).abs() < 1e-15);
        // The base (z = 200 mm in the file) is at the body's origin, and
        // the 40 mm footprint is centred on it.
        let o = geometry.origin.translation;
        assert!(
            (o.z + 0.2).abs() < 1e-12 && (o.x + 0.02).abs() < 1e-12 && (o.y + 0.02).abs() < 1e-12,
            "{o:?}"
        );
        // 40^3/6 mm3 at 1000 kg/m3.
        let expected = 40.0_f64.powi(3) / 6.0 * 1.0e-9 * ASSUMED_DENSITY;
        assert!((object.body.physical.mass - expected).abs() < 1e-12);
        assert_eq!(object.body.kind, BodyKind::Manipulable);
        assert_eq!(object.resources.len(), 1);
    }

    #[test]
    fn stated_units_are_used_and_bad_ones_refused() {
        let text = tetra(0.04);
        let object = import_mesh_object("wedge", "wedge.stl", text.as_bytes(), Some(1.0)).unwrap();
        assert!(
            !object.warnings.iter().any(|w| w.contains("units")),
            "{:?}",
            object.warnings
        );
        let guessed = import_mesh_object("wedge", "wedge.stl", text.as_bytes(), None).unwrap();
        assert!(guessed
            .warnings
            .iter()
            .any(|w| w.contains("read as metres")));
        assert!(import_mesh_object("w", "w.stl", text.as_bytes(), Some(0.0)).is_err());
        assert!(import_mesh_object("w", "w.ply", text.as_bytes(), None).is_err());
        assert!(import_mesh_object("w", "w.stl", b"solid x\nendsolid x\n", None).is_err());
    }
}
