//! Two real COLLADA meshes, exported from CAD with Blender, against the STL
//! collision meshes the same project ships for the same links. The two were
//! modelled separately, so agreement checks the reader's units, scene
//! transforms and index handling against an outside source.

use std::path::Path;

use axisforge_project_io::mesh;

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn ur5_visual_meshes_fit_their_collision_meshes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for link in ["base", "wrist3"] {
        let visual = mesh::parse_parts(
            "v.dae",
            &read(&root.join(format!("tests/data/ur5/visual/{link}.dae"))),
        )
        .unwrap();
        assert!(visual.notes.is_empty(), "{link}: {:?}", visual.notes);
        // Blender kept the CAD's two materials: grey body, black trim.
        assert_eq!(visual.parts.len(), 2, "{link}");
        assert!(visual.parts.iter().all(|p| p.color.is_some()), "{link}");
        let visual = visual.merged().unwrap();
        assert!(visual.triangle_count() > 1000, "{link}");

        let collision = mesh::parse(
            "c.stl",
            &read(&root.join(format!("../../apps/viewer/samples/ur5/meshes/{link}.stl"))),
        )
        .unwrap();
        let (vl, vh) = visual.bounds().unwrap();
        let (cl, ch) = collision.bounds().unwrap();
        for axis in 0..3 {
            for (v, c) in [(vl[axis], cl[axis]), (vh[axis], ch[axis])] {
                assert!(
                    (v - c).abs() < 1.0e-3,
                    "{link} axis {axis}: visual {v} vs collision {c}"
                );
            }
        }
    }
}
