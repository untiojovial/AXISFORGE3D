//! Parts given as meshes, planned end to end in the demo cell.
//!
//! Each part is a primitive tessellated into a triangle mesh and handed to
//! the planner as a mesh resource, so nothing tells it the shape: the
//! surface generator must find the grasps from the triangles alone. Each
//! must be picked in force closure within a bounded search.

use axisforge_domain::{
    templates, GeometryInstance, GeometryRole, PhysicalProperties, Project, Resource, Shape,
    Transform, Vec3, WorldBody,
};
use axisforge_grasping::{plan_pick, PartSurface, PlannerOptions};
use axisforge_kinematics::CancellationToken;

/// The demo cell with its cube replaced by `shape` as a mesh, resting on the
/// table where the cube was.
fn cell_with(shape: Shape) -> Project {
    let mut project = templates::pick_and_place_cell().unwrap();
    let alone = WorldBody::manipulable(
        "probe",
        "probe",
        Transform::IDENTITY,
        GeometryInstance::visual("g", "g", shape).with_role(GeometryRole::Both),
        PhysicalProperties {
            mass: 0.1,
            friction_coefficient: 0.5,
        },
    );
    let surface = PartSurface::of(&alone, &[]).unwrap();
    let lowest = surface
        .vertices
        .iter()
        .map(|v| v.z)
        .fold(f64::INFINITY, f64::min);
    project.resources.push(Resource {
        id: "part-mesh".to_owned(),
        name: "part.stl".to_owned(),
        media_type: "model/stl".to_owned(),
        vertices: surface.vertices.iter().flat_map(|v| v.to_array()).collect(),
        indices: surface.triangles.iter().flatten().copied().collect(),
    });
    let cube = project
        .environment
        .bodies
        .iter_mut()
        .find(|b| b.name == "cube")
        .unwrap();
    cube.pose = Transform::from_translation(Vec3::new(0.42, -0.1, 0.42 - lowest));
    cube.geometries = vec![GeometryInstance::visual(
        "part-mesh-geometry",
        "part",
        Shape::Mesh {
            resource_id: "part-mesh".to_owned(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        },
    )
    .with_role(GeometryRole::Both)];
    project.validate().unwrap();
    project
}

#[test]
fn tessellated_parts_are_picked_through_their_surface() {
    let seed = [0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0];
    let options = PlannerOptions {
        max_candidates: Some(300),
        ..PlannerOptions::default()
    };
    for (label, shape) in [
        ("sphere", Shape::Sphere { radius: 0.025 }),
        (
            "cylinder",
            Shape::Cylinder {
                radius: 0.015,
                length: 0.06,
            },
        ),
        (
            "flat box",
            Shape::Box {
                size: Vec3::new(0.03, 0.05, 0.02),
            },
        ),
        (
            "capsule",
            Shape::Capsule {
                radius: 0.012,
                length: 0.05,
            },
        ),
    ] {
        let project = cell_with(shape);
        let id = project.environment.body_by_name("cube").unwrap().id.clone();
        let report = plan_pick(&project, &id, &seed, &options, &CancellationToken::new()).unwrap();
        assert_eq!(report.candidate_generator, "surface_normal", "{label}");
        assert!(
            !report.plans.is_empty(),
            "{label}: no plan in {} candidates examined: {:?}",
            report.candidates_examined,
            report
                .rejections
                .iter()
                .map(|r| r.reason)
                .collect::<Vec<_>>()
        );
        for plan in &report.plans {
            assert!(plan.quality.force_closure, "{label}");
            assert!(plan.quality.quality > 0.0, "{label}");
            // Approached from above, as a part on a table must be.
            assert!(
                plan.candidate.approach.z < 0.0,
                "{label}: {:?}",
                plan.candidate.approach
            );
        }
    }
}
