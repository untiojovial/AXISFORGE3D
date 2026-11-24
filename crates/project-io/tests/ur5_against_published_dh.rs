//! Imports a real UR5 URDF and checks forward kinematics against the
//! Denavit-Hartenberg parameters Universal Robots publishes for the UR5.
//!
//! The DH parameters are the manufacturer's, not derived from the URDF, so
//! this check shares nothing with the importer: an error in reading URDF
//! origins, roll-pitch-yaw, axes or the tree would show up here.

use axisforge_domain::Transform;
use axisforge_kinematics::{forward_kinematics, KinematicModel, Rng};
use axisforge_project_io::import::{ImportOptions, NoMeshes};
use axisforge_project_io::urdf::import_urdf;

const UR5: &str = include_str!("data/ur5/ur5_robot.urdf");

/// Standard DH parameters for the UR5, as published by Universal Robots.
const D: [f64; 6] = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823];
const A: [f64; 6] = [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0];
const ALPHA: [f64; 6] = [
    std::f64::consts::FRAC_PI_2,
    0.0,
    0.0,
    std::f64::consts::FRAC_PI_2,
    -std::f64::consts::FRAC_PI_2,
    0.0,
];
const JOINTS: [&str; 6] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
];

type Matrix = [[f64; 4]; 4];

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn dh(theta: f64, d: f64, a: f64, alpha: f64) -> Matrix {
    let (st, ct) = theta.sin_cos();
    let (sa, ca) = alpha.sin_cos();
    [
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn matrix(t: &Transform) -> Matrix {
    let mut out = [[0.0; 4]; 4];
    for (column, axis) in [
        axisforge_domain::Vec3::X,
        axisforge_domain::Vec3::Y,
        axisforge_domain::Vec3::Z,
    ]
    .iter()
    .enumerate()
    {
        let v = t.apply_direction(axis);
        out[0][column] = v.x;
        out[1][column] = v.y;
        out[2][column] = v.z;
    }
    out[0][3] = t.translation.x;
    out[1][3] = t.translation.y;
    out[2][3] = t.translation.z;
    out[3][3] = 1.0;
    out
}

#[test]
fn the_imported_ur5_matches_the_published_dh_parameters() {
    let imported = import_urdf(UR5, &NoMeshes, &ImportOptions::default()).expect("imports");
    let robot = &imported.project.robot;
    assert_eq!(robot.axis_count(), 6);
    let model = KinematicModel::from_robot(robot).unwrap();
    let base = model
        .link_index(&robot.link_by_name("base").unwrap().id)
        .unwrap();
    let tool = model
        .link_index(&robot.link_by_name("tool0").unwrap().id)
        .unwrap();
    let slot = |name: &str| {
        (0..model.dof())
            .find(|i| model.joint_for_value(*i).unwrap().name == name)
            .unwrap()
    };

    let mut rng = Rng::new(2026);
    let mut worst_position = 0.0_f64;
    let mut worst_rotation = 0.0_f64;
    for _ in 0..50 {
        let q: Vec<f64> = (0..6)
            .map(|_| rng.next_in(-std::f64::consts::PI, std::f64::consts::PI))
            .collect();
        let mut values = vec![0.0; model.dof()];
        for (name, value) in JOINTS.iter().zip(&q) {
            values[slot(name)] = *value;
        }
        let pose = forward_kinematics(&model, &values).unwrap();
        let got = matrix(&pose.link(base).inverse().compose(&pose.link(tool)));
        let mut expected = dh(q[0], D[0], A[0], ALPHA[0]);
        for i in 1..6 {
            expected = multiply(&expected, &dh(q[i], D[i], A[i], ALPHA[i]));
        }
        let position = (0..3)
            .map(|i| (got[i][3] - expected[i][3]).powi(2))
            .sum::<f64>()
            .sqrt();
        let rotation = (0..3)
            .flat_map(|i| (0..3).map(move |k| (i, k)))
            .map(|(i, k)| (got[i][k] - expected[i][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        worst_position = worst_position.max(position);
        worst_rotation = worst_rotation.max(rotation);
    }
    // The URDF writes pi/2 as 1.57079632679, so agreement is limited to about
    // 1e-11; anything near the 0.1 mm acceptance bound would be a real error.
    assert!(
        worst_position < 1.0e-9,
        "position differs by {worst_position} m"
    );
    assert!(
        worst_rotation < 1.0e-9,
        "rotation differs by {worst_rotation}"
    );
}

#[test]
fn the_imported_ur5_reports_what_it_could_not_carry_over() {
    let imported = import_urdf(UR5, &NoMeshes, &ImportOptions::default()).unwrap();
    let joined = imported.warnings.join("\n");
    assert!(
        joined.contains("mesh file(s) were not imported"),
        "{joined}"
    );
    assert!(joined.contains("<transmission>"), "{joined}");
    assert!(joined.contains("default tool frame"), "{joined}");
}
