//! Phase 3 acceptance: in the demo cell and in two cells with obstacles, the
//! whole cycle is planned with every segment proven clear, a dense sampled
//! check agrees, the obstacles force detours, and the program a controller
//! would run takes the cycle time the profiles give, to well within 2 %.

use axisforge_collision::{Attachment, Scene, SceneOptions};
use axisforge_domain::templates::{self, GripperKind, Obstacles};
use axisforge_domain::{Project, Vec3};
use axisforge_kinematics::{forward_kinematics, CancellationToken, KinematicModel};
use axisforge_motion::{
    plan_cycle, program, sampled, simulate_program, Cycle, CycleOptions, MotionChecker,
};
use axisforge_service::demo_home;

/// What may touch in each segment, rebuilt from the cycle as the planner
/// defines it: the fingers and the part while closing and opening, the part
/// and the table while lifting and lowering.
fn scene_for(project: &Project, cycle: &Cycle, segment: &str) -> (Project, SceneOptions) {
    let gripper = project.robot.gripper.as_ref().unwrap();
    let part = cycle.body_id.clone();
    let table = project
        .environment
        .body_by_name("table")
        .unwrap()
        .id
        .clone();
    let mut placed = project.clone();
    placed
        .environment
        .bodies
        .iter_mut()
        .find(|b| b.id == part)
        .unwrap()
        .pose = cycle.place_pose;
    let fingers: Vec<(String, String)> = gripper
        .finger_link_ids
        .iter()
        .map(|f| (f.clone(), part.clone()))
        .collect();
    let carried = |extra: Vec<(String, String)>| SceneOptions {
        attachment: Some(Attachment {
            body_id: part.clone(),
            link_id: gripper.palm_link_id.clone(),
            relative_pose: cycle.grasp.part_in_palm,
        }),
        extra_ignored: extra,
    };
    let holding = SceneOptions {
        attachment: None,
        extra_ignored: fingers,
    };
    match segment {
        "open" | "approach" | "descend" => (project.clone(), SceneOptions::default()),
        "close" => (project.clone(), holding),
        "lift" => (project.clone(), carried(vec![(part.clone(), table)])),
        "transfer" => (project.clone(), carried(Vec::new())),
        "lower" => (project.clone(), carried(vec![(part.clone(), table)])),
        "release" => (placed, holding),
        _ => (placed, SceneOptions::default()),
    }
}

fn accept(obstacles: Obstacles, gripper: GripperKind) -> (Cycle, f64) {
    let project = templates::demo_cell(gripper, obstacles).unwrap();
    let model = KinematicModel::from_robot(&project.robot).unwrap();
    let home = demo_home(&project, &model).unwrap();
    let cube = project.environment.body_by_name("cube").unwrap();
    let target = Vec3::new(0.33, 0.21, cube.pose.translation.z);
    let options = CycleOptions::default();
    let report = plan_cycle(
        &project,
        &cube.id,
        target,
        &home,
        &options,
        &CancellationToken::new(),
    )
    .unwrap();
    let cycle = report
        .cycle
        .unwrap_or_else(|| panic!("{obstacles:?} {gripper:?}: {:?}", report.attempts));
    let margin = options.grasp.collision_margin;
    assert!(
        cycle.clearance > margin,
        "certified clear of the margin everywhere"
    );

    for segment in &cycle.segments {
        let (scene_project, scene) = scene_for(&project, &cycle, &segment.name);
        // Proven clear again, independently of the planner's own checkers.
        let checker = MotionChecker::new(&scene_project, &model, &scene, margin).unwrap();
        for m in &segment.moves {
            let verdict = checker.check_segment(&m.from, &m.to).unwrap();
            assert!(
                verdict.is_clear(),
                "{obstacles:?} {}: {verdict:?}",
                segment.name
            );
            // Dense samples agree: nothing within the margin at any of them.
            for i in 0..=40 {
                let t = f64::from(i) / 40.0;
                let q: Vec<f64> = m
                    .from
                    .iter()
                    .zip(&m.to)
                    .map(|(a, b)| a + (b - a) * t)
                    .collect();
                let pose = forward_kinematics(&model, &q).unwrap();
                let report = Scene::build(&scene_project, &model, &pose, &scene)
                    .unwrap()
                    .check(margin);
                assert!(
                    report.is_clear(),
                    "{obstacles:?} {} sample {i}: {:?}",
                    segment.name,
                    report.contacts
                );
            }
        }
    }

    // The profiles keep every axis within its limits.
    let samples = sampled(&cycle, 0.004).unwrap();
    for pair in samples.points.windows(2) {
        let dt = pair[1].time_from_start - pair[0].time_from_start;
        for (j, v) in pair[1].velocities.iter().enumerate() {
            assert!(v.abs() <= cycle.limits.velocity[j] * (1.0 + 1e-9));
            if dt > 1e-9 && (pair[1].velocities[j] - pair[0].velocities[j]).abs() > 0.0 {
                // Finite differences across a profile corner can exceed the
                // limit by one sample's worth; within a phase they do not.
                let a = (pair[1].velocities[j] - pair[0].velocities[j]) / dt;
                assert!(a.abs() <= cycle.limits.acceleration[j] * (1.0 + 1e-6) + 1e-9);
            }
        }
    }

    // A controller interpolating every 8 ms takes the planned time.
    let simulated = simulate_program(&program(&cycle), 0.008);
    let error = (simulated - cycle.cycle_time).abs() / cycle.cycle_time;
    eprintln!(
        "{obstacles:?} {gripper:?}: cycle {:.3} s, program {:.3} s ({:.3} %), {} segments, \
         certified clearance {:.2} mm, closest measured {:.1} mm",
        cycle.cycle_time,
        simulated,
        error * 100.0,
        cycle.segments.len(),
        cycle.clearance * 1000.0,
        cycle.closest * 1000.0
    );
    assert!(error < 0.02);
    (cycle, error)
}

/// The straight line from lift to above the target is blocked, so the
/// transfer had to go round.
fn transfer_detours(obstacles: Obstacles, cycle: &Cycle) -> bool {
    let project = templates::demo_cell(GripperKind::Parallel, obstacles).unwrap();
    let model = KinematicModel::from_robot(&project.robot).unwrap();
    let (scene_project, scene) = scene_for(&project, cycle, "transfer");
    let checker = MotionChecker::new(&scene_project, &model, &scene, 0.001).unwrap();
    let transfer = cycle
        .segments
        .iter()
        .find(|s| s.name == "transfer")
        .unwrap();
    let from = &transfer.moves.first().unwrap().from;
    let to = &transfer.moves.last().unwrap().to;
    transfer.moves.len() > 1 && !checker.check_segment(from, to).unwrap().is_clear()
}

#[test]
fn the_demo_cell_cycle_is_proven_and_timed() {
    accept(Obstacles::None, GripperKind::Parallel);
}

#[test]
fn a_wall_is_gone_round_and_the_cycle_is_proven_and_timed() {
    let (cycle, _) = accept(Obstacles::Wall, GripperKind::Parallel);
    assert!(transfer_detours(Obstacles::Wall, &cycle));
}

#[test]
fn posts_are_gone_round_and_the_cycle_is_proven_and_timed() {
    let (cycle, _) = accept(Obstacles::Posts, GripperKind::Parallel);
    assert!(transfer_detours(Obstacles::Posts, &cycle));
}

#[test]
fn a_hand_runs_the_cycle_in_the_wall_cell() {
    accept(Obstacles::Wall, GripperKind::Hand);
}
