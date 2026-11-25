//! The whole cycle in the demo cell.

use axisforge_domain::{templates, Vec3};
use axisforge_kinematics::{CancellationToken, KinematicModel};
use axisforge_motion::{
    csv, plan_cycle, program, sampled, simulate_program, urscript, CycleOptions, PartState,
    SegmentKind,
};

const HOME: [f64; 8] = [0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0];

#[test]
fn the_cube_is_carried_across_the_table() {
    let project = templates::pick_and_place_cell().unwrap();
    let cube = project.environment.body_by_name("cube").unwrap();
    let target = Vec3::new(0.36, 0.2, cube.pose.translation.z);
    let t = std::time::Instant::now();
    let report = plan_cycle(
        &project,
        &cube.id,
        target,
        &HOME,
        &CycleOptions::default(),
        &CancellationToken::new(),
    )
    .unwrap();
    eprintln!(
        "planned in {:.2} s; attempts {:?}",
        t.elapsed().as_secs_f64(),
        report.attempts
    );
    let cycle = report.cycle.expect("a cycle");
    for s in &cycle.segments {
        eprintln!(
            "{:>9} {:>8} {:?} {} moves {:.3} s (+{:.2}) clearance {:.1} mm, closest {:.1} mm",
            s.name,
            s.planner,
            s.part,
            s.moves.len(),
            s.duration,
            s.dwell,
            s.clearance * 1000.0,
            s.closest * 1000.0
        );
    }
    eprintln!(
        "cycle {:.3} s, clearance {:.1} mm",
        cycle.cycle_time,
        cycle.clearance * 1000.0
    );
    assert!(cycle.clearance > 0.001);
    // The segments join up.
    for pair in cycle.segments.windows(2) {
        let end = &pair[0].moves.last().unwrap().to;
        let start = &pair[1].moves[0].from;
        assert!(
            end.iter().zip(start).all(|(a, b)| (a - b).abs() < 1e-12),
            "{} -> {}",
            pair[0].name,
            pair[1].name
        );
    }
    assert_eq!(cycle.state_at(0.0).values, HOME.to_vec());
    let end = cycle.state_at(cycle.cycle_time + 1.0);
    assert!(end
        .values
        .iter()
        .zip(&HOME)
        .all(|(a, b)| (a - b).abs() < 1e-12));
    assert_eq!(end.part, PartState::Placed);
    let model = KinematicModel::from_robot(&project.robot).unwrap();
    assert_eq!(cycle.joint_names.len(), model.dof());

    // The program, run by a controller interpolating every millisecond,
    // takes the time the profiles say.
    let steps = program(&cycle);
    let simulated = simulate_program(&steps, 1.0e-3);
    let error = (simulated - cycle.cycle_time).abs() / cycle.cycle_time;
    eprintln!(
        "program {simulated:.4} s vs profile {:.4} s: {:.4} %",
        cycle.cycle_time,
        error * 100.0
    );
    assert!(error < 0.02, "{simulated} vs {}", cycle.cycle_time);

    let script = urscript(&cycle, "cube").unwrap();
    let movej = script
        .lines()
        .filter(|l| l.trim_start().starts_with("movej("))
        .count();
    let arm_moves: usize = cycle
        .segments
        .iter()
        .filter(|s| s.kind == SegmentKind::Arm)
        .map(|s| s.moves.len())
        .sum();
    assert_eq!(
        movej,
        arm_moves + 1,
        "one per arm move, and one to the start"
    );
    assert!(
        script.contains("axisforge_gripper_close(") && script.contains("axisforge_gripper_open(")
    );
    assert!(script.trim_end().ends_with("end"));

    let samples = sampled(&cycle, 0.008).unwrap();
    assert!((samples.points.last().unwrap().time_from_start - cycle.cycle_time).abs() < 1e-12);
    let text = csv(&samples);
    assert_eq!(text.lines().count(), samples.points.len() + 1);
    // Sampled velocities stay within the scaled limits.
    for point in &samples.points {
        for (j, v) in point.velocities.iter().enumerate() {
            assert!(v.abs() <= cycle.limits.velocity[j] * (1.0 + 1e-9));
        }
    }
}
