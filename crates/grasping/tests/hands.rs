//! Hands plan picks in the demo cell: the fingers close until they touch,
//! the contacts are judged in the wrench space, and the arm reaches the grasp
//! without collision.

use axisforge_domain::templates::{self, GripperKind};
use axisforge_grasping::{plan_pick, PlannerOptions, HAND_GENERATOR};
use axisforge_kinematics::{CancellationToken, KinematicModel};

fn home(model: &KinematicModel) -> Vec<f64> {
    let mut seed = vec![0.0; model.dof()];
    seed[..6].copy_from_slice(&[0.0, 0.3, 1.0, 0.0, 1.84, 0.0]);
    model.clamped(&seed)
}

#[test]
fn hands_pick_both_demo_parts() {
    for kind in [GripperKind::ThreeFinger, GripperKind::Hand] {
        let project = templates::pick_and_place_cell_with(kind).unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let gripper = project.robot.gripper.as_ref().unwrap();
        for part in ["cube", "dowel"] {
            let body = project.environment.body_by_name(part).unwrap();
            let start = std::time::Instant::now();
            let report = plan_pick(
                &project,
                &body.id,
                &home(&model),
                &PlannerOptions {
                    // Fewer samples than the default keep the test quick in
                    // debug builds; the demo parts need no more.
                    surface_samples: 24,
                    angular_samples: 4,
                    ..PlannerOptions::default()
                },
                &CancellationToken::new(),
            )
            .unwrap();
            eprintln!(
                "{kind:?} {part}: {} plans, {} of {} examined, {:.2} s",
                report.plans.len(),
                report.candidates_examined,
                report.candidates_total,
                start.elapsed().as_secs_f64()
            );
            let mut tally = std::collections::BTreeMap::new();
            for r in &report.rejections {
                *tally.entry(format!("{:?}", r.reason)).or_insert(0) += 1;
            }
            eprintln!("  {tally:?}");
            assert_eq!(report.candidate_generator, HAND_GENERATOR);
            assert!(!report.plans.is_empty(), "{kind:?} {part}: {tally:?}");
            for plan in &report.plans {
                assert!(plan.quality.force_closure);
                let hand = plan.candidate.hand.as_ref().unwrap();
                assert!(hand.fingers_in_contact >= 2);
                // The configurations carry the finger postures.
                for (index, joint_id) in gripper.jaw_joint_ids.iter().enumerate() {
                    let joint = model.joint_index(joint_id).unwrap();
                    let slot = model.joints()[joint].value_index.unwrap();
                    assert!((plan.pregrasp[slot] - hand.open[index]).abs() < 1e-12);
                    assert!((plan.grasp_closed[slot] - hand.closed[index]).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn hands_refuse_two_jaw_extensions_and_jaws_refuse_the_hand_generator() {
    let hand = templates::pick_and_place_cell_with(GripperKind::Hand).unwrap();
    let model = KinematicModel::from_robot(&hand.robot).unwrap();
    let cube = hand.environment.body_by_name("cube").unwrap();
    let plan = |project: &axisforge_domain::Project, options: PlannerOptions| {
        plan_pick(
            project,
            &cube.id,
            &home(&model),
            &options,
            &CancellationToken::new(),
        )
    };
    for options in [
        PlannerOptions {
            candidate_generator: "surface_normal".to_owned(),
            ..PlannerOptions::default()
        },
        PlannerOptions {
            quality_measure: "antipodal".to_owned(),
            ..PlannerOptions::default()
        },
        PlannerOptions {
            preshape: Some("fist".to_owned()),
            ..PlannerOptions::default()
        },
    ] {
        assert!(plan(&hand, options).is_err());
    }
    let jaws = templates::pick_and_place_cell().unwrap();
    let error = plan_pick(
        &jaws,
        &jaws.environment.body_by_name("cube").unwrap().id,
        &[0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0],
        &PlannerOptions {
            candidate_generator: HAND_GENERATOR.to_owned(),
            ..PlannerOptions::default()
        },
        &CancellationToken::new(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("parallel jaws"), "{error}");
}

#[test]
fn a_named_preshape_is_the_one_approached_in() {
    let project = templates::pick_and_place_cell_with(GripperKind::ThreeFinger).unwrap();
    let model = KinematicModel::from_robot(&project.robot).unwrap();
    let dowel = project.environment.body_by_name("dowel").unwrap();
    let report = plan_pick(
        &project,
        &dowel.id,
        &home(&model),
        &PlannerOptions {
            preshape: Some("half".to_owned()),
            surface_samples: 12,
            angular_samples: 2,
            nr_grasps: 1,
            ..PlannerOptions::default()
        },
        &CancellationToken::new(),
    )
    .unwrap();
    let plan = report
        .plans
        .first()
        .expect("a 30 mm dowel fits the half preshape");
    let hand = plan.candidate.hand.as_ref().unwrap();
    assert_eq!(hand.preshape, "half");
    assert!(hand.open.iter().all(|v| (v - 0.025).abs() < 1e-12));
}
