//! An outside prior changes the order candidates are examined in, and
//! nothing else.

use axisforge_domain::templates;
use axisforge_grasping::{plan_pick, CandidatePrior, PlanReport, PlannerOptions};
use axisforge_kinematics::CancellationToken;

const HOME: [f64; 8] = [0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0];

fn plan(options: &PlannerOptions) -> PlanReport {
    let project = templates::pick_and_place_cell().unwrap();
    let cube = project.environment.body_by_name("cube").unwrap();
    plan_pick(
        &project,
        &cube.id,
        &HOME,
        options,
        &CancellationToken::new(),
    )
    .unwrap()
}

fn first_examined(report: &PlanReport) -> String {
    report
        .rejections
        .first()
        .map(|r| r.label.clone())
        .or_else(|| report.plans.first().map(|p| p.candidate.label.clone()))
        .unwrap()
}

#[test]
fn a_prior_reorders_the_search_and_changes_no_verdict() {
    let all = plan(&PlannerOptions {
        nr_grasps: 50,
        ..PlannerOptions::default()
    });
    let default_first = plan(&PlannerOptions {
        nr_grasps: 1,
        ..PlannerOptions::default()
    });
    // A feasible grasp the default order does not try first.
    let favoured = all
        .plans
        .iter()
        .map(|p| p.candidate.label.clone())
        .find(|l| *l != first_examined(&default_first))
        .expect("the cube has more than one feasible grasp");
    let with_prior = plan(&PlannerOptions {
        nr_grasps: 1,
        candidate_prior: vec![CandidatePrior {
            label: favoured.clone(),
            score: 1.0,
        }],
        ..PlannerOptions::default()
    });
    assert_eq!(with_prior.candidates_examined, 1);
    assert_eq!(with_prior.plans[0].candidate.label, favoured);
    assert_eq!(with_prior.prior_matched, 1);
    assert_eq!(with_prior.prior_unmatched, 0);
    // The same grasp, checked the same way.
    let unprimed = all
        .plans
        .iter()
        .find(|p| p.candidate.label == favoured)
        .unwrap();
    assert_eq!(with_prior.plans[0], *unprimed);
}

#[test]
fn a_favoured_grasp_that_fails_is_still_refused() {
    let all = plan(&PlannerOptions {
        nr_grasps: 50,
        ..PlannerOptions::default()
    });
    let doomed = all
        .rejections
        .last()
        .expect("some grasp is refused")
        .clone();
    let with_prior = plan(&PlannerOptions {
        nr_grasps: 1,
        candidate_prior: vec![CandidatePrior {
            label: doomed.label.clone(),
            score: 1.0,
        }],
        ..PlannerOptions::default()
    });
    assert_eq!(first_examined(&with_prior), doomed.label);
    assert_eq!(with_prior.rejections[0].reason, doomed.reason);
    assert_eq!(with_prior.plans.len(), 1, "the search goes on after it");
}

#[test]
fn a_prior_for_something_else_is_reported_and_harmless() {
    let plain = plan(&PlannerOptions::default());
    let stray = plan(&PlannerOptions {
        candidate_prior: vec![CandidatePrior {
            label: "no such grasp".to_owned(),
            score: 0.9,
        }],
        ..PlannerOptions::default()
    });
    assert_eq!(stray.prior_matched, 0);
    assert_eq!(stray.prior_unmatched, 1);
    assert_eq!(stray.plans, plain.plans);
    assert_eq!(stray.rejections, plain.rejections);
}

#[test]
fn malformed_priors_are_refused() {
    for options in [
        PlannerOptions {
            candidate_prior: vec![CandidatePrior {
                label: "a".to_owned(),
                score: 1.5,
            }],
            ..PlannerOptions::default()
        },
        PlannerOptions {
            candidate_prior: vec![
                CandidatePrior {
                    label: "a".to_owned(),
                    score: 0.5,
                },
                CandidatePrior {
                    label: "a".to_owned(),
                    score: 0.6,
                },
            ],
            ..PlannerOptions::default()
        },
        PlannerOptions {
            prior_weight: 9.0,
            ..PlannerOptions::default()
        },
    ] {
        assert!(options.validate().is_err());
    }
}
