//! The native learned-order models reproduce scikit-learn's probabilities for the frozen Phase VII models.
//!
//! Fixture: `research/phase7_extension/native/fixture_sklearn.json` (seeded random feature vectors with missing
//! values, and the frozen models' `predict_proba`). Models: `research/phase7_extension/native/learned_order_phase7.json`.

use axisforge_grasping::learned::FEATURES;
use axisforge_grasping::LearnedModels;
use std::path::PathBuf;

fn repo(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

#[test]
fn native_trees_reproduce_scikit_learn_probabilities() {
    let models: LearnedModels = serde_json::from_str(
        &std::fs::read_to_string(repo(
            "research/phase7_extension/native/learned_order_phase7.json",
        ))
        .unwrap(),
    )
    .unwrap();
    models.validate().unwrap();
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo(
            "research/phase7_extension/native/fixture_sklearn.json",
        ))
        .unwrap(),
    )
    .unwrap();
    let rows = fixture["features"].as_array().unwrap();
    assert_eq!(rows.len(), 64);
    let robustness = models.robustness.as_ref().unwrap();
    let mut worst = 0.0f64;
    for (i, row) in rows.iter().enumerate() {
        let mut x = [0.0; FEATURES];
        for (k, v) in row.as_array().unwrap().iter().enumerate() {
            x[k] = v.as_f64().unwrap_or(f64::NAN);
        }
        let pf = models.feasibility.probability(&x);
        let ph = robustness.probability(&x);
        worst = worst
            .max((pf - fixture["feasibility"][i].as_f64().unwrap()).abs())
            .max((ph - fixture["robustness"][i].as_f64().unwrap()).abs());
    }
    assert!(worst <= 1e-12, "largest difference {worst}");
}
