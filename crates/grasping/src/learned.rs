//! Learned examination orders computed inside the planner.
//!
//! The research pipeline used to export every candidate to Python, score it
//! with gradient-boosting models and hand the scores back as a prior file, so
//! candidates were generated twice. This module evaluates the same models on
//! the planner's own candidates: one generation, the same checks, a different
//! order. Nothing here accepts or rejects a grasp; it only decides which
//! candidate is examined next.
//!
//! The models are scikit-learn `HistGradientBoostingClassifier`s exported tree
//! by tree (`tools/research/phase7_extension/export_models.py`). Thresholds,
//! leaf values and the baseline are stored as IEEE-754 bit patterns so they
//! load exactly. A tree sends a sample left when its feature is `<=` the
//! threshold, and a missing (NaN) feature the way the tree recorded, as
//! scikit-learn does; the raw score is the baseline plus the leaves in tree
//! order, and the probability is the logistic function of it.
//!
//! The orders reproduce the research simulator's arithmetic
//! (`research_grasping/phase5/catalogue.py`, `phase6/fusion.py`).

use axisforge_domain::{DomainError, DomainResult, Project, Shape, Vec3, WorldBody};
use serde::{Deserialize, Serialize};

use crate::GraspCandidate;

/// Number of candidate features the models read, in this order.
pub const FEATURES: usize = 19;

/// Names of the features, as the models were trained on them.
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "approach_x",
    "approach_y",
    "approach_z",
    "closing_x",
    "closing_y",
    "closing_z",
    "width",
    "rel_x",
    "rel_y",
    "rel_z",
    "height",
    "log_extent_x",
    "log_extent_y",
    "log_extent_z",
    "antipodal_a",
    "antipodal_b",
    "centre_offset",
    "preliminary",
    "makeable",
];

/// Weight the simulator gives a fused score's prior, far above the
/// preliminary score so that it only breaks ties.
const FUSION_WEIGHT: f64 = 1e6;

/// One regression tree, as parallel node arrays (node 0 is the root).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tree {
    /// Feature index read at each split node.
    pub feature: Vec<u32>,
    /// Split threshold bits (`f64::from_bits`): left when `x <= threshold`.
    pub threshold: Vec<u64>,
    /// Left child index.
    pub left: Vec<u32>,
    /// Right child index.
    pub right: Vec<u32>,
    /// `true` at a leaf.
    pub leaf: Vec<bool>,
    /// Leaf value bits (already scaled by the learning rate).
    pub value: Vec<u64>,
    /// Where a NaN feature goes at each split node.
    pub missing_left: Vec<bool>,
}

impl Tree {
    fn validate(&self, context: &str) -> DomainResult<()> {
        let n = self.leaf.len();
        let lengths = [
            self.feature.len(),
            self.threshold.len(),
            self.left.len(),
            self.right.len(),
            self.value.len(),
            self.missing_left.len(),
        ];
        if n == 0 || lengths.iter().any(|&l| l != n) {
            return Err(DomainError::unsupported(format!(
                "{context}: node arrays differ in length or are empty"
            )));
        }
        for i in 0..n {
            if self.leaf[i] {
                if !f64::from_bits(self.value[i]).is_finite() {
                    return Err(DomainError::unsupported(format!(
                        "{context}: leaf {i} is not finite"
                    )));
                }
                continue;
            }
            let (l, r) = (self.left[i] as usize, self.right[i] as usize);
            if l <= i || r <= i || l >= n || r >= n {
                return Err(DomainError::unsupported(format!(
                    "{context}: node {i} has children out of order"
                )));
            }
            if self.feature[i] as usize >= FEATURES {
                return Err(DomainError::unsupported(format!(
                    "{context}: node {i} reads feature {}",
                    self.feature[i]
                )));
            }
        }
        Ok(())
    }

    fn predict(&self, x: &[f64; FEATURES]) -> f64 {
        let mut i = 0usize;
        loop {
            if self.leaf[i] {
                return f64::from_bits(self.value[i]);
            }
            let v = x[self.feature[i] as usize];
            let left = if v.is_nan() {
                self.missing_left[i]
            } else {
                v <= f64::from_bits(self.threshold[i])
            };
            i = if left { self.left[i] } else { self.right[i] } as usize;
        }
    }
}

/// A binary gradient-boosting classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeModel {
    /// Baseline raw score bits.
    pub baseline: u64,
    /// Trees, added in this order.
    pub trees: Vec<Tree>,
}

impl TreeModel {
    /// Probability of the positive class.
    pub fn probability(&self, x: &[f64; FEATURES]) -> f64 {
        let mut raw = f64::from_bits(self.baseline);
        for tree in &self.trees {
            raw += tree.predict(x);
        }
        1.0 / (1.0 + (-raw).exp())
    }
}

/// The models and constants of the learned orders, as exported for the planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LearnedModels {
    /// `axisforge-learned-order`.
    pub format: String,
    /// Format version, 1.
    pub version: u32,
    /// Feature names in model order; must equal [`FEATURE_NAMES`].
    pub features: Vec<String>,
    /// P(feasible | candidate).
    pub feasibility: TreeModel,
    /// P(dynamically robust | feasible, candidate), used by M9, Joint and the hybrid.
    pub robustness: Option<TreeModel>,
    /// M4's exponent on the feasibility prior.
    pub alpha: f64,
    /// FORTIS-R Joint's exponent on the centring term.
    pub beta: f64,
    /// Floor applied to the centring term before a power.
    pub q_floor: f64,
    /// Finger depth used to normalise the centre offset, metres.
    pub finger_depth: f64,
}

impl LearnedModels {
    /// Checks the format, the feature list, the trees and the constants.
    ///
    /// # Errors
    /// Anything that would make the models read the wrong feature or fail at
    /// prediction time.
    pub fn validate(&self) -> DomainResult<()> {
        if self.format != "axisforge-learned-order" || self.version != 1 {
            return Err(DomainError::unsupported(format!(
                "learned-order file has format '{}' version {}; expected axisforge-learned-order version 1",
                self.format, self.version
            )));
        }
        if self
            .features
            .iter()
            .map(String::as_str)
            .ne(FEATURE_NAMES.iter().copied())
        {
            return Err(DomainError::unsupported(
                "learned-order features differ from the planner's feature list".to_owned(),
            ));
        }
        for (name, model) in [
            ("feasibility", Some(&self.feasibility)),
            ("robustness", self.robustness.as_ref()),
        ] {
            if let Some(m) = model {
                if !f64::from_bits(m.baseline).is_finite() || m.trees.is_empty() {
                    return Err(DomainError::unsupported(format!(
                        "{name} model: no trees or a non-finite baseline"
                    )));
                }
                for (i, t) in m.trees.iter().enumerate() {
                    t.validate(&format!("{name} tree {i}"))?;
                }
            }
        }
        for (name, v) in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("qFloor", self.q_floor),
            ("fingerDepth", self.finger_depth),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(DomainError::unsupported(format!(
                    "learned-order {name} must be finite and non-negative"
                )));
            }
        }
        if self.finger_depth <= 0.0 {
            return Err(DomainError::unsupported(
                "learned-order fingerDepth must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Which learned order to examine candidates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OrderMode {
    /// M1: preliminary + 2 `p~` (`p~ = P(F) / max`).
    Feasibility,
    /// M4: `p~^alpha * max(Q_cen, floor)^(1 - alpha)`.
    FeasibilityCentring,
    /// M9: `p~ * P(H|F)`.
    FeasibilityRobustness,
    /// FORTIS-R Joint: `p~ * P(H|F) * max(Q_cen, floor)^beta`.
    Joint,
    /// Round robin of Joint and the original order, skipping repeats.
    Hybrid,
}

impl OrderMode {
    /// Parses the command-line name.
    ///
    /// # Errors
    /// An unknown name.
    pub fn from_name(name: &str) -> DomainResult<Self> {
        Ok(match name {
            "m1" | "feasibility" => Self::Feasibility,
            "m4" | "feasibility-centring" => Self::FeasibilityCentring,
            "m9" | "feasibility-robustness" => Self::FeasibilityRobustness,
            "joint" => Self::Joint,
            "hybrid" => Self::Hybrid,
            other => {
                return Err(DomainError::unsupported(format!(
                    "unknown learned order '{other}'; use m1, m4, m9, joint or hybrid"
                )))
            }
        })
    }

    fn needs_robustness(self) -> bool {
        matches!(
            self,
            Self::FeasibilityRobustness | Self::Joint | Self::Hybrid
        )
    }
}

/// A learned order for the planner: models plus the mode to use.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnedOrder {
    /// The models.
    pub models: std::sync::Arc<LearnedModels>,
    /// The order.
    pub mode: OrderMode,
}

impl LearnedOrder {
    /// Pairs validated models with a mode.
    ///
    /// # Errors
    /// Invalid models, or a mode that needs a robustness model the file lacks.
    pub fn new(models: LearnedModels, mode: OrderMode) -> DomainResult<Self> {
        models.validate()?;
        if mode.needs_robustness() && models.robustness.is_none() {
            return Err(DomainError::unsupported(format!(
                "order {mode:?} needs a robustness model"
            )));
        }
        Ok(Self {
            models: std::sync::Arc::new(models),
            mode,
        })
    }
}

/// Axis-aligned bounds of a body's visual geometry in world coordinates, as
/// the research features measure the part (mesh geometries only).
///
/// # Errors
/// A body without visual mesh geometry, or a missing mesh resource.
pub fn visual_bounds(project: &Project, body: &WorldBody) -> DomainResult<(Vec3, Vec3)> {
    let mut low = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut high = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut seen = false;
    for geometry in body.geometries.iter().filter(|g| g.is_visual()) {
        let Shape::Mesh { resource_id, scale } = &geometry.shape else {
            return Err(DomainError::unsupported(format!(
                "learned orders measure mesh parts; body '{}' has a primitive visual geometry",
                body.name
            )));
        };
        let resource = project
            .resources
            .iter()
            .find(|r| &r.id == resource_id)
            .ok_or_else(|| {
                DomainError::dangling("resource", resource_id.clone(), "visual_bounds")
            })?;
        for v in resource.vertices.chunks_exact(3) {
            let local = Vec3::new(v[0] * scale.x, v[1] * scale.y, v[2] * scale.z);
            let w = body.pose.apply(&geometry.origin.apply(&local));
            low = Vec3::new(low.x.min(w.x), low.y.min(w.y), low.z.min(w.z));
            high = Vec3::new(high.x.max(w.x), high.y.max(w.y), high.z.max(w.z));
            seen = true;
        }
    }
    if !seen {
        return Err(DomainError::unsupported(format!(
            "body '{}' has no visual mesh vertices",
            body.name
        )));
    }
    Ok((low, high))
}

/// The 19 research features of one candidate.
pub fn features(
    c: &GraspCandidate,
    preliminary: f64,
    makeable: bool,
    low: &Vec3,
    high: &Vec3,
) -> [f64; FEATURES] {
    let ext = Vec3::new(
        (high.x - low.x).max(1e-6),
        (high.y - low.y).max(1e-6),
        (high.z - low.z).max(1e-6),
    );
    let norm = |v: &Vec3| (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    let (na, nk) = (norm(&c.approach), norm(&c.closing));
    let a = Vec3::new(c.approach.x / na, c.approach.y / na, c.approach.z / na);
    let k = Vec3::new(c.closing.x / nk, c.closing.y / nk, c.closing.z / nk);
    let dot = |u: &Vec3, v: &Vec3| u.x * v.x + u.y * v.y + u.z * v.z;
    let offset = if c.center_offset.is_nan() {
        0.0
    } else {
        c.center_offset
    };
    [
        a.x,
        a.y,
        a.z,
        k.x.abs(),
        k.y.abs(),
        k.z.abs(),
        c.width,
        (c.center.x - low.x) / ext.x,
        (c.center.y - low.y) / ext.y,
        (c.center.z - low.z) / ext.z,
        c.center.z - low.z,
        ext.x.ln(),
        ext.y.ln(),
        ext.z.ln(),
        dot(&c.normal_a, &k).abs(),
        dot(&c.normal_b, &k).abs(),
        offset,
        preliminary,
        if makeable { 1.0 } else { 0.0 },
    ]
}

/// Per-candidate model outputs, as `research-candidates --learned` reports them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LearnedScores {
    /// P(F) per candidate, generator order.
    pub feasibility: Vec<f64>,
    /// P(H|F) per candidate, when the file has a robustness model.
    pub robustness: Option<Vec<f64>>,
}

/// Evaluates the models on every candidate (generator order).
pub fn score_candidates(models: &LearnedModels, features: &[[f64; FEATURES]]) -> LearnedScores {
    LearnedScores {
        feasibility: features
            .iter()
            .map(|x| models.feasibility.probability(x))
            .collect(),
        robustness: models
            .robustness
            .as_ref()
            .map(|m| features.iter().map(|x| m.probability(x)).collect()),
    }
}

fn minmax(x: &[f64]) -> Vec<f64> {
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    if span > 0.0 {
        x.iter().map(|v| (v - lo) / span).collect()
    } else {
        vec![0.0; x.len()]
    }
}

/// Stable descending order of `score` (Rust's and Python's sorts are both stable).
fn descending(score: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    idx
}

/// Planner arithmetic: preliminary + weight * prior, + 10 when makeable.
fn planner_order(
    preliminary: &[f64],
    makeable: &[bool],
    prior: Option<(&[f64], f64)>,
) -> Vec<usize> {
    let score: Vec<f64> = preliminary
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let s = match prior {
                Some((pr, w)) => p + w * pr[i],
                None => p,
            };
            if makeable[i] {
                s + 10.0
            } else {
                s
            }
        })
        .collect();
    descending(&score)
}

fn fusion_order(preliminary: &[f64], makeable: &[bool], s: &[f64]) -> Vec<usize> {
    let m = minmax(s);
    let prior: Vec<f64> = m
        .iter()
        .zip(makeable)
        .map(|(v, &mk)| (v + 2.0 * if mk { 1.0 } else { 0.0 }) / 3.0)
        .collect();
    planner_order(preliminary, makeable, Some((&prior, FUSION_WEIGHT)))
}

/// Round robin of two orders over the same candidates, skipping repeats.
pub fn interleave(first: &[usize], second: &[usize]) -> Vec<usize> {
    let mut seen = vec![false; first.len().max(second.len())];
    let mut out = Vec::with_capacity(seen.len());
    for k in 0..first.len().max(second.len()) {
        for order in [first, second] {
            if let Some(&i) = order.get(k) {
                if !seen[i] {
                    seen[i] = true;
                    out.push(i);
                }
            }
        }
    }
    out
}

/// The examination order (indices into the generator's candidates).
///
/// `scores` are the model outputs for the same candidates; `centre_offset` is
/// each candidate's offset from the part centre (NaN read as 0).
///
/// # Panics
///
/// If `mode` needs the robustness model and `scores` has none; validated
/// models and [`score_candidates`] never produce that combination.
pub fn order(
    models: &LearnedModels,
    mode: OrderMode,
    preliminary: &[f64],
    makeable: &[bool],
    centre_offset: &[f64],
    scores: &LearnedScores,
) -> Vec<usize> {
    let p = &scores.feasibility;
    let pmax = p.iter().copied().fold(0.0, f64::max);
    let pt: Vec<f64> = if pmax > 0.0 {
        p.iter().map(|v| v / pmax).collect()
    } else {
        p.clone()
    };
    let q = minmax(
        &centre_offset
            .iter()
            .map(|&co| {
                let co = if co.is_nan() { 0.0 } else { co };
                (1.0 - co / models.finger_depth).clamp(0.0, 1.0)
            })
            .collect::<Vec<_>>(),
    );
    let h = scores.robustness.as_deref();
    let joint = || -> Vec<f64> {
        let h = h.expect("validated: joint needs robustness");
        (0..pt.len())
            .map(|i| pt[i] * h[i] * q[i].max(models.q_floor).powf(models.beta))
            .collect()
    };
    match mode {
        OrderMode::Feasibility => planner_order(preliminary, makeable, Some((&pt, 2.0))),
        OrderMode::FeasibilityCentring => {
            let s: Vec<f64> = (0..pt.len())
                .map(|i| {
                    pt[i].powf(models.alpha) * q[i].max(models.q_floor).powf(1.0 - models.alpha)
                })
                .collect();
            fusion_order(preliminary, makeable, &s)
        }
        OrderMode::FeasibilityRobustness => {
            let h = h.expect("validated: m9 needs robustness");
            let s: Vec<f64> = (0..pt.len()).map(|i| pt[i] * h[i]).collect();
            fusion_order(preliminary, makeable, &s)
        }
        OrderMode::Joint => fusion_order(preliminary, makeable, &joint()),
        OrderMode::Hybrid => interleave(
            &fusion_order(preliminary, makeable, &joint()),
            &planner_order(preliminary, makeable, None),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(v: f64) -> Tree {
        Tree {
            feature: vec![0],
            threshold: vec![0],
            left: vec![0],
            right: vec![0],
            leaf: vec![true],
            value: vec![v.to_bits()],
            missing_left: vec![false],
        }
    }

    fn stump(feature: u32, threshold: f64, lo: f64, hi: f64) -> Tree {
        Tree {
            feature: vec![feature, 0, 0],
            threshold: vec![threshold.to_bits(), 0, 0],
            left: vec![1, 0, 0],
            right: vec![2, 0, 0],
            leaf: vec![false, true, true],
            value: vec![0, lo.to_bits(), hi.to_bits()],
            missing_left: vec![true, false, false],
        }
    }

    #[test]
    fn a_split_sends_equal_values_left_and_nan_where_recorded() {
        let t = stump(6, 0.03, -1.0, 2.0);
        let mut x = [0.0; FEATURES];
        x[6] = 0.03;
        assert_eq!(t.predict(&x).to_bits(), (-1.0f64).to_bits());
        x[6] = 0.030_000_000_000_000_002;
        assert_eq!(t.predict(&x).to_bits(), 2.0f64.to_bits());
        x[6] = f64::NAN;
        assert_eq!(t.predict(&x).to_bits(), (-1.0f64).to_bits());
    }

    #[test]
    fn the_probability_is_the_logistic_of_baseline_plus_leaves() {
        let m = TreeModel {
            baseline: 0.5f64.to_bits(),
            trees: vec![leaf(0.25), leaf(-1.0)],
        };
        let raw: f64 = 0.5 + 0.25 - 1.0;
        assert!((m.probability(&[0.0; FEATURES]) - 1.0 / (1.0 + (-raw).exp())).abs() < 1e-15);
    }

    #[test]
    fn trees_with_children_out_of_order_are_refused() {
        let mut t = stump(0, 0.0, 0.0, 0.0);
        t.left[0] = 0;
        assert!(t.validate("t").is_err());
        let mut t = stump(30, 0.0, 0.0, 0.0);
        assert!(t.validate("t").is_err());
        t.feature[0] = 1;
        assert!(t.validate("t").is_ok());
    }

    #[test]
    fn interleaving_skips_repeats_and_keeps_every_candidate() {
        let out = interleave(&[2, 0, 1, 3], &[0, 1, 2, 3]);
        assert_eq!(out, vec![2, 0, 1, 3]);
        let out = interleave(&[3, 2, 1, 0], &[0, 1, 2, 3]);
        assert_eq!(out, vec![3, 0, 2, 1]);
    }

    #[test]
    fn interleaving_bounds_the_first_hit_by_twice_the_better_order() {
        // exhaustive over every pair of orders of 5 candidates and every feasible set
        let perms = permutations(5);
        for feasible in 1u32..32 {
            let ok = |i: usize| feasible & (1 << i) != 0;
            let first_hit = |o: &[usize]| o.iter().position(|&i| ok(i)).unwrap() + 1;
            for a in &perms {
                for b in perms.iter().step_by(7) {
                    let h = interleave(a, b);
                    assert_eq!(h.len(), 5);
                    assert!(first_hit(&h) <= 2 * first_hit(a).min(first_hit(b)));
                }
            }
        }
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for p in permutations(n - 1) {
            for k in 0..=p.len() {
                let mut q = p.clone();
                q.insert(k, n - 1);
                out.push(q);
            }
        }
        out
    }

    #[test]
    fn the_gate_stays_first_in_every_mode() {
        let models = LearnedModels {
            format: "axisforge-learned-order".into(),
            version: 1,
            features: FEATURE_NAMES.iter().map(|s| (*s).to_owned()).collect(),
            feasibility: TreeModel {
                baseline: 0f64.to_bits(),
                trees: vec![leaf(0.0)],
            },
            robustness: Some(TreeModel {
                baseline: 0f64.to_bits(),
                trees: vec![leaf(0.0)],
            }),
            alpha: 0.8,
            beta: 0.3,
            q_floor: 1e-3,
            finger_depth: 0.045,
        };
        models.validate().unwrap();
        let pre = [0.9, 0.1, 0.5, 0.7];
        let mk = [false, true, false, true];
        let co = [0.0, 0.01, 0.02, 0.03];
        let scores = LearnedScores {
            feasibility: vec![0.9, 0.2, 0.8, 0.3],
            robustness: Some(vec![0.5, 0.5, 0.5, 0.9]),
        };
        for mode in [
            OrderMode::Feasibility,
            OrderMode::FeasibilityCentring,
            OrderMode::FeasibilityRobustness,
            OrderMode::Joint,
        ] {
            let o = order(&models, mode, &pre, &mk, &co, &scores);
            assert!(
                mk[o[0]] && mk[o[1]] && !mk[o[2]] && !mk[o[3]],
                "{mode:?}: {o:?}"
            );
        }
    }
}
