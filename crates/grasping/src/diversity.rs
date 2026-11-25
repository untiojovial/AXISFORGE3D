//! Geometry-only downsampling of a research catalogue, before any native attempt.
use crate::ResearchCandidate;

/// Select a nested sequence of native IDs by farthest-point sampling. Every
/// fifth slot requests an opening-incompatible control when available. The
/// other slots prefer the opening-compatible stratum. Both fall back to the
/// other stratum when exhausted. Original IDs remain valid for evaluation.
pub fn diverse_research_candidates(
    records: &[ResearchCandidate],
    limit: usize,
    max_opening: f64,
    jaw_clearance: f64,
) -> Vec<ResearchCandidate> {
    let count = records.len().min(limit);
    let scale = max_opening.max(1e-9);
    let mut nearest = vec![f64::INFINITY; records.len()];
    let mut chosen = vec![false; records.len()];
    let compatible: Vec<_> = records
        .iter()
        .map(|record| {
            let c = &record.candidate;
            c.hand.is_some() || c.width + 2.0 * jaw_clearance <= max_opening
        })
        .collect();
    let descriptors: Vec<_> = records
        .iter()
        .map(|record| {
            let c = &record.candidate;
            [
                c.center.x / scale,
                c.center.y / scale,
                c.center.z / scale,
                c.width / scale,
                c.approach.x,
                c.approach.y,
                c.approach.z,
                c.closing.x,
                c.closing.y,
                c.closing.z,
            ]
        })
        .collect();
    let mut ordered = Vec::with_capacity(count);
    for slot in 0..count {
        let preferred = slot % 5 != 4;
        // A tie keeps the first native index, including at initialization.
        let mut best: Option<usize> = None;
        for i in 0..records.len() {
            if chosen[i] {
                continue;
            }
            if best.is_none_or(|j| {
                let want_i = compatible[i] == preferred;
                let want_j = compatible[j] == preferred;
                (want_i && !want_j) || (want_i == want_j && nearest[i] > nearest[j])
            }) {
                best = Some(i);
            }
        }
        let index = best.expect("slot count cannot exceed input length");
        chosen[index] = true;
        ordered.push(records[index].clone());
        let selected = descriptors[index];
        for (i, d) in descriptors.iter().enumerate() {
            let distance = d
                .iter()
                .zip(selected)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            nearest[i] = nearest[i].min(distance);
        }
    }
    ordered
}
