//! A small deterministic generator for inverse-kinematics restarts.
//!
//! Random restarts make IK far more reliable, but a solver whose answer depends
//! on a system entropy source cannot be tested, cannot be reproduced from a bug
//! report, and makes a project file's saved pose unreproducible. So the seed is
//! part of the request: the same seed gives the same restarts on every machine
//! and every target, this crate's WebAssembly build included.
//!
//! `SplitMix64` is used because it is a few lines, needs no state beyond a `u64`,
//! and passes the statistical tests that matter for sampling joint angles. It is
//! not cryptographic and must not be used where that is required.

/// A seeded `SplitMix64` generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Builds a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 raw bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// The next value in `[0, 1)`.
    pub fn next_unit(&mut self) -> f64 {
        // 53 bits is exactly the mantissa of an f64, so every value in the
        // range is representable and none is favoured by rounding.
        #[allow(clippy::cast_precision_loss)] // 53 bits fit an f64 mantissa exactly.
        let mantissa = (self.next_u64() >> 11) as f64;
        mantissa * (1.0 / 9_007_199_254_740_992.0)
    }

    /// The next value in `[lower, upper]`.
    pub fn next_in(&mut self, lower: f64, upper: f64) -> f64 {
        lower + self.next_unit() * (upper - lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let first: Vec<u64> = (0..8).map(|_| Rng::new(42).next_u64()).collect();
        let mut rng = Rng::new(42);
        let second: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
        assert_eq!(first[0], second[0]);
        let mut again = Rng::new(42);
        let third: Vec<u64> = (0..8).map(|_| again.next_u64()).collect();
        assert_eq!(second, third);
    }

    #[test]
    fn different_seeds_diverge() {
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn unit_values_stay_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let value = rng.next_unit();
            assert!((0.0..1.0).contains(&value), "{value}");
        }
    }

    #[test]
    fn ranged_values_respect_their_bounds() {
        let mut rng = Rng::new(99);
        for _ in 0..10_000 {
            let value = rng.next_in(-2.5, 1.5);
            assert!((-2.5..=1.5).contains(&value), "{value}");
        }
    }

    #[test]
    fn the_mean_of_many_unit_samples_is_near_a_half() {
        let mut rng = Rng::new(2024);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| rng.next_unit()).sum::<f64>() / f64::from(n);
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }
}
