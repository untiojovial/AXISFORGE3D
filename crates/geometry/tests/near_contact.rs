//! GJK near contact, where it matters most: a part a hair above a large floor.
//!
//! Each case places a randomly tilted box a known gap above a 100 m slab and
//! checks three properties over gaps from 10 cm down to 0.1 micrometre:
//!
//! - the distance is never overestimated, which would invent clearance;
//! - every gap of a micrometre or more is measured conclusively, and at
//!   most a micrometre short;
//! - no case runs anywhere near the iteration limit.
//!
//! Below a micrometre, rounding at this scale (about 5 micrometres from the
//! slab's 70 m support points) exceeds the gap, so an inconclusive answer
//! there is correct; it is still never an overestimate.

use axisforge_domain::{Quat, Transform, Vec3};
use axisforge_geometry::{gjk, ConvexBody, ConvexCore, GjkStatus};

fn generator(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let value = (state >> 11) as f64 / (1_u64 << 53) as f64;
        value
    }
}

#[test]
fn a_box_a_hair_above_a_large_floor_is_measured_safely() {
    let mut rng = generator(7);
    let slab = ConvexBody::new(
        ConvexCore::Box {
            half_extents: Vec3::new(50.0, 50.0, 0.01),
        },
        Transform::from_translation(Vec3::new(0.0, 0.0, -0.01)),
    );
    let mut worst_iterations = 0;
    let mut worst_shortfall = 0.0_f64;
    for case in 0..600 {
        let axis = Vec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5);
        let tilt = Quat::from_axis_angle(&axis, rng() * 3.0).unwrap();
        let half = Vec3::new(
            0.005 + rng() * 0.05,
            0.005 + rng() * 0.05,
            0.005 + rng() * 0.05,
        );
        let mut lowest = f64::INFINITY;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    lowest = lowest.min(
                        tilt.rotate(&Vec3::new(sx * half.x, sy * half.y, sz * half.z))
                            .z,
                    );
                }
            }
        }
        let gap = 10_f64.powf(-1.0 - rng() * 6.0);
        let part = ConvexBody::new(
            ConvexCore::Box { half_extents: half },
            Transform::new(Vec3::new(rng() - 0.5, rng() - 0.5, gap - lowest), tilt),
        );
        let result = gjk::distance(&slab, &part);
        worst_iterations = worst_iterations.max(result.iterations);
        assert!(
            result.distance <= gap + 1.0e-12,
            "case {case}: gap {gap:e} reported as {:e}, an overestimate",
            result.distance
        );
        if gap >= 1.0e-6 {
            assert_eq!(
                result.status,
                GjkStatus::Separated,
                "case {case}: gap {gap:e}: {result:?}"
            );
            // At most a micrometre short: the bound's rounding at this scale.
            worst_shortfall = worst_shortfall.max(gap - result.distance);
            assert!(
                gap - result.distance <= 1.0e-6,
                "case {case}: gap {gap:e} reported as {:e}",
                result.distance
            );
        }
    }
    assert!(
        worst_iterations <= 16,
        "worst case took {worst_iterations} iterations"
    );
    println!("worst shortfall {worst_shortfall:e} m, worst {worst_iterations} iterations");
}
