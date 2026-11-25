//! Time along a path: synchronized trapezoidal profiles within every axis'
//! velocity and acceleration limits.
//!
//! Each straight joint-space segment is followed exactly: all axes move
//! along it together, `q(t) = a + s(t) (b − a)`, so the motion executed is
//! the one whose collisions were checked. The path parameter `s` runs from 0
//! to 1 with a trapezoidal rate: constant acceleration, cruise, constant
//! deceleration — a triangle when the segment is too short to reach the
//! cruise rate. Its peak rate and acceleration are the largest every axis
//! allows: `ṡ ≤ v_j / |Δq_j|` and `s̈ ≤ a_j / |Δq_j|` for each axis `j`. The
//! motion stops at the end of every segment.
//!
//! This is the motion of a joint move on most controllers (a `movej` with
//! no blending), whose speed and acceleration are given for the axis that
//! travels furthest; [`Move::leading_axis`] is that axis and
//! [`Move::leading_velocity`] and [`Move::leading_acceleration`] are the
//! values to give it.

use axisforge_domain::{JointKind, RobotSystem};
use axisforge_kinematics::KinematicModel;
use serde::{Deserialize, Serialize};

/// Speed assumed for a revolute joint that declares none, rad/s.
pub const ASSUMED_REVOLUTE_VELOCITY: f64 = 1.0;
/// Acceleration assumed for a revolute joint that declares none, rad/s².
pub const ASSUMED_REVOLUTE_ACCELERATION: f64 = 2.0;
/// Speed assumed for a prismatic joint that declares none, m/s.
pub const ASSUMED_PRISMATIC_VELOCITY: f64 = 0.1;
/// Acceleration assumed for a prismatic joint that declares none, m/s².
pub const ASSUMED_PRISMATIC_ACCELERATION: f64 = 0.5;

/// Velocity and acceleration limits of every actuated joint, in
/// configuration order, already scaled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AxisLimits {
    /// Largest speed of each joint, SI units per second.
    pub velocity: Vec<f64>,
    /// Largest acceleration of each joint, SI units per second squared.
    pub acceleration: Vec<f64>,
    /// Joints with a limit that was not declared and was assumed, each with
    /// which: `shoulder (acceleration)`.
    pub assumed: Vec<String>,
}

impl AxisLimits {
    /// The robot's declared limits times `speed_scale` (velocity) and
    /// `speed_scale²` (acceleration, so that a slower motion keeps the same
    /// shape), with the assumed values above where a joint declares none.
    pub fn of(robot: &RobotSystem, model: &KinematicModel, speed_scale: f64) -> Self {
        let scale = if speed_scale.is_finite() && speed_scale > 0.0 {
            speed_scale.min(1.0)
        } else {
            1.0
        };
        let mut velocity = Vec::with_capacity(model.dof());
        let mut acceleration = Vec::with_capacity(model.dof());
        let mut assumed = Vec::new();
        for slot in 0..model.dof() {
            let node = model.joint_for_value(slot);
            let joint = node.and_then(|n| robot.joint(&n.id));
            let revolute = node.is_none_or(|n| n.kind == JointKind::Revolute);
            let (v, a) = if revolute {
                (ASSUMED_REVOLUTE_VELOCITY, ASSUMED_REVOLUTE_ACCELERATION)
            } else {
                (ASSUMED_PRISMATIC_VELOCITY, ASSUMED_PRISMATIC_ACCELERATION)
            };
            let declared_v = joint.and_then(|j| j.max_velocity).filter(|v| *v > 0.0);
            let declared_a = joint.and_then(|j| j.max_acceleration).filter(|a| *a > 0.0);
            let missing = match (declared_v.is_none(), declared_a.is_none()) {
                (true, true) => Some("speed, acceleration"),
                (true, false) => Some("speed"),
                (false, true) => Some("acceleration"),
                (false, false) => None,
            };
            if let Some(missing) = missing {
                let name = node.map_or_else(|| format!("joint {slot}"), |n| n.name.clone());
                assumed.push(format!("{name} ({missing})"));
            }
            velocity.push(declared_v.unwrap_or(v) * scale);
            acceleration.push(declared_a.unwrap_or(a) * scale * scale);
        }
        Self {
            velocity,
            acceleration,
            assumed,
        }
    }
}

/// One straight joint-space move and its time profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Move {
    /// Configuration at the start, full width.
    pub from: Vec<f64>,
    /// Configuration at the end, full width.
    pub to: Vec<f64>,
    /// Duration, seconds.
    pub duration: f64,
    /// Time spent accelerating, and again decelerating, seconds.
    pub ramp: f64,
    /// Peak rate of the path parameter, 1/s.
    pub peak_rate: f64,
    /// Acceleration of the path parameter, 1/s².
    pub rate_acceleration: f64,
    /// Configuration slot of the axis that travels furthest.
    pub leading_axis: usize,
    /// That axis' peak speed, SI units per second.
    pub leading_velocity: f64,
    /// That axis' acceleration, SI units per second squared.
    pub leading_acceleration: f64,
}

impl Move {
    /// Profiles the straight move from `from` to `to` within `limits`.
    pub fn new(from: &[f64], to: &[f64], limits: &AxisLimits) -> Self {
        let mut rate = f64::INFINITY;
        let mut accel = f64::INFINITY;
        let mut leading = (0, 0.0);
        for (j, (a, b)) in from.iter().zip(to).enumerate() {
            let delta = (b - a).abs();
            if delta <= 0.0 {
                continue;
            }
            rate = rate.min(limits.velocity[j] / delta);
            accel = accel.min(limits.acceleration[j] / delta);
            if delta > leading.1 {
                leading = (j, delta);
            }
        }
        if !rate.is_finite() || !accel.is_finite() {
            return Self {
                from: from.to_vec(),
                to: to.to_vec(),
                duration: 0.0,
                ramp: 0.0,
                peak_rate: 0.0,
                rate_acceleration: 0.0,
                leading_axis: leading.0,
                leading_velocity: 0.0,
                leading_acceleration: 0.0,
            };
        }
        // Distance 1 at rate `rate` and acceleration `accel`.
        let (ramp, peak, duration) = if rate * rate / accel >= 1.0 {
            let ramp = (1.0 / accel).sqrt();
            (ramp, accel * ramp, 2.0 * ramp)
        } else {
            let ramp = rate / accel;
            (ramp, rate, 1.0 / rate + rate / accel)
        };
        Self {
            from: from.to_vec(),
            to: to.to_vec(),
            duration,
            ramp,
            peak_rate: peak,
            rate_acceleration: accel,
            leading_axis: leading.0,
            // The cruise speed the leading axis is commanded with; in a
            // triangular move it is never reached, as on the controller.
            leading_velocity: rate * leading.1,
            leading_acceleration: accel * leading.1,
        }
    }

    /// The path parameter, its rate and its acceleration at time `t`.
    pub fn parameter(&self, t: f64) -> (f64, f64, f64) {
        if self.duration <= 0.0 {
            return (1.0, 0.0, 0.0);
        }
        let t = t.clamp(0.0, self.duration);
        let (a, ramp, peak) = (self.rate_acceleration, self.ramp, self.peak_rate);
        if t < ramp {
            (0.5 * a * t * t, a * t, a)
        } else if t <= self.duration - ramp {
            (0.5 * a * ramp * ramp + peak * (t - ramp), peak, 0.0)
        } else {
            let left = self.duration - t;
            (1.0 - 0.5 * a * left * left, a * left, -a)
        }
    }

    /// Configuration at time `t`.
    pub fn position(&self, t: f64) -> Vec<f64> {
        let (s, _, _) = self.parameter(t);
        self.from
            .iter()
            .zip(&self.to)
            .map(|(a, b)| a + (b - a) * s)
            .collect()
    }

    /// Joint velocities at time `t`.
    pub fn velocity(&self, t: f64) -> Vec<f64> {
        let (_, rate, _) = self.parameter(t);
        self.from
            .iter()
            .zip(&self.to)
            .map(|(a, b)| (b - a) * rate)
            .collect()
    }

    /// `true` when only the joints in `slots` move.
    pub fn moves_only(&self, slots: &[usize]) -> bool {
        self.from
            .iter()
            .zip(&self.to)
            .enumerate()
            .all(|(j, (a, b))| a.to_bits() == b.to_bits() || slots.contains(&j))
    }
}

/// Profiles a path of straight segments, stopping at each waypoint.
pub fn profile_path(path: &[Vec<f64>], limits: &AxisLimits) -> Vec<Move> {
    path.windows(2)
        .map(|pair| Move::new(&pair[0], &pair[1], limits))
        .collect()
}

/// How long a controller's joint move takes, simulated in steps of `dt`:
/// the leading axis covers `distance` from rest to rest, accelerating at
/// `acceleration` up to `velocity` and braking just in time.
///
/// This re-derives a move's duration from the numbers a program gives the
/// controller, not from the profile, so comparing the two checks both.
pub fn simulate_move(distance: f64, velocity: f64, acceleration: f64, dt: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    if velocity <= 0.0 || acceleration <= 0.0 || dt <= 0.0 {
        return f64::INFINITY;
    }
    let (mut x, mut v, mut t) = (0.0_f64, 0.0_f64, 0.0_f64);
    // A bound on the steps, so a mistake cannot loop forever.
    let limit = 10.0 * (distance / velocity + velocity / acceleration) / dt + 1.0e6;
    let mut steps = 0.0;
    while steps < limit {
        let remaining = distance - x;
        if remaining <= 0.0 {
            break;
        }
        // The fastest speed that, after this cycle's step, still leaves room
        // to stop: v'²/2a + v'dt/2 ≤ remaining − v dt/2. This is what an
        // online trajectory generator commands each cycle.
        let room = (remaining - 0.5 * v * dt).max(0.0);
        let half = 0.5 * acceleration * dt;
        let stop = -half + (half * half + 2.0 * acceleration * room).sqrt();
        let next = (v + acceleration * dt)
            .min(velocity)
            .min(stop)
            .max(v - acceleration * dt)
            .max(0.0);
        let advance = 0.5 * (v + next) * dt;
        if advance >= remaining || advance <= 0.0 {
            t += if advance > 0.0 {
                dt * remaining / advance
            } else {
                dt
            };
            break;
        }
        x += advance;
        t += dt;
        v = next;
        steps += 1.0;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(n: usize) -> AxisLimits {
        AxisLimits {
            velocity: vec![1.0; n],
            acceleration: vec![2.0; n],
            assumed: Vec::new(),
        }
    }

    #[test]
    fn a_long_move_cruises_and_a_short_one_does_not() {
        let l = limits(2);
        // 2 rad at 1 rad/s and 2 rad/s²: 0.5 s up, 1.5 s cruise, 0.5 s down.
        let long = Move::new(&[0.0, 0.0], &[2.0, 0.5], &l);
        assert!((long.duration - 2.5).abs() < 1e-12, "{}", long.duration);
        assert!((long.ramp - 0.5).abs() < 1e-12);
        assert_eq!(long.leading_axis, 0);
        assert!((long.leading_velocity - 1.0).abs() < 1e-12);
        assert!((long.leading_acceleration - 2.0).abs() < 1e-12);
        // 0.2 rad never reaches 1 rad/s: a triangle of 2 * sqrt(0.2 / 2).
        let short = Move::new(&[0.0, 0.0], &[0.0, -0.2], &l);
        assert!((short.duration - 2.0 * 0.1_f64.sqrt()).abs() < 1e-12);
        assert_eq!(short.leading_axis, 1);
    }

    #[test]
    fn every_axis_stays_within_its_own_limits_and_ends_on_time() {
        let l = AxisLimits {
            velocity: vec![1.0, 0.25, 3.0],
            acceleration: vec![4.0, 0.5, 1.0],
            assumed: Vec::new(),
        };
        let m = Move::new(&[0.0, 0.0, 0.0], &[1.5, -0.4, 0.9], &l);
        let dt = 1.0e-4;
        let mut previous_v: Option<Vec<f64>> = None;
        let mut t = 0.0;
        while t <= m.duration {
            let v = m.velocity(t);
            for (j, value) in v.iter().enumerate() {
                assert!(
                    value.abs() <= l.velocity[j] * (1.0 + 1e-9),
                    "axis {j} at {t}"
                );
            }
            if let Some(p) = &previous_v {
                for (j, (a, b)) in p.iter().zip(&v).enumerate() {
                    assert!(((b - a) / dt).abs() <= l.acceleration[j] * (1.0 + 1e-6) + 1e-9);
                }
            }
            previous_v = Some(v);
            t += dt;
        }
        let end = m.position(m.duration);
        assert!(end
            .iter()
            .zip(&[1.5, -0.4, 0.9])
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert!(m.position(0.0).iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn a_controller_given_the_leading_axis_numbers_takes_the_same_time() {
        let l = AxisLimits {
            velocity: vec![1.0, 0.25, 3.0],
            acceleration: vec![4.0, 0.5, 1.0],
            assumed: Vec::new(),
        };
        for target in [[1.5, -0.4, 0.9], [0.01, 0.0, 0.0], [0.0, 2.0, -3.0]] {
            let m = Move::new(&[0.0, 0.0, 0.0], &target, &l);
            let distance = target[m.leading_axis].abs();
            let simulated =
                simulate_move(distance, m.leading_velocity, m.leading_acceleration, 1.0e-3);
            assert!(
                (simulated - m.duration).abs() <= 2.0e-3,
                "{simulated} vs {}",
                m.duration
            );
        }
    }

    #[test]
    fn declared_limits_are_used_and_missing_ones_assumed() {
        let robot = axisforge_domain::templates::six_axis_with_gripper().unwrap();
        let model = KinematicModel::from_robot(&robot).unwrap();
        let full = AxisLimits::of(&robot, &model, 1.0);
        assert!((full.velocity[0] - std::f64::consts::PI).abs() < 1e-12);
        assert!(full.assumed.is_empty());
        let half = AxisLimits::of(&robot, &model, 0.5);
        assert!((half.velocity[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((half.acceleration[0] - 0.25 * full.acceleration[0]).abs() < 1e-12);
        let mut bare = robot.clone();
        bare.joints.iter_mut().for_each(|j| j.max_velocity = None);
        let assumed = AxisLimits::of(&bare, &model, 1.0);
        assert_eq!(assumed.assumed.len(), 8);
        assert!((assumed.velocity[0] - ASSUMED_REVOLUTE_VELOCITY).abs() < 1e-12);
    }
}
