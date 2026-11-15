//! Pose error: the six numbers that say how far a frame is from its target.
//!
//! Position error is a subtraction. Orientation error is not: quaternions,
//! Euler angles and matrices are all poor as *errors*, because the quantity a
//! solver needs is a vector it can add to a velocity. The rotation vector —
//! the axis of the residual rotation, scaled by its angle — is that vector. It
//! is the logarithm of the error quaternion, it lives in the same space as the
//! angular rows of the Jacobian, and its norm is the angle in radians, so a
//! tolerance on it means what a user expects.
//!
//! Euler angles are not used anywhere in this module. Near a gimbal
//! configuration their derivatives blow up, which would make a solver's step
//! size depend on where the tool happened to be pointing.

use axisforge_domain::{Quat, Transform, Vec3};

/// Below this rotation angle the small-angle form of the log map is used.
///
/// At a sixteenth of a microradian `sin(theta/2)` has lost enough precision
/// that dividing by it is worse than the series, and the series is exact to
/// double precision well above this point.
const SMALL_ANGLE: f64 = 1.0e-8;

/// The rotation vector of a quaternion: its axis scaled by its angle, radians.
///
/// The returned vector always represents the *shorter* of the two rotations a
/// quaternion could mean, so its norm never exceeds pi. Without that, a solver
/// handed `q` and `-q` — the same rotation — would be told to turn one way or
/// the long way round depending on a sign it cannot see.
pub fn rotation_vector(rotation: &Quat) -> Vec3 {
    let normalized = if rotation.w < 0.0 {
        Quat::new(-rotation.x, -rotation.y, -rotation.z, -rotation.w)
    } else {
        *rotation
    };
    let vector = Vec3::new(normalized.x, normalized.y, normalized.z);
    let sine_half = vector.norm();
    if sine_half < SMALL_ANGLE {
        // theta ~ 2 * sin(theta/2) for small theta, with no division.
        return vector.scaled(2.0);
    }
    let angle = 2.0 * sine_half.atan2(normalized.w.clamp(-1.0, 1.0));
    vector.scaled(angle / sine_half)
}

/// The quaternion whose rotation vector is `vector`.
///
/// The inverse of [`rotation_vector`], used to step an orientation by an
/// angular increment.
pub fn from_rotation_vector(vector: &Vec3) -> Quat {
    let angle = vector.norm();
    if angle < SMALL_ANGLE {
        // The first-order expansion, then normalized so the result is a
        // rotation even for a tiny-but-nonzero input.
        return Quat::new(vector.x * 0.5, vector.y * 0.5, vector.z * 0.5, 1.0)
            .normalized("from_rotation_vector")
            .unwrap_or(Quat::IDENTITY);
    }
    let axis = vector.scaled(1.0 / angle);
    Quat::from_axis_angle(&axis, angle).unwrap_or(Quat::IDENTITY)
}

/// How far one pose is from another, split into position and orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseError {
    /// Translation from the current position to the target, metres.
    pub linear: Vec3,
    /// Rotation vector taking the current orientation to the target, radians.
    pub angular: Vec3,
}

impl PoseError {
    /// The error between `current` and `target`, in world coordinates.
    pub fn between(current: &Transform, target: &Transform) -> Self {
        let residual = target.rotation.mul(&current.rotation.conjugate());
        Self {
            linear: target.translation.sub(&current.translation),
            angular: rotation_vector(&residual),
        }
    }

    /// Position error magnitude, metres.
    pub fn position_norm(&self) -> f64 {
        self.linear.norm()
    }

    /// Orientation error magnitude, radians.
    pub fn orientation_norm(&self) -> f64 {
        self.angular.norm()
    }

    /// `true` when both components are within their tolerances.
    pub fn within(&self, position_tolerance: f64, orientation_tolerance: f64) -> bool {
        self.position_norm() <= position_tolerance
            && self.orientation_norm() <= orientation_tolerance
    }

    /// The error as a six-vector: three linear rows then three angular rows.
    ///
    /// The row order matches [`crate::jacobian::Jacobian`], so the two can be
    /// multiplied without a reshuffle.
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.linear.x,
            self.linear.y,
            self.linear.z,
            self.angular.x,
            self.angular.y,
            self.angular.z,
        ]
    }

    /// A scalar combining both components, for comparing candidate solutions.
    ///
    /// `orientation_weight` converts radians into metres; it has to be chosen,
    /// not derived, because there is no physical conversion between them. The
    /// default elsewhere in this crate is 0.1 m/rad, which makes a tenth of a
    /// radian cost as much as a centimetre — about the ratio at which a user of
    /// a metre-scale arm stops caring which is worse.
    pub fn weighted_norm(&self, orientation_weight: f64) -> f64 {
        self.position_norm() + orientation_weight * self.orientation_norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn the_rotation_vector_of_the_identity_is_zero() {
        assert!(rotation_vector(&Quat::IDENTITY).norm() < 1e-15);
    }

    #[test]
    fn the_rotation_vector_recovers_the_axis_and_angle() {
        let axis = Vec3::new(1.0, 2.0, -3.0).normalized("t").unwrap();
        let q = Quat::from_axis_angle(&axis, 1.1).unwrap();
        let vector = rotation_vector(&q);
        assert!((vector.norm() - 1.1).abs() < 1e-12, "{vector:?}");
        assert!(
            vector.normalized("t").unwrap().distance(&axis) < 1e-12,
            "{vector:?}"
        );
    }

    #[test]
    fn a_quaternion_and_its_negation_give_the_same_error() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let q = Quat::from_axis_angle(&axis, 2.5).unwrap();
        let negated = Quat::new(-q.x, -q.y, -q.z, -q.w);
        let a = rotation_vector(&q);
        let b = rotation_vector(&negated);
        assert!(a.distance(&b) < 1e-12, "{a:?} vs {b:?}");
    }

    #[test]
    fn the_rotation_vector_never_takes_the_long_way_round() {
        // 350 degrees one way is 10 degrees the other.
        let q = Quat::from_axis_angle(&Vec3::Z, 350.0_f64.to_radians()).unwrap();
        let vector = rotation_vector(&q);
        assert!(
            (vector.norm() - 10.0_f64.to_radians()).abs() < 1e-12,
            "expected the short way, got {} rad",
            vector.norm()
        );
        assert!(vector.z < 0.0, "and in the opposite direction: {vector:?}");
    }

    #[test]
    fn the_log_and_exponential_maps_are_inverse() {
        for angle in [1.0e-12, 1.0e-9, 1.0e-4, 0.5, 1.5, 3.0] {
            let axis = Vec3::new(0.3, -0.5, 0.8).normalized("t").unwrap();
            let vector = axis.scaled(angle);
            let recovered = rotation_vector(&from_rotation_vector(&vector));
            assert!(
                recovered.distance(&vector) < 1e-11,
                "angle {angle}: {recovered:?} vs {vector:?}"
            );
        }
    }

    #[test]
    fn a_pi_rotation_is_representable() {
        let q = Quat::from_axis_angle(&Vec3::X, PI).unwrap();
        let vector = rotation_vector(&q);
        assert!((vector.norm() - PI).abs() < 1e-9, "{vector:?}");
    }

    #[test]
    fn pose_error_separates_position_from_orientation() {
        let current = Transform::IDENTITY;
        let target = Transform::new(
            Vec3::new(0.1, 0.0, 0.0),
            Quat::from_axis_angle(&Vec3::Z, FRAC_PI_2).unwrap(),
        );
        let error = PoseError::between(&current, &target);
        assert!((error.position_norm() - 0.1).abs() < 1e-12);
        assert!((error.orientation_norm() - FRAC_PI_2).abs() < 1e-12);
        assert!(!error.within(0.05, 1.0), "position is out of tolerance");
        assert!(!error.within(0.2, 0.1), "orientation is out of tolerance");
        assert!(error.within(0.2, 2.0));
    }

    #[test]
    fn the_six_vector_puts_linear_rows_first() {
        let error = PoseError {
            linear: Vec3::new(1.0, 2.0, 3.0),
            angular: Vec3::new(4.0, 5.0, 6.0),
        };
        for (got, expected) in error.to_array().iter().zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
            assert!((got - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn a_pose_has_no_error_against_itself() {
        let pose = Transform::new(
            Vec3::new(0.4, -0.2, 0.9),
            Quat::from_axis_angle(&Vec3::new(1.0, 1.0, 0.0), 0.8).unwrap(),
        );
        let error = PoseError::between(&pose, &pose);
        assert!(error.position_norm() < 1e-15);
        assert!(error.orientation_norm() < 1e-15);
    }
}
