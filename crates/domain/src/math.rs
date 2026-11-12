//! Minimal, dependency-free linear algebra for the domain layer.
//!
//! The domain owns the serialized representation of poses and deliberately
//! depends on no linear-algebra crate, so the on-disk format never inherits
//! another library's internal layout. Conversion to `nalgebra` happens in
//! `axisforge-kinematics`.
//!
//! # Conventions
//!
//! Right-handed, Z up. Quaternions are Hamilton, stored `(x, y, z, w)`, unit
//! norm. A [`Transform`] maps **child coordinates into parent coordinates**,
//! and `a.compose(&b)` applies `b` first. Lengths are metres, angles radians.

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};

/// Tolerance for "is this effectively zero" tests on lengths.
pub const LENGTH_EPSILON: f64 = 1e-12;

/// Tolerance when checking that a quaternion is of unit norm.
pub const UNIT_NORM_EPSILON: f64 = 1e-9;

/// A 3D vector in metres, or a dimensionless direction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    /// Unit vector along +X.
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    /// Unit vector along +Y.
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    /// Unit vector along +Z.
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Builds a vector from components.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// `true` when every component is finite.
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Fails with [`DomainError::NonFinite`] when any component is `NaN`/inf.
    pub fn require_finite(&self, context: &str) -> DomainResult<()> {
        if self.is_finite() {
            Ok(())
        } else {
            Err(DomainError::NonFinite {
                context: context.to_owned(),
            })
        }
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Squared Euclidean norm.
    pub fn norm_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// The unit vector with the same direction.
    ///
    /// # Errors
    /// [`DomainError::ZeroLengthVector`] for a degenerate vector.
    pub fn normalized(&self, context: &str) -> DomainResult<Self> {
        self.require_finite(context)?;
        let n = self.norm();
        if n <= LENGTH_EPSILON {
            return Err(DomainError::ZeroLengthVector {
                context: context.to_owned(),
            });
        }
        Ok(Self::new(self.x / n, self.y / n, self.z / n))
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product `self × other`.
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Component-wise sum.
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Component-wise difference `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Uniform scaling.
    pub fn scaled(&self, k: f64) -> Self {
        Self::new(self.x * k, self.y * k, self.z * k)
    }

    /// Distance to another point.
    pub fn distance(&self, other: &Self) -> f64 {
        self.sub(other).norm()
    }

    /// Any unit vector perpendicular to this one.
    ///
    /// Used wherever a frame has to be completed from a single axis — a grasp
    /// approach direction, for instance, fixes only one axis of three.
    ///
    /// # Errors
    /// [`DomainError::ZeroLengthVector`] for a degenerate vector.
    pub fn any_perpendicular(&self, context: &str) -> DomainResult<Self> {
        let unit = self.normalized(context)?;
        // Cross with whichever basis vector is least parallel, so the result is
        // never near-degenerate.
        let seed = if unit.x.abs() < 0.9 { Self::X } else { Self::Y };
        unit.cross(&seed).normalized(context)
    }

    /// Component array, for FFI and serialization boundaries.
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// A Hamilton quaternion `(x, y, z, w)` representing a rotation.
///
/// `q` and `-q` encode the same rotation; compare with
/// [`Quat::approx_same_rotation`], never component-wise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    /// Vector part, X.
    pub x: f64,
    /// Vector part, Y.
    pub y: f64,
    /// Vector part, Z.
    pub z: f64,
    /// Scalar part.
    pub w: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// Builds a quaternion from raw components (not normalized).
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// `true` when every component is finite.
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Quaternion norm.
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// The unit quaternion with the same rotation.
    ///
    /// # Errors
    /// [`DomainError::ZeroLengthVector`] for a degenerate quaternion.
    pub fn normalized(&self, context: &str) -> DomainResult<Self> {
        if !self.is_finite() {
            return Err(DomainError::NonFinite {
                context: context.to_owned(),
            });
        }
        let n = self.norm();
        if n <= LENGTH_EPSILON {
            return Err(DomainError::ZeroLengthVector {
                context: context.to_owned(),
            });
        }
        Ok(Self::new(self.x / n, self.y / n, self.z / n, self.w / n))
    }

    /// `true` when the norm is within [`UNIT_NORM_EPSILON`] of 1.
    pub fn is_unit(&self) -> bool {
        self.is_finite() && (self.norm() - 1.0).abs() <= UNIT_NORM_EPSILON
    }

    /// Rotation of `angle` radians about `axis`, which is normalized first.
    ///
    /// # Errors
    /// Propagates [`Vec3::normalized`].
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> DomainResult<Self> {
        if !angle.is_finite() {
            return Err(DomainError::NonFinite {
                context: "angle".to_owned(),
            });
        }
        let a = axis.normalized("rotation axis")?;
        let (s, c) = (angle * 0.5).sin_cos();
        Ok(Self::new(a.x * s, a.y * s, a.z * s, c))
    }

    /// Hamilton product `self * other`: apply `other` first, then `self`.
    pub fn mul(&self, other: &Self) -> Self {
        Self::new(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )
    }

    /// Conjugate; the inverse for unit quaternions.
    pub fn conjugate(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// Rotates a vector.
    pub fn rotate(&self, v: &Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v).scaled(2.0);
        v.add(&t.scaled(self.w)).add(&qv.cross(&t))
    }

    /// `true` when both encode the same rotation, accounting for `q` and `-q`.
    pub fn approx_same_rotation(&self, other: &Self, epsilon: f64) -> bool {
        let dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        (dot.abs() - 1.0).abs() <= epsilon
    }

    /// The shortest rotation taking unit vector `from` onto unit vector `to`.
    ///
    /// # Errors
    /// Propagates [`Vec3::normalized`].
    pub fn from_rotation_arc(from: &Vec3, to: &Vec3) -> DomainResult<Self> {
        let a = from.normalized("rotation arc `from`")?;
        let b = to.normalized("rotation arc `to`")?;
        let dot = a.dot(&b);
        if dot >= 1.0 - 1e-12 {
            return Ok(Self::IDENTITY);
        }
        if dot <= -1.0 + 1e-12 {
            let axis = a.any_perpendicular("rotation arc fallback axis")?;
            return Ok(Self::new(axis.x, axis.y, axis.z, 0.0));
        }
        let axis = a.cross(&b);
        Self::new(axis.x, axis.y, axis.z, 1.0 + dot).normalized("rotation arc")
    }

    /// Builds a rotation from an orthonormal basis given column-wise.
    ///
    /// A grasp is specified by its axes — where the fingers close, where the
    /// gripper approaches from — so building a rotation from three directions
    /// is the natural constructor for one.
    ///
    /// # Errors
    /// [`DomainError::InvalidTree`] when the basis is not right-handed and
    /// orthonormal to within `1e-6`.
    pub fn from_basis(x_axis: &Vec3, y_axis: &Vec3, z_axis: &Vec3) -> DomainResult<Self> {
        let reject = |reason: &str| {
            Err(DomainError::InvalidTree {
                reason: format!("rotation basis is not usable: {reason}"),
            })
        };
        for (axis, name) in [(x_axis, "x"), (y_axis, "y"), (z_axis, "z")] {
            if (axis.norm() - 1.0).abs() > 1e-6 {
                return reject(&format!("the {name} axis is not a unit vector"));
            }
        }
        if x_axis.dot(y_axis).abs() > 1e-6
            || y_axis.dot(z_axis).abs() > 1e-6
            || z_axis.dot(x_axis).abs() > 1e-6
        {
            return reject("the axes are not mutually perpendicular");
        }
        if x_axis.cross(y_axis).dot(z_axis) < 0.0 {
            return reject("the basis is left-handed");
        }

        // Shepperd's method: pick the branch with the largest denominator so the
        // division never amplifies rounding.
        let (m00, m01, m02) = (x_axis.x, y_axis.x, z_axis.x);
        let (m10, m11, m12) = (x_axis.y, y_axis.y, z_axis.y);
        let (m20, m21, m22) = (x_axis.z, y_axis.z, z_axis.z);
        let trace = m00 + m11 + m22;

        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Self::new((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Self::new(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            Self::new((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            Self::new((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
        };
        q.normalized("basis rotation")
    }

    /// Intrinsic Z-Y-X Euler angles, returned as `(roll, pitch, yaw)` to match
    /// the URDF `rpy` ordering. Presentation only.
    pub fn to_euler_rpy(&self) -> (f64, f64, f64) {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        (roll, pitch, yaw)
    }

    /// Inverse of [`Quat::to_euler_rpy`].
    pub fn from_euler_rpy(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr, cr) = (roll * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();
        Self::new(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    }
}

/// A rigid pose: rotation then translation, child into parent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Transform {
    /// Origin of the child frame in the parent frame, metres.
    pub translation: Vec3,
    /// Orientation of the child frame in the parent frame.
    pub rotation: Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity pose.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Builds a pose.
    pub const fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// Pure translation.
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
        }
    }

    /// Pure rotation.
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
        }
    }

    /// Validates finiteness and re-normalizes the rotation.
    ///
    /// # Errors
    /// Propagates non-finite and degenerate-quaternion failures.
    pub fn normalized(&self, context: &str) -> DomainResult<Self> {
        self.translation.require_finite(context)?;
        Ok(Self::new(
            self.translation,
            self.rotation.normalized(context)?,
        ))
    }

    /// Applies this transform to a point in the child frame.
    pub fn apply(&self, point: &Vec3) -> Vec3 {
        self.rotation.rotate(point).add(&self.translation)
    }

    /// Rotates a direction, leaving the translation out.
    pub fn apply_direction(&self, direction: &Vec3) -> Vec3 {
        self.rotation.rotate(direction)
    }

    /// Composition `self ∘ other`: `other` is applied first.
    pub fn compose(&self, other: &Self) -> Self {
        Self::new(
            self.rotation
                .rotate(&other.translation)
                .add(&self.translation),
            self.rotation.mul(&other.rotation),
        )
    }

    /// The inverse pose. Assumes a unit rotation.
    pub fn inverse(&self) -> Self {
        let inv = self.rotation.conjugate();
        Self::new(inv.rotate(&self.translation).scaled(-1.0), inv)
    }

    /// The pose translated along one of its own axes.
    ///
    /// A grasp approach retreats along the gripper's approach axis, which is a
    /// motion in the frame's own coordinates rather than the world's.
    pub fn translated_locally(&self, offset: &Vec3) -> Self {
        Self::new(self.apply(offset), self.rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn normalizing_zero_vector_is_an_error() {
        assert!(matches!(
            Vec3::ZERO.normalized("axis"),
            Err(DomainError::ZeroLengthVector { .. })
        ));
    }

    #[test]
    fn quarter_turn_about_z_maps_x_to_y() {
        let q = Quat::from_axis_angle(&Vec3::Z, std::f64::consts::FRAC_PI_2).unwrap();
        let v = q.rotate(&Vec3::X);
        assert!(
            close(v.x, 0.0) && close(v.y, 1.0) && close(v.z, 0.0),
            "{v:?}"
        );
    }

    #[test]
    fn q_and_negative_q_are_the_same_rotation() {
        let q = Quat::from_axis_angle(&Vec3::new(0.2, 0.3, 0.9), 2.0).unwrap();
        let neg = Quat::new(-q.x, -q.y, -q.z, -q.w);
        assert!(q.approx_same_rotation(&neg, 1e-12));
    }

    #[test]
    fn transform_inverse_round_trips() {
        let t = Transform::new(
            Vec3::new(0.3, -1.2, 0.05),
            Quat::from_axis_angle(&Vec3::new(1.0, 2.0, -0.5), 1.1).unwrap(),
        );
        let p = Vec3::new(0.11, 0.22, 0.33);
        let back = t.inverse().apply(&t.apply(&p));
        assert!(close(back.x, p.x) && close(back.y, p.y) && close(back.z, p.z));
    }

    #[test]
    fn composition_applies_the_right_hand_side_first() {
        let rot = Transform::from_rotation(
            Quat::from_axis_angle(&Vec3::Z, std::f64::consts::FRAC_PI_2).unwrap(),
        );
        let trans = Transform::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let origin = rot.compose(&trans).apply(&Vec3::ZERO);
        assert!(close(origin.x, 0.0) && close(origin.y, 1.0), "{origin:?}");
    }

    #[test]
    fn a_perpendicular_is_always_perpendicular_and_unit() {
        for v in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-3.0, 0.2, 0.0),
        ] {
            let p = v.any_perpendicular("v").unwrap();
            assert!(close(p.norm(), 1.0), "{v:?} -> {p:?}");
            assert!(
                p.dot(&v.normalized("v").unwrap()).abs() < 1e-9,
                "{v:?} -> {p:?}"
            );
        }
    }

    #[test]
    fn a_basis_round_trips_through_a_quaternion() {
        // The case a grasp produces: an arbitrary closing direction completed
        // into a full frame.
        for seed in [
            Vec3::X,
            Vec3::new(0.3, -0.8, 0.5),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let z = seed.normalized("z").unwrap();
            let x = z.any_perpendicular("x").unwrap();
            let y = z.cross(&x);

            let q = Quat::from_basis(&x, &y, &z).unwrap();
            for (axis, expected) in [(Vec3::X, x), (Vec3::Y, y), (Vec3::Z, z)] {
                let got = q.rotate(&axis);
                assert!(
                    close(got.x, expected.x)
                        && close(got.y, expected.y)
                        && close(got.z, expected.z),
                    "{seed:?}: {got:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn a_left_handed_basis_is_refused() {
        assert!(Quat::from_basis(&Vec3::X, &Vec3::Z, &Vec3::Y).is_err());
    }

    #[test]
    fn a_non_orthogonal_basis_is_refused() {
        let skew = Vec3::new(1.0, 0.3, 0.0).normalized("s").unwrap();
        assert!(Quat::from_basis(&Vec3::X, &skew, &Vec3::Z).is_err());
    }

    #[test]
    fn local_translation_moves_along_the_frames_own_axes() {
        // A gripper retreating 10 cm along its approach axis after a quarter
        // turn moves in Y, not X.
        let pose = Transform::new(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::from_axis_angle(&Vec3::Z, std::f64::consts::FRAC_PI_2).unwrap(),
        );
        let retreated = pose.translated_locally(&Vec3::new(0.1, 0.0, 0.0));
        assert!(close(retreated.translation.x, 1.0), "{retreated:?}");
        assert!(close(retreated.translation.y, 0.1), "{retreated:?}");
    }

    #[test]
    fn euler_round_trips_through_quaternion() {
        for &(r, p, y) in &[(0.0, 0.0, 0.0), (0.3, -0.4, 1.2), (-1.0, 0.2, -2.5)] {
            let (r2, p2, y2) = Quat::from_euler_rpy(r, p, y).to_euler_rpy();
            assert!(close(r, r2) && close(p, p2) && close(y, y2));
        }
    }
}
