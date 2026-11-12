//! Joints: the degrees of freedom of a robot.

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};
use crate::math::{Quat, Transform, Vec3};

/// The kind of motion a joint allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    /// Rotation about [`Joint::axis`]. Value in radians.
    Revolute,
    /// Translation along [`Joint::axis`]. Value in metres.
    Prismatic,
    /// Rigid attachment. No degree of freedom.
    Fixed,
}

impl JointKind {
    /// `true` when the joint contributes a degree of freedom.
    pub fn is_actuated(self) -> bool {
        !matches!(self, Self::Fixed)
    }

    /// Suffix for the joint's value in SI units.
    pub fn si_suffix(self) -> &'static str {
        match self {
            Self::Revolute => "rad",
            Self::Prismatic => "m",
            Self::Fixed => "",
        }
    }
}

/// What a joint is *for*.
///
/// The 1..=6 axis budget of a manipulator counts only [`JointRole::Manipulator`]
/// joints. A parallel-jaw gripper adds two prismatic joints that are genuine
/// degrees of freedom — they move, they are drawn, they collide — but they
/// position fingers rather than the tool, so a six-axis arm with a gripper is
/// still a six-axis arm. Inverse kinematics holds end-effector joints fixed and
/// solves over the manipulator joints; the grasp planner does the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum JointRole {
    /// Positions the tool. Counts toward the axis budget; solved by IK.
    #[default]
    Manipulator,
    /// Actuates the end effector, e.g. a gripper jaw. Not an axis; not solved by IK.
    EndEffector,
}

/// A closed interval of allowed joint values, in SI units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointLimits {
    /// Lower bound, inclusive.
    pub lower: f64,
    /// Upper bound, inclusive.
    pub upper: f64,
}

impl JointLimits {
    /// Builds limits, checking that they are finite and ordered.
    ///
    /// # Errors
    /// [`DomainError::NonFinite`] or [`DomainError::InvalidLimits`].
    pub fn new(lower: f64, upper: f64, subject: &str) -> DomainResult<Self> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(DomainError::NonFinite {
                context: format!("{subject}.limits"),
            });
        }
        if lower > upper {
            return Err(DomainError::InvalidLimits {
                subject: subject.to_owned(),
                lower,
                upper,
            });
        }
        Ok(Self { lower, upper })
    }

    /// Symmetric limits `[-half, half]`.
    ///
    /// # Errors
    /// [`DomainError::NegativeValue`] when `half` is negative.
    pub fn symmetric(half: f64, subject: &str) -> DomainResult<Self> {
        if half < 0.0 {
            return Err(DomainError::NegativeValue {
                context: format!("{subject}.limits half-range"),
                value: half,
            });
        }
        Self::new(-half, half, subject)
    }

    /// Width of the interval.
    pub fn span(&self) -> f64 {
        self.upper - self.lower
    }

    /// Midpoint of the interval.
    pub fn center(&self) -> f64 {
        0.5 * (self.lower + self.upper)
    }

    /// `true` when `value` lies inside, allowing `tolerance` of slack.
    pub fn contains(&self, value: f64, tolerance: f64) -> bool {
        value >= self.lower - tolerance && value <= self.upper + tolerance
    }

    /// `value` clamped into the interval.
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.lower, self.upper)
    }

    /// Signed distance to the nearest bound; negative when outside.
    ///
    /// Zero means the value sits exactly on a bound.
    pub fn margin(&self, value: f64) -> f64 {
        (value - self.lower).min(self.upper - value)
    }

    /// Position within the interval as a fraction in `[0, 1]`.
    ///
    /// Returns `0.5` for a zero-width interval, where no fraction is meaningful.
    pub fn fraction(&self, value: f64) -> f64 {
        let span = self.span();
        if span <= 0.0 {
            return 0.5;
        }
        ((value - self.lower) / span).clamp(0.0, 1.0)
    }
}

/// A joint connecting a parent link to a child link.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Joint {
    /// Stable identifier.
    pub id: String,
    /// Display name, unique within the robot.
    pub name: String,
    /// The kind of motion.
    pub kind: JointKind,
    /// What the joint is for. Defaults to [`JointRole::Manipulator`].
    #[serde(default)]
    pub role: JointRole,
    /// Identifier of the parent link.
    pub parent_link_id: String,
    /// Identifier of the child link.
    pub child_link_id: String,
    /// Pose of the joint frame in the parent link's frame, at value zero.
    #[serde(default)]
    pub origin: Transform,
    /// Motion axis, a unit vector in the joint frame.
    pub axis: Vec3,
    /// Allowed range of the joint value, SI units.
    pub limits: JointLimits,
    /// Maximum magnitude of velocity, SI units per second. `None` means unset.
    #[serde(default)]
    pub max_velocity: Option<f64>,
    /// Maximum magnitude of acceleration, SI units per second squared.
    #[serde(default)]
    pub max_acceleration: Option<f64>,
    /// Maximum magnitude of effort: newton-metres for revolute, newtons for prismatic.
    #[serde(default)]
    pub max_effort: Option<f64>,
}

impl Joint {
    /// Builds a revolute joint with symmetric limits.
    ///
    /// # Errors
    /// Propagates axis and limit validation.
    pub fn revolute(
        id: impl Into<String>,
        name: impl Into<String>,
        parent_link_id: impl Into<String>,
        child_link_id: impl Into<String>,
        origin: Transform,
        axis: Vec3,
        half_range: f64,
    ) -> DomainResult<Self> {
        let name = name.into();
        let limits = JointLimits::symmetric(half_range, &name)?;
        Ok(Self {
            id: id.into(),
            name,
            kind: JointKind::Revolute,
            role: JointRole::Manipulator,
            parent_link_id: parent_link_id.into(),
            child_link_id: child_link_id.into(),
            origin,
            axis,
            limits,
            max_velocity: None,
            max_acceleration: None,
            max_effort: None,
        })
    }

    /// Builds a prismatic joint over `[lower, upper]` metres.
    ///
    /// # Errors
    /// Propagates axis and limit validation.
    #[allow(clippy::too_many_arguments)]
    pub fn prismatic(
        id: impl Into<String>,
        name: impl Into<String>,
        parent_link_id: impl Into<String>,
        child_link_id: impl Into<String>,
        origin: Transform,
        axis: Vec3,
        lower: f64,
        upper: f64,
    ) -> DomainResult<Self> {
        let name = name.into();
        let limits = JointLimits::new(lower, upper, &name)?;
        Ok(Self {
            id: id.into(),
            name,
            kind: JointKind::Prismatic,
            role: JointRole::Manipulator,
            parent_link_id: parent_link_id.into(),
            child_link_id: child_link_id.into(),
            origin,
            axis,
            limits,
            max_velocity: None,
            max_acceleration: None,
            max_effort: None,
        })
    }

    /// Builds a fixed joint. Its limits are the degenerate interval `[0, 0]`.
    pub fn fixed(
        id: impl Into<String>,
        name: impl Into<String>,
        parent_link_id: impl Into<String>,
        child_link_id: impl Into<String>,
        origin: Transform,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind: JointKind::Fixed,
            role: JointRole::Manipulator,
            parent_link_id: parent_link_id.into(),
            child_link_id: child_link_id.into(),
            origin,
            axis: Vec3::Z,
            limits: JointLimits {
                lower: 0.0,
                upper: 0.0,
            },
            max_velocity: None,
            max_acceleration: None,
            max_effort: None,
        }
    }

    /// Marks the joint as an end-effector actuator, builder style.
    pub fn as_end_effector(mut self) -> Self {
        self.role = JointRole::EndEffector;
        self
    }

    /// Sets the motion limits, builder style.
    pub fn with_dynamics(
        mut self,
        max_velocity: f64,
        max_acceleration: f64,
        max_effort: f64,
    ) -> Self {
        self.max_velocity = Some(max_velocity);
        self.max_acceleration = Some(max_acceleration);
        self.max_effort = Some(max_effort);
        self
    }

    /// `true` when this joint contributes a degree of freedom.
    pub fn is_actuated(&self) -> bool {
        self.kind.is_actuated()
    }

    /// `true` when this joint counts toward the manipulator's axis budget.
    pub fn is_manipulator_axis(&self) -> bool {
        self.is_actuated() && matches!(self.role, JointRole::Manipulator)
    }

    /// `true` when this joint actuates the end effector.
    pub fn is_end_effector(&self) -> bool {
        self.is_actuated() && matches!(self.role, JointRole::EndEffector)
    }

    /// The transform contributed by the joint at `value`, in the joint frame.
    ///
    /// The axis is normalized here, so a joint whose axis drifted from unit
    /// length still produces a rigid transform rather than a scaled one.
    ///
    /// # Errors
    /// [`DomainError::NonFinite`] for a non-finite value, or
    /// [`DomainError::ZeroLengthVector`] for a degenerate axis.
    pub fn motion(&self, value: f64) -> DomainResult<Transform> {
        if !value.is_finite() {
            return Err(DomainError::NonFinite {
                context: format!("joint '{}' value", self.name),
            });
        }
        let context = format!("joint '{}' axis", self.name);
        match self.kind {
            JointKind::Revolute => {
                let axis = self.axis.normalized(&context)?;
                Ok(Transform::from_rotation(Quat::from_axis_angle(
                    &axis, value,
                )?))
            }
            JointKind::Prismatic => {
                let axis = self.axis.normalized(&context)?;
                Ok(Transform::from_translation(axis.scaled(value)))
            }
            JointKind::Fixed => Ok(Transform::IDENTITY),
        }
    }

    /// The pose of the child link in the parent's frame at `value`.
    ///
    /// # Errors
    /// Propagates [`Joint::motion`].
    pub fn child_pose(&self, value: f64) -> DomainResult<Transform> {
        Ok(self.origin.compose(&self.motion(value)?))
    }

    /// The motion axis expressed in the parent link's frame, at any value.
    ///
    /// For a revolute joint this is the rotation axis; for a prismatic joint,
    /// the translation direction. Both are invariant in the parent frame
    /// because the joint rotates or slides about its own axis.
    ///
    /// # Errors
    /// [`DomainError::ZeroLengthVector`] for a degenerate axis.
    pub fn axis_in_parent(&self) -> DomainResult<Vec3> {
        let context = format!("joint '{}' axis", self.name);
        Ok(self
            .origin
            .apply_direction(&self.axis.normalized(&context)?))
    }

    /// Validates identifiers, topology references, axis, origin and dynamics.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "joint id" });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "joint name" });
        }
        if self.parent_link_id == self.child_link_id {
            return Err(DomainError::InvalidTree {
                reason: format!(
                    "joint '{}' connects link '{}' to itself",
                    self.name, self.parent_link_id
                ),
            });
        }
        self.origin
            .normalized(&format!("joint '{}' origin", self.name))?;
        if self.kind.is_actuated() {
            self.axis
                .normalized(&format!("joint '{}' axis", self.name))?;
        }
        JointLimits::new(self.limits.lower, self.limits.upper, &self.name)?;
        for (value, what) in [
            (self.max_velocity, "maxVelocity"),
            (self.max_acceleration, "maxAcceleration"),
            (self.max_effort, "maxEffort"),
        ] {
            if let Some(value) = value {
                if !value.is_finite() {
                    return Err(DomainError::NonFinite {
                        context: format!("joint '{}' {what}", self.name),
                    });
                }
                if value <= 0.0 {
                    return Err(DomainError::NegativeValue {
                        context: format!("joint '{}' {what} must be positive", self.name),
                        value,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn revolute_z() -> Joint {
        Joint::revolute(
            "j",
            "axis 1",
            "a",
            "b",
            Transform::from_translation(Vec3::new(0.0, 0.0, 0.3)),
            Vec3::Z,
            std::f64::consts::PI,
        )
        .expect("valid joint")
    }

    #[test]
    fn revolute_motion_rotates_about_its_axis() {
        let pose = revolute_z().motion(std::f64::consts::FRAC_PI_2).unwrap();
        let moved = pose.apply(&Vec3::X);
        assert!(moved.distance(&Vec3::Y) < 1e-12, "{moved:?}");
    }

    #[test]
    fn prismatic_motion_slides_along_its_axis() {
        let joint = Joint::prismatic(
            "j",
            "rail",
            "a",
            "b",
            Transform::IDENTITY,
            Vec3::X,
            0.0,
            0.2,
        )
        .unwrap();
        let pose = joint.motion(0.15).unwrap();
        assert!(pose.translation.distance(&Vec3::new(0.15, 0.0, 0.0)) < 1e-12);
    }

    #[test]
    fn a_non_unit_axis_still_yields_a_rigid_motion() {
        let mut joint = revolute_z();
        joint.axis = Vec3::new(0.0, 0.0, 4.0);
        let pose = joint.motion(std::f64::consts::FRAC_PI_2).unwrap();
        let moved = pose.apply(&Vec3::X);
        assert!(
            (moved.norm() - 1.0).abs() < 1e-12,
            "scaled, not rotated: {moved:?}"
        );
    }

    #[test]
    fn inverted_limits_are_rejected() {
        assert!(matches!(
            JointLimits::new(1.0, -1.0, "axis 1"),
            Err(DomainError::InvalidLimits { .. })
        ));
    }

    #[test]
    fn margin_is_negative_outside_the_interval() {
        let limits = JointLimits::new(-1.0, 1.0, "a").unwrap();
        assert!(limits.margin(1.25) < 0.0);
        assert!((limits.margin(0.25) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn fraction_of_a_degenerate_interval_is_the_middle() {
        let limits = JointLimits::new(0.0, 0.0, "a").unwrap();
        assert!((limits.fraction(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gripper_jaws_are_actuated_but_not_axes() {
        let jaw = Joint::prismatic(
            "j",
            "jaw",
            "palm",
            "finger",
            Transform::IDENTITY,
            Vec3::Y,
            0.0,
            0.04,
        )
        .unwrap()
        .as_end_effector();
        assert!(jaw.is_actuated());
        assert!(!jaw.is_manipulator_axis());
        assert!(jaw.is_end_effector());
    }

    #[test]
    fn a_joint_may_not_connect_a_link_to_itself() {
        let mut joint = revolute_z();
        joint.child_link_id = joint.parent_link_id.clone();
        assert!(matches!(
            joint.validate(),
            Err(DomainError::InvalidTree { .. })
        ));
    }

    #[test]
    fn axis_in_parent_follows_the_origin_rotation() {
        let mut joint = revolute_z();
        joint.origin = Transform::new(
            Vec3::ZERO,
            Quat::from_axis_angle(&Vec3::X, std::f64::consts::FRAC_PI_2).unwrap(),
        );
        let axis = joint.axis_in_parent().unwrap();
        assert!(
            axis.distance(&Vec3::new(0.0, -1.0, 0.0)) < 1e-12,
            "{axis:?}"
        );
    }

    #[test]
    fn role_defaults_to_manipulator_when_absent_from_json() {
        let json = serde_json::to_value(revolute_z()).unwrap();
        let mut map = json.as_object().unwrap().clone();
        map.remove("role");
        let restored: Joint = serde_json::from_value(serde_json::Value::Object(map)).unwrap();
        assert_eq!(restored.role, JointRole::Manipulator);
    }
}
