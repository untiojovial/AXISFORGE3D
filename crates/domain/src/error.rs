//! Errors for violated domain invariants.

use thiserror::Error;

/// A violated domain invariant. The value cannot exist.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum DomainError {
    /// A `NaN` or infinite float reached a place that requires a real number.
    #[error("non-finite number in `{context}`")]
    NonFinite {
        /// Where the offending value was found.
        context: String,
    },

    /// A vector that must define a direction had (near) zero length.
    #[error("cannot normalize a zero-length vector in `{context}`")]
    ZeroLengthVector {
        /// Where the offending value was found.
        context: String,
    },

    /// A bounded interval was empty or inverted.
    #[error("invalid limits for `{subject}`: lower {lower} must be <= upper {upper}")]
    InvalidLimits {
        /// What the interval belongs to.
        subject: String,
        /// Lower bound, SI.
        lower: f64,
        /// Upper bound, SI.
        upper: f64,
    },

    /// A negative value was supplied where only non-negative ones are meaningful.
    #[error("`{context}` must be >= 0, got {value}")]
    NegativeValue {
        /// Where the offending value was found.
        context: String,
        /// The offending value.
        value: f64,
    },

    /// Two entities shared an identifier that must be unique.
    #[error("duplicate {kind} identifier `{id}`")]
    DuplicateId {
        /// Entity kind.
        kind: &'static str,
        /// The duplicated identifier.
        id: String,
    },

    /// Two entities shared a name that must be unique within its scope.
    #[error("duplicate {kind} name `{name}`")]
    DuplicateName {
        /// Entity kind.
        kind: &'static str,
        /// The duplicated name.
        name: String,
    },

    /// A reference pointed at an entity that does not exist.
    #[error("{kind} `{id}` referenced by `{referenced_by}` does not exist")]
    DanglingReference {
        /// Entity kind expected to exist.
        kind: &'static str,
        /// The missing identifier.
        id: String,
        /// The entity holding the dangling reference.
        referenced_by: String,
    },

    /// The kinematic graph was not a tree rooted at a single link.
    #[error("kinematic tree is invalid: {reason}")]
    InvalidTree {
        /// Human-readable explanation.
        reason: String,
    },

    /// An empty or whitespace-only name was supplied.
    #[error("`{kind}` name must not be empty")]
    EmptyName {
        /// Entity kind.
        kind: &'static str,
    },

    /// A manipulator axis count outside the supported range was requested.
    #[error(
        "unsupported manipulator axis count {count} (supported: {min}..={max}); \
         end-effector joints such as gripper jaws do not count toward this"
    )]
    UnsupportedAxisCount {
        /// Requested count.
        count: usize,
        /// Minimum supported.
        min: usize,
        /// Maximum supported.
        max: usize,
    },

    /// A joint value was outside its configured limits.
    #[error("value {value} for joint `{joint}` is outside limits [{lower}, {upper}]")]
    ValueOutOfLimits {
        /// Joint name.
        joint: String,
        /// Offending value, SI.
        value: f64,
        /// Lower bound, SI.
        lower: f64,
        /// Upper bound, SI.
        upper: f64,
    },

    /// The number of supplied joint values did not match the model.
    #[error("expected {expected} joint values, got {actual}")]
    JointCountMismatch {
        /// Expected count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },

    /// A request was well-formed but asks for something that cannot be done.
    #[error("{reason}")]
    Unsupported {
        /// What was asked and why it cannot be done.
        reason: String,
    },

    /// A gripper was required but the robot does not have one.
    #[error("robot `{robot}` has no gripper: {detail}")]
    NoGripper {
        /// Robot name.
        robot: String,
        /// What was missing.
        detail: String,
    },
}

impl DomainError {
    /// Builds a [`DomainError::DanglingReference`].
    pub fn dangling(
        kind: &'static str,
        id: impl Into<String>,
        referenced_by: impl Into<String>,
    ) -> Self {
        Self::DanglingReference {
            kind,
            id: id.into(),
            referenced_by: referenced_by.into(),
        }
    }

    /// Builds a [`DomainError::InvalidTree`].
    pub fn tree(reason: impl Into<String>) -> Self {
        Self::InvalidTree {
            reason: reason.into(),
        }
    }

    /// Builds a [`DomainError::Unsupported`].
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    /// Builds a [`DomainError::NoGripper`].
    pub fn no_gripper(robot: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::NoGripper {
            robot: robot.into(),
            detail: detail.into(),
        }
    }
}

/// Convenience alias for domain fallible operations.
pub type DomainResult<T> = Result<T, DomainError>;
