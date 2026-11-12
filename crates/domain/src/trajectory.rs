//! Named configurations and the paths between them.

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};

/// A named joint configuration, in SI units, in the robot's canonical order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Waypoint {
    /// Stable identifier.
    pub id: String,
    /// Display name, unique within the trajectory.
    pub name: String,
    /// One value per actuated joint, in [`crate::robot::RobotSystem::actuated_joints`] order.
    pub joint_values: Vec<f64>,
    /// How long to dwell here before moving on, seconds.
    #[serde(default)]
    pub dwell_seconds: f64,
}

impl Waypoint {
    /// Builds a waypoint with no dwell.
    pub fn new(id: impl Into<String>, name: impl Into<String>, joint_values: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            joint_values,
            dwell_seconds: 0.0,
        }
    }

    /// Validates the waypoint against an expected joint count.
    ///
    /// # Errors
    /// [`DomainError::JointCountMismatch`] when the width is wrong, or a value
    /// error for non-finite entries.
    pub fn validate(&self, expected_joints: usize) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "waypoint id",
            });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "waypoint name",
            });
        }
        if self.joint_values.len() != expected_joints {
            return Err(DomainError::JointCountMismatch {
                expected: expected_joints,
                actual: self.joint_values.len(),
            });
        }
        for (index, value) in self.joint_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("waypoint '{}' joint {index}", self.name),
                });
            }
        }
        if !self.dwell_seconds.is_finite() {
            return Err(DomainError::NonFinite {
                context: format!("waypoint '{}' dwellSeconds", self.name),
            });
        }
        if self.dwell_seconds < 0.0 {
            return Err(DomainError::NegativeValue {
                context: format!("waypoint '{}' dwellSeconds", self.name),
                value: self.dwell_seconds,
            });
        }
        Ok(())
    }
}

/// An ordered list of waypoints the robot visits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Trajectory {
    /// Stable identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// The waypoints, in visiting order.
    #[serde(default)]
    pub waypoints: Vec<Waypoint>,
    /// Fraction of each joint's maximum velocity to use, in `(0, 1]`.
    #[serde(default = "default_speed_scale")]
    pub speed_scale: f64,
}

fn default_speed_scale() -> f64 {
    0.25
}

impl Trajectory {
    /// Builds an empty trajectory at the default speed.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            waypoints: Vec::new(),
            speed_scale: default_speed_scale(),
        }
    }

    /// Looks up a waypoint by identifier.
    pub fn waypoint(&self, id: &str) -> Option<&Waypoint> {
        self.waypoints.iter().find(|w| w.id == id)
    }

    /// Validates the trajectory against an expected joint count.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self, expected_joints: usize) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "trajectory id",
            });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "trajectory name",
            });
        }
        if !self.speed_scale.is_finite() {
            return Err(DomainError::NonFinite {
                context: format!("trajectory '{}' speedScale", self.name),
            });
        }
        if !(0.0..=1.0).contains(&self.speed_scale) || self.speed_scale <= 0.0 {
            return Err(DomainError::NegativeValue {
                context: format!(
                    "trajectory '{}' speedScale must be within (0, 1]",
                    self.name
                ),
                value: self.speed_scale,
            });
        }
        let mut ids: Vec<&str> = Vec::with_capacity(self.waypoints.len());
        let mut names: Vec<&str> = Vec::with_capacity(self.waypoints.len());
        for waypoint in &self.waypoints {
            waypoint.validate(expected_joints)?;
            if ids.contains(&waypoint.id.as_str()) {
                return Err(DomainError::DuplicateId {
                    kind: "waypoint",
                    id: waypoint.id.clone(),
                });
            }
            if names.contains(&waypoint.name.as_str()) {
                return Err(DomainError::DuplicateName {
                    kind: "waypoint",
                    name: waypoint.name.clone(),
                });
            }
            ids.push(&waypoint.id);
            names.push(&waypoint.name);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_waypoint_of_the_wrong_width_is_rejected() {
        let waypoint = Waypoint::new("w", "home", vec![0.0, 0.0]);
        assert!(matches!(
            waypoint.validate(3),
            Err(DomainError::JointCountMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn zero_speed_scale_is_rejected() {
        let mut trajectory = Trajectory::new("t", "cycle");
        trajectory.speed_scale = 0.0;
        assert!(trajectory.validate(0).is_err());
    }

    #[test]
    fn speed_scale_defaults_when_absent_from_json() {
        let restored: Trajectory =
            serde_json::from_str(r#"{"id":"t","name":"cycle"}"#).expect("parses");
        assert!((restored.speed_scale - 0.25).abs() < 1e-12);
        assert!(restored.waypoints.is_empty());
    }
}
