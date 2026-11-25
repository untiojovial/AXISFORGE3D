//! The AXISFORGE 3D domain model.
//!
//! This crate holds what a robot *is*: links, joints, tool frames, grippers,
//! the world around them, and the rules that say whether a description is
//! coherent. It computes no kinematics, draws nothing, and knows nothing about
//! the user interface — those live in `axisforge-kinematics`, the renderer and
//! the application respectively. The dependency arrow points inward only, so
//! this crate pulls in neither Tauri, nor Three.js, nor any rendering or UI
//! library. See `docs/adr/ADR-001-layering.md`.
//!
//! # Units
//!
//! Everything stored and computed is SI: **metres, radians, seconds,
//! kilograms, newtons**. [`units`] exists to render and parse at the edges;
//! changing a project's display units never moves a stored value.
//!
//! # Identifiers
//!
//! Entities are referenced by string id, never by index into a vector. An index
//! changes meaning the moment a list is reordered or an element removed, which
//! would silently retarget a command the user has already issued; an id does
//! not. [`templates::stable_id`] mints ids deterministically, so regenerating a
//! project file produces no diff.
//!
//! # Safety notice
//!
//! AXISFORGE 3D is a design and visualization tool. Its kinematics, collision
//! and grasp results are approximations intended for engineering exploration.
//! **They are not a functional-safety certification and must not be used as the
//! sole basis for operating physical machinery.** Validate every motion on the
//! real cell, under the appropriate safety measures, before relying on it.
//!
//! ```
//! use axisforge_domain::templates;
//!
//! let project = templates::pick_and_place_cell()?;
//! assert_eq!(project.robot.axis_count(), 6);
//! // The gripper's two jaws are degrees of freedom, but not axes.
//! assert_eq!(project.robot.total_dof(), 8);
//! project.validate()?;
//! # Ok::<(), axisforge_domain::DomainError>(())
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::return_self_not_must_use,
    clippy::similar_names
)]

pub mod environment;
pub mod error;
pub mod geometry;
pub mod joint;
pub mod link;
pub mod math;
pub mod project;
pub mod robot;
pub mod templates;
pub mod trajectory;
pub mod units;

pub use environment::{
    BodyKind, Environment, IgnoredPair, IgnoredPairs, PhysicalProperties, WorldBody,
    GROUND_OWNER_ID,
};
pub use error::{DomainError, DomainResult};
pub use geometry::{GeometryInstance, GeometryRole, Material, Shape};
pub use joint::{Joint, JointKind, JointLimits, JointRole};
pub use link::{Inertial, Link, ToolFrame};
pub use math::{Quat, Transform, Vec3};
pub use project::{Project, ProjectMetadata, Resource};
pub use robot::{
    Actor, ActorJoint, Gripper, JawCoupling, Preshape, RobotSystem, MAX_AXES, MIN_AXES,
};
pub use trajectory::{Trajectory, Waypoint};
pub use units::{AngleUnit, LengthUnit, UnitPreferences};

/// Standard gravity, metres per second squared.
///
/// Used to turn a part's mass into the weight a grasp must hold.
pub const STANDARD_GRAVITY: f64 = 9.806_65;

/// The notice that must accompany any simulation or analysis result shown to a user.
///
/// Every surface that reports a kinematic, collision or grasp result displays
/// this. It is a constant rather than copy in each view so that the wording
/// cannot drift apart between the desktop app, the command line and reports.
pub const SAFETY_NOTICE: &str = "AXISFORGE 3D results are engineering approximations, not a \
     functional-safety certification. Validate all motion on the real cell before operating \
     machinery.";

/// Tolerance used when comparing poses and joint values for equality, SI units.
///
/// One micrometre and one microradian: finer than any manipulator this tool
/// targets can repeat, and coarse enough that accumulated floating-point error
/// over a six-link chain does not trip it.
pub const COMPARISON_TOLERANCE: f64 = 1.0e-6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_safety_notice_says_what_it_must() {
        assert!(SAFETY_NOTICE.contains("not a functional-safety certification"));
    }

    #[test]
    fn standard_gravity_turns_mass_into_weight() {
        let part = PhysicalProperties {
            mass: 2.0,
            friction_coefficient: 0.5,
        };
        assert!((part.weight() - 2.0 * STANDARD_GRAVITY).abs() < 1e-12);
    }
}
