//! Robotic grasping for AXISFORGE 3D: synthesis, quality and pick planning.
//!
//! Three layers, each usable on its own:
//!
//! - [`candidate`] enumerates the antipodal two-finger grasps of a box,
//!   cylinder, capsule or sphere from its dimensions; [`surface`] finds the
//!   grasps of any other part, a mesh included, by sampling its surface and
//!   closing the pads on it until they touch.
//! - [`quality`] scores a grasp for a gripper and a part: friction-cone force
//!   closure, stroke margin, required grip force, and how far off-centre it is.
//! - [`planner`] turns a part into executable picks — pre-grasp, grasp, close,
//!   lift — keeping only grasps the arm can reach and perform without
//!   collision, and reporting why each of the others was rejected.
//!
//! The crate adds no kinematics or collision code of its own: a gripper is
//! ordinary links and prismatic joints, so it reuses the engine unchanged.
//!
//! Plans are engineering proposals from a simplified contact model and sampled
//! collision checks, not a functional-safety certification; see
//! [`axisforge_domain::SAFETY_NOTICE`].
//!
//! ```
//! use axisforge_domain::templates;
//! use axisforge_grasping::{plan_pick, PlannerOptions};
//! use axisforge_kinematics::CancellationToken;
//!
//! let project = templates::pick_and_place_cell()?;
//! let cube = project.environment.body_by_name("cube").expect("the demo cube");
//! let home = [0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0];
//! let report = plan_pick(
//!     &project,
//!     &cube.id,
//!     &home,
//!     &PlannerOptions::default(),
//!     &CancellationToken::new(),
//! )?;
//! assert!(!report.plans.is_empty());
//! # Ok::<(), axisforge_domain::DomainError>(())
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::similar_names
)]

pub mod candidate;
pub mod hand;
pub mod hull;
pub mod planner;
pub mod quality;
pub mod registry;
pub mod surface;
pub mod wrench;

pub use candidate::{synthesize, GraspCandidate, SynthesisOptions};
pub use hand::{synthesize_hand, Hand, HandGrasp, HandSynthesis, HAND_GENERATOR};
pub use planner::{
    plan_pick, plan_pick_with, GraspPlan, PlanReport, PlannerOptions, Rejection, RejectionReason,
    AUTO_GENERATOR,
};
pub use quality::{
    evaluate, AntipodalMeasure, GraspQuality, Measurement, QualityInput, QualityMeasure,
    WrenchSpaceMeasure, GRIP_SAFETY_FACTOR,
};
pub use registry::{
    CandidateGenerator, ExtensionInfo, ExtensionKind, PrimitiveAntipodalGenerator, Registry,
    SurfaceNormalGenerator,
};
pub use surface::{synthesize_surface, Closure, PartSurface};
pub use wrench::{grasp_wrench_space, ContactPoint, WrenchOptions, WrenchSpace};
