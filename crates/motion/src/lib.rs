//! Collision-free motion for AXISFORGE 3D.
//!
//! - [`continuous`] proves a straight joint-space segment clear of
//!   collisions, rather than sampling it.
//! - [`rrt`] finds collision-free paths with a bidirectional RRT and
//!   shortens them with shortcuts, every segment proven clear.
//! - [`profile`] times a path with synchronized trapezoidal profiles within
//!   every axis' velocity and acceleration limits.
//! - [`cycle`] plans the whole pick, transport and place cycle, every
//!   segment proven clear.
//! - [`export`] writes the cycle as a controller program (`URScript`, or
//!   sampled joint positions) and re-times the program independently.
//!
//! Results are engineering approximations, not a functional-safety
//! certification; see [`axisforge_domain::SAFETY_NOTICE`].

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

pub mod continuous;
pub mod cycle;
pub mod export;
pub mod profile;
pub mod rrt;

pub use continuous::{Closest, MotionChecker, SegmentVerdict, MAX_DEPTH};
pub use cycle::{
    plan_cycle, Cycle, CycleAttempt, CycleOptions, CycleReport, CycleSegment, CycleState,
    PartState, SegmentKind,
};
pub use export::{
    csv, program, sampled, simulate_program, urscript, ProgramStep, SampledTrajectory,
    TrajectoryPoint,
};
pub use profile::{profile_path, simulate_move, AxisLimits, Move};
pub use rrt::{plan_path, PathFailure, PathPlan, RrtOptions};
