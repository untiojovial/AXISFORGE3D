//! Collision-free motion for AXISFORGE 3D.
//!
//! - [`continuous`] proves a straight joint-space segment clear of
//!   collisions, rather than sampling it.
//! - [`rrt`] finds collision-free paths with a bidirectional RRT and
//!   shortens them with shortcuts, every segment proven clear.
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
pub mod rrt;

pub use continuous::{Closest, MotionChecker, SegmentVerdict, MAX_DEPTH};
pub use rrt::{plan_path, PathFailure, PathPlan, RrtOptions};
