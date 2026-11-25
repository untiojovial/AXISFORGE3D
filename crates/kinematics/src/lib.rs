//! Kinematics for AXISFORGE 3D: forward kinematics, Jacobians and IK.
//!
//! All robotics mathematics lives here and nowhere else; the user interface
//! asks this crate for answers rather than recomputing them. Joint-value
//! vectors are in the model's configuration order (depth-first tree order),
//! SI units throughout.
//!
//! Results are engineering approximations, not a functional-safety
//! certification; see [`axisforge_domain::SAFETY_NOTICE`].
#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::return_self_not_must_use,
    clippy::similar_names
)]

pub mod cancel;
pub mod fk;
pub mod ik;
pub mod jacobian;
pub mod model;
pub mod orientation;
pub mod rng;

pub use cancel::{CancellationToken, Completion};
pub use fk::{forward_kinematics, forward_kinematics_unchecked, frame_pose, tool_pose, Pose};
pub use ik::{IkOptions, IkSolution};
pub use jacobian::{conditioning, jacobian, ConditioningReport, Jacobian, TaskSpace};
pub use model::{FrameNode, JointNode, KinematicModel, LinkNode};
pub use orientation::PoseError;
pub use rng::Rng;
