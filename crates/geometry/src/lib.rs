//! Convex geometry for AXISFORGE 3D: bounding volumes and distance queries.
//!
//! This crate turns the domain's parametric shapes into convex bodies with
//! support functions, bounds them with axis-aligned boxes for the broad phase,
//! and measures the distance between any two of them with GJK. It depends on
//! `axisforge-domain` and on nothing else: no rendering, no Tauri, no UI.
//!
//! It computes *distance*, not penetration depth — see [`gjk`] for why.
//!
//! ```
//! use axisforge_domain::{Transform, Vec3};
//! use axisforge_geometry::{gjk, ConvexBody, ConvexCore};
//!
//! let table = ConvexBody::new(
//!     ConvexCore::Box { half_extents: Vec3::new(0.25, 0.35, 0.015) },
//!     Transform::from_translation(Vec3::new(0.0, 0.0, 0.4)),
//! );
//! let part = ConvexBody::new(
//!     ConvexCore::Box { half_extents: Vec3::new(0.02, 0.02, 0.02) },
//!     Transform::from_translation(Vec3::new(0.0, 0.0, 0.5)),
//! );
//! let result = gjk::distance(&table, &part);
//! // Table top at 0.415, part bottom at 0.48.
//! assert!((result.distance - 0.065).abs() < 1e-9);
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

pub mod aabb;
pub mod gjk;
pub mod hull3;
pub mod support;

pub use aabb::Aabb;
pub use gjk::{Distance, GjkStatus};
pub use hull3::{convex_hull, Hull3};
pub use support::{body_from_geometry, body_from_shape, ConvexBody, ConvexCore};
