//! Convex bodies described by their support function.
//!
//! A support function answers one question — "which point of this body is
//! furthest along direction `d`?" — and that single question is enough for GJK
//! to compute the distance between any two convex bodies. Writing each shape
//! this way means the distance code has no per-shape cases at all.
//!
//! Round shapes are handled by *inflation* rather than by their own support
//! case: a sphere is an inflated point, a capsule an inflated segment. Inflation
//! is exact, not an approximation — the support point of an inflated body is the
//! core's support point pushed `inflation` along `d` — and it keeps the
//! iteration running on a polytope, where convergence is finite.

use axisforge_domain::{DomainError, DomainResult, GeometryInstance, Shape, Transform, Vec3};

use crate::aabb::Aabb;

/// The polytope or exact-support core of a convex body, in its own frame.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvexCore {
    /// A single point at the origin. Inflated, this is a sphere.
    Point,
    /// A segment along local Z, from `-half_length` to `+half_length`.
    ///
    /// Inflated, this is a capsule.
    Segment {
        /// Half the segment's length, metres.
        half_length: f64,
    },
    /// A box centred on the origin.
    Box {
        /// Half-extents along X, Y, Z, metres.
        half_extents: Vec3,
    },
    /// A cylinder centred on the origin, axis along local Z.
    ///
    /// Its support function is exact, so a cylinder needs no inflation and
    /// keeps its sharp rim — inflating a disc would round the edges off.
    Cylinder {
        /// Radius, metres.
        radius: f64,
        /// Half the length along Z, metres.
        half_length: f64,
    },
    /// An explicit point set; the body is its convex hull.
    ///
    /// Mesh geometry enters here. The hull is what gets collided, so a
    /// non-convex mesh is treated as its hull — conservative for clearance
    /// checks, and stated rather than silently assumed.
    Hull {
        /// The points, in the body's own frame.
        points: Vec<Vec3>,
    },
}

impl ConvexCore {
    /// The core's support point along `direction`, in the core's own frame.
    ///
    /// `direction` need not be normalized. A zero direction yields the origin
    /// for every core, which keeps the function total.
    pub fn support(&self, direction: &Vec3) -> Vec3 {
        match self {
            Self::Point => Vec3::ZERO,
            Self::Segment { half_length } => {
                Vec3::new(0.0, 0.0, half_length * sign_or_zero(direction.z))
            }
            Self::Box { half_extents } => Vec3::new(
                half_extents.x * sign_or_zero(direction.x),
                half_extents.y * sign_or_zero(direction.y),
                half_extents.z * sign_or_zero(direction.z),
            ),
            Self::Cylinder {
                radius,
                half_length,
            } => {
                let radial = (direction.x * direction.x + direction.y * direction.y).sqrt();
                let (x, y) = if radial > f64::EPSILON {
                    (radius * direction.x / radial, radius * direction.y / radial)
                } else {
                    // Looking straight down the axis, every rim point is
                    // equally extreme; the centre of the cap is the stable
                    // choice and keeps the iteration deterministic.
                    (0.0, 0.0)
                };
                Vec3::new(x, y, half_length * sign_or_zero(direction.z))
            }
            Self::Hull { points } => {
                let mut best = Vec3::ZERO;
                let mut best_dot = f64::NEG_INFINITY;
                for point in points {
                    let dot = point.dot(direction);
                    if dot > best_dot {
                        best_dot = dot;
                        best = *point;
                    }
                }
                best
            }
        }
    }

    /// Half-extents of the smallest local box containing the core.
    pub fn local_half_extents(&self) -> Vec3 {
        match self {
            Self::Point => Vec3::ZERO,
            Self::Segment { half_length } => Vec3::new(0.0, 0.0, *half_length),
            Self::Box { half_extents } => *half_extents,
            Self::Cylinder {
                radius,
                half_length,
            } => Vec3::new(*radius, *radius, *half_length),
            Self::Hull { points } => {
                let mut half = Vec3::ZERO;
                for point in points {
                    half = Vec3::new(
                        half.x.max(point.x.abs()),
                        half.y.max(point.y.abs()),
                        half.z.max(point.z.abs()),
                    );
                }
                half
            }
        }
    }

    /// Validates the core's parameters.
    ///
    /// # Errors
    /// [`DomainError::NonFinite`] or [`DomainError::NegativeValue`].
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        let check = |value: f64, what: &str| -> DomainResult<()> {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("{context}.{what}"),
                });
            }
            if value < 0.0 {
                return Err(DomainError::NegativeValue {
                    context: format!("{context}.{what}"),
                    value,
                });
            }
            Ok(())
        };
        match self {
            Self::Point => Ok(()),
            Self::Segment { half_length } => check(*half_length, "halfLength"),
            Self::Box { half_extents } => {
                half_extents.require_finite(&format!("{context}.halfExtents"))?;
                check(half_extents.x, "halfExtents.x")?;
                check(half_extents.y, "halfExtents.y")?;
                check(half_extents.z, "halfExtents.z")
            }
            Self::Cylinder {
                radius,
                half_length,
            } => {
                check(*radius, "radius")?;
                check(*half_length, "halfLength")
            }
            Self::Hull { points } => {
                if points.is_empty() {
                    return Err(DomainError::ZeroLengthVector {
                        context: format!("{context}.points is empty"),
                    });
                }
                for point in points {
                    point.require_finite(&format!("{context}.points"))?;
                }
                Ok(())
            }
        }
    }
}

fn sign_or_zero(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// A convex body placed in the world: a core, an inflation radius and a pose.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexBody {
    /// The core shape in its own frame.
    pub core: ConvexCore,
    /// Radius added uniformly around the core, metres.
    pub inflation: f64,
    /// Pose of the core's frame in world coordinates.
    pub pose: Transform,
}

impl ConvexBody {
    /// Builds a body from a core at a pose, with no inflation.
    pub fn new(core: ConvexCore, pose: Transform) -> Self {
        Self {
            core,
            inflation: 0.0,
            pose,
        }
    }

    /// Builds an inflated body.
    pub fn inflated(core: ConvexCore, inflation: f64, pose: Transform) -> Self {
        Self {
            core,
            inflation,
            pose,
        }
    }

    /// A sphere: an inflated point.
    pub fn sphere(radius: f64, pose: Transform) -> Self {
        Self::inflated(ConvexCore::Point, radius, pose)
    }

    /// A capsule: an inflated segment along local Z.
    pub fn capsule(radius: f64, cylindrical_length: f64, pose: Transform) -> Self {
        Self::inflated(
            ConvexCore::Segment {
                half_length: cylindrical_length * 0.5,
            },
            radius,
            pose,
        )
    }

    /// The *core's* support point in world coordinates along a world direction.
    ///
    /// Inflation is deliberately excluded. GJK runs on the cores and subtracts
    /// the two inflation radii from the result once, at the end: pushing every
    /// support point outward during the iteration would round the Minkowski
    /// difference into a shape with no vertices, and the iteration would not
    /// terminate.
    pub fn core_support(&self, direction: &Vec3) -> Vec3 {
        let local_direction = self.pose.rotation.conjugate().rotate(direction);
        self.pose.apply(&self.core.support(&local_direction))
    }

    /// The smallest world-aligned box containing the body, inflation included.
    pub fn aabb(&self) -> Aabb {
        Aabb::from_local_half_extents(&self.pose, self.core.local_half_extents())
            .expanded(self.inflation)
    }

    /// The largest distance from the body's own origin to its surface.
    ///
    /// Used to scale tolerances to the size of what is being measured.
    pub fn bounding_radius(&self) -> f64 {
        self.core.local_half_extents().norm() + self.inflation
    }

    /// Validates the core, the inflation radius and the pose.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        self.core.validate(context)?;
        if !self.inflation.is_finite() {
            return Err(DomainError::NonFinite {
                context: format!("{context}.inflation"),
            });
        }
        if self.inflation < 0.0 {
            return Err(DomainError::NegativeValue {
                context: format!("{context}.inflation"),
                value: self.inflation,
            });
        }
        self.pose.normalized(&format!("{context}.pose"))?;
        Ok(())
    }
}

/// Turns a domain shape into a convex body at `pose`.
///
/// `mesh_vertices` resolves [`Shape::Mesh`]; pass the vertices of the
/// referenced resource, already flattened as `[x, y, z, ...]`.
///
/// # Errors
/// [`DomainError::DanglingReference`] when a mesh's vertices were not supplied,
/// or a value error from the shape's parameters.
pub fn body_from_shape(
    shape: &Shape,
    pose: Transform,
    mesh_vertices: Option<&[f64]>,
) -> DomainResult<ConvexBody> {
    let body = match shape {
        Shape::Box { size } => ConvexBody::new(
            ConvexCore::Box {
                half_extents: size.scaled(0.5),
            },
            pose,
        ),
        Shape::Sphere { radius } => ConvexBody::sphere(*radius, pose),
        Shape::Cylinder { radius, length } => ConvexBody::new(
            ConvexCore::Cylinder {
                radius: *radius,
                half_length: length * 0.5,
            },
            pose,
        ),
        Shape::Capsule { radius, length } => ConvexBody::capsule(*radius, *length, pose),
        Shape::Mesh { resource_id, scale } => {
            let vertices = mesh_vertices.ok_or_else(|| {
                DomainError::dangling("mesh resource", resource_id.clone(), "body_from_shape")
            })?;
            if vertices.len() < 3 {
                return Err(DomainError::ZeroLengthVector {
                    context: format!("mesh resource '{resource_id}' has no vertices"),
                });
            }
            let points: Vec<Vec3> = vertices
                .chunks_exact(3)
                .map(|c| Vec3::new(c[0] * scale.x, c[1] * scale.y, c[2] * scale.z))
                .collect();
            ConvexBody::new(ConvexCore::Hull { points }, pose)
        }
    };
    body.validate("shape")?;
    Ok(body)
}

/// Turns a posed geometry instance into a convex body in world coordinates.
///
/// `owner_pose` is the world pose of whatever the geometry is attached to — a
/// link's forward-kinematics pose, or a world body's pose.
///
/// # Errors
/// Propagates [`body_from_shape`].
pub fn body_from_geometry(
    geometry: &GeometryInstance,
    owner_pose: &Transform,
    mesh_vertices: Option<&[f64]>,
) -> DomainResult<ConvexBody> {
    body_from_shape(
        &geometry.shape,
        owner_pose.compose(&geometry.origin),
        mesh_vertices,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::Quat;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn a_box_support_picks_the_matching_corner() {
        let body = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
            },
            Transform::IDENTITY,
        );
        let point = body.core_support(&Vec3::new(1.0, -1.0, 1.0));
        assert!(
            point.distance(&Vec3::new(1.0, -2.0, 3.0)) < 1e-12,
            "{point:?}"
        );
    }

    #[test]
    fn support_follows_the_pose() {
        let pose = Transform::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::from_axis_angle(&Vec3::Z, FRAC_PI_2).unwrap(),
        );
        let body = ConvexBody::new(
            ConvexCore::Box {
                half_extents: Vec3::new(1.0, 0.1, 0.1),
            },
            pose,
        );
        // The box's long axis now points along world +Y.
        let point = body.core_support(&Vec3::Y);
        assert!((point.y - 1.0).abs() < 1e-12, "{point:?}");
    }

    #[test]
    fn a_cylinder_support_lands_on_the_rim() {
        let body = ConvexBody::new(
            ConvexCore::Cylinder {
                radius: 2.0,
                half_length: 0.5,
            },
            Transform::IDENTITY,
        );
        let point = body.core_support(&Vec3::new(3.0, 4.0, 1.0));
        assert!((point.x - 1.2).abs() < 1e-12, "{point:?}");
        assert!((point.y - 1.6).abs() < 1e-12, "{point:?}");
        assert!((point.z - 0.5).abs() < 1e-12, "{point:?}");
    }

    #[test]
    fn a_cylinder_viewed_down_its_axis_returns_the_cap_centre() {
        let body = ConvexBody::new(
            ConvexCore::Cylinder {
                radius: 2.0,
                half_length: 0.5,
            },
            Transform::IDENTITY,
        );
        let point = body.core_support(&Vec3::Z);
        assert!(
            point.distance(&Vec3::new(0.0, 0.0, 0.5)) < 1e-12,
            "{point:?}"
        );
    }

    #[test]
    fn a_zero_direction_is_total_not_a_panic() {
        for core in [
            ConvexCore::Point,
            ConvexCore::Segment { half_length: 1.0 },
            ConvexCore::Box {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            ConvexCore::Cylinder {
                radius: 1.0,
                half_length: 1.0,
            },
        ] {
            assert!(core.support(&Vec3::ZERO).norm() < 1e-12, "{core:?}");
        }
    }

    #[test]
    fn inflation_shows_up_in_the_bounds_but_not_in_core_support() {
        let body = ConvexBody::sphere(0.25, Transform::IDENTITY);
        assert!(
            body.core_support(&Vec3::X).norm() < 1e-12,
            "the core is a point"
        );
        let bounds = body.aabb();
        assert!((bounds.size().x - 0.5).abs() < 1e-12, "{bounds:?}");
    }

    #[test]
    fn a_capsule_is_an_inflated_segment() {
        let body = ConvexBody::capsule(0.1, 0.4, Transform::IDENTITY);
        assert_eq!(body.core, ConvexCore::Segment { half_length: 0.2 });
        // Total extent along Z is the cylindrical section plus both caps.
        assert!((body.aabb().size().z - 0.6).abs() < 1e-12);
    }

    #[test]
    fn a_negative_radius_is_rejected() {
        let body = ConvexBody::sphere(-1.0, Transform::IDENTITY);
        assert!(matches!(
            body.validate("s"),
            Err(DomainError::NegativeValue { .. })
        ));
    }

    #[test]
    fn an_unsupplied_mesh_is_a_dangling_reference_not_an_empty_body() {
        let shape = Shape::Mesh {
            resource_id: "r".to_owned(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(matches!(
            body_from_shape(&shape, Transform::IDENTITY, None),
            Err(DomainError::DanglingReference { .. })
        ));
    }

    #[test]
    fn a_mesh_becomes_its_scaled_hull() {
        let shape = Shape::Mesh {
            resource_id: "r".to_owned(),
            scale: Vec3::new(2.0, 1.0, 1.0),
        };
        let vertices = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let body = body_from_shape(&shape, Transform::IDENTITY, Some(&vertices)).unwrap();
        let point = body.core_support(&Vec3::X);
        assert!(
            (point.x - 2.0).abs() < 1e-12,
            "the scale must be applied: {point:?}"
        );
    }
}
