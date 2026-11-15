//! Geometry attached to links and to bodies in the world.
//!
//! A shape is a description, not a mesh: tessellation is the renderer's job and
//! distance queries are `axisforge-geometry`'s. Keeping the description here
//! means a project file stays small and exact, and a grasp planner can reason
//! about an object's actual dimensions rather than a triangle soup.

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};
use crate::math::{Transform, Vec3};

/// A parametric shape, in SI units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Shape {
    /// Axis-aligned box centred on its own origin.
    Box {
        /// Full extents along X, Y, Z, metres.
        size: Vec3,
    },
    /// Sphere centred on its own origin.
    Sphere {
        /// Radius, metres.
        radius: f64,
    },
    /// Cylinder centred on its own origin, axis along local +Z.
    Cylinder {
        /// Radius, metres.
        radius: f64,
        /// Total length along Z, metres.
        length: f64,
    },
    /// Capsule centred on its own origin, axis along local +Z.
    ///
    /// `length` is the cylindrical section only; total extent is
    /// `length + 2 * radius`.
    Capsule {
        /// Radius, metres.
        radius: f64,
        /// Cylindrical section length, metres.
        length: f64,
    },
    /// A mesh resource referenced by id, with a per-axis scale.
    Mesh {
        /// Identifier into [`crate::project::Project::resources`].
        resource_id: String,
        /// Dimensionless scale applied to the stored mesh.
        scale: Vec3,
    },
}

impl Shape {
    /// Validates the shape's parameters.
    ///
    /// # Errors
    /// [`DomainError::NonFinite`] or [`DomainError::NegativeValue`].
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        let non_negative = |value: f64, what: &str| -> DomainResult<()> {
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
            Self::Box { size } => {
                size.require_finite(&format!("{context}.size"))?;
                non_negative(size.x, "size.x")?;
                non_negative(size.y, "size.y")?;
                non_negative(size.z, "size.z")
            }
            Self::Sphere { radius } => non_negative(*radius, "radius"),
            Self::Cylinder { radius, length } | Self::Capsule { radius, length } => {
                non_negative(*radius, "radius")?;
                non_negative(*length, "length")
            }
            Self::Mesh { resource_id, scale } => {
                if resource_id.trim().is_empty() {
                    return Err(DomainError::EmptyName {
                        kind: "mesh resource id",
                    });
                }
                scale.require_finite(&format!("{context}.scale"))
            }
        }
    }

    /// Half-extents of a conservative box around the shape, in its own frame.
    ///
    /// Mesh shapes return `None`: their bounds need the vertex data, which the
    /// domain does not hold.
    pub fn local_half_extents(&self) -> Option<Vec3> {
        match self {
            Self::Box { size } => Some(size.scaled(0.5)),
            Self::Sphere { radius } => Some(Vec3::new(*radius, *radius, *radius)),
            Self::Cylinder { radius, length } => Some(Vec3::new(*radius, *radius, length * 0.5)),
            Self::Capsule { radius, length } => {
                Some(Vec3::new(*radius, *radius, length * 0.5 + radius))
            }
            Self::Mesh { .. } => None,
        }
    }

    /// The largest straight-line extent of the shape, metres.
    ///
    /// Used by diagnostics to spot a model whose units were never converted.
    pub fn largest_extent(&self) -> Option<f64> {
        self.local_half_extents()
            .map(|half| half.x.max(half.y).max(half.z) * 2.0)
    }
}

/// A simple non-physical material for the viewport.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Material {
    /// sRGB hex colour, `#rrggbb`.
    pub color: String,
    /// Opacity in `[0, 1]`.
    pub opacity: f64,
    /// Metalness in `[0, 1]`.
    pub metalness: f64,
    /// Roughness in `[0, 1]`.
    pub roughness: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: "#9aa7b4".to_owned(),
            opacity: 1.0,
            metalness: 0.1,
            roughness: 0.65,
        }
    }
}

impl Material {
    /// Validates colour syntax and the `[0, 1]` ranges.
    ///
    /// # Errors
    /// [`DomainError::EmptyName`] for a malformed colour, or a range failure.
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        let hex = self.color.strip_prefix('#').unwrap_or("");
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(DomainError::EmptyName {
                kind: "material color (expected #rrggbb)",
            });
        }
        for (value, what) in [
            (self.opacity, "opacity"),
            (self.metalness, "metalness"),
            (self.roughness, "roughness"),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("{context}.{what}"),
                });
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(DomainError::NegativeValue {
                    context: format!("{context}.{what} must be within [0, 1]"),
                    value,
                });
            }
        }
        Ok(())
    }
}

/// Whether a geometry is drawn, collided, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GeometryRole {
    /// Drawn, never collided.
    #[default]
    Visual,
    /// Collided, never drawn unless the overlay is on.
    Collision,
    /// Both.
    Both,
}

/// A shape placed relative to its owner's frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryInstance {
    /// Stable identifier, unique within the owner.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Pose of the shape in the owner's frame.
    #[serde(default)]
    pub origin: Transform,
    /// The shape itself.
    pub shape: Shape,
    /// Rendering material.
    #[serde(default)]
    pub material: Material,
    /// Role in the render and collision pipelines.
    #[serde(default)]
    pub role: GeometryRole,
}

impl GeometryInstance {
    /// Builds a visual geometry with the default material.
    pub fn visual(id: impl Into<String>, name: impl Into<String>, shape: Shape) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            origin: Transform::IDENTITY,
            shape,
            material: Material::default(),
            role: GeometryRole::Visual,
        }
    }

    /// Sets the pose, builder style.
    pub fn with_origin(mut self, origin: Transform) -> Self {
        self.origin = origin;
        self
    }

    /// Sets the material colour, builder style.
    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.material.color = color.into();
        self
    }

    /// Sets how metallic and how rough the surface looks, builder style.
    pub fn with_finish(mut self, metalness: f64, roughness: f64) -> Self {
        self.material.metalness = metalness;
        self.material.roughness = roughness;
        self
    }

    /// Sets the role, builder style.
    pub fn with_role(mut self, role: GeometryRole) -> Self {
        self.role = role;
        self
    }

    /// `true` when this geometry should be drawn.
    pub fn is_visual(&self) -> bool {
        matches!(self.role, GeometryRole::Visual | GeometryRole::Both)
    }

    /// `true` when this geometry should be collision-checked.
    pub fn is_collision(&self) -> bool {
        matches!(self.role, GeometryRole::Collision | GeometryRole::Both)
    }

    /// Validates identifiers, pose, shape and material.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "geometry id",
            });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "geometry name",
            });
        }
        self.origin.normalized(&format!("{context}.origin"))?;
        self.shape.validate(&format!("{context}.shape"))?;
        self.material.validate(&format!("{context}.material"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_radius_is_rejected() {
        assert!(matches!(
            Shape::Sphere { radius: -1.0 }.validate("s"),
            Err(DomainError::NegativeValue { .. })
        ));
    }

    #[test]
    fn capsule_half_extent_includes_both_caps() {
        let he = Shape::Capsule {
            radius: 0.1,
            length: 0.4,
        }
        .local_half_extents()
        .unwrap();
        assert!((he.z - 0.3).abs() < 1e-12);
    }

    #[test]
    fn largest_extent_reports_the_full_span() {
        assert!(
            (Shape::Box {
                size: Vec3::new(0.4, 0.2, 0.09)
            }
            .largest_extent()
            .unwrap()
                - 0.4)
                .abs()
                < 1e-12
        );
        assert!((Shape::Sphere { radius: 0.05 }.largest_extent().unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn a_mesh_has_no_known_extent() {
        let mesh = Shape::Mesh {
            resource_id: "m".into(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(mesh.largest_extent().is_none());
    }

    #[test]
    fn malformed_color_is_rejected() {
        let m = Material {
            color: "blue".to_owned(),
            ..Material::default()
        };
        assert!(m.validate("mat").is_err());
    }
}
