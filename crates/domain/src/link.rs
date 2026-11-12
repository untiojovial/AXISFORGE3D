//! Links: the rigid bodies of a robot.

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};
use crate::geometry::GeometryInstance;
use crate::math::{Transform, Vec3};

/// Mass properties of a rigid body, expressed about its centre of mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Inertial {
    /// Mass, kilograms.
    pub mass: f64,
    /// Centre of mass in the link frame, metres.
    #[serde(default)]
    pub center_of_mass: Vec3,
    /// Diagonal of the inertia tensor about the centre of mass, kg*m^2.
    #[serde(default)]
    pub inertia_diagonal: Vec3,
}

impl Default for Inertial {
    fn default() -> Self {
        Self {
            mass: 0.0,
            center_of_mass: Vec3::ZERO,
            inertia_diagonal: Vec3::ZERO,
        }
    }
}

impl Inertial {
    /// Builds mass properties with a point mass at the origin.
    pub fn point_mass(mass: f64) -> Self {
        Self {
            mass,
            ..Self::default()
        }
    }

    /// Validates mass and inertia for finiteness and sign.
    ///
    /// # Errors
    /// [`DomainError::NonFinite`] or [`DomainError::NegativeValue`].
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        if !self.mass.is_finite() {
            return Err(DomainError::NonFinite {
                context: format!("{context}.mass"),
            });
        }
        if self.mass < 0.0 {
            return Err(DomainError::NegativeValue {
                context: format!("{context}.mass"),
                value: self.mass,
            });
        }
        self.center_of_mass
            .require_finite(&format!("{context}.centerOfMass"))?;
        self.inertia_diagonal
            .require_finite(&format!("{context}.inertiaDiagonal"))?;
        for (value, axis) in [
            (self.inertia_diagonal.x, "x"),
            (self.inertia_diagonal.y, "y"),
            (self.inertia_diagonal.z, "z"),
        ] {
            if value < 0.0 {
                return Err(DomainError::NegativeValue {
                    context: format!("{context}.inertiaDiagonal.{axis}"),
                    value,
                });
            }
        }
        Ok(())
    }
}

/// A rigid body in the robot's kinematic tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Link {
    /// Stable identifier.
    pub id: String,
    /// Display name, unique within the robot.
    pub name: String,
    /// Mass properties.
    #[serde(default)]
    pub inertial: Inertial,
    /// Shapes attached to this link.
    #[serde(default)]
    pub geometries: Vec<GeometryInstance>,
}

impl Link {
    /// Builds a link with no geometry and no mass.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            inertial: Inertial::default(),
            geometries: Vec::new(),
        }
    }

    /// Adds a geometry, builder style.
    pub fn with_geometry(mut self, geometry: GeometryInstance) -> Self {
        self.geometries.push(geometry);
        self
    }

    /// Sets the mass properties, builder style.
    pub fn with_inertial(mut self, inertial: Inertial) -> Self {
        self.inertial = inertial;
        self
    }

    /// Geometries that participate in collision checking.
    pub fn collision_geometries(&self) -> impl Iterator<Item = &GeometryInstance> {
        self.geometries.iter().filter(|g| g.is_collision())
    }

    /// Geometries that are drawn.
    pub fn visual_geometries(&self) -> impl Iterator<Item = &GeometryInstance> {
        self.geometries.iter().filter(|g| g.is_visual())
    }

    /// Looks up a geometry by identifier.
    pub fn geometry(&self, id: &str) -> Option<&GeometryInstance> {
        self.geometries.iter().find(|g| g.id == id)
    }

    /// Validates the link and every geometry on it.
    ///
    /// # Errors
    /// Propagates the first failing check, or [`DomainError::DuplicateId`] when
    /// two geometries share an identifier.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "link id" });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "link name" });
        }
        self.inertial.validate(&format!("link '{}'", self.name))?;
        let mut seen: Vec<&str> = Vec::with_capacity(self.geometries.len());
        for geometry in &self.geometries {
            geometry.validate(&format!(
                "link '{}' geometry '{}'",
                self.name, geometry.name
            ))?;
            if seen.contains(&geometry.id.as_str()) {
                return Err(DomainError::DuplicateId {
                    kind: "geometry",
                    id: geometry.id.clone(),
                });
            }
            seen.push(&geometry.id);
        }
        Ok(())
    }
}

/// A named frame rigidly attached to a link.
///
/// Tool frames are what the user commands: an IK target is a tool frame pose,
/// not a link pose. A gripper's grasp frame is a tool frame too, which is why
/// the same solver drives both a welding torch and a pair of jaws.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolFrame {
    /// Stable identifier.
    pub id: String,
    /// Display name, unique within the robot.
    pub name: String,
    /// Identifier of the link this frame is attached to.
    pub link_id: String,
    /// Pose of the frame in the link's frame.
    #[serde(default)]
    pub offset: Transform,
}

impl ToolFrame {
    /// Builds a tool frame.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        link_id: impl Into<String>,
        offset: Transform,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            link_id: link_id.into(),
            offset,
        }
    }

    /// Validates identifiers and the offset's rotation.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "tool frame id",
            });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "tool frame name",
            });
        }
        self.offset
            .normalized(&format!("tool frame '{}' offset", self.name))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Shape;

    #[test]
    fn duplicate_geometry_ids_are_rejected() {
        let link = Link::new("l", "base")
            .with_geometry(GeometryInstance::visual(
                "g",
                "a",
                Shape::Sphere { radius: 0.1 },
            ))
            .with_geometry(GeometryInstance::visual(
                "g",
                "b",
                Shape::Sphere { radius: 0.1 },
            ));
        assert!(matches!(
            link.validate(),
            Err(DomainError::DuplicateId { .. })
        ));
    }

    #[test]
    fn negative_mass_is_rejected() {
        let link = Link::new("l", "base").with_inertial(Inertial::point_mass(-1.0));
        assert!(matches!(
            link.validate(),
            Err(DomainError::NegativeValue { .. })
        ));
    }

    #[test]
    fn role_filters_split_visual_from_collision() {
        use crate::geometry::GeometryRole;
        let link = Link::new("l", "base")
            .with_geometry(GeometryInstance::visual(
                "a",
                "shell",
                Shape::Sphere { radius: 0.1 },
            ))
            .with_geometry(
                GeometryInstance::visual("b", "hull", Shape::Sphere { radius: 0.11 })
                    .with_role(GeometryRole::Collision),
            )
            .with_geometry(
                GeometryInstance::visual("c", "both", Shape::Sphere { radius: 0.12 })
                    .with_role(GeometryRole::Both),
            );
        assert_eq!(link.visual_geometries().count(), 2);
        assert_eq!(link.collision_geometries().count(), 2);
    }
}
