//! The project aggregate: everything a saved file contains.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::environment::Environment;
use crate::error::{DomainError, DomainResult};
use crate::robot::RobotSystem;
use crate::trajectory::Trajectory;
use crate::units::UnitPreferences;

/// A mesh or other binary resource referenced by geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Resource {
    /// Stable identifier, referenced by [`crate::geometry::Shape::Mesh`].
    pub id: String,
    /// Display name.
    pub name: String,
    /// Media type, e.g. `model/stl`.
    pub media_type: String,
    /// Triangle vertex positions, flattened as `[x, y, z, ...]`, metres.
    ///
    /// Stored inline so a project file is self-contained: opening one on
    /// another machine never hunts for a mesh on a path that does not exist.
    #[serde(default)]
    pub vertices: Vec<f64>,
    /// Triangle indices into [`Resource::vertices`], three per face.
    #[serde(default)]
    pub indices: Vec<u32>,
}

impl Resource {
    /// Validates identifiers and index bounds.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "resource id",
            });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "resource name",
            });
        }
        if self.vertices.len() % 3 != 0 {
            return Err(DomainError::InvalidTree {
                reason: format!(
                    "resource '{}' has {} vertex coordinates, which is not a multiple of 3",
                    self.name,
                    self.vertices.len()
                ),
            });
        }
        if self.indices.len() % 3 != 0 {
            return Err(DomainError::InvalidTree {
                reason: format!(
                    "resource '{}' has {} indices, which is not a multiple of 3",
                    self.name,
                    self.indices.len()
                ),
            });
        }
        for value in &self.vertices {
            if !value.is_finite() {
                return Err(DomainError::NonFinite {
                    context: format!("resource '{}' vertices", self.name),
                });
            }
        }
        // A mesh with more than 4 billion vertices cannot be indexed by the
        // u32 index buffer at all, so the count itself is the fault to report.
        let vertex_count = u32::try_from(self.vertices.len() / 3).map_err(|_| {
            DomainError::tree(format!(
                "resource '{}' has {} vertices, more than a u32 index can address",
                self.name,
                self.vertices.len() / 3
            ))
        })?;
        if let Some(bad) = self.indices.iter().find(|index| **index >= vertex_count) {
            return Err(DomainError::dangling(
                "mesh vertex",
                bad.to_string(),
                format!("resource '{}' indices", self.name),
            ));
        }
        Ok(())
    }
}

/// Free-form notes and authorship, carried with the project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct ProjectMetadata {
    /// Who authored the project.
    pub author: String,
    /// What it is for.
    pub description: String,
    /// Free-form tags.
    pub tags: Vec<String>,
}

/// A complete project: one robot, its world, its trajectories and its settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    /// Stable identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Notes and authorship.
    #[serde(default)]
    pub metadata: ProjectMetadata,
    /// The robot.
    pub robot: RobotSystem,
    /// The world the robot works in.
    #[serde(default)]
    pub environment: Environment,
    /// Saved trajectories.
    #[serde(default)]
    pub trajectories: Vec<Trajectory>,
    /// Inline mesh resources.
    #[serde(default)]
    pub resources: Vec<Resource>,
    /// Display units. Presentation only.
    #[serde(default)]
    pub units: UnitPreferences,
}

impl Project {
    /// Builds a project around a robot, with an empty world.
    pub fn new(id: impl Into<String>, name: impl Into<String>, robot: RobotSystem) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            metadata: ProjectMetadata::default(),
            robot,
            environment: Environment::default(),
            trajectories: Vec::new(),
            resources: Vec::new(),
            units: UnitPreferences::default(),
        }
    }

    /// Looks up a resource by identifier.
    pub fn resource(&self, id: &str) -> Option<&Resource> {
        self.resources.iter().find(|r| r.id == id)
    }

    /// Looks up a trajectory by identifier.
    pub fn trajectory(&self, id: &str) -> Option<&Trajectory> {
        self.trajectories.iter().find(|t| t.id == id)
    }

    /// Validates the whole aggregate, including cross-references.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName { kind: "project id" });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "project name",
            });
        }
        self.robot.validate()?;

        let mut resource_ids: BTreeSet<&str> = BTreeSet::new();
        for resource in &self.resources {
            resource.validate()?;
            if !resource_ids.insert(&resource.id) {
                return Err(DomainError::DuplicateId {
                    kind: "resource",
                    id: resource.id.clone(),
                });
            }
        }

        // Mesh geometry anywhere in the project must point at a resource the
        // file actually carries, or the viewport would show a hole and the
        // collision checker would silently skip a body.
        let link_owners: BTreeSet<&str> = self.robot.links.iter().map(|l| l.id.as_str()).collect();
        for link in &self.robot.links {
            for geometry in &link.geometries {
                if let crate::geometry::Shape::Mesh { resource_id, .. } = &geometry.shape {
                    if !resource_ids.contains(resource_id.as_str()) {
                        return Err(DomainError::dangling(
                            "resource",
                            resource_id.clone(),
                            format!("link '{}' geometry '{}'", link.name, geometry.name),
                        ));
                    }
                }
            }
        }
        for body in &self.environment.bodies {
            for geometry in &body.geometries {
                if let crate::geometry::Shape::Mesh { resource_id, .. } = &geometry.shape {
                    if !resource_ids.contains(resource_id.as_str()) {
                        return Err(DomainError::dangling(
                            "resource",
                            resource_id.clone(),
                            format!("body '{}' geometry '{}'", body.name, geometry.name),
                        ));
                    }
                }
            }
        }

        self.environment.validate(&link_owners)?;

        let joint_count = self.robot.actuated_joints().len();
        let mut trajectory_ids: BTreeSet<&str> = BTreeSet::new();
        for trajectory in &self.trajectories {
            trajectory.validate(joint_count)?;
            if !trajectory_ids.insert(&trajectory.id) {
                return Err(DomainError::DuplicateId {
                    kind: "trajectory",
                    id: trajectory.id.clone(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{GeometryInstance, Shape};
    use crate::math::Vec3;
    use crate::templates;

    #[test]
    fn a_template_project_validates() {
        let robot = templates::planar_2r().expect("template builds");
        Project::new("p", "demo", robot).validate().expect("valid");
    }

    #[test]
    fn mesh_geometry_without_its_resource_is_rejected() {
        let mut robot = templates::planar_2r().expect("template builds");
        robot.links[0].geometries.push(GeometryInstance::visual(
            "g-mesh",
            "shell",
            Shape::Mesh {
                resource_id: "missing".to_owned(),
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
        ));
        let project = Project::new("p", "demo", robot);
        let error = project.validate().expect_err("dangling mesh");
        assert!(
            matches!(error, DomainError::DanglingReference { .. }),
            "{error}"
        );
        assert!(error.to_string().contains("missing"), "{error}");
    }

    #[test]
    fn a_trajectory_of_the_wrong_width_is_rejected() {
        let robot = templates::planar_2r().expect("template builds");
        let mut project = Project::new("p", "demo", robot);
        let mut trajectory = Trajectory::new("t", "cycle");
        trajectory
            .waypoints
            .push(crate::trajectory::Waypoint::new("w", "home", vec![0.0]));
        project.trajectories.push(trajectory);
        assert!(matches!(
            project.validate(),
            Err(DomainError::JointCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn an_out_of_range_mesh_index_is_rejected() {
        let resource = Resource {
            id: "r".to_owned(),
            name: "block".to_owned(),
            media_type: "model/stl".to_owned(),
            vertices: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 5],
        };
        assert!(matches!(
            resource.validate(),
            Err(DomainError::DanglingReference { .. })
        ));
    }
}
