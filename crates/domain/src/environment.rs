//! The world around the robot: fixtures, obstacles and objects to pick up.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::{DomainError, DomainResult};
use crate::geometry::GeometryInstance;
use crate::math::{Transform, Vec3};

/// Owner identifier used for the ground plane in collision reports.
///
/// The ground is not a body in [`Environment::bodies`]; it is implied by
/// [`Environment::ground_height`]. Collision results still need something to
/// name as the other side of a contact, so they use this.
pub const GROUND_OWNER_ID: &str = "world::ground";

/// What a body in the world is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BodyKind {
    /// Static scenery: a table, a fence, a wall. Never grasped, always collided.
    #[default]
    Obstacle,
    /// A part the robot is meant to pick up.
    Manipulable,
    /// A fixture that holds parts: a jig, a bin, a conveyor stop.
    Fixture,
}

impl BodyKind {
    /// `true` when a grasp planner should consider this body a target.
    pub fn is_graspable(self) -> bool {
        matches!(self, Self::Manipulable)
    }
}

/// Physical properties a grasp depends on.
///
/// These are the only physics the domain carries, and they exist because a
/// grasp's feasibility is not geometric alone: the same pose on the same box
/// holds or slips depending on friction and weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalProperties {
    /// Mass, kilograms.
    pub mass: f64,
    /// Coulomb friction coefficient between this body and a gripper pad.
    ///
    /// Dimensionless. Rubber on cardboard is around 0.8; steel on steel
    /// around 0.2. The half-angle of the friction cone is `atan(mu)`.
    pub friction_coefficient: f64,
}

impl Default for PhysicalProperties {
    fn default() -> Self {
        // 0.5 kg and mu = 0.5 describe a small plastic part against a rubber
        // pad: enough to make force-closure arithmetic meaningful without
        // claiming to know a part the user has not described.
        Self {
            mass: 0.5,
            friction_coefficient: 0.5,
        }
    }
}

impl PhysicalProperties {
    /// The half-angle of the friction cone, radians.
    pub fn friction_cone_half_angle(&self) -> f64 {
        self.friction_coefficient.atan()
    }

    /// Weight under standard gravity, newtons.
    pub fn weight(&self) -> f64 {
        self.mass * crate::STANDARD_GRAVITY
    }

    /// Validates mass and friction.
    ///
    /// # Errors
    /// [`DomainError::NonFinite`] or [`DomainError::NegativeValue`].
    pub fn validate(&self, context: &str) -> DomainResult<()> {
        for (value, what) in [
            (self.mass, "mass"),
            (self.friction_coefficient, "frictionCoefficient"),
        ] {
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
        }
        Ok(())
    }
}

/// A rigid body in the world, independent of the robot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldBody {
    /// Stable identifier.
    pub id: String,
    /// Display name, unique within the environment.
    pub name: String,
    /// What the body is for.
    #[serde(default)]
    pub kind: BodyKind,
    /// Pose in world coordinates.
    #[serde(default)]
    pub pose: Transform,
    /// Shapes making up the body, posed relative to [`WorldBody::pose`].
    pub geometries: Vec<GeometryInstance>,
    /// Mass and friction. Only meaningful for a manipulable body.
    #[serde(default)]
    pub physical: PhysicalProperties,
}

impl WorldBody {
    /// Builds an obstacle from a single geometry.
    pub fn obstacle(
        id: impl Into<String>,
        name: impl Into<String>,
        pose: Transform,
        geometry: GeometryInstance,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind: BodyKind::Obstacle,
            pose,
            geometries: vec![geometry],
            physical: PhysicalProperties::default(),
        }
    }

    /// Builds a manipulable part from a single geometry.
    pub fn manipulable(
        id: impl Into<String>,
        name: impl Into<String>,
        pose: Transform,
        geometry: GeometryInstance,
        physical: PhysicalProperties,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind: BodyKind::Manipulable,
            pose,
            geometries: vec![geometry],
            physical,
        }
    }

    /// `true` when a grasp planner should consider this body a target.
    pub fn is_graspable(&self) -> bool {
        self.kind.is_graspable()
    }

    /// The body's sole geometry, when it has exactly one.
    ///
    /// Grasp synthesis works from a single primitive; a body assembled from
    /// several shapes has no single set of face normals to be antipodal about,
    /// so the planner refuses it rather than picking one shape arbitrarily.
    pub fn single_geometry(&self) -> Option<&GeometryInstance> {
        match self.geometries.as_slice() {
            [only] => Some(only),
            _ => None,
        }
    }

    /// Validates the body and its geometries.
    ///
    /// # Errors
    /// Propagates the first failing check.
    pub fn validate(&self) -> DomainResult<()> {
        if self.id.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "world body id",
            });
        }
        if self.name.trim().is_empty() {
            return Err(DomainError::EmptyName {
                kind: "world body name",
            });
        }
        if self.geometries.is_empty() {
            return Err(DomainError::EmptyName {
                kind: "world body geometry list",
            });
        }
        self.pose
            .normalized(&format!("body '{}' pose", self.name))?;
        self.physical.validate(&format!("body '{}'", self.name))?;
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for geometry in &self.geometries {
            geometry.validate(&format!(
                "body '{}' geometry '{}'",
                self.name, geometry.name
            ))?;
            if !seen.insert(&geometry.id) {
                return Err(DomainError::DuplicateId {
                    kind: "geometry",
                    id: geometry.id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// A pair of collision owners that is deliberately not checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnoredPair {
    /// One owner: a link id, a body id, or [`GROUND_OWNER_ID`].
    pub first: String,
    /// The other owner.
    pub second: String,
    /// Why the pair is ignored, shown in the UI so the exemption is auditable.
    #[serde(default)]
    pub reason: String,
}

/// The set of pairs excluded from collision checking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct IgnoredPairs {
    pairs: Vec<IgnoredPair>,
}

impl IgnoredPairs {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a pair, with a reason recorded for the user to review.
    pub fn ignore(&mut self, first: &str, second: &str, reason: impl Into<String>) {
        if self.contains(first, second) {
            return;
        }
        let (first, second) = Self::ordered(first, second);
        self.pairs.push(IgnoredPair {
            first,
            second,
            reason: reason.into(),
        });
    }

    /// `true` when the pair is excluded, in either order.
    pub fn contains(&self, first: &str, second: &str) -> bool {
        let (a, b) = Self::ordered(first, second);
        self.pairs.iter().any(|p| p.first == a && p.second == b)
    }

    /// The recorded pairs.
    pub fn pairs(&self) -> &[IgnoredPair] {
        &self.pairs
    }

    /// Removes a pair if present, returning whether it was there.
    pub fn allow(&mut self, first: &str, second: &str) -> bool {
        let (a, b) = Self::ordered(first, second);
        let before = self.pairs.len();
        self.pairs.retain(|p| !(p.first == a && p.second == b));
        self.pairs.len() != before
    }

    fn ordered(first: &str, second: &str) -> (String, String) {
        if first <= second {
            (first.to_owned(), second.to_owned())
        } else {
            (second.to_owned(), first.to_owned())
        }
    }
}

/// The world: a ground plane, bodies, and collision exemptions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Environment {
    /// Height of the ground plane along world Z, metres.
    pub ground_height: f64,
    /// Whether the ground plane participates in collision checking.
    pub ground_collides: bool,
    /// Bodies in the world.
    pub bodies: Vec<WorldBody>,
    /// Pairs excluded from collision checking.
    pub ignored_pairs: IgnoredPairs,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            ground_height: 0.0,
            ground_collides: true,
            bodies: Vec::new(),
            ignored_pairs: IgnoredPairs::new(),
        }
    }
}

impl Environment {
    /// Looks up a body by identifier.
    pub fn body(&self, id: &str) -> Option<&WorldBody> {
        self.bodies.iter().find(|b| b.id == id)
    }

    /// Looks up a body by display name.
    pub fn body_by_name(&self, name: &str) -> Option<&WorldBody> {
        self.bodies.iter().find(|b| b.name == name)
    }

    /// Bodies a grasp planner should consider targets.
    pub fn graspable_bodies(&self) -> impl Iterator<Item = &WorldBody> {
        self.bodies.iter().filter(|b| b.is_graspable())
    }

    /// Normal of the ground plane, pointing away from the floor.
    pub fn ground_normal(&self) -> Vec3 {
        Vec3::Z
    }

    /// Validates every body and the exemption list.
    ///
    /// # Errors
    /// Propagates the first failing check, or
    /// [`DomainError::DanglingReference`] for an exemption naming an unknown
    /// owner that is not a robot link.
    pub fn validate(&self, known_owners: &BTreeSet<&str>) -> DomainResult<()> {
        if !self.ground_height.is_finite() {
            return Err(DomainError::NonFinite {
                context: "environment.groundHeight".to_owned(),
            });
        }
        let mut ids: BTreeSet<&str> = BTreeSet::new();
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for body in &self.bodies {
            body.validate()?;
            if !ids.insert(&body.id) {
                return Err(DomainError::DuplicateId {
                    kind: "world body",
                    id: body.id.clone(),
                });
            }
            if !names.insert(&body.name) {
                return Err(DomainError::DuplicateName {
                    kind: "world body",
                    name: body.name.clone(),
                });
            }
        }
        for pair in self.ignored_pairs.pairs() {
            for owner in [&pair.first, &pair.second] {
                let known = owner == GROUND_OWNER_ID
                    || ids.contains(owner.as_str())
                    || known_owners.contains(owner.as_str());
                if !known {
                    return Err(DomainError::dangling(
                        "collision owner",
                        owner.clone(),
                        "environment.ignoredPairs",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Shape;

    #[test]
    fn ignored_pairs_are_order_independent() {
        let mut pairs = IgnoredPairs::new();
        pairs.ignore("b", "a", "adjacent links always touch");
        assert!(pairs.contains("a", "b"));
        assert!(pairs.contains("b", "a"));
        assert_eq!(pairs.pairs().len(), 1);
        pairs.ignore("a", "b", "duplicate");
        assert_eq!(
            pairs.pairs().len(),
            1,
            "a repeated pair should not be stored twice"
        );
        assert!(pairs.allow("b", "a"));
        assert!(!pairs.contains("a", "b"));
    }

    #[test]
    fn an_exemption_naming_an_unknown_owner_is_rejected() {
        let mut env = Environment::default();
        env.ignored_pairs.ignore("ghost", GROUND_OWNER_ID, "typo");
        let known = BTreeSet::new();
        assert!(matches!(
            env.validate(&known),
            Err(DomainError::DanglingReference { .. })
        ));
    }

    #[test]
    fn the_ground_is_always_a_known_owner() {
        let mut env = Environment::default();
        env.bodies.push(WorldBody::obstacle(
            "b",
            "table",
            Transform::IDENTITY,
            GeometryInstance::visual(
                "g",
                "slab",
                Shape::Box {
                    size: Vec3::new(1.0, 1.0, 0.1),
                },
            ),
        ));
        env.ignored_pairs
            .ignore("b", GROUND_OWNER_ID, "the table rests on the floor");
        env.validate(&BTreeSet::new()).expect("ground is implicit");
    }

    #[test]
    fn friction_cone_half_angle_matches_atan_mu() {
        let props = PhysicalProperties {
            mass: 1.0,
            friction_coefficient: 1.0,
        };
        assert!(
            (props.friction_cone_half_angle() - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
            "mu = 1 is a 45 degree cone"
        );
    }

    #[test]
    fn only_manipulable_bodies_are_graspable() {
        let mut env = Environment::default();
        env.bodies.push(WorldBody::obstacle(
            "o",
            "wall",
            Transform::IDENTITY,
            GeometryInstance::visual(
                "g",
                "slab",
                Shape::Box {
                    size: Vec3::new(1.0, 0.1, 1.0),
                },
            ),
        ));
        env.bodies.push(WorldBody::manipulable(
            "p",
            "part",
            Transform::IDENTITY,
            GeometryInstance::visual(
                "g",
                "cube",
                Shape::Box {
                    size: Vec3::new(0.05, 0.05, 0.05),
                },
            ),
            PhysicalProperties::default(),
        ));
        let names: Vec<&str> = env.graspable_bodies().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["part"]);
    }
}
