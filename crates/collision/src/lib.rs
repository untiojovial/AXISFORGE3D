//! Collision scenes and clearance queries for AXISFORGE 3D.
//!
//! A [`Scene`] is a snapshot: every collision geometry of the robot at one
//! configuration, every body in the world, and the ground, each as a convex
//! body with a bounding box. [`Scene::check`] then measures the pairs that
//! matter — anything involving the robot or a part it is carrying — and
//! reports which are in contact and which pair is closest.
//!
//! Three kinds of pair are never measured:
//!
//! - two bodies of the same owner (a link does not collide with itself);
//! - links joined by a joint, which touch by construction;
//! - pairs the project explicitly exempts, each with a recorded reason.
//!
//! Static scenery is not checked against itself either: a part resting on a
//! table is the scene working as intended, not a fault.
//!
//! Distances come from GJK, so contact reports carry no penetration depth, and
//! mesh geometry is collided as its convex hull. Results are engineering
//! approximations, not a functional-safety certification; see
//! [`axisforge_domain::SAFETY_NOTICE`].

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

use std::collections::{BTreeMap, BTreeSet};

pub mod motion;
pub use motion::MotionBounds;

use axisforge_domain::{
    DomainError, DomainResult, GeometryInstance, Project, Shape, Transform, Vec3, GROUND_OWNER_ID,
};
use axisforge_geometry::{body_from_geometry, gjk, Aabb, ConvexBody, ConvexCore};
use axisforge_kinematics::{KinematicModel, Pose};
use serde::{Deserialize, Serialize};

/// Half-width of the slab standing in for the ground plane, metres.
const GROUND_HALF_EXTENT: f64 = 50.0;
/// Thickness of that slab, metres.
const GROUND_THICKNESS: f64 = 0.02;

/// What a collision body belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerKind {
    /// A link of the robot.
    Link,
    /// A body in the world that is not being carried.
    World,
    /// A world body rigidly attached to a robot link — a grasped part.
    Attached,
    /// The ground plane.
    Ground,
}

impl OwnerKind {
    /// `true` for owners that move with the robot.
    fn is_mobile(self) -> bool {
        matches!(self, Self::Link | Self::Attached)
    }
}

/// One convex body in a scene.
#[derive(Debug, Clone)]
pub struct SceneBody {
    /// Identifier of the owning link or world body, or [`GROUND_OWNER_ID`].
    pub owner_id: String,
    /// Display name of the owner.
    pub owner_name: String,
    /// What the owner is.
    pub kind: OwnerKind,
    /// The body in world coordinates.
    pub body: ConvexBody,
    /// Its bounding box.
    pub aabb: Aabb,
}

/// A world body carried by a robot link.
#[derive(Debug, Clone, PartialEq)]
pub struct Attachment {
    /// Identifier of the world body being carried.
    pub body_id: String,
    /// Identifier of the link carrying it.
    pub link_id: String,
    /// Pose of the body in the link's frame.
    pub relative_pose: Transform,
}

/// How a scene is assembled.
#[derive(Debug, Clone, Default)]
pub struct SceneOptions {
    /// A part the robot is carrying, if any.
    ///
    /// The part is posed from its link instead of its stored world pose, and
    /// it is never checked against the gripper holding it.
    pub attachment: Option<Attachment>,
    /// Extra owner pairs to leave unmeasured for this scene only.
    ///
    /// A grasp planner uses this to permit finger-to-part contact at the grasp
    /// pose, where touching is the point.
    pub extra_ignored: Vec<(String, String)>,
}

/// A snapshot of everything that can collide, at one configuration.
#[derive(Debug, Clone)]
pub struct Scene {
    bodies: Vec<SceneBody>,
    ignored: BTreeSet<(String, String)>,
}

/// The measured relation between two owners.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PairReport {
    /// Identifier of the first owner.
    pub first_id: String,
    /// Display name of the first owner.
    pub first_name: String,
    /// Identifier of the second owner.
    pub second_id: String,
    /// Display name of the second owner.
    pub second_name: String,
    /// Separation, metres. Zero when touching or overlapping.
    pub distance: f64,
}

/// The outcome of checking a scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollisionReport {
    /// Pairs closer than the requested margin, closest first.
    pub contacts: Vec<PairReport>,
    /// The closest measured pair, in contact or not. `None` when nothing was
    /// measured.
    pub closest: Option<PairReport>,
    /// How many body pairs were measured with GJK.
    pub pairs_measured: usize,
    /// `false` when any measurement hit its iteration limit, in which case the
    /// distances are lower bounds and "clear" should not be trusted.
    pub conclusive: bool,
}

impl CollisionReport {
    /// `true` when no pair is within the margin and every measurement converged.
    pub fn is_clear(&self) -> bool {
        self.contacts.is_empty() && self.conclusive
    }
}

fn ordered(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_owned(), b.to_owned())
    } else {
        (b.to_owned(), a.to_owned())
    }
}

fn mesh_vertices<'a>(project: &'a Project, geometry: &GeometryInstance) -> Option<&'a [f64]> {
    match &geometry.shape {
        Shape::Mesh { resource_id, .. } => project
            .resource(resource_id)
            .map(|resource| resource.vertices.as_slice()),
        _ => None,
    }
}

/// Pairs of robot links that are never measured.
///
/// Links joined by a joint touch by construction. Links joined only through
/// fixed joints are one rigid body: their distance never changes, so measuring
/// it can only ever repeat the same answer, and an overlap between them would
/// be a permanent contact.
fn robot_exemptions(robot: &axisforge_domain::RobotSystem) -> BTreeSet<(String, String)> {
    let mut ignored: BTreeSet<(String, String)> = robot
        .joints
        .iter()
        .map(|joint| ordered(&joint.parent_link_id, &joint.child_link_id))
        .collect();
    for group in rigid_groups(robot) {
        for (i, first) in group.iter().enumerate() {
            for second in &group[i + 1..] {
                ignored.insert(ordered(first, second));
            }
        }
    }
    ignored
}

/// The representative of `id`'s set, compressing the path on the way.
fn find_root(parents: &mut BTreeMap<String, String>, id: &str) -> String {
    let parent = parents[id].clone();
    if parent == id {
        return parent;
    }
    let top = find_root(parents, &parent);
    parents.insert(id.to_owned(), top.clone());
    top
}

/// Sets of link ids joined to each other only through fixed joints.
fn rigid_groups(robot: &axisforge_domain::RobotSystem) -> Vec<Vec<String>> {
    let mut parents: BTreeMap<String, String> = robot
        .links
        .iter()
        .map(|l| (l.id.clone(), l.id.clone()))
        .collect();
    for joint in robot.joints.iter().filter(|j| !j.is_actuated()) {
        let parent_set = find_root(&mut parents, &joint.parent_link_id);
        let child_set = find_root(&mut parents, &joint.child_link_id);
        if parent_set != child_set {
            parents.insert(parent_set, child_set);
        }
    }
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for link in &robot.links {
        let top = find_root(&mut parents, &link.id);
        groups.entry(top).or_default().push(link.id.clone());
    }
    groups.into_values().filter(|g| g.len() > 1).collect()
}

impl Scene {
    /// Assembles the scene for `project` with the robot at `pose`.
    ///
    /// `model` and `pose` must come from `project.robot`.
    ///
    /// # Errors
    /// [`DomainError::DanglingReference`] when an attachment names an unknown
    /// body or link, or when a geometry cannot be turned into a convex body.
    pub fn build(
        project: &Project,
        model: &KinematicModel,
        pose: &Pose,
        options: &SceneOptions,
    ) -> DomainResult<Self> {
        let mut bodies = Vec::new();
        let mut ignored: BTreeSet<(String, String)> = BTreeSet::new();

        for link in &project.robot.links {
            let index = model
                .link_index(&link.id)
                .ok_or_else(|| DomainError::dangling("link", link.id.clone(), "Scene::build"))?;
            let world = pose.link(index);
            for geometry in link.collision_geometries() {
                let body = body_from_geometry(geometry, &world, mesh_vertices(project, geometry))?;
                bodies.push(SceneBody {
                    owner_id: link.id.clone(),
                    owner_name: link.name.clone(),
                    kind: OwnerKind::Link,
                    aabb: body.aabb(),
                    body,
                });
            }
        }
        ignored.extend(robot_exemptions(&project.robot));

        for world_body in &project.environment.bodies {
            let attachment = options
                .attachment
                .as_ref()
                .filter(|a| a.body_id == world_body.id);
            let (world, kind) = match attachment {
                Some(attachment) => {
                    let link = model.link_index(&attachment.link_id).ok_or_else(|| {
                        DomainError::dangling("link", attachment.link_id.clone(), "attachment")
                    })?;
                    (
                        pose.link(link).compose(&attachment.relative_pose),
                        OwnerKind::Attached,
                    )
                }
                None => (world_body.pose, OwnerKind::World),
            };
            for geometry in world_body.geometries.iter().filter(|g| g.is_collision()) {
                let body = body_from_geometry(geometry, &world, mesh_vertices(project, geometry))?;
                bodies.push(SceneBody {
                    owner_id: world_body.id.clone(),
                    owner_name: world_body.name.clone(),
                    kind,
                    aabb: body.aabb(),
                    body,
                });
            }
        }
        if let Some(attachment) = &options.attachment {
            if project.environment.body(&attachment.body_id).is_none() {
                return Err(DomainError::dangling(
                    "world body",
                    attachment.body_id.clone(),
                    "attachment",
                ));
            }
            // A carried part is in contact with whatever holds it.
            ignored.insert(ordered(&attachment.body_id, &attachment.link_id));
            if let Some(gripper) = &project.robot.gripper {
                for link in gripper
                    .finger_link_ids
                    .iter()
                    .chain([&gripper.palm_link_id])
                {
                    ignored.insert(ordered(&attachment.body_id, link));
                }
            }
        }

        if project.environment.ground_collides {
            let body = ConvexBody::new(
                ConvexCore::Box {
                    half_extents: Vec3::new(
                        GROUND_HALF_EXTENT,
                        GROUND_HALF_EXTENT,
                        GROUND_THICKNESS * 0.5,
                    ),
                },
                Transform::from_translation(Vec3::new(
                    0.0,
                    0.0,
                    project.environment.ground_height - GROUND_THICKNESS * 0.5,
                )),
            );
            bodies.push(SceneBody {
                owner_id: GROUND_OWNER_ID.to_owned(),
                owner_name: "ground".to_owned(),
                kind: OwnerKind::Ground,
                aabb: body.aabb(),
                body,
            });
        }

        for pair in project.environment.ignored_pairs.pairs() {
            ignored.insert(ordered(&pair.first, &pair.second));
        }
        for (first, second) in &options.extra_ignored {
            ignored.insert(ordered(first, second));
        }

        Ok(Self { bodies, ignored })
    }

    /// The bodies in the scene.
    pub fn bodies(&self) -> &[SceneBody] {
        &self.bodies
    }

    /// `true` when the pair of owners is left unmeasured.
    pub fn is_ignored(&self, first: &str, second: &str) -> bool {
        self.ignored.contains(&ordered(first, second))
    }

    /// Measures every relevant pair and reports contacts and the closest pair.
    ///
    /// A pair counts as a contact when its distance is at most `margin`
    /// metres; a margin of zero reports only touching or overlapping pairs.
    pub fn check(&self, margin: f64) -> CollisionReport {
        let mut contacts: Vec<PairReport> = Vec::new();
        let mut closest: Option<PairReport> = None;
        let mut pairs_measured = 0;
        let mut conclusive = true;

        for (i, a) in self.bodies.iter().enumerate() {
            for b in &self.bodies[i + 1..] {
                if a.owner_id == b.owner_id
                    || !(a.kind.is_mobile() || b.kind.is_mobile())
                    || self.is_ignored(&a.owner_id, &b.owner_id)
                {
                    continue;
                }
                // The box gap is a lower bound on the true distance, so a pair
                // whose boxes are further apart than both the margin and the
                // best distance so far cannot change the report.
                let bound = a.aabb.distance(&b.aabb);
                let best = closest.as_ref().map_or(f64::INFINITY, |c| c.distance);
                if bound > margin && bound >= best {
                    continue;
                }
                let result = gjk::distance(&a.body, &b.body);
                pairs_measured += 1;
                conclusive &= result.status.is_conclusive();
                let report = PairReport {
                    first_id: a.owner_id.clone(),
                    first_name: a.owner_name.clone(),
                    second_id: b.owner_id.clone(),
                    second_name: b.owner_name.clone(),
                    distance: result.distance,
                };
                if result.distance < best {
                    closest = Some(report.clone());
                }
                if result.distance <= margin {
                    contacts.push(report);
                }
            }
        }
        contacts.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        // A mesh collides as several pieces; two owners touching through
        // more than one pair of pieces are still one contact, reported at its
        // closest.
        let mut seen = BTreeSet::new();
        contacts.retain(|c| seen.insert(ordered(&c.first_id, &c.second_id)));
        CollisionReport {
            contacts,
            closest,
            pairs_measured,
            conclusive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates;
    use axisforge_kinematics::forward_kinematics;

    fn cell() -> (Project, KinematicModel) {
        let project = templates::pick_and_place_cell().unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        (project, model)
    }

    fn check(project: &Project, model: &KinematicModel, values: &[f64]) -> CollisionReport {
        let pose = forward_kinematics(model, values).unwrap();
        Scene::build(project, model, &pose, &SceneOptions::default())
            .unwrap()
            .check(0.0)
    }

    #[test]
    fn the_demo_cell_is_clear_standing_upright() {
        let (project, model) = cell();
        let report = check(&project, &model, &vec![0.0; model.dof()]);
        assert!(report.is_clear(), "{:?}", report.contacts);
        assert!(report.closest.is_some());
    }

    #[test]
    fn a_closed_gripper_does_not_collide_with_itself() {
        // The fingers meet at zero opening; that pair is exempt, and the
        // palm-to-finger pairs are adjacent links.
        let (project, model) = cell();
        let report = check(&project, &model, &vec![0.0; model.dof()]);
        let gripper = project.robot.require_gripper().unwrap();
        assert!(!report
            .contacts
            .iter()
            .any(|c| gripper.finger_link_ids.contains(&c.first_id)
                && gripper.finger_link_ids.contains(&c.second_id)));
    }

    #[test]
    fn folding_the_arm_into_the_table_is_reported() {
        let (project, model) = cell();
        // Shoulder pitched far forward drives the forearm through the table.
        let mut values = vec![0.0; model.dof()];
        values[1] = 1.45;
        let report = check(&project, &model, &values);
        assert!(!report.is_clear());
        assert!(
            report.contacts.iter().any(|c| c.first_name == "table"
                || c.second_name == "table"
                || c.first_name == "ground"
                || c.second_name == "ground"),
            "{:?}",
            report.contacts
        );
    }

    #[test]
    fn folding_the_arm_onto_itself_is_reported() {
        let (project, model) = cell();
        let mut values = vec![0.0; model.dof()];
        values[2] = 2.4;
        values[4] = 2.0;
        let report = check(&project, &model, &values);
        let link_ids: Vec<&str> = project.robot.links.iter().map(|l| l.id.as_str()).collect();
        assert!(
            report
                .contacts
                .iter()
                .any(|c| link_ids.contains(&c.first_id.as_str())
                    && link_ids.contains(&c.second_id.as_str())),
            "expected a link-to-link contact, got {:?} (closest {:?})",
            report.contacts,
            report.closest
        );
    }

    #[test]
    fn scenery_resting_on_scenery_is_not_a_contact() {
        // The cube and the dowel sit on the table at distance zero.
        let (project, model) = cell();
        let report = check(&project, &model, &vec![0.0; model.dof()]);
        assert!(!report
            .contacts
            .iter()
            .any(|c| c.first_name == "cube" || c.second_name == "cube"));
    }

    #[test]
    fn a_margin_turns_a_near_miss_into_a_contact() {
        let (project, model) = cell();
        let pose = forward_kinematics(&model, &vec![0.0; model.dof()]).unwrap();
        let scene = Scene::build(&project, &model, &pose, &SceneOptions::default()).unwrap();
        let closest = scene.check(0.0).closest.unwrap().distance;
        assert!(closest > 0.0);
        assert!(!scene.check(closest + 0.001).contacts.is_empty());
        assert!(scene.check(closest * 0.5).contacts.is_empty());
    }

    #[test]
    fn an_attached_part_moves_with_its_link_and_ignores_the_gripper() {
        let (project, model) = cell();
        let gripper = project.robot.require_gripper().unwrap();
        let cube = project.environment.body_by_name("cube").unwrap();
        let options = SceneOptions {
            attachment: Some(Attachment {
                body_id: cube.id.clone(),
                link_id: gripper.palm_link_id.clone(),
                // Between the fingers, at the grasp centre.
                relative_pose: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0725)),
            }),
            extra_ignored: Vec::new(),
        };
        let mut values = vec![0.0; model.dof()];
        for index in model.end_effector_value_indices() {
            values[index] = 0.02;
        }
        let pose = forward_kinematics(&model, &values).unwrap();
        let scene = Scene::build(&project, &model, &pose, &options).unwrap();
        let carried = scene
            .bodies()
            .iter()
            .find(|b| b.owner_id == cube.id)
            .unwrap();
        assert_eq!(carried.kind, OwnerKind::Attached);
        // The arm is upright, so the part is carried far above its table pose.
        assert!(
            carried.body.pose.translation.z > 1.0,
            "{:?}",
            carried.body.pose
        );
        assert!(
            scene.check(0.0).is_clear(),
            "{:?}",
            scene.check(0.0).contacts
        );
    }

    #[test]
    fn parts_fixed_to_the_same_link_are_not_measured_against_each_other() {
        // Two bodies bolted to the last link through separate fixed joints,
        // overlapping: they are one rigid body, and must not read as a contact.
        let (mut project, _) = cell();
        let wrist = project
            .robot
            .joint_by_name("axis 6")
            .unwrap()
            .child_link_id
            .clone();
        for name in ["bracket a", "bracket b"] {
            let id = axisforge_domain::templates::stable_id("test", "link", name);
            project.robot.links.push(
                axisforge_domain::Link::new(&id, name).with_geometry(
                    axisforge_domain::GeometryInstance::visual(
                        format!("{id}-g"),
                        "block",
                        Shape::Box {
                            size: Vec3::new(0.03, 0.03, 0.03),
                        },
                    )
                    .with_origin(Transform::from_translation(Vec3::new(0.0, 0.07, 0.03)))
                    .with_role(axisforge_domain::GeometryRole::Both),
                ),
            );
            project.robot.joints.push(axisforge_domain::Joint::fixed(
                format!("{id}-j"),
                format!("{name} mount"),
                &wrist,
                &id,
                Transform::IDENTITY,
            ));
        }
        project.validate().unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let report = check(&project, &model, &vec![0.0; model.dof()]);
        assert!(
            !report.contacts.iter().any(
                |c| c.first_name.starts_with("bracket") && c.second_name.starts_with("bracket")
            ),
            "{:?}",
            report.contacts
        );
    }

    #[test]
    fn an_owner_touching_through_several_pieces_is_one_contact() {
        // A link made of three boxes, all sunk into the table: one contact.
        let (mut project, _) = cell();
        let table = project.environment.body_by_name("table").unwrap().clone();
        let top = table.pose.translation.z;
        let id = axisforge_domain::templates::stable_id("test", "link", "pieces");
        let mut link = axisforge_domain::Link::new(&id, "pieces");
        for (i, x) in [0.3, 0.4, 0.5].iter().enumerate() {
            link.geometries.push(
                axisforge_domain::GeometryInstance::visual(
                    format!("{id}-{i}"),
                    "piece",
                    Shape::Box {
                        size: Vec3::new(0.05, 0.05, 0.05),
                    },
                )
                .with_origin(Transform::from_translation(Vec3::new(*x, 0.0, top)))
                .with_role(axisforge_domain::GeometryRole::Collision),
            );
        }
        project.robot.links.push(link);
        let root = project.robot.root_link_id.clone();
        project.robot.joints.push(axisforge_domain::Joint::fixed(
            format!("{id}-j"),
            "pieces mount",
            &root,
            &id,
            Transform::IDENTITY,
        ));
        project.validate().unwrap();
        let model = KinematicModel::from_robot(&project.robot).unwrap();
        let report = check(&project, &model, &vec![0.0; model.dof()]);
        let with_table: Vec<_> = report
            .contacts
            .iter()
            .filter(|c| c.first_name == "pieces" || c.second_name == "pieces")
            .collect();
        assert_eq!(with_table.len(), 1, "{:?}", report.contacts);
        assert!(with_table[0].distance <= 0.0, "sunk pieces touch");
    }

    #[test]
    fn an_attachment_to_an_unknown_body_is_an_error() {
        let (project, model) = cell();
        let pose = forward_kinematics(&model, &vec![0.0; model.dof()]).unwrap();
        let options = SceneOptions {
            attachment: Some(Attachment {
                body_id: "ghost".to_owned(),
                link_id: project.robot.root_link_id.clone(),
                relative_pose: Transform::IDENTITY,
            }),
            extra_ignored: Vec::new(),
        };
        assert!(Scene::build(&project, &model, &pose, &options).is_err());
    }
}
