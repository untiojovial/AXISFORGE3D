//! The application service: one command interface over the whole engine.
//!
//! Every front end — the command line, the WebAssembly build behind the
//! browser viewer, a desktop shell — talks to the engine through
//! [`Session::handle`], which takes a [`Request`] and returns a [`Response`].
//! [`Session::handle_json`] is the same thing over text. Because there is one
//! interface, there is one place where robotics is computed: a front end draws
//! what a [`Snapshot`] tells it and never recomputes a pose.
//!
//! A session owns a validated project and the robot's current configuration.
//! Requests that would leave the project invalid are refused and change
//! nothing.

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

use axisforge_collision::{Attachment, CollisionReport, Scene, SceneOptions};
use axisforge_domain::{
    templates, DomainError, JointKind, JointRole, Project, Quat, Shape, Transform, Vec3,
    SAFETY_NOTICE,
};
use axisforge_grasping::{plan_pick_with, ExtensionInfo, PlanReport, PlannerOptions, Registry};
use axisforge_kinematics::{
    conditioning, forward_kinematics, frame_pose, ik, jacobian, CancellationToken,
    ConditioningReport, IkOptions, IkSolution, KinematicModel, TaskSpace,
};
use axisforge_project_io::ProjectIoError;
use serde::{Deserialize, Serialize};

/// Smallest singular value below which a pose is flagged near-singular.
const SINGULARITY_THRESHOLD: f64 = 0.01;

/// What can go wrong handling a request.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// The request was well-formed but the engine refused it.
    #[error(transparent)]
    Domain(#[from] DomainError),
    /// A project file could not be read or written.
    #[error(transparent)]
    ProjectIo(#[from] ProjectIoError),
    /// The request text was not a valid request.
    #[error("malformed request: {0}")]
    Malformed(String),
}

/// A part carried by the gripper, for drawing and collision checking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Carry {
    /// Identifier of the carried world body.
    pub body_id: String,
    /// Pose of the body in the palm link's frame.
    pub part_in_palm: Transform,
}

impl From<serde_json::Error> for ServiceError {
    fn from(error: serde_json::Error) -> Self {
        Self::Malformed(error.to_string())
    }
}

/// A command for the session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum Request {
    /// Replace the project with the pick-and-place demo cell.
    LoadDemo,
    /// Replace the project with one parsed from project-file text.
    LoadProject {
        /// The file's contents.
        text: String,
    },
    /// Return the current project as project-file text.
    ExportProject,
    /// Describe the robot, its joints and the world's bodies.
    Describe,
    /// List the quality measures and candidate generators that are plugged in.
    ListExtensions,
    /// Set the robot's configuration and return what it looks like.
    SetJoints {
        /// One value per actuated joint, in configuration order.
        values: Vec<f64>,
    },
    /// Evaluate a configuration without storing it.
    Snapshot {
        /// The configuration; the session's current one when omitted.
        values: Option<Vec<f64>>,
        /// A part to draw and check as carried by the gripper.
        carry: Option<Carry>,
    },
    /// Solve inverse kinematics for the default tool frame.
    SolveIk {
        /// Target position, metres.
        position: Vec3,
        /// Target orientation as roll, pitch, yaw in radians; position-only when omitted.
        rpy: Option<[f64; 3]>,
    },
    /// Plan picks of a world body from the current configuration.
    PlanPick {
        /// Identifier of the body to pick.
        body_id: String,
        /// Planner settings; defaults when omitted.
        options: Option<PlannerOptions>,
    },
    /// Move a world body or change its physical properties.
    UpdateBody {
        /// Identifier of the body.
        body_id: String,
        /// New position, metres.
        position: Option<Vec3>,
        /// New rotation about world Z, radians.
        yaw: Option<f64>,
        /// New mass, kilograms.
        mass: Option<f64>,
        /// New friction coefficient.
        friction_coefficient: Option<f64>,
    },
}

/// One actuated joint, as a front end needs it for a slider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JointInfo {
    /// Display name.
    pub name: String,
    /// Kind of motion.
    pub kind: JointKind,
    /// What the joint is for.
    pub role: JointRole,
    /// Lower limit, SI units.
    pub lower: f64,
    /// Upper limit, SI units.
    pub upper: f64,
    /// Current value, SI units.
    pub value: f64,
}

/// One world body, as a front end needs it for a list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyInfo {
    /// Identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// `true` when the body can be a pick target.
    pub graspable: bool,
    /// World position, metres.
    pub position: Vec3,
    /// Mass, kilograms.
    pub mass: f64,
    /// Friction coefficient against the gripper pads.
    pub friction_coefficient: f64,
}

/// What the session holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Description {
    /// Project name.
    pub project_name: String,
    /// Robot name.
    pub robot_name: String,
    /// Number of manipulator axes.
    pub axis_count: usize,
    /// Actuated joints, in configuration order.
    pub joints: Vec<JointInfo>,
    /// World bodies.
    pub bodies: Vec<BodyInfo>,
    /// Name of the fitted gripper, if any.
    pub gripper: Option<String>,
    /// Greatest jaw opening, metres, if a gripper is fitted.
    pub gripper_max_opening: Option<f64>,
    /// Ground height, metres.
    pub ground_height: f64,
    /// The notice every front end must display alongside results.
    pub safety_notice: String,
}

/// One shape to draw, in world coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Drawable {
    /// Identifier of the owning link or body.
    pub owner_id: String,
    /// Display name of the owner.
    pub owner_name: String,
    /// `true` when the owner is a robot link.
    pub is_link: bool,
    /// The shape.
    pub shape: Shape,
    /// sRGB colour, `#rrggbb`.
    pub color: String,
    /// How metallic the surface looks, `[0, 1]`.
    pub metalness: f64,
    /// How rough the surface looks, `[0, 1]`.
    pub roughness: f64,
    /// `true` when the shape is part of what is normally drawn.
    pub visual: bool,
    /// `true` when the shape is what collision checking uses.
    pub collision: bool,
    /// World pose of the shape.
    pub pose: Transform,
}

/// Everything a front end needs to draw and annotate one configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// The configuration evaluated.
    pub values: Vec<f64>,
    /// Every shape, posed: what is drawn and what is collided, each flagged.
    pub drawables: Vec<Drawable>,
    /// World pose of the default tool frame, when the robot has one.
    pub tool_pose: Option<Transform>,
    /// Contacts and the closest pair.
    pub collision: CollisionReport,
    /// Conditioning of the manipulator Jacobian, when there is a tool frame.
    pub conditioning: Option<ConditioningReport>,
    /// Names of joints outside their limits.
    pub out_of_limits: Vec<String>,
}

/// The answer to a [`Request`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum Response {
    /// What the session holds.
    Description(Description),
    /// Project-file text.
    ProjectText(String),
    /// A drawn configuration.
    Snapshot(Box<Snapshot>),
    /// An inverse-kinematics result.
    Ik(IkSolution),
    /// A pick-planning result.
    Plan(Box<PlanReport>),
    /// The plugged-in extensions.
    Extensions(Vec<ExtensionInfo>),
}

fn drawable(
    owner_id: &str,
    owner_name: &str,
    is_link: bool,
    geometry: &axisforge_domain::GeometryInstance,
    owner_pose: &Transform,
) -> Drawable {
    Drawable {
        owner_id: owner_id.to_owned(),
        owner_name: owner_name.to_owned(),
        is_link,
        shape: geometry.shape.clone(),
        color: geometry.material.color.clone(),
        metalness: geometry.material.metalness,
        roughness: geometry.material.roughness,
        visual: geometry.is_visual(),
        collision: geometry.is_collision(),
        pose: owner_pose.compose(&geometry.origin),
    }
}

/// A project, a robot configuration, and the engine wired around them.
#[derive(Debug, Clone)]
pub struct Session {
    project: Project,
    model: KinematicModel,
    values: Vec<f64>,
    registry: Registry,
}

impl Session {
    /// A session holding the demo cell in an elbow-up home posture, gripper
    /// pointing down above the table and clear of everything.
    ///
    /// # Errors
    /// Never in practice; the template is validated by the test suite.
    pub fn demo() -> Result<Self, ServiceError> {
        let mut session = Self::with_project(templates::pick_and_place_cell()?)?;
        let home = [0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0];
        if session.values.len() == home.len() {
            session.values = home.to_vec();
        }
        Ok(session)
    }

    /// A session holding `project`, with the robot at zero.
    ///
    /// # Errors
    /// When the project does not validate.
    pub fn with_project(project: Project) -> Result<Self, ServiceError> {
        project.validate()?;
        let model = KinematicModel::from_robot(&project.robot)?;
        let values = model.clamped(&vec![0.0; model.dof()]);
        Ok(Self {
            project,
            model,
            values,
            registry: Registry::standard(),
        })
    }

    /// Swaps in another session's project while keeping this one's registered
    /// extensions: loading a file must not unplug a library.
    fn replace(&mut self, other: Self) {
        let registry = std::mem::take(&mut self.registry);
        *self = other;
        self.registry = registry;
    }

    /// The extension registry, for a host to plug libraries into.
    pub fn registry_mut(&mut self) -> &mut Registry {
        &mut self.registry
    }

    /// The project.
    pub fn project(&self) -> &Project {
        &self.project
    }

    /// The current configuration.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Handles a request given as JSON text and answers in JSON text.
    ///
    /// The answer is `{"ok":true,"kind":...,"data":...}` or
    /// `{"ok":false,"error":"..."}`; this function itself never fails, so a
    /// front end has exactly one shape of reply to handle.
    pub fn handle_json(&mut self, text: &str) -> String {
        let result = serde_json::from_str::<Request>(text)
            .map_err(ServiceError::from)
            .and_then(|request| self.handle(request, &CancellationToken::new()))
            .and_then(|response| serde_json::to_value(response).map_err(ServiceError::from));
        let reply = match result {
            Ok(mut value) => {
                if let Some(object) = value.as_object_mut() {
                    object.insert("ok".to_owned(), true.into());
                }
                value
            }
            Err(error) => serde_json::json!({ "ok": false, "error": error.to_string() }),
        };
        reply.to_string()
    }

    /// Handles a request.
    ///
    /// # Errors
    /// When the engine refuses the request; the session is then unchanged.
    pub fn handle(
        &mut self,
        request: Request,
        cancel: &CancellationToken,
    ) -> Result<Response, ServiceError> {
        match request {
            Request::LoadDemo => {
                self.replace(Self::demo()?);
                Ok(Response::Description(self.describe()))
            }
            Request::LoadProject { text } => {
                self.replace(Self::with_project(axisforge_project_io::from_json(&text)?)?);
                Ok(Response::Description(self.describe()))
            }
            Request::ExportProject => Ok(Response::ProjectText(axisforge_project_io::to_json(
                &self.project,
            )?)),
            Request::Describe => Ok(Response::Description(self.describe())),
            Request::ListExtensions => Ok(Response::Extensions(self.registry.list())),
            Request::SetJoints { values } => {
                let snapshot = self.snapshot(&values, None)?;
                self.values = values;
                Ok(Response::Snapshot(Box::new(snapshot)))
            }
            Request::Snapshot { values, carry } => {
                let values = values.unwrap_or_else(|| self.values.clone());
                Ok(Response::Snapshot(Box::new(
                    self.snapshot(&values, carry.as_ref())?,
                )))
            }
            Request::SolveIk { position, rpy } => {
                let frame = self.default_frame()?;
                let (rotation, position_only) = match rpy {
                    Some([roll, pitch, yaw]) => (Quat::from_euler_rpy(roll, pitch, yaw), false),
                    None => (Quat::IDENTITY, true),
                };
                let options = IkOptions {
                    position_only,
                    ..IkOptions::default()
                };
                let target = Transform::new(position, rotation);
                let solution =
                    ik::solve(&self.model, frame, &target, &self.values, &options, cancel)?;
                Ok(Response::Ik(solution))
            }
            Request::PlanPick { body_id, options } => {
                let report = plan_pick_with(
                    &self.registry,
                    &self.project,
                    &body_id,
                    &self.values,
                    &options.unwrap_or_default(),
                    cancel,
                )?;
                Ok(Response::Plan(Box::new(report)))
            }
            Request::UpdateBody {
                body_id,
                position,
                yaw,
                mass,
                friction_coefficient,
            } => {
                // Edit a copy and validate it, so a refused edit changes nothing.
                let mut project = self.project.clone();
                let body = project
                    .environment
                    .bodies
                    .iter_mut()
                    .find(|b| b.id == body_id)
                    .ok_or_else(|| DomainError::dangling("world body", body_id, "update_body"))?;
                if let Some(position) = position {
                    body.pose.translation = position;
                }
                if let Some(yaw) = yaw {
                    body.pose.rotation = Quat::from_axis_angle(&Vec3::Z, yaw)?;
                }
                if let Some(mass) = mass {
                    body.physical.mass = mass;
                }
                if let Some(mu) = friction_coefficient {
                    body.physical.friction_coefficient = mu;
                }
                project.validate()?;
                self.project = project;
                Ok(Response::Description(self.describe()))
            }
        }
    }

    fn default_frame(&self) -> Result<usize, DomainError> {
        self.model.default_frame().ok_or_else(|| {
            DomainError::unsupported(format!(
                "robot '{}' has several tool frames and no default; declare one",
                self.project.robot.name
            ))
        })
    }

    fn describe(&self) -> Description {
        let limits = self.model.limits();
        let joints = (0..self.model.dof())
            .filter_map(|index| {
                let joint = self.model.joint_for_value(index)?;
                Some(JointInfo {
                    name: joint.name.clone(),
                    kind: joint.kind,
                    role: joint.role,
                    lower: limits[index].lower,
                    upper: limits[index].upper,
                    value: self.values[index],
                })
            })
            .collect();
        let bodies = self
            .project
            .environment
            .bodies
            .iter()
            .map(|body| BodyInfo {
                id: body.id.clone(),
                name: body.name.clone(),
                graspable: body.is_graspable(),
                position: body.pose.translation,
                mass: body.physical.mass,
                friction_coefficient: body.physical.friction_coefficient,
            })
            .collect();
        let gripper = self.project.robot.gripper.as_ref();
        Description {
            project_name: self.project.name.clone(),
            robot_name: self.project.robot.name.clone(),
            axis_count: self.project.robot.axis_count(),
            joints,
            bodies,
            gripper: gripper.map(|g| g.name.clone()),
            gripper_max_opening: gripper.map(|g| g.max_opening),
            ground_height: self.project.environment.ground_height,
            safety_notice: SAFETY_NOTICE.to_owned(),
        }
    }

    fn snapshot(&self, values: &[f64], carry: Option<&Carry>) -> Result<Snapshot, ServiceError> {
        let pose = forward_kinematics(&self.model, values)?;
        let attachment = match (carry, &self.project.robot.gripper) {
            (Some(carry), Some(gripper)) => Some(Attachment {
                body_id: carry.body_id.clone(),
                link_id: gripper.palm_link_id.clone(),
                relative_pose: carry.part_in_palm,
            }),
            (Some(_), None) => {
                return Err(self.project.robot.require_gripper().unwrap_err().into())
            }
            (None, _) => None,
        };

        let mut drawables = Vec::new();
        for link in &self.project.robot.links {
            let index = self
                .model
                .link_index(&link.id)
                .ok_or_else(|| DomainError::dangling("link", link.id.clone(), "snapshot"))?;
            for geometry in &link.geometries {
                drawables.push(drawable(
                    &link.id,
                    &link.name,
                    true,
                    geometry,
                    &pose.link(index),
                ));
            }
        }
        for body in &self.project.environment.bodies {
            let world = match attachment.as_ref().filter(|a| a.body_id == body.id) {
                Some(attachment) => {
                    let palm = self.model.link_index(&attachment.link_id).ok_or_else(|| {
                        DomainError::dangling("link", attachment.link_id.clone(), "snapshot")
                    })?;
                    pose.link(palm).compose(&attachment.relative_pose)
                }
                None => body.pose,
            };
            for geometry in &body.geometries {
                drawables.push(drawable(&body.id, &body.name, false, geometry, &world));
            }
        }

        let options = SceneOptions {
            attachment,
            extra_ignored: Vec::new(),
        };
        let collision = Scene::build(&self.project, &self.model, &pose, &options)?.check(0.0);
        let frame = self.model.default_frame();
        let conditioning = match frame {
            Some(frame) => Some(conditioning(
                &jacobian(&self.model, &pose, frame)?,
                &self.model.manipulator_value_indices(),
                TaskSpace::Pose,
                SINGULARITY_THRESHOLD,
            )),
            None => None,
        };
        let limits = self.model.limits();
        let out_of_limits = (0..self.model.dof())
            .filter(|index| !limits[*index].contains(values[*index], 1.0e-9))
            .filter_map(|index| self.model.joint_for_value(index).map(|j| j.name.clone()))
            .collect();

        Ok(Snapshot {
            values: values.to_vec(),
            drawables,
            tool_pose: frame.map(|frame| frame_pose(&self.model, &pose, frame)),
            collision,
            conditioning,
            out_of_limits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(session: &mut Session, json: &str) -> serde_json::Value {
        serde_json::from_str(&session.handle_json(json)).expect("replies are always JSON")
    }

    #[test]
    fn describe_reports_axes_joints_and_parts() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"describe"}"#);
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["data"]["axisCount"], 6);
        assert_eq!(reply["data"]["joints"].as_array().unwrap().len(), 8);
        assert_eq!(reply["data"]["joints"][6]["role"], "end_effector");
        let graspable: Vec<&str> = reply["data"]["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["graspable"] == true)
            .map(|b| b["name"].as_str().unwrap())
            .collect();
        assert_eq!(graspable, ["cube", "dowel"]);
        assert!(reply["data"]["safetyNotice"]
            .as_str()
            .unwrap()
            .contains("not a functional-safety"));
    }

    #[test]
    fn a_snapshot_draws_every_link_and_body() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"snapshot"}"#);
        assert_eq!(reply["kind"], "snapshot");
        let drawables = reply["data"]["drawables"].as_array().unwrap();
        // What collision checking sees: base, six links, palm, two fingers,
        // the table top and legs, and the two parts.
        assert_eq!(
            drawables.iter().filter(|d| d["collision"] == true).count(),
            17
        );
        // Every owner is drawn, and each drawn shape carries its finish.
        for owner in [
            "base",
            "axis 3 link",
            "gripper palm",
            "gripper left finger",
            "table",
            "cube",
        ] {
            assert!(
                drawables
                    .iter()
                    .any(|d| d["ownerName"] == owner && d["visual"] == true),
                "{owner} is not drawn"
            );
        }
        assert!(drawables
            .iter()
            .all(|d| d["roughness"].is_number() && d["metalness"].is_number()));
        assert_eq!(
            reply["data"]["collision"]["contacts"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert!(reply["data"]["toolPose"].is_object());
    }

    #[test]
    fn a_bad_configuration_is_refused_and_changes_nothing() {
        let mut session = Session::demo().unwrap();
        let before = session.values().to_vec();
        let reply = call(&mut session, r#"{"op":"set_joints","values":[0,0]}"#);
        assert_eq!(reply["ok"], false);
        assert!(reply["error"].as_str().unwrap().contains("expected 8"));
        assert_eq!(session.values(), before);
    }

    #[test]
    fn malformed_text_gets_an_error_reply_not_a_panic() {
        let mut session = Session::demo().unwrap();
        assert_eq!(call(&mut session, "not json")["ok"], false);
        assert_eq!(
            call(&mut session, r#"{"op":"launch_missiles"}"#)["ok"],
            false
        );
    }

    #[test]
    fn planning_and_carrying_work_end_to_end() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}"}}"#),
        );
        assert_eq!(reply["kind"], "plan", "{reply}");
        let plan = &reply["data"]["plans"][0];
        assert!(plan.is_object(), "{}", reply["data"]["rejections"]);

        // Draw the lifted state with the part carried: no contacts, part raised.
        let request = serde_json::json!({
            "op": "snapshot",
            "values": plan["lift"],
            "carry": { "bodyId": cube, "partInPalm": plan["partInPalm"] },
        });
        let lifted = call(&mut session, &request.to_string());
        assert_eq!(lifted["ok"], true, "{lifted}");
        assert_eq!(
            lifted["data"]["collision"]["contacts"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        let z = lifted["data"]["drawables"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["ownerName"] == "cube")
            .unwrap()["pose"]["translation"]["z"]
            .as_f64()
            .unwrap();
        assert!(
            (z - 0.54).abs() < 1e-3,
            "the cube should be 10 cm above its rest height, z = {z}"
        );
    }

    #[test]
    fn moving_a_part_out_of_reach_changes_the_plan_outcome() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let moved = call(
            &mut session,
            &format!(
                r#"{{"op":"update_body","bodyId":"{cube}","position":{{"x":2.5,"y":0,"z":0.44}}}}"#
            ),
        );
        assert_eq!(moved["ok"], true, "{moved}");
        let options = r#"{"ik":{"positionTolerance":1e-5,"orientationTolerance":1e-4,"maxIterations":40,"restarts":1,"rngSeed":1,"damping":0.05,"positionOnly":false}}"#;
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}","options":{options}}}"#),
        );
        assert_eq!(reply["data"]["plans"].as_array().unwrap().len(), 0);
        assert_eq!(
            reply["data"]["rejections"][0]["reason"],
            "pregrasp_unreachable"
        );
    }

    #[test]
    fn an_invalid_edit_is_refused_and_the_project_is_unchanged() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let before = session.project().clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"update_body","bodyId":"{cube}","mass":-3}}"#),
        );
        assert_eq!(reply["ok"], false);
        assert_eq!(session.project(), &before);
    }

    #[test]
    fn export_then_load_round_trips() {
        let mut session = Session::demo().unwrap();
        let text = call(&mut session, r#"{"op":"export_project"}"#)["data"]
            .as_str()
            .unwrap()
            .to_owned();
        let before = session.project().clone();
        let reply = call(
            &mut session,
            &serde_json::json!({"op":"load_project","text":text}).to_string(),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(session.project(), &before);
    }

    #[test]
    fn extensions_are_listed_and_selectable_by_name() {
        let mut session = Session::demo().unwrap();
        let reply = call(&mut session, r#"{"op":"list_extensions"}"#);
        let ids: Vec<&str> = reply["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["antipodal", "wrench_space", "primitive_antipodal"]);

        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_pick","bodyId":"{cube}","options":{{"qualityMeasure":"antipodal","nrGrasps":1}}}}"#
            ),
        );
        assert_eq!(
            reply["data"]["plans"][0]["quality"]["measure"], "antipodal",
            "{reply}"
        );
        let reply = call(
            &mut session,
            &format!(
                r#"{{"op":"plan_pick","bodyId":"{cube}","options":{{"qualityMeasure":"nope"}}}}"#
            ),
        );
        assert_eq!(reply["ok"], false);
        assert!(reply["error"].as_str().unwrap().contains("wrench_space"));
    }

    #[test]
    fn a_plugged_in_library_survives_loading_another_project() {
        use axisforge_grasping::{Measurement, QualityInput, QualityMeasure};
        struct Flat;
        impl QualityMeasure for Flat {
            fn id(&self) -> &'static str {
                "flat"
            }
            fn summary(&self) -> &'static str {
                "Always the same."
            }
            fn measure(&self, _: &QualityInput<'_>) -> Measurement {
                Measurement {
                    quality: 1.0,
                    normalized: 1.0,
                    force_closure: true,
                }
            }
        }
        let mut session = Session::demo().unwrap();
        session
            .registry_mut()
            .register_measure(std::sync::Arc::new(Flat))
            .unwrap();
        call(&mut session, r#"{"op":"load_demo"}"#);
        let reply = call(&mut session, r#"{"op":"list_extensions"}"#);
        assert!(
            reply["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["id"] == "flat"),
            "{reply}"
        );
    }

    #[test]
    fn the_wasm_build_would_refuse_a_time_budget_but_native_accepts_it() {
        let mut session = Session::demo().unwrap();
        let cube = session
            .project()
            .environment
            .body_by_name("cube")
            .unwrap()
            .id
            .clone();
        let reply = call(
            &mut session,
            &format!(r#"{{"op":"plan_pick","bodyId":"{cube}","options":{{"timeoutSeconds":30}}}}"#),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["data"]["timedOut"], false);
    }

    #[test]
    fn position_only_ik_reaches_a_point() {
        let mut session = Session::demo().unwrap();
        let reply = call(
            &mut session,
            r#"{"op":"solve_ik","position":{"x":0.35,"y":0.2,"z":0.7}}"#,
        );
        assert_eq!(reply["kind"], "ik");
        assert_eq!(reply["data"]["converged"], true, "{reply}");
    }
}
