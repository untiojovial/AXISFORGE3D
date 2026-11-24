//! `axisforge`: the command-line front end.
//!
//! Every subcommand goes through the same service the other front ends use,
//! so what the terminal prints is what the viewer would draw.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use axisforge_domain::{templates, SAFETY_NOTICE};
use axisforge_kinematics::{forward_kinematics, KinematicModel};
use axisforge_project_io::import::{ImportOptions, Imported};
use axisforge_project_io::simox::{import_simox_object, import_simox_robot};
use axisforge_project_io::urdf::import_urdf;
use axisforge_service::{Request, Response, Session};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "axisforge",
    version,
    about = "Robot cell design and grasp planning."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write the pick-and-place demo cell to a project file.
    Demo {
        /// Where to write it.
        #[arg(long, default_value = "examples/pick_and_place_cell.axisforge")]
        out: PathBuf,
    },
    /// Check that a project file is well-formed and coherent.
    Validate {
        /// The project file.
        file: PathBuf,
    },
    /// List a project's joints and bodies.
    Describe {
        /// The project file.
        file: PathBuf,
    },
    /// Import a URDF robot description into a project file.
    ImportUrdf {
        /// The URDF file. Mesh files are looked for relative to it.
        file: PathBuf,
        /// Where to write the project.
        #[arg(long)]
        out: PathBuf,
        /// Joints that drive an end effector, comma separated. They do not
        /// count toward the six-axis limit.
        #[arg(long, value_delimiter = ',')]
        end_effector: Vec<String>,
    },
    /// Import a Simox robot XML into a project file.
    ImportSimox {
        /// The robot XML. Mesh files are looked for relative to it.
        file: PathBuf,
        /// Where to write the project.
        #[arg(long)]
        out: PathBuf,
        /// Extra joints that drive an end effector, comma separated. Actors of
        /// the file's own <Endeffector> are recognized without this.
        #[arg(long, value_delimiter = ',')]
        end_effector: Vec<String>,
    },
    /// Add a Simox manipulation object to a project as a part to pick.
    AddObject {
        /// The project file.
        project: PathBuf,
        /// The Simox <ManipulationObject> XML.
        object: PathBuf,
        /// Where to put the object, metres, as `x,y,z`.
        #[arg(long, value_delimiter = ',', num_args = 3)]
        at: Vec<f64>,
        /// Where to write the project; the input project when omitted.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Print the world pose of every link for given joint values, as JSON.
    Fk {
        /// The project file.
        file: PathBuf,
        /// Joint values as `name=value`, SI units. Unnamed joints stay at zero.
        #[arg(long = "set")]
        set: Vec<String>,
    },
    /// Plan picks of a body and report plans and rejections.
    Plan {
        /// The project file.
        file: PathBuf,
        /// The body's display name.
        body: String,
        /// Print the full report as JSON instead of a summary.
        #[arg(long)]
        json: bool,
    },
}

/// Saves an imported project and prints what was and was not carried over.
fn report_import(imported: &Imported, out: &Path) -> Result<()> {
    axisforge_project_io::save(&imported.project, out)?;
    let robot = &imported.project.robot;
    println!(
        "wrote {}: {} ({} axes, {} links, {} meshes)",
        out.display(),
        robot.name,
        robot.axis_count(),
        robot.links.len(),
        imported.project.resources.len()
    );
    for warning in &imported.warnings {
        println!("  warning: {warning}");
    }
    Ok(())
}

/// Finds a mesh a description refers to, relative to the description's folder.
///
/// `package://name/rest` has no fixed meaning outside ROS, so `rest` is tried
/// under the folder and each of its ancestors, with and without the package
/// name. Only STL and OBJ files are ever opened: a description is untrusted,
/// and a "mesh" reference is no reason to read arbitrary files.
fn load_mesh(folder: &Path, reference: &str) -> Option<Vec<u8>> {
    let lower = reference.to_ascii_lowercase();
    if !(lower.ends_with(".stl") || lower.ends_with(".obj")) {
        return None;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(rest) = reference.strip_prefix("package://") {
        let without_package = rest.split_once('/').map_or(rest, |(_, tail)| tail);
        for ancestor in folder.ancestors().take(6) {
            candidates.push(ancestor.join(rest));
            candidates.push(ancestor.join(without_package));
        }
    } else if let Some(path) = reference.strip_prefix("file://") {
        candidates.push(PathBuf::from(path));
    } else {
        candidates.push(folder.join(reference));
    }
    candidates
        .into_iter()
        .find_map(|path| std::fs::read(path).ok())
}

fn session_for(file: &std::path::Path) -> Result<Session> {
    let project =
        axisforge_project_io::load(file).with_context(|| format!("loading {}", file.display()))?;
    let is_demo = project.name == "Pick and place cell";
    // The demo cell has a known clear home posture; any other project starts at zero.
    if is_demo {
        let mut session = Session::demo()?;
        let text = axisforge_project_io::to_json(&project)?;
        let values = session.values().to_vec();
        session.handle(Request::LoadProject { text }, &Default::default())?;
        session.handle(Request::SetJoints { values }, &Default::default())?;
        return Ok(session);
    }
    Ok(Session::with_project(project)?)
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Demo { out } => {
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            axisforge_project_io::save(&templates::pick_and_place_cell()?, &out)?;
            println!("wrote {}", out.display());
        }
        Command::Validate { file } => {
            let project = axisforge_project_io::load(&file)?;
            println!(
                "{}: valid ({} axes, {} links, {} bodies)",
                file.display(),
                project.robot.axis_count(),
                project.robot.links.len(),
                project.environment.bodies.len()
            );
        }
        Command::Describe { file } => {
            let mut session = session_for(&file)?;
            let Response::Description(d) =
                session.handle(Request::Describe, &Default::default())?
            else {
                return Err(anyhow!("unexpected reply"));
            };
            println!(
                "{} / {} ({} axes)",
                d.project_name, d.robot_name, d.axis_count
            );
            for joint in &d.joints {
                println!(
                    "  joint {:<22} {:?}/{:?} [{:+.3}, {:+.3}] = {:+.3}",
                    joint.name, joint.kind, joint.role, joint.lower, joint.upper, joint.value
                );
            }
            for body in &d.bodies {
                println!(
                    "  body  {:<22} graspable={} mass={} kg mu={}",
                    body.name, body.graspable, body.mass, body.friction_coefficient
                );
            }
        }
        Command::ImportUrdf {
            file,
            out,
            end_effector,
        } => {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            let folder = file.parent().map(Path::to_path_buf).unwrap_or_default();
            let meshes = move |reference: &str| load_mesh(&folder, reference);
            let options = ImportOptions {
                end_effector_joints: end_effector,
            };
            let imported = import_urdf(&text, &meshes, &options)?;
            report_import(&imported, &out)?;
        }
        Command::ImportSimox {
            file,
            out,
            end_effector,
        } => {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            let folder = file.parent().map(Path::to_path_buf).unwrap_or_default();
            let meshes = move |reference: &str| load_mesh(&folder, reference);
            let options = ImportOptions {
                end_effector_joints: end_effector,
            };
            let imported = import_simox_robot(&text, &meshes, &options)?;
            report_import(&imported, &out)?;
        }
        Command::AddObject {
            project,
            object,
            at,
            out,
        } => {
            let mut target = axisforge_project_io::load(&project)?;
            let text = std::fs::read_to_string(&object)
                .with_context(|| format!("reading {}", object.display()))?;
            let folder = object.parent().map(Path::to_path_buf).unwrap_or_default();
            let meshes = move |reference: &str| load_mesh(&folder, reference);
            let mut imported = import_simox_object(&text, &meshes)?;
            if target
                .environment
                .body_by_name(&imported.body.name)
                .is_some()
            {
                return Err(anyhow!(
                    "the project already has a body named '{}'",
                    imported.body.name
                ));
            }
            if let [x, y, z] = at[..] {
                imported.body.pose.translation = axisforge_domain::Vec3::new(x, y, z);
            }
            let name = imported.body.name.clone();
            target.environment.bodies.push(imported.body);
            target.resources.extend(imported.resources);
            let out = out.unwrap_or(project);
            axisforge_project_io::save(&target, &out)?;
            println!("added '{name}' to {}", out.display());
            for warning in &imported.warnings {
                println!("  warning: {warning}");
            }
        }
        Command::Fk { file, set } => {
            let project = axisforge_project_io::load(&file)?;
            let model = KinematicModel::from_robot(&project.robot)?;
            let mut values = vec![0.0; model.dof()];
            for assignment in &set {
                let (name, value) = assignment
                    .split_once('=')
                    .ok_or_else(|| anyhow!("`{assignment}` is not of the form name=value"))?;
                let index = (0..model.dof())
                    .find(|i| model.joint_for_value(*i).is_some_and(|j| j.name == name))
                    .ok_or_else(|| anyhow!("no actuated joint named '{name}'"))?;
                values[index] = value.parse().with_context(|| format!("value of {name}"))?;
            }
            let pose = forward_kinematics(&model, &values)?;
            let mut out = serde_json::Map::new();
            for (index, link) in model.links().iter().enumerate() {
                let t = pose.link(index);
                out.insert(
                    link.name.clone(),
                    serde_json::json!([
                        t.translation.x,
                        t.translation.y,
                        t.translation.z,
                        t.rotation.x,
                        t.rotation.y,
                        t.rotation.z,
                        t.rotation.w
                    ]),
                );
            }
            println!("{}", serde_json::Value::Object(out));
        }
        Command::Plan { file, body, json } => {
            let mut session = session_for(&file)?;
            let body_id = session
                .project()
                .environment
                .body_by_name(&body)
                .ok_or_else(|| anyhow!("no body named '{body}' in {}", file.display()))?
                .id
                .clone();
            let Response::Plan(report) = session.handle(
                Request::PlanPick {
                    body_id,
                    options: None,
                },
                &Default::default(),
            )?
            else {
                return Err(anyhow!("unexpected reply"));
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }
            println!(
                "{body}: {} plan(s), {} rejected, {} of {} candidates examined",
                report.plans.len(),
                report.rejections.len(),
                report.candidates_examined,
                report.candidates_total
            );
            for plan in &report.plans {
                println!(
                    "  plan   {:<44} score {:.3}  width {:.1} mm  grip {:.1} N  clearance {:.1} mm",
                    plan.candidate.label,
                    plan.quality.score,
                    plan.candidate.width * 1e3,
                    plan.quality.required_grip_force.unwrap_or(f64::NAN),
                    plan.min_clearance * 1e3
                );
            }
            for rejection in &report.rejections {
                println!(
                    "  reject {:<44} {:?}: {}",
                    rejection.label, rejection.reason, rejection.detail
                );
            }
            println!("\n{SAFETY_NOTICE}");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
