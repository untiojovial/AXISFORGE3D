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
use axisforge_service::{prepare_import, Request, Response, Session};
use clap::{Args, Parser, Subcommand, ValueEnum};

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

/// What to add around an imported robot so it can plan picks right away.
#[derive(Args)]
struct Placement {
    /// Mount one of AXISFORGE's grippers on this tool frame; `default` picks
    /// the robot's default tool frame.
    #[arg(long, value_name = "FRAME")]
    gripper: Option<String>,
    /// Which gripper `--gripper` mounts.
    #[arg(long, value_enum, default_value = "parallel")]
    gripper_kind: DemoGripper,
    /// Put the robot in the demo pick-and-place cell: a table, a cube and a
    /// dowel.
    #[arg(long)]
    demo_cell: bool,
}

impl Placement {
    fn apply(&self, imported: &mut Imported) -> Result<()> {
        if self.gripper.is_none() && !self.demo_cell {
            return Ok(());
        }
        let kind = self.gripper_kind.into();
        let frame = self
            .gripper
            .as_deref()
            .map(|f| (if f == "default" { "" } else { f }, kind));
        imported.project = prepare_import(
            imported.project.clone(),
            frame,
            self.demo_cell,
            &mut imported.warnings,
        )?;
        Ok(())
    }
}

/// The grippers the demo cell can be built with.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum DemoGripper {
    /// Two parallel jaws.
    Parallel,
    /// A centric gripper with three radial jaws.
    ThreeFinger,
    /// A hand of three two-phalanx fingers.
    Hand,
}

impl From<DemoGripper> for templates::GripperKind {
    fn from(value: DemoGripper) -> Self {
        match value {
            DemoGripper::Parallel => Self::Parallel,
            DemoGripper::ThreeFinger => Self::ThreeFinger,
            DemoGripper::Hand => Self::Hand,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Write the pick-and-place demo cell to a project file.
    Demo {
        /// Where to write it.
        #[arg(long, default_value = "examples/pick_and_place_cell.axisforge")]
        out: PathBuf,
        /// Leave the table empty: no cube, no dowel.
        #[arg(long)]
        no_parts: bool,
        /// The gripper on the arm.
        #[arg(long, value_enum, default_value = "parallel")]
        gripper: DemoGripper,
        /// Obstacles on the table.
        #[arg(long, value_enum, default_value = "none")]
        obstacles: DemoObstacles,
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
        #[command(flatten)]
        placement: Placement,
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
        #[command(flatten)]
        placement: Placement,
    },
    /// Add a Simox manipulation object to a project as a part to pick.
    AddObject {
        /// The project file.
        project: PathBuf,
        /// The Simox <ManipulationObject> XML, or a bare STL, OBJ or COLLADA
        /// mesh, whose origin is then the centre of its base.
        object: PathBuf,
        /// Where to put the object, metres, as `x y z`.
        #[arg(long, num_args = 3, allow_negative_numbers = true)]
        at: Vec<f64>,
        /// A bare mesh's units: `m` or `mm`; guessed from its size and
        /// reported when omitted.
        #[arg(long)]
        units: Option<String>,
        /// A bare mesh's part name; the file name without extension when omitted.
        #[arg(long)]
        name: Option<String>,
        /// The part's mass, kilograms, replacing the estimate or default.
        #[arg(long)]
        mass: Option<f64>,
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
    /// Plan the whole cycle: pick a part, carry it to a target and set it
    /// down, from the start posture and back, every motion proven clear.
    Cycle {
        /// The project file.
        file: PathBuf,
        /// The part's display name.
        body: String,
        /// Where to set the part's origin down, metres: X Y, and Z when it is
        /// not to stay at its current height.
        #[arg(long, num_args = 2..=3, value_name = "X Y [Z]", allow_negative_numbers = true, required = true)]
        to: Vec<f64>,
        /// Fraction of every joint's speed limit to use.
        #[arg(long, default_value_t = 0.5)]
        speed: f64,
        /// The start posture, radians or metres: one value per manipulator
        /// axis (the gripper as it is) or per joint. The demo cell's home, or
        /// all zeros, when omitted.
        #[arg(long, num_args = 1.., value_name = "VALUES", allow_negative_numbers = true)]
        start: Option<Vec<f64>>,
        /// Write the program: `urscript`, `csv` or `json`.
        #[arg(long, value_name = "FORMAT")]
        export: Option<String>,
        /// Where to write it; the format's usual file name when omitted.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Print the full report as JSON instead of a summary.
        #[arg(long)]
        json: bool,
    },
}

/// The obstacles the demo cell can be built with.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum DemoObstacles {
    /// None.
    None,
    /// A wall across the table.
    Wall,
    /// Two posts on the table.
    Posts,
}

impl From<DemoObstacles> for templates::Obstacles {
    fn from(value: DemoObstacles) -> Self {
        match value {
            DemoObstacles::None => Self::None,
            DemoObstacles::Wall => Self::Wall,
            DemoObstacles::Posts => Self::Posts,
        }
    }
}

/// Plans a cycle and prints, or writes, what came of it.
#[allow(clippy::too_many_arguments)]
fn run_cycle(
    file: &Path,
    body: &str,
    to: &[f64],
    speed: f64,
    start: Option<&[f64]>,
    export: Option<&str>,
    out: Option<&Path>,
    json: bool,
) -> Result<()> {
    let mut session = session_for(file)?;
    if let Some(start) = start {
        let mut values = session.values().to_vec();
        let model = KinematicModel::from_robot(&session.project().robot)?;
        let arm = model.manipulator_value_indices();
        if start.len() == values.len() {
            values.copy_from_slice(start);
        } else if start.len() == arm.len() {
            for (slot, value) in arm.iter().zip(start) {
                values[*slot] = *value;
            }
        } else {
            return Err(anyhow!(
                "--start takes {} values (the manipulator axes) or {} (every joint); {} given",
                arm.len(),
                values.len(),
                start.len()
            ));
        }
        session.handle(Request::SetJoints { values }, &Default::default())?;
    }
    let part = session
        .project()
        .environment
        .body_by_name(body)
        .ok_or_else(|| anyhow!("no body named '{body}' in {}", file.display()))?;
    let body_id = part.id.clone();
    let target = axisforge_domain::Vec3::new(
        to[0],
        to[1],
        to.get(2).copied().unwrap_or(part.pose.translation.z),
    );
    let options = axisforge_motion::CycleOptions {
        speed_scale: speed,
        ..Default::default()
    };
    let Response::Cycle(view) = session.handle(
        Request::PlanCycle {
            body_id,
            target,
            options: Some(options),
        },
        &Default::default(),
    )?
    else {
        return Err(anyhow!("unexpected reply"));
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&view.report)?);
    }
    let Some(cycle) = &view.report.cycle else {
        if !json {
            println!(
                "{body}: no cycle found; {} grasp(s) were feasible",
                view.report.grasps_found
            );
            for attempt in &view.report.attempts {
                println!(
                    "  {:<44} {}: {}",
                    attempt.grasp, attempt.segment, attempt.reason
                );
            }
            println!("\n{SAFETY_NOTICE}");
        }
        return Err(anyhow!("no cycle found for '{body}'"));
    };
    if !json {
        println!(
            "{body}: cycle {:.3} s (program re-timed: {:.3} s), grasp {}, the fastest of {} \
             complete cycle(s) from {} feasible grasp(s)",
            cycle.cycle_time,
            view.program_seconds.unwrap_or(f64::NAN),
            cycle.grasp.candidate.label,
            view.report.cycles_found,
            view.report.grasps_found
        );
        for segment in &cycle.segments {
            println!(
                "  {:<9} {:<8} {:>2} move(s) {:>7.3} s{}  clearance ≥ {:.1} mm (closest measured {:.1} mm){}",
                segment.name,
                segment.planner,
                segment.moves.len(),
                segment.duration,
                if segment.dwell > 0.0 {
                    format!(" + {:.2} s", segment.dwell)
                } else {
                    String::new()
                },
                segment.clearance * 1e3,
                segment.closest * 1e3,
                if segment.allowed_contacts.is_empty() {
                    String::new()
                } else {
                    format!("; allowed: {}", segment.allowed_contacts.join(", "))
                }
            );
        }
        for attempt in &view.report.attempts {
            println!(
                "  grasp without a cycle {}: {}: {}",
                attempt.grasp, attempt.segment, attempt.reason
            );
        }
        if !view.report.assumed_limits.is_empty() {
            println!(
                "  warning: limits not declared, so assumed: {}",
                view.report.assumed_limits.join(", ")
            );
        }
    }
    if let Some(format) = export {
        let Response::Program(program) = session.handle(
            Request::ExportCycle {
                format: format.to_owned(),
                period: None,
            },
            &Default::default(),
        )?
        else {
            return Err(anyhow!("unexpected reply"));
        };
        let path = out.map_or_else(|| PathBuf::from(&program.file_name), Path::to_path_buf);
        std::fs::write(&path, program.text)
            .with_context(|| format!("writing {}", path.display()))?;
        if !json {
            println!("wrote {}", path.display());
        }
    }
    if !json {
        println!("\n{SAFETY_NOTICE}");
    }
    Ok(())
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
/// name. Failing that, a file of the same name anywhere under the folder is
/// used if there is exactly one, as the viewer does with the files it is
/// given. Only STL, OBJ and COLLADA files are ever opened: a description is untrusted,
/// and a "mesh" reference is no reason to read arbitrary files.
fn load_mesh(folder: &Path, reference: &str) -> Option<Vec<u8>> {
    if !axisforge_project_io::mesh::is_supported(reference) {
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
    if let Some(bytes) = candidates
        .into_iter()
        .find_map(|path| std::fs::read(path).ok())
    {
        return Some(bytes);
    }
    let name = reference.rsplit(['/', '\\']).next()?;
    let mut found = Vec::new();
    find_by_name(folder, name, 4, &mut found);
    match found.as_slice() {
        [only] => std::fs::read(only).ok(),
        _ => None,
    }
}

/// Collects up to two files called `name` under `folder`, at most `depth`
/// folders down, without following symbolic links.
fn find_by_name(folder: &Path, name: &str, depth: usize, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        if found.len() > 1 {
            return;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_file() && entry.file_name() == name {
            found.push(entry.path());
        } else if kind.is_dir() && depth > 0 {
            find_by_name(&entry.path(), name, depth - 1, found);
        }
    }
}

fn session_for(file: &std::path::Path) -> Result<Session> {
    let project =
        axisforge_project_io::load(file).with_context(|| format!("loading {}", file.display()))?;
    let is_demo = project.name == "Pick and place cell";
    let mut session = Session::with_project(project)?;
    // The demo cell has a known clear home posture; any other project starts at zero.
    if is_demo {
        let model = KinematicModel::from_robot(&session.project().robot)?;
        if let Some(values) = axisforge_service::demo_home(session.project(), &model) {
            session.handle(Request::SetJoints { values }, &Default::default())?;
        }
    }
    Ok(session)
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Demo {
            out,
            no_parts,
            gripper,
            obstacles,
        } => {
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            let mut cell = templates::demo_cell(gripper.into(), obstacles.into())?;
            if no_parts {
                cell.environment.bodies.retain(|b| !b.is_graspable());
                let kept: std::collections::BTreeSet<String> = cell
                    .environment
                    .bodies
                    .iter()
                    .map(|b| b.id.clone())
                    .chain(cell.robot.links.iter().map(|l| l.id.clone()))
                    .chain([axisforge_domain::GROUND_OWNER_ID.to_owned()])
                    .collect();
                let pairs: Vec<_> = cell.environment.ignored_pairs.pairs().to_vec();
                cell.environment.ignored_pairs = axisforge_domain::IgnoredPairs::new();
                for pair in pairs
                    .into_iter()
                    .filter(|p| kept.contains(&p.first) && kept.contains(&p.second))
                {
                    cell.environment
                        .ignored_pairs
                        .ignore(&pair.first, &pair.second, pair.reason);
                }
                cell.validate()?;
            }
            axisforge_project_io::save(&cell, &out)?;
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
            placement,
        } => {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            let folder = file.parent().map(Path::to_path_buf).unwrap_or_default();
            let meshes = move |reference: &str| load_mesh(&folder, reference);
            let options = ImportOptions {
                end_effector_joints: end_effector,
            };
            let mut imported = import_urdf(&text, &meshes, &options)?;
            placement.apply(&mut imported)?;
            report_import(&imported, &out)?;
        }
        Command::ImportSimox {
            file,
            out,
            end_effector,
            placement,
        } => {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            let folder = file.parent().map(Path::to_path_buf).unwrap_or_default();
            let meshes = move |reference: &str| load_mesh(&folder, reference);
            let options = ImportOptions {
                end_effector_joints: end_effector,
            };
            let mut imported = import_simox_robot(&text, &meshes, &options)?;
            placement.apply(&mut imported)?;
            report_import(&imported, &out)?;
        }
        Command::AddObject {
            project,
            object,
            at,
            units,
            name,
            mass,
            out,
        } => {
            let mut target = axisforge_project_io::load(&project)?;
            let file_name = object
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_owned();
            let mut imported = if axisforge_project_io::mesh::is_supported(&file_name) {
                let scale = match units.as_deref() {
                    None => None,
                    Some("m") => Some(1.0),
                    Some("mm") => Some(0.001),
                    Some(other) => return Err(anyhow!("units `{other}`: use `m` or `mm`")),
                };
                let bytes = std::fs::read(&object)
                    .with_context(|| format!("reading {}", object.display()))?;
                let stem = object
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .unwrap_or("part")
                    .to_owned();
                axisforge_project_io::object::import_mesh_object(
                    &name.unwrap_or(stem),
                    &file_name,
                    &bytes,
                    scale,
                )?
            } else {
                let text = std::fs::read_to_string(&object)
                    .with_context(|| format!("reading {}", object.display()))?;
                let folder = object.parent().map(Path::to_path_buf).unwrap_or_default();
                let meshes = move |reference: &str| load_mesh(&folder, reference);
                import_simox_object(&text, &meshes)?
            };
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
            if let Some(mass) = mass {
                if !(mass.is_finite() && mass > 0.0) {
                    return Err(anyhow!("a mass of {mass} kg is not a positive number"));
                }
                imported.body.physical.mass = mass;
                imported
                    .warnings
                    .retain(|w| !w.contains("mass") || !w.contains("before planning"));
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
            // On stderr, so `--json` output stays a single document.
            let blind = axisforge_service::links_without_collision(&session.project().robot);
            if !blind.is_empty() {
                eprintln!(
                    "warning: {} moving link(s) have no collision geometry and were not checked: {}",
                    blind.len(),
                    blind.join(", ")
                );
            }
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
                let hold = match &plan.candidate.hand {
                    Some(hand) => format!(
                        "{} fingers, {} contacts",
                        hand.fingers_in_contact,
                        plan.candidate.contacts.len()
                    ),
                    None => format!("width {:.1} mm", plan.candidate.width * 1e3),
                };
                println!(
                    "  plan   {:<44} score {:.3}  {hold}  grip {:.1} N  clearance {:.1} mm",
                    plan.candidate.label,
                    plan.quality.score,
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
        Command::Cycle {
            file,
            body,
            to,
            speed,
            start,
            export,
            out,
            json,
        } => run_cycle(
            &file,
            &body,
            &to,
            speed,
            start.as_deref(),
            export.as_deref(),
            out.as_deref(),
            json,
        )?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshes_are_found_by_path_then_by_unambiguous_name() {
        let root = std::env::temp_dir().join(format!("axisforge-cli-mesh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("pkg/meshes/collision")).unwrap();
        std::fs::create_dir_all(root.join("flat/meshes")).unwrap();
        std::fs::create_dir_all(root.join("flat/a")).unwrap();
        std::fs::create_dir_all(root.join("flat/b")).unwrap();
        std::fs::write(root.join("pkg/meshes/collision/base.stl"), b"by path").unwrap();
        std::fs::write(root.join("flat/meshes/base.stl"), b"by name").unwrap();
        std::fs::write(root.join("flat/a/twice.stl"), b"a").unwrap();
        std::fs::write(root.join("flat/b/twice.stl"), b"b").unwrap();
        std::fs::write(root.join("flat/secret.txt"), b"no").unwrap();

        let by_path = load_mesh(&root.join("pkg"), "package://pkg/meshes/collision/base.stl");
        assert_eq!(by_path.as_deref(), Some(&b"by path"[..]));
        let by_name = load_mesh(
            &root.join("flat"),
            "package://robot/meshes/ur5/collision/base.stl",
        );
        assert_eq!(by_name.as_deref(), Some(&b"by name"[..]));
        assert_eq!(
            load_mesh(&root.join("flat"), "package://robot/twice.stl"),
            None,
            "ambiguous"
        );
        assert_eq!(
            load_mesh(&root.join("flat"), "secret.txt"),
            None,
            "not a mesh"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
