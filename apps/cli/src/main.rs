//! `axisforge`: the command-line front end.
//!
//! Every subcommand goes through the same service the other front ends use,
//! so what the terminal prints is what the viewer would draw.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use axisforge_domain::{templates, SAFETY_NOTICE};
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
