//! Programs for a controller, and the check that they take the time the
//! profiles say.
//!
//! A cycle becomes a list of [`ProgramStep`]s: a joint move per straight
//! segment, given as the target and the speed and acceleration of its
//! leading axis, a gripper action per finger move, and the dwells. From it:
//!
//! - [`urscript`] writes a `URScript` program for a six-axis arm: `movej` with
//!   `a` and `v` for the leading axis and no blending, which is the motion
//!   the profiles describe, and two gripper functions that only wait, to be
//!   replaced with the gripper's own commands;
//! - [`sampled`] samples positions and velocities at a fixed period, for
//!   controllers that take a streamed or tabulated joint trajectory, and
//!   [`csv`] writes those samples as text.
//!
//! [`simulate_program`] runs the steps as a controller's interpolator
//! would, cycle by cycle, from the numbers in the program alone. Its total
//! is the program's time; comparing it with [`Cycle::cycle_time`] checks the
//! profiles and the export against each other.

use std::fmt::Write as _;

use axisforge_domain::{DomainError, DomainResult, SAFETY_NOTICE};
use serde::{Deserialize, Serialize};

use crate::cycle::{Cycle, PartState, SegmentKind};
use crate::profile::simulate_move;

/// One instruction of a cycle program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "step",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum ProgramStep {
    /// A joint move with no blending.
    MoveJoints {
        /// The segment it belongs to.
        segment: String,
        /// Target values of the manipulator axes, in order.
        target: Vec<f64>,
        /// Commanded speed of the leading axis.
        velocity: f64,
        /// Commanded acceleration of the leading axis.
        acceleration: f64,
        /// How far the leading axis travels.
        leading_distance: f64,
        /// Duration of the move's profile, seconds.
        duration: f64,
    },
    /// The fingers move.
    Gripper {
        /// The segment it belongs to: `open`, `close`, `release` or `rest`.
        segment: String,
        /// `true` when the fingers close on the part.
        close: bool,
        /// Target values of the finger joints, in order.
        target: Vec<f64>,
        /// How long the move takes, seconds.
        seconds: f64,
    },
    /// A pause.
    Wait {
        /// The segment it belongs to.
        segment: String,
        /// Seconds.
        seconds: f64,
    },
}

/// The program of a cycle.
pub fn program(cycle: &Cycle) -> Vec<ProgramStep> {
    let arm = &cycle.manipulator_slots;
    let fingers: Vec<usize> = (0..cycle.joint_names.len())
        .filter(|slot| !arm.contains(slot))
        .collect();
    let mut steps = Vec::new();
    for segment in &cycle.segments {
        for m in &segment.moves {
            if m.duration <= 0.0 {
                continue;
            }
            if segment.kind == SegmentKind::Gripper || m.moves_only(&fingers) {
                steps.push(ProgramStep::Gripper {
                    segment: segment.name.clone(),
                    close: segment.name == "close",
                    target: fingers.iter().map(|s| m.to[*s]).collect(),
                    seconds: m.duration,
                });
            } else {
                steps.push(ProgramStep::MoveJoints {
                    segment: segment.name.clone(),
                    target: arm.iter().map(|s| m.to[*s]).collect(),
                    velocity: m.leading_velocity,
                    acceleration: m.leading_acceleration,
                    leading_distance: (m.to[m.leading_axis] - m.from[m.leading_axis]).abs(),
                    duration: m.duration,
                });
            }
        }
        if segment.dwell > 0.0 {
            steps.push(ProgramStep::Wait {
                segment: segment.name.clone(),
                seconds: segment.dwell,
            });
        }
    }
    steps
}

/// How long a program takes when its joint moves are run by a controller
/// interpolating every `dt` seconds; gripper moves and waits take the time
/// they say.
pub fn simulate_program(steps: &[ProgramStep], dt: f64) -> f64 {
    steps
        .iter()
        .map(|step| match step {
            ProgramStep::MoveJoints {
                velocity,
                acceleration,
                leading_distance,
                ..
            } => simulate_move(*leading_distance, *velocity, *acceleration, dt),
            ProgramStep::Gripper { seconds, .. } | ProgramStep::Wait { seconds, .. } => *seconds,
        })
        .sum()
}

fn list(values: &[f64]) -> String {
    values
        .iter()
        // Values that round to zero are written as zero, not as -0.000000.
        .map(|v| format!("{:.6}", if v.abs() < 5.0e-7 { 0.0 } else { *v }))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A `URScript` program running the cycle once.
///
/// # Errors
/// When the robot does not have six manipulator axes, which is what
/// `movej` drives.
pub fn urscript(cycle: &Cycle, part_name: &str) -> DomainResult<String> {
    if cycle.manipulator_slots.len() != 6 {
        return Err(DomainError::unsupported(format!(
            "URScript drives six-axis arms; this robot has {} manipulator axes",
            cycle.manipulator_slots.len()
        )));
    }
    let names: Vec<&str> = cycle
        .manipulator_slots
        .iter()
        .map(|s| cycle.joint_names[*s].as_str())
        .collect();
    let start = cycle.state_at(0.0).values;
    let start_arm: Vec<f64> = cycle.manipulator_slots.iter().map(|s| start[*s]).collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# AXISFORGE 3D: pick-and-place cycle for '{part_name}'.\n\
         # Planned cycle time {:.3} s, from the start configuration back to it.\n\
         # Joint order: {}.\n\
         # {SAFETY_NOTICE}\n\
         # The two gripper functions only wait; replace their bodies with your\n\
         # gripper's own commands before running the program.",
        cycle.cycle_time,
        names.join(", ")
    );
    out.push_str("def axisforge_cycle():\n");
    out.push_str("  def axisforge_gripper_close(seconds):\n    sleep(seconds)\n  end\n");
    out.push_str("  def axisforge_gripper_open(seconds):\n    sleep(seconds)\n  end\n");
    let _ = writeln!(
        out,
        "  # Go to the start configuration (not part of the cycle time).\n  movej([{}], a=0.5, v=0.25, r=0)",
        list(&start_arm)
    );
    let mut current = String::new();
    for step in program(cycle) {
        let segment = match &step {
            ProgramStep::MoveJoints { segment, .. }
            | ProgramStep::Gripper { segment, .. }
            | ProgramStep::Wait { segment, .. } => segment.clone(),
        };
        if segment != current {
            let _ = writeln!(out, "  # {segment}");
            current = segment;
        }
        match step {
            ProgramStep::MoveJoints {
                target,
                velocity,
                acceleration,
                duration,
                ..
            } => {
                let _ = writeln!(
                    out,
                    "  movej([{}], a={acceleration:.6}, v={velocity:.6}, r=0)  # {duration:.3} s",
                    list(&target)
                );
            }
            ProgramStep::Gripper { close, seconds, .. } => {
                let _ = writeln!(
                    out,
                    "  axisforge_gripper_{}({seconds:.3})",
                    if close { "close" } else { "open" }
                );
            }
            ProgramStep::Wait { seconds, .. } => {
                let _ = writeln!(out, "  sleep({seconds:.3})");
            }
        }
    }
    out.push_str("end\n");
    Ok(out)
}

/// One sample of a cycle's trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryPoint {
    /// Seconds from the start.
    pub time_from_start: f64,
    /// Every joint, in configuration order.
    pub positions: Vec<f64>,
    /// Their velocities.
    pub velocities: Vec<f64>,
    /// Where the part is.
    pub part: PartState,
}

/// A cycle sampled at a fixed period.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SampledTrajectory {
    /// Joint names, in configuration order.
    pub joint_names: Vec<String>,
    /// Sampling period, seconds.
    pub period: f64,
    /// The samples, the last one at the end of the cycle.
    pub points: Vec<TrajectoryPoint>,
}

/// Samples a cycle every `period` seconds, and at its end.
///
/// # Errors
/// A period that is not positive, or so short that the samples would
/// exceed a million.
pub fn sampled(cycle: &Cycle, period: f64) -> DomainResult<SampledTrajectory> {
    if !period.is_finite() || period <= 0.0 {
        return Err(DomainError::NegativeValue {
            context: "sampling period".to_owned(),
            value: period,
        });
    }
    let count = (cycle.cycle_time / period).ceil();
    if count > 1.0e6 {
        return Err(DomainError::unsupported(format!(
            "{count} samples; use a longer period"
        )));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let count = count as usize;
    let mut points = Vec::with_capacity(count + 1);
    for k in 0..=count {
        #[allow(clippy::cast_precision_loss)]
        let t = (k as f64 * period).min(cycle.cycle_time);
        let state = cycle.state_at(t);
        let velocities = velocity_at(cycle, t);
        points.push(TrajectoryPoint {
            time_from_start: t,
            positions: state.values,
            velocities,
            part: state.part,
        });
        if t >= cycle.cycle_time {
            break;
        }
    }
    Ok(SampledTrajectory {
        joint_names: cycle.joint_names.clone(),
        period,
        points,
    })
}

fn velocity_at(cycle: &Cycle, t: f64) -> Vec<f64> {
    let zero = vec![0.0; cycle.joint_names.len()];
    for segment in &cycle.segments {
        let mut local = t - segment.start_time;
        if local < 0.0 || local > segment.duration {
            continue;
        }
        for m in &segment.moves {
            if local <= m.duration {
                return m.velocity(local);
            }
            local -= m.duration;
        }
    }
    zero
}

/// The samples as comma-separated text: a header, then one row per sample.
pub fn csv(trajectory: &SampledTrajectory) -> String {
    let mut out = String::from("time_s");
    for name in &trajectory.joint_names {
        out.push(',');
        out.push_str(&name.replace([',', '"', '\n'], " "));
    }
    out.push_str(",part\n");
    for point in &trajectory.points {
        let _ = write!(out, "{:.4}", point.time_from_start);
        for value in &point.positions {
            let _ = write!(out, ",{value:.6}");
        }
        let part = match point.part {
            PartState::AtStart => "at_start",
            PartState::Carried => "carried",
            PartState::Placed => "placed",
        };
        let _ = writeln!(out, ",{part}");
    }
    out
}
