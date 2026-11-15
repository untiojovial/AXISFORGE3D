//! The AXISFORGE project file: a versioned JSON envelope around a project.
//!
//! ```json
//! { "format": "axisforge-project", "version": 2, "project": { ... } }
//! ```
//!
//! Reading validates: a file that parses but describes an incoherent robot is
//! rejected here, so nothing downstream ever holds an unvalidated project.
//! Writing is atomic: the bytes go to a temporary file beside the target, are
//! flushed to disk, and are then renamed over it, so a crash or a full disk
//! leaves the previous file intact instead of a truncated one.
//!
//! Version 1 files (written before grippers existed) are read as they are:
//! every field added since has a serde default, so no rewriting is needed. A
//! file from a *newer* version is refused rather than half-understood.

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::must_use_candidate, clippy::missing_errors_doc)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use axisforge_domain::{DomainError, Project};
use serde::{Deserialize, Serialize};

/// The `format` tag every project file carries.
pub const FORMAT_TAG: &str = "axisforge-project";
/// The version this build writes.
pub const CURRENT_VERSION: u32 = 2;
/// Largest file this crate will read, bytes.
///
/// A project is text describing a robot cell; even with inline meshes, a file
/// beyond this is far more likely to be a mistake or an attack than a project.
pub const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// What can go wrong reading or writing a project file.
#[derive(Debug, thiserror::Error)]
pub enum ProjectIoError {
    /// The file could not be read or written.
    #[error("could not {action} `{path}`: {source}")]
    Io {
        /// What was being attempted.
        action: &'static str,
        /// The file involved.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The file is larger than [`MAX_FILE_BYTES`].
    #[error("`{path}` is {size} bytes; the limit is {limit}")]
    TooLarge {
        /// The file involved.
        path: PathBuf,
        /// Its size.
        size: u64,
        /// The limit.
        limit: u64,
    },
    /// The text is not valid JSON of the expected shape.
    #[error("not a valid project file: {0}")]
    Malformed(String),
    /// The `format` tag is not [`FORMAT_TAG`].
    #[error("unexpected format tag `{found}`; expected `{FORMAT_TAG}`")]
    WrongFormat {
        /// The tag found.
        found: String,
    },
    /// The file was written by a newer version of the format.
    #[error("file is format version {found}; this build reads up to {CURRENT_VERSION}")]
    TooNew {
        /// The version found.
        found: u32,
    },
    /// The file parsed but the project in it is not coherent.
    #[error("the project is invalid: {0}")]
    Invalid(DomainError),
}

// The messages above already include the cause, so these conversions do not
// also expose it as a `source`; a chained printer would say everything twice.
impl From<serde_json::Error> for ProjectIoError {
    fn from(error: serde_json::Error) -> Self {
        Self::Malformed(error.to_string())
    }
}

impl From<DomainError> for ProjectIoError {
    fn from(error: DomainError) -> Self {
        Self::Invalid(error)
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    format: String,
    version: u32,
    project: Project,
}

/// Serializes a project to file text, validating it first.
///
/// # Errors
/// [`ProjectIoError::Invalid`] when the project does not validate.
pub fn to_json(project: &Project) -> Result<String, ProjectIoError> {
    project.validate()?;
    let envelope = Envelope {
        format: FORMAT_TAG.to_owned(),
        version: CURRENT_VERSION,
        project: project.clone(),
    };
    let mut text = serde_json::to_string_pretty(&envelope)?;
    text.push('\n');
    Ok(text)
}

/// Parses and validates file text.
///
/// # Errors
/// Any [`ProjectIoError`] except the I/O variants.
pub fn from_json(text: &str) -> Result<Project, ProjectIoError> {
    // Check the tag and version before deserializing the project, so a foreign
    // or future file gets the message about what it is, not about whichever
    // field happened to be missing.
    let header: serde_json::Value = serde_json::from_str(text)?;
    let found = header
        .get("format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if found != FORMAT_TAG {
        return Err(ProjectIoError::WrongFormat {
            found: found.to_owned(),
        });
    }
    let version = header
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if version > u64::from(CURRENT_VERSION) {
        return Err(ProjectIoError::TooNew {
            found: u32::try_from(version).unwrap_or(u32::MAX),
        });
    }
    let envelope: Envelope = serde_json::from_value(header)?;
    envelope.project.validate()?;
    Ok(envelope.project)
}

/// Reads and validates a project file.
///
/// # Errors
/// Any [`ProjectIoError`].
pub fn load(path: &Path) -> Result<Project, ProjectIoError> {
    let io = |action| {
        move |source| ProjectIoError::Io {
            action,
            path: path.to_path_buf(),
            source,
        }
    };
    let size = fs::metadata(path).map_err(io("inspect"))?.len();
    if size > MAX_FILE_BYTES {
        return Err(ProjectIoError::TooLarge {
            path: path.to_path_buf(),
            size,
            limit: MAX_FILE_BYTES,
        });
    }
    from_json(&fs::read_to_string(path).map_err(io("read"))?)
}

/// Writes a project file atomically.
///
/// # Errors
/// [`ProjectIoError::Invalid`] before anything is written when the project
/// does not validate, or [`ProjectIoError::Io`]; in both cases any existing
/// file at `path` is untouched.
pub fn save(project: &Project, path: &Path) -> Result<(), ProjectIoError> {
    let text = to_json(project)?;
    let io = |action| {
        move |source| ProjectIoError::Io {
            action,
            path: path.to_path_buf(),
            source,
        }
    };
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(text.as_bytes())?;
        // Without the sync, a rename can land before the data does, and a
        // power cut would leave an empty file under the real name.
        file.sync_all()?;
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(io("write"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::templates;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("axisforge-io-{}-{name}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_project_survives_a_round_trip_bit_for_bit() {
        let project = templates::pick_and_place_cell().unwrap();
        let text = to_json(&project).unwrap();
        let restored = from_json(&text).unwrap();
        assert_eq!(project, restored);
        assert_eq!(
            text,
            to_json(&restored).unwrap(),
            "saving twice must not change the file"
        );
    }

    #[test]
    fn awkward_floats_are_not_rounded() {
        let mut project = templates::pick_and_place_cell().unwrap();
        project.environment.bodies[1].pose.translation.x = -0.027_777_954_397_958_354;
        let restored = from_json(&to_json(&project).unwrap()).unwrap();
        assert_eq!(
            restored.environment.bodies[1].pose.translation.x.to_bits(),
            (-0.027_777_954_397_958_354_f64).to_bits()
        );
    }

    #[test]
    fn a_foreign_file_is_named_as_such() {
        let error = from_json(r#"{"format":"something-else","version":1}"#).unwrap_err();
        assert!(
            matches!(error, ProjectIoError::WrongFormat { .. }),
            "{error}"
        );
        assert!(matches!(
            from_json("{}").unwrap_err(),
            ProjectIoError::WrongFormat { .. }
        ));
        assert!(matches!(
            from_json("not json").unwrap_err(),
            ProjectIoError::Malformed(_)
        ));
    }

    #[test]
    fn a_newer_file_is_refused() {
        let text = to_json(&templates::pick_and_place_cell().unwrap())
            .unwrap()
            .replace("\"version\": 2", "\"version\": 3");
        assert!(matches!(
            from_json(&text).unwrap_err(),
            ProjectIoError::TooNew { found: 3 }
        ));
    }

    #[test]
    fn a_version_one_file_without_gripper_fields_still_loads() {
        let project = axisforge_domain::Project::new("p", "old", templates::six_axis().unwrap());
        let mut value: serde_json::Value =
            serde_json::from_str(&to_json(&project).unwrap()).unwrap();
        value["version"] = 1.into();
        let robot = value["project"]["robot"].as_object_mut().unwrap();
        robot.remove("gripper");
        for joint in robot["joints"].as_array_mut().unwrap() {
            joint.as_object_mut().unwrap().remove("role");
        }
        let restored = from_json(&value.to_string()).unwrap();
        assert_eq!(restored.robot.axis_count(), 6);
        assert!(restored.robot.gripper.is_none());
    }

    #[test]
    fn an_incoherent_project_is_rejected_on_read_and_on_write() {
        let mut project = templates::pick_and_place_cell().unwrap();
        let text = to_json(&project)
            .unwrap()
            .replace("\"mass\": 0.18", "\"mass\": -1.0");
        assert!(matches!(
            from_json(&text).unwrap_err(),
            ProjectIoError::Invalid(_)
        ));
        project.robot.root_link_id = "ghost".to_owned();
        assert!(matches!(
            to_json(&project).unwrap_err(),
            ProjectIoError::Invalid(_)
        ));
    }

    #[test]
    fn saving_replaces_the_file_and_leaves_no_temporary_behind() {
        let dir = scratch("save");
        let path = dir.join("cell.axisforge");
        let project = templates::pick_and_place_cell().unwrap();
        save(&project, &path).unwrap();
        save(&project, &path).unwrap();
        assert_eq!(load(&path).unwrap(), project);
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "{names:?}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_save_leaves_the_previous_file_intact() {
        let dir = scratch("keep");
        let path = dir.join("cell.axisforge");
        let good = templates::pick_and_place_cell().unwrap();
        save(&good, &path).unwrap();
        let mut bad = good.clone();
        bad.name = String::new();
        assert!(save(&bad, &path).is_err());
        assert_eq!(load(&path).unwrap(), good);
        fs::remove_dir_all(dir).unwrap();
    }
}
