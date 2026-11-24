//! The extension registry: how other libraries plug into the planner.
//!
//! The planner does not know which quality measure or which candidate
//! generator it is running; it asks this registry for them by name. The
//! built-in ones are registered by [`Registry::standard`], and a host adds its
//! own with [`Registry::register_measure`] and
//! [`Registry::register_generator`]. A crate wrapping another grasp library —
//! a learned grasp scorer, a sampler for mesh parts — implements one of the two
//! traits and registers it; nothing in the planner changes.
//!
//! Names are unique. Registering a second extension under a name already
//! taken is refused, so one library cannot silently replace another's.

use std::collections::BTreeMap;
use std::sync::Arc;

use axisforge_domain::{DomainError, DomainResult, Resource, WorldBody};
use serde::{Deserialize, Serialize};

use crate::candidate::{synthesize, GraspCandidate, SynthesisOptions};
use crate::quality::{AntipodalMeasure, QualityMeasure, WrenchSpaceMeasure};
use crate::surface::synthesize_surface;

/// A way of proposing grasps for a part. Implement this to plug in another library.
pub trait CandidateGenerator: Send + Sync {
    /// The name the generator is selected by.
    fn id(&self) -> &'static str;
    /// One line saying what it generates, for listings.
    fn summary(&self) -> &'static str;
    /// Proposes grasps for `body`, in world coordinates. `resources` are
    /// the project's meshes, for bodies that refer to them.
    ///
    /// # Errors
    /// When the generator does not support the body.
    fn generate(
        &self,
        body: &WorldBody,
        resources: &[Resource],
        options: &SynthesisOptions,
    ) -> DomainResult<Vec<GraspCandidate>>;
}

/// The built-in generator: antipodal grasps of primitive shapes.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimitiveAntipodalGenerator;

impl CandidateGenerator for PrimitiveAntipodalGenerator {
    fn id(&self) -> &'static str {
        "primitive_antipodal"
    }

    fn summary(&self) -> &'static str {
        "Antipodal two-finger grasps of boxes, cylinders, capsules and spheres, enumerated \
         from their dimensions."
    }

    fn generate(
        &self,
        body: &WorldBody,
        _resources: &[Resource],
        options: &SynthesisOptions,
    ) -> DomainResult<Vec<GraspCandidate>> {
        synthesize(body, options)
    }
}

/// Grasps of any part, sampled on its surface: meshes, assembled parts, and
/// primitives alike. See [`synthesize_surface`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SurfaceNormalGenerator;

impl CandidateGenerator for SurfaceNormalGenerator {
    fn id(&self) -> &'static str {
        "surface_normal"
    }

    fn summary(&self) -> &'static str {
        "Points sampled on the part's surface; the jaws close along each point's normal,          approached from several directions, and the contacts are where the pads stop.          Works on meshes and assembled parts."
    }

    fn generate(
        &self,
        body: &WorldBody,
        resources: &[Resource],
        options: &SynthesisOptions,
    ) -> DomainResult<Vec<GraspCandidate>> {
        synthesize_surface(body, resources, options)
    }
}

/// What kind of extension a registry entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionKind {
    /// A [`QualityMeasure`].
    QualityMeasure,
    /// A [`CandidateGenerator`].
    CandidateGenerator,
}

/// One registered extension, for listings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionInfo {
    /// What it is.
    pub kind: ExtensionKind,
    /// The name it is selected by.
    pub id: String,
    /// What it does.
    pub summary: String,
}

/// The quality measures and candidate generators available to the planner.
#[derive(Clone, Default)]
pub struct Registry {
    measures: BTreeMap<String, Arc<dyn QualityMeasure>>,
    generators: BTreeMap<String, Arc<dyn CandidateGenerator>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("measures", &self.measures.keys().collect::<Vec<_>>())
            .field("generators", &self.generators.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn check_id(id: &str) -> DomainResult<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(DomainError::unsupported(format!(
            "extension id '{id}' must be non-empty lower-case letters, digits and underscores"
        )));
    }
    Ok(())
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding the built-in measures and generators.
    pub fn standard() -> Self {
        let mut registry = Self::new();
        // The built-in ids are valid and distinct, so these cannot fail.
        let _ = registry.register_measure(Arc::new(WrenchSpaceMeasure));
        let _ = registry.register_measure(Arc::new(AntipodalMeasure));
        let _ = registry.register_generator(Arc::new(PrimitiveAntipodalGenerator));
        let _ = registry.register_generator(Arc::new(SurfaceNormalGenerator));
        registry
    }

    /// Adds a quality measure.
    ///
    /// # Errors
    /// When its id is malformed or already taken.
    pub fn register_measure(&mut self, measure: Arc<dyn QualityMeasure>) -> DomainResult<()> {
        let id = measure.id().to_owned();
        check_id(&id)?;
        if self.measures.contains_key(&id) {
            return Err(DomainError::unsupported(format!(
                "a quality measure named '{id}' is already registered"
            )));
        }
        self.measures.insert(id, measure);
        Ok(())
    }

    /// Adds a candidate generator.
    ///
    /// # Errors
    /// When its id is malformed or already taken.
    pub fn register_generator(
        &mut self,
        generator: Arc<dyn CandidateGenerator>,
    ) -> DomainResult<()> {
        let id = generator.id().to_owned();
        check_id(&id)?;
        if self.generators.contains_key(&id) {
            return Err(DomainError::unsupported(format!(
                "a candidate generator named '{id}' is already registered"
            )));
        }
        self.generators.insert(id, generator);
        Ok(())
    }

    /// Looks up a quality measure.
    ///
    /// # Errors
    /// When no measure has that name; the message lists the ones that exist.
    pub fn measure(&self, id: &str) -> DomainResult<&dyn QualityMeasure> {
        self.measures.get(id).map(AsRef::as_ref).ok_or_else(|| {
            DomainError::unsupported(format!(
                "no quality measure named '{id}'; available: {}",
                self.measures.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })
    }

    /// Looks up a candidate generator.
    ///
    /// # Errors
    /// When no generator has that name; the message lists the ones that exist.
    pub fn generator(&self, id: &str) -> DomainResult<&dyn CandidateGenerator> {
        self.generators.get(id).map(AsRef::as_ref).ok_or_else(|| {
            DomainError::unsupported(format!(
                "no candidate generator named '{id}'; available: {}",
                self.generators
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }

    /// Every registered extension, measures first, each group by name.
    pub fn list(&self) -> Vec<ExtensionInfo> {
        let measures = self.measures.values().map(|m| ExtensionInfo {
            kind: ExtensionKind::QualityMeasure,
            id: m.id().to_owned(),
            summary: m.summary().to_owned(),
        });
        let generators = self.generators.values().map(|g| ExtensionInfo {
            kind: ExtensionKind::CandidateGenerator,
            id: g.id().to_owned(),
            summary: g.summary().to_owned(),
        });
        measures.chain(generators).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::{Measurement, QualityInput};

    struct Constant;
    impl QualityMeasure for Constant {
        fn id(&self) -> &'static str {
            "constant"
        }
        fn summary(&self) -> &'static str {
            "Always 0.5."
        }
        fn measure(&self, _: &QualityInput<'_>) -> Measurement {
            Measurement {
                quality: 0.5,
                normalized: 0.5,
                force_closure: true,
            }
        }
    }

    #[test]
    fn the_standard_registry_lists_the_built_ins() {
        let ids: Vec<String> = Registry::standard()
            .list()
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(
            ids,
            [
                "antipodal",
                "wrench_space",
                "primitive_antipodal",
                "surface_normal"
            ]
        );
    }

    #[test]
    fn a_library_can_register_its_own_measure() {
        let mut registry = Registry::standard();
        registry.register_measure(Arc::new(Constant)).unwrap();
        assert_eq!(
            registry.measure("constant").unwrap().summary(),
            "Always 0.5."
        );
    }

    #[test]
    fn a_name_cannot_be_taken_twice() {
        let mut registry = Registry::standard();
        registry.register_measure(Arc::new(Constant)).unwrap();
        let error = registry.register_measure(Arc::new(Constant)).unwrap_err();
        assert!(error.to_string().contains("already registered"), "{error}");
        assert!(registry
            .register_measure(Arc::new(WrenchSpaceMeasure))
            .is_err());
    }

    #[test]
    fn an_unknown_name_lists_what_exists() {
        let registry = Registry::standard();
        let error = registry.measure("nope").err().unwrap();
        assert!(error.to_string().contains("wrench_space"), "{error}");
        let error = registry.generator("nope").err().unwrap();
        assert!(error.to_string().contains("primitive_antipodal"), "{error}");
    }
}
