//! Unit handling.
//!
//! Everything stored is SI: metres, radians, seconds, kilograms. Units exist
//! only to render and parse at the edges, so changing a project's display
//! units never moves a value.

use serde::{Deserialize, Serialize};

/// How lengths are shown to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LengthUnit {
    /// Metres, the internal unit.
    #[default]
    Meters,
    /// Millimetres.
    Millimeters,
    /// Centimetres.
    Centimeters,
    /// Inches.
    Inches,
}

impl LengthUnit {
    /// How many metres one unit represents.
    pub fn meters_per_unit(self) -> f64 {
        match self {
            Self::Meters => 1.0,
            Self::Millimeters => 1.0e-3,
            Self::Centimeters => 1.0e-2,
            Self::Inches => 0.0254,
        }
    }

    /// Converts an SI length into this display unit.
    pub fn from_si(self, meters: f64) -> f64 {
        meters / self.meters_per_unit()
    }

    /// Converts a value in this unit into SI metres.
    pub fn to_si(self, value: f64) -> f64 {
        value * self.meters_per_unit()
    }

    /// Short suffix used in the UI.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Meters => "m",
            Self::Millimeters => "mm",
            Self::Centimeters => "cm",
            Self::Inches => "in",
        }
    }
}

/// How angles are shown to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AngleUnit {
    /// Degrees, the usual choice for operators.
    #[default]
    Degrees,
    /// Radians, the internal unit.
    Radians,
}

impl AngleUnit {
    /// Converts an SI angle into this display unit.
    pub fn from_si(self, radians: f64) -> f64 {
        match self {
            Self::Degrees => radians.to_degrees(),
            Self::Radians => radians,
        }
    }

    /// Converts a value in this unit into SI radians.
    pub fn to_si(self, value: f64) -> f64 {
        match self {
            Self::Degrees => value.to_radians(),
            Self::Radians => value,
        }
    }

    /// Short suffix used in the UI.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Degrees => "deg",
            Self::Radians => "rad",
        }
    }
}

/// Display preferences. Presentation only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct UnitPreferences {
    /// Length unit shown in the inspector and reports.
    pub length: LengthUnit,
    /// Angle unit shown in the inspector and reports.
    pub angle: AngleUnit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_round_trips_through_every_unit() {
        for unit in [
            LengthUnit::Meters,
            LengthUnit::Millimeters,
            LengthUnit::Centimeters,
            LengthUnit::Inches,
        ] {
            assert!(
                (unit.to_si(unit.from_si(0.4321)) - 0.4321).abs() < 1e-12,
                "{unit:?}"
            );
        }
    }

    #[test]
    fn one_hundred_eighty_degrees_is_pi_radians() {
        assert!((AngleUnit::Degrees.to_si(180.0) - std::f64::consts::PI).abs() < 1e-12);
    }
}
