//! Axis-aligned bounding boxes, the broad phase of collision checking.

use axisforge_domain::{DomainError, DomainResult, Transform, Vec3};

/// An axis-aligned box in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Corner with the smallest coordinate on every axis.
    pub min: Vec3,
    /// Corner with the largest coordinate on every axis.
    pub max: Vec3,
}

impl Aabb {
    /// A box containing exactly one point.
    pub fn from_point(point: Vec3) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// The smallest box containing every point given.
    ///
    /// # Errors
    /// [`DomainError::ZeroLengthVector`] when `points` is empty, since an empty
    /// set has no bounds and silently returning a zero box would make an empty
    /// body look like one sitting at the origin.
    pub fn from_points(points: &[Vec3]) -> DomainResult<Self> {
        let mut iter = points.iter();
        let first = iter.next().ok_or(DomainError::ZeroLengthVector {
            context: "Aabb::from_points received no points".to_owned(),
        })?;
        let mut bounds = Self::from_point(*first);
        for point in iter {
            bounds = bounds.including(*point);
        }
        Ok(bounds)
    }

    /// This box grown to contain `point`.
    pub fn including(&self, point: Vec3) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            ),
            max: Vec3::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            ),
        }
    }

    /// This box grown to contain `other`.
    pub fn union(&self, other: &Self) -> Self {
        self.including(other.min).including(other.max)
    }

    /// This box grown by `margin` on every side.
    pub fn expanded(&self, margin: f64) -> Self {
        let delta = Vec3::new(margin, margin, margin);
        Self {
            min: self.min.sub(&delta),
            max: self.max.add(&delta),
        }
    }

    /// Centre of the box.
    pub fn center(&self) -> Vec3 {
        self.min.add(&self.max).scaled(0.5)
    }

    /// Full extents along each axis.
    pub fn size(&self) -> Vec3 {
        self.max.sub(&self.min)
    }

    /// Half-extents along each axis.
    pub fn half_extents(&self) -> Vec3 {
        self.size().scaled(0.5)
    }

    /// Length of the box's main diagonal.
    pub fn diagonal(&self) -> f64 {
        self.size().norm()
    }

    /// `true` when the two boxes overlap or touch, allowing `margin` of slack.
    pub fn intersects(&self, other: &Self, margin: f64) -> bool {
        self.min.x - margin <= other.max.x
            && self.max.x + margin >= other.min.x
            && self.min.y - margin <= other.max.y
            && self.max.y + margin >= other.min.y
            && self.min.z - margin <= other.max.z
            && self.max.z + margin >= other.min.z
    }

    /// `true` when `point` lies inside or on the boundary.
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// `point` clamped into the box.
    pub fn clamp(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            point.x.clamp(self.min.x, self.max.x),
            point.y.clamp(self.min.y, self.max.y),
            point.z.clamp(self.min.z, self.max.z),
        )
    }

    /// Lower bound on the distance between the two boxes; zero when they overlap.
    ///
    /// This is a *bound*, not the distance between the bodies inside them: it
    /// is what lets the broad phase discard a pair without running GJK.
    pub fn distance(&self, other: &Self) -> f64 {
        let gap = |a_min: f64, a_max: f64, b_min: f64, b_max: f64| -> f64 {
            if a_max < b_min {
                b_min - a_max
            } else if b_max < a_min {
                a_min - b_max
            } else {
                0.0
            }
        };
        let dx = gap(self.min.x, self.max.x, other.min.x, other.max.x);
        let dy = gap(self.min.y, self.max.y, other.min.y, other.max.y);
        let dz = gap(self.min.z, self.max.z, other.min.z, other.max.z);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// The smallest world-aligned box containing a local box under `pose`.
    ///
    /// Each world axis is bounded by the sum of the absolute projections of the
    /// local half-extents onto it, so a rotated unit cube spans `sqrt(2)`
    /// across the axes it was rotated about rather than staying at 1.
    pub fn from_local_half_extents(pose: &Transform, half_extents: Vec3) -> Self {
        let x = pose.apply_direction(&Vec3::X);
        let y = pose.apply_direction(&Vec3::Y);
        let z = pose.apply_direction(&Vec3::Z);
        let radius = Vec3::new(
            half_extents.x * x.x.abs() + half_extents.y * y.x.abs() + half_extents.z * z.x.abs(),
            half_extents.x * x.y.abs() + half_extents.y * y.y.abs() + half_extents.z * z.y.abs(),
            half_extents.x * x.z.abs() + half_extents.y * y.z.abs() + half_extents.z * z.z.abs(),
        );
        Self {
            min: pose.translation.sub(&radius),
            max: pose.translation.add(&radius),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axisforge_domain::Quat;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn an_empty_point_set_has_no_bounds() {
        assert!(Aabb::from_points(&[]).is_err());
    }

    #[test]
    fn a_rotated_unit_cube_spans_the_square_root_of_two() {
        let pose = Transform::from_rotation(Quat::from_axis_angle(&Vec3::Z, FRAC_PI_4).unwrap());
        let bounds = Aabb::from_local_half_extents(&pose, Vec3::new(0.5, 0.5, 0.5));
        let size = bounds.size();
        assert!((size.x - 2.0_f64.sqrt()).abs() < 1e-12, "{size:?}");
        assert!((size.y - 2.0_f64.sqrt()).abs() < 1e-12, "{size:?}");
        assert!(
            (size.z - 1.0).abs() < 1e-12,
            "the rotation axis is unchanged: {size:?}"
        );
    }

    #[test]
    fn separated_boxes_report_their_gap() {
        let a = Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let b = Aabb {
            min: Vec3::new(4.0, 0.0, 0.0),
            max: Vec3::new(5.0, 1.0, 1.0),
        };
        assert!((a.distance(&b) - 3.0).abs() < 1e-12);
        assert!(!a.intersects(&b, 0.0));
        assert!(
            a.intersects(&b, 3.5),
            "a generous margin should bring them together"
        );
    }

    #[test]
    fn overlapping_boxes_report_zero_distance() {
        let a = Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let b = Aabb {
            min: Vec3::new(0.5, 0.5, 0.5),
            max: Vec3::new(2.0, 2.0, 2.0),
        };
        assert!(a.distance(&b) < 1e-12);
        assert!(a.intersects(&b, 0.0));
    }

    #[test]
    fn the_diagonal_gap_is_euclidean_not_axis_wise() {
        let a = Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let b = Aabb {
            min: Vec3::new(4.0, 5.0, 1.0),
            max: Vec3::new(5.0, 6.0, 2.0),
        };
        // 3 along x, 4 along y, touching in z.
        assert!((a.distance(&b) - 5.0).abs() < 1e-12);
    }
}
