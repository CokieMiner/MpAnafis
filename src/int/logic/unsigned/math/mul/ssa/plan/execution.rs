//! Retained coefficient execution trees for prepared transforms.

use core::ops::Deref;

use alloc::borrow::Cow;

use super::{FftPlan, PointwiseMulPlan, PointwiseSquarePlan};

/// Multiplication geometry with its complete sequential coefficient subtree.
#[derive(Clone, Debug)]
pub struct MulTransformPlan {
    geometry: FftPlan,
    pointwise: PointwiseMulPlan,
}

/// Square geometry with its complete sequential coefficient subtree.
#[derive(Clone, Debug)]
pub struct SquareTransformPlan {
    geometry: FftPlan,
    pointwise: PointwiseSquarePlan,
}

/// A prepared execution tree or a geometry-only request at a transform boundary.
pub enum MulTransformInput<'plan> {
    Geometry(Option<&'plan FftPlan>),
    Prepared(&'plan MulTransformPlan),
}

/// The square counterpart of a transform boundary request.
pub enum SquareTransformInput<'plan> {
    Geometry(Option<&'plan FftPlan>),
    Prepared(&'plan SquareTransformPlan),
}

impl MulTransformPlan {
    /// Borrows the immutable preselected coefficient strategy.
    pub const fn pointwise(&self) -> &PointwiseMulPlan {
        &self.pointwise
    }

    /// Builds all sequential coefficient descendants outside execution.
    pub fn new(geometry: FftPlan) -> Self {
        Self {
            pointwise: PointwiseMulPlan::from(geometry.inner_bits),
            geometry,
        }
    }

    /// Sizes the retained tree without constructing any descendant plans.
    pub fn transform_mul_scratch(&self, workers: usize) -> usize {
        self.geometry
            .transform_mul_scratch_with_pointwise(workers, self.pointwise.scratch_len)
    }
}

impl SquareTransformPlan {
    /// Borrows the immutable preselected square strategy.
    pub const fn pointwise(&self) -> &PointwiseSquarePlan {
        &self.pointwise
    }

    /// Builds all square descendants outside execution.
    pub fn new(geometry: FftPlan) -> Self {
        Self {
            pointwise: PointwiseSquarePlan::from(geometry.inner_bits),
            geometry,
        }
    }

    /// Sizes the retained square tree without reconstructing descendants.
    pub fn transform_sqr_scratch(&self, workers: usize) -> usize {
        self.geometry
            .transform_sqr_scratch_with_pointwise(workers, self.pointwise.scratch_len)
    }
}

impl Deref for MulTransformPlan {
    type Target = FftPlan;
    fn deref(&self) -> &FftPlan {
        &self.geometry
    }
}

impl Deref for SquareTransformPlan {
    type Target = FftPlan;
    fn deref(&self) -> &FftPlan {
        &self.geometry
    }
}

impl<'plan> From<Option<&'plan FftPlan>> for MulTransformInput<'plan> {
    fn from(plan: Option<&'plan FftPlan>) -> Self {
        Self::Geometry(plan)
    }
}

impl<'plan> From<&'plan MulTransformPlan> for MulTransformInput<'plan> {
    fn from(plan: &'plan MulTransformPlan) -> Self {
        Self::Prepared(plan)
    }
}

impl<'plan> From<Option<&'plan FftPlan>> for SquareTransformInput<'plan> {
    fn from(plan: Option<&'plan FftPlan>) -> Self {
        Self::Geometry(plan)
    }
}

impl<'plan> From<&'plan SquareTransformPlan> for SquareTransformInput<'plan> {
    fn from(plan: &'plan SquareTransformPlan) -> Self {
        Self::Prepared(plan)
    }
}

impl<'plan> MulTransformInput<'plan> {
    /// Geometry-only calls prepare once; retained calls borrow their entire tree.
    pub fn resolve(self, bits: usize, pair: bool) -> Cow<'plan, MulTransformPlan> {
        match self {
            Self::Prepared(plan) => Cow::Borrowed(plan),
            Self::Geometry(geometry) => Cow::Owned(MulTransformPlan::new(
                geometry.copied().unwrap_or_else(|| {
                    if pair {
                        FftPlan::new_for_pair(bits)
                    } else {
                        FftPlan::new(bits)
                    }
                }),
            )),
        }
    }
}

impl<'plan> SquareTransformInput<'plan> {
    /// Resolves a square-specific tree at the outer transform boundary.
    pub fn resolve(self, bits: usize) -> Cow<'plan, SquareTransformPlan> {
        match self {
            Self::Prepared(plan) => Cow::Borrowed(plan),
            Self::Geometry(geometry) => Cow::Owned(SquareTransformPlan::new(
                geometry
                    .copied()
                    .unwrap_or_else(|| FftPlan::new_for_square(bits)),
            )),
        }
    }
}
