//! Reusable forced-tier squaring tuner.

use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Ssa, Toom3, Toom4, Toom6, Toom8,
    TransformChoice,
};

/// Root squaring tier measured by [`Tuner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Algorithm {
    /// Quadratic schoolbook squaring.
    Schoolbook,
    /// One forced Karatsuba square level with normal child dispatch.
    Karatsuba,
    /// One forced Toom-Cook 3 square level with normal child dispatch.
    ToomCook3,
    /// One Toom-Cook 4 square level with normal child dispatch.
    ToomCook4,
    /// One Toom-Cook 6 square level with normal child dispatch.
    ToomCook6,
    /// One Toom-Cook 8/8.5 square level with normal child dispatch.
    ToomCook85,
    /// Schonhage-Strassen squaring.
    #[cfg(not(target_pointer_width = "16"))]
    Ssa,
}

/// Allocation-free reusable state for one squaring crossover sample.
#[derive(Debug)]
pub struct Tuner {
    algorithm: Algorithm,
    len: usize,
    scratch: ScratchBuffer,
}

impl Tuner {
    /// Pre-allocates the exact scratch required for `algorithm` at this width.
    ///
    /// # Panics
    ///
    /// Panics if the operand width is zero or SSA cannot represent the
    /// requested square width.
    #[must_use]
    pub fn new(algorithm: Algorithm, len: usize) -> Self {
        assert!(len != 0, "squaring tuner operand must be nonzero-width");
        #[cfg(not(target_pointer_width = "16"))]
        if algorithm == Algorithm::Ssa {
            assert!(
                Ssa::admits_sqr(len),
                "SSA cannot represent the requested tuning width"
            );
        }
        let scratch_len = match algorithm {
            Algorithm::Schoolbook => 0,
            Algorithm::Karatsuba => Multiplication::karatsuba_sqr_forced_scratch_len(len),
            Algorithm::ToomCook3 => Multiplication::toom3_sqr_forced_scratch_len(len),
            Algorithm::ToomCook4 => Multiplication::toom4_sqr_scratch_len(len),
            Algorithm::ToomCook6 => Multiplication::toom6_sqr_scratch_len(len),
            Algorithm::ToomCook85 => Multiplication::toom8_sqr_scratch_len(len),
            #[cfg(not(target_pointer_width = "16"))]
            Algorithm::Ssa => Ssa::sqr_scratch_len(len),
        };
        let mut scratch = ScratchBuffer::acquire(scratch_len);
        // SAFETY: forced tiers write or clear every scratch region before
        // reading it, and their sizing functions include child workspaces.
        unsafe {
            scratch.set_len(scratch_len);
        }
        Self {
            algorithm,
            len,
            scratch,
        }
    }

    /// Squares with the configured root tier without caller-side allocation.
    ///
    /// # Panics
    ///
    /// Panics if the operand or destination width differs from construction,
    /// or if a transform rejects the measured shape.
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb]) {
        debug_assert_eq!(a.len(), self.len, "tuner operand width changed");
        debug_assert_eq!(
            dst.len(),
            self.len.wrapping_mul(2),
            "tuner destination width changed"
        );
        dst.fill(0);
        match self.algorithm {
            Algorithm::Schoolbook => Schoolbook::sqr(dst, a),
            Algorithm::Karatsuba => Karatsuba::sqr_forced(dst, a, &mut self.scratch),
            Algorithm::ToomCook3 => {
                Toom3::sqr_forced(dst, a, &mut self.scratch);
            }
            Algorithm::ToomCook4 => Toom4::sqr(dst, a, &mut self.scratch),
            Algorithm::ToomCook6 => Toom6::sqr(dst, a, &mut self.scratch),
            Algorithm::ToomCook85 => Toom8::sqr(dst, a, &mut self.scratch),
            #[cfg(not(target_pointer_width = "16"))]
            Algorithm::Ssa => {
                assert!(
                    Ssa::try_sqr(dst, a, TransformChoice::PLANNED, Some(&mut self.scratch)),
                    "SSA rejected the tuning shape"
                );
            }
        }
    }
}
