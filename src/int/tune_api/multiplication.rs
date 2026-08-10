//! Reusable forced-tier multiplication tuner.

use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Ssa, Toom3, Toom4, Toom6, Toom8,
    TransformChoice,
};

/// Root multiplication tier measured by [`Tuner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Algorithm {
    /// Quadratic schoolbook multiplication.
    Schoolbook,
    /// One forced Karatsuba level with normal child dispatch.
    Karatsuba,
    /// One forced Toom-Cook 3 level with normal child dispatch.
    ToomCook3,
    /// One Toom-Cook 4 level with normal child dispatch.
    ToomCook4,
    /// One Toom-Cook 6/6.5 level with normal child dispatch.
    ToomCook6,
    /// One forced Toom-Cook 8.5 level with normal child dispatch.
    ToomCook85,
    /// Schonhage-Strassen exact multiplication.
    #[cfg(not(target_pointer_width = "16"))]
    Ssa,
}

/// Allocation-free reusable state for one multiplication crossover sample.
#[derive(Debug)]
pub struct Tuner {
    algorithm: Algorithm,
    len_a: usize,
    len_b: usize,
    scratch: ScratchBuffer,
}

impl Tuner {
    /// Pre-allocates the exact scratch required for `algorithm` at this shape.
    ///
    /// # Panics
    ///
    /// Panics if either operand width is zero or SSA cannot represent the
    /// requested product width.
    #[must_use]
    pub fn new(algorithm: Algorithm, len_a: usize, len_b: usize) -> Self {
        assert!(
            len_a != 0 && len_b != 0,
            "multiplication tuner operands must be nonzero-width"
        );
        #[cfg(not(target_pointer_width = "16"))]
        if algorithm == Algorithm::Ssa {
            assert!(
                Ssa::admits_mul(len_a, len_b),
                "SSA cannot represent the requested tuning shape"
            );
        }
        let scratch_len = match algorithm {
            Algorithm::Schoolbook => 0,
            Algorithm::Karatsuba => Multiplication::karatsuba_mul_forced_scratch_len(len_a, len_b),
            Algorithm::ToomCook3 => Multiplication::toom3_mul_forced_scratch_len(len_a, len_b),
            Algorithm::ToomCook4 => Multiplication::toom4_mul_scratch_len(len_a, len_b),
            Algorithm::ToomCook6 => Multiplication::toom6_mul_scratch_len(len_a, len_b),
            Algorithm::ToomCook85 => Multiplication::toom8_mul_scratch_len(len_a, len_b),
            #[cfg(not(target_pointer_width = "16"))]
            Algorithm::Ssa => Ssa::mul_scratch_len(len_a, len_b),
        };
        let mut scratch = ScratchBuffer::acquire(scratch_len);
        // SAFETY: forced tiers write or clear every scratch region before
        // reading it, and their sizing functions include child workspaces.
        unsafe {
            scratch.set_len(scratch_len);
        }
        Self {
            algorithm,
            len_a,
            len_b,
            scratch,
        }
    }

    /// Multiplies with the configured root tier without caller-side allocation.
    ///
    /// # Panics
    ///
    /// Panics if an operand or destination width differs from construction, or
    /// if a transform rejects the measured shape.
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        debug_assert_eq!(a.len(), self.len_a, "left tuner operand width changed");
        debug_assert_eq!(b.len(), self.len_b, "right tuner operand width changed");
        debug_assert_eq!(
            dst.len(),
            self.len_a.wrapping_add(self.len_b),
            "tuner destination width changed"
        );
        match self.algorithm {
            Algorithm::Schoolbook => Schoolbook::mul(dst, a, b),
            Algorithm::Karatsuba => Karatsuba::mul_forced(dst, a, b, &mut self.scratch),
            Algorithm::ToomCook3 => {
                Toom3::mul_forced(dst, a, b, &mut self.scratch);
            }
            Algorithm::ToomCook4 => Toom4::mul(dst, a, b, &mut self.scratch),
            Algorithm::ToomCook6 => Toom6::mul(dst, a, b, &mut self.scratch),
            Algorithm::ToomCook85 => Toom8::mul(dst, a, b, &mut self.scratch),
            #[cfg(not(target_pointer_width = "16"))]
            Algorithm::Ssa => {
                assert!(
                    Ssa::try_mul(dst, a, b, TransformChoice::PLANNED, Some(&mut self.scratch)),
                    "SSA rejected the tuning shape"
                );
            }
        }
    }
}
