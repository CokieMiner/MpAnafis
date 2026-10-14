//! Validated transform geometry and addressable buffer sizes.

use core::mem::{align_of, size_of};

use super::{
    DefaultExecutor, LIMB_BITS, Limb, MAX_PRIMES, PRIME_CATALOG, ParallelExecutor, PlannedProfile,
    Prime,
};

mod multiply;
mod workspace;

pub use multiply::{Multiplication, Ntt};
pub use workspace::NttWorkspace;

/// Immutable coefficient, transform-order, native-shape, and storage bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NttPlan {
    len_a: usize,
    len_b: usize,
    planned: PlannedProfile,
    arena_len: usize,
    mul_f64_scratch: usize,
    sqr_f64_scratch: usize,
    mul_limb_scratch: usize,
    sqr_limb_scratch: usize,
}

impl NttPlan {
    /// Plans positive operand lengths, rejecting unavailable or unaddressable geometry.
    #[must_use]
    pub fn for_multiplication(len_a: usize, len_b: usize) -> Option<Self> {
        let planned = PlannedProfile::select(len_a, len_b, LIMB_BITS)?;
        Planning::admit(len_a, len_b, planned)
    }

    /// Returns native-limb multiplication scratch, including alignment padding.
    #[must_use]
    pub const fn required_mul_scratch_limbs(&self) -> usize {
        self.mul_limb_scratch
    }

    /// Returns native-limb square scratch, including alignment padding.
    #[must_use]
    pub const fn required_sqr_scratch_limbs(&self) -> usize {
        self.sqr_limb_scratch
    }
}

/// Internal admission of coefficient geometry and complete native storage spans.
#[derive(Debug)]
pub struct Planning;

impl Planning {
    pub fn admit(len_a: usize, len_b: usize, planned: PlannedProfile) -> Option<NttPlan> {
        let arena_len = planned.profile.np.checked_mul(planned.transform_len)?;
        let mul_f64_scratch = arena_len.checked_mul(4)?;
        let sqr_f64_scratch = arena_len.checked_mul(3)?;
        let mul_bytes = mul_f64_scratch.checked_mul(size_of::<f64>())?;
        let sqr_bytes = sqr_f64_scratch.checked_mul(size_of::<f64>())?;
        let padding = align_of::<f64>()
            .saturating_sub(align_of::<Limb>())
            .div_ceil(size_of::<Limb>());
        let mul_limb_scratch = mul_bytes.div_ceil(size_of::<Limb>()).checked_add(padding)?;
        let sqr_limb_scratch = sqr_bytes.div_ceil(size_of::<Limb>()).checked_add(padding)?;
        let address_limit = usize::try_from(isize::MAX).ok()?;
        if mul_bytes > address_limit
            || mul_limb_scratch.checked_mul(size_of::<Limb>())? > address_limit
        {
            return None;
        }
        let output_len = len_a.checked_add(len_b)?;
        let _output_bits = output_len.checked_mul(LIMB_BITS)?;
        let _stream_bits = planned
            .conv_len
            .checked_mul(usize::try_from(planned.profile.chunk_bits).ok()?)?
            .checked_add(MAX_PRIMES.checked_mul(50)?.checked_next_multiple_of(64)?)?;
        if output_len.checked_mul(size_of::<Limb>())? > address_limit {
            return None;
        }
        Some(NttPlan {
            len_a,
            len_b,
            planned,
            arena_len,
            mul_f64_scratch,
            sqr_f64_scratch,
            mul_limb_scratch,
            sqr_limb_scratch,
        })
    }
}
