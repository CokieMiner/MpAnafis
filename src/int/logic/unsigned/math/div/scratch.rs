//! Reusable working storage for the division tower.
//!
//! Normalization, products and recursive remainders retain their allocations
//! across calls. Reciprocal construction also owns temporary integer values;
//! its refinement reuses the preceding approximation's allocation.

use super::{InternalMpUint, MulScratch, ScratchBuffer};

/// Pre-allocated scratch space for division.
#[derive(Debug, Clone)]
pub struct DivScratch {
    pub v_norm: ScratchBuffer,
    pub u_norm: ScratchBuffer,
    pub v_padded: ScratchBuffer,
    pub q_den_low: ScratchBuffer,
    pub den_pad: ScratchBuffer,
    /// Block-repair products of the divide-and-conquer recursion.
    pub recursive_product: ScratchBuffer,
    pub newton_v_norm: ScratchBuffer,
    pub newton_u_norm: ScratchBuffer,
    pub newton_r_cur: ScratchBuffer,
    pub newton_p_buf: ScratchBuffer,
    pub newton_c_buf: ScratchBuffer,
    pub mul_scratch: MulScratch,
    pub dummy_rem: InternalMpUint,
    /// Reciprocal divisor or unrequested Burnikel quotient, in disjoint branches.
    pub dummy_quot: InternalMpUint,
}

impl Default for DivScratch {
    fn default() -> Self {
        Self {
            v_norm: ScratchBuffer::acquire(0),
            u_norm: ScratchBuffer::acquire(0),
            v_padded: ScratchBuffer::acquire(0),
            q_den_low: ScratchBuffer::acquire(0),
            den_pad: ScratchBuffer::acquire(0),
            recursive_product: ScratchBuffer::acquire(0),
            newton_v_norm: ScratchBuffer::acquire(0),
            newton_u_norm: ScratchBuffer::acquire(0),
            newton_r_cur: ScratchBuffer::acquire(0),
            newton_p_buf: ScratchBuffer::acquire(0),
            newton_c_buf: ScratchBuffer::acquire(0),
            mul_scratch: MulScratch::default(),
            dummy_rem: InternalMpUint::zero(),
            dummy_quot: InternalMpUint::zero(),
        }
    }
}
