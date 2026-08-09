//! Reusable working storage for the division tower.
//!
//! Every divider in this module tree writes its temporaries into one of these
//! buffers rather than allocating per call. Callers that divide repeatedly with
//! the same operand shapes therefore pay the allocation once.

use super::{InternalMpUint, MulScratch, ScratchBuffer};

/// Pre-allocated scratch space for division.
#[derive(Debug, Clone)]
pub struct DivScratch {
    pub v_norm: ScratchBuffer,
    pub u_norm: ScratchBuffer,
    pub v_padded: ScratchBuffer,
    pub q_den_low: ScratchBuffer,
    pub den_pad: ScratchBuffer,
    pub bz_d: ScratchBuffer,
    pub bz_r1: ScratchBuffer,
    pub bz_a_pad: ScratchBuffer,
    pub bz_q: ScratchBuffer,
    pub bz_r0: ScratchBuffer,
    pub bz_r1_iter: ScratchBuffer,
    pub bz_rem_final: ScratchBuffer,
    pub newton_v_norm: ScratchBuffer,
    pub newton_u_norm: ScratchBuffer,
    pub newton_a_pad: ScratchBuffer,
    pub newton_total_quo: ScratchBuffer,
    pub newton_cur_2n: ScratchBuffer,
    pub newton_q_i: ScratchBuffer,
    pub newton_r_cur: ScratchBuffer,
    pub newton_r_next: ScratchBuffer,
    pub mul_scratch: MulScratch,
    pub dummy_rem: InternalMpUint,
    pub dummy_quot: InternalMpUint,
    pub dummy_u: InternalMpUint,
    pub mod_rem: InternalMpUint,
}

impl Default for DivScratch {
    fn default() -> Self {
        Self {
            v_norm: ScratchBuffer::acquire(0),
            u_norm: ScratchBuffer::acquire(0),
            v_padded: ScratchBuffer::acquire(0),
            q_den_low: ScratchBuffer::acquire(0),
            den_pad: ScratchBuffer::acquire(0),
            bz_d: ScratchBuffer::acquire(0),
            bz_r1: ScratchBuffer::acquire(0),
            bz_a_pad: ScratchBuffer::acquire(0),
            bz_q: ScratchBuffer::acquire(0),
            bz_r0: ScratchBuffer::acquire(0),
            bz_r1_iter: ScratchBuffer::acquire(0),
            bz_rem_final: ScratchBuffer::acquire(0),
            newton_v_norm: ScratchBuffer::acquire(0),
            newton_u_norm: ScratchBuffer::acquire(0),
            newton_a_pad: ScratchBuffer::acquire(0),
            newton_total_quo: ScratchBuffer::acquire(0),
            newton_cur_2n: ScratchBuffer::acquire(0),
            newton_q_i: ScratchBuffer::acquire(0),
            newton_r_cur: ScratchBuffer::acquire(0),
            newton_r_next: ScratchBuffer::acquire(0),
            mul_scratch: MulScratch::default(),
            dummy_rem: InternalMpUint::zero(),
            dummy_quot: InternalMpUint::zero(),
            dummy_u: InternalMpUint::zero(),
            mod_rem: InternalMpUint::zero(),
        }
    }
}
