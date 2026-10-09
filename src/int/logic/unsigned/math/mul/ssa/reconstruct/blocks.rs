//! Disjoint signed block accumulation followed by overlap carry resolution.

#![expect(
    unsafe_code,
    reason = "Validated reconstruction geometry bounds coefficient blocks, overlap guards, and disjoint worker arenas"
)]

use core::num::NonZeroUsize;

use crate::parallel::ParallelExecutor;

use super::{LIMB_BITS, Limb, SharedEval, SsaCarry, SsaCoefficients, SsaRing, SsaTransform};

/// Geometry of equally sized, limb-aligned coefficient blocks.
#[derive(Clone, Copy)]
pub struct ReconstructionBlocks {
    count: usize,
    coefficients: NonZeroUsize,
    span: usize,
    magnitude: NonZeroUsize,
    inner_limbs: usize,
    chunk_bits: NonZeroUsize,
    accumulator_len: NonZeroUsize,
}

impl ReconstructionBlocks {
    /// Selects blocks only when the established prefix spans multiple blocks
    /// with enough work and every block starts on a limb boundary. The complete
    /// transform sets the alignment and magnitude bound even for a short prefix.
    /// All dimension products are checked.
    pub fn new(
        len: usize,
        active: usize,
        chunk: NonZeroUsize,
        inner_bits: usize,
        workers: usize,
    ) -> Option<Self> {
        if workers <= 1 {
            return None;
        }
        let outer_bits = len.checked_mul(chunk.get())?;
        let outer_limbs = outer_bits.div_euclid(LIMB_BITS);
        let alignment_limit = 1_usize.checked_shl(outer_limbs.trailing_zeros())?;
        let limit = workers.min(len).min(alignment_limit);
        if limit < 2 {
            return None;
        }
        let block_log = limit.ilog2();
        // SAFETY: limit>=2 is a representable usize, so floor(log2(limit))
        // is strictly below usize::BITS and its power of two is representable.
        let count = unsafe { 1_usize.unchecked_shl(block_log) };
        // SAFETY: limit>=2 and count=2^floor(log2(limit))<=limit<=len,
        // so the divisor and the resulting complete-block width are positive.
        let coefficients = unsafe { NonZeroUsize::new_unchecked(len >> block_log) };
        let inner_limbs = SsaRing::mod_limbs(inner_bits);
        if active <= coefficients.get()
            || !SsaTransform::has_parallel_work(active, inner_limbs, workers)
        {
            return None;
        }
        // count is a power of two dividing outer_limbs: its exponent does
        // not exceed the trailing-zero alignment limit established above.
        let span = outer_limbs >> block_log;
        #[expect(
            clippy::as_conversions,
            reason = "a usize trailing-zero count is at most 64, representable even on 16-bit targets"
        )]
        let transform_log = len.trailing_zeros() as usize;
        let magnitude_bits = chunk.get().checked_mul(2)?.checked_add(transform_log)?;
        let magnitude = magnitude_bits.div_ceil(LIMB_BITS);
        let accumulator_len = span.checked_add(magnitude)?.checked_add(1)?;
        Some(Self {
            count,
            coefficients,
            span,
            // SAFETY: chunk>=1 gives magnitude_bits>=2, so its ceiling limb
            // count is positive; the admitted centered bound fits the inner ring.
            magnitude: unsafe { NonZeroUsize::new_unchecked(magnitude) },
            inner_limbs,
            chunk_bits: chunk,
            // SAFETY: the checked guard addition makes accumulator_len>=1.
            accumulator_len: unsafe { NonZeroUsize::new_unchecked(accumulator_len) },
        })
    }

    /// Complete live block arena, in addition to the outer biased accumulator.
    pub const fn scratch_len(self) -> usize {
        self.accumulator_len.get().saturating_mul(self.count)
    }

    /// Accumulates complete canonical coefficients in parallel, then merges
    /// disjoint block outputs in index order with explicit signed overlaps.
    ///
    /// # Safety
    /// The matrix contains a complete-coefficient prefix of the geometry's
    /// convolution; omitted coefficients are proved zero.
    /// Each coefficient has the centered magnitude bound used by construction.
    /// `acc` is the zeroed outer accumulator biased by `2^outer_bits+1`, with
    /// room for the outer data plus the inner coefficient and one carry limb.
    /// `scratch` is disjoint and covers `scratch_len()` initialized limbs.
    pub unsafe fn run<E: ParallelExecutor>(
        self,
        matrix: &[Limb],
        acc: &mut [Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        // SAFETY: the caller's admitted plan rejects a saturated scratch_len
        // and supplies the complete count*accumulator_len disjoint block arena.
        let (blocks, _) = unsafe { scratch.split_at_mut_unchecked(self.scratch_len()) };
        // SAFETY: all block and matrix partitions derive from checked geometry;
        // each leaf receives a private accumulator for streamed magnitude digits.
        unsafe {
            self.accumulate(matrix, blocks, self.count, executor);
        }
        // SAFETY: the limb-aligned inner ring has representable 4*inner_bits;
        // accumulator_len=span+magnitude+1 is positive and checked by new.
        let (cl, bias_index) = unsafe {
            (
                self.inner_limbs.unchecked_add(1),
                self.accumulator_len.get().unchecked_sub(1),
            )
        };
        let active_blocks = matrix
            .len()
            .div_euclid(cl)
            .div_ceil(self.coefficients.get());
        for (index, leaf) in blocks
            .chunks_exact(self.accumulator_len.get())
            .take(active_blocks)
            .enumerate()
        {
            // SAFETY: index<count and count*span=outer_limbs; each complete
            // leaf has accumulator_len=bias_index+1 initialized limbs.
            let (start, (digits, sign)) = unsafe {
                (
                    index.unchecked_mul(self.span),
                    leaf.split_at_unchecked(bias_index),
                )
            };
            // Each local sum S obeys |S| < B^(span+magnitude). Its stored
            // value is B^(span+magnitude)+S, so a top digit of zero denotes
            // a negative overlap carry -1; a top digit of one denotes zero.
            // Add the unsigned digits before applying that signed carry.
            // SAFETY: checked block geometry makes start <= acc.len(), so this
            // suffix contains the complete destination overlap.
            let escaped =
                unsafe { SsaCarry::add_full_in_place(acc.get_unchecked_mut(start..), digits) };
            debug_assert_eq!(
                escaped, 0,
                "the global carry arena covers each unsigned block"
            );
            // SAFETY: the leaf includes its accumulator's top limb, so the
            // split suffix is nonempty.
            if unsafe { *sign.get_unchecked(0) } == 0 {
                // SAFETY: bias_index is inside the block accumulator and the
                // checked block span keeps carry_start within the global arena.
                let borrowed = unsafe {
                    let carry_start = start.unchecked_add(bias_index);
                    SsaCarry::propagate_borrow(acc.get_unchecked_mut(carry_start..))
                };
                // After each complete block the accumulator is q+1 plus an
                // index-order convolution prefix. Total negative magnitude
                // is below q, so every signed overlap resolves internally.
                debug_assert!(!borrowed, "the outer bias bounds every block prefix");
            }
        }
    }

    /// Forks independent blocks without sharing their writes or carry chains.
    ///
    /// # Safety
    /// `leaves` is a power of two. Scratch contains that many complete blocks.
    /// The matrix contains a nonempty complete-coefficient prefix of their
    /// convolution, with the coefficient and arena widths held by self.
    unsafe fn accumulate<E: ParallelExecutor>(
        &self,
        matrix: &[Limb],
        scratch: &mut [Limb],
        leaves: usize,
        executor: &E,
    ) {
        if leaves == 1 {
            let acc = scratch;
            // SAFETY: accumulator_len is positive by ReconstructionBlocks::new,
            // and the leaf owns exactly that many initialized limbs. Its top
            // digit is the bias; only the magnitude prefix needs zeroing.
            unsafe {
                let last = self.accumulator_len.get().unchecked_sub(1);
                acc.get_unchecked_mut(..last).fill(0);
                *acc.get_unchecked_mut(last) = 1;
            }
            // With R=2^chunk_bits and |c_i|<B^magnitude, each local prefix
            // has magnitude < B^magnitude * sum(R^i, i<coefficients)
            // <= B^magnitude * R^coefficients = B^(magnitude+span).
            // Biasing by that bound prevents underflow, and one top limb
            // stores the entire positive result below twice the bound.
            // SAFETY: inner_bits has representable 4*inner_bits, so its data
            // width plus one guard fits usize on both SSA pointer widths.
            let cl = unsafe { self.inner_limbs.unchecked_add(1) };
            for (index, coefficient) in matrix.chunks_exact(cl).enumerate() {
                // SAFETY: index<coefficients and coefficients*chunk_bits is
                // this block's checked span*LIMB_BITS <= outer_bits.
                let shift = unsafe { index.unchecked_mul(self.chunk_bits.get()) };
                let limbs = shift.div_euclid(LIMB_BITS);
                // SAFETY: the remainder is below LIMB_BITS, which is at most
                // 64 on every supported target and therefore fits in u32.
                let bits = unsafe { shift.rem_euclid(LIMB_BITS).try_into().unwrap_unchecked() };
                // SAFETY: each initialized coefficient is complete; both sign
                // digits and its bound magnitude prefix are within the data.
                let negative = unsafe {
                    *coefficient.get_unchecked(self.inner_limbs) != 0
                        || coefficient.get_unchecked(self.inner_limbs.unchecked_sub(1))
                            >> (Limb::BITS - 1)
                            != 0
                };
                // SAFETY: the checked local arena includes the shifted full
                // magnitude and a carry limb. Magnitude digits feed it directly.
                // The local bias proves every signed prefix remains positive.
                unsafe {
                    if negative {
                        SsaCoefficients::shift_sub_magnitude_run(
                            acc,
                            limbs,
                            coefficient,
                            self.magnitude,
                            self.inner_limbs,
                            bits,
                        );
                    } else {
                        let active = SharedEval::active_len(
                            coefficient.get_unchecked(..self.magnitude.get()),
                        );
                        if let Some(active_len) = NonZeroUsize::new(active) {
                            SsaCoefficients::process_positive_coeff(
                                coefficient,
                                active_len,
                                limbs,
                                bits,
                                acc,
                            );
                        }
                    }
                }
            }
            return;
        }
        let half = leaves >> 1;
        // SAFETY: leaves<=count; complete matrix and block arenas were checked
        // by the parent plan and new. These half spans are bounded by them.
        let (matrix_span, (left_work, right_work)) = unsafe {
            let matrix_span = half
                .unchecked_mul(self.coefficients.get())
                .unchecked_mul(self.inner_limbs.unchecked_add(1));
            let arena_span = half.unchecked_mul(self.accumulator_len.get());
            (matrix_span, scratch.split_at_mut_unchecked(arena_span))
        };
        if matrix.len() <= matrix_span {
            // SAFETY: the complete established prefix lies in the left child.
            // Empty right blocks need no initialization and are never merged.
            unsafe {
                self.accumulate(matrix, left_work, half, executor);
            }
            return;
        }
        // SAFETY: the preceding branch handles prefixes contained in the left
        // child, so matrix_span is strictly below matrix.len() here.
        let (left, right) = unsafe { matrix.split_at_unchecked(matrix_span) };
        let ((), ()) = executor.join(
            // SAFETY: exact block partitions preserve initialized disjoint arenas.
            || unsafe {
                self.accumulate(left, left_work, half, executor);
            },
            // SAFETY: the right block partitions satisfy the identical bounds.
            || unsafe {
                self.accumulate(right, right_work, half, executor);
            },
        );
    }
}
