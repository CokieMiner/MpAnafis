//! Inverse DIT execution restores contiguous children before their parent stage.

#![expect(
    unsafe_code,
    reason = "Admitted power-of-two geometry proves ascending stages and disjoint worker spans"
)]

use super::{NttFloatKernels, ParallelExecutor, Prime, Stage, Transform};

const CACHE_LEAF: usize = 256;

impl Transform {
    /// Executes an admitted DIT transform, leaving scaling to the active output prefix.
    pub fn inverse<E: ParallelExecutor>(
        data: &mut [f64],
        twiddles: &[f64],
        prime: &Prime,
        kernels: &NttFloatKernels,
        budget: usize,
        executor: &E,
    ) {
        let len = data.len();
        let odd = len.trailing_zeros() & 1 != 0;
        let child_len = if odd { len >> 1 } else { len >> 2 };
        // SAFETY: child_len >= 1 and child_len-1 is the complete inverse
        // child-table prefix inside the admitted len-1 table.
        let (child_twiddles, stage) =
            unsafe { twiddles.split_at_unchecked(child_len.unchecked_sub(1)) };
        if len <= CACHE_LEAF {
            let limit = if odd { len >> 1 } else { len };
            let mut step = 4;
            let mut roots = twiddles;
            while step <= limit {
                // SAFETY: ascending stages consume limit-1 roots. Every
                // stage has 3*step/4 entries in the admitted inverse table.
                let (active, tail) =
                    unsafe { roots.split_at_unchecked((step >> 2).unchecked_mul(3)) };
                for block in data.chunks_exact_mut(step) {
                    Stage::run(block, active, kernels.dit, prime, 1, executor);
                }
                roots = tail;
                // SAFETY: step <= limit <= CACHE_LEAF, so multiplying by four fits usize.
                step = unsafe { step.unchecked_mul(4) };
            }
            if odd {
                Stage::radix2::<true>(data, stage, len, prime);
            }
            return;
        }
        if budget > 1 && len >= 32768 {
            // SAFETY: len is even; budget > 1 gives two positive sub-budgets.
            let ((left, right), left_budget, right_budget) = unsafe {
                let left_budget = budget >> 1;
                (
                    data.split_at_mut_unchecked(len >> 1),
                    left_budget,
                    budget.unchecked_sub(left_budget),
                )
            };
            executor.join(
                || {
                    for child in left.chunks_exact_mut(child_len) {
                        Self::inverse(child, child_twiddles, prime, kernels, left_budget, executor);
                    }
                },
                || {
                    for child in right.chunks_exact_mut(child_len) {
                        Self::inverse(
                            child,
                            child_twiddles,
                            prime,
                            kernels,
                            right_budget,
                            executor,
                        );
                    }
                },
            );
        } else {
            for child in data.chunks_exact_mut(child_len) {
                Self::inverse(child, child_twiddles, prime, kernels, 1, executor);
            }
        }
        if odd {
            Stage::radix2::<true>(data, stage, len, prime);
        } else {
            Stage::run(data, stage, kernels.dit, prime, budget, executor);
        }
    }
}
