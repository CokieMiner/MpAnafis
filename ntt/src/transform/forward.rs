//! Forward DIF stages followed by contiguous cache-resident descendants.

#![expect(
    unsafe_code,
    reason = "Admitted power-of-two geometry proves stage spans and worker partitions"
)]

use super::{NttFloatKernels, ParallelExecutor, Prime, Scalar, Stage};

const CACHE_LEAF: usize = 256;

/// Cache-blocked power-of-two transform execution.
#[derive(Debug)]
pub struct Transform;

impl Transform {
    /// Executes an admitted DIF transform with a known nonzero input prefix.
    pub fn forward<E: ParallelExecutor>(
        data: &mut [f64],
        twiddles: &[f64],
        input_len: usize,
        prime: &Prime,
        kernels: &NttFloatKernels,
        budget: usize,
        executor: &E,
    ) {
        let len = data.len();
        let odd = len.trailing_zeros() & 1 != 0;
        let child_len = if odd { len >> 1 } else { len >> 2 };
        // SAFETY: len >= 4 is an admitted power of two. Radix-2 uses len/2
        // roots; radix-4 uses 3*len/4. Both prefixes fit the len-1 root table.
        let (stage, mut descendants) = unsafe {
            twiddles.split_at_unchecked(if odd {
                child_len
            } else {
                child_len.unchecked_mul(3)
            })
        };
        if odd {
            Stage::radix2::<false>(data, stage, input_len, prime);
        } else if input_len <= len >> 1 {
            pruned_radix4(data, stage, input_len, prime);
        } else {
            Stage::run(data, stage, kernels.dif, prime, budget, executor);
        }
        if child_len < 4 {
            return;
        }
        if len <= CACHE_LEAF {
            // All descendants have even transform order. Their table stages
            // are shared by equal-length contiguous blocks inside this leaf.
            let mut step = child_len;
            while step >= 4 {
                // SAFETY: the descending geometric series uses exactly
                // child_len-1 roots, already admitted in the remaining table.
                let (roots, tail) =
                    unsafe { descendants.split_at_unchecked((step >> 2).unchecked_mul(3)) };
                for block in data.chunks_exact_mut(step) {
                    Stage::run(block, roots, kernels.dif, prime, 1, executor);
                }
                descendants = tail;
                step >>= 2;
            }
        } else if budget > 1 && len >= 32768 {
            // SAFETY: len is even, and budget > 1 gives two positive budgets.
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
                        Self::forward(
                            child,
                            descendants,
                            child_len,
                            prime,
                            kernels,
                            left_budget,
                            executor,
                        );
                    }
                },
                || {
                    for child in right.chunks_exact_mut(child_len) {
                        Self::forward(
                            child,
                            descendants,
                            child_len,
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
                Self::forward(child, descendants, child_len, prime, kernels, 1, executor);
            }
        }
    }
}

fn pruned_radix4(data: &mut [f64], twiddles: &[f64], input_len: usize, prime: &Prime) {
    let quarter = data.len() >> 2;
    let modulus = prime.float();
    let reciprocal = prime.reciprocal();
    // SAFETY: data has four disjoint quarters and the admitted stage has
    // three twiddle quarters. Read slots outside input_len are zero; the
    // omitted two or three quarters are overwritten before any descendant reads them.
    let (first, second, third, fourth, w0, w1, w2) = unsafe {
        let (first, tail) = data.split_at_mut_unchecked(quarter);
        let (second, final_half) = tail.split_at_mut_unchecked(quarter);
        let (third, fourth) = final_half.split_at_mut_unchecked(quarter);
        let (w0, roots) = twiddles.split_at_unchecked(quarter);
        let (w1, w2) = roots.split_at_unchecked(quarter);
        (first, second, third, fourth, w0, w1, w2)
    };
    if input_len <= quarter {
        // Three zero quarters reduce each butterfly to three products.
        for (((((a, b), c), d), &root), &square_root) in first
            .iter_mut()
            .zip(second)
            .zip(third)
            .zip(fourth)
            .zip(w0)
            .zip(w2)
        {
            let value = Scalar::reduce(*a, modulus, reciprocal);
            let rotated = Scalar::mulmod(value, root, modulus, reciprocal);
            *a = value;
            *b = Scalar::mulmod(value, square_root, modulus, reciprocal);
            *c = rotated;
            *d = Scalar::mulmod(rotated, square_root, modulus, reciprocal);
        }
    } else {
        // Two zero quarters remove their contributions before evaluation.
        for ((((((a, b), c), d), &root), &shifted_root), &square_root) in first
            .iter_mut()
            .zip(second)
            .zip(third)
            .zip(fourth)
            .zip(w0)
            .zip(w1)
            .zip(w2)
        {
            let left = Scalar::reduce(*a, modulus, reciprocal);
            let right = Scalar::reduce(*b, modulus, reciprocal);
            let left_rotated = Scalar::mulmod(left, root, modulus, reciprocal);
            let right_rotated = Scalar::mulmod(right, shifted_root, modulus, reciprocal);
            *a = left + right;
            *b = Scalar::mulmod(left - right, square_root, modulus, reciprocal);
            *c = left_rotated + right_rotated;
            *d = Scalar::mulmod(
                left_rotated - right_rotated,
                square_root,
                modulus,
                reciprocal,
            );
        }
    }
}
