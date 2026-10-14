//! Disjoint butterfly ranges scheduled under one explicit worker budget.

#![expect(
    unsafe_code,
    reason = "Validated transform geometry proves pointer spans and disjoint synchronous task ranges"
)]

use super::{ParallelExecutor, Prime, Radix4Kernel, Scalar};

/// Raw stage admission and range scheduling.
#[derive(Debug)]
pub struct Stage;

struct RangeJob {
    values: *mut f64,
    twiddles: *const f64,
    quarter: usize,
    kernel: Radix4Kernel,
    p: f64,
    pinv: f64,
}

// SAFETY: RangeJob is private and borrowed only by run_ranges. Its recursive
// splits give each task disjoint indices in all four quarters. Twiddles are
// immutable, and synchronous joins finish before Stage::run releases either slice.
unsafe impl Sync for RangeJob {}

impl Stage {
    /// Executes a complete stage after its slice geometry and ISA have been admitted.
    pub fn run<E: ParallelExecutor>(
        values: &mut [f64],
        twiddles: &[f64],
        kernel: Radix4Kernel,
        prime: &Prime,
        budget: usize,
        executor: &E,
    ) {
        let quarter = values.len() >> 2;
        let job = RangeJob {
            values: values.as_mut_ptr(),
            twiddles: twiddles.as_ptr(),
            quarter,
            kernel,
            p: prime.float(),
            pinv: prime.reciprocal(),
        };
        run_ranges(&job, 0, quarter, budget, executor);
    }

    /// Executes the odd-order radix-2 stage, pruning a zero upper input half.
    pub fn radix2<const INVERSE: bool>(
        values: &mut [f64],
        twiddles: &[f64],
        input_len: usize,
        prime: &Prime,
    ) {
        let half = values.len() >> 1;
        let modulus = prime.float();
        let reciprocal = prime.reciprocal();
        // SAFETY: the admitted transform has two disjoint initialized halves.
        let (left, right) = unsafe { values.split_at_mut_unchecked(half) };
        if !INVERSE && input_len <= half {
            for ((a, b), &root) in left.iter_mut().zip(right).zip(twiddles) {
                let value = Scalar::reduce(*a, modulus, reciprocal);
                *a = value;
                *b = Scalar::mulmod(value, root, modulus, reciprocal);
            }
        } else {
            for ((a, b), &root) in left.iter_mut().zip(right).zip(twiddles) {
                let first = Scalar::reduce(*a, modulus, reciprocal);
                let second = if INVERSE {
                    Scalar::mulmod(*b, root, modulus, reciprocal)
                } else {
                    *b
                };
                *a = Scalar::reduce(first + second, modulus, reciprocal);
                *b = if INVERSE {
                    Scalar::reduce(first - second, modulus, reciprocal)
                } else {
                    Scalar::mulmod(first - second, root, modulus, reciprocal)
                };
            }
        }
    }
}

fn run_ranges<E: ParallelExecutor>(
    job: &RangeJob,
    start: usize,
    end: usize,
    budget: usize,
    executor: &E,
) {
    // SAFETY: recursive ranges satisfy start <= end <= quarter. Their
    // midpoint is within the parent span; positive budgets split only above one.
    let length = unsafe { end.unchecked_sub(start) };
    if budget > 1 && length >= 8192 {
        // SAFETY: start + length/2 <= end; 0 < budget/2 < budget.
        let (middle, left, right) = unsafe {
            let left = budget >> 1;
            (
                start.unchecked_add(length >> 1),
                left,
                budget.unchecked_sub(left),
            )
        };
        executor.join(
            || run_ranges(job, start, middle, left, executor),
            || run_ranges(job, middle, end, right, executor),
        );
    } else {
        // SAFETY: Stage::run admits four initialized quarter spans and three
        // twiddle spans. Recursive tasks access disjoint start..end ranges,
        // and the kernel table establishes target prerequisites.
        unsafe {
            (job.kernel)(
                job.values,
                job.twiddles,
                job.quarter,
                start,
                end,
                job.p,
                job.pinv,
            );
        }
    }
}
