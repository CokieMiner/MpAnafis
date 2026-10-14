//! Scratch admission, prime-local execution, and sequential native reconstruction.

#![expect(
    unsafe_code,
    reason = "Immutable plans and exclusive aligned arenas prove kernel spans and residue bounds"
)]

use crate::{
    arch::NttFloatKernels,
    conversion::Conversion,
    crt::Crt,
    transform::{Tables, Transform, Twiddles},
};

use super::{
    DefaultExecutor, Limb, MAX_PRIMES, NttPlan, PRIME_CATALOG, ParallelExecutor, PlannedProfile,
    Prime,
};

/// Exact native-limb multiplication using a planned prime prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Ntt;

/// Internal execution of an admitted prime convolution.
#[derive(Debug)]
pub struct Multiplication;

struct RunContext<'operation, E> {
    operands: (&'operation [Limb], &'operation [Limb]),
    geometry: &'operation PlannedProfile,
    kernels: &'operation NttFloatKernels,
    executor: &'operation E,
}

impl Ntt {
    /// Multiplies the exact destination prefix using the default executor.
    ///
    /// Admission failures and destination suffixes preserve their original values.
    pub fn try_mul_with_scratch(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) -> bool {
        DefaultExecutor::with_resolved(|executor| {
            Self::try_mul_with_scratch_and_executor(dst, a, b, scratch, executor)
        })
    }

    /// Squares the exact destination prefix using the default executor.
    ///
    /// Admission failures and destination suffixes preserve their original values.
    pub fn try_sqr_with_scratch(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) -> bool {
        DefaultExecutor::with_resolved(|executor| {
            Self::try_sqr_with_scratch_and_executor(dst, a, scratch, executor)
        })
    }

    /// Multiplies with caller-owned scratch and an explicit execution policy.
    pub fn try_mul_with_scratch_and_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) -> bool {
        let Some(output_len) = a.len().checked_add(b.len()) else {
            return false;
        };
        let Some(output) = dst.get_mut(..output_len) else {
            return false;
        };
        if a.is_empty() || b.is_empty() {
            output.fill(0);
            return true;
        }
        let Some(plan) = NttPlan::for_multiplication(a.len(), b.len()) else {
            return false;
        };
        if scratch.len() < plan.required_mul_scratch_limbs() {
            return false;
        }
        // SAFETY: all native integer and float bit patterns are valid.
        // The admitted scratch size includes worst-case alignment padding,
        // so align_to_mut retains the complete four-arena float span.
        let (operands, forward, inverse) = unsafe {
            let (_, aligned, _) = scratch.align_to_mut::<f64>();
            let floats = aligned.get_unchecked_mut(..plan.mul_f64_scratch);
            let (values, roots) = floats.split_at_mut_unchecked(plan.arena_len.unchecked_mul(2));
            let (fwd, inv) = roots.split_at_mut_unchecked(plan.arena_len);
            (values, fwd, inv)
        };
        let geometry = &plan.planned;
        // SAFETY: the immutable plan admits a nonempty catalog prefix.
        let primes = unsafe { PRIME_CATALOG.get_unchecked(..geometry.profile.np) };
        Twiddles::prepare(
            forward,
            inverse,
            geometry.transform_len,
            primes,
            executor.parallelism().get(),
            executor,
        );
        Multiplication::execute::<false, E>(
            output,
            (a, b),
            operands,
            Tables { forward, inverse },
            &plan,
            &NttFloatKernels::get(),
            executor,
        );
        true
    }

    /// Squares with caller-owned scratch and an explicit execution policy.
    pub fn try_sqr_with_scratch_and_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) -> bool {
        let Some(output_len) = a.len().checked_mul(2) else {
            return false;
        };
        let Some(output) = dst.get_mut(..output_len) else {
            return false;
        };
        if a.is_empty() {
            output.fill(0);
            return true;
        }
        let Some(plan) = NttPlan::for_multiplication(a.len(), a.len()) else {
            return false;
        };
        if scratch.len() < plan.required_sqr_scratch_limbs() {
            return false;
        }
        // SAFETY: all storage bit patterns are valid, and the checked native
        // scratch count includes alignment padding for the three float arenas.
        let (operands, forward, inverse) = unsafe {
            let (_, aligned, _) = scratch.align_to_mut::<f64>();
            let floats = aligned.get_unchecked_mut(..plan.sqr_f64_scratch);
            let (values, roots) = floats.split_at_mut_unchecked(plan.arena_len);
            let (fwd, inv) = roots.split_at_mut_unchecked(plan.arena_len);
            (values, fwd, inv)
        };
        let geometry = &plan.planned;
        // SAFETY: the immutable plan admits a nonempty catalog prefix.
        let primes = unsafe { PRIME_CATALOG.get_unchecked(..geometry.profile.np) };
        Twiddles::prepare(
            forward,
            inverse,
            geometry.transform_len,
            primes,
            executor.parallelism().get(),
            executor,
        );
        Multiplication::execute::<true, E>(
            output,
            (a, a),
            operands,
            Tables { forward, inverse },
            &plan,
            &NttFloatKernels::get(),
            executor,
        );
        true
    }
}

impl Multiplication {
    /// Runs admitted operands in exclusive arenas and immutable transform tables.
    pub fn execute<const SQUARE: bool, E: ParallelExecutor>(
        dst: &mut [Limb],
        operands: (&[Limb], &[Limb]),
        arena: &mut [f64],
        tables: Tables<'_>,
        plan: &NttPlan,
        kernels: &NttFloatKernels,
        executor: &E,
    ) {
        let geometry = &plan.planned;
        // SAFETY: both callers retain arena storage for the admitted operation.
        // Splits are exclusive; multiplication needs two spans and squaring one.
        let (left, right, primes) = unsafe {
            let span = plan.arena_len;
            let values = if SQUARE {
                (arena.get_unchecked_mut(..span), &mut [][..])
            } else {
                let (prefix, suffix) = arena.split_at_mut_unchecked(span);
                (prefix, suffix.get_unchecked_mut(..span))
            };
            (
                values.0,
                values.1,
                PRIME_CATALOG.get_unchecked(..geometry.profile.np),
            )
        };
        let context = RunContext {
            operands,
            geometry,
            kernels,
            executor,
        };
        run_primes::<SQUARE, E>(
            left,
            right,
            tables,
            primes,
            &context,
            executor.parallelism().get(),
        );
        let mut residues: [&[f64]; MAX_PRIMES] = [&[]; MAX_PRIMES];
        for (slot, spectrum) in residues
            .iter_mut()
            .zip(left.chunks_exact(geometry.transform_len))
        {
            *slot = spectrum;
        }
        // SAFETY: geometry admits coefficient uniqueness and native output
        // capacity. Active spectra contain initialized scaled integer residues
        // of magnitude below 2*p over the entire convolution coefficient span.
        unsafe {
            Crt::reconstruct(dst, &residues, geometry);
        }
    }
}

fn run_primes<const SQUARE: bool, E: ParallelExecutor>(
    left: &mut [f64],
    right: &mut [f64],
    tables: Tables<'_>,
    primes: &[Prime],
    context: &RunContext<'_, E>,
    budget: usize,
) {
    let len = context.geometry.transform_len;
    if budget > 1 && primes.len() > 1 && len >= 4096 {
        let half = primes.len() >> 1;
        let workers = budget >> 1;
        // SAFETY: these arenas and tables cover exactly the admitted prime
        // prefix. half is a nonempty proper prefix; budget > 1 ensures both
        // worker partitions remain positive and sum to the parent's budget.
        let ((lp, ls), (rp, rs), (fp, fs), (ip, is), (pp, ps), remaining) = unsafe {
            let span = half.unchecked_mul(len);
            (
                left.split_at_mut_unchecked(span),
                if SQUARE {
                    (&mut [][..], &mut [][..])
                } else {
                    right.split_at_mut_unchecked(span)
                },
                tables.forward.split_at_unchecked(span),
                tables.inverse.split_at_unchecked(span),
                primes.split_at_unchecked(half),
                budget.unchecked_sub(workers),
            )
        };
        context.executor.join(
            || {
                run_primes::<SQUARE, E>(
                    lp,
                    rp,
                    Tables {
                        forward: fp,
                        inverse: ip,
                    },
                    pp,
                    context,
                    workers,
                );
            },
            || {
                run_primes::<SQUARE, E>(
                    ls,
                    rs,
                    Tables {
                        forward: fs,
                        inverse: is,
                    },
                    ps,
                    context,
                    remaining,
                );
            },
        );
    } else {
        let mut remaining_right = right;
        for (((lbuf, forward), inverse), prime) in left
            .chunks_exact_mut(len)
            .zip(tables.forward.chunks_exact(len))
            .zip(tables.inverse.chunks_exact(len))
            .zip(primes)
        {
            let rbuf = if SQUARE {
                &mut [][..]
            } else {
                // SAFETY: each prime iteration consumes one of the admitted
                // disjoint right spectra, so len never exceeds the remaining span.
                let (head, tail) = unsafe { remaining_right.split_at_mut_unchecked(len) };
                remaining_right = tail;
                head
            };
            process_prime::<SQUARE, E>(
                lbuf,
                rbuf,
                Tables { forward, inverse },
                prime,
                context,
                budget,
            );
        }
    }
}

fn process_prime<const SQUARE: bool, E: ParallelExecutor>(
    left: &mut [f64],
    right: &mut [f64],
    tables: Tables<'_>,
    prime: &Prime,
    context: &RunContext<'_, E>,
    budget: usize,
) {
    let geometry = context.geometry;
    let kernels = context.kernels;
    let executor = context.executor;
    let len = geometry.transform_len;
    Conversion::pack(
        left,
        context.operands.0,
        prime,
        geometry.profile.chunk_bits,
        geometry.digits_a,
    );
    if !SQUARE {
        Conversion::pack(
            right,
            context.operands.1,
            prime,
            geometry.profile.chunk_bits,
            geometry.digits_b,
        );
    }
    if !SQUARE && budget > 1 && len >= 32768 {
        let workers = budget >> 1;
        // SAFETY: budget > 1 proves 0 < workers < budget.
        let remaining = unsafe { budget.unchecked_sub(workers) };
        executor.join(
            || {
                Transform::forward(
                    left,
                    tables.forward,
                    geometry.digits_a,
                    prime,
                    kernels,
                    workers,
                    executor,
                );
            },
            || {
                Transform::forward(
                    right,
                    tables.forward,
                    geometry.digits_b,
                    prime,
                    kernels,
                    remaining,
                    executor,
                );
            },
        );
    } else {
        Transform::forward(
            left,
            tables.forward,
            geometry.digits_a,
            prime,
            kernels,
            budget,
            executor,
        );
        if !SQUARE {
            Transform::forward(
                right,
                tables.forward,
                geometry.digits_b,
                prime,
                kernels,
                1,
                executor,
            );
        }
    }
    let pointwise = if SQUARE {
        kernels.square
    } else {
        kernels.multiply
    };
    let source = if SQUARE {
        core::ptr::null()
    } else {
        right.as_ptr()
    };
    // SAFETY: initialized spectra cover len residues below 2*p in magnitude.
    // Multiplication reads a disjoint right span; squaring ignores its null
    // pointer. The selected table establishes the kernel's ISA prerequisites.
    unsafe {
        pointwise(
            left.as_mut_ptr(),
            source,
            len,
            prime.float(),
            prime.reciprocal(),
        );
    }
    Transform::inverse(left, tables.inverse, prime, kernels, budget, executor);
    // SAFETY: the inverse initializes all len entries; conv_len <= len and
    // Twiddles::generate stores the canonical inverse scale in its last slot.
    // The kernel's ISA is already admitted.
    unsafe {
        (kernels.scale)(
            left.as_mut_ptr(),
            geometry.conv_len,
            *tables.inverse.get_unchecked(len.unchecked_sub(1)),
            prime.float(),
            prime.reciprocal(),
        );
    }
}
