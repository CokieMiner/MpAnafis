//! Forward and inverse tables generated from the same admitted stage geometry.

#![expect(
    unsafe_code,
    reason = "Admitted power-of-two geometry proves disjoint table spans and canonical field products"
)]

use super::{ParallelExecutor, Prime};

/// Immutable tables for all admitted prime spectra.
#[derive(Clone, Copy, Debug)]
pub struct Tables<'tables> {
    pub forward: &'tables [f64],
    pub inverse: &'tables [f64],
}

/// Prepared power-of-two roots and stage tables.
#[derive(Debug)]
pub struct Twiddles;

impl Twiddles {
    /// Initializes independent prime tables under a partitioned worker budget.
    pub fn prepare<E: ParallelExecutor>(
        forward: &mut [f64],
        inverse: &mut [f64],
        len: usize,
        primes: &[Prime],
        budget: usize,
        executor: &E,
    ) {
        if budget > 1 && primes.len() > 1 && len >= 4096 {
            let count = primes.len() >> 1;
            let left = budget >> 1;
            // SAFETY: prime_count*len is the admitted table span. count is a
            // positive proper prefix, and both table borrows are exclusively split.
            let ((fp, fs), (ip, is), (pp, ps)) = unsafe {
                (
                    forward.split_at_mut_unchecked(count.unchecked_mul(len)),
                    inverse.split_at_mut_unchecked(count.unchecked_mul(len)),
                    primes.split_at_unchecked(count),
                )
            };
            // SAFETY: budget > 1 proves 0 < left < budget.
            let right = unsafe { budget.unchecked_sub(left) };
            executor.join(
                || {
                    Self::prepare(fp, ip, len, pp, left, executor);
                },
                || {
                    Self::prepare(fs, is, len, ps, right, executor);
                },
            );
        } else {
            for ((fwd, inv), prime) in forward
                .chunks_exact_mut(len)
                .zip(inverse.chunks_exact_mut(len))
                .zip(primes)
            {
                Self::generate(fwd, inv, prime);
            }
        }
    }

    /// Generates `len-1` DIF and DIT roots, retaining the inverse scale in the final DIT slot.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Every canonical twiddle is below 2^50 and exactly representable in binary64"
    )]
    pub fn generate(forward: &mut [f64], inverse: &mut [f64], prime: &Prime) {
        let len = forward.len();
        debug_assert!(
            len.is_power_of_two() && len >= 4 && inverse.len() == len,
            "table geometry is admitted"
        );
        let mut base = prime.primitive_root_of_unity(len.trailing_zeros());
        let mut base_inverse = prime.inv_mod(base);
        let modulus = prime.modulus().cast_signed();
        let half_modulus = prime.modulus() >> 1;
        let mut step = len;
        let mut offset = 0;
        // SAFETY: admitted lengths have power-of-two roots in this prime.
        // Each radix-4 stage has 3*quarter slots. Forward prefixes sum to at
        // most len-1; inverse stage offsets are quarter-1. The two mutable
        // tables are disjoint. All generated roots stay canonical under mulmod.
        unsafe {
            if len.trailing_zeros() & 1 != 0 {
                let half = len >> 1;
                let fp = forward.get_unchecked_mut(..half);
                let ip = inverse.get_unchecked_mut(half.unchecked_sub(1)..len.unchecked_sub(1));
                let mut fw = 1_u64;
                let mut iw = 1_u64;
                for (f, i) in fp.iter_mut().zip(ip) {
                    *f = fw.cast_signed().unchecked_sub(modulus.unchecked_mul(i64::from(fw > half_modulus))) as f64;
                    *i = iw.cast_signed().unchecked_sub(modulus.unchecked_mul(i64::from(iw > half_modulus))) as f64;
                    fw = prime.mulmod_fast(fw, base);
                    iw = prime.mulmod_fast(iw, base_inverse);
                }
                base = prime.mulmod_fast(base, base);
                base_inverse = prime.mulmod_fast(base_inverse, base_inverse);
                offset = half;
                step = half;
            }
            while step >= 4 {
                let quarter = step >> 2;
                let count = quarter.unchecked_mul(3);
                let fstage = forward.get_unchecked_mut(offset..offset.unchecked_add(count));
                let istage =
                    inverse.get_unchecked_mut(quarter.unchecked_sub(1)..step.unchecked_sub(1));
                let (flinear, fsquared) = fstage.split_at_mut_unchecked(quarter << 1);
                let (ilinear, isquared) = istage.split_at_mut_unchecked(quarter << 1);
                let mut fw = 1_u64;
                let mut iw = 1_u64;
                for (f, i) in flinear.iter_mut().zip(ilinear.iter_mut()) {
                    *f = fw.cast_signed().unchecked_sub(modulus.unchecked_mul(i64::from(fw > half_modulus))) as f64;
                    *i = iw.cast_signed().unchecked_sub(modulus.unchecked_mul(i64::from(iw > half_modulus))) as f64;
                    fw = prime.mulmod_fast(fw, base);
                    iw = prime.mulmod_fast(iw, base_inverse);
                }
                // The third quarter uses W^(2*j), already in the first half.
                for ((f, i), (ff, ii)) in fsquared
                    .iter_mut()
                    .zip(isquared)
                    .zip(flinear.iter().step_by(2).zip(ilinear.iter().step_by(2)))
                {
                    *f = *ff;
                    *i = *ii;
                }
                let squared = prime.mulmod_fast(base, base);
                base = prime.mulmod_fast(squared, squared);
                let squared_inverse = prime.mulmod_fast(base_inverse, base_inverse);
                base_inverse = prime.mulmod_fast(squared_inverse, squared_inverse);
                offset = offset.unchecked_add(count);
                step >>= 2;
            }
            // The power-of-two length fits u64 on every supported target and
            // is nonzero modulo this odd prime. The unused final root slot
            // retains its canonical inverse for every operation with this table.
            *inverse.get_unchecked_mut(len.unchecked_sub(1)) = prime.pow_mod(
                u64::try_from(len).unwrap_unchecked(),
                prime.modulus().unchecked_sub(2),
            ) as f64;
        }
    }
}
