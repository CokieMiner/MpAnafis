//! Radix inverses with exact carries from the known low product.
//!
//! For `M*I=1 mod B^k`, write `M=M0+B^k*M1`. Extending to `m=k+s`,
//! `1<=s<=k`, requires `E=floor(M0*I/B^k)+M1*I mod B^s` and
//! the appended digits `-I*E mod B^s`. No uncomputed inverse digits enter
//! either product. A high approximation's two guards determine its carry
//! exactly because `M0*I mod B^k=1`.

#![expect(
    unsafe_code,
    reason = "The odd-domain seed and materialized modulus width bound geometric precision, disjoint product arenas, initialized guards and inverse suffixes"
)]

use core::num::NonZeroUsize;

#[cfg(not(target_pointer_width = "16"))]
use super::Multiplication;
use super::{
    Addition, ArchKernels, HighProduct, Limb, LowProduct, MontgomeryDomain, MontgomeryScratch,
    ScratchBuffer,
};

impl MontgomeryDomain {
    /// Builds `M^-1 mod B^n` from the constructor's exact one-limb inverse.
    ///
    /// The constructor supplies a materialized odd n-limb modulus with
    /// `n>=2` and `M[0]*seed=1 mod B`.
    /// Precision follows `ceil(n/2^j)` from one limb to n, so every update
    /// has `k<m<=2k` and appends exactly `s=m-k` initialized digits.
    pub fn inverse_mod_radix(modulus: &[Limb], seed: Limb) -> ScratchBuffer {
        let n = modulus.len();
        debug_assert!(n >= 2, "only wide domains build a radix inverse");
        debug_assert_eq!(
            modulus.first().copied().unwrap_or(0).wrapping_mul(seed),
            1,
            "the constructor supplies the exact low inverse"
        );
        // SAFETY: n>=2 proves n-1>0; n is a materialized slice length.
        let reduced_width = unsafe { NonZeroUsize::new_unchecked(n.unchecked_sub(1)) };
        // SAFETY: ilog2 of a nonzero usize is below Limb::BITS, so its
        // successor fits u32 and supplies every reverse precision exponent.
        let levels = unsafe { reduced_width.ilog2().unchecked_add(1) };
        let mut inverse = ScratchBuffer::acquire(n);
        // SAFETY: n>=2 and reservation supplies n writable limbs.
        // The seed initializes the only digit exposed before refinement.
        unsafe {
            inverse.as_mut_ptr().write(seed);
            inverse.set_len(1);
        }
        let mut scratch = MontgomeryScratch::default();
        for shift in (0..levels).rev() {
            let known = inverse.len();
            // SAFETY: shift<Limb::BITS. ((n-1)>>shift)+1<=n is the
            // ceiling n/2^shift; consecutive ceilings give known<next<=2*known.
            let next = unsafe { reduced_width.get().unchecked_shr(shift).unchecked_add(1) };
            // SAFETY: known<next<=n bounds the head and a nonempty suffix
            // of width next-known<=known. The inverse exposes only known digits.
            let (head, suffix) = unsafe {
                (
                    modulus.get_unchecked(..next),
                    NonZeroUsize::new_unchecked(next.unchecked_sub(known)),
                )
            };
            #[cfg(not(target_pointer_width = "16"))]
            let wrapped = Multiplication::try_mul_mod_bnm1::<false, false>(
                head,
                &inverse,
                next,
                &mut scratch.high_product,
                &mut scratch.multiplication,
            );
            #[cfg(target_pointer_width = "16")]
            let wrapped = false;
            let error_start = if wrapped {
                scratch.coefficients.reset_with_capacity(suffix.get());
                // Folding P=M*I into width w>=next adds F<=B^known-2
                // only to its known low 1. A canonical cyclic zero denotes
                // the all-maximum folded residue; its escaping low borrow
                // identifies the all-maximum error digits.
                // SAFETY: cyclic admission supplies w>=next>known digits.
                let (low, high) = unsafe { scratch.high_product.split_at_mut_unchecked(known) };
                if Addition::propagate_borrow(low, 1) != 0 {
                    // SAFETY: w-known>=suffix bounds these initialized digits.
                    unsafe {
                        high.get_unchecked_mut(..suffix.get()).fill(Limb::MAX);
                    }
                }
                known
            } else {
                inverse_refinement_error(head, &inverse, suffix, &mut scratch)
            };
            // SAFETY: error_start..error_start+suffix is initialized by either
            // product path. Both paths reserve suffix coefficient digits and
            // leave their exposed length zero. Both factors have suffix digits
            // and the destination is disjoint. The inverse has capacity n>=next;
            // negation initializes all appended digits before their length commits.
            unsafe {
                LowProduct::mul(
                    scratch
                        .coefficients
                        .spare_capacity_mut()
                        .get_unchecked_mut(..suffix.get()),
                    &inverse,
                    scratch
                        .high_product
                        .get_unchecked(error_start..error_start.unchecked_add(suffix.get())),
                    suffix.get(),
                    &mut scratch.multiplication,
                );
                let _ = Addition::negate_with_borrow(
                    inverse.as_mut_ptr().add(known),
                    scratch.coefficients.as_ptr(),
                    suffix.get(),
                    0,
                );
                inverse.set_len(next);
            }
        }
        inverse
    }
}

/// Forms `E=floor(M0*I/B^k)+M1*I mod B^s` in the high-product arena.
///
/// The driver supplies `k=head.len()-s=inverse.len()`, `1<=s<=k`,
/// `M0*I=1 mod B^k`, and a materialized modulus of at least k limbs.
/// Returns the start of the initialized s-digit error.
fn inverse_refinement_error(
    head: &[Limb],
    inverse: &[Limb],
    suffix: NonZeroUsize,
    scratch: &mut MontgomeryScratch,
) -> usize {
    let known = inverse.len();
    // SAFETY: known<=modulus.len(); its slice spans at most isize::MAX
    // bytes with at least two bytes per limb, so 2*known fits usize.
    let double_known = unsafe { known.unchecked_mul(2) };
    scratch.high_product.reset_with_capacity(double_known);
    // Leaf diagonals need no work arena. Recursive high products need
    // at most 2*k digits; the cross product then reuses that allocation.
    let work_len = HighProduct::scratch_len(inverse).max(suffix.get());
    scratch.coefficients.reset_with_capacity(work_len);
    // SAFETY: head contains known initialized digits; both factors are
    // nonempty. Output reserves their complete 2*k product; recursive
    // admission reserves the proved 2*k work bound. Both arenas and
    // multiplication scratch are disjoint from the immutable inputs.
    let (start, end) = unsafe {
        HighProduct::high_product_blocks(
            head.get_unchecked(..known),
            inverse,
            known,
            scratch
                .high_product
                .spare_capacity_mut()
                .get_unchecked_mut(..double_known),
            scratch.coefficients.spare_capacity_mut(),
            &mut scratch.multiplication,
        )
    };
    // SAFETY: a full leaf or the root's first full block initializes
    // [0,end). Subsequent cross additions only modify that prefix.
    unsafe {
        scratch.high_product.set_len(end);
    }
    if known >= 3 {
        // Put c=k-2 and T=floor(M0*I/B^c)-delta. Leaf omissions give
        // 0<=delta<k*B<B^2; recursive omissions give delta<=7<B^2.
        // Since M0*I mod B^k=1 and c>=1, the exact c-scaled product
        // is divisible by B^2. Therefore its exact k-scaled high is
        // floor(T/B^2)+[T mod B^2!=0].
        // SAFETY: k>=3 gives two initialized guards before start.
        let guard = unsafe {
            *scratch.high_product.get_unchecked(start.unchecked_sub(2))
                | *scratch.high_product.get_unchecked(start.unchecked_sub(1))
        };
        if guard != 0 {
            // The corrected high is below B^k. Its lower approximation
            // is below B^k-1, so one digit below MAX absorbs this carry.
            // SAFETY: start..end holds k initialized digits; the absorbing
            // digit proves that every pointer remains within that span.
            unsafe {
                let mut digit = scratch.high_product.as_mut_ptr().add(start);
                loop {
                    let (sum, pending) = digit.read().overflowing_add(1);
                    digit.write(sum);
                    if !pending {
                        break;
                    }
                    digit = digit.add(1);
                }
            }
        }
    }
    // SAFETY: M1 has suffix digits and I has known>=suffix digits. The
    // recursive work reservation supplies at least suffix coefficient
    // slots and leaves their exposed length zero. The low writer initializes
    // those raw slots. start..end has known>=suffix initialized high digits.
    unsafe {
        LowProduct::mul(
            scratch
                .coefficients
                .spare_capacity_mut()
                .get_unchecked_mut(..suffix.get()),
            head.get_unchecked(known..),
            inverse,
            suffix.get(),
            &mut scratch.multiplication,
        );
        // The escaping carry is discarded modulo B^suffix.
        let _ = ArchKernels::add_limbs_unchecked(
            scratch.high_product.as_mut_ptr().add(start),
            scratch.coefficients.as_ptr(),
            suffix.get(),
        );
    }
    start
}
