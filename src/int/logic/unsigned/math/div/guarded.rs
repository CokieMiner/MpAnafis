//! Quotient certification from a normalized low quotient guard digit.
//!
//! Let `B` be the limb radix, `Q=floor(U/D)<B^q`, and normalize both operands
//! by the same bit shift. Write `D_s=P*B^k+d`, with `0<=d<B^k`, retaining
//! `q+1` divisor limbs so `P>=B^(q+1)/2`. Dividing the longer numerator
//! prefix gives `T=floor(floor(U_s/B^(k-1))/P)`. Recursive approximation
//! returns `A>=T` with `A-T<E=2*Limb::BITS`, or proves `A=T` with E=0.
//!
//! Since `U_s>=Q*P*B^k`, `A>=T>=B*Q`. With `x=U_s/D_s<Q+1<=B^q`,
//! `T<B*x+B*x/P<B*x+2`. Hence `A<B*x+E+2` and `E+2<B` on every
//! supported limb width. Thus `floor(A/B)` is Q or Q+1, and an overestimate
//! requires `A mod B<E+2`. Every other guard certifies the quotient.
//!
//! An ambiguous guard compares the complete candidate product with the
//! original numerator. The one-unit bound permits exactly one decrement.
//! No prefix remainder is reconstructed or consumed.
//!
//! With complete operands, `T=floor(B*U_s/D_s)`, so an overestimate requires
//! `A mod B<E`. For an exact division x=Q, either bound gives A<B*(Q+1).
//! Guard removal therefore returns Q without a product comparison.
//! One guard suffices; a second would add a full-width quotient-digit step.

#![expect(
    unsafe_code,
    reason = "truncation admission bounds normalized suffixes, disjoint scratch spans, quotient guards and omitted-product correction"
)]

use core::{
    cmp::Ordering,
    mem::replace,
    ptr::{copy_nonoverlapping, write_bytes},
};

use super::{
    ArchKernels, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb, Multiplication,
    NEWTON_QUOTIENT_THRESHOLD, ScratchBuffer,
};

impl Division {
    /// Writes a quotient using one low quotient guard limb.
    ///
    /// With `split=0`, both complete operands are retained and the numerator
    /// receives one zero low limb. Otherwise `1<=split<divisor.len()` retains
    /// `q+1` divisor limbs, where `q=numerator.len()-divisor.len()+1`, and one
    /// additional numerator limb. The divisor has at least three limbs, the
    /// numerator exceeds it, and both slices are canonical. `EXACT` proves
    /// divisibility of the complete operands, including any discarded limbs.
    #[expect(
        clippy::too_many_lines,
        reason = "normalization, recursive quotient bounds and one-unit correction share the same operand and workspace owners"
    )]
    pub fn guarded_quotient<const EXACT: bool>(
        numerator: &[Limb],
        divisor: &[Limb],
        quotient: &mut InternalMpUint,
        split: usize,
        scratch: &mut DivScratch,
    ) {
        // SAFETY: admission proves split=0 or 1<=split<divisor.len(), with
        // divisor.len()<=numerator.len(). Both operand suffixes are initialized.
        // The canonical nonzero divisor has a positive high limb.
        let (num_split, padding, shift, num_head, den_head) = unsafe {
            let padding = usize::from(split == 0);
            let num_split = if split == 0 {
                0
            } else {
                split.unchecked_sub(1)
            };
            (
                num_split,
                padding,
                divisor.last().unwrap_unchecked().leading_zeros(),
                numerator.get_unchecked(num_split..),
                divisor.get_unchecked(split..),
            )
        };
        // The Newton driver also uses u_norm/v_norm while constructing its
        // reciprocal. Borrow their allocations as independent owners until
        // the prefix division and its possible correction are complete.
        let mut num_prefix = replace(&mut scratch.u_norm, ScratchBuffer::acquire(0));
        let mut den_prefix = replace(&mut scratch.v_norm, ScratchBuffer::acquire(0));
        // SAFETY: a materialized Limb slice is limited to isize::MAX bytes;
        // its length plus one low guard and one high guard fits usize.
        let num_capacity = unsafe { num_head.len().unchecked_add(padding).unchecked_add(1) };
        num_prefix.reset_with_capacity(num_capacity);
        // Initialize the normalized operand at its final guard offset. This
        // combines the normalization copy and multiplication by B^padding.
        // SAFETY: capacity >= num_head.len()+padding+1. Both owners are
        // disjoint and Limb-aligned; the input is nonempty and initialized.
        // The prefix zeros, shifted limbs and final carry initialize the whole
        // exposed span. Canonicality gives shift<Limb::BITS; the architecture
        // facade selects a valid leaf for 0<shift<Limb::BITS.
        unsafe {
            let destination = num_prefix.as_mut_ptr();
            write_bytes(destination, 0, padding);
            let shifted = destination.add(padding);
            let carry = if shift == 0 {
                copy_nonoverlapping(num_head.as_ptr(), shifted, num_head.len());
                0
            } else {
                ArchKernels::lshift_into_unchecked(
                    shifted,
                    num_head.as_ptr(),
                    num_head.len(),
                    shift,
                )
            };
            *shifted.add(num_head.len()) = carry;
            num_prefix.set_len(num_capacity);
        }
        let den = if shift == 0 {
            den_head
        } else {
            Self::shift_limbs_left::<false>(den_head, shift, &mut den_prefix);
            // SAFETY: 0 < shift < Limb::BITS. Both shifted suffixes remain
            // nonempty. Their preceding limbs supply precisely the low bits
            // of the normalized full-operand prefixes.
            unsafe {
                let carry_shift = Limb::BITS.unchecked_sub(shift);
                if split != 0 {
                    *den_prefix.get_unchecked_mut(0) |=
                        *divisor.get_unchecked(split.unchecked_sub(1)) >> carry_shift;
                }
                if num_split != 0 {
                    *num_prefix.get_unchecked_mut(0) |=
                        *numerator.get_unchecked(num_split.unchecked_sub(1)) >> carry_shift;
                }
            }
            den_prefix.as_slice()
        };
        let n = den.len();
        // SAFETY: the numerator retains one extra limb and one carry slot.
        // Its carry has at most shift bits, below the normalized divisor's
        // leading bit; hence its high n-limb window is below den.
        let quotient_len = unsafe { num_prefix.len().unchecked_sub(n) };
        let digits = quotient.ensure_capacity_set_len_get_limbs(quotient_len);
        let error_bound = if n < NEWTON_QUOTIENT_THRESHOLD {
            scratch.recursive_product.reset_with_capacity(n);
            // SAFETY: reservation supplies n disjoint writable spare limbs.
            // Each repair product is initialized before recursion reads it.
            let product = unsafe {
                scratch
                    .recursive_product
                    .spare_capacity_mut()
                    .get_unchecked_mut(..n)
            };
            Self::burnikel_div_approximate(
                &mut num_prefix,
                den,
                digits,
                product,
                &mut scratch.mul_scratch,
            )
        } else {
            // SAFETY: normalization gives den its high bit and num_prefix one
            // guard below it. The disjoint initialized quotient spans exactly
            // num_prefix.len()-den.len() limbs; no caller consumes its residue.
            unsafe {
                Self::newton_div_blocks::<true, false, false>(
                    &mut num_prefix,
                    den,
                    digits.as_mut_ptr(),
                    digits.len(),
                    scratch,
                );
            }
            0
        };
        // SAFETY: U > D implies A >= T >= B*Q >= B, so the low guard exists.
        let guard = unsafe { *digits.get_unchecked(0) };
        quotient.shr_assign(LIMB_BITS);
        quotient.normalize();
        // SAFETY: recursion gives E<=2*Limb::BITS<=128. Exact short leaves
        // and Newton give E=0. Prefix truncation adds at most two; the sum
        // is below B even on 16-bit limbs.
        let ambiguous_guard = unsafe { error_bound.unchecked_add(if split == 0 { 0 } else { 2 }) };
        if !EXACT && guard < ambiguous_guard {
            // C is Q or Q+1. This rare comparison decides the single correction
            // directly, without retaining a recursive prefix residue.
            // SAFETY: materialized Limb slices each have at most isize::MAX
            // bytes, so their limb-count sum fits usize on every pointer width.
            let product_len = unsafe { quotient.limbs().len().unchecked_add(divisor.len()) };
            scratch.q_den_low.reset_with_capacity(product_len);
            // SAFETY: U>D gives Q>=1 and the candidate is Q or Q+1. The
            // canonical divisor and quotient occupy disjoint owners; reserved
            // spare capacity holds their complete product. Every output limb
            // is initialized before its length is committed.
            unsafe {
                let product = Multiplication::mul_nonempty_distinct_into_uninit(
                    quotient.limbs(),
                    divisor,
                    scratch
                        .q_den_low
                        .spare_capacity_mut()
                        .get_unchecked_mut(..product_len),
                    &mut scratch.mul_scratch,
                );
                // Both operands are canonical and positive: their product
                // has product_len or product_len-1 limbs, with no lower gap.
                let top_is_zero = *product.get_unchecked(product_len.unchecked_sub(1)) == 0;
                scratch
                    .q_den_low
                    .set_len(product_len.unchecked_sub(usize::from(top_is_zero)));
            }
            if InternalMpUint::cmp_limbs(&scratch.q_den_low, numerator) == Ordering::Greater {
                quotient.decrement();
            }
        }
        scratch.u_norm = num_prefix;
        scratch.v_norm = den_prefix;
    }
}
