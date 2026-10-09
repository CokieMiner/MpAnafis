//! Karatsuba square root in one partitioned limb workspace.
//!
//! References:
//! - P. Zimmermann, "Karatsuba Square Root", INRIA RR-3805, 1999.
//!   <https://inria.hal.science/inria-00072854>
//! - R. P. Brent and P. Zimmermann, "Modern Computer Arithmetic",
//!   Cambridge University Press, 2011, Section 1.5.1.

#![expect(
    unsafe_code,
    reason = "normalized root widths bound disjoint arithmetic spans, scalar shifts, and infallible inline basecases"
)]

use core::{mem::MaybeUninit, ptr::copy_nonoverlapping, slice::from_raw_parts_mut};

use super::{
    Addition, ArchKernels, DivScratch, DoubleLimb, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb,
    MulScratch, Roots, ScratchBuffer,
};

/// For roots of at most twice the inline width, the first recursive divisor
/// fits inline. Their workspace contains 2n input limbs and n+floor(n/2)+2
/// scratch limbs, requiring at most `7*INLINE_LIMBS+2` initialized limbs.
const STACK_WORKSPACE_LIMBS: usize = 7 * INLINE_LIMBS + 2;

impl Roots {
    /// Computes a floor square root and optionally its exact remainder.
    #[expect(
        clippy::too_many_lines,
        reason = "normalization, workspace partitioning, and exact remainder reconstruction share the root driver's width invariants"
    )]
    pub fn sqrt<const REMAINDER: bool>(a: &InternalMpUint) -> (InternalMpUint, InternalMpUint) {
        if let Some(root) = Self::isqrt_inline(a) {
            let remainder = if REMAINDER {
                a.sub(&root.square())
            } else {
                InternalMpUint::zero()
            };
            return (root, remainder);
        }
        let root_len = a.limbs().len().div_ceil(2);
        // SAFETY: the materialized input occupies at most isize::MAX bytes,
        // with at least two bytes per limb. Thus root_len<=2^(usize::BITS-3),
        // and input+scratch <= 7*root_len/2+2 < usize::MAX on every target.
        let (input_len, total) = unsafe {
            let input_len = root_len.unchecked_mul(2);
            let scratch_len = root_len.unchecked_add(root_len >> 1).unchecked_add(2);
            (input_len, input_len.unchecked_add(scratch_len))
        };
        // An odd limb count gains one low zero limb. The remaining even shift
        // normalizes the highest two bits without forming a total bit count,
        // which can exceed usize on small pointer-width targets.
        let whole_shift = a.limbs().len() & 1;
        // SAFETY: the inline branch returned for every input of at most four
        // limbs, so the canonical, nonzero high limb exists.
        let bit_shift = unsafe { a.limbs().last().unwrap_unchecked().leading_zeros() } & !1;
        // SAFETY: whole_shift <= 1 and bit_shift < LIMB_BITS <= 64. Their
        // combined shift is below 128 and fits usize on every supported target.
        let shift_bits = unsafe {
            usize::try_from(bit_shift)
                .unwrap_unchecked()
                .unchecked_add(whole_shift.unchecked_mul(LIMB_BITS))
        };
        let mut stack;
        let mut storage;
        let workspace = if total <= STACK_WORKSPACE_LIMBS {
            stack = [MaybeUninit::<Limb>::uninit(); STACK_WORKSPACE_LIMBS];
            let (workspace, _) = stack.split_at_mut(total);
            workspace
        } else {
            storage = ScratchBuffer::acquire(total);
            // SAFETY: acquisition reserves at least total writable limbs;
            // their MaybeUninit view permits initialization without a zero pass.
            unsafe { storage.spare_capacity_mut().get_unchecked_mut(..total) }
        };
        // SAFETY: workspace has total=input_len+scratch_len writable slots;
        // the input and scratch partitions retain disjoint MaybeUninit views.
        let (input_storage, scratch_storage) =
            unsafe { workspace.split_at_mut_unchecked(input_len) };
        // SAFETY: whole_shift is zero or one, below input_len>=6.
        let (low_padding, shifted) = unsafe { input_storage.split_at_mut_unchecked(whole_shift) };
        low_padding.fill(MaybeUninit::new(0));
        // SAFETY: input_len=original_len+whole_shift, so shifted retains
        // exactly the original input's limb count.
        let (destination, _) = unsafe { shifted.split_at_mut_unchecked(a.limbs().len()) };
        if bit_shift == 0 {
            // SAFETY: both owners are disjoint; the destination has exactly
            // the initialized input length and is writable without prior values.
            unsafe {
                copy_nonoverlapping(
                    a.limbs().as_ptr(),
                    destination.as_mut_ptr().cast::<Limb>(),
                    destination.len(),
                );
            }
        } else {
            // SAFETY: the source is initialized and the destination has the same
            // writable length; their owners are disjoint and 0<bit_shift<LIMB_BITS.
            // The shift does not exceed the leading zeros, so no carry escapes.
            let carry = unsafe {
                ArchKernels::lshift_into_unchecked(
                    destination.as_mut_ptr().cast::<Limb>(),
                    a.limbs().as_ptr(),
                    destination.len(),
                    bit_shift,
                )
            };
            debug_assert_eq!(
                carry, 0,
                "the shift uses only the high limb's leading zeros"
            );
        }
        // Only scratch is read before a first-write kernel initializes it.
        // The normalized input is already complete: an odd original limb count
        // gains one low zero, and an even count requires no padding.
        scratch_storage.fill(MaybeUninit::new(0));
        // SAFETY: copying or shifting initializes every original input limb;
        // low_padding completes the exact input_len span. Scratch was initialized
        // above. split_at_mut makes these aligned mutable spans disjoint.
        let (input, scratch) = unsafe {
            (
                from_raw_parts_mut(
                    input_storage.as_mut_ptr().cast::<Limb>(),
                    input_storage.len(),
                ),
                from_raw_parts_mut(
                    scratch_storage.as_mut_ptr().cast::<Limb>(),
                    scratch_storage.len(),
                ),
            )
        };
        let mut root = InternalMpUint::zero();
        root.resize(root_len);
        Self::sqrt_rem_recursive::<REMAINDER>(
            input,
            root.limbs_mut(),
            scratch,
            &mut DivScratch::default(),
            &mut MulScratch::default(),
        );

        let root_shift = shift_bits >> 1;
        let mut remainder = InternalMpUint::zero();
        if REMAINDER {
            // For normalized root S = s*2^root_shift + low, the original
            // remainder is (R + 2*S*low - low^2)/2^shift_bits. Normalization
            // shifts at most 2*LIMB_BITS-2 bits, so root_shift < LIMB_BITS and
            // low < B/2. Thus 2*low fits one limb and one scalar product forms
            // 2*S*low. The correction fits root_len+2 limbs, available because
            // root_len >= 3.
            // SAFETY: root_len >= 3 and input has 2*root_len limbs, hence
            // root_len+2 fits the existing span. The root is nonempty and
            // root_shift < LIMB_BITS bounds the discarded-bit mask.
            let (correction_len, low) = unsafe {
                (
                    root_len.unchecked_add(2),
                    *root.limbs().get_unchecked(0) & (1_usize << root_shift).unchecked_sub(1),
                )
            };
            // SAFETY: root_len>=3 proves correction_len=root_len+2<=2*root_len.
            let (rem, _) = unsafe { input.split_at_mut_unchecked(correction_len) };
            // SAFETY: rem has root_len+2 initialized limbs, leaving two guards.
            let (_, remainder_guards) = unsafe { rem.split_at_mut_unchecked(root_len) };
            // SAFETY: correction_len=root_len+2 leaves two guards. The first
            // contains the normalized remainder's high limb; the second is scratch.
            unsafe {
                *remainder_guards.get_unchecked_mut(1) = 0;
            }
            if low != 0 {
                // SAFETY: root has root_len initialized limbs, the remainder
                // has root_len+2, and both owners are disjoint. Since low < B/2,
                // low<<1 is exactly 2*low on every supported pointer width.
                let carry = unsafe {
                    ArchKernels::add_mul_limbs_unchecked(
                        rem.as_mut_ptr(),
                        root.limbs().as_ptr(),
                        root_len,
                        low << 1,
                    )
                };
                // SAFETY: rem retains root_len initialized digits and two guards.
                let (_, guards) = unsafe { rem.split_at_mut_unchecked(root_len) };
                // SAFETY: two initialized guards follow the product span.
                // The full-limb product carry is added to the first guard;
                // only its binary overflow is propagated through the second.
                let (first_guard, rest) = unsafe { guards.split_first_mut().unwrap_unchecked() };
                let (sum, carry_bit) = first_guard.overflowing_add(carry);
                *first_guard = sum;
                let overflow = Addition::propagate_carry(rest, Limb::from(carry_bit));
                debug_assert_eq!(overflow, 0, "two guard limbs contain 2*S*low+R");
                let low_square: [Limb; 2] = ArchKernels::mul_limb_lo_hi(low, low).into();
                let borrow = Addition::sub_slice_in_place(rem, &low_square);
                // SAFETY: rem has root_len+2>=5 digits, including the low square.
                let (_, tail) = unsafe { rem.split_at_mut_unchecked(2) };
                let underflow = Addition::propagate_borrow(tail, borrow);
                debug_assert_eq!(underflow, 0, "2*S*low >= low^2");
            }
            remainder.clone_from_slice(rem);
            remainder.shr_assign(shift_bits);
        }
        root.shr_assign(root_shift);
        (root, remainder)
    }

    /// Computes the floor square root of up to four limbs without allocations.
    #[expect(
        clippy::as_conversions,
        reason = "Native roots and extracted columns fit their target limbs on 16-, 32-, and 64-bit targets."
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "DoubleLimb columns narrow to the low Limb, or to a proved single-limb root."
        )
    )]
    #[must_use]
    pub fn isqrt_inline(a: &InternalMpUint) -> Option<InternalMpUint> {
        if a.is_zero() || a.is_one() {
            return Some(a.clone());
        }
        let len = a.limbs().len();
        if len > 4 {
            return None;
        }
        let [a0, a1, a2, a3] = a.extract_4();
        if len == 1 {
            return Some(InternalMpUint::from_limb(a0.isqrt()));
        }
        if len == 2 {
            let val = ((a1 as DoubleLimb) << LIMB_BITS) | (a0 as DoubleLimb);
            return Some(InternalMpUint::from_limb(val.isqrt() as Limb));
        }

        // The normalized three- or four-limb operand has a nonzero high half.
        // Zimmermann's 2-by-1 step operates on two DoubleLimb columns.
        let np_hi = ((a3 as DoubleLimb) << LIMB_BITS) | (a2 as DoubleLimb);
        let np_lo = ((a1 as DoubleLimb) << LIMB_BITS) | (a0 as DoubleLimb);
        let lz = (np_hi.leading_zeros() & !1) as usize;
        let (norm_hi, norm_lo) = if lz == 0 {
            (np_hi, np_lo)
        } else {
            (
                (np_hi << lz) | (np_lo >> ((LIMB_BITS * 2).wrapping_sub(lz))),
                np_lo << lz,
            )
        };
        // 0 < norm_hi < B^2 proves its square root fits one Limb on every
        // target. Widen only for the square and two-limb reconstruction.
        let sp0 = norm_hi.isqrt() as Limb;
        let sp0_wide = sp0 as DoubleLimb;
        let rp0_init = norm_hi.wrapping_sub(sp0_wide.wrapping_mul(sp0_wide));
        let rp0 = (rp0_init << LIMB_BITS.wrapping_sub(1)) | (norm_lo >> LIMB_BITS.wrapping_add(1));
        // 0 < norm_hi < B^2 gives 0 < sp0 < B. A high numerator limb
        // at least sp0 would give q >= B; clamp that estimate before dividing.
        // Otherwise the two-by-one kernel returns both exact intermediates.
        let high = (rp0 >> LIMB_BITS) as Limb;
        let (q, u) = if high >= sp0 {
            let quotient = Limb::MAX as DoubleLimb;
            (quotient, rp0.wrapping_sub(quotient.wrapping_mul(sp0_wide)))
        } else {
            // SAFETY: sp0 is positive and fits Limb; high < sp0 guarantees
            // a one-limb quotient. The low cast extracts rp0 modulo B on
            // every pointer width, and the high cast loses no bits.
            let (quotient, remainder) =
                unsafe { ArchKernels::divrem_1_unchecked(rp0 as Limb, high, sp0) };
            (quotient as DoubleLimb, remainder as DoubleLimb)
        };
        let one = DoubleLimb::from(1_u8);
        let mut root = (sp0_wide << LIMB_BITS) | q;
        let mut carry = isize::from((u >> LIMB_BITS.wrapping_sub(1)) != 0);
        let mut rem = (u << LIMB_BITS.wrapping_add(1))
            | (norm_lo & ((one << LIMB_BITS.wrapping_add(1)).wrapping_sub(1)));
        let q2 = q.wrapping_mul(q);
        if rem < q2 {
            carry = carry.wrapping_sub(1);
        }
        rem = rem.wrapping_sub(q2);
        while carry < 0 {
            let (r1, c1) = rem.overflowing_add(root);
            if c1 {
                carry = carry.wrapping_add(1);
            }
            root = root.wrapping_sub(1);
            let (r2, c2) = r1.overflowing_add(root);
            if c2 {
                carry = carry.wrapping_add(1);
            }
            rem = r2;
            carry = carry.wrapping_add(1);
        }
        root >>= lz >> 1;
        Some(InternalMpUint::from_limbs_2(
            root as Limb,
            (root >> LIMB_BITS) as Limb,
        ))
    }
}
