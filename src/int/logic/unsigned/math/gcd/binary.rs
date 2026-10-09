//! Native-width binary GCD and odd-scalar cancellation.
//!
//! Reference: J. Stein, "Computational problems associated with Racah algebra",
//! Journal of Computational Physics 1(3), 397-405, 1967.
//! DOI: 10.1016/0021-9991(67)90047-2.

#![expect(
    unsafe_code,
    reason = "binary GCD validates operand widths and nonzero scalar divisors before limb access and reduction"
)]

use core::num::NonZeroUsize;

use super::{
    ArchKernels, BINARY_EUCLID_DIVISION_SHIFT, DivScratch, Division, Gcd, InternalMpUint,
    LIMB_BITS, Limb,
};

impl Gcd {
    /// Resolves zero, unit, scalar, and two-limb reductions before the full tower.
    /// Wider operands with unrelated high prefixes require a general reduction.
    pub fn small_gcd(left: &InternalMpUint, right: &InternalMpUint) -> Option<InternalMpUint> {
        if left.is_zero() {
            return Some(right.clone());
        }
        if right.is_zero() {
            return Some(left.clone());
        }
        // A unit operand answers before the leaf dispatch: gcd(1, x) is one
        // without entering the binary kernel.
        if left.is_one() || right.is_one() {
            return Some(InternalMpUint::one());
        }
        let left_len = left.limbs().len();
        let right_len = right.limbs().len();
        // Equal native widths enter the register kernel. A single-limb
        // divisor first reduces its wider partner modulo its odd part,
        // avoiding a two-limb binary recurrence across the width gap.
        if left_len == right_len && left_len <= 2 {
            return Some(Self::gcd_leaf_pair(left, right));
        }

        let difference;
        let (large, small) = if right_len <= 2 && left_len >= right_len {
            (left, right)
        } else if left_len <= 2 {
            (right, left)
        } else {
            // For equal high parts, gcd(H*B^2+a, H*B^2+b) equals
            // gcd(H*B^2+a, |a-b|). The difference fits the existing
            // two-limb leaf without copying either full-width operand.
            // SAFETY: both lengths exceed two, so the high suffixes and
            // their two preceding low limbs are initialized and in bounds.
            let (first, second) = unsafe {
                if left.limbs().get_unchecked(2..) != right.limbs().get_unchecked(2..) {
                    return None;
                }
                (
                    (
                        *left.limbs().get_unchecked(1),
                        *left.limbs().get_unchecked(0),
                    ),
                    (
                        *right.limbs().get_unchecked(1),
                        *right.limbs().get_unchecked(0),
                    ),
                )
            };
            let (larger, smaller) = if first >= second {
                (first, second)
            } else {
                (second, first)
            };
            let (low, borrow) = larger.1.overflowing_sub(smaller.1);
            // SAFETY: lexicographic (high, low) ordering proves the complete
            // two-limb difference is nonnegative, including its low borrow.
            let high = unsafe {
                larger
                    .0
                    .unchecked_sub(smaller.0)
                    .unchecked_sub(Limb::from(borrow))
            };
            if low | high == 0 {
                return Some(left.clone());
            }
            difference = InternalMpUint::from_limbs_2(low, high);
            (left, &difference)
        };
        if let [scalar] = small.limbs() {
            if scalar.is_power_of_two() {
                // gcd(a, 2^k) = 2^min(v_2(a), k), with k < LIMB_BITS.
                // SAFETY: both original inputs are nonzero, and `large`
                // always refers to an original input. Its low limb exists.
                let low = unsafe { *large.limbs().get_unchecked(0) };
                let twos = low.trailing_zeros().min(scalar.trailing_zeros());
                return Some(InternalMpUint::from_limb(1 << twos));
            }
            let y_tz = scalar.trailing_zeros();
            // SAFETY: small is a nonzero single-limb operand; removing its
            // trailing zero bits leaves a positive odd native divisor.
            let divisor = unsafe { NonZeroUsize::new_unchecked(*scalar >> y_tz) };
            // SAFETY: both original inputs are nonzero, and `large`
            // always refers to an original input. Its low limb exists.
            let low = unsafe { *large.limbs().get_unchecked(0) };
            let common_twos = low.trailing_zeros().min(y_tz);
            let g = Self::gcd_odd_limb(large.limbs(), divisor);
            return Some(InternalMpUint::from_limb(g << common_twos));
        }

        let mut rem = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        Division::rem_into(large, small, &mut rem, &mut scratch);
        Some(Self::gcd_leaf_pair(small, &rem))
    }

    pub fn gcd_leaf_pair(left: &InternalMpUint, right: &InternalMpUint) -> InternalMpUint {
        debug_assert!(
            left.limbs().len() <= 2 && right.limbs().len() <= 2,
            "binary leaves accept at most two limbs"
        );
        let [u0, u1, _, _] = left.extract_4();
        let [v0, v1, _, _] = right.extract_4();
        if (u1 | v1) == 0 {
            return InternalMpUint::from_limb(Self::gcd_1(u0, v0));
        }
        let [g0, g1] = Self::gcd_2([u0, u1], [v0, v1]);
        InternalMpUint::from_limbs_2(g0, g1)
    }

    /// Reduces a multi-limb integer against a positive odd scalar divisor.
    /// The caller supplies at least two initialized limbs and an odd divisor.
    ///
    /// Low-to-high cancellation gives `A-Q*d = B^(len-1)*(last-carry)`
    /// with `0<=carry<d`. Since B is invertible modulo d, taking the GCD of
    /// d with `abs(last-carry)` preserves the answer. The difference fits a
    /// limb. A difference below d needs no multiplication; one below 2d
    /// needs one subtraction. Larger differences use one more cancellation
    /// with the existing inverse. Each case gives a residue below d, allowing
    /// the first binary subtraction to use a known operand ordering.
    pub fn gcd_odd_limb(numerator: &[Limb], denominator: NonZeroUsize) -> Limb {
        let divisor = denominator.get();
        debug_assert!(numerator.len() >= 2, "multi-limb scalar reduction");
        debug_assert_eq!(divisor & 1, 1, "positive odd divisor");
        if divisor == 1 {
            return 1;
        }
        let inverse = Division::modular_inverse_limb(divisor);
        // SAFETY: at least two initialized limbs prove last_index>=1. The
        // first and last columns exist, and 1..last_index is a valid middle.
        let (first, middle, last) = unsafe {
            let index = numerator.len().unchecked_sub(1);
            (
                *numerator.get_unchecked(0),
                numerator.get_unchecked(1..index),
                *numerator.get_unchecked(index),
            )
        };
        // The first column has zero incoming carry. Its low cancellation is
        // exact, so the high product initializes carry without a subtraction.
        let (_, mut carry) = ArchKernels::mul_limb_lo_hi(first.wrapping_mul(inverse), divisor);
        for &limb in middle {
            let (difference, borrow) = limb.overflowing_sub(carry);
            // q*d equals difference modulo B, so multiplication by the
            // modular inverse intentionally discards every high product bit.
            let quotient = difference.wrapping_mul(inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(quotient, divisor);
            // high<=d-1. Equality forces the low product<=B-d, whereas a
            // borrow forces difference>=B-d+1; both conditions cannot hold.
            // SAFETY: high+borrow<d<=Limb::MAX preserves the carry invariant.
            carry = unsafe { high.unchecked_add(Limb::from(borrow)) };
        }
        let mut residue = last.abs_diff(carry);
        if residue == 0 {
            return divisor;
        }
        if residue >= divisor {
            // SAFETY: this branch proves the difference is nonnegative.
            let difference = unsafe { residue.unchecked_sub(divisor) };
            if difference < divisor {
                // residue<2*d gives its exact remainder after one subtraction,
                // without forming the potentially overflowing bound 2*d.
                residue = difference;
            } else {
                // q*d has low limb residue, so residue-q*d=-high*B. Since B
                // is invertible modulo odd d, gcd(d,residue)=gcd(d,high).
                // q<B proves high<d, matching the other reduction cases.
                (_, residue) = ArchKernels::mul_limb_lo_hi(residue.wrapping_mul(inverse), divisor);
            }
            if residue == 0 {
                return divisor;
            }
        }
        // Removing powers of two preserves the GCD with the odd divisor.
        residue >>= residue.trailing_zeros();
        // SAFETY: 0<residue<divisor and both values are odd, so their
        // difference is positive and even. Peel the first binary subtraction
        // before the loop without an ordering comparison or conditional move.
        let first_difference =
            unsafe { NonZeroUsize::new_unchecked(divisor.unchecked_sub(residue)) };
        let mut left = (first_difference.get() >> first_difference.trailing_zeros()) >> 1;
        let mut right = residue >> 1;
        // Both values have an implicit low one-bit. Modular subtraction and
        // its sign mask encode absolute differences without signed overflow.
        while left != right {
            let difference = left.wrapping_sub(right);
            let mask = if left < right { Limb::MAX } else { 0 };
            right = right.wrapping_add(mask & difference);
            left = (difference ^ mask).wrapping_sub(mask);
            left = (left >> 1) >> difference.trailing_zeros();
        }
        (left << 1) | 1
    }

    /// In-register GCD for integers up to two limbs.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "LIMB_BITS fits in u32"
    )]
    pub const fn gcd_2(mut u: [Limb; 2], mut v: [Limb; 2]) -> [Limb; 2] {
        if (u[0] | u[1]) == 0 {
            return v;
        }
        if (v[0] | v[1]) == 0 {
            return u;
        }

        let u_tz = trailing_zeros_2(u);
        let v_tz = trailing_zeros_2(v);
        let common_shift = if u_tz < v_tz { u_tz } else { v_tz };
        rshift_2(&mut u, u_tz);
        rshift_2(&mut v, v_tz);

        if (u[1] | v[1]) == 0 {
            let mut res = [Self::gcd_1(u[0], v[0]), 0];
            lshift_2(&mut res, common_shift);
            return res;
        }

        // Implicit least significant bit: both u and v are odd.
        let limb_bits = LIMB_BITS as u32;
        // SAFETY: supported limb widths are 16, 32, and 64, all positive.
        let high_shift = unsafe { limb_bits.unchecked_sub(1) };
        let mut u0 = (u[0] >> 1) | (u[1] << high_shift);
        let mut u1 = u[1] >> 1;
        let mut v0 = (v[0] >> 1) | (v[1] << high_shift);
        let mut v1 = v[1] >> 1;

        while u1 != 0 || v1 != 0 {
            let (diff0, b0) = u0.overflowing_sub(v0);
            let (diff1_pre, b1a) = u1.overflowing_sub(v1);
            let (diff1, b1b) = diff1_pre.overflowing_sub(if b0 { 1 } else { 0 });
            let borrow = b1a || b1b;
            let vgtu = if borrow { !0 } else { 0 };

            if diff0 == 0 {
                if diff1 == 0 {
                    let mut res = [(u0 << 1) | 1, (u1 << 1) | (u0 >> high_shift)];
                    lshift_2(&mut res, common_shift);
                    return res;
                }
                let c = diff1.trailing_zeros();
                v1 = v1.wrapping_add(vgtu & diff1);
                u0 = (diff1 ^ vgtu).wrapping_sub(vgtu);
                // SAFETY: diff1 is nonzero and represents a difference
                // between high halves below B/2, so c < LIMB_BITS-1.
                u0 >>= unsafe { c.unchecked_add(1) };
                u1 = 0;
            } else {
                // SAFETY: diff0!=0 gives trailing_zeros < LIMB_BITS <= 64.
                let c = unsafe { diff0.trailing_zeros().unchecked_add(1) };
                // v <-- min(u, v)
                let (v0_new, c0) = v0.overflowing_add(vgtu & diff0);
                let (v1_new, _) =
                    v1.overflowing_add((vgtu & diff1).wrapping_add(if c0 { 1 } else { 0 }));
                v0 = v0_new;
                v1 = v1_new;

                // u <-- |u - v|
                u0 = (diff0 ^ vgtu).wrapping_sub(vgtu);
                u1 = diff1 ^ vgtu;

                if c == limb_bits {
                    u0 = u1;
                    u1 = 0;
                } else {
                    // SAFETY: c is in [1,LIMB_BITS], and equality returned
                    // above; the complementary shift is in [1,LIMB_BITS).
                    let complementary = unsafe { limb_bits.unchecked_sub(c) };
                    u0 = (u0 >> c) | (u1 << complementary);
                    u1 >>= c;
                }
            }
        }

        while ((v0 | u0) & (1 << high_shift)) != 0 {
            let (diff0, borrow) = u0.overflowing_sub(v0);
            if diff0 == 0 {
                // The implicit odd value is 2*u0+1. Although the shifted
                // high limb is zero, u0's top bit still belongs to the high
                // limb of that reconstructed value.
                let mut res = [(u0 << 1) | 1, u0 >> high_shift];
                lshift_2(&mut res, common_shift);
                return res;
            }
            let vgtu = if borrow { !0 } else { 0 };
            v0 = v0.wrapping_add(vgtu & diff0);
            u0 = (diff0 ^ vgtu).wrapping_sub(vgtu);
            let c = diff0.trailing_zeros();
            u0 = (u0 >> 1) >> c;
        }

        let g0 = Self::gcd_1((u0 << 1) | 1, (v0 << 1) | 1);
        let mut res = [g0, 0];
        lshift_2(&mut res, common_shift);
        res
    }

    /// In-register binary GCD, with an initial remainder for a large bit gap.
    #[must_use]
    pub const fn gcd_1(mut u: Limb, mut v: Limb) -> Limb {
        if u == 0 {
            return v;
        }
        if v == 0 {
            return u;
        }

        let u_tz = u.trailing_zeros();
        let v_tz = v.trailing_zeros();
        let common_shift = if u_tz < v_tz { u_tz } else { v_tz };
        u >>= u_tz;
        v >>= v_tz;

        let (larger, smaller) = if u < v { (v, u) } else { (u, v) };
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "profile validation bounds the scalar bit gap by 64, which fits u32 on every target"
        )]
        let division_shift = BINARY_EUCLID_DIVISION_SHIFT as u32;
        if let Some(high) = larger.checked_shr(division_shift)
            && high > smaller
        {
            // SAFETY: removing trailing zeros from nonzero inputs leaves
            // two positive odd operands, so the smaller divisor is nonzero.
            let remainder = unsafe { larger.checked_rem(smaller).unwrap_unchecked() };
            if remainder == 0 {
                return smaller.wrapping_shl(common_shift);
            }
            u = remainder >> remainder.trailing_zeros();
            v = smaller;
        }

        // Implicit low one-bits remove repeated parity tests from reduction.
        u >>= 1;
        v >>= 1;

        while u != v {
            let t = u.wrapping_sub(v);
            let vgtu = if u < v { !0 } else { 0 };
            v = v.wrapping_add(vgtu & t);
            u = (t ^ vgtu).wrapping_sub(vgtu);
            let c = t.trailing_zeros();
            u = (u >> 1) >> c;
        }

        ((u << 1) | 1).wrapping_shl(common_shift)
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "LIMB_BITS fits in u32"
)]
const fn trailing_zeros_2(u: [Limb; 2]) -> u32 {
    let limb_bits = LIMB_BITS as u32;
    if u[0] != 0 {
        u[0].trailing_zeros()
    } else if u[1] != 0 {
        // SAFETY: a nonzero high limb has fewer than LIMB_BITS trailing
        // zeros; their sum is below 2*LIMB_BITS <= 128.
        unsafe { u[1].trailing_zeros().unchecked_add(limb_bits) }
    } else {
        // SAFETY: twice the supported limb width is at most 128.
        unsafe { limb_bits.unchecked_mul(2) }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "LIMB_BITS fits in u32"
)]
const fn rshift_2(u: &mut [Limb; 2], shift: u32) {
    let limb_bits = LIMB_BITS as u32;
    if shift == 0 {
        return;
    }
    let word_shift = shift.wrapping_shr(LIMB_BITS.trailing_zeros()) as usize;
    // SAFETY: supported limb widths are positive powers of two.
    let bit_shift = shift & unsafe { limb_bits.unchecked_sub(1) };
    if word_shift == 1 {
        u[0] = u[1];
        u[1] = 0;
    } else if word_shift > 1 {
        *u = [0, 0];
        return;
    }
    if bit_shift > 0 {
        // SAFETY: masking and this branch give 0<bit_shift<LIMB_BITS.
        let carry_shift = unsafe { limb_bits.unchecked_sub(bit_shift) };
        u[0] = u[0].wrapping_shr(bit_shift) | u[1].wrapping_shl(carry_shift);
        u[1] = u[1].wrapping_shr(bit_shift);
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "LIMB_BITS fits in u32"
)]
pub const fn lshift_2(u: &mut [Limb; 2], shift: u32) {
    let limb_bits = LIMB_BITS as u32;
    if shift == 0 {
        return;
    }
    let word_shift = shift.wrapping_shr(LIMB_BITS.trailing_zeros()) as usize;
    // SAFETY: supported limb widths are positive powers of two.
    let bit_shift = shift & unsafe { limb_bits.unchecked_sub(1) };
    if word_shift == 1 {
        u[1] = u[0];
        u[0] = 0;
    } else if word_shift > 1 {
        *u = [0, 0];
        return;
    }
    if bit_shift > 0 {
        // SAFETY: masking and this branch give 0<bit_shift<LIMB_BITS.
        let carry_shift = unsafe { limb_bits.unchecked_sub(bit_shift) };
        u[1] = u[1].wrapping_shl(bit_shift) | u[0].wrapping_shr(carry_shift);
        u[0] = u[0].wrapping_shl(bit_shift);
    }
}
