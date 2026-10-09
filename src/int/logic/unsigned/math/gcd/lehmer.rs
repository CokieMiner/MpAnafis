//! Lehmer quotient simulation and full-operand matrix updates.
//!
//! Reference: D. H. Lehmer, "Euclid's Algorithm for Large Numbers",
//! The American Mathematical Monthly 45(4), 227-233, 1938.
//! DOI: 10.2307/2302609.

#![expect(
    unsafe_code,
    reason = "Lehmer updates use validated non-aliasing limb spans and initialize reserved destinations before committing matrix results"
)]

use core::cmp::max;

use super::{
    ArchKernels, DoubleLimb, Gcd, InternalMpUint, LEHMER_FUSED_UPDATE_MAX_LIMBS, LIMB_BITS, Limb,
};

#[derive(Clone, Copy, Debug)]
pub struct SignedLimbCarry {
    pub low: Limb,
    pub negative: bool,
}

impl SignedLimbCarry {
    pub const ZERO: Self = Self {
        low: 0,
        negative: false,
    };
}

impl Gcd {
    /// Applies one Lehmer batch with a per-call fused-kernel policy.
    ///
    /// `None` selects the single-pass fused kernel while the current operands
    /// fit the tuned fused width and the three-pass separate kernel above it,
    /// so shrinking operands transparently drop to one limb pass per batch
    /// instead of inheriting the entry-width policy for the whole reduction.
    /// `Some` forces a policy for tuning comparisons. The flag only affects
    /// equal-width pairs; asymmetric pairs share one partitioned path.
    #[expect(
        clippy::too_many_arguments,
        reason = "The dispatched Lehmer batch forwards the full transition matrix, both destinations, and the fused policy in one call."
    )]
    pub fn lehmer_update_dispatched(
        u: &mut InternalMpUint,
        v: &mut InternalMpUint,
        next_u: &mut InternalMpUint,
        next_v: &mut InternalMpUint,
        u0: Limb,
        v0: Limb,
        u1: Limb,
        v1: Limb,
        even: bool,
        force_fused: Option<bool>,
    ) -> bool {
        let fused = force_fused.unwrap_or_else(|| {
            max(u.limbs().len(), v.limbs().len()) <= LEHMER_FUSED_UPDATE_MAX_LIMBS
        });
        if fused {
            lehmer_update::<true>(u, v, next_u, next_v, u0, v0, u1, v1, even)
        } else {
            lehmer_update::<false>(u, v, next_u, next_v, u0, v0, u1, v1, even)
        }
    }
}

/// Applies one Lehmer transition matrix to full-precision operands in place.
#[expect(
    clippy::too_many_arguments,
    clippy::similar_names,
    reason = "The Lehmer transition matrix requires four conventional coefficients, an even flag, and paired u/v destination names."
)]
pub fn lehmer_update<const FUSED_EQUAL_LENGTH: bool>(
    u: &mut InternalMpUint,
    v: &mut InternalMpUint,
    next_u: &mut InternalMpUint,
    next_v: &mut InternalMpUint,
    u0: Limb,
    v0: Limb,
    u1: Limb,
    v1: Limb,
    even: bool,
) -> bool {
    let u_limbs = u.limbs();
    let v_limbs = v.limbs();
    let max_len = max(u_limbs.len(), v_limbs.len());
    if max_len == 0 {
        next_u.clear();
        next_v.clear();
        return true;
    }
    // Every kernel path initializes all max_len output limbs before
    // reading them, so output preparation requires no zero-fill.
    let mut write_u = next_u.prepare_limb_write(max_len);
    let mut write_v = next_v.prepare_limb_write(max_len);
    let dst_u = write_u.as_mut_ptr();
    let dst_v = write_v.as_mut_ptr();
    // SAFETY: `write_*` guarantee `max_len` writable limbs disjoint from the
    // borrowed sources; the kernel initializes every one of them and reports
    // trimmed lengths within `max_len`.
    let (next_u_len, next_v_len, ok) = unsafe {
        lehmer_update_raw::<FUSED_EQUAL_LENGTH>(
            u_limbs, v_limbs, dst_u, dst_v, max_len, u0, v0, u1, v1, even,
        )
    };
    // SAFETY: the kernel initialized all `max_len` limbs; committing exposes
    // them and the trimmed lengths only shorten the logical span.
    unsafe {
        let _ = write_u.commit();
        let _ = write_v.commit();
        next_u.set_len(next_u_len);
        next_v.set_len(next_v_len);
    }
    if ok {
        u.swap(next_u);
        v.swap(next_v);
    }
    ok
}

#[expect(
    clippy::too_many_arguments,
    clippy::similar_names,
    reason = "The Lehmer matrix update coordinates fused equal-length and general asymmetric passes with four conventional coefficients."
)]
unsafe fn lehmer_update_raw<const FUSED_EQUAL_LENGTH: bool>(
    u: &[Limb],
    v: &[Limb],
    next_u: *mut Limb,
    next_v: *mut Limb,
    max_len: usize,
    u0: Limb,
    v0: Limb,
    u1: Limb,
    v1: Limb,
    even: bool,
) -> (usize, usize, bool) {
    let u_len = u.len();
    let v_len = v.len();
    debug_assert_eq!(max_len, max(u_len, v_len), "destination has wrong length");

    let (u_pos_coeff, u_neg_coeff, v_pos_coeff, v_neg_coeff) = if even {
        (u0, v0, v1, u1)
    } else {
        (v0, u0, u1, v1)
    };

    // SAFETY: the caller guarantees `next_u`/`next_v` writable for `max_len`
    // limbs disjoint from the sources; each branch initializes all of them.
    let valid = unsafe {
        if u_len == v_len {
            if FUSED_EQUAL_LENGTH {
                // SAFETY: both sources and both destinations have exactly `max_len`
                // elements, and the separately owned destinations cannot overlap the
                // source values.
                sub_mul_two_streams_unchecked(
                    next_u,
                    next_v,
                    u.as_ptr(),
                    v.as_ptr(),
                    max_len,
                    u_pos_coeff,
                    u_neg_coeff,
                    v_pos_coeff,
                    v_neg_coeff,
                    even,
                )
            } else {
                // SAFETY: both sources and both destinations have exactly `max_len`
                // elements, and the separately owned destinations cannot overlap the
                // source values.
                sub_mul_separate_passes(
                    next_u,
                    next_v,
                    u.as_ptr(),
                    v.as_ptr(),
                    max_len,
                    u_pos_coeff,
                    u_neg_coeff,
                    v_pos_coeff,
                    v_neg_coeff,
                    even,
                )
            }
        } else {
            Gcd::lehmer_update_unequal(
                u,
                v,
                next_u,
                next_v,
                u_pos_coeff,
                u_neg_coeff,
                v_pos_coeff,
                v_neg_coeff,
                even,
            )
        }
    };

    let mut new_u_len = max_len;
    while new_u_len > 0 {
        // SAFETY: `new_u_len` is in `1..=max_len` over initialized destinations.
        if unsafe { *next_u.add(new_u_len.wrapping_sub(1)) } != 0 {
            break;
        }
        new_u_len = new_u_len.wrapping_sub(1);
    }

    let mut new_v_len = max_len;
    while new_v_len > 0 {
        // SAFETY: `new_v_len` is in `1..=max_len` over initialized destinations.
        if unsafe { *next_v.add(new_v_len.wrapping_sub(1)) } != 0 {
            break;
        }
        new_v_len = new_v_len.wrapping_sub(1);
    }

    (new_u_len, new_v_len, valid)
}

/// Fused two-destination multiply-subtract streaming kernel:
/// evaluates both Lehmer stream outputs in a single pass over the source operands:
/// `dst_u[i] = u_pos[i] * u_pos_coeff - u_neg[i] * u_neg_coeff + carry_u`
/// `dst_v[i] = v_pos[i] * v_pos_coeff - v_neg[i] * v_neg_coeff + carry_v`
/// across `len` limbs, returning whether both final corrections are valid (zero).
///
/// # Safety
///
/// - `src_u` and `src_v` must be valid for reads of `len` elements.
/// - `dst_u` and `dst_v` must be valid for writes of `len` elements and must not alias each other or sources.
#[expect(
    clippy::as_conversions,
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "Critical primitive leaf arithmetic kernel; DoubleLimb casts are exact widening and consumes dual streams with 4 coefficients"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "Products are split at LIMB_BITS into exact Limb halves"
    )
)]
#[inline(always)]
unsafe fn sub_mul_two_streams_unchecked(
    dst_u: *mut Limb,
    dst_v: *mut Limb,
    src_u: *const Limb,
    src_v: *const Limb,
    len: usize,
    u_pos_coeff: Limb,
    u_neg_coeff: Limb,
    v_pos_coeff: Limb,
    v_neg_coeff: Limb,
    even: bool,
) -> bool {
    let (ptr1, ptr2) = if even { (src_u, src_v) } else { (src_v, src_u) };
    let u_pos_coeff_wide = u_pos_coeff as DoubleLimb;
    let u_neg_coeff_wide = u_neg_coeff as DoubleLimb;
    let v_pos_coeff_wide = v_pos_coeff as DoubleLimb;
    let v_neg_coeff_wide = v_neg_coeff as DoubleLimb;

    // For coefficient c, carry < c (or carry=0 when c=0) is inductive:
    // (B-1)*c + carry < B*c. All four product carries fit one limb.
    let mut u_pos_carry: Limb = 0;
    let mut u_neg_carry: Limb = 0;
    let mut u_borrow = false;

    let mut v_pos_carry: Limb = 0;
    let mut v_neg_carry: Limb = 0;
    let mut v_borrow = false;

    for i in 0..len {
        // SAFETY: caller guarantees src_u and src_v are valid for len reads,
        // and dst_u, dst_v are valid for len writes with no aliasing.
        let (p1, p2) = unsafe { (*ptr1.add(i) as DoubleLimb, *ptr2.add(i) as DoubleLimb) };

        // SAFETY: p1,p2 <= B-1 and each carry is below its coefficient.
        // Each product plus carry is < B*c <= B*(B-1), fitting DoubleLimb.
        let (u_pos_prod, u_neg_prod) = unsafe {
            (
                p1.unchecked_mul(u_pos_coeff_wide)
                    .unchecked_add(u_pos_carry as DoubleLimb),
                p2.unchecked_mul(u_neg_coeff_wide)
                    .unchecked_add(u_neg_carry as DoubleLimb),
            )
        };
        // SAFETY: the inductive product bound above puts both high halves
        // in Limb on every pointer width, including a zero coefficient.
        unsafe {
            u_pos_carry = Limb::try_from(u_pos_prod >> LIMB_BITS).unwrap_unchecked();
            u_neg_carry = Limb::try_from(u_neg_prod >> LIMB_BITS).unwrap_unchecked();
        }
        let (u_diff, u_b1) = (u_pos_prod as Limb).overflowing_sub(u_neg_prod as Limb);
        let (u_low, u_b2) = u_diff.overflowing_sub(Limb::from(u_borrow));
        u_borrow = u_b1 || u_b2;

        // SAFETY: the same coefficient/carry bound gives exact DoubleLimb
        // products and sums for the independent second stream.
        let (v_pos_prod, v_neg_prod) = unsafe {
            (
                p2.unchecked_mul(v_pos_coeff_wide)
                    .unchecked_add(v_pos_carry as DoubleLimb),
                p1.unchecked_mul(v_neg_coeff_wide)
                    .unchecked_add(v_neg_carry as DoubleLimb),
            )
        };
        // SAFETY: the same coefficient/carry bound applies independently to
        // the second stream, so its high halves also fit Limb.
        unsafe {
            v_pos_carry = Limb::try_from(v_pos_prod >> LIMB_BITS).unwrap_unchecked();
            v_neg_carry = Limb::try_from(v_neg_prod >> LIMB_BITS).unwrap_unchecked();
        }
        let (v_diff, v_b1) = (v_pos_prod as Limb).overflowing_sub(v_neg_prod as Limb);
        let (v_low, v_b2) = v_diff.overflowing_sub(Limb::from(v_borrow));
        v_borrow = v_b1 || v_b2;

        // SAFETY: caller guarantees dst_u and dst_v have len writable elements.
        unsafe {
            *dst_u.add(i) = u_low;
            *dst_v.add(i) = v_low;
        }
    }

    // SAFETY: each negative product carry is at most coefficient-1 <= B-2
    // (or zero). Its binary subtraction borrow therefore fits in the same
    // limb. Equality of the positive carry and this sum is exactly the
    // condition that the full signed result has zero high correction.
    unsafe {
        let net_u = u_neg_carry.unchecked_add(Limb::from(u_borrow));
        let net_v = v_neg_carry.unchecked_add(Limb::from(v_borrow));
        u_pos_carry == net_u && v_pos_carry == net_v
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Consumes dual source pointers and destinations with four conventional coefficients"
)]
#[expect(
    clippy::as_conversions,
    reason = "limb products widen exactly to DoubleLimb; low halves and shifted high halves are the two base-B digits on every pointer width"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "each product is split at LIMB_BITS into its exact low and high base-B digits"
    )
)]
#[inline]
unsafe fn sub_mul_separate_passes(
    dst_u: *mut Limb,
    dst_v: *mut Limb,
    src_u: *const Limb,
    src_v: *const Limb,
    len: usize,
    u_pos_coeff: Limb,
    u_neg_coeff: Limb,
    v_pos_coeff: Limb,
    v_neg_coeff: Limb,
    even: bool,
) -> bool {
    let (ptr1, ptr2) = if even { (src_u, src_v) } else { (src_v, src_u) };

    // Two independent multiplication carries share the loop. For coefficient c,
    // (B-1)*c + carry < B*c preserves carry < c (or zero when c=0), so each
    // carry occupies one limb before the subtraction passes consume the output.
    let mut carry_u: Limb = 0;
    let mut carry_v: Limb = 0;
    for index in 0..len {
        // SAFETY: the caller supplies len initialized source limbs and disjoint
        // writable destinations. This loop initializes both output spans. The
        // carry bound above makes each high product fit Limb on every target.
        unsafe {
            let product_u = (*ptr1.add(index) as DoubleLimb)
                .unchecked_mul(u_pos_coeff as DoubleLimb)
                .unchecked_add(carry_u as DoubleLimb);
            let product_v = (*ptr2.add(index) as DoubleLimb)
                .unchecked_mul(v_pos_coeff as DoubleLimb)
                .unchecked_add(carry_v as DoubleLimb);
            dst_u.add(index).write(product_u as Limb);
            dst_v.add(index).write(product_v as Limb);
            carry_u = Limb::try_from(product_u >> LIMB_BITS).unwrap_unchecked();
            carry_v = Limb::try_from(product_v >> LIMB_BITS).unwrap_unchecked();
        }
    }
    // Select the process-stable architecture kernel once for both streams.
    let sub_mul = ArchKernels::selected_sub_mul_limbs_unchecked();

    // Stream u: dst_u = ptr1 * u_pos_coeff - ptr2 * u_neg_coeff.
    // SAFETY: ptr2 is valid for len reads and dst_u for len initialized,
    // nonaliasing reads/writes. The selector establishes target prerequisites.
    let (carry_neg_u, borrow_u) = unsafe { sub_mul(dst_u, ptr2, len, u_neg_coeff) };
    // SAFETY: the product carry is at most u_neg_coeff-1 (or zero), and
    // the kernel's borrow is binary. Their sum is at most Limb::MAX.
    let net_u = unsafe { carry_neg_u.unchecked_add(borrow_u) };
    let valid_u = carry_u == net_u;

    // Stream v: dst_v = ptr2 * v_pos_coeff - ptr1 * v_neg_coeff.
    // SAFETY: ptr1 is valid for len reads and dst_v for len initialized,
    // nonaliasing reads/writes; the same selected target remains valid.
    let (carry_neg_v, borrow_v) = unsafe { sub_mul(dst_v, ptr1, len, v_neg_coeff) };
    // SAFETY: the same scalar-product bound and binary borrow apply to v.
    let net_v = unsafe { carry_neg_v.unchecked_add(borrow_v) };
    let valid_v = carry_v == net_v;

    valid_u && valid_v
}

/// Multiplies `src[0..len]` by `scalar` and writes the product directly into `dst[0..len]`,
/// returning the high carry limb.
///
/// Eliminates the preliminary zero-initialization memset required by `add_mul_limbs_unchecked`.
///
/// # Safety
///
/// - `dst` must be valid for writes of `len` limbs.
/// - `src` must be valid for reads of `len` limbs.
/// - `dst` and `src` must not alias.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "Critical leaf kernel for Lehmer vector updates; exact widening and splitting"
)]
impl Gcd {
    #[cfg_attr(
        target_pointer_width = "32",
        expect(
            clippy::cast_possible_truncation,
            reason = "DoubleLimb (u64) is narrowed to Limb (u32) on 32-bit targets after shift"
        )
    )]
    #[inline(always)]
    pub unsafe fn mul_limbs_scalar_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        scalar: Limb,
    ) -> Limb {
        let scalar_wide = scalar as DoubleLimb;
        let mut carry: DoubleLimb = 0;
        for i in 0..len {
            // SAFETY: caller guarantees src is valid for len reads and dst for len writes.
            let value = unsafe { *src.add(i) } as DoubleLimb;
            let product = value.wrapping_mul(scalar_wide).wrapping_add(carry);
            // SAFETY: i < len, within caller-guaranteed dst allocation.
            unsafe {
                *dst.add(i) = product as Limb;
            }
            carry = product >> LIMB_BITS;
        }
        carry as Limb
    }
}
