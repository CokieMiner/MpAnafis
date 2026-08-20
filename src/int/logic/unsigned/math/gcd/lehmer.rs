//! Lehmer quotient simulation and full-operand matrix updates.

use core::cmp::max;

use alloc::vec::Vec;

use super::{ArchKernels, DoubleLimb, Gcd, InternalMpUint, LIMB_BITS, Limb};

impl Gcd {
    pub fn extract_top_limb(u_limbs: &[Limb], v_limbs: &[Limb]) -> (Limb, Limb) {
        let u_len = u_limbs.len();
        let v_len = v_limbs.len();
        if u_len == 0 {
            return (0, 0);
        }
        let top_u = get_limb(u_limbs, u_len.wrapping_sub(1));
        let lz = top_u.leading_zeros();
        let next_u = if u_len > 1 {
            get_limb(u_limbs, u_len.wrapping_sub(2))
        } else {
            0
        };

        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS fits in u32 even on 16-bit targets (where LIMB_BITS is 16): avoids range checks and compiles to branchless register truncation."
        )]
        let limb_bits_u32 = LIMB_BITS as u32;

        let u_hat = if lz == 0 {
            top_u
        } else {
            top_u.wrapping_shl(lz) | next_u.wrapping_shr(limb_bits_u32.wrapping_sub(lz))
        };

        let limb_diff = u_len.wrapping_sub(v_len);
        let v_hat = if limb_diff > 1 || v_len == 0 {
            0
        } else if limb_diff == 1 {
            // v has one fewer limb than u; v's top limb must be right-shifted
            // by (LIMB_BITS - lz) to align with u's top bits.
            let top_v = get_limb(v_limbs, v_len.wrapping_sub(1));
            // v_hat: top_v right-shifted to align with u_hat's scale.
            // u_hat = top_u << lz | next_u >> (LIMB_BITS - lz)
            // v_hat at same scale = top_v >> (LIMB_BITS - lz)
            // When lz == 0, u's full top limb is used, v's contribution is top_v >> 64 = 0.
            if lz == 0 {
                0
            } else {
                top_v.wrapping_shr(limb_bits_u32.wrapping_sub(lz))
            }
        } else {
            let top_v = get_limb(v_limbs, v_len.wrapping_sub(1));
            let next_v = if v_len > 1 {
                get_limb(v_limbs, v_len.wrapping_sub(2))
            } else {
                0
            };
            if lz == 0 {
                top_v
            } else {
                top_v.wrapping_shl(lz) | next_v.wrapping_shr(limb_bits_u32.wrapping_sub(lz))
            }
        };

        (u_hat, v_hat)
    }

    /// Extracts the two leading normalized limbs used by wide Lehmer simulation.
    ///
    /// Both values are aligned with the leading limb of `u`.  Keeping the second
    /// limb makes the quotient sequence stable for more Euclidean steps while the
    /// returned transition coefficients remain single-limb values, which is the
    /// representation consumed by [`Gcd::lehmer_update`].
    pub fn extract_top_two_limbs(u_limbs: &[Limb], v_limbs: &[Limb]) -> (DoubleLimb, DoubleLimb) {
        let u_len = u_limbs.len();
        let v_len = v_limbs.len();
        if u_len < 2 || v_len < 2 || u_len != v_len {
            return (0, 0);
        }

        let top_u = get_limb(u_limbs, u_len.wrapping_sub(1));
        let shift = top_u.leading_zeros();
        let top_v = get_limb(v_limbs, v_len.wrapping_sub(1));

        let u_hat = normalize_top_two(
            top_u,
            get_limb(u_limbs, u_len.wrapping_sub(2)),
            get_limb(u_limbs, u_len.wrapping_sub(3)),
            shift,
        );
        let v_hat = normalize_top_two(
            top_v,
            get_limb(v_limbs, v_len.wrapping_sub(2)),
            get_limb(v_limbs, v_len.wrapping_sub(3)),
            shift,
        );
        (u_hat, v_hat)
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The normalized high and low limbs are each Limb-sized and are packed into DoubleLimb by construction."
)]
const fn normalize_top_two(top: Limb, next: Limb, third: Limb, shift: u32) -> DoubleLimb {
    if shift == 0 {
        return ((top as DoubleLimb) << LIMB_BITS) | next as DoubleLimb;
    }
    let lower_shift = (LIMB_BITS as u32).wrapping_sub(shift);
    let high = top.wrapping_shl(shift) | next.wrapping_shr(lower_shift);
    let low = next.wrapping_shl(shift) | third.wrapping_shr(lower_shift);
    ((high as DoubleLimb) << LIMB_BITS) | low as DoubleLimb
}

#[allow(
    unsafe_code,
    reason = "checked_div is unwrapped only inside the loop branch that proves v_hat != 0."
)]
impl Gcd {
    pub const fn lehmer_simulate(
        mut u_hat: Limb,
        mut v_hat: Limb,
    ) -> (Limb, Limb, Limb, Limb, bool) {
        let mut u_0: Limb = 1;
        let mut v_0: Limb = 0;
        let mut u_1: Limb = 0;
        let mut v_1: Limb = 1;
        let mut even = true;

        loop {
            if v_hat == 0 {
                break;
            }

            // Optimize division with subtractions for small quotients
            let mut q: Limb = 0;
            let mut rem = u_hat;
            if rem >= v_hat {
                rem = rem.wrapping_sub(v_hat);
                q = q.wrapping_add(1);
                if rem >= v_hat {
                    rem = rem.wrapping_sub(v_hat);
                    q = q.wrapping_add(1);
                    if rem >= v_hat {
                        rem = rem.wrapping_sub(v_hat);
                        q = q.wrapping_add(1);
                        if rem >= v_hat {
                            // SAFETY: v_hat is non-zero
                            let div = unsafe { rem.checked_div(v_hat).unwrap_unchecked() };
                            q = q.wrapping_add(div);
                            rem = rem.wrapping_sub(div.wrapping_mul(v_hat));
                        }
                    }
                }
            }

            let update_u = u_0.wrapping_add(q.wrapping_mul(u_1));
            let update_v = v_0.wrapping_add(q.wrapping_mul(v_1));

            if even {
                if v_hat < v_1 || rem < update_u {
                    break;
                }
            } else if v_hat < u_1 || rem < update_v {
                break;
            }

            u_hat = v_hat;
            v_hat = rem;

            u_0 = update_u;
            v_0 = update_v;

            let temp_u = u_0;
            u_0 = u_1;
            u_1 = temp_u;

            let temp_v = v_0;
            v_0 = v_1;
            v_1 = temp_v;

            even = !even;
        }

        (u_0, v_0, u_1, v_1, even)
    }

    /// Simulates Lehmer steps from two leading limbs.
    ///
    /// The matrix coefficients are capped at `Limb::MAX`; once a coefficient would
    /// need a wider representation, the valid prefix accumulated so far is
    /// returned and the caller can resume with the full operands.
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "The coefficient bound is checked against Limb::MAX before narrowing from DoubleLimb."
    )]
    pub const fn lehmer_simulate_wide(
        mut u_hat: DoubleLimb,
        mut v_hat: DoubleLimb,
    ) -> (Limb, Limb, Limb, Limb, bool) {
        let mut u_0: Limb = 1;
        let mut v_0: Limb = 0;
        let mut u_1: Limb = 0;
        let mut v_1: Limb = 1;
        let mut even = true;

        loop {
            if v_hat == 0 {
                break;
            }

            let (q, rem) = if u_hat >= v_hat {
                let Some(q) = u_hat.checked_div(v_hat) else {
                    break;
                };
                let Some(rem) = u_hat.checked_rem(v_hat) else {
                    break;
                };
                (q, rem)
            } else {
                (0, u_hat)
            };

            if q > Limb::MAX as DoubleLimb {
                break;
            }

            let Some(product_first) = q.checked_mul(u_1 as DoubleLimb) else {
                break;
            };
            let Some(product_second) = q.checked_mul(v_1 as DoubleLimb) else {
                break;
            };
            let Some(candidate_first) = (u_0 as DoubleLimb).checked_add(product_first) else {
                break;
            };
            let Some(candidate_second) = (v_0 as DoubleLimb).checked_add(product_second) else {
                break;
            };
            if candidate_first > Limb::MAX as DoubleLimb
                || candidate_second > Limb::MAX as DoubleLimb
            {
                break;
            }

            let update_u = candidate_first as Limb;
            let update_v = candidate_second as Limb;
            if even {
                if v_hat < v_1 as DoubleLimb || rem < update_u as DoubleLimb {
                    break;
                }
            } else if v_hat < u_1 as DoubleLimb || rem < update_v as DoubleLimb {
                break;
            }

            u_hat = v_hat;
            v_hat = rem;
            u_0 = update_u;
            v_0 = update_v;

            let temp_u = u_0;
            u_0 = u_1;
            u_1 = temp_u;
            let temp_v = v_0;
            v_0 = v_1;
            v_1 = temp_v;
            even = !even;
        }

        (u_0, v_0, u_1, v_1, even)
    }
}

#[allow(
    unsafe_code,
    reason = "The only caller resizes both u and v to max(u_len, v_len); every unchecked range is bounded by those original lengths or max_len, and backup lengths equal the copied original prefixes."
)]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The Lehmer matrix has four conventional coefficients plus an even flag and two backups; Limb products fit u128/i128 on supported widths and are split back to one Limb."
)]
fn lehmer_update_slice(
    u: &mut [Limb],
    u_len: &mut usize,
    v: &mut [Limb],
    v_len: &mut usize,
    u_backup: &mut Vec<Limb>,
    v_backup: &mut Vec<Limb>,
    u0: Limb,
    v0: Limb,
    u1: Limb,
    v1: Limb,
    even: bool,
) -> bool {
    let max_len = max(*u_len, *v_len);
    if max_len == 0 {
        return true;
    }

    u_backup.resize(max_len, 0);
    v_backup.resize(max_len, 0);

    let mut carry_u: i128 = 0;
    let mut carry_v: i128 = 0;
    let limb_bits_u32 = LIMB_BITS as u32;

    let (u_pos_coeff, u_neg_coeff, v_pos_coeff, v_neg_coeff) = if even {
        (u0, v0, v1, u1)
    } else {
        (v0, u0, u1, v1)
    };

    let u_orig_len = *u_len;
    let v_orig_len = *v_len;

    for i in 0..max_len {
        let ui = if i < u_orig_len {
            // SAFETY: i < u_orig_len <= u.len()
            unsafe { *u.get_unchecked(i) }
        } else {
            0
        };
        let vi = if i < v_orig_len {
            // SAFETY: i < v_orig_len <= v.len()
            unsafe { *v.get_unchecked(i) }
        } else {
            0
        };

        let (u_pos_val, u_neg_val) = if even { (ui, vi) } else { (vi, ui) };
        let (u_pos_lo, u_pos_hi) = ArchKernels::mul_limb_lo_hi(u_pos_val, u_pos_coeff);
        let (u_neg_lo, u_neg_hi) = ArchKernels::mul_limb_lo_hi(u_neg_val, u_neg_coeff);
        let u_lo_diff = (u_pos_lo as i128)
            .wrapping_sub(u_neg_lo as i128)
            .wrapping_add(carry_u);
        let u_lo = (u_lo_diff.cast_unsigned() & Limb::MAX as u128) as Limb;
        let u_hi_carry = u_lo_diff.wrapping_shr(limb_bits_u32);
        carry_u = (u_pos_hi as i128)
            .wrapping_sub(u_neg_hi as i128)
            .wrapping_add(u_hi_carry);

        let (v_pos_val, v_neg_val) = if even { (vi, ui) } else { (ui, vi) };
        let (v_pos_lo, v_pos_hi) = ArchKernels::mul_limb_lo_hi(v_pos_val, v_pos_coeff);
        let (v_neg_lo, v_neg_hi) = ArchKernels::mul_limb_lo_hi(v_neg_val, v_neg_coeff);
        let v_lo_diff = (v_pos_lo as i128)
            .wrapping_sub(v_neg_lo as i128)
            .wrapping_add(carry_v);
        let v_lo = (v_lo_diff.cast_unsigned() & Limb::MAX as u128) as Limb;
        let v_hi_carry = v_lo_diff.wrapping_shr(limb_bits_u32);
        carry_v = (v_pos_hi as i128)
            .wrapping_sub(v_neg_hi as i128)
            .wrapping_add(v_hi_carry);

        // SAFETY: u_backup and v_backup were resized to max_len; i < max_len
        unsafe {
            *u_backup.get_unchecked_mut(i) = u_lo;
            *v_backup.get_unchecked_mut(i) = v_lo;
        }
    }

    if carry_u != 0 || carry_v != 0 {
        return false;
    }

    // SAFETY: u and v have at least max_len allocated capacity; u_backup and v_backup have max_len elements.
    unsafe {
        u.get_unchecked_mut(..max_len).copy_from_slice(u_backup);
        v.get_unchecked_mut(..max_len).copy_from_slice(v_backup);
    }

    let mut new_u_len = max_len;
    while new_u_len > 0 {
        // SAFETY: new_u_len <= max_len and new_u_len > 0, so index new_u_len - 1 is < max_len
        if unsafe { *u.get_unchecked(new_u_len.wrapping_sub(1)) } != 0 {
            break;
        }
        new_u_len = new_u_len.wrapping_sub(1);
    }
    *u_len = new_u_len;

    let mut new_v_len = max_len;
    while new_v_len > 0 {
        // SAFETY: new_v_len <= max_len and new_v_len > 0, so index new_v_len - 1 is < max_len
        if unsafe { *v.get_unchecked(new_v_len.wrapping_sub(1)) } != 0 {
            break;
        }
        new_v_len = new_v_len.wrapping_sub(1);
    }
    *v_len = new_v_len;

    true
}

#[allow(
    unsafe_code,
    reason = "max_len is the maximum current operand length; both buffers are expanded to exactly max_len, newly exposed tails are initialized by lehmer_update_slice, and final scanned lengths remain <= max_len."
)]
#[allow(
    clippy::too_many_arguments,
    reason = "The Lehmer transition matrix requires four coefficients, an even flag, and two reusable backup buffers."
)]
impl Gcd {
    pub fn lehmer_update(
        u: &mut InternalMpUint,
        v: &mut InternalMpUint,
        u_backup: &mut Vec<Limb>,
        v_backup: &mut Vec<Limb>,
        u0: Limb,
        v0: Limb,
        u1: Limb,
        v1: Limb,
        even: bool,
    ) -> bool {
        let max_len = max(u.limbs().len(), v.limbs().len());
        let mut u_len = u.limbs().len();
        let mut v_len = v.limbs().len();
        // SAFETY: `max_len` is at least each current initialized length. The
        // callee copies those prefixes, then initializes every newly exposed
        // tail element before reading it.
        let (u_slice, v_slice) = unsafe {
            (
                u.ensure_capacity_set_len_get_limbs(max_len),
                v.ensure_capacity_set_len_get_limbs(max_len),
            )
        };
        let ok = lehmer_update_slice(
            u_slice, &mut u_len, v_slice, &mut v_len, u_backup, v_backup, u0, v0, u1, v1, even,
        );
        // SAFETY: `u_len` and `v_len` are obtained by scanning downward from
        // `max_len`, so both are within the already initialized buffers.
        unsafe {
            let _ = u.ensure_capacity_set_len_get_limbs(u_len);
            let _ = v.ensure_capacity_set_len_get_limbs(v_len);
        }
        ok
    }
}

#[allow(
    unsafe_code,
    reason = "The unchecked limb access occurs only in the branch that proves idx < limbs.len()."
)]
fn get_limb(limbs: &[Limb], idx: usize) -> Limb {
    if idx < limbs.len() {
        // SAFETY: this branch proves `idx < limbs.len()`.
        unsafe { *limbs.get_unchecked(idx) }
    } else {
        0
    }
}
