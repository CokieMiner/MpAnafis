//! Binary GCD fallback for short operands.

use core::{cmp::Ordering, mem::swap};

use super::{ArchKernels, Gcd, InternalMpUint, LIMB_BITS, Limb};

/// Fused trailing-zeros count + right-shift in a single pass over the limbs.
///
/// Equivalent to `n.shr_assign(n.trailing_zeros())` but scans the limbs once
/// instead of twice (once for `trailing_zeros`, once for `shr_assign`).
#[allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "get_unchecked_mut avoids bounds checks; LIMB_BITS fits in u32 even on 16-bit targets (where LIMB_BITS is 16): avoids checked conversions and is branchless."
)]
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "Inlining this helper eliminates call overhead and exposes loop invariants to the optimizer."
)]
fn shr_trailing_zeros_assign(n: &mut InternalMpUint) {
    let limbs = n.limbs();
    let len = limbs.len();
    if len == 0 {
        return;
    }

    // Find the first non-zero limb and count trailing zeros within it
    let mut word_shift: usize = 0;
    let mut bit_shift: u32 = 0;
    for i in 0..len {
        // SAFETY: i < len
        let limb = unsafe { *limbs.get_unchecked(i) };
        if limb == 0 {
            word_shift = word_shift.wrapping_add(1);
        } else {
            bit_shift = limb.trailing_zeros();
            break;
        }
    }

    if word_shift >= len {
        // Value is zero
        // SAFETY: setting len to 0 is safe
        unsafe {
            n.set_len(0);
        }
        n.normalize();
        return;
    }

    let total_shift = word_shift
        .wrapping_mul(LIMB_BITS)
        .wrapping_add(bit_shift as usize);
    if total_shift == 0 {
        return;
    }

    // Perform the combined word+bit shift.
    let new_len = len.wrapping_sub(word_shift);

    if word_shift > 0 {
        let mut_limbs = n.limbs_mut();
        // Word shift: move limbs down
        for i in 0..new_len {
            // SAFETY: i + word_shift < len, i < new_len
            unsafe {
                *mut_limbs.get_unchecked_mut(i) =
                    *mut_limbs.get_unchecked(i.wrapping_add(word_shift));
            }
        }
    }

    // SAFETY: new_len <= len
    unsafe {
        n.set_len(new_len);
    }

    if bit_shift > 0 {
        // Use arch-optimized kernel for the bit-level right shift
        let mut_limbs = n.limbs_mut();
        // SAFETY: limbs pointer is valid for new_len elements; 0 < bit_shift < LIMB_BITS
        unsafe {
            let _ = ArchKernels::rshift_unchecked(mut_limbs.as_mut_ptr(), new_len, bit_shift);
        }
    }

    n.normalize();
}

/// Completes binary GCD after the common power-of-two factor has been removed.
///
/// Leaves the odd-part GCD in `u` and zero in `v`.
#[allow(
    unsafe_code,
    reason = "Bypassing vector length checks during final Stein GCD limb truncation to avoid branchy checks."
)]
impl Gcd {
    pub fn binary_gcd_odd_part_assign(u: &mut InternalMpUint, v: &mut InternalMpUint) {
        shr_trailing_zeros_assign(u);
        shr_trailing_zeros_assign(v);

        loop {
            // Fallback to primitive 64-bit GCD when both fit in a single limb.
            if u.limbs().len() <= 1 && v.limbs().len() <= 1 {
                let ans = Self::gcd_limb(
                    *u.limbs().first().unwrap_or(&0),
                    *v.limbs().first().unwrap_or(&0),
                );
                if ans != 0 {
                    u.clone_from_slice(&[ans]);
                } else if u.is_zero() {
                    #[allow(
                        unsafe_code,
                        reason = "Bypassing vector length checks during final Stein GCD limb truncation to avoid branchy checks."
                    )]
                    // SAFETY: setting length to 0 is always safe
                    unsafe {
                        u.set_len(0);
                    }
                    break;
                }
                #[allow(
                    unsafe_code,
                    reason = "Bypassing vector length checks during final Stein GCD limb truncation to avoid branchy checks."
                )]
                // SAFETY: setting length to 0 is always safe
                unsafe {
                    v.set_len(0);
                }
                u.normalize();
                v.normalize();
                break;
            }

            if (*u).cmp(&*v) == Ordering::Greater {
                swap(u, v);
            }

            v.sub_assign(u);
            if v.is_zero() {
                break;
            }
            shr_trailing_zeros_assign(v);
        }
    }

    pub const fn gcd_limb(mut a: Limb, mut b: Limb) -> Limb {
        while b != 0 {
            let next_a = b;
            b = match a.checked_rem(b) {
                Some(rem) => rem,
                None => return a,
            };
            a = next_a;
        }
        a
    }
}
