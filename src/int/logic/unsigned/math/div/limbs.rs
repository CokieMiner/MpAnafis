//! Limb-slice primitives shared by the division kernels.
//!
//! These are the innermost loops the Knuth and Burnikel-Ziegler drivers reach
//! for: a single-limb division, and the add/subtract-with-propagation steps that
//! maintain a running remainder in place.

use core::ptr::{copy_nonoverlapping, write_bytes};

use alloc::vec::Vec;

use super::{Addition, ArchKernels, Division, InternalMpUint, LIMB_BITS, Limb};

impl Division {
    /// Divides a multi-limb numerator by a single limb divisor.
    ///
    /// Returns the remainder, writing the quotient limbs into `quotient_out`
    /// when `WRITE_QUOTIENT` is set.
    ///
    /// # Safety
    ///
    /// `den_v` must be non-zero.
    pub fn div_rem_1<const WRITE_QUOTIENT: bool>(
        num_limbs: &[Limb],
        den_v: Limb,
        quotient_out: &mut InternalMpUint,
    ) -> Limb {
        let len = num_limbs.len();
        if WRITE_QUOTIENT {
            // SAFETY: the loop below writes all `len` elements before any read.
            let _ = unsafe { quotient_out.ensure_capacity_set_len_get_limbs(len) };
        }
        let mut rem: Limb = 0;
        for i in (0..len).rev() {
            // SAFETY: i is always within bounds [0, len)
            let limb = unsafe { *num_limbs.get_unchecked(i) };
            // SAFETY: divisor den_v is non-zero (caller contract); rem < den_v holds by induction
            let (q_val, new_rem) = unsafe { ArchKernels::divrem_1_unchecked(limb, rem, den_v) };
            rem = new_rem;
            if WRITE_QUOTIENT {
                // SAFETY: `quotient_out` was initialized to `len` limbs above and
                // the reversed loop keeps `i` in `0..len`.
                unsafe {
                    *quotient_out.limbs_mut().get_unchecked_mut(i) = q_val;
                }
            }
        }
        if WRITE_QUOTIENT {
            quotient_out.normalize();
        }
        rem
    }

    /// Subtracts `q_hat * v_limbs` from `u_window` in place, returning the
    /// borrow out of the window's top limb.
    ///
    /// `u_window.len()` must be at least `v_limbs.len() + 1`.
    #[allow(
        clippy::inline_always,
        reason = "Inlining Knuth Algorithm D helpers eliminates function call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[inline(always)]
    pub fn mul_sub_in_place(
        u_window: &mut [Limb],
        v_limbs: &[Limb],
        q_hat: Limb,
        sub_mul: unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb),
    ) -> Limb {
        let v_len = v_limbs.len();
        // SAFETY: Caller ensures u_window.len() >= v_len + 1, and pointers are valid
        let (carry, borrow) =
            unsafe { sub_mul(u_window.as_mut_ptr(), v_limbs.as_ptr(), v_len, q_hat) };
        // SAFETY: v_len < u_window.len() (guaranteed by caller)
        let u_val = unsafe { *u_window.get_unchecked(v_len) };
        let (diff1, b1) = u_val.overflowing_sub(carry);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        // SAFETY: v_len < u_window.len()
        unsafe {
            *u_window.get_unchecked_mut(v_len) = diff2;
        }
        Limb::from(b1 | b2)
    }

    /// Adds `v_limbs` back into `u_window`, absorbing the carry into the
    /// window's top limb. Used by the Algorithm D add-back correction.
    #[allow(
        clippy::inline_always,
        reason = "Inlining Knuth Algorithm D helpers eliminates function call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[inline(always)]
    pub fn add_in_place(u_window: &mut [Limb], v_limbs: &[Limb]) {
        let v_len = v_limbs.len();
        let carry = Addition::add_slice_in_place(u_window, v_limbs);

        if carry != 0 {
            // SAFETY: v_len < u_window.len() (guaranteed by caller)
            unsafe {
                let val = *u_window.get_unchecked(v_len);
                *u_window.get_unchecked_mut(v_len) = val.wrapping_add(carry);
            }
        }
    }

    /// Adds `src` into the low limbs of `dst`, propagating the carry through the
    /// remaining limbs. Returns the carry out of `dst`.
    pub fn add_limbs_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        let src_len = src.len();
        if src_len == 0 {
            return 0;
        }
        let carry = Addition::add_slice_in_place(dst, src);
        if carry == 0 {
            return 0;
        }
        // SAFETY: dst has length strictly >= src_len by check above.
        let dst_upper = unsafe { dst.get_unchecked_mut(src_len..) };
        Addition::propagate_carry(dst_upper, carry)
    }

    /// Subtracts `src` from the low limbs of `dst`, propagating the borrow
    /// through the remaining limbs. Returns the borrow out of `dst`.
    pub fn sub_limbs_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        let src_len = src.len();
        if src_len == 0 {
            return 0;
        }
        let borrow = Addition::sub_slice_in_place(dst, src);
        if borrow == 0 {
            return 0;
        }
        // SAFETY: dst has length strictly >= src_len by check above.
        let dst_upper = unsafe { dst.get_unchecked_mut(src_len..) };
        Addition::propagate_borrow(dst_upper, borrow)
    }

    /// Normalizes a limb slice by shifting it left into `out`.
    #[allow(
        unsafe_code,
        reason = "The validated shift loop uses raw pointer access to initialize reserved output storage without redundant bounds checks."
    )]
    pub fn shift_limbs_left(limbs: &[Limb], shift: u32, out: &mut Vec<Limb>) {
        out.clear();
        let len = limbs.len();
        if shift == 0 || len == 0 {
            out.extend_from_slice(limbs);
            return;
        }

        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS safely fits in u32"
        )]
        let limb_bits = LIMB_BITS as u32;
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "shift mathematically fits in usize bounds"
        )]
        let whole = (shift >> LIMB_BITS.trailing_zeros()) as usize;
        let bit_shift = shift & limb_bits.wrapping_sub(1);
        let new_len = whole.wrapping_add(len);

        if bit_shift == 0 {
            out.reserve(new_len);
            let out_ptr = out.as_mut_ptr();
            // SAFETY: reserve made `new_len` elements available and the writes
            // initialize exactly the zero prefix followed by all input limbs.
            unsafe {
                write_bytes(out_ptr, 0, whole);
                copy_nonoverlapping(limbs.as_ptr(), out_ptr.add(whole), len);
                out.set_len(new_len);
            }
        } else {
            out.reserve(new_len.wrapping_add(1));
            let out_ptr = out.as_mut_ptr();
            // SAFETY: reserve made at least `new_len + 1` elements available.
            unsafe {
                write_bytes(out_ptr, 0, whole);
            }

            // SAFETY: whole is within the reserved output allocation.
            let dest_ptr = unsafe { out_ptr.add(whole) };
            let src_ptr = limbs.as_ptr();
            let mut carry: Limb = 0;
            let carry_shift = limb_bits.wrapping_sub(bit_shift);

            let mut index: usize = 0;
            while index.wrapping_add(3) < len {
                // SAFETY: index + 3 < len, and the destination has room for
                // `whole + len + 1` initialized output limbs.
                unsafe {
                    let value_0 = src_ptr.add(index).read();
                    let value_1 = src_ptr.add(index.wrapping_add(1)).read();
                    let value_2 = src_ptr.add(index.wrapping_add(2)).read();
                    let value_3 = src_ptr.add(index.wrapping_add(3)).read();

                    dest_ptr
                        .add(index)
                        .write(value_0.wrapping_shl(bit_shift) | carry);
                    dest_ptr
                        .add(index.wrapping_add(1))
                        .write(value_1.wrapping_shl(bit_shift) | value_0.wrapping_shr(carry_shift));
                    dest_ptr
                        .add(index.wrapping_add(2))
                        .write(value_2.wrapping_shl(bit_shift) | value_1.wrapping_shr(carry_shift));
                    dest_ptr
                        .add(index.wrapping_add(3))
                        .write(value_3.wrapping_shl(bit_shift) | value_2.wrapping_shr(carry_shift));

                    carry = value_3.wrapping_shr(carry_shift);
                }
                index = index.wrapping_add(4);
            }

            while index < len {
                // SAFETY: index < len and the destination reserves one output
                // position per source limb after the zero prefix.
                unsafe {
                    let value = src_ptr.add(index).read();
                    dest_ptr
                        .add(index)
                        .write(value.wrapping_shl(bit_shift) | carry);
                    carry = value.wrapping_shr(carry_shift);
                }
                index = index.wrapping_add(1);
            }

            let final_len = if carry != 0 {
                // SAFETY: the extra reserved limb follows the `len` outputs.
                unsafe {
                    dest_ptr.add(len).write(carry);
                }
                new_len.wrapping_add(1)
            } else {
                new_len
            };

            // SAFETY: every element below final_len was initialized above.
            unsafe {
                out.set_len(final_len);
            }
        }

        debug_assert!(
            out.last().copied().unwrap_or(0) != 0,
            "division normalization expects normalized nonzero input when shift is nonzero"
        );
    }
}
