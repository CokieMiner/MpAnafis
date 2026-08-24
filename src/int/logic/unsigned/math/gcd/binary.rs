//! In-register binary GCD kernels for up to four limbs (256 bits).

use super::{Gcd, LIMB_BITS, Limb};

impl Gcd {
    /// In-register binary GCD for integers up to four limbs (256 bits on 64-bit targets).
    ///
    /// Keeps all state in CPU registers, eliminating vector allocations, length checks,
    /// and memory loads during binary reduction.
    #[must_use]
    pub const fn gcd_4(mut u: [Limb; 4], mut v: [Limb; 4]) -> [Limb; 4] {
        if Self::is_zero_4(u) {
            return v;
        }
        if Self::is_zero_4(v) {
            return u;
        }

        let u_tz = Self::trailing_zeros_4(u);
        let v_tz = Self::trailing_zeros_4(v);
        let common_shift = if u_tz < v_tz { u_tz } else { v_tz };

        Self::rshift_4(&mut u, u_tz);

        loop {
            let v_shift = Self::trailing_zeros_4(v);
            Self::rshift_4(&mut v, v_shift);

            if Self::is_greater_4(u, v) {
                let temp = u;
                u = v;
                v = temp;
            }

            Self::sub_4(&mut v, &u);

            if Self::is_zero_4(v) {
                break;
            }
        }

        Self::lshift_4(&mut u, common_shift);
        u
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "LIMB_BITS fits in u32"
    )]
    const fn trailing_zeros_4(u: [Limb; 4]) -> u32 {
        let limb_bits = LIMB_BITS as u32;
        if u[0] != 0 {
            u[0].trailing_zeros()
        } else if u[1] != 0 {
            u[1].trailing_zeros().wrapping_add(limb_bits)
        } else if u[2] != 0 {
            u[2].trailing_zeros()
                .wrapping_add(limb_bits.wrapping_mul(2))
        } else if u[3] != 0 {
            u[3].trailing_zeros()
                .wrapping_add(limb_bits.wrapping_mul(3))
        } else {
            limb_bits.wrapping_mul(4)
        }
    }

    const fn is_zero_4(u: [Limb; 4]) -> bool {
        u[0] == 0 && u[1] == 0 && u[2] == 0 && u[3] == 0
    }

    const fn is_greater_4(u: [Limb; 4], v: [Limb; 4]) -> bool {
        if u[3] != v[3] {
            u[3] > v[3]
        } else if u[2] != v[2] {
            u[2] > v[2]
        } else if u[1] != v[1] {
            u[1] > v[1]
        } else {
            u[0] > v[0]
        }
    }

    const fn sub_4(v: &mut [Limb; 4], u: &[Limb; 4]) {
        let (v0, b0) = v[0].overflowing_sub(u[0]);
        let (s1, b1a) = v[1].overflowing_sub(u[1]);
        let borrow0: Limb = if b0 { 1 } else { 0 };
        let (v1, b1b) = s1.overflowing_sub(borrow0);
        let b1 = b1a || b1b;

        let (s2, b2a) = v[2].overflowing_sub(u[2]);
        let borrow1: Limb = if b1 { 1 } else { 0 };
        let (v2, b2b) = s2.overflowing_sub(borrow1);
        let b2 = b2a || b2b;

        let (s3, _) = v[3].overflowing_sub(u[3]);
        let borrow2: Limb = if b2 { 1 } else { 0 };
        let (v3, _) = s3.overflowing_sub(borrow2);

        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        v[3] = v3;
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "LIMB_BITS fits in u32"
    )]
    const fn rshift_4(u: &mut [Limb; 4], shift: u32) {
        let limb_bits = LIMB_BITS as u32;
        if shift == 0 {
            return;
        }
        let word_shift = shift.wrapping_shr(LIMB_BITS.trailing_zeros()) as usize;
        let bit_shift = shift & limb_bits.wrapping_sub(1);

        match word_shift {
            0 => {}
            1 => {
                u[0] = u[1];
                u[1] = u[2];
                u[2] = u[3];
                u[3] = 0;
            }
            2 => {
                u[0] = u[2];
                u[1] = u[3];
                u[2] = 0;
                u[3] = 0;
            }
            3 => {
                u[0] = u[3];
                u[1] = 0;
                u[2] = 0;
                u[3] = 0;
            }
            _ => {
                *u = [0, 0, 0, 0];
                return;
            }
        }

        if bit_shift > 0 {
            let carry_shift = limb_bits.wrapping_sub(bit_shift);
            u[0] = u[0].wrapping_shr(bit_shift) | u[1].wrapping_shl(carry_shift);
            u[1] = u[1].wrapping_shr(bit_shift) | u[2].wrapping_shl(carry_shift);
            u[2] = u[2].wrapping_shr(bit_shift) | u[3].wrapping_shl(carry_shift);
            u[3] = u[3].wrapping_shr(bit_shift);
        }
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "LIMB_BITS fits in u32"
    )]
    const fn lshift_4(u: &mut [Limb; 4], shift: u32) {
        let limb_bits = LIMB_BITS as u32;
        if shift == 0 {
            return;
        }
        let word_shift = shift.wrapping_shr(LIMB_BITS.trailing_zeros()) as usize;
        let bit_shift = shift & limb_bits.wrapping_sub(1);

        match word_shift {
            0 => {}
            1 => {
                u[3] = u[2];
                u[2] = u[1];
                u[1] = u[0];
                u[0] = 0;
            }
            2 => {
                u[3] = u[1];
                u[2] = u[0];
                u[1] = 0;
                u[0] = 0;
            }
            3 => {
                u[3] = u[0];
                u[2] = 0;
                u[1] = 0;
                u[0] = 0;
            }
            _ => {
                *u = [0, 0, 0, 0];
                return;
            }
        }

        if bit_shift > 0 {
            let carry_shift = limb_bits.wrapping_sub(bit_shift);
            u[3] = u[3].wrapping_shl(bit_shift) | u[2].wrapping_shr(carry_shift);
            u[2] = u[2].wrapping_shl(bit_shift) | u[1].wrapping_shr(carry_shift);
            u[1] = u[1].wrapping_shl(bit_shift) | u[0].wrapping_shr(carry_shift);
            u[0] = u[0].wrapping_shl(bit_shift);
        }
    }
}
