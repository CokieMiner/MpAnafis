//! Radix string parsing for the unsigned integer engine.

#![allow(
    unsafe_code,
    reason = "Conversion algorithms use unchecked slice access on pre-sized buffers and String::from_utf8_unchecked after constructing ASCII-only output bytes."
)]

use alloc::vec::Vec;

use crate::error::{ParseMpUintError, ParseMpUintErrorKind};

use super::{
    ArchKernels, DoubleLimb, INLINE_LIMBS, InternalMpUint, KARATSUBA_THRESHOLD, LIMB_BITS, Limb,
};

// --- Const lookup tables for radix conversion ---

/// Limb-sized parameters shared by radix parsing and formatting.
#[derive(Clone, Copy)]
pub struct RadixParameters {
    /// Largest `k` such that `radix^k` fits in a [`Limb`].
    pub max_digits: usize,
    /// `radix^max_digits`.
    pub max_power: Limb,
}

impl RadixParameters {
    /// Returns the limb-sized parameters for a radix in `2..=36`.
    ///
    /// The table is computed at compile time, avoiding a runtime
    /// `checked_mul` loop on every conversion.
    #[allow(clippy::too_many_lines, reason = "Lookup tables take many lines")]
    #[inline]
    pub const fn for_limb(radix: u32) -> Self {
        // Table indexed by radix (2..=36). Each entry is (max_digits, max_power).
        // Computed offline for both 32-bit and 64-bit Limb sizes.
        #[cfg(target_pointer_width = "64")]
        const TABLE: [(usize, Limb); 37] = [
            (0, 0),                      // radix 0 (unused)
            (0, 0),                      // radix 1 (unused)
            (63, 0x8000_0000_0000_0000), // radix 2
            (40, 0xa8b8_b452_291f_e821), // radix 3
            (31, 0x4000_0000_0000_0000), // radix 4
            (27, 0x6765_c793_fa10_079d), // radix 5
            (24, 0x41c2_1cb8_e100_0000), // radix 6
            (22, 0x3642_7987_5022_6111), // radix 7
            (21, 0x8000_0000_0000_0000), // radix 8
            (20, 0xa8b8_b452_291f_e821), // radix 9
            (19, 0x8ac7_2304_89e8_0000), // radix 10
            (18, 0x4d28_cb56_c33f_a539), // radix 11
            (17, 0x1eca_170c_0000_0000), // radix 12
            (17, 0x780c_7372_621b_d74d), // radix 13
            (16, 0x1e39_a505_7d81_0000), // radix 14
            (16, 0x5b27_ac99_3df9_7701), // radix 15
            (15, 0x1000_0000_0000_0000), // radix 16
            (15, 0x27b9_5e99_7e21_d9f1), // radix 17
            (15, 0x5da0_e1e5_3c5c_8000), // radix 18
            (15, 0xd2ae_3299_c1c4_aedb), // radix 19
            (14, 0x16bc_c41e_9000_0000), // radix 20
            (14, 0x2d04_b7fd_d9c0_ef49), // radix 21
            (14, 0x5658_597b_caa2_4000), // radix 22
            (14, 0xa0e2_0737_3760_9371), // radix 23
            (13, 0x0c29_e980_0000_0000), // radix 24
            (13, 0x14ad_f4b7_3203_34b9), // radix 25
            (13, 0x226e_d364_78bf_a000), // radix 26
            (13, 0x383d_9170_b85f_f80b), // radix 27
            (13, 0x5a3c_23e3_9c00_0000), // radix 28
            (13, 0x8e65_1373_8812_2bcd), // radix 29
            (13, 0xdd41_bb36_d259_e000), // radix 30
            (12, 0x0aee_5720_ee83_0681), // radix 31
            (12, 0x1000_0000_0000_0000), // radix 32
            (12, 0x1725_88ad_4f5f_0981), // radix 33
            (12, 0x211e_44f7_d02c_1000), // radix 34
            (12, 0x2ee5_6725_f06e_5c71), // radix 35
            (12, 0x41c2_1cb8_e100_0000), // radix 36
        ];
        #[cfg(target_pointer_width = "32")]
        const TABLE: [(usize, Limb); 37] = [
            (0, 0),            // radix 0 (unused)
            (0, 0),            // radix 1 (unused)
            (31, 0x8000_0000), // radix 2
            (20, 0xcfd4_1b91), // radix 3
            (15, 0x4000_0000), // radix 4
            (13, 0x48c2_7395), // radix 5
            (12, 0x81bf_1000), // radix 6
            (11, 0x75db_9c97), // radix 7
            (10, 0x4000_0000), // radix 8
            (10, 0xcfd4_1b91), // radix 9
            (9, 0x3b9a_ca00),  // radix 10
            (9, 0x8c8b_6d2b),  // radix 11
            (8, 0x19a1_0000),  // radix 12
            (8, 0x309f_1021),  // radix 13
            (8, 0x57f6_c100),  // radix 14
            (8, 0x98c2_9b81),  // radix 15
            (7, 0x1000_0000),  // radix 16
            (7, 0x1875_4571),  // radix 17
            (7, 0x247d_bc80),  // radix 18
            (7, 0x3547_667b),  // radix 19
            (7, 0x4c4b_4000),  // radix 20
            (7, 0x6b5a_6e1d),  // radix 21
            (7, 0x94ac_e180),  // radix 22
            (7, 0xcaf1_8367),  // radix 23
            (6, 0x0b64_0000),  // radix 24
            (6, 0x0e8d_4a51),  // radix 25
            (6, 0x1269_ae40),  // radix 26
            (6, 0x1717_9149),  // radix 27
            (6, 0x1cb9_1000),  // radix 28
            (6, 0x2374_4899),  // radix 29
            (6, 0x2b73_a840),  // radix 30
            (6, 0x34e6_3b41),  // radix 31
            (6, 0x4000_0000),  // radix 32
            (6, 0x4cfa_3cc1),  // radix 33
            (6, 0x5c13_d840),  // radix 34
            (6, 0x6d91_b519),  // radix 35
            (6, 0x81bf_1000),  // radix 36
        ];
        #[cfg(target_pointer_width = "16")]
        const TABLE: [(usize, Limb); 37] = [
            (0, 0x0000),  // radix 0 (unused)
            (0, 0x0000),  // radix 1 (unused)
            (15, 0x8000), // radix 2
            (10, 0xe6a9), // radix 3
            (7, 0x4000),  // radix 4
            (6, 0x3d09),  // radix 5
            (6, 0xb640),  // radix 6
            (5, 0x41a7),  // radix 7
            (5, 0x8000),  // radix 8
            (5, 0xe6a9),  // radix 9
            (4, 0x2710),  // radix 10
            (4, 0x3931),  // radix 11
            (4, 0x5100),  // radix 12
            (4, 0x6f91),  // radix 13
            (4, 0x9610),  // radix 14
            (4, 0xc5c1),  // radix 15
            (3, 0x1000),  // radix 16
            (3, 0x1331),  // radix 17
            (3, 0x16c8),  // radix 18
            (3, 0x1acb),  // radix 19
            (3, 0x1f40),  // radix 20
            (3, 0x242d),  // radix 21
            (3, 0x2998),  // radix 22
            (3, 0x2f87),  // radix 23
            (3, 0x3600),  // radix 24
            (3, 0x3d09),  // radix 25
            (3, 0x44a8),  // radix 26
            (3, 0x4ce3),  // radix 27
            (3, 0x55c0),  // radix 28
            (3, 0x5f45),  // radix 29
            (3, 0x6978),  // radix 30
            (3, 0x745f),  // radix 31
            (3, 0x8000),  // radix 32
            (3, 0x8c61),  // radix 33
            (3, 0x9988),  // radix 34
            (3, 0xa77b),  // radix 35
            (3, 0xb640),  // radix 36
        ];
        #[allow(
            clippy::as_conversions,
            clippy::indexing_slicing,
            reason = "radix is bounded to 2..=36 by caller"
        )]
        let (max_digits, max_power) = TABLE[radix as usize];
        Self {
            max_digits,
            max_power,
        }
    }
}

impl InternalMpUint {
    /// Parses a string in the given radix (2..=36).
    ///
    /// Supports both uppercase and lowercase digits.
    ///
    /// # Errors
    ///
    /// Returns `ParseMpUintError` if the string is empty, the radix is out of range,
    /// or a character is not a valid digit for the radix.
    #[allow(
        clippy::too_many_lines,
        reason = "parsing loop is tightly coupled and efficient"
    )]
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseMpUintError> {
        if !(2..=36).contains(&radix) {
            return Err(ParseMpUintError {
                kind: ParseMpUintErrorKind::InvalidRadix,
            });
        }

        if s.is_empty() {
            return Err(ParseMpUintError {
                kind: ParseMpUintErrorKind::Empty,
            });
        }

        if s.starts_with('-') {
            return Err(ParseMpUintError {
                kind: ParseMpUintErrorKind::Negative,
            });
        }

        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "radix <= 36 fits in Limb"
        )]
        let radix_limb = radix as Limb;
        let RadixParameters {
            max_digits,
            max_power,
        } = RadixParameters::for_limb(radix);

        // Fast path for small strings (fewer than KARATSUBA_THRESHOLD chunks)
        if s.len() <= max_digits.wrapping_mul(KARATSUBA_THRESHOLD) {
            let mut result = Self::zero();
            let mut current_chunk_val: Limb = 0;
            let mut current_chunk_len = 0_usize;
            let mut current_chunk_power: Limb = 1;

            for &byte in s.as_bytes() {
                let digit = digit_from_ascii_byte(byte, radix).ok_or(ParseMpUintError {
                    kind: ParseMpUintErrorKind::InvalidDigit,
                })?;
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "digit < radix <= 36 fits in Limb"
                )]
                let digit_limb = digit as Limb;
                current_chunk_val = current_chunk_val
                    .wrapping_mul(radix_limb)
                    .wrapping_add(digit_limb);
                current_chunk_power = current_chunk_power.wrapping_mul(radix_limb);
                current_chunk_len = current_chunk_len.saturating_add(1);

                if current_chunk_len == max_digits {
                    result = mul_small_add(&result, max_power, current_chunk_val);
                    current_chunk_val = 0;
                    current_chunk_len = 0;
                    current_chunk_power = 1;
                }
            }

            if current_chunk_len > 0 {
                result = mul_small_add(&result, current_chunk_power, current_chunk_val);
            }
            return Ok(result);
        }

        // Divide and Conquer path for huge strings.
        // Validation is done inline during chunk processing to avoid a
        // separate full-string pass. Chunks are stored as bare Limb values,
        // avoiding per-chunk InternalMpUint allocation overhead.
        let bytes = s.as_bytes();
        let mut chunk_vals: Vec<Limb> = Vec::new();
        let mut i = bytes.len();
        while i > 0 {
            let start = i.saturating_sub(max_digits);
            // SAFETY: start < i <= bytes.len(), so bytes[start..i] is in bounds.
            let chunk_bytes = unsafe { bytes.get_unchecked(start..i) };
            let mut chunk_val: Limb = 0;
            for &byte in chunk_bytes {
                let digit = digit_from_ascii_byte(byte, radix).ok_or(ParseMpUintError {
                    kind: ParseMpUintErrorKind::InvalidDigit,
                })?;
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "digit < radix <= 36 fits in Limb"
                )]
                let digit_limb = digit as Limb;
                chunk_val = chunk_val.wrapping_mul(radix_limb).wrapping_add(digit_limb);
            }
            chunk_vals.push(chunk_val);
            i = start;
        }

        // Pad chunks to a power of 2 (pad with 0 limbs)
        let mut len_pow2 = 1;
        while len_pow2 < chunk_vals.len() {
            len_pow2 = len_pow2.wrapping_mul(2);
        }
        chunk_vals.resize(len_pow2, 0);

        // Convert bare Limb chunks to InternalMpUint for combine_chunks.
        // Allocation is deferred to here (one Vec per chunk) rather than
        // performed during the parsing loop.
        let chunks: Vec<Self> = chunk_vals.into_iter().map(Self::from_limb).collect();

        // Precompute powers: powers[k] = max_power ^ (2^k)
        let mut powers = Vec::new();
        powers.push(Self::from_limb(max_power));
        for _ in 1..len_pow2.trailing_zeros() {
            // SAFETY: the initial push above means `powers` is never empty on
            // any iteration of this loop.
            let last = unsafe { powers.last().unwrap_unchecked() };
            powers.push(last.mul(last));
        }

        Ok(combine_chunks(&chunks, &powers))
    }
}

fn combine_chunks(chunks: &[InternalMpUint], powers: &[InternalMpUint]) -> InternalMpUint {
    if chunks.is_empty() {
        return InternalMpUint::zero();
    }
    if chunks.len() == 1 {
        // SAFETY: the early return for an empty chunks slice proves length >= 1.
        return unsafe { chunks.first().unwrap_unchecked() }.clone();
    }
    // The single-element case returned above, so `chunks.len() >= 2` and
    // `0 < mid < chunks.len()`: both halves split at `mid` are in bounds.
    let mid = chunks.len() >> 1;
    // SAFETY: `mid = chunks.len() >> 1` with `chunks.len() >= 2` bounds both
    // `..mid` and `mid..` ranges.
    let (lower, upper) = unsafe {
        (
            combine_chunks(chunks.get_unchecked(..mid), powers),
            combine_chunks(chunks.get_unchecked(mid..), powers),
        )
    };

    // `chunks.len()` is a power of two and `mid` is its exact half, so `mid` is
    // a power of two and `k` is its split level. `powers` holds one entry per
    // split level up to `len_pow2`, so `k < powers.len()`.
    // SAFETY: `mid.trailing_zeros()` is at most `usize::BITS - 1` and therefore
    // always fits in `usize` on 16/32/64-bit targets.
    let k = unsafe { usize::try_from(mid.trailing_zeros()).unwrap_unchecked() };
    // SAFETY: `k` is the precomputed power index for `mid`'s split level and is
    // below `powers.len()`.
    let power = unsafe { powers.get_unchecked(k) };

    let mut upper_shifted = upper.mul(power);
    upper_shifted.add_assign(&lower);
    upper_shifted
}

/// Multiplies `value` by `mul` and adds `add`, producing a new number.
///
/// Uses the architecture-optimised `add_mul_limbs_unchecked` kernel to compute
/// `result = value * mul`, then folds in the `add` term with carry propagation.
///
/// Used internally by string parsing.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Limb→DoubleLimb is a widening cast; DoubleLimb→Limb truncation \
              is correct — the low LIMB_BITS bits form the result limb"
)]
fn mul_small_add(value: &InternalMpUint, mul: Limb, add: Limb) -> InternalMpUint {
    let limbs = value.limbs();
    let len = limbs.len();
    if len < INLINE_LIMBS {
        let mut result_limbs = [0_usize; INLINE_LIMBS];
        // SAFETY: result_limbs has INLINE_LIMBS elements, limbs has len elements, len < INLINE_LIMBS.
        let kernel_carry = unsafe {
            ArchKernels::add_mul_limbs_unchecked(
                result_limbs.as_mut_ptr(),
                limbs.as_ptr(),
                len,
                mul,
            )
        };
        // SAFETY: len < INLINE_LIMBS, so index len is in bounds.
        unsafe {
            *result_limbs.get_unchecked_mut(len) = kernel_carry;
        }
        let mut carry: DoubleLimb = add as DoubleLimb;
        for i in 0..=len {
            // SAFETY: i <= len < INLINE_LIMBS.
            let d = unsafe { *result_limbs.get_unchecked(i) };
            carry = carry.wrapping_add(d as DoubleLimb);
            // SAFETY: i <= len < INLINE_LIMBS.
            unsafe {
                *result_limbs.get_unchecked_mut(i) = carry as Limb;
            }
            carry >>= LIMB_BITS;
            if carry == 0 {
                break;
            }
        }
        return InternalMpUint::from_limbs_4(
            result_limbs[0],
            result_limbs[1],
            result_limbs[2],
            result_limbs[3],
        );
    }

    // Allocate result with room for a possible carry limb
    let mut result_limbs: Vec<Limb> = alloc::vec![0; len.wrapping_add(1)];

    // SAFETY: result_limbs has len+1 elements, limbs has len elements.
    // The kernel writes len limbs and returns the carry.
    let kernel_carry = unsafe {
        ArchKernels::add_mul_limbs_unchecked(result_limbs.as_mut_ptr(), limbs.as_ptr(), len, mul)
    };

    // Store kernel carry at position len
    // SAFETY: result_limbs has len+1 elements, index len is valid
    unsafe {
        *result_limbs.as_mut_ptr().add(len) = kernel_carry;
    }

    // Add `add` to the result with carry propagation
    let mut carry: DoubleLimb = add as DoubleLimb;
    for i in 0..len.wrapping_add(1) {
        // SAFETY: result_limbs has len+1 elements
        let d = unsafe { *result_limbs.get_unchecked(i) };
        carry = carry.wrapping_add(d as DoubleLimb);
        // SAFETY: result_limbs has len+1 elements
        unsafe {
            *result_limbs.get_unchecked_mut(i) = carry as Limb;
        }
        carry >>= LIMB_BITS;
        if carry == 0 {
            break;
        }
    }

    InternalMpUint::from_limbs(result_limbs)
}

/// Converts an ASCII digit byte into its numeric value for `radix`.
///
/// Huge-input parsing chunks the original UTF-8 string by byte count. Since
/// every valid radix digit is ASCII, validating bytes directly avoids creating
/// invalid UTF-8 subslices when an invalid multibyte character straddles a
/// chunk boundary.
fn digit_from_ascii_byte(byte: u8, radix: u32) -> Option<u32> {
    let digit = match byte {
        b'0'..=b'9' => u32::from(byte.wrapping_sub(b'0')),
        b'a'..=b'z' => u32::from(byte.wrapping_sub(b'a')).wrapping_add(10),
        b'A'..=b'Z' => u32::from(byte.wrapping_sub(b'A')).wrapping_add(10),
        _ => return None,
    };
    (digit < radix).then_some(digit)
}
