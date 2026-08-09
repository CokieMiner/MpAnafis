//! One-shot x86-64 CPU dispatch for complete schoolbook multiplication.
//!
//! Three backends: ADX+BMI2 (modern, with fixed-width kernels), BMI2 (Haswell,
//! paired-row with `mulx`), and a vanilla fallback for everything else.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, add_mul_2_limbs_bmi2_backend, add_mul_2_limbs_vanilla_backend,
    add_mul_limbs_adx_backend, add_mul_limbs_bmi2_backend, add_mul_limbs_vanilla_backend,
    mul_2_limbs_bmi2_backend, mul_2_limbs_vanilla_backend, mul_2x2_portable_unchecked,
    mul_3x3_portable_unchecked, selected_x86_backend,
    x86_64_adx::{
        add_mul_4_limbs_unchecked as add_mul_4_adx, add_mul_5_limbs_unchecked as add_mul_5_adx,
        add_mul_6_limbs_unchecked as add_mul_6_adx, add_mul_7_limbs_unchecked as add_mul_7_adx,
        add_mul_8_limbs_unchecked as add_mul_8_adx, add_mul_9_limbs_unchecked as add_mul_9_adx,
        add_mul_10_limbs_unchecked as add_mul_10_adx, add_mul_11_limbs_unchecked as add_mul_11_adx,
        add_mul_12_limbs_unchecked as add_mul_12_adx, add_mul_13_limbs_unchecked as add_mul_13_adx,
        mul_2x4_limbs_unchecked as mul_2x4_adx, mul_2x5_limbs_unchecked as mul_2x5_adx,
        mul_2x6_limbs_unchecked as mul_2x6_adx, mul_2x7_limbs_unchecked as mul_2x7_adx,
        mul_2x8_limbs_unchecked as mul_2x8_adx, mul_2x9_limbs_unchecked as mul_2x9_adx,
        mul_2x10_limbs_unchecked as mul_2x10_adx, mul_2x11_limbs_unchecked as mul_2x11_adx,
        mul_2x12_limbs_unchecked as mul_2x12_adx, mul_2x13_limbs_unchecked as mul_2x13_adx,
    },
    x86_64_adx_tail::{
        add_mul_14_limbs_unchecked as add_mul_14_adx, add_mul_15_limbs_unchecked as add_mul_15_adx,
        add_mul_16_limbs_unchecked as add_mul_16_adx, add_mul_17_limbs_unchecked as add_mul_17_adx,
    },
};

type BasecaseFn = unsafe fn(*mut Limb, *const Limb, usize, *const Limb, usize);

static KERNEL: OnceLock<BasecaseFn> = OnceLock::new();

// ---------------------------------------------------------------------------
// Macros for generating complete fixed-width basecase variants
// ---------------------------------------------------------------------------

/// Complete fixed-width basecase using a custom two-row initializer and a
/// fixed-width add-mul row kernel. Used for inner widths 4–13 where both
/// the first-row and subsequent-row ADX kernels are specialized.
macro_rules! define_fixed_width_basecase_init {
    ($name:ident, $len:literal, $init:ident, $add_mul:ident) => {
        unsafe fn $name(dst: *mut Limb, a: *const Limb, len_a: usize, b: *const Limb) {
            // SAFETY: the caller guarantees len_a >= 2, the complete product
            // span, and the fixed-width source selected by the outer match.
            unsafe {
                $init(dst, b, *a, *a.add(1));
            }
            let mut index = 2_usize;
            while index < len_a {
                // SAFETY: every shifted fixed row and carry limb remains in
                // the inherited len_a + fixed-width destination span.
                let carry = unsafe { $add_mul(dst.add(index), b, *a.add(index)) };
                // SAFETY: index < len_a proves index + fixed width is inside
                // the complete product destination.
                unsafe {
                    *dst.add(index.wrapping_add($len)) = carry;
                }
                index = index.wrapping_add(1);
            }
        }
    };
}

/// Complete fixed-width basecase using the generic BMI2 two-row initializer
/// and a fixed-width add-mul row kernel. Used for inner widths 14–17 where
/// only the subsequent-row kernel is specialized.
macro_rules! define_fixed_width_basecase {
    ($name:ident, $len:literal, $add_mul:ident) => {
        unsafe fn $name(dst: *mut Limb, a: *const Limb, len_a: usize, b: *const Limb) {
            // SAFETY: the caller guarantees len_a >= 2, the complete product
            // span, and the fixed-width source selected by the outer match.
            unsafe {
                mul_2_limbs_bmi2_backend(dst, b, $len, *a, *a.add(1));
            }
            let mut index = 2_usize;
            while index < len_a {
                // SAFETY: every shifted fixed row and carry limb remains in
                // the inherited len_a + fixed-width destination span.
                let carry = unsafe { $add_mul(dst.add(index), b, *a.add(index)) };
                // SAFETY: index < len_a proves index + fixed width is inside
                // the complete product destination.
                unsafe {
                    *dst.add(index.wrapping_add($len)) = carry;
                }
                index = index.wrapping_add(1);
            }
        }
    };
}

/// Paired-row basecase using the given two-row, paired, and single-row
/// multiply-add backends. Used for the BMI2 and vanilla fallback paths.
macro_rules! define_paired_row_basecase {
    ($name:ident, $mul_two:ident, $add_mul_two:ident, $add_mul_one:ident) => {
        unsafe fn $name(
            dst: *mut Limb,
            a: *const Limb,
            len_a: usize,
            b: *const Limb,
            len_b: usize,
        ) {
            // SAFETY: len_a >= 2 and the caller guarantees the complete spans.
            unsafe { $mul_two(dst, b, len_b, *a, *a.add(1)) };
            let mut index = 2_usize;
            while index.wrapping_add(1) < len_a {
                let carry_index0 = index.wrapping_add(len_b);
                let carry_index1 = carry_index0.wrapping_add(1);
                // SAFETY: both rows and carry positions lie inside the proven spans.
                unsafe {
                    *dst.add(carry_index0) = 0;
                    let (carry0, carry1) = $add_mul_two(
                        dst.add(index),
                        b,
                        len_b,
                        *a.add(index),
                        *a.add(index.wrapping_add(1)),
                    );
                    let existing = *dst.add(carry_index0);
                    let (sum, overflow) = existing.overflowing_add(carry0);
                    *dst.add(carry_index0) = sum;
                    let (top, top_overflow) = carry1.overflowing_add(Limb::from(overflow));
                    debug_assert!(!top_overflow, "dual-row carry exceeded the result width");
                    *dst.add(carry_index1) = top;
                }
                index = index.wrapping_add(2);
            }
            if index < len_a {
                // SAFETY: the final row and carry position lie inside dst.
                let carry = unsafe { $add_mul_one(dst.add(index), b, len_b, *a.add(index)) };
                // SAFETY: index + len_b < len_a + len_b.
                unsafe {
                    *dst.add(index.wrapping_add(len_b)) = carry;
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ADX+BMI2 backend: one match at entry, no per-row dispatch
// ---------------------------------------------------------------------------

// Widths 4–13: fixed ADX two-row init + fixed ADX add-mul rows.
define_fixed_width_basecase_init!(basecase_adx_bmi2_4, 4, mul_2x4_adx, add_mul_4_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_5, 5, mul_2x5_adx, add_mul_5_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_6, 6, mul_2x6_adx, add_mul_6_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_7, 7, mul_2x7_adx, add_mul_7_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_8, 8, mul_2x8_adx, add_mul_8_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_9, 9, mul_2x9_adx, add_mul_9_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_10, 10, mul_2x10_adx, add_mul_10_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_11, 11, mul_2x11_adx, add_mul_11_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_12, 12, mul_2x12_adx, add_mul_12_adx);
define_fixed_width_basecase_init!(basecase_adx_bmi2_13, 13, mul_2x13_adx, add_mul_13_adx);

// Widths 14–17: generic BMI2 two-row init + fixed ADX add-mul rows.
define_fixed_width_basecase!(basecase_adx_bmi2_14, 14, add_mul_14_adx);
define_fixed_width_basecase!(basecase_adx_bmi2_15, 15, add_mul_15_adx);
define_fixed_width_basecase!(basecase_adx_bmi2_16, 16, add_mul_16_adx);
define_fixed_width_basecase!(basecase_adx_bmi2_17, 17, add_mul_17_adx);

/// Compute the complete product `dst[0..len_a + len_b] = a[0..len_a] ×
/// b[0..len_b]` using the ADX+BMI2 backend.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len_a + len_b` elements.
/// - `a` must be valid for reads of `len_a` elements.
/// - `b` must be valid for reads of `len_b` elements.
/// - `dst`, `a`, and `b` spans must be pairwise disjoint: the kernel writes
///   rows into `dst` while reading `a` and `b`.
unsafe fn mul_basecase_adx_bmi2(
    dst: *mut Limb,
    a: *const Limb,
    len_a: usize,
    b: *const Limb,
    len_b: usize,
) {
    // SAFETY: every fixed-width branch encodes the exact inner width; the
    // generic branch receives the caller-proven len_b. All branches inherit
    // the complete-product and disjoint-span invariants.
    unsafe {
        match len_b {
            4 => basecase_adx_bmi2_4(dst, a, len_a, b),
            5 => basecase_adx_bmi2_5(dst, a, len_a, b),
            6 => basecase_adx_bmi2_6(dst, a, len_a, b),
            7 => basecase_adx_bmi2_7(dst, a, len_a, b),
            8 => basecase_adx_bmi2_8(dst, a, len_a, b),
            9 => basecase_adx_bmi2_9(dst, a, len_a, b),
            10 => basecase_adx_bmi2_10(dst, a, len_a, b),
            11 => basecase_adx_bmi2_11(dst, a, len_a, b),
            12 => basecase_adx_bmi2_12(dst, a, len_a, b),
            13 => basecase_adx_bmi2_13(dst, a, len_a, b),
            14 => basecase_adx_bmi2_14(dst, a, len_a, b),
            15 => basecase_adx_bmi2_15(dst, a, len_a, b),
            16 => basecase_adx_bmi2_16(dst, a, len_a, b),
            17 => basecase_adx_bmi2_17(dst, a, len_a, b),
            _ => {
                mul_2_limbs_bmi2_backend(dst, b, len_b, *a, *a.add(1));
                let mut index = 2_usize;
                while index < len_a {
                    let carry = add_mul_limbs_adx_backend(dst.add(index), b, len_b, *a.add(index));
                    *dst.add(index.wrapping_add(len_b)) = carry;
                    index = index.wrapping_add(1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BMI2 backend (Haswell): paired-row with mulx, no ADX
// ---------------------------------------------------------------------------

define_paired_row_basecase!(
    mul_basecase_bmi2,
    mul_2_limbs_bmi2_backend,
    add_mul_2_limbs_bmi2_backend,
    add_mul_limbs_bmi2_backend
);

// ---------------------------------------------------------------------------
// Vanilla fallback: paired-row without special instructions
// ---------------------------------------------------------------------------

define_paired_row_basecase!(
    mul_basecase_fallback,
    mul_2_limbs_vanilla_backend,
    add_mul_2_limbs_vanilla_backend,
    add_mul_limbs_vanilla_backend
);

// ---------------------------------------------------------------------------
// Kernel selection and public API
// ---------------------------------------------------------------------------

fn select_kernel() -> BasecaseFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => mul_basecase_adx_bmi2,
        X86Backend::Bmi2 => mul_basecase_bmi2,
        X86Backend::Adx | X86Backend::Baseline => mul_basecase_fallback,
    }
}

/// Dispatch and compute a complete schoolbook product.
///
/// # Safety
///
/// `a` and `b` must cover `len_a >= 2` and `len_b > 0` limbs, `dst` must
/// cover `len_a + len_b` limbs, and neither input may overlap `dst`.
#[inline]
pub unsafe fn mul_basecase_unchecked(
    dst: *mut Limb,
    a: *const Limb,
    len_a: usize,
    b: *const Limb,
    len_b: usize,
) {
    debug_assert!(len_a >= 2, "basecase outer operand needs two limbs");
    debug_assert!(len_b > 0, "basecase inner operand must be nonempty");
    if len_a == 2 && len_b == 2 {
        // The fixed portable kernel needs no CPU dispatch. Keeping this check
        // at the architecture boundary also helps direct basecase callers.
        // SAFETY: this branch proves both inputs and the four-limb destination.
        unsafe {
            mul_2x2_portable_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 3 && len_b == 3 {
        // The portable fixed kernel is faster than the feature-specific path
        // and needs no CPU dispatch. Keeping the check at this boundary also
        // makes direct basecase callers pay no OnceLock/function-call cost.
        // SAFETY: this branch proves both inputs have three limbs; the caller
        // provides the disjoint six-limb destination.
        unsafe {
            mul_3x3_portable_unchecked(dst, a, b);
        }
        return;
    }
    let kernel = *KERNEL.get_or_init(select_kernel);
    // SAFETY: the caller establishes all spans and the selector proves any
    // backend CPU feature requirements.
    unsafe {
        kernel(dst, a, len_a, b, len_b);
    }
}
