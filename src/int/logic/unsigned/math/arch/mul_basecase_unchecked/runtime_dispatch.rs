//! One-shot x86-64 CPU dispatch for complete schoolbook multiplication.
//!
//! ADX+BMI2 uses fixed-width rows; BMI2 uses paired `mulx` rows; the baseline
//! uses paired `mulq` rows. CPU selection occurs once per complete product.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, add_mul_2_limbs_bmi2_backend, add_mul_2_limbs_vanilla_backend,
    add_mul_limbs_adx_backend, add_mul_limbs_bmi2_backend, add_mul_limbs_vanilla_backend,
    mul_2_limbs_bmi2_backend, mul_2_limbs_vanilla_backend, mul_2x2_portable_unchecked,
    mul_3x3_portable_unchecked, selected_x86_backend,
    mul_4x4_portable_unchecked, mul_8x8_portable_unchecked,
    add_mul_4_adx, add_mul_5_adx, add_mul_6_adx, add_mul_7_adx,
    add_mul_8_adx, add_mul_9_adx, add_mul_10_adx, add_mul_11_adx,
    add_mul_12_adx, add_mul_13_adx, add_mul_14_adx, add_mul_15_adx,
    add_mul_16_adx, add_mul_17_adx, mul_2x4_adx, mul_2x5_adx, mul_2x6_adx,
    mul_2x7_adx, mul_2x8_adx, mul_2x9_adx, mul_2x10_adx, mul_2x11_adx,
    mul_2x12_adx, mul_2x13_adx, mul_4x4_adx, mul_8x8_adx,
};

type BasecaseFn = unsafe fn(*mut Limb, *const Limb, usize, *const Limb, usize);
type FixedBasecaseFn = unsafe fn(*mut Limb, *const Limb, *const Limb);

static KERNEL: OnceLock<BasecaseFn> = OnceLock::new();
static MUL_4_KERNEL: OnceLock<FixedBasecaseFn> = OnceLock::new();
static MUL_8_KERNEL: OnceLock<FixedBasecaseFn> = OnceLock::new();

/// Dispatches once and writes the complete schoolbook product.
///
/// # Safety
///
/// Aligned initialized inputs cover `len_a >= 2` and `len_b > 0` limbs.
/// The aligned exclusive destination covers `len_a + len_b` writable limbs,
/// which may be uninitialized, and does not overlap either input. The length
/// sum fits `usize`; all byte spans fit `isize::MAX`. Read-only inputs may alias.
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
        // SAFETY: this branch fixes both aligned inputs at two limbs and the
        // disjoint write-only destination at four limbs; no CPU features are needed.
        unsafe {
            mul_2x2_portable_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 3 && len_b == 3 {
        // SAFETY: this branch fixes both aligned inputs at three limbs and the
        // disjoint write-only destination at six limbs; no CPU features are needed.
        unsafe {
            mul_3x3_portable_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 4 && len_b == 4 {
        // SAFETY: both initialized inputs have four limbs and the disjoint
        // write-only destination has eight. The cached selector proves ADX/BMI2.
        unsafe {
            mul_4x4_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 8 && len_b == 8 {
        // SAFETY: both initialized inputs have eight limbs and the disjoint
        // write-only destination has sixteen. The cached selector proves ADX/BMI2.
        unsafe {
            mul_8x8_unchecked(dst, a, b);
        }
        return;
    }
    let kernel = *KERNEL.get_or_init(select_kernel);
    // SAFETY: the caller supplies aligned initialized inputs, a disjoint
    // exclusive destination, and representable spans; selection proves CPU features.
    unsafe { kernel(dst, a, len_a, b, len_b) }
}

/// Dispatches to the selected four-by-four product kernel.
///
/// # Safety
/// Aligned inputs cover four initialized limbs each; the disjoint exclusive
/// aligned destination covers eight writable limbs, which may be uninitialized.
#[inline]
pub unsafe fn mul_4x4_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    let kernel = *MUL_4_KERNEL.get_or_init(select_mul_4_kernel);
    // SAFETY: the caller supplies both aligned inputs and the disjoint eight-limb
    // destination. The cached selector proves any ADX/BMI2 requirement.
    unsafe { kernel(dst, a, b) }
}

/// Dispatches to the selected eight-by-eight product kernel.
///
/// # Safety
/// Aligned inputs cover eight initialized limbs each; the disjoint exclusive
/// aligned destination covers sixteen writable limbs, which may be uninitialized.
#[inline]
pub unsafe fn mul_8x8_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    let kernel = *MUL_8_KERNEL.get_or_init(select_mul_8_kernel);
    // SAFETY: the caller supplies both aligned inputs and the disjoint sixteen-limb
    // destination. The cached selector proves any ADX/BMI2 requirement.
    unsafe { kernel(dst, a, b) }
}

fn select_kernel() -> BasecaseFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => mul_basecase_adx_bmi2,
        X86Backend::Bmi2 => mul_basecase_bmi2,
        X86Backend::Adx | X86Backend::Baseline => mul_basecase_fallback,
    }
}

fn select_mul_4_kernel() -> FixedBasecaseFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => mul_4x4_adx,
        X86Backend::Adx | X86Backend::Bmi2 | X86Backend::Baseline => mul_4x4_portable_unchecked,
    }
}

fn select_mul_8_kernel() -> FixedBasecaseFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => mul_8x8_adx,
        X86Backend::Adx | X86Backend::Bmi2 | X86Backend::Baseline => mul_8x8_portable_unchecked,
    }
}

// ---------------------------------------------------------------------------
// Macros for generating complete fixed-width basecase variants
// ---------------------------------------------------------------------------

/// Complete fixed-width basecase using a custom two-row initializer and a
/// fixed-width add-mul row kernel. Used for inner widths 4-13 where both
/// the first-row and subsequent-row ADX kernels are specialized.
macro_rules! define_fixed_width_basecase_init {
    ($name:ident, $len:literal, $init:ident, $add_mul:ident) => {
        unsafe fn $name(dst: *mut Limb, a: *const Limb, len_a: usize, b: *const Limb) {
            // SAFETY: the caller guarantees len_a >= 2, the complete product
            // span, and the fixed-width source selected by the outer match.
            unsafe {
                $init(dst, b, *a, *a.add(1));
            }
            for index in 2..len_a {
                // SAFETY: every shifted fixed row and carry limb remains in
                // the inherited len_a + fixed-width destination span.
                let carry = unsafe { $add_mul(dst.add(index), b, *a.add(index)) };
                // SAFETY: index < len_a proves index + fixed width is inside
                // the complete product destination.
                unsafe {
                    *dst.add(index.unchecked_add($len)) = carry;
                }
            }
        }
    };
}

/// Complete fixed-width basecase using the generic BMI2 two-row initializer
/// and a fixed-width add-mul row kernel. Used for inner widths 14-17 where
/// only the subsequent-row kernel is specialized.
macro_rules! define_fixed_width_basecase {
    ($name:ident, $len:literal, $add_mul:ident) => {
        unsafe fn $name(dst: *mut Limb, a: *const Limb, len_a: usize, b: *const Limb) {
            // SAFETY: the caller guarantees len_a >= 2, the complete product
            // span, and the fixed-width source selected by the outer match.
            unsafe {
                mul_2_limbs_bmi2_backend(dst, b, $len, *a, *a.add(1));
            }
            for index in 2..len_a {
                // SAFETY: every shifted fixed row and carry limb remains in
                // the inherited len_a + fixed-width destination span.
                let carry = unsafe { $add_mul(dst.add(index), b, *a.add(index)) };
                // SAFETY: index < len_a proves index + fixed width is inside
                // the complete product destination.
                unsafe {
                    *dst.add(index.unchecked_add($len)) = carry;
                }
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
            if len_a == 4 && len_b == 4 {
                // SAFETY: this branch proves both exact input widths and the
                // inherited contract provides the disjoint product span.
                unsafe {
                    mul_4x4_portable_unchecked(dst, a, b);
                }
                return;
            }
            if len_a == 8 && len_b == 8 {
                // SAFETY: this branch proves both exact input widths and the
                // inherited contract provides the disjoint product span.
                unsafe {
                    mul_8x8_portable_unchecked(dst, a, b);
                }
                return;
            }
            // SAFETY: len_a >= 2 and the caller guarantees the complete spans.
            unsafe { $mul_two(dst, b, len_b, *a, *a.add(1)) };
            let mut index = 2_usize;
            // SAFETY: index starts at two and remains <= len_a. Two rows run
            // only when len_a-index >= 2; the length sum bounds all offsets.
            // Setting the extra overlap limb to zero makes existing < B^len_b,
            // so the paired-row sum is < B^(len_b+2) and the high addition is
            // exact. The selected backends have established CPU prerequisites.
            unsafe {
            while len_a.unchecked_sub(index) >= 2 {
                let carry_index0 = index.unchecked_add(len_b);
                let carry_index1 = carry_index0.unchecked_add(1);
                    *dst.add(carry_index0) = 0;
                    let (carry0, carry1) = $add_mul_two(
                        dst.add(index),
                        b,
                        len_b,
                        *a.add(index),
                        *a.add(index.unchecked_add(1)),
                    );
                    let existing = *dst.add(carry_index0);
                    let (sum, overflow) = existing.overflowing_add(carry0);
                    *dst.add(carry_index0) = sum;
                    let top = carry1.unchecked_add(Limb::from(overflow));
                    *dst.add(carry_index1) = top;
                index = index.unchecked_add(2);
            }
            }
            if index < len_a {
                // SAFETY: the final row and carry position lie inside dst.
                let carry = unsafe { $add_mul_one(dst.add(index), b, len_b, *a.add(index)) };
                // SAFETY: index + len_b < len_a + len_b.
                unsafe {
                    *dst.add(index.unchecked_add(len_b)) = carry;
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ADX+BMI2 backend: one match at entry, no per-row dispatch
// ---------------------------------------------------------------------------

// Widths 4-13: fixed ADX two-row init + fixed ADX add-mul rows.
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

// Widths 14-17: generic BMI2 two-row init + fixed ADX add-mul rows.
define_fixed_width_basecase!(basecase_adx_bmi2_14, 14, add_mul_14_adx);
define_fixed_width_basecase!(basecase_adx_bmi2_15, 15, add_mul_15_adx);
define_fixed_width_basecase!(basecase_adx_bmi2_16, 16, add_mul_16_adx);
define_fixed_width_basecase!(basecase_adx_bmi2_17, 17, add_mul_17_adx);

/// Compute the complete product `dst[0..len_a + len_b] = a[0..len_a] x
/// b[0..len_b]` using the ADX+BMI2 backend.
///
/// # Safety
///
/// Aligned initialized inputs cover `len_a >= 2` and `len_b > 0` limbs.
/// Aligned `dst` covers `len_a + len_b` writable, possibly uninitialized limbs;
/// the length sum fits `usize` and all byte spans fit `isize::MAX`.
/// Neither input overlaps `dst`; the CPU supports ADX and BMI2.
unsafe fn mul_basecase_adx_bmi2(
    dst: *mut Limb,
    a: *const Limb,
    len_a: usize,
    b: *const Limb,
    len_b: usize,
) {
    // SAFETY: every fixed-width branch encodes the exact inner width; the
    // generic branch receives the caller-proven len_b. Initialization writes
    // two rows before accumulation reads them. Each closing offset is below
    // len_a+len_b, which the caller proves fits usize. ADX/BMI2 is established.
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
                for index in 2..len_a {
                    let carry = add_mul_limbs_adx_backend(dst.add(index), b, len_b, *a.add(index));
                    *dst.add(index.unchecked_add(len_b)) = carry;
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
