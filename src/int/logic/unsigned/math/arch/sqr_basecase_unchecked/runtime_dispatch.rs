//! One-shot x86-64 CPU dispatch for complete schoolbook squaring.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, adx_square, bmi2_square, portable_square, selected_x86_backend,
};

type SqrFn = unsafe fn(*mut Limb, *const Limb, usize);
static KERNEL: OnceLock<SqrFn> = OnceLock::new();

/// Selects the complete-square backend once for widths above eight limbs.
///
/// The dual-chain kernel needs both ADX and BMI2, the `mulx`+`adcq` kernel
/// needs only BMI2, and everything else runs the portable driver. ADX without
/// BMI2 has no accelerated kernel because every hardware variant relies on
/// `mulx`.
///
/// # Safety
///
/// `a` must cover `len` aligned initialized readable limbs. `dst` must cover
/// `2 * len` aligned writable limbs, disjoint from `a`; its contents may be
/// uninitialized. The complete output byte span must fit in `isize::MAX`.
/// `len == 0` performs no pointer access.
#[inline]
pub unsafe fn sqr_basecase_unchecked(dst: *mut Limb, a: *const Limb, len: usize) {
    if len <= 8 {
        // SAFETY: fixed-width kernels for N <= 8 need no CPU dispatch.
        unsafe {
            portable_square(dst, a, len);
        }
        return;
    }
    let kernel = *KERNEL.get_or_init(|| match selected_x86_backend() {
        X86Backend::AdxBmi2 => adx_square,
        X86Backend::Bmi2 => bmi2_square,
        X86Backend::Adx | X86Backend::Baseline => portable_square,
    });
    // SAFETY: the caller provides the complete aligned disjoint spans. len>8
    // meets both hardware backends' size bounds; selection proves their CPU
    // features. Every backend initializes all 2*len output limbs.
    unsafe {
        kernel(dst, a, len);
    }
}
