//! Untimed GMP reference products and limb ABI validation.

#![expect(
    unsafe_code,
    reason = "GMP reference calls use validated limb layouts and disjoint output buffers"
)]

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::Limb;

/// Computes an equal-width product outside the measured closure.
pub fn gmp_equal_reference(left: &[Limb], right: &[Limb]) -> Vec<Limb> {
    const { assert_gmp_limb_width() }
    assert_eq!(left.len(), right.len(), "equal-width operands differ");
    let count = validated_gmp_count(left.len());
    let result_len = left.len().checked_mul(2).expect("product width fits usize");
    let mut expected = vec![Limb::MIN; result_len];
    // SAFETY: both inputs contain `count > 0` limbs with GMP's size/alignment;
    // the fresh output is disjoint and has exactly 2*count initialized limbs.
    // Read-only inputs may alias, including when computing a square.
    unsafe {
        gmp::mpn_mul_n(
            expected.as_mut_ptr().cast::<limb_t>(),
            left.as_ptr().cast::<limb_t>(),
            right.as_ptr().cast::<limb_t>(),
            count,
        );
    }
    expected
}

/// Computes an ordered rectangular product outside the measured closure.
#[cfg(target_os = "linux")]
pub fn gmp_pair_reference(larger: &[Limb], smaller: &[Limb]) -> Vec<Limb> {
    const { assert_gmp_limb_width() }
    let (larger_count, smaller_count) = validated_gmp_counts(larger.len(), smaller.len());
    let result_len = larger
        .len()
        .checked_add(smaller.len())
        .expect("product width fits usize");
    let mut expected = vec![Limb::MIN; result_len];
    // SAFETY: validated counts satisfy larger >= smaller >= 1; input size and
    // alignment match GMP. The fresh output is disjoint and holds their full
    // product. Read-only inputs may overlap.
    unsafe {
        let _high = gmp::mpn_mul(
            expected.as_mut_ptr().cast::<limb_t>(),
            larger.as_ptr().cast::<limb_t>(),
            larger_count,
            smaller.as_ptr().cast::<limb_t>(),
            smaller_count,
        );
    }
    expected
}

/// Validates GMP's ordered, nonempty operand counts before timing.
#[cfg(target_os = "linux")]
pub fn validated_gmp_counts(larger: usize, smaller: usize) -> (size_t, size_t) {
    assert!(larger >= smaller, "GMP requires the longer operand first");
    (validated_gmp_count(larger), validated_gmp_count(smaller))
}

/// Validates GMP's positive signed limb count before timing.
pub fn validated_gmp_count(len: usize) -> size_t {
    assert!(len > 0, "GMP requires nonempty operands");
    size_t::try_from(len).expect("benchmark width must fit a GMP size")
}

/// Proves the layout required by the raw limb pointer casts.
pub const fn assert_gmp_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<limb_t>(),
        "GMP limb size differs"
    );
    assert!(
        align_of::<Limb>() == align_of::<limb_t>(),
        "GMP limb alignment differs"
    );
}
