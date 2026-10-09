//! Public constructors generate values across native limb and precision widths.

use proptest::prelude::{Strategy, any, prop_oneof};

use crate::{BoundedPrecision, MpInt, MpUint};

use super::support::uint_from_words;

/// Generates unlimited unsigned values with at most `max_limbs` native limbs.
pub fn uint(max_limbs: usize) -> impl Strategy<Value = MpUint> {
    proptest::collection::vec(any::<usize>(), 0..=max_limbs)
        .prop_map(|words| uint_from_words(&words))
}

/// Generates nonzero unlimited unsigned values.
pub fn uint_nonzero(max_limbs: usize) -> impl Strategy<Value = MpUint> {
    uint(max_limbs).prop_filter("nonzero divisor", |value| !value.is_zero())
}

/// Generates unsigned residues modulo `2^bits`.
pub fn bounded_uint_wrapped(bits: usize) -> impl Strategy<Value = MpUint> {
    let width = BoundedPrecision::new(bits.max(1)).expect("valid test width");
    let max_limbs = width.get().div_ceil(usize::BITS as usize);
    uint(max_limbs).prop_map(move |value| MpUint::with_precision_wrapping(value, width))
}

/// Generates unlimited signed magnitudes with both signs and canonical zero.
pub fn int(max_limbs: usize) -> impl Strategy<Value = MpInt> {
    (uint(max_limbs), any::<bool>()).prop_map(|(magnitude, positive)| {
        let value = MpInt::from(magnitude);
        if positive { value } else { -value }
    })
}

/// Generates signed two's-complement residues with the requested width.
pub fn bounded_int_wrapped(bits: usize) -> impl Strategy<Value = MpInt> {
    let width = BoundedPrecision::new(bits.max(1)).expect("valid test width");
    let max_limbs = width.get().div_ceil(usize::BITS as usize);
    int(max_limbs).prop_map(move |value| MpInt::with_precision_wrapping(value, width))
}

/// Generates either bounded residues or unlimited signed values.
pub fn int_maybe_bounded(bits: usize) -> impl Strategy<Value = MpInt> {
    prop_oneof![
        bounded_int_wrapped(bits),
        int(bits.div_ceil(usize::BITS as usize)),
    ]
}
