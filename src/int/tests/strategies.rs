//! Proptest strategies for [`MpUint`] and [`MpInt`].
//!
//! These strategies are used by all property-based tests in the `tests/` module.
//! They generate arbitrarily-sized values (up to a configurable max-limb count)
//! and rely on proptest's shrinking to find minimal failing cases.

use proptest::prelude::*;

use super::super::{
    api::{BoundedPrecision, MpInt, MpUint, Precision},
    logic::{InternalMpInt, InternalMpUint},
    types::Limb,
};

// ---------------------------------------------------------------------------
// Strategies for InternalMpUint (the raw unsigned representation)
// ---------------------------------------------------------------------------

/// Strategy that generates a random `InternalMpUint` with up to `max_limbs` limbs.
fn internal_uint_limbs(max_limbs: usize) -> impl Strategy<Value = InternalMpUint> {
    proptest::collection::vec(any::<Limb>(), 0..=max_limbs).prop_map(InternalMpUint::from_limbs)
}

/// Strategy that generates a non-zero `InternalMpUint` with up to `max_limbs` limbs.
fn internal_uint_nonzero(max_limbs: usize) -> impl Strategy<Value = InternalMpUint> {
    internal_uint_limbs(max_limbs).prop_filter("value must be non-zero", |v| !v.is_zero())
}

// ---------------------------------------------------------------------------
// Strategies for MpUint (the public unsigned wrapper)
// ---------------------------------------------------------------------------

/// Strategy that generates an `MpUint` with unlimited precision and up to
/// `max_limbs` limbs.
pub fn uint(max_limbs: usize) -> impl Strategy<Value = MpUint> {
    internal_uint_limbs(max_limbs).prop_map(|value| MpUint {
        value,
        precision: Precision::Unlimited,
    })
}

/// Strategy that generates a non-zero `MpUint` with unlimited precision.
pub fn uint_nonzero(max_limbs: usize) -> impl Strategy<Value = MpUint> {
    internal_uint_nonzero(max_limbs).prop_map(|value| MpUint {
        value,
        precision: Precision::Unlimited,
    })
}

/// Strategy that generates a bounded-precision `MpUint` with `bits` width,
/// and applies unsigned wrapping to ensure the value fits.
pub fn bounded_uint_wrapped(bits: usize) -> impl Strategy<Value = MpUint> {
    let effective_bits = bits.max(1);
    let bounded_width = BoundedPrecision::new(effective_bits).expect("bits are in range");
    let max_limbs = effective_bits
        .div_ceil(crate::int::types::Limb::BITS as usize)
        .max(1);
    internal_uint_limbs(max_limbs).prop_map(move |value| MpUint {
        value: value.apply_wrapping(effective_bits),
        precision: Precision::Bounded(bounded_width),
    })
}

// ---------------------------------------------------------------------------
// Strategies for MpInt (the public signed wrapper)
// ---------------------------------------------------------------------------

/// Strategy that generates an `MpInt` with unlimited precision and up to
/// `max_limbs` limbs for the magnitude.
pub fn int(max_limbs: usize) -> impl Strategy<Value = MpInt> {
    (internal_uint_limbs(max_limbs), any::<bool>()).prop_map(|(mag, positive)| {
        let pos = positive || mag.is_zero();
        MpInt {
            value: InternalMpInt {
                abs: mag,
                is_positive: pos,
            },
            precision: Precision::Unlimited,
        }
    })
}

/// Strategy that generates a non-zero `MpInt` with unlimited precision.
pub fn int_nonzero(max_limbs: usize) -> impl Strategy<Value = MpInt> {
    (internal_uint_nonzero(max_limbs), any::<bool>()).prop_map(|(mag, is_positive)| MpInt {
        value: InternalMpInt {
            abs: mag,
            is_positive,
        },
        precision: Precision::Unlimited,
    })
}

// ---------------------------------------------------------------------------
// Strategies for bounded-precision integers
// ---------------------------------------------------------------------------

/// Strategy that generates a bounded-precision `MpInt` with `bits` width,
/// and applies signed wrapping to ensure the value fits.
pub fn bounded_int_wrapped(bits: usize) -> impl Strategy<Value = MpInt> {
    fn apply_signed_wrapping_for_bits(v: InternalMpInt, bits: usize) -> InternalMpInt {
        v.apply_wrapping(bits.max(1))
    }
    let bounded_width = BoundedPrecision::new(bits.max(1)).expect("bits are in range");
    let max_limbs = bits.div_ceil(crate::int::types::Limb::BITS as usize).max(1);
    (internal_uint_limbs(max_limbs), any::<bool>()).prop_map(move |(mag, positive)| {
        let pos = positive || mag.is_zero();
        let raw = InternalMpInt {
            abs: mag,
            is_positive: pos,
        };
        let value = apply_signed_wrapping_for_bits(raw, bits);
        MpInt {
            value,
            precision: Precision::Bounded(bounded_width),
        }
    })
}

/// Strategy that generates an `MpInt` with either bounded or unlimited precision.
pub fn int_maybe_bounded(bits: usize) -> impl Strategy<Value = MpInt> {
    proptest::bool::weighted(0.5).prop_flat_map(move |bounded| {
        if bounded {
            bounded_int_wrapped(bits).boxed()
        } else {
            int(bits.div_ceil(crate::int::types::Limb::BITS as usize)).boxed()
        }
    })
}
