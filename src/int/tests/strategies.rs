//! Proptest strategies for [`ArbiUint`] and [`ArbiInt`].
//!
//! These strategies are used by all property-based tests in the `tests/` module.
//! They generate arbitrarily-sized values (up to a configurable max-limb count)
//! and rely on proptest's shrinking to find minimal failing cases.

use proptest::prelude::*;

use super::super::{
    api::{ArbiInt, ArbiUint, BoundedPrecision, Precision},
    logic::{InternalArbiInt, InternalArbiUint},
    types::Limb,
};

// ---------------------------------------------------------------------------
// Strategies for InternalArbiUint (the raw unsigned representation)
// ---------------------------------------------------------------------------

/// Strategy that generates a random `InternalArbiUint` with up to `max_limbs` limbs.
fn internal_uint_limbs(max_limbs: usize) -> impl Strategy<Value = InternalArbiUint> {
    proptest::collection::vec(any::<Limb>(), 0..=max_limbs).prop_map(InternalArbiUint::from_limbs)
}

/// Strategy that generates a non-zero `InternalArbiUint` with up to `max_limbs` limbs.
fn internal_uint_nonzero(max_limbs: usize) -> impl Strategy<Value = InternalArbiUint> {
    internal_uint_limbs(max_limbs).prop_filter("value must be non-zero", |v| !v.is_zero())
}

// ---------------------------------------------------------------------------
// Strategies for ArbiUint (the public unsigned wrapper)
// ---------------------------------------------------------------------------

/// Strategy that generates an `ArbiUint` with unlimited precision and up to
/// `max_limbs` limbs.
pub fn uint(max_limbs: usize) -> impl Strategy<Value = ArbiUint> {
    internal_uint_limbs(max_limbs).prop_map(|value| ArbiUint {
        value,
        precision: Precision::Unlimited,
    })
}

/// Strategy that generates a non-zero `ArbiUint` with unlimited precision.
pub fn uint_nonzero(max_limbs: usize) -> impl Strategy<Value = ArbiUint> {
    internal_uint_nonzero(max_limbs).prop_map(|value| ArbiUint {
        value,
        precision: Precision::Unlimited,
    })
}

/// Strategy that generates a bounded-precision `ArbiUint` with `bits` width,
/// and applies unsigned wrapping to ensure the value fits.
pub fn bounded_uint_wrapped(bits: usize) -> impl Strategy<Value = ArbiUint> {
    let effective_bits = bits.max(1);
    let bounded_width = BoundedPrecision::new(effective_bits).expect("bits are in range");
    let max_limbs = effective_bits
        .div_ceil(crate::int::types::Limb::BITS as usize)
        .max(1);
    internal_uint_limbs(max_limbs).prop_map(move |value| ArbiUint {
        value: value.apply_wrapping(effective_bits),
        precision: Precision::Bounded(bounded_width),
    })
}

// ---------------------------------------------------------------------------
// Strategies for ArbiInt (the public signed wrapper)
// ---------------------------------------------------------------------------

/// Strategy that generates an `ArbiInt` with unlimited precision and up to
/// `max_limbs` limbs for the magnitude.
pub fn int(max_limbs: usize) -> impl Strategy<Value = ArbiInt> {
    (internal_uint_limbs(max_limbs), any::<bool>()).prop_map(|(mag, positive)| {
        let pos = positive || mag.is_zero();
        ArbiInt {
            value: InternalArbiInt {
                abs: mag,
                is_positive: pos,
            },
            precision: Precision::Unlimited,
        }
    })
}

/// Strategy that generates a non-zero `ArbiInt` with unlimited precision.
pub fn int_nonzero(max_limbs: usize) -> impl Strategy<Value = ArbiInt> {
    (internal_uint_nonzero(max_limbs), any::<bool>()).prop_map(|(mag, is_positive)| ArbiInt {
        value: InternalArbiInt {
            abs: mag,
            is_positive,
        },
        precision: Precision::Unlimited,
    })
}

// ---------------------------------------------------------------------------
// Strategies for bounded-precision integers
// ---------------------------------------------------------------------------

/// Strategy that generates a bounded-precision `ArbiInt` with `bits` width,
/// and applies signed wrapping to ensure the value fits.
pub fn bounded_int_wrapped(bits: usize) -> impl Strategy<Value = ArbiInt> {
    fn apply_signed_wrapping_for_bits(v: InternalArbiInt, bits: usize) -> InternalArbiInt {
        v.apply_wrapping(bits.max(1))
    }
    let bounded_width = BoundedPrecision::new(bits.max(1)).expect("bits are in range");
    let max_limbs = bits.div_ceil(crate::int::types::Limb::BITS as usize).max(1);
    (internal_uint_limbs(max_limbs), any::<bool>()).prop_map(move |(mag, positive)| {
        let pos = positive || mag.is_zero();
        let raw = InternalArbiInt {
            abs: mag,
            is_positive: pos,
        };
        let value = apply_signed_wrapping_for_bits(raw, bits);
        ArbiInt {
            value,
            precision: Precision::Bounded(bounded_width),
        }
    })
}

/// Strategy that generates an `ArbiInt` with either bounded or unlimited precision.
pub fn int_maybe_bounded(bits: usize) -> impl Strategy<Value = ArbiInt> {
    proptest::bool::weighted(0.5).prop_flat_map(move |bounded| {
        if bounded {
            bounded_int_wrapped(bits).boxed()
        } else {
            int(bits.div_ceil(crate::int::types::Limb::BITS as usize)).boxed()
        }
    })
}
