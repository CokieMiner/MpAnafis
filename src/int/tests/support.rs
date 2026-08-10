//! Shared constructors and strategies for integer properties.

#[cfg(feature = "num-traits")]
use super::ArbiInt;
use super::{ArbiUint, BoundedPrecision, InternalArbiUint, Limb, Precision, Strategy, Vec, any};
#[cfg(feature = "std")]
use super::{DefaultHasher, Hash, Hasher};

pub fn nz(bits: usize) -> BoundedPrecision {
    BoundedPrecision::new(bits).expect("valid bounded width")
}

pub fn uint(value: u64) -> ArbiUint {
    ArbiUint {
        value: InternalArbiUint::from_u64(value),
        precision: Precision::Unlimited,
    }
}

#[cfg(feature = "num-traits")]
pub fn int_from_i64(value: i64) -> ArbiInt {
    ArbiInt::from(value)
}

pub fn exact_limb_vec(len: usize) -> impl Strategy<Value = Vec<Limb>> {
    proptest::collection::vec(any::<Limb>(), len)
}

#[cfg(feature = "std")]
pub fn hash_u64(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
