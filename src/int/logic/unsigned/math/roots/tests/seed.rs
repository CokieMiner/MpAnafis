//! Exact integer certificates for fixed-point logarithm and exponential tables.

use super::super::{EXP2_LOWER, InternalMpUint, LOG2_LOWER};

#[test]
fn logarithmic_tables_satisfy_their_integer_bounds() {
    let one = InternalMpUint::one();
    let sample = |index: &usize| !cfg!(miri) || [0, 1, 127, 128, 254, 255].contains(index);
    for (index, &entry) in LOG2_LOWER
        .iter()
        .enumerate()
        .filter(|(index, _)| sample(index))
    {
        let value = 256_usize.checked_add(index).expect("nine-bit mantissa");
        let power = InternalMpUint::from_limb(value).pow(256);
        let shift = 2048_usize
            .checked_add(usize::from(entry))
            .expect("table exponent");
        let lower = one.shl(shift);
        assert!(lower <= power);
        assert!(power < lower.shl(1));
    }
    for (index, &entry) in EXP2_LOWER
        .iter()
        .enumerate()
        .filter(|(index, _)| sample(index))
    {
        let value = 256_usize
            .checked_add(usize::from(entry))
            .expect("nine-bit mantissa");
        let successor = value.checked_add(1).expect("mantissa successor");
        let shift = 2048_usize.checked_add(index).expect("table exponent");
        let power = one.shl(shift);
        assert!(InternalMpUint::from_limb(value).pow(256) <= power);
        assert!(power < InternalMpUint::from_limb(successor).pow(256));
    }
}
