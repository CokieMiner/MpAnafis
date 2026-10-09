//! Signed operators, primitive conversions, iterators, and optional numeric traits.

use core::{
    hash::{Hash, Hasher},
    str::FromStr,
};
use std::collections::hash_map::DefaultHasher;

use mp_anafis::{BoundedPrecision, MpError, MpInt, MpUint, Precision};
#[cfg(feature = "num-traits")]
use num_traits::{FromPrimitive, Num, One, Signed, ToPrimitive, Zero};
use rug::Integer;

use crate::{Bounds, Input, assert_integer};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    match input.operation % 5 {
        0 => operators(a, b, ra, rb, input),
        1 => conversions(a, ra, rb),
        2 => shifts(a, ra, input.parameter),
        3 => values(a, b, ra, rb),
        _ => {
            #[cfg(feature = "num-traits")]
            numeric(a, b, ra, rb);
            #[cfg(not(feature = "num-traits"))]
            conversions(a, ra, rb);
        }
    }
}

fn operators(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let finite = input.flags & 1 == 0;
    let a = if finite {
        MpInt::with_precision_wrapping(a.clone(), BoundedPrecision::new(bits).unwrap())
    } else {
        a.clone()
    };
    let bounds = Bounds {
        bits: finite.then_some(bits),
        signed: true,
    };
    let ra = bounds.wrap(ra);
    let remainder_overflow = rb == &-1
        && bounds
            .bits
            .is_some_and(|bits| ra == -(Integer::from(1) << (bits - 1)));
    macro_rules! operator {
        ($op:tt, $assign:tt, $expected:expr) => {{
            let expected = $expected;
            for result in [a.clone() $op b.clone(), a.clone() $op b, &a $op b.clone(), &a $op b] {
                assert_integer(&result, &expected); assert_eq!(result.precision(), Precision::Unlimited);
            }
            if bounds.fits(&expected) && (stringify!($op) != "%" || !remainder_overflow) {
                let mut owned = a.clone(); owned $assign b.clone();
                let mut borrowed = a.clone(); borrowed $assign b;
                assert_integer(&owned, &expected); assert_integer(&borrowed, &expected);
                assert_eq!(owned.precision(), a.precision()); assert_eq!(borrowed.precision(), a.precision());
            }
        }};
    }
    operator!(+, +=, Integer::from(&ra + rb));
    operator!(-, -=, Integer::from(&ra - rb));
    operator!(*, *=, Integer::from(&ra * rb));
    if rb != &0 {
        operator!(/, /=, Integer::from(&ra / rb));
        operator!(%, %=, Integer::from(&ra % rb));
    }
    operator!(&, &=, Integer::from(&ra & rb));
    operator!(|, |=, Integer::from(&ra | rb));
    operator!(^, ^=, Integer::from(&ra ^ rb));
    if bounds.fits(&-ra.clone()) {
        assert_integer(-&a, &-ra.clone());
        assert_integer(-a.clone(), &-ra);
    }
}

fn conversions(a: &MpInt, ra: &Integer, rb: &Integer) {
    macro_rules! primitive {
        ($($ty:ty, $ref:ident);+ $(;)?) => {$(
            // Narrowing selects and interprets low bits at each primitive width.
            let input = rb.to_i128_wrapping() as $ty;
            assert_integer(MpInt::from(input), &Integer::from(input));
            let expected = ra.$ref().and_then(|value| <$ty>::try_from(value).ok());
            assert_eq!(<$ty>::try_from(a.clone()), expected.ok_or(MpError::IntegerConversionLoss));
        )+};
    }
    primitive!(u8, to_u128; u16, to_u128; u32, to_u128; u64, to_u128; u128, to_u128; usize, to_u128; i8, to_i128; i16, to_i128; i32, to_i128; i64, to_i128; i128, to_i128; isize, to_i128);
    let unsigned = MpUint::try_from(a.clone());
    if ra < &0 {
        assert_eq!(unsigned, Err(MpError::NegativeInput));
    } else {
        assert_integer(MpInt::from(unsigned.unwrap()), ra);
    }
}

fn shifts(a: &MpInt, ra: &Integer, parameter: u16) {
    // Counts below 128 fit every supported signed and unsigned shift operand.
    let shift = parameter % 128;
    let left = ra.clone() << usize::from(shift);
    let right = ra.clone() >> usize::from(shift);
    macro_rules! counts {
        ($($ty:ty),+ $(,)?) => {$(
            let count = <$ty>::try_from(shift).unwrap();
            assert_integer(a.clone() << count, &left); assert_integer(a << count, &left);
            assert_integer(a.clone() >> count, &right); assert_integer(a >> count, &right);
            let mut value = a.clone(); value <<= count; assert_integer(&value, &left);
            let mut value = a.clone(); value >>= count; assert_integer(&value, &right);
        )+};
    }
    counts!(
        u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
    );
}

fn values(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer) {
    assert!(MpInt::default().is_zero());
    assert_integer(MpInt::from_str(&ra.to_string()).unwrap(), ra);
    assert!(!format!("{a:?}").is_empty());
    assert_eq!(format!("{a:b}"), format!("{ra:b}"));
    assert_eq!(format!("{a:o}"), format!("{ra:o}"));
    assert_eq!(format!("{a:x}"), format!("{ra:x}"));
    assert_eq!(format!("{a:X}"), format!("{ra:X}"));
    assert_eq!(format!("{a:+#032x}"), format!("{ra:+#032x}"));
    assert_eq!(format!("{a:>40}"), format!("{ra:>40}"));
    let sum = Integer::from(ra + rb);
    let product = Integer::from(ra * rb);
    for result in [
        [a.clone(), b.clone()].into_iter().sum::<MpInt>(),
        [a, b].into_iter().sum(),
    ] {
        assert_integer(&result, &sum);
        assert_eq!(result.precision(), Precision::Unlimited);
    }
    for result in [
        [a.clone(), b.clone()].into_iter().product::<MpInt>(),
        [a, b].into_iter().product(),
    ] {
        assert_integer(&result, &product);
        assert_eq!(result.precision(), Precision::Unlimited);
    }
    assert!(core::iter::empty::<MpInt>().sum::<MpInt>().is_zero());
    assert!(core::iter::empty::<&MpInt>().sum::<MpInt>().is_zero());
    assert!(core::iter::empty::<MpInt>().product::<MpInt>().is_one());
    assert!(core::iter::empty::<&MpInt>().product::<MpInt>().is_one());
    let bits = usize::try_from(ra.significant_bits()).unwrap() + 1;
    let bounded =
        MpInt::with_precision_checked(a.clone(), BoundedPrecision::new(bits).unwrap()).unwrap();
    let mut hash_a = DefaultHasher::new();
    a.hash(&mut hash_a);
    let mut hash_b = DefaultHasher::new();
    bounded.hash(&mut hash_b);
    assert_eq!(hash_a.finish(), hash_b.finish());
    assert_eq!(a, &bounded);
    let unsigned_b = MpUint::from_str_radix(&rb.clone().abs().to_string(), 10).unwrap();
    assert_eq!(a.partial_cmp(&unsigned_b), Some(ra.cmp(&rb.clone().abs())));
    assert_eq!(a == &unsigned_b, ra == &rb.clone().abs());
}

#[cfg(feature = "num-traits")]
fn numeric(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer) {
    let mut zero = a.clone();
    <MpInt as Zero>::set_zero(&mut zero);
    assert_eq!(zero.precision(), Precision::Unlimited);
    assert!(<MpInt as Zero>::is_zero(&zero));
    let mut one = a.clone();
    <MpInt as One>::set_one(&mut one);
    assert_eq!(one.precision(), Precision::Unlimited);
    assert!(<MpInt as One>::is_one(&one));
    assert!(<MpInt as Zero>::is_zero(&<MpInt as Zero>::zero()));
    assert!(<MpInt as One>::is_one(&<MpInt as One>::one()));
    assert_integer(
        <MpInt as Num>::from_str_radix(&ra.to_string(), 10).unwrap(),
        ra,
    );
    assert_integer(<MpInt as Signed>::abs(a), &ra.clone().abs());
    assert_integer(
        <MpInt as Signed>::abs_sub(a, b),
        &Integer::from(ra - rb).max(Integer::new()),
    );
    assert_integer(
        <MpInt as Signed>::signum(a),
        &Integer::from(i32::from(ra > &0) - i32::from(ra < &0)),
    );
    assert_eq!(<MpInt as Signed>::is_positive(a), ra > &0);
    assert_eq!(<MpInt as Signed>::is_negative(a), ra < &0);
    macro_rules! primitive {
        ($($ty:ty, $from:ident, $to:ident, $ref:ident);+ $(;)?) => {$(
            // Narrowing provides every primitive domain on all pointer widths.
            let value = rb.to_i128_wrapping() as $ty;
            assert_integer(<MpInt as FromPrimitive>::$from(value).unwrap(), &Integer::from(value));
            assert_eq!(<MpInt as ToPrimitive>::$to(a), ra.$ref().and_then(|value| <$ty>::try_from(value).ok()));
        )+};
    }
    primitive!(u8, from_u8, to_u8, to_u128; u16, from_u16, to_u16, to_u128; u32, from_u32, to_u32, to_u128; u64, from_u64, to_u64, to_u128; u128, from_u128, to_u128, to_u128; usize, from_usize, to_usize, to_u128; i8, from_i8, to_i8, to_i128; i16, from_i16, to_i16, to_i128; i32, from_i32, to_i32, to_i128; i64, from_i64, to_i64, to_i128; i128, from_i128, to_i128, to_i128; isize, from_isize, to_isize, to_i128);
    assert_eq!(<MpInt as ToPrimitive>::to_f32(a), crate::float32(ra));
    assert_eq!(<MpInt as ToPrimitive>::to_f64(a), crate::float64(ra));
    let float = f64::from_bits(rb.to_u64_wrapping());
    let expected = float
        .to_i64()
        .map(Integer::from)
        .or_else(|| float.to_u64().map(Integer::from));
    assert_eq!(
        <MpInt as FromPrimitive>::from_f64(float).map(|value| value.to_string()),
        expected.map(|value| value.to_string())
    );
    let float = f32::from_bits(rb.to_u32_wrapping());
    let expected = float
        .to_i64()
        .map(Integer::from)
        .or_else(|| float.to_u64().map(Integer::from));
    assert_eq!(
        <MpInt as FromPrimitive>::from_f32(float).map(|value| value.to_string()),
        expected.map(|value| value.to_string())
    );
}
