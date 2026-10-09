//! Unsigned operators, primitive conversions, iterators, and optional numeric traits.

use core::{
    hash::{Hash, Hasher},
    str::FromStr,
};
use std::collections::hash_map::DefaultHasher;

use mp_anafis::{BoundedPrecision, MpError, MpInt, MpUint, Precision};
#[cfg(feature = "num-traits")]
use num_traits::{FromPrimitive, Num, One, ToPrimitive, Unsigned, Zero};
use rug::Integer;

use crate::{Bounds, Input, assert_integer};

pub fn fuzz_all(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    match input.operation % 5 {
        0 => operators(a, b, ra, rb, input),
        1 => conversions(a, ra, rb),
        2 => shifts(a, ra, input.parameter),
        3 => values(a, b, ra, rb),
        _ => {
            #[cfg(feature = "num-traits")]
            numeric(a, ra, rb);
            #[cfg(not(feature = "num-traits"))]
            conversions(a, ra, rb);
        }
    }
}

fn operators(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let finite = input.flags & 1 == 0;
    let a = if finite {
        MpUint::with_precision_wrapping(a.clone(), BoundedPrecision::new(bits).unwrap())
    } else {
        a.clone()
    };
    let bounds = Bounds {
        bits: finite.then_some(bits),
        signed: false,
    };
    let ra = bounds.wrap(ra);
    macro_rules! operator {
        ($op:tt, $assign:tt, $expected:expr) => {{
            let expected = $expected;
            for result in [a.clone() $op b.clone(), a.clone() $op b, &a $op b.clone(), &a $op b] {
                assert_integer(&result, &expected);
                assert_eq!(result.precision(), Precision::Unlimited);
            }
            if bounds.fits(&expected) {
                let mut owned = a.clone(); owned $assign b.clone();
                let mut borrowed = a.clone(); borrowed $assign b;
                assert_integer(&owned, &expected); assert_integer(&borrowed, &expected);
                assert_eq!(owned.precision(), a.precision()); assert_eq!(borrowed.precision(), a.precision());
            }
        }};
    }
    operator!(+, +=, Integer::from(&ra + rb));
    if ra >= *rb {
        operator!(-, -=, Integer::from(&ra - rb));
    }
    operator!(*, *=, Integer::from(&ra * rb));
    if rb != &0 {
        operator!(/, /=, Integer::from(&ra / rb));
        operator!(%, %=, Integer::from(&ra % rb));
    }
    operator!(&, &=, Integer::from(&ra & rb));
    operator!(|, |=, Integer::from(&ra | rb));
    operator!(^, ^=, Integer::from(&ra ^ rb));
}

fn conversions(a: &MpUint, ra: &Integer, rb: &Integer) {
    macro_rules! unsigned {
        ($($ty:ty),+ $(,)?) => {$(
            // Narrowing selects the low bits at each primitive's width, including usize.
            let input = rb.to_u128_wrapping() as $ty;
            assert_integer(MpUint::from(input), &Integer::from(input));
            let expected = ra.to_u128().and_then(|value| <$ty>::try_from(value).ok());
            assert_eq!(<$ty>::try_from(a.clone()), expected.ok_or(MpError::IntegerConversionLoss));
        )+};
    }
    unsigned!(u8, u16, u32, u64, u128, usize);
    macro_rules! signed {
        ($($ty:ty),+ $(,)?) => {$(
            // Signed narrowing interprets the selected low bits as two's complement.
            let input = rb.to_i128_wrapping() as $ty;
            let actual = MpUint::try_from(input);
            if input < 0 { assert_eq!(actual, Err(MpError::NegativeInput)); }
            else { assert_integer(actual.unwrap(), &Integer::from(input)); }
        )+};
    }
    signed!(i8, i16, i32, i64, i128, isize);
    let signed = MpInt::from(a.clone());
    assert_integer(&signed, ra);
    assert_integer(MpUint::try_from(signed).unwrap(), ra);
}

fn shifts(a: &MpUint, ra: &Integer, parameter: u16) {
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

fn values(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer) {
    assert!(MpUint::default().is_zero());
    assert_integer(MpUint::from_str(&ra.to_string()).unwrap(), ra);
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
        [a.clone(), b.clone()].into_iter().sum::<MpUint>(),
        [a, b].into_iter().sum(),
    ] {
        assert_integer(&result, &sum);
        assert_eq!(result.precision(), Precision::Unlimited);
    }
    for result in [
        [a.clone(), b.clone()].into_iter().product::<MpUint>(),
        [a, b].into_iter().product(),
    ] {
        assert_integer(&result, &product);
        assert_eq!(result.precision(), Precision::Unlimited);
    }
    assert!(core::iter::empty::<MpUint>().sum::<MpUint>().is_zero());
    assert!(core::iter::empty::<&MpUint>().sum::<MpUint>().is_zero());
    assert!(core::iter::empty::<MpUint>().product::<MpUint>().is_one());
    assert!(core::iter::empty::<&MpUint>().product::<MpUint>().is_one());
    let bounded = MpUint::with_precision_checked(
        a.clone(),
        BoundedPrecision::new(usize::try_from(ra.significant_bits()).unwrap().max(1)).unwrap(),
    )
    .unwrap();
    let mut hash_a = DefaultHasher::new();
    a.hash(&mut hash_a);
    let mut hash_b = DefaultHasher::new();
    bounded.hash(&mut hash_b);
    assert_eq!(hash_a.finish(), hash_b.finish());
    assert_eq!(a, &bounded);
    let signed_b = MpInt::from(b.clone());
    assert_eq!(a.partial_cmp(&signed_b), Some(ra.cmp(rb)));
    assert_eq!(a == &signed_b, ra == rb);
    let negative_b = -signed_b;
    assert_eq!(a.partial_cmp(&negative_b), Some(ra.cmp(&-rb.clone())));
    assert_eq!(a == &negative_b, ra == &-rb.clone());
}

#[cfg(feature = "num-traits")]
fn numeric(a: &MpUint, ra: &Integer, rb: &Integer) {
    require_unsigned(a);
    let mut zero = a.clone();
    <MpUint as Zero>::set_zero(&mut zero);
    assert_eq!(zero.precision(), Precision::Unlimited);
    assert!(<MpUint as Zero>::is_zero(&zero));
    let mut one = a.clone();
    <MpUint as One>::set_one(&mut one);
    assert_eq!(one.precision(), Precision::Unlimited);
    assert!(<MpUint as One>::is_one(&one));
    assert!(<MpUint as Zero>::is_zero(&<MpUint as Zero>::zero()));
    assert!(<MpUint as One>::is_one(&<MpUint as One>::one()));
    assert_integer(
        <MpUint as Num>::from_str_radix(&ra.to_string(), 10).unwrap(),
        ra,
    );
    macro_rules! primitive {
        ($($ty:ty, $from:ident, $to:ident, $ref:ident);+ $(;)?) => {$(
            // Narrowing provides a value in every primitive domain on all pointer widths.
            let value = rb.to_i128_wrapping() as $ty;
            let expected = Integer::from(value);
            assert_eq!(<MpUint as FromPrimitive>::$from(value).map(|value| value.to_string()), (expected >= 0).then(|| expected.to_string()));
            assert_eq!(<MpUint as ToPrimitive>::$to(a), ra.$ref().and_then(|value| <$ty>::try_from(value).ok()));
        )+};
    }
    primitive!(u8, from_u8, to_u8, to_u128; u16, from_u16, to_u16, to_u128; u32, from_u32, to_u32, to_u128; u64, from_u64, to_u64, to_u128; u128, from_u128, to_u128, to_u128; usize, from_usize, to_usize, to_u128; i8, from_i8, to_i8, to_i128; i16, from_i16, to_i16, to_i128; i32, from_i32, to_i32, to_i128; i64, from_i64, to_i64, to_i128; i128, from_i128, to_i128, to_i128; isize, from_isize, to_isize, to_i128);
    assert_eq!(<MpUint as ToPrimitive>::to_f32(a), crate::float32(ra));
    assert_eq!(<MpUint as ToPrimitive>::to_f64(a), crate::float64(ra));
    let float = f64::from_bits(rb.to_u64_wrapping());
    let expected = float
        .to_i64()
        .map(Integer::from)
        .or_else(|| float.to_u64().map(Integer::from))
        .filter(|value| value >= &0);
    assert_eq!(
        <MpUint as FromPrimitive>::from_f64(float).map(|value| value.to_string()),
        expected.map(|value| value.to_string())
    );
    let float = f32::from_bits(rb.to_u32_wrapping());
    let expected = float
        .to_i64()
        .map(Integer::from)
        .or_else(|| float.to_u64().map(Integer::from))
        .filter(|value| value >= &0);
    assert_eq!(
        <MpUint as FromPrimitive>::from_f32(float).map(|value| value.to_string()),
        expected.map(|value| value.to_string())
    );
}

#[cfg(feature = "num-traits")]
fn require_unsigned<T: Unsigned>(_: &T) {}
