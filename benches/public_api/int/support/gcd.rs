//! Shared operand distributions for GCD, cofactors, inversion, and Jacobi.
//! General pairs span two seed streams. Fibonacci fixtures repeat one exact
//! consecutive pair ten times; fast doubling keeps their setup quasilinear.

use core::mem::swap;

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use super::{FlintInt, pin_flint_to_one_thread};
use super::{SAMPLES, mp_uint};

/// Algebraic structure varied independently of the benchmarked operation.
#[derive(Clone, Copy, Debug)]
pub enum GcdShape {
    NearEqualSmall,
    NearEqualWide,
    Fibonacci,
    Uneven,
    SharedFactor,
    TrailingZeros,
    ExactMultiple,
    Scalar,
}

/// Builds the same positive pairs for all reduction consumers. Denominators
/// are odd, including the Jacobi and noninvertible shared-factor cases.
pub fn mp_gcd_pairs(bits: usize, shape: GcdShape) -> Vec<(MpUint, MpUint)> {
    assert!(
        bits >= 64 && bits.is_multiple_of(4),
        "GCD widths start at 64 bits"
    );
    let mut fibonacci = None;
    let half = (bits >> 3) << 2;
    let high_bit = MpUint::one() << bits.checked_sub(1).expect("positive width");
    (0..SAMPLES)
        .map(|index| {
            let seed = index.wrapping_add(if index < (SAMPLES >> 1) { 42 } else { 2_021 });
            // Prefix 10 leaves room for bounded differences without changing
            // the registered width. Odd denominators admit Jacobi on all pairs.
            let mut right =
                (mp_uint(bits, seed.wrapping_add(1_337)) >> 2_usize) | &high_bit | MpUint::one();
            let left = match shape {
                GcdShape::NearEqualSmall => &right + MpUint::from(2_u8),
                GcdShape::NearEqualWide => &right + mp_uint(half, seed),
                GcdShape::Uneven => mp_uint((bits >> 5).max(1) << 2, seed),
                GcdShape::SharedFactor => {
                    let factor = (MpUint::one() << half) + MpUint::one();
                    let left_cofactor = mp_uint(half, seed) | MpUint::one();
                    let right_cofactor = mp_uint(half, seed.wrapping_add(1_337)) | MpUint::one();
                    right = right_cofactor * &factor;
                    left_cofactor * factor
                }
                GcdShape::TrailingZeros => {
                    let base = mp_uint(half, seed) | MpUint::one();
                    base << bits
                        .checked_sub(half)
                        .expect("half width is below full width")
                }
                GcdShape::ExactMultiple => {
                    right = (right >> 2_usize) | MpUint::one();
                    &right * MpUint::from(3_u8)
                }
                GcdShape::Scalar => MpUint::from(seed.wrapping_mul(2).wrapping_add(5)),
                GcdShape::Fibonacci => {
                    return fibonacci
                        .get_or_insert_with(|| fibonacci_pair(bits))
                        .clone();
                }
            };
            assert!(right.is_odd(), "Jacobi denominator stays odd");
            assert!(
                left.significant_bits() <= bits && right.significant_bits() <= bits,
                "fixture respects its width bound"
            );
            (left, right)
        })
        .collect()
}

/// Converts prepared magnitudes outside timing, preserving exact input identity.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn rug_gcd_pairs(bits: usize, shape: GcdShape) -> Vec<(Integer, Integer)> {
    mp_gcd_pairs(bits, shape)
        .into_iter()
        .map(|(left, right)| {
            (
                Integer::from_str_radix(&left.to_string_radix(16), 16)
                    .expect("left operand parses"),
                Integer::from_str_radix(&right.to_string_radix(16), 16)
                    .expect("right operand parses"),
            )
        })
        .collect()
}

/// Constructs consecutive Fibonacci numbers near the requested width using
/// F(2k)=F(k)*(2*F(k+1)-F(k)) and F(2k+1)=F(k)^2+F(k+1)^2.
fn fibonacci_pair(bits: usize) -> (MpUint, MpUint) {
    let index = u64::try_from(bits)
        .expect("benchmark width fits u64")
        .checked_mul(144_042)
        .expect("Fibonacci index estimate fits")
        .checked_div(100_000)
        .expect("positive scale");
    let mut previous = MpUint::zero();
    let mut current = MpUint::one();
    for shift in (0..u64::BITS
        .checked_sub(index.leading_zeros())
        .expect("bit length fits"))
        .rev()
    {
        let even = &previous * ((&current << 1_usize) - &previous);
        let odd = &previous * &previous + &current * &current;
        if index & (1_u64 << shift) == 0 {
            previous = even;
            current = odd;
        } else {
            previous = odd;
            current = even + &previous;
        }
    }
    loop {
        let next = &previous + &current;
        if next.significant_bits() > bits {
            break;
        }
        previous = current;
        current = next;
    }
    assert_eq!(
        current.significant_bits(),
        bits,
        "largest Fibonacci operand has the registered width"
    );
    // Consecutive Fibonacci numbers cannot both be even. Orientation preserves
    // the same long quotient-one trajectory and provides an odd denominator.
    if previous.is_even() {
        swap(&mut previous, &mut current);
    }
    (current, previous)
}

/// Adds the same distributions below each operation's existing random case.
#[macro_export]
macro_rules! gcd_scenarios {
    ($widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr
        $(, flint: $flint_op:expr)?) => {
        $crate::int::support::gcd_scenarios!(@cases $widths, $size, $count, $mp_op, $rug_op, $verify, [$(flint: $flint_op)?];
            near_equal_2 = NearEqualSmall, near_equal_wide = NearEqualWide,
            fibonacci = Fibonacci, uneven = Uneven, shared_factor = SharedFactor,
            many_trailing_zeros = TrailingZeros, exact_multiple = ExactMultiple, scalar = Scalar);
    };
    (@cases $widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr, $references:tt;
        $($name:ident = $shape:ident),+ $(,)?) => { $(
        $crate::int::support::gcd_scenarios!(@case $widths, $size, $count, $mp_op, $rug_op, $verify, $references; $name = $shape);
    )+ };
    (@case $widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr,
        [$(flint: $flint_op:expr)?]; $name:ident = $shape:ident) => {
        $crate::int::support::paired_bench!($name, $widths, samples = ($size, $count),
            mp: |bits| $crate::int::support::mp_gcd_pairs(bits, $crate::int::support::GcdShape::$shape)
                => $mp_op,
            rug: |bits| $crate::int::support::rug_gcd_pairs(bits, $crate::int::support::GcdShape::$shape)
                => $rug_op,
            verify = $verify,
            $(flint: |bits| $crate::int::support::flint_gcd_pairs(bits, $crate::int::support::GcdShape::$shape)
                => $flint_op,)?
        );
    };
}

/// Converts the verified positive GCD fixtures without timing conversion.
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub fn flint_gcd_pairs(bits: usize, shape: GcdShape) -> Vec<(FlintInt, FlintInt)> {
    pin_flint_to_one_thread();
    mp_gcd_pairs(bits, shape)
        .iter()
        .map(|(a, b)| {
            (
                FlintInt::from_str_radix(&a.to_string_radix(16), 16),
                FlintInt::from_str_radix(&b.to_string_radix(16), 16),
            )
        })
        .collect()
}
