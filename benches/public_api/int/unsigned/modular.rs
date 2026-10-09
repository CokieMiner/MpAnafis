//! Modular operations on identical operands with nonzero odd moduli.
//! Rug composes arithmetic with Euclidean reduction. Montgomery and Barrett
//! include domain setup. The Montgomery reference evaluates a*b*R^-1 modulo m.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, ops::RemRounding};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::{EXTENDED_GCD, MODULAR, MODULAR_EXP},
    support::{
        SAMPLE_COUNT_GCD, SAMPLE_SIZE_GCD, coprime_hex_pairs, mp_uint, odd_hex, paired_bench,
    },
};

#[derive(Clone, Copy)]
enum ExponentShape {
    OneBit,
    TwoBits,
    ShortSparse,
    SparseThree,
    Alternating,
    Dense,
}

// Exponent distributions share the operation, sampling, and independent oracle
// with the random public pow_mod case. The numeric argument is modulus width.
macro_rules! pow_scenarios {
    ($widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr) => {
        pow_scenarios!(@cases $widths, $size, $count, $mp_op, $rug_op, $verify;
            one_bit = OneBit, two_bits = TwoBits, short_sparse = ShortSparse,
            sparse_three = SparseThree, alternating = Alternating, dense = Dense);
    };
    (@cases $widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr;
        $($name:ident = $shape:ident),+ $(,)?) => { $(
        crate::int::support::paired_bench!($name, $widths, samples = ($size, $count),
            mp: |bits| mp_power_inputs(bits, ExponentShape::$shape) => $mp_op,
            rug: |bits| rug_power_inputs(bits, ExponentShape::$shape) => $rug_op,
            verify = $verify,
        );
    )+ };
}

paired_bench!(add_mod, MODULAR,
    mp: mp_inputs => |(a, b, m): &(MpUint, MpUint, MpUint)| a.add_mod(b, m),
    rug: rug_inputs => |(a, b, m): &(Integer, Integer, Integer)| Some((Integer::from(a + b)).rem_euc(m)),
);
paired_bench!(sub_mod, MODULAR,
    mp: mp_inputs => |(a, b, m): &(MpUint, MpUint, MpUint)| a.sub_mod(b, m),
    rug: rug_inputs => |(a, b, m): &(Integer, Integer, Integer)| Some((Integer::from(a - b)).rem_euc(m)),
);
paired_bench!(mul_mod, MODULAR,
    mp: mp_inputs => |(a, b, m): &(MpUint, MpUint, MpUint)| a.mul_mod(b, m),
    rug: rug_inputs => |(a, b, m): &(Integer, Integer, Integer)| Some((Integer::from(a * b)).rem_euc(m)),
);
paired_bench!(pow_mod, MODULAR_EXP, samples = (4, 20),
    mp: mp_inputs => |(a, b, m): &(MpUint, MpUint, MpUint)| a.pow_mod(b, m),
    rug: rug_inputs => |(a, b, m): &(Integer, Integer, Integer)| a.pow_mod_ref(b, m).map(Integer::from),
    scenarios = pow_scenarios,
);
paired_bench!(montgomery_mul, MODULAR,
    mp: mp_inputs => |(a, b, m): &(MpUint, MpUint, MpUint)| a.montgomery_mul(b, m),
    rug: rug_inputs => |(a, b, m): &(Integer, Integer, Integer)| {
        let radix_bits = m.significant_bits().div_ceil(usize::BITS).checked_mul(usize::BITS).expect("radix width fits Rug");
        let inverse = (Integer::from(1) << radix_bits).invert(m).expect("odd modulus is coprime to radix");
        Some((Integer::from(a * b) * inverse).rem_euc(m))
    },
);
paired_bench!(barrett_reduce, MODULAR,
    mp: |bits| mp_inputs(bits).into_iter().map(|(a, _, m)| (a, m)).collect::<Vec<_>>()
        => |(a, m): &(MpUint, MpUint)| a.barrett_reduce(m),
    rug: |bits| rug_inputs(bits).into_iter().map(|(a, _, m)| (a, m)).collect::<Vec<_>>()
        => |(a, m): &(Integer, Integer)| Some(Integer::from(a.rem_euc(m))),
);
paired_bench!(invert, EXTENDED_GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: |bits| {
        coprime_hex_pairs(bits)
            .into_iter()
            .map(|(a, m)| {
                (
                    MpUint::from_str_radix(&a, 16).expect("value parses"),
                    MpUint::from_str_radix(&m, 16).expect("modulus parses"),
                )
            })
            .collect::<Vec<_>>()
    } => |(a, m): &(MpUint, MpUint)| a.invert(m),
    rug: |bits| {
        coprime_hex_pairs(bits)
            .into_iter()
            .map(|(a, m)| {
                (
                    Integer::from_str_radix(&a, 16).expect("value parses"),
                    Integer::from_str_radix(&m, 16).expect("modulus parses"),
                )
            })
            .collect::<Vec<_>>()
    } => |(a, m): &(Integer, Integer)| a.invert_ref(m).map(Integer::from),
    scenarios = crate::int::support::gcd_scenarios,
);

fn mp_inputs(bits: usize) -> Vec<(MpUint, MpUint, MpUint)> {
    vec![(
        mp_uint(bits, 42),
        mp_uint(bits, 1_337),
        MpUint::from_str_radix(&odd_hex(bits, 9_999), 16).expect("odd modulus parses"),
    )]
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_inputs(bits: usize) -> Vec<(Integer, Integer, Integer)> {
    vec![(
        rug_uint(bits, 42),
        rug_uint(bits, 1_337),
        Integer::from_str_radix(&odd_hex(bits, 9_999), 16).expect("odd modulus parses"),
    )]
}

fn mp_power_inputs(bits: usize, shape: ExponentShape) -> Vec<(MpUint, MpUint, MpUint)> {
    let high = MpUint::one() << bits.checked_sub(1).expect("positive exponent width");
    let exponent = match shape {
        ExponentShape::OneBit => high,
        ExponentShape::TwoBits => high + MpUint::one(),
        ExponentShape::ShortSparse => MpUint::from(73_u8),
        ExponentShape::SparseThree => high + (MpUint::one() << bits.div_euclid(2)) + MpUint::one(),
        ExponentShape::Alternating => MpUint::from_str_radix(&"a".repeat(bits.div_euclid(4)), 16)
            .expect("alternating exponent parses"),
        ExponentShape::Dense => (high << 1_usize) - MpUint::one(),
    };
    vec![(
        mp_uint(bits, 42),
        exponent,
        MpUint::from_str_radix(&odd_hex(bits, 9_999), 16).expect("odd modulus parses"),
    )]
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_power_inputs(bits: usize, shape: ExponentShape) -> Vec<(Integer, Integer, Integer)> {
    mp_power_inputs(bits, shape)
        .into_iter()
        .map(|(base, exponent, modulus)| {
            (
                Integer::from_str_radix(&base.to_string_radix(16), 16).expect("base parses"),
                Integer::from_str_radix(&exponent.to_string_radix(16), 16)
                    .expect("exponent parses"),
                Integer::from_str_radix(&modulus.to_string_radix(16), 16).expect("modulus parses"),
            )
        })
        .collect()
}
