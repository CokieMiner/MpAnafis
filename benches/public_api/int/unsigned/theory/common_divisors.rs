//! Unsigned common-divisor functions, compared with Rug/GMP.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, ops::RemRounding};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::{EXTENDED_GCD, GCD},
    support::{SAMPLE_COUNT_GCD, SAMPLE_SIZE_GCD, mp_uint_pairs, paired_bench},
};

paired_bench!(gcd, GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.gcd(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.gcd_ref(b)),
    scenarios = crate::int::support::gcd_scenarios,
);
paired_bench!(lcm, GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.lcm(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.lcm_ref(b))),
    scenarios = crate::int::support::gcd_scenarios,
);
paired_bench!(gcd_lcm, GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.gcd_lcm(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| {
        let gcd = Integer::from(a.gcd_ref(b));
        let lcm = Integer::from(a / &gcd) * b;
        Some((gcd, lcm))
    },
    scenarios = crate::int::support::gcd_scenarios,
);
paired_bench!(extended_gcd, EXTENDED_GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.extended_gcd(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| {
        let (gcd, x, y) = a.clone().extended_gcd(b.clone(), Integer::new());
        Some((gcd, x.rem_euc(b), y.rem_euc(a)))
    },
    verify = crate::int::support::verify_extended_gcd_pairs,
    scenarios = crate::int::support::gcd_scenarios,
);
paired_bench!(is_coprime, GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.is_coprime(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.gcd_ref(b)) == 1,
    scenarios = crate::int::support::gcd_scenarios,
);
