//! Direct product runners compared with independent public residues.

#[cfg(not(target_pointer_width = "16"))]
use crate::tune_api::{CyclicProductAlgorithm, CyclicProductRunner};
use crate::{MpUint, tune_api::MontgomeryProductRunner};

use super::strategies::integer;

#[test]
fn prepared_montgomery_products_preserve_radix_congruence_across_reuse() {
    for width in [1_usize, 2, 3, 4, 16, 31, 32, 33, 63, 64, 65, 128, 129] {
        if cfg!(miri) && width > 33 {
            continue;
        }
        let modulus = vec![usize::MAX; width];
        let mut left = modulus.clone();
        *left.last_mut().expect("positive width") >>= 1;
        let right = vec![usize::MAX - 2; width];
        let mut runner = MontgomeryProductRunner::new(&left, &right, &modulus);
        let modulus_value = integer(&modulus);
        let expected = (integer(&left) * integer(&right)) % &modulus_value;
        let bits = width
            .checked_mul(usize::try_from(usize::BITS).expect("native width fits"))
            .expect("bounded radix");
        for _ in 0..3 {
            let result = integer(runner.run());
            assert!(result < modulus_value);
            assert_eq!((result << bits) % &modulus_value, expected);
        }
    }
}

#[test]
#[cfg(not(target_pointer_width = "16"))]
fn cyclic_strategies_compute_the_same_modulus_for_balanced_and_uneven_inputs() {
    for width in [
        1_usize, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 512, 513, 3_072,
        3_073, 3_074,
    ] {
        if cfg!(miri) && width > 2 {
            continue;
        }
        for right_width in [width, width.div_ceil(2)] {
            let left = vec![usize::MAX; width];
            let right = vec![usize::MAX - 2; right_width];
            let product = integer(&left) * integer(&right);
            let mut runner = CyclicProductRunner::new(&left, &right, width);
            for _ in 0..2 {
                for algorithm in [
                    CyclicProductAlgorithm::Full,
                    CyclicProductAlgorithm::Cyclic,
                    CyclicProductAlgorithm::Production,
                ] {
                    let result = runner.run(algorithm);
                    let modulus = integer(&vec![usize::MAX; result.len()]);
                    assert_eq!(integer(result) % &modulus, &product % &modulus);
                    assert!(modulus > MpUint::from(0_u8));
                }
            }
        }
    }
}
