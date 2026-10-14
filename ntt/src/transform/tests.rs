use alloc::vec;

use crate::{
    PRIME_CATALOG, SequentialExecutor,
    arch::{NttFloatKernels, Scalar},
};

use super::{Transform, Twiddles};

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Canonical test residues and inverse scales are below 2^50 and exactly representable as f64"
)]
fn forward_and_inverse_preserve_prime_fields_across_cache_boundaries_and_zero_prefixes() {
    let kernels = NttFloatKernels::get();
    for prime in PRIME_CATALOG {
        for len in [4_usize, 8, 16, 64, 128, 256, 512, 1024] {
            let mut forward = vec![0.0; len];
            let mut inverse = vec![0.0; len];
            Twiddles::generate(&mut forward, &mut inverse, prime);
            for active in [
                1,
                len.div_euclid(4),
                len.div_euclid(2),
                len.checked_sub(1).expect("positive length"),
                len,
            ] {
                let mut values = vec![0.0; len];
                for (index, value) in values.iter_mut().take(active).enumerate() {
                    *value = (prime
                        .modulus()
                        .checked_sub(1)
                        .expect("positive prime")
                        .checked_sub(u64::try_from(index).expect("small index"))
                        .expect("test index is below prime")) as f64;
                }
                let original = values.clone();
                Transform::forward(
                    &mut values,
                    &forward,
                    active,
                    prime,
                    &kernels,
                    1,
                    &SequentialExecutor,
                );
                Transform::inverse(
                    &mut values,
                    &inverse,
                    prime,
                    &kernels,
                    1,
                    &SequentialExecutor,
                );
                let factor =
                    prime.inv_mod(u64::try_from(len).expect("small transform length")) as f64;
                for (actual, expected) in values.iter().zip(&original) {
                    let scaled = Scalar::mulmod(*actual, factor, prime.float(), prime.reciprocal());
                    let centered = Scalar::reduce(scaled, prime.float(), prime.reciprocal());
                    let canonical = if centered < 0.0 {
                        centered + prime.float()
                    } else {
                        centered
                    };
                    assert_eq!(
                        canonical,
                        *expected,
                        "len={len}, active={active}, prime={}",
                        prime.modulus()
                    );
                }
            }
        }
    }
}
