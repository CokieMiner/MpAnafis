#![expect(
    unsafe_code,
    reason = "Tests allocate complete disjoint buffers and invoke only detected ISA backends"
)]

extern crate std;

use alloc::{vec, vec::Vec};

use crate::PRIME_CATALOG;

use super::{NttFloatKernels, Scalar};

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Catalog boundary residues are exact integers below 2^50 in f64 and u64"
)]
fn admitted_backends_match_exact_integer_products_and_scalar_spans() {
    let mut backends = vec![NttFloatKernels {
        dif: Scalar::radix4::<false>,
        dit: Scalar::radix4::<true>,
        multiply: Scalar::pointwise::<false>,
        square: Scalar::pointwise::<true>,
        scale: Scalar::scale,
    }];
    backends.push(NttFloatKernels::get());
    #[cfg(all(
        not(miri),
        target_arch = "x86_64",
        any(
            feature = "std",
            all(
                not(target_feature = "avx512f"),
                target_feature = "avx2",
                target_feature = "fma"
            )
        )
    ))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        use super::Avx2;
        backends.push(NttFloatKernels {
            dif: Avx2::radix4::<false>,
            dit: Avx2::radix4::<true>,
            multiply: Avx2::pointwise::<false>,
            square: Avx2::pointwise::<true>,
            scale: Avx2::scale,
        });
    }
    for prime in PRIME_CATALOG {
        let modulus = prime.modulus();
        let inputs = [
            0,
            1,
            2,
            modulus.div_euclid(2),
            modulus.checked_sub(1).expect("positive prime"),
        ];
        let left: Vec<f64> = inputs
            .iter()
            .flat_map(|&a| inputs.iter().map(move |_| a as f64))
            .collect();
        let right: Vec<f64> = inputs
            .iter()
            .flat_map(|_| inputs.iter().map(|&b| b as f64))
            .collect();
        for backend in &backends {
            let mut values = left.clone();
            // SAFETY: arrays have equal initialized lengths and are disjoint.
            // Each backend was admitted by runtime detection or is scalar.
            unsafe {
                (backend.multiply)(
                    values.as_mut_ptr(),
                    right.as_ptr(),
                    values.len(),
                    prime.float(),
                    prime.reciprocal(),
                );
            }
            for ((&actual, &a), &b) in values.iter().zip(&left).zip(&right) {
                let expected = u128::from(a as u64)
                    .checked_mul(u128::from(b as u64))
                    .expect("100-bit product")
                    % u128::from(modulus);
                let centered = Scalar::reduce(actual, prime.float(), prime.reciprocal());
                let canonical = if centered < 0.0 {
                    centered + prime.float()
                } else {
                    centered
                };
                assert_eq!(
                    canonical,
                    u64::try_from(expected).expect("canonical residue") as f64
                );
            }
        }
    }
}
