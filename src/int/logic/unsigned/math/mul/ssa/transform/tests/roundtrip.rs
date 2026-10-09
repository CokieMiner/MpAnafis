//! Dense and zero-tail FFT round trips across coefficient and matrix widths.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded ring and power-of-two matrix geometries establish complete initialized coefficients and primitive roots"
)]

use alloc::vec;

use proptest::prelude::*;

use crate::parallel::SequentialExecutor;

use super::super::{LIMB_BITS, Limb, SsaRing, SsaTransform};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 24 }))]

    #[test]
    fn forward_inverse_roundtrips_cover_dense_sparse_and_wide_coefficients(
        (bits, len, data) in (
            prop::sample::select(if cfg!(miri) { vec![64_usize, 128] } else { vec![512_usize, 1024, 8192, 16384] }),
            prop::sample::select(if cfg!(miri) { vec![2_usize, 4, 8] } else { vec![2_usize, 4, 8, 16, 256] }),
        ).prop_flat_map(|(bits, len)| (Just(bits), Just(len), prop::collection::vec(any::<Limb>(), len * bits.div_euclid(LIMB_BITS)))),
        zero_tail in any::<bool>(),
    ) {
        check_roundtrip(bits, len, &data, zero_tail);
    }
}

#[cfg_attr(
    miri,
    ignore = "the fixed sweep exercises 256-slot transforms and 16384-bit coefficients; smaller versions of the same round trips run under Miri"
)]
#[test]
fn full_coefficient_carries_cover_scalar_radix_four_and_matrix_entries() {
    for (bits, len) in [
        (512_usize, 2_usize),
        (1024, 4),
        (8192, 8),
        (16384, 8),
        (512, 256),
        (1024, 256),
    ] {
        for fill in [0, 1, Limb::MAX] {
            for zero_tail in [false, true] {
                check_roundtrip(
                    bits,
                    len,
                    &vec![fill; len * bits.div_euclid(LIMB_BITS)],
                    zero_tail,
                );
            }
        }
    }
}

fn check_roundtrip(bits: usize, len: usize, data: &[Limb], zero_tail: bool) {
    let ml = bits.div_euclid(LIMB_BITS);
    let cl = ml + 1;
    let active = if zero_tail { len >> 1 } else { len };
    let mut matrix = vec![0; len * cl];
    for (slot, input) in matrix
        .chunks_exact_mut(cl)
        .zip(data.chunks_exact(ml))
        .take(active)
    {
        slot[..ml].copy_from_slice(input);
    }
    let expected = matrix.clone();
    let mut scratch = vec![Limb::MAX; cl];
    let root = bits
        .checked_mul(2)
        .expect("test period fits")
        .div_euclid(len);
    // SAFETY: every complete initialized coefficient has a zero guard;
    // len divides 2*bits, the declared zero tail is physically zero, and
    // the scratch coefficient is complete and disjoint from the matrix.
    unsafe {
        SsaTransform::fft_in_place_with_executor(
            &mut matrix,
            len,
            root,
            bits,
            false,
            active,
            &SequentialExecutor,
            &mut scratch,
        );
        SsaTransform::fft_in_place_with_executor(
            &mut matrix,
            len,
            root,
            bits,
            true,
            len,
            &SequentialExecutor,
            &mut scratch,
        );
    }
    let logarithm = usize::try_from(len.trailing_zeros()).expect("test logarithm fits");
    let inverse_scale = 2 * bits - logarithm;
    for slot in matrix.chunks_exact_mut(cl) {
        // SAFETY: the inverse preserves complete semi-normalized coefficients;
        // normalization precedes the reduced inverse-scale exponent.
        unsafe {
            let _ = SsaRing::normalize(slot, bits);
            SsaRing::shift_in_place(slot, inverse_scale, bits, &mut scratch);
        }
    }
    assert_eq!(matrix, expected);
}
