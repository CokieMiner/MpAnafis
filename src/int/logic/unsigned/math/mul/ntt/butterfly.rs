//! Multi-prime NTT butterfly stages with architecture vector acceleration.

use super::{ArchKernels, Modulus, Ntt};

impl Ntt {
    /// Executes one forward DIF butterfly stage across `values`.
    #[cfg(test)]
    pub fn forward_dif_stage(
        values: &mut [u32],
        block_len: usize,
        block_root: u32,
        modulus: Modulus,
        twiddle_buf: &mut [u32],
    ) {
        let half_len = block_len >> 1;
        Self::generate_stage_twiddles(twiddle_buf, half_len, block_root, modulus);
        // SAFETY: generation initialized the first half_len twiddle entries.
        let stage_twiddles = unsafe { twiddle_buf.get_unchecked(..half_len) };
        Self::forward_dif_stage_with_twiddles(values, block_len, modulus, stage_twiddles);
    }

    /// Generates one stage's twiddles exactly once for all disjoint blocks.
    pub fn generate_stage_twiddles(
        twiddle_buf: &mut [u32],
        half_len: usize,
        block_root: u32,
        modulus: Modulus,
    ) {
        debug_assert!(
            twiddle_buf.len() >= half_len,
            "twiddle buffer must cover the requested NTT stage"
        );
        let mut current_twiddle = Self::to_montgomery(1, modulus);
        for twiddle in twiddle_buf.iter_mut().take(half_len) {
            *twiddle = current_twiddle;
            current_twiddle = Self::montgomery_mul(current_twiddle, block_root, modulus);
        }
    }

    /// Applies a precomputed DIF stage to cache-contiguous block ranges.
    pub fn forward_dif_stage_with_twiddles(
        values: &mut [u32],
        block_len: usize,
        modulus: Modulus,
        stage_twiddles: &[u32],
    ) {
        let half_len = block_len >> 1;
        debug_assert!(
            stage_twiddles.len() >= half_len,
            "stage twiddles must cover every butterfly"
        );
        for block in values.chunks_exact_mut(block_len) {
            let (low, high) = block.split_at_mut(half_len);
            // SAFETY: low and high have length half_len matching stage_twiddles.
            unsafe {
                ArchKernels::ntt_dif_butterfly_unchecked(
                    low.as_mut_ptr(),
                    high.as_mut_ptr(),
                    stage_twiddles.as_ptr(),
                    half_len,
                    modulus.prime,
                    modulus.neg_inverse,
                );
            }
        }
    }

    /// Applies two consecutive forward DIF stages as one radix-4 pass.
    pub fn forward_dif_radix4_stage_with_twiddles(
        values: &mut [u32],
        block_len: usize,
        modulus: Modulus,
        stage_twiddles: &[u32],
    ) {
        let quarter_len = block_len >> 2;
        debug_assert!(
            block_len.is_multiple_of(4) && stage_twiddles.len() >= quarter_len.saturating_mul(2),
            "radix-4 stage requires two complete twiddle quarters"
        );
        for block in values.chunks_exact_mut(block_len) {
            // SAFETY: the block contains four disjoint quarter spans and the
            // stage twiddles contain both quarter twiddle spans.
            unsafe {
                ArchKernels::ntt_radix4_dif_unchecked(
                    block.as_mut_ptr(),
                    stage_twiddles.as_ptr(),
                    quarter_len,
                    modulus.prime,
                    modulus.neg_inverse,
                );
            }
        }
    }

    /// Executes one inverse DIT butterfly stage across `values`.
    #[cfg(test)]
    pub fn inverse_dit_stage(
        values: &mut [u32],
        block_len: usize,
        block_root: u32,
        modulus: Modulus,
        twiddle_buf: &mut [u32],
    ) {
        let half_len = block_len >> 1;
        Self::generate_stage_twiddles(twiddle_buf, half_len, block_root, modulus);
        // SAFETY: generation initialized the first half_len twiddle entries.
        let stage_twiddles = unsafe { twiddle_buf.get_unchecked(..half_len) };
        Self::inverse_dit_stage_with_twiddles(values, block_len, modulus, stage_twiddles);
    }

    /// Applies a precomputed inverse DIT stage to cache-contiguous blocks.
    pub fn inverse_dit_stage_with_twiddles(
        values: &mut [u32],
        block_len: usize,
        modulus: Modulus,
        stage_twiddles: &[u32],
    ) {
        let half_len = block_len >> 1;
        debug_assert!(
            stage_twiddles.len() >= half_len,
            "stage twiddles must cover every butterfly"
        );
        for block in values.chunks_exact_mut(block_len) {
            let (low, high) = block.split_at_mut(half_len);
            // SAFETY: low and high have length half_len matching stage_twiddles.
            unsafe {
                ArchKernels::ntt_dit_butterfly_unchecked(
                    low.as_mut_ptr(),
                    high.as_mut_ptr(),
                    stage_twiddles.as_ptr(),
                    half_len,
                    modulus.prime,
                    modulus.neg_inverse,
                );
            }
        }
    }

    /// Applies two consecutive inverse DIT stages as one radix-4 pass.
    pub fn inverse_dit_radix4_stage_with_twiddles(
        values: &mut [u32],
        block_len: usize,
        modulus: Modulus,
        stage_twiddles: &[u32],
    ) {
        let quarter_len = block_len >> 2;
        debug_assert!(
            block_len.is_multiple_of(4) && stage_twiddles.len() >= quarter_len.saturating_mul(2),
            "radix-4 stage requires two complete twiddle quarters"
        );
        for block in values.chunks_exact_mut(block_len) {
            // SAFETY: the block contains four disjoint quarter spans and the
            // stage twiddles contain both quarter twiddle spans.
            unsafe {
                ArchKernels::ntt_radix4_dit_unchecked(
                    block.as_mut_ptr(),
                    stage_twiddles.as_ptr(),
                    quarter_len,
                    modulus.prime,
                    modulus.neg_inverse,
                );
            }
        }
    }

    /// Pointwise in-place Montgomery multiplication across two coefficient slices.
    #[cfg(test)]
    pub fn pointwise_monty_mul(dst: &mut [u32], src: &[u32], modulus: Modulus) {
        let len = dst.len().min(src.len());
        let (dst_prefix, _) = dst.split_at_mut(len);
        let (src_prefix, _) = src.split_at(len);
        Self::pointwise_monty_mul_canonical(dst_prefix, src_prefix, modulus);
    }

    /// Applies Montgomery multiplication to equal-length lazy residue slices.
    ///
    /// The selected architecture canonicalizes each `[0, 2p)` operand while
    /// loading it, so no separate `% prime` normalization pass is required.
    pub fn pointwise_monty_mul_canonical(dst: &mut [u32], src: &[u32], modulus: Modulus) {
        // SAFETY: dst and src have equal lengths and contain lazy residues
        // within the architecture kernel's validated `[0, 2p)` contract.
        unsafe {
            ArchKernels::ntt_monty_mul_slice_unchecked(
                dst.as_mut_ptr(),
                dst.as_ptr(),
                src.as_ptr(),
                dst.len(),
                modulus.prime,
                modulus.neg_inverse,
            );
        }
    }

    /// Pointwise in-place Montgomery squaring across coefficient slice.
    #[cfg(test)]
    pub fn pointwise_monty_sqr(dst: &mut [u32], modulus: Modulus) {
        Self::pointwise_monty_sqr_canonical(dst, modulus);
    }

    /// Applies Montgomery squaring to a lazy residue slice.
    ///
    /// The selected architecture canonicalizes each `[0, 2p)` operand while
    /// loading it, so no separate `% prime` normalization pass is required.
    pub fn pointwise_monty_sqr_canonical(dst: &mut [u32], modulus: Modulus) {
        // SAFETY: dst contains lazy residues within the architecture kernel's
        // validated `[0, 2p)` contract and aliases both inputs intentionally.
        unsafe {
            ArchKernels::ntt_monty_mul_slice_unchecked(
                dst.as_mut_ptr(),
                dst.as_ptr(),
                dst.as_ptr(),
                dst.len(),
                modulus.prime,
                modulus.neg_inverse,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::arithmetic_side_effects,
        reason = "Reference arithmetic is bounded by the fixed NTT test modulus"
    )]
    #![allow(
        clippy::as_conversions,
        reason = "Reference Montgomery helpers intentionally narrow fixed-width residues"
    )]
    #![allow(
        clippy::cast_possible_truncation,
        reason = "Reference residues are bounded below the u32 modulus range"
    )]
    #![allow(
        clippy::indexing_slicing,
        reason = "Fixed-size test arrays are indexed only by loops with proven bounds"
    )]
    #![allow(
        clippy::integer_division,
        reason = "Radix-4 reference geometry uses exact powers-of-two test lengths"
    )]
    #![allow(
        clippy::unwrap_used,
        reason = "Fixed-size radix-4 test geometry proves every conversion succeeds"
    )]
    #![allow(
        clippy::shadow_unrelated,
        reason = "Expected and actual buffers are intentionally rebound per transform direction"
    )]
    #![allow(
        clippy::too_many_lines,
        reason = "The fused-kernel equivalence test keeps both DIF and DIT proofs together"
    )]

    use alloc::vec;

    use super::{ArchKernels, Modulus, Ntt};
    use crate::int::logic::unsigned::math::mul::ntt::plan::MODULI;

    fn lazy_mod_2p(value: u64, modulus: Modulus) -> u32 {
        (value % (2 * u64::from(modulus.prime))) as u32
    }

    fn lazy_redc(a: u32, b: u32, modulus: Modulus) -> u32 {
        let product = u64::from(a) * u64::from(b);
        let factor = (product as u32).wrapping_mul(modulus.neg_inverse);
        product
            .wrapping_add(u64::from(factor) * u64::from(modulus.prime))
            .wrapping_shr(32) as u32
    }

    #[test]
    fn lazy_butterflies_widen_33_bit_adds_and_borrows() {
        let modulus = MODULI[0];
        let two_p = modulus.prime * 2;
        let cases = [(two_p - 1, two_p - 2), (two_p - 3, 7), (13, two_p - 5)];
        let twiddle = Ntt::to_montgomery(1, modulus);

        for (u, v) in cases {
            let mut dif = [u, v];
            let mut dif_twiddles = [0];
            Ntt::forward_dif_stage(&mut dif, 2, twiddle, modulus, &mut dif_twiddles);
            let expected_sum = lazy_mod_2p(u64::from(u) + u64::from(v), modulus);
            let expected_diff =
                lazy_mod_2p(u64::from(u) + u64::from(two_p) - u64::from(v), modulus);
            assert!(dif[0] < two_p && dif[1] < two_p);
            assert_eq!(dif[0] % modulus.prime, expected_sum % modulus.prime);
            assert_eq!(dif[1] % modulus.prime, expected_diff % modulus.prime);

            let mut dit = [u, v];
            let mut dit_stage_twiddles = [0];
            Ntt::inverse_dit_stage(&mut dit, 2, twiddle, modulus, &mut dit_stage_twiddles);
            assert!(dit[0] < two_p && dit[1] < two_p);
            assert_eq!(dit[0] % modulus.prime, expected_sum % modulus.prime);
            assert_eq!(dit[1] % modulus.prime, expected_diff % modulus.prime);
        }
    }

    #[test]
    fn pointwise_paths_normalize_lazy_operands() {
        let modulus = MODULI[0];
        let p = modulus.prime;
        for len in [1, 2, 4, 8, 9, 16] {
            let mut product = vec![0; len];
            let mut source = vec![0; len];
            for (index, (left, right)) in product.iter_mut().zip(&mut source).enumerate() {
                *left = p + 1 + u32::try_from(index).unwrap();
                *right = p + 2 + u32::try_from(index * 3).unwrap();
            }
            let expected = product
                .iter()
                .zip(&source)
                .map(|(&left, &right)| Ntt::montgomery_mul(left - p, right - p, modulus))
                .collect::<alloc::vec::Vec<_>>();
            Ntt::pointwise_monty_mul(&mut product, &source, modulus);
            assert_eq!(product, expected, "product len={len}");

            let mut square = (0..len)
                .map(|index| p + 1 + u32::try_from(index).unwrap())
                .collect::<alloc::vec::Vec<_>>();
            let expected = square
                .iter()
                .map(|&value| Ntt::montgomery_mul(value - p, value - p, modulus))
                .collect::<alloc::vec::Vec<_>>();
            Ntt::pointwise_monty_sqr(&mut square, modulus);
            assert_eq!(square, expected, "square len={len}");
        }
    }

    #[test]
    fn lazy_dif_vector_lanes_handle_carry_and_borrow() {
        let modulus = MODULI[0];
        let two_p = modulus.prime * 2;
        let root = Ntt::to_montgomery(3, modulus);
        let mut values = [0_u32; 16];
        for i in 0..8 {
            values[i] = if i % 2 == 0 { two_p - 1 } else { two_p - 2 };
            values[8 + i] = if i % 2 == 0 { two_p - 2 } else { two_p - 1 };
        }
        let mut twiddles = [0_u32; 8];
        let mut generated_twiddles = [0_u32; 8];
        Ntt::forward_dif_stage(&mut values, 16, root, modulus, &mut twiddles);

        generated_twiddles[0] = Ntt::to_montgomery(1, modulus);
        for i in 1..8 {
            generated_twiddles[i] = Ntt::montgomery_mul(generated_twiddles[i - 1], root, modulus);
        }
        for i in 0..8 {
            let u = if i % 2 == 0 { two_p - 1 } else { two_p - 2 };
            let v = if i % 2 == 0 { two_p - 2 } else { two_p - 1 };
            let expected_sum = lazy_mod_2p(u64::from(u) + u64::from(v), modulus);
            let diff = lazy_mod_2p(u64::from(u) + u64::from(two_p) - u64::from(v), modulus);
            let expected_product = lazy_redc(diff, generated_twiddles[i], modulus);
            assert!(values[i] < two_p && values[8 + i] < two_p);
            assert_eq!(values[i] % modulus.prime, expected_sum % modulus.prime);
            assert_eq!(
                values[8 + i] % modulus.prime,
                expected_product % modulus.prime
            );
        }

        let mut dit_values = [0_u32; 16];
        for i in 0..8 {
            dit_values[i] = if i % 2 == 0 { two_p - 1 } else { two_p - 2 };
            dit_values[8 + i] = if i % 2 == 0 { two_p - 2 } else { two_p - 1 };
        }
        let mut dit_twiddles = [0_u32; 8];
        Ntt::inverse_dit_stage(&mut dit_values, 16, root, modulus, &mut dit_twiddles);
        for i in 0..8 {
            let u = if i % 2 == 0 { two_p - 1 } else { two_p - 2 };
            let v = if i % 2 == 0 { two_p - 2 } else { two_p - 1 };
            let product = lazy_redc(v, generated_twiddles[i], modulus);
            let expected_sum = lazy_mod_2p(u64::from(u) + u64::from(product), modulus);
            let expected_diff = lazy_mod_2p(
                u64::from(u) + u64::from(two_p) - u64::from(product),
                modulus,
            );
            assert!(dit_values[i] < two_p && dit_values[8 + i] < two_p);
            assert_eq!(dit_values[i] % modulus.prime, expected_sum % modulus.prime);
            assert_eq!(
                dit_values[8 + i] % modulus.prime,
                expected_diff % modulus.prime
            );
        }
    }

    #[test]
    fn radix4_kernels_match_two_radix2_stages() {
        for modulus in MODULI {
            for quarter_len in [1, 2, 4, 8] {
                let block_len = quarter_len * 4;
                let root = Ntt::to_montgomery(modulus.primitive_root, modulus);
                let root4 = Ntt::montgomery_pow(
                    root,
                    (modulus.prime - 1) / u32::try_from(block_len).unwrap(),
                    modulus,
                );
                let root2 = Ntt::montgomery_pow(
                    root,
                    (modulus.prime - 1) / u32::try_from(block_len / 2).unwrap(),
                    modulus,
                );
                let mut twiddles = vec![0; block_len / 2];
                let mut small_twiddles = vec![0; quarter_len];
                Ntt::generate_stage_twiddles(&mut twiddles, block_len / 2, root4, modulus);
                Ntt::generate_stage_twiddles(&mut small_twiddles, quarter_len, root2, modulus);
                let mut input = vec![0_u32; block_len];
                for (index, value) in input.iter_mut().enumerate() {
                    *value = Ntt::to_montgomery(
                        u32::try_from(index * 37 + 11).unwrap() % modulus.prime,
                        modulus,
                    );
                }
                let mut expected = input.clone();
                let mut actual = input;
                // SAFETY: each pointer pair covers disjoint spans of the
                // complete block and its corresponding twiddle span.
                unsafe {
                    ArchKernels::ntt_dif_butterfly_unchecked(
                        expected.as_mut_ptr(),
                        expected.as_mut_ptr().add(block_len / 2),
                        twiddles.as_ptr(),
                        block_len / 2,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                    ArchKernels::ntt_dif_butterfly_unchecked(
                        expected.as_mut_ptr(),
                        expected.as_mut_ptr().add(quarter_len),
                        small_twiddles.as_ptr(),
                        quarter_len,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                    ArchKernels::ntt_dif_butterfly_unchecked(
                        expected.as_mut_ptr().add(block_len / 2),
                        expected.as_mut_ptr().add(block_len / 2 + quarter_len),
                        small_twiddles.as_ptr(),
                        quarter_len,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                    ArchKernels::ntt_radix4_dif_unchecked(
                        actual.as_mut_ptr(),
                        twiddles.as_ptr(),
                        quarter_len,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                }
                let two_prime = modulus.prime * 2;
                assert!(actual.iter().all(|&value| value < two_prime));
                assert!(expected.iter().all(|&value| value < two_prime));
                for (actual_value, expected_value) in actual.iter().zip(&expected) {
                    assert_eq!(
                        actual_value % modulus.prime,
                        expected_value % modulus.prime,
                        "DIF p={modulus:?}, q={quarter_len}"
                    );
                }

                let mut expected = actual.clone();
                let mut actual = expected.clone();
                // SAFETY: the two small DIT stages and the fused stage use
                // the same disjoint spans and validated twiddle lengths.
                unsafe {
                    ArchKernels::ntt_dit_butterfly_unchecked(
                        expected.as_mut_ptr(),
                        expected.as_mut_ptr().add(quarter_len),
                        small_twiddles.as_ptr(),
                        quarter_len,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                    ArchKernels::ntt_dit_butterfly_unchecked(
                        expected.as_mut_ptr().add(block_len / 2),
                        expected.as_mut_ptr().add(block_len / 2 + quarter_len),
                        small_twiddles.as_ptr(),
                        quarter_len,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                    ArchKernels::ntt_dit_butterfly_unchecked(
                        expected.as_mut_ptr(),
                        expected.as_mut_ptr().add(block_len / 2),
                        twiddles.as_ptr(),
                        block_len / 2,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                    ArchKernels::ntt_radix4_dit_unchecked(
                        actual.as_mut_ptr(),
                        twiddles.as_ptr(),
                        quarter_len,
                        modulus.prime,
                        modulus.neg_inverse,
                    );
                }
                assert!(actual.iter().all(|&value| value < two_prime));
                assert!(expected.iter().all(|&value| value < two_prime));
                for (actual_value, expected_value) in actual.iter().zip(&expected) {
                    assert_eq!(
                        actual_value % modulus.prime,
                        expected_value % modulus.prime,
                        "DIT p={modulus:?}, q={quarter_len}"
                    );
                }
            }
        }
    }
}
