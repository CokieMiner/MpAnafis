//! Montgomery-step recurrences, carry boundaries, and span preservation.

#![expect(
    unsafe_code,
    reason = "Tests provide initialized spans, a Montgomery inverse, and detected CPU prerequisites"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::{super::ArchKernels, Limb, MontyKernel};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn steps_match_reference_and_preserve_spans(
        words in collection::vec((any::<Limb>(), any::<Limb>(), any::<Limb>()), 0..=80),
        input_limb in any::<Limb>(),
    ) {
        let output = words.iter().map(|row| row.0).collect::<Vec<_>>();
        let multiplier = words.iter().map(|row| row.1).collect::<Vec<_>>();
        let modulus = words.iter().map(|row| row.2).collect::<Vec<_>>();
        check(&output, &multiplier, modulus, input_limb);
    }
}

#[test]
fn carries_cross_scalar_block_and_tail_boundaries() {
    for len in [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    ] {
        if cfg!(miri) && len > 5 {
            continue;
        }
        for words in row_patterns(len) {
            for scalar in [0, 1, Limb::MAX] {
                check(&words, &vec![Limb::MAX; len], words.clone(), scalar);
            }
        }
    }
}

fn check(output: &[Limb], multiplier: &[Limb], mut modulus: Vec<Limb>, scalar: Limb) {
    let len = output.len();
    let kernels: &[Option<MontyKernel>] = &[
        Some(ArchKernels::selected_monty_redc_step_unchecked()),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        Some(super::fallback::monty_redc_step_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        std::arch::is_x86_feature_detected!("bmi2")
            .then_some::<MontyKernel>(super::x86_64_bmi2::monty_redc_step_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        (std::arch::is_x86_feature_detected!("adx") && std::arch::is_x86_feature_detected!("bmi2"))
            .then_some::<MontyKernel>(super::x86_64_adx::monty_redc_step_unchecked),
    ];
    if len == 0 {
        for kernel in kernels.iter().flatten() {
            assert_eq!(
                // SAFETY: empty steps access none of the pointers; the table
                // establishes each backend's CPU prerequisites.
                unsafe {
                    kernel(
                        core::ptr::null_mut(),
                        core::ptr::null(),
                        core::ptr::null(),
                        0,
                        scalar,
                        0,
                    )
                },
                0
            );
        }
        return;
    }
    *modulus.first_mut().expect("nonempty modulus") |= 1;
    let low = *modulus.first().expect("nonempty modulus");
    let mut inverse = low;
    // An odd low word satisfies low^2 = 1 mod 8. Each Newton iteration
    // doubles the correct bits, so five iterations cover every limb width.
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(2_usize.wrapping_sub(low.wrapping_mul(inverse)));
    }
    inverse = inverse.wrapping_neg();
    let original_modulus = modulus.clone();
    let original_multiplier = multiplier.to_vec();
    for shared_input in [false, true] {
        let source = if shared_input { &modulus } else { multiplier };
        let mut expected = output.to_vec();
        let expected_carry =
            reference_montgomery_step(&mut expected, source, &modulus, scalar, inverse);
        for kernel in kernels.iter().flatten() {
            let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
            actual
                .get_mut(1..=len)
                .expect("output window")
                .copy_from_slice(output);
            // SAFETY: the output window contains len initialized writable
            // limbs and is disjoint from both immutable inputs. The inputs
            // may be identical. The odd modulus and inverse cancel limb zero;
            // the table establishes CPU prerequisites.
            let carry = unsafe {
                kernel(
                    actual.as_mut_ptr().add(1),
                    source.as_ptr(),
                    modulus.as_ptr(),
                    len,
                    scalar,
                    inverse,
                )
            };
            assert_eq!(carry, expected_carry);
            assert_eq!(actual.get(1..=len).expect("result window"), expected);
            assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
        }
    }
    assert_eq!(modulus, original_modulus);
    assert_eq!(multiplier, original_multiplier);
}

fn reference_montgomery_step(
    out: &mut [Limb],
    multiplier: &[Limb],
    modulus: &[Limb],
    input_limb: Limb,
    inverse: Limb,
) -> Limb {
    let input_carry = Oracle::add_mul(out, multiplier, input_limb);
    let quotient_limb = out.first().expect("nonempty output").wrapping_mul(inverse);
    let modulus_carry = Oracle::add_mul(out, modulus, quotient_limb);
    assert_eq!(out.first(), Some(&0), "the inverse cancels the low limb");
    out.rotate_left(1);
    let (high_limb, carry_out) = input_carry.overflowing_add(modulus_carry);
    *out.last_mut().expect("nonempty output") = high_limb;
    Limb::from(carry_out)
}
