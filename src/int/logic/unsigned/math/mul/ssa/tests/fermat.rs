//! Scalar Fermat products and explicit versus implicit zero guards.

#![expect(
    unsafe_code,
    reason = "Fixtures establish exact guarded or proved short output spans and complete disjoint transform workspaces"
)]

use super::*;

#[test]
fn single_limb_fermat_product_uses_scalar_basecase() {
    let mut actual = [Limb::MAX; 2];
    let mut scratch = vec![0; SsaPointwise::fermat_basecase_scratch_len(LIMB_BITS)];
    // SAFETY: complete canonical ordinary coefficients have two initialized
    // limbs; their disjoint scratch retains the exact scalar product layout.
    unsafe {
        SsaPointwise::fermat_basecase_mul_into(
            &mut actual,
            &[Limb::MAX, 0],
            &[Limb::MAX.wrapping_sub(1), 0],
            LIMB_BITS,
            &mut scratch,
        );
    }
    // B-1=-2 and B-2=-3 modulo B+1.
    assert_eq!(actual, [6, 0]);
}

#[test]
fn implicit_zero_guards_and_proved_short_outputs_match_guarded_operands() {
    let bits = 512;
    let cl = SsaRing::coeff_limbs(bits).get();
    let plan = FftPlan::new(bits);
    let mut guarded_left = vec![0; cl];
    let mut guarded_right = vec![0; cl];
    *guarded_left.first_mut().expect("coefficient") = 3;
    *guarded_right.first_mut().expect("coefficient") = 5;
    let mut guarded = vec![Limb::MAX; cl];
    let mut implicit = vec![Limb::MAX; cl];
    let mut exact = [Limb::MAX];
    let mut scratch = vec![Limb::MAX; plan.transform_mul_scratch(1)];
    // SAFETY: canonical guarded operands and ordinary short operands 3,5 have
    // exact significant widths two and three. Their five-bit product fits the
    // one-limb output; other outputs hold complete guarded coefficients.
    // All buffers are disjoint and scratch has the retained plan's exact size.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut guarded,
            &guarded_left,
            &guarded_right,
            bits,
            None,
            true,
            Some(&plan),
            &SequentialExecutor,
            &mut scratch,
        );
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut implicit,
            &[3],
            &[5],
            bits,
            Some((2, 3)),
            true,
            Some(&plan),
            &SequentialExecutor,
            &mut scratch,
        );
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut exact,
            &[3],
            &[5],
            bits,
            Some((2, 3)),
            true,
            Some(&plan),
            &SequentialExecutor,
            &mut scratch,
        );
    }
    assert_eq!(implicit, guarded);
    assert_eq!(guarded.first(), Some(&15));
    assert_eq!(exact, [15]);
}
