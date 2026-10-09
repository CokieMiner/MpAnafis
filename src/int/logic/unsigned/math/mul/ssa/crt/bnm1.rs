//! Prepared products and squares modulo `B^n - 1`, with CRT reconstruction.
//!
//! Each recursive level splits `B^(2h) - 1` into `B^h + 1` and `B^h - 1`.
//!
//! # References
//!
//! - Gaudry, P., Kruppa, A., & Zimmermann, P. (2007). A GMP-based Implementation
//!   of Schönhage-Strassen's Large Integer Multiplication Algorithm.
//!   *Proceedings of ISSAC '07*, ACM, 167–174. <https://doi.org/10.1145/1277548.1277572>

#![expect(
    unsafe_code,
    reason = "Prepared halving trees and sized disjoint arenas bound complete residue writes, folding, and exact CRT reconstruction"
)]

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    ArchKernels, CrtMulLevel, CrtSquareLevel, Limb, LimbOutput, Multiplication, SsaCarry,
    SsaTransform,
};

/// Namespace for the `B^n - 1` half of the CRT split and its reconstructions.
///
/// The top-level entry points pair one `B^n + 1` transform with one `B^n - 1`
/// product and merge the two residues; this is that second half. Scratch layouts
/// and destination-prefix reconstructions share this namespace.
/// The shared-operand two-by-one recursion lives in [`two_by_one`](super::two_by_one).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaCrt;

impl SsaCrt {
    /// Computes `dst = a * b mod (B^n - 1)`, where `n` is the common width of all
    /// three slices.
    ///
    /// Splits `n` in half at every level until it reaches
    /// `SSA_BNM1_BASECASE_LIMBS`, so `n` must stay even the whole way down. `n` need
    /// not be a power of two, as the planner's `SsaPlan::crt_half_width` guarantees the
    /// weaker property that suffices, namely that the odd part of `n` already fits
    /// the basecase. The supplied executor is reused for each independent transform;
    /// nested pointwise coefficient products deliberately select their sequential
    /// child executor, so the outer worker budget is not multiplied recursively.
    /// Executes a retained Mersenne product tree without planning descendants.
    ///
    /// # Safety
    /// Operands and destination have the width used to construct `prepared`;
    /// scratch covers the layout computed for the supplied executor's width.
    pub unsafe fn mul_mod_bnm1_prepared<E: ParallelExecutor>(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
        prepared: &[CrtMulLevel],
    ) {
        let n = a.len();
        debug_assert_eq!(n, b.len(), "mul_mod_bnm1 widths must match");
        debug_assert_eq!(n, dst.len(), "mul_mod_bnm1 dst width must match");

        // SAFETY: CrtMulPlan guarantees at least one level is always present.
        let (first, child_levels) = unsafe { prepared.split_first().unwrap_unchecked() };

        let ring = match first {
            CrtMulLevel::Basecase { plan, work } => {
                // Basecase: perform exact multiplication and fold.
                // SAFETY: the basecase plan reserves a 2n-limb product and
                // work limbs; both exclusive initialized prefixes fit scratch.
                let (prod, mul_scratch) = unsafe {
                    let prod_len = n.unchecked_mul(2);
                    let (prod, rest) = scratch.split_at_mut_unchecked(prod_len);
                    (prod, rest.get_unchecked_mut(..*work))
                };
                Multiplication::execute_plan_with_executor(
                    *plan,
                    prod,
                    a,
                    b,
                    mul_scratch,
                    &SequentialExecutor,
                );
                Self::fold_bnm1_product(dst, prod);
                return;
            }
            CrtMulLevel::Split { ring } => ring,
        };

        debug_assert!(
            n.is_multiple_of(2),
            "recursive mul_mod_bnm1 width must be even"
        );
        let h = n >> 1;
        // SAFETY: the retained split reserves h Mersenne limbs and the larger
        // child arena; h=n/2 bounds the guarded width h+1<=n.
        let (cl, xm, rest2) = unsafe {
            let cl = h.unchecked_add(1);
            let (xm, work) = scratch.split_at_mut_unchecked(h);
            (cl, xm, work)
        };

        // 1. Compute xp = a * b mod (B^h + 1)
        {
            // SAFETY: h>=1 and dst.len()==2h, so the Fermat writer can
            // initialize the complete h+1-limb prefix directly in the output.
            let xp = unsafe { dst.get_unchecked_mut(..cl) };
            // SAFETY: layout_len reserves two complete cl-limb operands and
            // this ring's workspace in the reusable tail, disjoint from xp/xm.
            let (left_padded, right_padded, ring_scratch) = unsafe {
                let (left, rest) = rest2.split_at_mut_unchecked(cl);
                let (right, work) = rest.split_at_mut_unchecked(cl);
                (left, right, work)
            };

            Self::stage_padded_difference(left_padded, a);
            Self::stage_padded_difference(right_padded, b);

            let modulus_bits = ring.modulus_bits;
            // SAFETY: staging initialized both canonical h+1-limb residues.
            // xp and the executor-sized ring arena are disjoint planned spans.
            unsafe {
                SsaTransform::fft_mul_mod_slices_with_executor(
                    xp,
                    left_padded,
                    right_padded,
                    modulus_bits,
                    None,
                    false,
                    ring,
                    executor,
                    ring_scratch,
                );
            }
        }

        // 2. Compute xm = a * b mod (B^h - 1)
        {
            // SAFETY: the same tail reserves two h-limb residues and the
            // retained Mersenne child's arena; the Fermat borrows have ended.
            let (left_folded, right_folded, xm_scratch) = unsafe {
                let (left, rest) = rest2.split_at_mut_unchecked(h);
                let (right, work) = rest.split_at_mut_unchecked(h);
                (left, right, work)
            };

            Self::stage_folded_sum(left_folded, a);
            Self::stage_folded_sum(right_folded, b);

            // SAFETY: exact halving partitions match the retained child levels.
            unsafe {
                Self::mul_mod_bnm1_prepared(
                    xm,
                    left_folded,
                    right_folded,
                    xm_scratch,
                    executor,
                    child_levels,
                );
            }
        }

        // SAFETY: the Fermat writer initialized dst[..=h]; xm is the complete
        // h-limb Mersenne residue and dst retains its full 2h-limb width.
        unsafe {
            Self::merge_crt_halves_in_place(dst, xm);
        }
    }

    /// Computes `dst = a * a mod (B^n - 1)`, where `n` is the common width of both
    /// slices.
    ///
    /// The same split as [`Self::mul_mod_bnm1_prepared`], specialised throughout. Modulo `B^h + 1`
    /// the base `B^h` is `-1`, so `a` reduces to `a_low - a_high` and the square of
    /// the residue is the square of that difference; modulo `B^h - 1` the base is
    /// `1` and the operand is `a_low + a_high`. Every level therefore runs one
    /// forward transform where the general product runs two, and the basecase
    /// reaches the tower's squaring tier rather than its product tier.
    ///
    /// Executes a retained Mersenne square tree without replanning descendants.
    ///
    /// # Safety
    /// Operand and destination match the plan's width, scratch covers its full
    /// arena for the supplied executor's width.
    pub unsafe fn sqr_mod_bnm1_prepared<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
        prepared: &[CrtSquareLevel],
    ) {
        let n = a.len();
        debug_assert_eq!(n, dst.len(), "sqr_mod_bnm1 dst width must match");

        // SAFETY: CrtSquarePlan guarantees at least one level is always present.
        let (first, child_levels) = unsafe { prepared.split_first().unwrap_unchecked() };

        let ring = match first {
            CrtSquareLevel::Basecase { plan, work } => {
                // SAFETY: the retained basecase reserves 2n output limbs and
                // work initialized scratch limbs in disjoint exclusive spans.
                let (prod, sqr_scratch) = unsafe {
                    let (prod, rest) = scratch.split_at_mut_unchecked(n.unchecked_mul(2));
                    (prod, rest.get_unchecked_mut(..*work))
                };
                Multiplication::execute_square_plan_with_executor(
                    *plan,
                    prod,
                    a,
                    sqr_scratch,
                    &SequentialExecutor,
                );
                Self::fold_bnm1_product(dst, prod);
                return;
            }
            CrtSquareLevel::Split { ring } => ring,
        };

        debug_assert!(
            n.is_multiple_of(2),
            "recursive sqr_mod_bnm1 width must be even"
        );
        let h = n >> 1;
        // SAFETY: the square layout reserves h Mersenne limbs and the larger
        // child arena; h=n/2 bounds the guarded width h+1<=n.
        let (cl, xm, rest2) = unsafe {
            let cl = h.unchecked_add(1);
            let (xm, work) = scratch.split_at_mut_unchecked(h);
            (cl, xm, work)
        };

        // 1. Compute xp = a^2 mod (B^h + 1) from a_low - a_high.
        {
            // SAFETY: h>=1 and dst.len()==2h bound the complete guarded prefix.
            let xp = unsafe { dst.get_unchecked_mut(..cl) };
            // SAFETY: sqr_layout_len reserves this cl-limb operand and ring arena.
            let (padded, ring_scratch) = unsafe { rest2.split_at_mut_unchecked(cl) };

            Self::stage_padded_difference(padded, a);

            let modulus_bits = ring.modulus_bits;
            // SAFETY: the operand is one complete guarded coefficient, disjoint from
            // xp, and the ring scratch is sized for this exact modulus width.
            unsafe {
                SsaTransform::fft_sqr_mod_slices_with_executor(
                    xp,
                    padded,
                    modulus_bits,
                    false,
                    ring,
                    executor,
                    ring_scratch,
                );
            }
        }

        // 2. Compute xm = a^2 mod (B^h - 1) from a_low + a_high.
        {
            // SAFETY: the dead Fermat tail covers h folded limbs and the child arena.
            let (folded, xm_scratch) = unsafe { rest2.split_at_mut_unchecked(h) };

            Self::stage_folded_sum(folded, a);

            // SAFETY: exact halving partitions match the retained child levels.
            unsafe {
                Self::sqr_mod_bnm1_prepared(xm, folded, xm_scratch, executor, child_levels);
            }
        }

        // SAFETY: the Fermat writer initialized dst[..=h], and xm contains
        // the complete h-limb Mersenne residue for this 2h-limb destination.
        unsafe {
            Self::merge_crt_halves_in_place(dst, xm);
        }
    }
    /// Folds an exact `2n`-limb product into its `B^n - 1` residue.
    pub fn fold_bnm1_product(dst: &mut [impl LimbOutput], prod: &[Limb]) {
        let n = dst.len();
        debug_assert!(n > 0, "a Mersenne ring contains at least one data limb");
        debug_assert_eq!(
            prod.len(),
            // SAFETY: n native limbs occupy at most isize::MAX bytes;
            // doubling n fits usize on every supported pointer width.
            unsafe { n.unchecked_mul(2) },
            "bnm1 folding requires the exact product"
        );
        // SAFETY: the admitted CRT geometry has n > 0; prod holds exactly two
        // initialized n-limb halves and dst is a disjoint writable n-limb span.
        // Each slice supplies native limb alignment. The architecture facade
        // selects a supported kernel, which writes the folded sum in one pass.
        let carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                dst.as_mut_ptr().cast(),
                prod.as_ptr(),
                prod.as_ptr().add(n),
                n,
            )
        };
        if carry > 0 {
            // Both halves are <=B^n-1; if their sum carries, its low part
            // is <=B^n-2 and therefore absorbs the end-around +1.
            // SAFETY: the three-operand addition initialized every n-limb
            // destination element before any end-around carry reads it.
            let escaped = SsaCarry::propagate_carry(unsafe { LimbOutput::assume_init_mut(dst) });
            debug_assert!(!escaped, "two product halves absorb the folded carry");
        }
    }
}
