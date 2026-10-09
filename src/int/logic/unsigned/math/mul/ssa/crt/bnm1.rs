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

use core::ptr::copy_nonoverlapping;

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    Addition, ArchKernels, CrtMulLevel, CrtSquareLevel, Limb, LimbOutput, Multiplication, SsaCarry,
    SsaTransform,
};

/// Namespace for the `B^n - 1` half of the CRT split and its reconstructions.
///
/// The top-level entry points pair one `B^n + 1` transform with one `B^n - 1`
/// product and merge the two residues; this is that second half, together with
/// the scratch layout both halves are cut from and the two reconstructions.
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
        // SAFETY: the retained split reserves h+1 Fermat limbs, h Mersenne
        // limbs, and the larger child arena; h=n/2 bounds the guard addition.
        let (cl, xp, xm, rest2) = unsafe {
            let cl = h.unchecked_add(1);
            let (xp, after_fermat) = scratch.split_at_mut_unchecked(cl);
            let (xm, work) = after_fermat.split_at_mut_unchecked(h);
            (cl, xp, xm, work)
        };

        // 1. Compute xp = a * b mod (B^h + 1)
        {
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

        Self::merge_crt_halves(dst, xp, xm);
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
        // SAFETY: the square layout reserves h+1 Fermat limbs, h Mersenne
        // limbs and both sequential child arenas; h=n/2 bounds the guard.
        let (cl, xp, xm, rest2) = unsafe {
            let cl = h.unchecked_add(1);
            let (xp, after_fermat) = scratch.split_at_mut_unchecked(cl);
            let (xm, work) = after_fermat.split_at_mut_unchecked(h);
            (cl, xp, xm, work)
        };

        // 1. Compute xp = a^2 mod (B^h + 1) from a_low - a_high.
        {
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

        Self::merge_crt_halves(dst, xp, xm);
    }

    /// Reconstructs the exact product `dst = xp + k * B^n + k` from the two
    /// top-level CRT residues, where `k = (xm - xp) / 2 mod (B^n - 1)` and `n` is
    /// the width of `xm`.
    ///
    /// Writes an exact integer prefix without end-around carry folding.
    /// A destination shorter than `2n` limbs receives the corresponding low prefix.
    ///
    /// Shared by the tower's product and its square, which arrive here with residues
    /// of identical shape and differ only in how they computed them. `xm` is dead on
    /// entry, so it becomes `k` in place rather than being copied to a third span.
    pub fn merge_exact_product(dst: &mut [impl LimbOutput], xp: &[Limb], xm: &mut [Limb]) {
        let n = xm.len();
        debug_assert!(n > 0, "the CRT half-width must be nonzero");
        // n is the length of a live xm slice, so on every supported usize width
        // n + 1 fits: the slice itself occupies n * size_of::<Limb>() bytes.
        // SAFETY: the live n-limb slice bounds n below isize::MAX/size_of::<Limb>().
        let expected_xp_len = unsafe { n.unchecked_add(1) };
        debug_assert_eq!(
            xp.len(),
            expected_xp_len,
            "the Fermat residue carries one guard limb above the CRT half-width"
        );

        // D = X_m - X_p mod (B^n - 1)
        let k = xm;
        // SAFETY: xp carries one guard limb above the n-limb residue, so both
        // the n-limb span and the guard slot at index n are in range.
        let mut borrow = Addition::sub_slice_in_place(k, unsafe { xp.get_unchecked(..n) });
        // B^n == 1 modulo B^n-1, so the Fermat guard joins the end-around
        // borrow directly; no separate subtraction from k is required.
        // SAFETY: xp carries one guard limb above the n-limb residue.
        let guard = unsafe { *xp.get_unchecked(n) };
        // SAFETY: the subtraction flag and canonical Fermat guard are bits.
        borrow = unsafe { borrow.unchecked_add(guard) };

        // Modulo B^n-1, a borrow of B^n is equivalent to 1.
        let b2 = SsaCarry::sub_full_in_place(k, &[borrow]);
        if b2 > 0 {
            let _ = SsaCarry::sub_full_in_place(k, &[b2]);
        }

        // k = D * 2^{-1} mod (B^n - 1)
        Self::halve_mod_bnm1(k, n);

        // The all-ones representative is the redundant form of zero.
        if k.iter().all(|limb| *limb == Limb::MAX) {
            k.fill(0);
        }

        // SAFETY: n native limbs occupy at most isize::MAX bytes; doubling n
        // fits usize on every supported pointer width.
        let full_width = unsafe { n.unchecked_mul(2) };
        let max_len = dst.len().min(full_width);
        // Every limb below max_len is overwritten by the two assignments below. Only
        // a caller-provided tail beyond the complete CRT width needs clearing; for
        // the dominant equal-width product that range is empty.
        // SAFETY: max_len is min(dst.len(), 2n), so the tail is within dst.
        unsafe { dst.get_unchecked_mut(max_len..) }.fill(LimbOutput::from_limb(0));
        if max_len == 0 {
            return;
        }

        let copy_xp = max_len.min(n);
        // SAFETY: dst, xp, and k each span copy_xp limbs and are disjoint. Fusing
        // the copy and addition removes one complete output-width memory pass.
        let carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                dst.as_mut_ptr().cast(),
                xp.as_ptr(),
                k.as_ptr(),
                copy_xp,
            )
        };

        if max_len > n {
            // SAFETY: this branch proves n<max_len<=2n.
            let copy_k = unsafe { max_len.unchecked_sub(n) };
            // SAFETY: max_len <= dst.len() and max_len <= 2n, so dst[n..max_len]
            // is in range and k[..copy_k] holds exactly copy_k of k's n limbs.
            let dst_span = unsafe { dst.get_unchecked_mut(n..max_len) };
            // SAFETY: copy_k == max_len - n <= n == k.len(), so k[..copy_k] is in range.
            let k_span = unsafe { k.get_unchecked(..copy_k) };
            // SAFETY: the spans have identical lengths and distinct owners;
            // LimbOutput has native limb layout. This initializes the high span
            // before the carry correction obtains a readable limb borrow.
            unsafe {
                copy_nonoverlapping(k_span.as_ptr(), dst_span.as_mut_ptr().cast(), copy_k);
            }
            // SAFETY: the limb-add carry and canonical Fermat guard are bits.
            let total_carry = unsafe { carry.unchecked_add(guard) };
            if total_carry != 0 {
                // SAFETY: the complete high span was initialized by the copy;
                // its exclusive borrow retains native limb layout and bounds.
                let _ = SsaCarry::add_full_in_place(
                    unsafe { LimbOutput::assume_init_mut(dst_span) },
                    &[total_carry],
                );
            }
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

    /// Reconstructs `dst = xp + k * B^h + k mod (B^n - 1)` from the two residues,
    /// where `k = (xm - xp) / 2 mod (B^h - 1)`.
    ///
    /// Shared by the product and square recursions, which differ only in how they
    /// obtain `xp` and `xm`. `xm` is dead once `k` is derived, so it is transformed
    /// into `k` in place rather than copied into a third residue span.
    pub fn merge_crt_halves(dst: &mut [impl LimbOutput], xp: &[Limb], xm: &mut [Limb]) {
        let h = xm.len();
        let n = dst.len();
        debug_assert!(h > 0, "the recursive CRT half-width must be nonzero");
        debug_assert_eq!(
            h.checked_mul(2),
            Some(n),
            "the recursive CRT width must equal twice its half-width"
        );
        // h == xm.len() == k.len() is the length of a live slice below, so h + 1
        // cannot wrap: the slice itself occupies h * size_of::<Limb>() bytes.
        debug_assert_eq!(
            xp.len(),
            // SAFETY: h is a live native-limb slice length, bounded by
            // isize::MAX/size_of::<Limb>(), so its guard width fits usize.
            unsafe { h.unchecked_add(1) },
            "the Fermat CRT residue width differs"
        );
        let k = xm;
        // SAFETY: xp holds h + 1 limbs, so the h-limb span is in range.
        let mut borrow = Addition::sub_slice_in_place(k, unsafe { xp.get_unchecked(..h) });
        // SAFETY: xp holds h + 1 limbs, so the guard slot at index h is in range.
        let guard = unsafe { *xp.get_unchecked(h) };
        // SAFETY: the subtraction flag and canonical Fermat guard are bits.
        borrow = unsafe { borrow.unchecked_add(guard) };

        let b2 = SsaCarry::sub_full_in_place(k, &[borrow]);
        if b2 > 0 {
            let _ = SsaCarry::sub_full_in_place(k, &[b2]);
        }

        Self::halve_mod_bnm1(k, h);

        // SAFETY: the h-limb destination and both inputs are complete and disjoint.
        // This writes xp+k directly instead of copying xp and adding in place.
        let carry1 = unsafe {
            ArchKernels::add_limbs_3_unchecked(dst.as_mut_ptr().cast(), xp.as_ptr(), k.as_ptr(), h)
        };

        // SAFETY: dst holds n = 2h limbs and k holds h limbs, so dst[h..] spans
        // exactly k's width and the wrap-around fold below stays within dst.
        unsafe {
            copy_nonoverlapping(k.as_ptr(), dst.as_mut_ptr().cast::<Limb>().add(h), h);
        }
        // SAFETY: the low add and high copy initialized the two disjoint h-limb
        // halves, exactly covering dst's n=2h elements. No output was read
        // before those first writes established the complete initialized span.
        let initialized = unsafe { LimbOutput::assume_init_mut(dst) };
        // SAFETY: the limb-add carry and canonical Fermat guard are bits.
        let carry_guard = unsafe { carry1.unchecked_add(guard) };
        // SAFETY: dst holds n = 2h limbs, so dst[h..] spans exactly k's width.
        let carry2 = SsaCarry::add_full_in_place(
            unsafe { initialized.get_unchecked_mut(h..) },
            &[carry_guard],
        );
        if carry2 > 0 {
            // k < B^h and carry_guard <= 2 bound their sum by B^h+1.
            // An escaped carry is one and leaves the high half at most one.
            // Since B^h >= 2^16, that half is below B^h-1: the complete
            // destination cannot be all ones and absorbs the end-around +1.
            let escaped = SsaCarry::propagate_carry(initialized);
            debug_assert!(!escaped, "the CRT high half absorbs the end-around carry");
        }
    }
}
