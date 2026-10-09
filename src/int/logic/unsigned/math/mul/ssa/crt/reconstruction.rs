//! CRT reconstruction with the Fermat residue in the destination prefix.
//!
//! For `Q=B^n`, `xp + (Q+1)k` has both requested residues when
//! `k=(xm-xp)/2 mod (Q-1)`. A complete high destination half receives `k`
//! directly; shorter exact outputs use the Mersenne buffer. The Fermat guard
//! is retained in a scalar before the high output overwrites it.

#![expect(
    unsafe_code,
    reason = "CRT dimensions and initialized Fermat prefixes bound in-place limb arithmetic and complete output initialization"
)]

use core::ptr::copy_nonoverlapping;

use super::{Addition, ArchKernels, Limb, LimbOutput, SsaCarry, SsaCrt};

impl SsaCrt {
    /// Reconstructs an exact integer prefix from a separately stored Fermat
    /// residue. This also supports destinations shorter than the guarded residue.
    /// `xm` becomes the canonical quotient `k` in place.
    pub fn merge_exact_product(dst: &mut [impl LimbOutput], xp: &[Limb], xm: &mut [Limb]) {
        let n = xm.len();
        let guard = recover_quotient(xp, xm);
        let k = xm;
        if k.iter().all(|limb| *limb == Limb::MAX) {
            k.fill(0);
        }

        // SAFETY: a live native-limb slice bounds 2n below usize::MAX on
        // every pointer width. max_len is a prefix of the writable destination.
        let max_len = dst.len().min(unsafe { n.unchecked_mul(2) });
        // SAFETY: max_len<=dst.len(); this initializes only the extra output tail.
        unsafe { dst.get_unchecked_mut(max_len..) }.fill(LimbOutput::from_limb(0));
        if max_len == 0 {
            return;
        }

        let copy_xp = max_len.min(n);
        // SAFETY: the three disjoint spans cover copy_xp native limbs; xp and
        // k are initialized, and the kernel initializes this output prefix.
        let carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                dst.as_mut_ptr().cast(),
                xp.as_ptr(),
                k.as_ptr(),
                copy_xp,
            )
        };
        if max_len > n {
            // SAFETY: n<max_len<=min(dst.len(),2n); the high width is at most
            // k.len()==n. The spans are disjoint and have native limb layout.
            let high = unsafe { dst.get_unchecked_mut(n..max_len) };
            // SAFETY: high.len()<=k.len(), and the output cannot alias k.
            unsafe {
                copy_nonoverlapping(k.as_ptr(), high.as_mut_ptr().cast(), high.len());
            }
            // SAFETY: the addition carry and Fermat guard are bits.
            let correction = unsafe { carry.unchecked_add(guard) };
            if correction != 0 {
                // SAFETY: the preceding copy initialized every high output limb.
                let _ = SsaCarry::add_full_in_place(
                    unsafe { LimbOutput::assume_init_mut(high) },
                    &[correction],
                );
            }
        }
    }

    /// Reconstructs the exact product from `xp` already stored in `dst[..=n]`
    /// and the `n`-limb Mersenne residue in `xm`. The complete destination is
    /// initialized; a width below `2n` receives the low integer prefix, and
    /// limbs beyond `2n` are zero. The quotient occupies the high destination
    /// half when it fits, and reuses `xm` for shorter outputs.
    ///
    /// # Safety
    /// `n=xm.len()` is nonzero, `dst` has at least `n+1` elements, and its first
    /// `n+1` elements contain an initialized canonical residue modulo `B^n+1`.
    pub unsafe fn merge_exact_product_in_place(dst: &mut [impl LimbOutput], xm: &mut [Limb]) {
        let n = xm.len();
        debug_assert!(
            n > 0 && dst.len() > n,
            "the output covers a nonzero guarded CRT half"
        );
        // SAFETY: n is a live native-limb slice length, so 2n fits usize on
        // 16-, 32-, and 64-bit targets, where a limb occupies at least two bytes.
        let full_len = unsafe { n.unchecked_mul(2) };
        if dst.len() >= full_len {
            // SAFETY: the complete product fits and its first n+1 limbs contain
            // xp. The remaining padding is disjoint from both CRT halves.
            let (product, padding) = unsafe { dst.split_at_mut_unchecked(full_len) };
            // SAFETY: product has exactly two n-limb halves, with the caller's
            // initialized Fermat prefix and a disjoint initialized xm source.
            let _ = unsafe { merge_full_in_place::<true>(product, xm) };
            padding.fill(LimbOutput::from_limb(0));
            return;
        }
        // SAFETY: the caller establishes bounds and initialization of xp's
        // n+1-limb prefix. Its shared borrow ends before output mutation.
        let guard = recover_quotient(
            unsafe { LimbOutput::assume_init(dst.get_unchecked(..=n)) },
            xm,
        );
        let k = xm;
        // The all-ones representative is redundant zero modulo B^n-1.
        if k.iter().all(|limb| *limb == Limb::MAX) {
            k.fill(0);
        }

        // SAFETY: n<dst.len()<2n bounds the initialized low prefix and the
        // shorter writable high span. A full quotient does not fit in high.
        let (low, high) = unsafe { dst.split_at_mut_unchecked(n) };
        // SAFETY: high.len()<=n==k.len(), with disjoint native-limb spans.
        // This initializes the high output, overwriting the cached Fermat guard.
        unsafe {
            copy_nonoverlapping(k.as_ptr(), high.as_mut_ptr().cast(), high.len());
        }
        // SAFETY: low contains the initialized n Fermat data limbs; k has n
        // initialized disjoint limbs. The native-layout kernel adds in place.
        let carry =
            unsafe { ArchKernels::add_limbs_unchecked(low.as_mut_ptr().cast(), k.as_ptr(), n) };
        // SAFETY: the addition carry and canonical Fermat guard are bits.
        let correction = unsafe { carry.unchecked_add(guard) };
        if correction != 0 {
            // SAFETY: the copy initialized the complete high output span.
            let _ = SsaCarry::add_full_in_place(
                unsafe { LimbOutput::assume_init_mut(high) },
                &[correction],
            );
        }
    }

    /// Reconstructs a `2h`-limb residue modulo `B^(2h)-1` from the guarded
    /// Fermat residue in the destination and the Mersenne residue in `xm`.
    /// The complete destination is initialized, with end-around carry folding.
    ///
    /// # Safety
    /// `h=xm.len()` is nonzero, `dst.len()==2h`, and `dst[..=h]` contains an
    /// initialized residue modulo `B^h+1` whose guard is zero or one.
    pub unsafe fn merge_crt_halves_in_place(dst: &mut [impl LimbOutput], xm: &mut [Limb]) {
        let h = xm.len();
        debug_assert!(h > 0, "a recursive CRT half is nonempty");
        debug_assert_eq!(
            h.checked_mul(2),
            Some(dst.len()),
            "the recursive output covers two CRT halves"
        );
        // SAFETY: the caller supplies two complete h-limb halves and the
        // initialized Fermat prefix. The Mersenne source is disjoint from dst.
        let carry2 = unsafe { merge_full_in_place::<false>(dst, xm) };
        if carry2 > 0 {
            // k<B^h and correction<=2 leave the high half at most one after
            // overflow. B^h>=2^16, so the full output absorbs the end-around +1.
            // SAFETY: merge_full_in_place initialized both complete halves.
            let escaped = SsaCarry::propagate_carry(unsafe { LimbOutput::assume_init_mut(dst) });
            debug_assert!(!escaped, "the CRT high half absorbs the end-around carry");
        }
    }
}

/// Forms the quotient directly in the high half and returns its escaping
/// correction carry. Exact products canonicalize quotient zero; modular
/// products retain the all-ones zero representative before end-around folding.
///
/// # Safety
/// `n=xm.len()>0`, `dst.len()==2n`, and `dst[..=n]` contains initialized
/// Fermat data with a binary guard. The high half may otherwise be uninitialized.
unsafe fn merge_full_in_place<const EXACT: bool>(dst: &mut [impl LimbOutput], xm: &[Limb]) -> Limb {
    let n = xm.len();
    // SAFETY: the caller initialized the guard at n<dst.len(). Cache its
    // native-layout value before the quotient's first write overwrites it.
    let guard = unsafe { *dst.as_ptr().cast::<Limb>().add(n) };
    // SAFETY: dst.len()==2n gives two disjoint complete n-limb spans;
    // every element of the low Fermat half is initialized by the caller.
    let (low, high) = unsafe { dst.split_at_mut_unchecked(n) };
    // SAFETY: xm and low contain n initialized readable limbs, both
    // disjoint from high's n aligned writable limbs. The kernel only reads
    // its sources and initializes the high half, overwriting the cached guard.
    let borrow = unsafe {
        ArchKernels::sub_limbs_3_unchecked(
            high.as_mut_ptr().cast(),
            xm.as_ptr(),
            low.as_ptr().cast(),
            n,
        )
    };
    // SAFETY: the subtraction initialized every element of high.
    let k = unsafe { LimbOutput::assume_init_mut(high) };
    // B^n==1 modulo B^n-1: subtract the borrow and Fermat guard, folding
    // one more escaped borrow, then rotate right by one to divide by two.
    // SAFETY: the subtraction borrow and Fermat guard are bits.
    let quotient_correction = unsafe { borrow.unchecked_add(guard) };
    let borrow2 = SsaCarry::sub_full_in_place(k, &[quotient_correction]);
    if borrow2 > 0 {
        let _ = SsaCarry::sub_full_in_place(k, &[borrow2]);
    }
    SsaCrt::halve_mod_bnm1(k, n);
    if EXACT && k.iter().all(|limb| *limb == Limb::MAX) {
        k.fill(0);
    }
    // SAFETY: low contains n initialized Fermat limbs, and the disjoint
    // high half now holds n initialized quotient limbs. No copy is needed.
    let carry = unsafe { ArchKernels::add_limbs_unchecked(low.as_mut_ptr().cast(), k.as_ptr(), n) };
    // SAFETY: the low addition carry and cached Fermat guard are bits.
    let correction = unsafe { carry.unchecked_add(guard) };
    SsaCarry::add_full_in_place(k, &[correction])
}

/// Replaces `xm` with `(xm-xp)/2 mod (B^n-1)` and retains the Fermat guard.
/// Callers provide an initialized `n+1`-limb Fermat residue and `n>0`.
fn recover_quotient(xp: &[Limb], xm: &mut [Limb]) -> Limb {
    let n = xm.len();
    debug_assert!(n > 0, "a CRT quotient has a nonzero half-width");
    // SAFETY: a live native-limb slice bounds n+1 below usize::MAX on
    // all supported pointer widths.
    let expected_width = unsafe { n.unchecked_add(1) };
    debug_assert_eq!(
        xp.len(),
        expected_width,
        "the Fermat residue has one guard above its data width"
    );
    // SAFETY: xp has exactly n+1 initialized limbs.
    let (low, guard) = unsafe { (xp.get_unchecked(..n), *xp.get_unchecked(n)) };
    let borrow = Addition::sub_slice_in_place(xm, low);
    // B^n==1 modulo B^n-1, so the guard joins the end-around borrow.
    // SAFETY: the subtraction flag and canonical Fermat guard are bits.
    let correction = unsafe { borrow.unchecked_add(guard) };
    let borrow2 = SsaCarry::sub_full_in_place(xm, &[correction]);
    if borrow2 > 0 {
        let _ = SsaCarry::sub_full_in_place(xm, &[borrow2]);
    }
    SsaCrt::halve_mod_bnm1(xm, n);
    guard
}
