//! Lower Newton approximation to `B^(2n) / D`.
//!
//! Reference: R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*
//! Cambridge University Press, 2011, Section 3.4.1,
//! Algorithm 3.5 (`ApproximateReciprocal`), Lemma 3.8.
//! The contract is `V·D ≤ B^(2n) < (V+2)·D`.
//! The derivation below establishes the precision schedule and error bound.
//!
//! The recursion starts with the leading half of the divisor and refines against
//! the full divisor. The basecase is exact; each refinement doubles precision.
//!
//! # Reciprocal contract
//!
//! With `T = B^(2n)/D` and `V` the returned value: `0 ≤ T − V < 2`.
//!
//! # Error bound
//!
//! ## Base case (`n ≤ NEWTON_RAPHSON_BASECASE_LIMBS`)
//!
//! `V = floor((B^(2n)−1)/D)`. From `B^(2n)−1 = V·D + r`, `0 ≤ r < D`:
//!
//! ```text
//! T − V = (r + 1)/D,   so  0 < T − V ≤ 1.
//! ```
//!
//! ## Refinement (`n > 4`)
//!
//! ```text
//! k  = floor(n/2) + 1
//! Dₕ = floor(D / B^(n−k))          (top k limbs of D)
//! S  = B^(n−k)
//! ```
//!
//! **Induction hypothesis on W₀ = reciprocal(Dₕ):**
//!
//! ```text
//! 0 ≤ B^(2k)/Dₕ − W₀ < 2,       B^k ≤ W₀ < 2·B^k.
//! ```
//!
//! **Residue construction.** Write `D = Dₕ·S + d`, `0 ≤ d < S`. Before
//! correction:
//!
//! ```text
//! R₀ = B^(n+k) − D·W₀ = S·(B^(2k) − Dₕ·W₀) − d·W₀.
//! ```
//!
//! Upper: `S·(B^(2k)−Dₕ·W₀) < 2·Dₕ·S ≤ 2·D`, `−d·W₀ ≤ 0`, so `R₀ < 2D`.
//! Lower: `S·(B^(2k)−Dₕ·W₀) ≥ 0`, `−d·W₀ > −2·B^n`, so `R₀ > −2·B^n`.
//!
//! The correction loop (linear or modular) decrements W₀ → W and adds D until
//! `R = B^(n+k) − D·W ≥ 0`. Each step crosses from negative to non-negative,
//! giving `0 ≤ R < D` when at least one decrement was applied, or `R = R₀ < 2D`
//! when none was needed. In all cases: **`0 ≤ R < 2D`**.
//!
//! **Refinement step.** With `X = W·S`:
//!
//! ```text
//! T − X = S·R/D.
//! C     = floor(R / B^(k−1)) · W.
//! V     = X + floor(C / B^(k+1)).
//! ```
//!
//! **Newton identity.** Using `R + W·D = B^(n+k)`:
//!
//! ```text
//! (T−X)²/T + R·W/B^(2k) = (S·R/D)²·D/B^(2n) + R·W/B^(2k)
//!                        = R²/(D·B^(2k)) + R·W/B^(2k)
//!                        = R·(R + W·D)/(D·B^(2k))
//!                        = R·B^(n+k)/(D·B^(2k))
//!                        = R·S/D = T − X.           ✓
//! ```
//!
//! Write `R = H_R·B^(k−1) + ρ`, `0 ≤ ρ < B^(k−1)`:
//!
//! ```text
//! R·W/B^(2k) − floor(H_R·W/B^(k+1))
//!   = ρ·W/B^(2k)  +  frac(H_R·W/B^(k+1))
//!   ≡ δ + η.
//! ```
//!
//! **Error bounds.**
//!
//! - `δ = ρ·W/B^(2k) < B^(k−1)·2·B^k / B^(2k) = 2/B`.
//! - `η = frac(H_R·W/B^(k+1)) < 1`.
//! - `(T−X)²/T < (2·S)² / (B^(2n)/D) = 4·S²·D/B^(2n) = 4·D/B^(2k)`.
//!   Since `D < B^n` and `2k ≥ n+1`: `4·D/B^(2k) < 4·B^(n−2k) ≤ 4/B`.
//!   Even n gives the stronger bound `4/B²`.
//!
//! **Result:**
//!
//! ```text
//! T − V = (T−X)²/T + δ + η  <  1 + 6/B.
//! ```
//!
//! On every supported target (`B ≥ 2^16`): `6/B < 1`, so **`T − V < 2`**.
//! The lower bound `T − V ≥ 0` follows from `T − V = (T−X)²/T + δ + η`, a sum of
//! non-negative terms.
//!
//! ## Tight integer form
//!
//! Multiplying the parity-dependent bound by `D` and then by `B²`:
//!
//! ```text
//! B²·Δ < (B² + 2B + 4)·D    for even n,
//! B²·Δ < (B² + 6B)·D        for odd n,
//! where Δ = B^(2n) − V·D.
//! ```

#![expect(
    unsafe_code,
    reason = "reciprocal precision bounds each refinement slice and initialized product span before pointer access"
)]

use core::{
    mem::{replace, swap},
    num::NonZeroUsize,
    ops::Rem,
    ptr::{copy, copy_nonoverlapping, write_bytes},
};

#[cfg(not(target_pointer_width = "16"))]
use super::Multiplication;
use super::{
    Addition, DivScratch, Division, InternalMpUint, Limb, LowProduct, NEWTON_RAPHSON_BASECASE_LIMBS,
};

impl Division {
    /// Constructs the reciprocal for an `n + block` by `n` division.
    ///
    /// Returns owned storage and the number of low guard limbs to skip. The
    /// remaining slice represents V <= B^(n+block)/D with absolute error < 2.
    /// `GUARD_FULL_BLOCK` also retains a guard when `block == n`, allowing
    /// quotient-only division to certify its final block without a residue.
    /// Skipping the guard through a slice avoids shifting the reciprocal buffer.
    /// Before discarding the guard, its error is below `2+1/α²` for
    /// `α=D/B^n`. For k=block+1<=n and S=B^(n-k), the upward-rounded
    /// divisor `D_r` obeys `0<=D_r-D<=S`. Rounding costs at most
    /// `B^(n+k)*S/(D*D_r)<=1/α²`; Newton contributes less than two.
    /// Full-width padded inversion has no rounding error, and discarding
    /// additional guard digits reduces the error below two.
    /// Its leading limb is one: the basecase lies in [B^k, 2B^k), and a
    /// refinement cannot decrement B^k because D*B^k < B^(n+k). Adding the
    /// nonnegative correction preserves the lower endpoint; at working width k,
    /// Newton refinement stays below its normalized target. For D=B^k/2,
    /// the recurrence returns 2B^k-1; every other normalized D has target
    /// strictly below 2B^k. The leading reciprocal digit is therefore one.
    pub fn newton_block_reciprocal<const GUARD_FULL_BLOCK: bool>(
        den: &[Limb],
        block: usize,
        scratch: &mut DivScratch,
    ) -> (InternalMpUint, usize) {
        let n = den.len();
        debug_assert!(
            block > 0 && block <= n,
            "reciprocal block must fit the nonempty divisor"
        );
        if block == n {
            if !GUARD_FULL_BLOCK {
                return (Self::newton_reciprocal(den, scratch), 0);
            }
            // Inverting D*B at width n+1 gives B^(2n+1)/D. Its discarded
            // low limb leaves an n-limb inverse with error < 1+2/B; retaining
            // that limb certifies the final quotient at one extra radix digit.
            // The divisor remains live during inversion; the basecase numerator
            // independently retains u_norm. Newton does not use dummy_quot.
            let mut padded = replace(&mut scratch.dummy_quot, InternalMpUint::zero());
            // SAFETY: a materialized Limb slice is bounded by isize::MAX bytes;
            // one additional limb fits usize on every supported pointer width.
            let width = unsafe { n.unchecked_add(1) };
            let mut destination = padded.prepare_limb_write(width);
            // SAFETY: preparation reserves n+1 aligned limbs disjoint from den.
            // The low zero and complete n-limb copy initialize every slot before
            // commit. The normalized divisor keeps the high limb nonzero.
            unsafe {
                let pointer = destination.as_mut_ptr();
                pointer.write(0);
                copy_nonoverlapping(den.as_ptr(), pointer.add(1), n);
                let _ = destination.commit();
            }
            let reciprocal = Self::newton_reciprocal(padded.limbs(), scratch);
            scratch.dummy_quot = padded;
            return (reciprocal, 1);
        }
        // One guard limb bounds the untruncated rounding error by 1/α²;
        // dropping it reduces that contribution below 4/B. Round the high
        // divisor upward to preserve a lower bound when its suffix is nonzero.
        // SAFETY: block < n proves block + 1 <= n and n - width is in bounds.
        let width = unsafe { block.unchecked_add(1) };
        if width == n {
            return (Self::newton_reciprocal(den, scratch), 1);
        }
        // Reciprocal recursion uses u_norm for its basecase numerator;
        // the discarded-quotient slot is free throughout the Newton branch.
        let mut rounded = replace(&mut scratch.dummy_quot, InternalMpUint::zero());
        // SAFETY: width <= n; the source and rounded storage have distinct owners.
        rounded.clone_from_slice(unsafe { den.get_unchecked(n.unchecked_sub(width)..) });
        rounded.increment();
        let reciprocal = if rounded.limbs().len() > width {
            // The rounded divisor is B^width. Its scaled inverse is B^width;
            // discarding one low limb yields the exact lower bound B^block.
            rounded
        } else {
            let value = Self::newton_reciprocal(rounded.limbs(), scratch);
            scratch.dummy_quot = rounded;
            value
        };
        // Before dropping the guard, the Newton error is < 2. Afterwards,
        // rounding the divisor costs < 4/B, reciprocal error costs < 2/B,
        // and discarding the guard costs < 1: total error < 1 + 6/B < 2.
        (reciprocal, 1)
    }

    /// Computes the reciprocal of a normalized divisor `den` (MSB set).
    ///
    /// With `T = B^(2n)/D` and `x = v_hi * B^(n-k)`, refinement targets
    /// `2x - x²/T = T - (T-x)²/T <= T`. Truncating the error product and
    /// rounding downward preserve a lower bound; recursive guard precision
    /// keeps the absolute error below two.
    #[expect(
        clippy::too_many_lines,
        reason = "Newton reciprocal refinement combines recursive basecase, modular/linear error product, and correction in one pipeline."
    )]
    pub fn newton_reciprocal(den: &[Limb], scratch: &mut DivScratch) -> InternalMpUint {
        let n = den.len();
        if n <= NEWTON_RAPHSON_BASECASE_LIMBS || n <= 4 {
            return Self::reciprocal_basecase(den, scratch);
        }

        // With k = floor(n/2) + 1 and a preceding reciprocal error below two,
        // correction establishes 0 <= T-x < 2*B^(n-k). Dropping k-1 low
        // error limbs adds less than 2/B before rounding.
        // T-result < 1 + 4*B^(n-2k) + 2/B <= 1 + 6/B < 2.
        // SAFETY: n > 4 proves floor(n/2) + 1 <= n, a materialized slice length.
        let k = unsafe { (n >> 1).unchecked_add(1) };
        // SAFETY: k <= n, so n-k and n are valid bounds for den.
        let den_hi = unsafe { den.get_unchecked(n.unchecked_sub(k)..n) };
        let mut v_hi = Self::newton_reciprocal(den_hi, scratch);

        let v_hi_limbs = v_hi.limbs();
        // SAFETY: k <= n and n <= isize::MAX/size_of::<Limb>(), so n+k fits usize.
        let target_len = unsafe { n.unchecked_add(k) };

        // Error product P = den * v_hi:
        // Attempt modular multiplication mod B^w - 1 when the operand widths reach the
        // transform threshold. The residue mod B^w - 1 (with w >= n + 1) uniquely identifies
        // the small error R = B^(n+k) - den * v_hi (|R| < 2*B^n), avoiding linear zero-padding.
        // SAFETY: n + 1 fits usize for any materialized divisor.
        let minimum = unsafe { n.unchecked_add(1) };
        #[cfg(not(target_pointer_width = "16"))]
        let wrapped = Multiplication::try_mul_mod_bnm1::<false, false>(
            den,
            v_hi_limbs,
            minimum,
            &mut scratch.newton_p_buf,
            &mut scratch.mul_scratch,
        );
        #[cfg(target_pointer_width = "16")]
        let wrapped = false;

        if wrapped {
            let w = scratch.newton_p_buf.len();
            // In the Mersenne ring B^w - 1, -P_mod == ~P_mod (ones' complement negation).
            // Compute R = B^s - P_mod == ~P_mod + B^s mod (B^w - 1) directly in place
            // with a single streaming negation pass and carry propagation.
            let p_slice = scratch.newton_p_buf.as_mut_slice();
            for limb in p_slice.iter_mut() {
                *limb = !*limb;
            }
            // SAFETY: w >= minimum >= n + 1 >= 2 > 0.
            let s = unsafe { Rem::rem(target_len, NonZeroUsize::new_unchecked(w)) };
            // Add B^s (add 1 at limb index s) with end-around carry for modulus B^w - 1.
            // SAFETY: s = target_len % w < w = p_slice.len().
            let carry = Addition::propagate_carry(unsafe { p_slice.get_unchecked_mut(s..) }, 1);
            if carry != 0 {
                let _ = Addition::propagate_carry(p_slice, 1);
            }

            // In [0, B^w - 1), a nonnegative residue R < 2*B^n <= 2*B^(w-1) has its most
            // significant limb X[w - 1] <= 1. An overestimate (P > B^(n+k)) produces R in
            // (-2*B^n, 0), represented mod (B^w - 1) by X[w - 1] >= Limb::MAX - 2 > 1.
            // A single limb test identifies this without scanning the buffer.
            // Correct by decrementing v_hi and adding D into the residue mod B^w - 1.
            // SAFETY: w >= minimum >= n + 1 >= 2, so w - 1 is a valid limb index.
            while unsafe {
                *scratch
                    .newton_p_buf
                    .as_slice()
                    .get_unchecked(w.unchecked_sub(1))
            } > 1
            {
                v_hi.decrement();
                let res_slice = scratch.newton_p_buf.as_mut_slice();
                let carry_d = Addition::add_slice_in_place(res_slice, den);
                let den_len = den.len();
                // SAFETY: w >= n + 1 > den_len; newton_p_buf has w limbs.
                let carry_tail = Addition::propagate_carry(
                    unsafe { res_slice.get_unchecked_mut(den_len..) },
                    carry_d,
                );
                if carry_tail != 0 {
                    let _ = Addition::propagate_carry(res_slice, 1);
                }
            }

            // The nonnegative residue fits in at most n + 1 limbs; truncate and transfer
            // ownership to newton_r_cur via O(1) buffer swap, avoiding all copy allocations.
            scratch.newton_p_buf.truncate(minimum);
            swap(&mut scratch.newton_p_buf, &mut scratch.newton_r_cur);
        } else {
            // Linear D·W path with truncated low product.
            // After correction 0 <= R < 2D < 2B^n < B^(n+1), and B^(n+k) = 0
            // mod B^(n+1). Hence R = (-P) mod B^(n+1) with P = D·W, so the low
            // n+1 limbs determine R exactly. This skips k-1 high zero limbs.
            scratch.newton_p_buf.reset_with_capacity(minimum);
            // W has a fixed leading one: D*W=D*W_low+D*B^k. Removing that
            // scalar row leaves only k inverse limbs in the multiplication.
            // SAFETY: the k-limb reciprocal is in [B^k,2B^k), so its low
            // k limbs precede an initialized leading one.
            let inverse_low = unsafe { v_hi_limbs.get_unchecked(..k) };
            // SAFETY: reservation supplies minimum=n+1 disjoint spare limbs.
            // The rectangular product initializes every limb before set_len;
            // the reciprocal and divisor have independent initialized owners.
            unsafe {
                let _ = LowProduct::mul_with_guard(
                    den,
                    inverse_low,
                    scratch
                        .newton_p_buf
                        .spare_capacity_mut()
                        .get_unchecked_mut(..minimum),
                    &mut scratch.v_padded,
                    &mut scratch.den_pad,
                    &mut scratch.mul_scratch,
                );
                scratch.newton_p_buf.set_len(minimum);
            }
            // SAFETY: k<n and minimum=n+1. The product suffix has n+1-k
            // initialized limbs, covered by the divisor's disjoint low prefix.
            let (upper, addend) = unsafe {
                (
                    scratch.newton_p_buf.get_unchecked_mut(k..),
                    den.get_unchecked(..minimum.unchecked_sub(k)),
                )
            };
            // Escaping carry is discarded modulo B^(n+1).
            let _ = Addition::add_slice_in_place(upper, addend);
            // R = (-P) mod B^(n+1). In two's complement the low zero limbs
            // stay zero; negating the first nonzero limb consumes the unit
            // carry, so every higher limb needs only a bitwise complement.
            // A zero product remains zero without writes. The rectangular
            // product initialized the complete minimum-limb span.
            let mut digits = scratch.newton_p_buf.iter_mut();
            for digit in digits.by_ref() {
                if *digit != 0 {
                    *digit = digit.wrapping_neg();
                    break;
                }
            }
            for digit in digits {
                *digit = !*digit;
            }
            swap(&mut scratch.newton_p_buf, &mut scratch.newton_r_cur);
            // A wrapped negative residue satisfies R = B^(n+1)+R0 with
            // R0 in (-2B^n, 0), so its top limb exceeds one. A true residue
            // has top limb zero or one. Correct until the top test passes.
            // SAFETY: minimum = n+1 >= 2, so minimum-1 is a valid limb index.
            let top_idx = unsafe { minimum.unchecked_sub(1) };
            // SAFETY: den has n = minimum-1 limbs; r_cur has minimum limbs.
            let den_len = unsafe { minimum.unchecked_sub(1) };
            // SAFETY: top_idx = minimum-1 < r_cur.len(), so the read is in bounds.
            while unsafe { *scratch.newton_r_cur.as_slice().get_unchecked(top_idx) } > 1 {
                v_hi.decrement();
                let carry_d = Addition::add_slice_in_place(&mut scratch.newton_r_cur, den);
                // SAFETY: tail is the one guard limb above den; overflow past
                // B^(n+1) is discarded, which is reduction mod B^(n+1).
                let _ = Addition::propagate_carry(
                    unsafe { scratch.newton_r_cur.get_unchecked_mut(den_len..) },
                    carry_d,
                );
            }
        }

        while scratch.newton_r_cur.last() == Some(&0) {
            let _ = scratch.newton_r_cur.pop();
        }

        // C = floor(R / B^(k-1)) * W with W the corrected reciprocal.
        // Truncating R to error_high keeps the exact C of the proof; its low
        // k-1 discarded limbs cost δ < 2/B. The certified high product is exact, so
        // floor(C / B^(k+1)) adds only η < 1.
        let v_hi_limbs_ref = v_hi.limbs();
        // SAFETY: k > 0 at recursive widths; clamp covers empty correction.
        let error_shift = unsafe { k.unchecked_sub(1) };
        let error_start = error_shift.min(scratch.newton_r_cur.len());
        // SAFETY: error_start <= len; view is disjoint from destination.
        let error_high = unsafe { scratch.newton_r_cur.get_unchecked(error_start..) };
        let correction_start = if error_high.is_empty() {
            // H=0 implies C=H*W=0; no product storage or correction is needed.
            scratch.newton_c_buf.clear();
            0
        } else {
            // R<2D<2B^n gives H at most n+2-k<=k+1 limbs. If H has k+1
            // limbs, n is odd and its leading digit is exactly one. Both H and
            // W's implicit leading digits therefore need addition, not products.
            // SAFETY: correction cannot decrement B^k because D*B^k<B^(n+k),
            // so the independent W slice retains k>0 low limbs and a leading one.
            Self::newton_correction_product(
                error_high,
                unsafe { v_hi_limbs_ref.get_unchecked(..k) },
                &mut scratch.newton_c_buf,
                &mut scratch.newton_p_buf,
                &mut scratch.mul_scratch,
            )
        };

        // Proof step: V = X + floor(C / B^(k+1)) = W·B^(n-k) + floor(C / B^(k+1)).
        // The final floor adds η = frac(H_R·W/B^(k+1)) < 1 to the total error.
        // Combined: T-V = (T-X)²/T + δ + η < 1 + 6/B < 2.
        // The product consumes the last read of v_hi, so reconstruction can
        // reuse its allocation directly.
        let source_len = v_hi.limbs().len();
        // SAFETY: k <= n and v_hi has k+1 limbs: normalized D gives
        // B^k <= v_hi <= 2*B^k. The destination therefore needs n+1 limbs,
        // which fits usize for a materialized n-limb divisor on all targets.
        let (offset, result_len) = unsafe {
            let offset = n.unchecked_sub(k);
            (offset, source_len.unchecked_add(offset))
        };
        let mut destination = v_hi.prepare_limb_write(result_len);
        let pointer = destination.as_mut_ptr();
        // SAFETY: the guard reserves result_len = source_len+offset limbs.
        // copy permits overlap and preserves every initialized source limb;
        // zeroing the low offset limbs initializes the complementary prefix.
        unsafe {
            copy(pointer, pointer.add(offset), source_len);
            write_bytes(pointer, 0, offset);
            let _ = destination.commit();
        }

        if scratch.newton_c_buf.len() > correction_start {
            // SAFETY: the product builder returns the first exact correction
            // digit, and every limb of its retained suffix is initialized.
            let c_slice = unsafe { scratch.newton_c_buf.get_unchecked(correction_start..) };
            let carry_out = Addition::add_slice_in_place(v_hi.limbs_mut(), c_slice);
            // SAFETY: R has at most n+1 limbs, so its suffix after k-1 has
            // at most n+2-k limbs. Multiplication by the k+1-limb inverse
            // gives c_len <= n+3. Removing k+1 leaves c_slice.len() <= n+2-k
            // <= n+1, the committed reciprocal width, because k >= 1.
            let tail = unsafe { v_hi.limbs_mut().get_unchecked_mut(c_slice.len()..) };
            let overflow = Addition::propagate_carry(tail, carry_out);
            debug_assert_eq!(overflow, 0, "the lower reciprocal bound fits n+1 limbs");
        }
        // Correction cannot decrement B^k, so shifting W retains its leading
        // one. The nonnegative addition fits n+1 limbs and cannot clear it;
        // the reconstructed reciprocal is canonical without a normalization scan.
        v_hi
    }
}
