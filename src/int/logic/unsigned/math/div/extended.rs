//! Extended Euclid with retained HGCD matrices and two absolute cofactors.
//!
//! The coefficients of the original `a` attached to `(r0, r1)` are
//! `((-1)^step * s0, -(-1)^step * s1)`. A retained matrix `M` maps the new
//! remainders to the old pair, so its inverse gives
//! `s0' = m11*s0 + m01*s1`, `s1' = m10*s0 + m00*s1`.
//! Only the determinant sign changes the coefficient parity.

#![expect(
    unsafe_code,
    reason = "normalized nonzero remainders bound limb reads and scalar division; cofactor guards fit materialized slice limits"
)]

use core::{cmp::Ordering, mem::swap};

use super::{
    ArchKernels, DivScratch, Division, EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS,
    EXTENDED_GCD_COFACTOR_BATCH_RATIO, EXTENDED_GCD_WIDE_THRESHOLD,
    EXTENDED_HGCD_CROSSOVER_THRESHOLD, Gcd, HgcdMatrix, HgcdWorkspace, InternalMpUint, Limb,
};

impl Division {
    /// Returns the GCD, the absolute coefficient of `a`, and its sign parity.
    #[expect(
        clippy::too_many_lines,
        reason = "The HGCD, Lehmer, scalar-tail and exact fallback transitions share one remainder/cofactor state and reusable workspace."
    )]
    pub fn compute_extended_euclid_core(
        a: &InternalMpUint,
        b: &InternalMpUint,
    ) -> (InternalMpUint, InternalMpUint, usize) {
        let mut scratch = DivScratch::default();
        let (mut r0, mut r1, mut s0, mut s1, mut step) =
            if a.limbs().len() > b.limbs().len() && !b.is_zero() {
                // For a = q*b + r, the coefficient of a in r is one,
                // independently of q. Reduce unequal widths before creating
                // cofactor or HGCD storage; an exact multiple needs neither.
                let mut remainder = InternalMpUint::zero();
                Self::rem_into(a, b, &mut remainder, &mut scratch);
                if remainder.is_zero() {
                    return (b.clone(), InternalMpUint::zero(), 1);
                }
                (
                    b.clone(),
                    remainder,
                    InternalMpUint::zero(),
                    InternalMpUint::one(),
                    1,
                )
            } else {
                (
                    a.clone(),
                    b.clone(),
                    InternalMpUint::one(),
                    InternalMpUint::zero(),
                    0,
                )
            };
        let mut temp = InternalMpUint::zero();
        let mut temp_b = InternalMpUint::zero();
        // b = s1*r0 + s0*r1 bounds the live cofactors by b while both
        // remainders are nonzero. Scalar products need two carry guards.
        // SAFETY: a valid limb slice occupies at most isize::MAX bytes and
        // each limb occupies at least two bytes; adding one or two fits usize.
        let (head_capacity, cofactor_capacity) = unsafe {
            (
                b.limbs().len().unchecked_add(1),
                b.limbs().len().unchecked_add(2),
            )
        };
        s0.reserve(head_capacity);
        s1.reserve(cofactor_capacity);
        temp_b.reserve(cofactor_capacity);

        let mut q = InternalMpUint::zero();
        let mut next_r = InternalMpUint::zero();
        let mut next_r0 = InternalMpUint::zero();
        let mut next_r1 = InternalMpUint::zero();
        // HGCD receives only these reduced operands. The initial wide division
        // used separate scratch, so its discarded width must not bypass the
        // bounded thread-local pool for the smaller HGCD problem.
        let workspace_len = r0.limbs().len().max(r1.limbs().len());
        let run = |mut hgcd_workspace: Option<&mut HgcdWorkspace>| {
            let mut block_matrix = HgcdMatrix::default();

            while !r1.is_zero() {
                if r0.cmp(&r1) == Ordering::Less {
                    // A quotient-zero transition exchanges both rows and parity.
                    swap(&mut r0, &mut r1);
                    swap(&mut s0, &mut s1);
                    step ^= 1;
                    continue;
                }

                if r0.limbs().len() == 1 {
                    // SAFETY: r0 >= r1 > 0 and r0 has one limb, hence both
                    // initialized slices contain exactly one element.
                    let (head, tail) =
                        unsafe { (*r0.limbs().get_unchecked(0), *r1.limbs().get_unchecked(0)) };
                    let (gcd, head_coefficient, tail_coefficient, parity) =
                        Self::extended_gcd_limb(head, tail);
                    Gcd::assign_linear_combination(
                        &mut temp,
                        &s0,
                        head_coefficient,
                        &s1,
                        tail_coefficient,
                    );
                    return (InternalMpUint::from_limb(gcd), temp, step ^ parity);
                }

                let cofactor_len = s0.limbs().len().max(s1.limbs().len());
                if cofactor_len >= EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS
                    && r0.limbs().len() < cofactor_len.div_euclid(EXTENDED_GCD_COFACTOR_BATCH_RATIO)
                {
                    // The measured width policy amortizes recursive completion
                    // over the large cofactors. Finish the smaller problem and
                    // compose only its surviving row, replacing repeated large
                    // cofactor updates with two products. Cofactors are bounded
                    // by the caller's b; the ratio is >= 1, so the strict width
                    // comparison proves recursive descent independently of tuning.
                    let (gcd, head_coefficient, parity) =
                        Self::compute_extended_euclid_core(&r0, &r1);
                    temp_b.assign_product_with_scratch(
                        &r0,
                        &head_coefficient,
                        &mut scratch.mul_scratch,
                    );
                    if parity == 0 {
                        temp_b.sub_assign(&gcd);
                    } else {
                        temp_b.add_assign(&gcd);
                    }
                    // |y| = (r0*|x| - (-1)^parity*gcd)/r1 is exact. Keep
                    // absolute coefficients throughout the internal composition.
                    Self::div_exact_into(&temp_b, &r1, &mut temp, &mut scratch);
                    // Opposite signs of the two live cofactors turn the signed
                    // Bezout row into a sum of absolute products. Its determinant
                    // sign multiplies the parity already accumulated above.
                    temp_b.assign_product_with_scratch(
                        &head_coefficient,
                        &s0,
                        &mut scratch.mul_scratch,
                    );
                    next_r.assign_product_with_scratch(&temp, &s1, &mut scratch.mul_scratch);
                    temp_b.add_assign(&next_r);
                    return (gcd, temp_b, step ^ parity);
                }

                // SAFETY: the quotient-zero transition establishes r0 >= r1;
                // canonical representations therefore have ordered widths.
                let width_gap = unsafe { r0.limbs().len().unchecked_sub(r1.limbs().len()) };
                if r1.limbs().len() >= EXTENDED_HGCD_CROSSOVER_THRESHOLD
                    && cofactor_len >= EXTENDED_GCD_WIDE_THRESHOLD
                    && width_gap <= 1
                    && let Some(workspace) = hgcd_workspace.as_deref_mut()
                    && Gcd::hgcd_block_matrix(
                        &mut r0,
                        &mut r1,
                        &mut block_matrix,
                        &mut scratch,
                        workspace,
                    )
                {
                    // The first accepted block already owns both cofactors when
                    // the live row is a unit basis vector. Transfer those owners
                    // instead of multiplying by zero/one and copying the entries.
                    if s1.is_zero() && s0.is_one() {
                        swap(&mut s0, &mut block_matrix.m11);
                        swap(&mut s1, &mut block_matrix.m10);
                        step ^= usize::from(!block_matrix.positive_det);
                        continue;
                    }
                    if s0.is_zero() && s1.is_one() {
                        swap(&mut s0, &mut block_matrix.m01);
                        swap(&mut s1, &mut block_matrix.m00);
                        step ^= usize::from(!block_matrix.positive_det);
                        continue;
                    }
                    temp.assign_product_with_scratch(
                        &block_matrix.m10,
                        &s0,
                        &mut scratch.mul_scratch,
                    );
                    temp_b.assign_product_with_scratch(
                        &block_matrix.m00,
                        &s1,
                        &mut scratch.mul_scratch,
                    );
                    temp.add_assign(&temp_b);
                    temp_b.assign_product_with_scratch(
                        &block_matrix.m11,
                        &s0,
                        &mut scratch.mul_scratch,
                    );
                    s0.assign_product_with_scratch(
                        &block_matrix.m01,
                        &s1,
                        &mut scratch.mul_scratch,
                    );
                    temp_b.add_assign(&s0);
                    swap(&mut s0, &mut temp_b);
                    swap(&mut s1, &mut temp);
                    step ^= usize::from(!block_matrix.positive_det);
                    continue;
                }

                // SAFETY: the materialized r1 slice has at most isize::MAX
                // bytes and at least two bytes per limb, so len+1 fits usize.
                let two_limb_window_limit = unsafe { r1.limbs().len().unchecked_add(1) };
                if r1.limbs().len() >= 2 && r0.limbs().len() <= two_limb_window_limit {
                    // A determinant-one batch keeps the coefficient parity.
                    if let Some((u0, v0, u1, v1)) = Gcd::hgcd2(r0.limbs(), r1.limbs())
                        && Gcd::lehmer_update_dispatched(
                            &mut r0,
                            &mut r1,
                            &mut next_r0,
                            &mut next_r1,
                            u0,
                            v0,
                            u1,
                            v1,
                            true,
                            None,
                        )
                    {
                        // A live unit row is the HGCD matrix column itself; skip
                        // the two zero products and the one-scaling copies.
                        if s1.is_zero() && s0.is_one() {
                            s0.set_limb(u0);
                            s1.set_limb(u1);
                            continue;
                        }
                        if s0.is_zero() && s1.is_one() {
                            s0.set_limb(v0);
                            s1.set_limb(v1);
                            continue;
                        }
                        Gcd::assign_linear_combinations(
                            &mut temp,
                            &mut temp_b,
                            &s0,
                            &s1,
                            u0,
                            v0,
                            u1,
                            v1,
                        );
                        swap(&mut s0, &mut temp);
                        swap(&mut s1, &mut temp_b);
                        continue;
                    }
                    if let Some(small_q) = Gcd::fast_small_div_step(&mut r0, &r1) {
                        swap(&mut r0, &mut r1);
                        if !r1.is_zero() {
                            // Fibonacci and other quotient-one trails add the
                            // live cofactor in place instead of writing a new
                            // combination through a temporary.
                            if small_q == 1 {
                                s0.add_assign(&s1);
                            } else {
                                Gcd::assign_linear_combination(&mut temp, &s0, 1, &s1, small_q);
                                swap(&mut s0, &mut temp);
                            }
                        }
                        swap(&mut s0, &mut s1);
                        step ^= 1;
                        continue;
                    }
                }

                // A one-limb divisor and rejected simulations share this exact
                // transition. The coefficient of a zero remainder is discarded.
                Self::div_rem_into(&r0, &r1, &mut q, &mut next_r, &mut scratch);
                swap(&mut r0, &mut r1);
                swap(&mut r1, &mut next_r);
                if !r1.is_zero() {
                    temp.assign_product_with_scratch(&q, &s1, &mut scratch.mul_scratch);
                    s0.add_assign(&temp);
                }
                swap(&mut s0, &mut s1);
                step ^= 1;
            }
            (r0, s0, step)
        };
        // Every later remainder is bounded by b. Below the recursive
        // crossover no HGCD block can run, so the coefficient loop needs
        // neither a thread-local borrow nor a transfer of pooled workspace.
        if b.limbs().len() >= EXTENDED_HGCD_CROSSOVER_THRESHOLD {
            HgcdWorkspace::with_thread_local(workspace_len, |workspace| run(Some(workspace)))
        } else {
            run(None)
        }
    }

    /// Finishes Euclid in registers and returns the surviving absolute row.
    ///
    /// Every accumulated coefficient is bounded by an original operand divided
    /// by the GCD, including the terminal zero row. Both operands fit a limb,
    /// so the complete multiply-adds below fit a limb on every pointer width.
    pub fn extended_gcd_limb(mut head: Limb, mut tail: Limb) -> (Limb, Limb, Limb, usize) {
        let mut head_coefficient: Limb = 1;
        let mut next_head_coefficient: Limb = 0;
        let mut tail_coefficient: Limb = 0;
        let mut next_tail_coefficient: Limb = 1;
        let mut parity = 0;
        while tail != 0 {
            // The borrow distinguishes an initial head<tail from the unit
            // quotient interval without treating the wrapped difference as exact.
            let (difference, underflow) = head.overflowing_sub(tail);
            let (remainder, new_head, new_tail) = if !underflow && difference < tail {
                // tail <= head < 2*tail proves q=1 without forming 2*tail,
                // which could overflow. Consecutive Fibonacci inputs retain
                // this relation until the last step; no division or coefficient
                // multiplication is needed for their unit-quotient tail.
                // SAFETY: these positive coefficient sums are entries of the
                // next Euclidean row, bounded by the original limb operands/GCD.
                unsafe {
                    (
                        difference,
                        head_coefficient.unchecked_add(next_head_coefficient),
                        tail_coefficient.unchecked_add(next_tail_coefficient),
                    )
                }
            } else {
                // SAFETY: tail is nonzero and the high numerator limb is zero,
                // so the quotient fits one limb and hardware division is valid.
                let (quotient, remainder) =
                    unsafe { ArchKernels::divrem_1_unchecked(head, 0, tail) };
                // SAFETY: each complete nonnegative multiply-add is an entry
                // of the next Euclidean row, bounded by an original operand/GCD.
                // Its product and partial sums therefore also fit a native limb.
                unsafe {
                    (
                        remainder,
                        head_coefficient
                            .unchecked_add(quotient.unchecked_mul(next_head_coefficient)),
                        tail_coefficient
                            .unchecked_add(quotient.unchecked_mul(next_tail_coefficient)),
                    )
                }
            };
            head_coefficient = next_head_coefficient;
            next_head_coefficient = new_head;
            tail_coefficient = next_tail_coefficient;
            next_tail_coefficient = new_tail;
            head = tail;
            tail = remainder;
            parity ^= 1;
        }
        (head, head_coefficient, tail_coefficient, parity)
    }
}
