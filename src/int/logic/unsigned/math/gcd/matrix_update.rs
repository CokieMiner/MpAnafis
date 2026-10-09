//! Exact sparse transitions and scalar matrix composition for HGCD leaves.

#![expect(
    unsafe_code,
    reason = "scalar matrix products validate disjoint source and destination spans and initialize all carry guards before commit"
)]

use core::{
    cmp::max,
    mem::swap,
    ptr::{copy_nonoverlapping, write_bytes},
};

use super::{
    Addition, ArchKernels, DivScratch, DoubleLimb, Gcd, HgcdMatrix, InternalMpUint, LIMB_BITS, Limb,
};

impl HgcdMatrix {
    /// Right-multiplies by the matrix represented by one Lehmer batch.
    ///
    /// The inverse action uses `[[v1, v0], [u1, u0]]`, with determinant
    /// positive exactly when `even` is true. Distinct output owners preserve
    /// all four input entries until their linear combinations are complete.
    pub fn update_small(
        &mut self,
        next: &mut Self,
        u0: Limb,
        v0: Limb,
        u1: Limb,
        v1: Limb,
        even: bool,
    ) {
        if self.is_identity() {
            next.m00.set_limb(v1);
            next.m01.set_limb(v0);
            next.m10.set_limb(u1);
            next.m11.set_limb(u0);
            next.positive_det = even;
            swap(self, next);
            return;
        }

        Gcd::assign_linear_combinations(
            &mut next.m00,
            &mut next.m01,
            &self.m00,
            &self.m01,
            v1,
            u1,
            v0,
            u0,
        );
        Gcd::assign_linear_combinations(
            &mut next.m10,
            &mut next.m11,
            &self.m10,
            &self.m11,
            v1,
            u1,
            v0,
            u0,
        );
        next.positive_det = self.positive_det == even;
        swap(self, next);
    }

    /// Records `u = q*v+r` using `M * [[q, 1], [1, 0]]`.
    ///
    /// The first column requires two products and two sums. The second column
    /// takes ownership of the original first column after its last read, so
    /// it needs no limb copies. The quotient must be positive.
    pub fn update_quotient(
        &mut self,
        next: &mut Self,
        quotient: &InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(!quotient.is_zero(), "Euclidean quotient must be nonzero");

        if quotient.limbs().len() == 1 {
            // SAFETY: length is verified to be 1.
            let q = unsafe { *quotient.limbs().get_unchecked(0) };
            self.update_quotient_scalar(next, q);
            return;
        }

        if self.is_identity() {
            next.m00.clone_from(quotient);
            next.m01.set_limb(1);
            next.m10.set_limb(1);
            next.m11.clear();
            next.positive_det = false;
            swap(self, next);
            return;
        }

        Self::assign_product_slice_two_by_one(
            &mut next.m00,
            &mut next.m10,
            &self.m00,
            &self.m10,
            quotient.limbs(),
            scratch,
        );
        next.m00.add_assign(&self.m01);
        swap(&mut next.m01, &mut self.m00);

        next.m10.add_assign(&self.m11);
        swap(&mut next.m11, &mut self.m10);
        next.positive_det = !self.positive_det;
        swap(self, next);
    }

    /// Records `u = q*v+r` using `M * [[q, 1], [1, 0]]` where `q` fits in a single limb.
    pub fn update_quotient_scalar(&mut self, next: &mut Self, q: Limb) {
        debug_assert!(q != 0, "Euclidean quotient must be nonzero");

        // M * [[1, 1], [1, 0]] factors as M * [[1, 1], [0, 1]] (subtraction)
        // followed by a column exchange, including the determinant-sign change.
        // Unit quotients avoid the generic scaled products below.
        if q == 1 {
            self.m01.add_assign(&self.m00);
            self.m11.add_assign(&self.m10);
            self.swap_columns();
            return;
        }

        if self.is_identity() {
            next.m00.set_limb(q);
            next.m01.set_limb(1);
            next.m10.set_limb(1);
            next.m11.clear();
            next.positive_det = false;
            swap(self, next);
            return;
        }

        assign_scaled_add(&mut next.m00, self.m00.limbs(), q, self.m01.limbs());
        swap(&mut next.m01, &mut self.m00);

        assign_scaled_add(&mut next.m10, self.m10.limbs(), q, self.m11.limbs());
        swap(&mut next.m11, &mut self.m10);
        next.positive_det = !self.positive_det;
        swap(self, next);
    }
}

impl Gcd {
    /// Writes two positive scalar dot products in one traversal of the inputs.
    /// Each output has two carry guards. The first product plus the incoming
    /// carry fits `DoubleLimb`; only the second product can overflow it.
    #[expect(
        clippy::as_conversions,
        clippy::too_many_arguments,
        reason = "four scalar coefficients define two simultaneous products; validated disjoint destinations receive every limb before commit, and casts split exact limb halves"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "DoubleLimb narrows to Limb on 32-bit and 64-bit targets"
        )
    )]
    pub fn assign_linear_combinations(
        out_first: &mut InternalMpUint,
        out_second: &mut InternalMpUint,
        left: &InternalMpUint,
        right: &InternalMpUint,
        first_left: Limb,
        first_right: Limb,
        second_left: Limb,
        second_right: Limb,
    ) {
        let left_limbs = left.limbs();
        let right_limbs = right.limbs();
        let common = left_limbs.len().min(right_limbs.len());
        let width = left_limbs.len().max(right_limbs.len());
        // SAFETY: valid Limb slices occupy at most isize::MAX bytes, and
        // Limb has at least two bytes; two guard limbs therefore fit usize.
        let capacity = unsafe { width.unchecked_add(2) };
        let mut first_write = out_first.prepare_limb_write(capacity);
        let mut second_write = out_second.prepare_limb_write(capacity);
        let first = first_write.as_mut_ptr();
        let second = second_write.as_mut_ptr();
        let mut first_carry: DoubleLimb = 0;
        let mut second_carry: DoubleLimb = 0;
        for index in 0..common {
            // SAFETY: index < common bounds both initialized source reads.
            let (a, b) = unsafe {
                (
                    *left_limbs.get_unchecked(index) as DoubleLimb,
                    *right_limbs.get_unchecked(index) as DoubleLimb,
                )
            };
            // If B=2^LIMB_BITS, carry <= 2B-2. Thus
            // (B-1)^2 + carry <= B^2-1. The second sum is below 2B^2,
            // and its overflow bit becomes bit LIMB_BITS of the next carry.
            // SAFETY: a,b and each coefficient are below B; each product is
            // at most (B-1)^2. Incoming carries are at most 2B-2, so the first
            // addition is at most B^2-1 and fits DoubleLimb on every target.
            // The second addition retains its legitimate overflow bit.
            let ((first_sum, first_overflow), (second_sum, second_overflow)) = unsafe {
                (
                    a.unchecked_mul(first_left as DoubleLimb)
                        .unchecked_add(first_carry)
                        .overflowing_add(b.unchecked_mul(first_right as DoubleLimb)),
                    a.unchecked_mul(second_left as DoubleLimb)
                        .unchecked_add(second_carry)
                        .overflowing_add(b.unchecked_mul(second_right as DoubleLimb)),
                )
            };
            first_carry =
                (first_sum >> LIMB_BITS) | ((DoubleLimb::from(first_overflow)) << LIMB_BITS);
            second_carry =
                (second_sum >> LIMB_BITS) | ((DoubleLimb::from(second_overflow)) << LIMB_BITS);
            // SAFETY: distinct write guards each reserve width+2 limbs,
            // index < common <= width, and neither destination aliases a source.
            unsafe {
                *first.add(index) = first_sum as Limb;
                *second.add(index) = second_sum as Limb;
            }
        }
        let (longer, first_scalar, second_scalar) = if left_limbs.len() > right_limbs.len() {
            (left_limbs, first_left, second_left)
        } else {
            (right_limbs, first_right, second_right)
        };
        for index in common..width {
            // SAFETY: longer.len() == width bounds this initialized read.
            let value = unsafe { *longer.get_unchecked(index) } as DoubleLimb;
            // SAFETY: value and coefficients are below B and carries at most
            // 2B-2; each sum is at most (B-1)^2+2B-2 = B^2-1.
            let (first_sum, second_sum) = unsafe {
                (
                    value
                        .unchecked_mul(first_scalar as DoubleLimb)
                        .unchecked_add(first_carry),
                    value
                        .unchecked_mul(second_scalar as DoubleLimb)
                        .unchecked_add(second_carry),
                )
            };
            first_carry = first_sum >> LIMB_BITS;
            second_carry = second_sum >> LIMB_BITS;
            // SAFETY: index < width < capacity and destinations are disjoint.
            unsafe {
                *first.add(index) = first_sum as Limb;
                *second.add(index) = second_sum as Limb;
            }
        }
        // SAFETY: the loops wrote 0..width. These two guards complete both
        // capacity spans before commit; each carry is at most 2B-2.
        unsafe {
            *first.add(width) = first_carry as Limb;
            *first.add(width.unchecked_add(1)) = (first_carry >> LIMB_BITS) as Limb;
            *second.add(width) = second_carry as Limb;
            *second.add(width.unchecked_add(1)) = (second_carry >> LIMB_BITS) as Limb;
            let _ = first_write.commit();
            let _ = second_write.commit();
        }
        out_first.normalize();
        out_second.normalize();
    }

    /// Initializes one product, then accumulates the other into that destination.
    /// Two scalar products of n-limb values sum to less than 2*B^(n+1), so two
    /// guard limbs cover both multiplication carries and the final addition.
    pub fn assign_linear_combination(
        out: &mut InternalMpUint,
        left: &InternalMpUint,
        left_scalar: Limb,
        right: &InternalMpUint,
        right_scalar: Limb,
    ) {
        let left_limbs = left.limbs();
        let right_limbs = right.limbs();
        if left_scalar == 0 || left_limbs.is_empty() {
            assign_scaled(out, right_limbs, right_scalar);
            return;
        }
        if right_scalar == 0 || right_limbs.is_empty() {
            assign_scaled(out, left_limbs, left_scalar);
            return;
        }

        let (major_left, major_scalar, minor_right, minor_scalar) =
            if left_limbs.len() >= right_limbs.len() {
                (left_limbs, left_scalar, right_limbs, right_scalar)
            } else {
                (right_limbs, right_scalar, left_limbs, left_scalar)
            };

        let max_len = major_left.len();
        // SAFETY: max_len comes from valid Limb slices, so it is at most
        // isize::MAX / size_of::<Limb>(). Limb has at least two bytes; adding
        // two guard limbs stays below usize::MAX on 16-, 32-, and 64-bit targets.
        let out_len = unsafe { max_len.unchecked_add(2) };
        let mut write = out.prepare_limb_write(out_len);
        let dst_ptr = write.as_mut_ptr();
        // SAFETY: out_len >= major_left.len()+2; write guarantees out_len writable limbs.
        // mul_limbs_scalar_unchecked writes the product of major_left into dst_ptr,
        // the first guard receives the carry, the second guard is initialized to zero,
        // and write.commit exposes the initialized destination slice.
        let dst = unsafe {
            let carry = Self::mul_limbs_scalar_unchecked(
                dst_ptr,
                major_left.as_ptr(),
                max_len,
                major_scalar,
            );
            *dst_ptr.add(max_len) = carry;
            *dst_ptr.add(max_len.unchecked_add(1)) = 0;
            write.commit()
        };
        add_scaled(dst, minor_right, minor_scalar);
        out.normalize();
    }
}

fn assign_scaled_add(out: &mut InternalMpUint, major: &[Limb], scalar: Limb, minor: &[Limb]) {
    if scalar == 0 || major.is_empty() {
        out.clone_from_slice(minor);
        return;
    }
    if minor.is_empty() {
        assign_scaled(out, major, scalar);
        return;
    }
    let major_len = major.len();
    let minor_len = minor.len();
    let max_len = max(major_len, minor_len);
    // SAFETY: max_len is at most isize::MAX / size_of::<Limb>(); adding one guard limb fits usize.
    let out_len = unsafe { max_len.unchecked_add(1) };
    let mut write = out.prepare_limb_write(out_len);
    let dst_ptr = write.as_mut_ptr();
    // SAFETY: the guard reserves out_len aligned limbs disjoint from both
    // initialized inputs; minor_len,major_len <= max_len < out_len. The copy
    // and zero fill initialize the entire span before the accumulation reads it.
    unsafe {
        copy_nonoverlapping(minor.as_ptr(), dst_ptr, minor_len);
        if minor_len < out_len {
            write_bytes(dst_ptr.add(minor_len), 0, out_len.unchecked_sub(minor_len));
        }
        let carry =
            ArchKernels::add_mul_limbs_unchecked(dst_ptr, major.as_ptr(), major_len, scalar);
        let rem = out_len.unchecked_sub(major_len);
        if rem > 0 && carry != 0 {
            // SAFETY: rem > 0 proves dst_ptr.add(major_len) is within out_len initialized limbs.
            let (first_sum, overflow) = (*dst_ptr.add(major_len)).overflowing_add(carry);
            *dst_ptr.add(major_len) = first_sum;
            if overflow && rem > 1 {
                // SAFETY: rem > 1 proves dst_ptr.add(major_len + 1) has rem - 1 initialized limbs.
                let tail_overflow = ArchKernels::propagate_carry_unchecked(
                    dst_ptr.add(major_len.unchecked_add(1)),
                    rem.unchecked_sub(1),
                    1,
                );
                debug_assert_eq!(tail_overflow, 0, "alloc_len guarantees space for carry");
            }
        }
        let _ = write.commit();
    }
    out.normalize();
}

fn assign_scaled(out: &mut InternalMpUint, src: &[Limb], scalar: Limb) {
    if src.is_empty() || scalar == 0 {
        out.clear();
        return;
    }
    if scalar == 1 {
        out.clone_from_slice(src);
        return;
    }
    // SAFETY: a valid source slice occupies at most isize::MAX bytes, so its
    // limb count is strictly below usize::MAX and one guard cannot overflow.
    let out_len = unsafe { src.len().unchecked_add(1) };
    let mut write = out.prepare_limb_write(out_len);
    let dst_ptr = write.as_mut_ptr();
    // SAFETY: out_len == src.len()+1; the guard owns aligned storage
    // disjoint from src. mul_limbs_scalar_unchecked initializes the reserved span.
    unsafe {
        let carry = Gcd::mul_limbs_scalar_unchecked(dst_ptr, src.as_ptr(), src.len(), scalar);
        *dst_ptr.add(src.len()) = carry;
        let _ = write.commit();
    }
    out.normalize();
}

fn add_scaled(dst: &mut [Limb], src: &[Limb], scalar: Limb) {
    // SAFETY: the caller allocates max(left.len(), right.len())+2 initialized
    // limbs and src is one of those independent immutable matrix entries.
    // Both aligned spans remain alive and disjoint for the kernel call.
    let carry = unsafe {
        ArchKernels::add_mul_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src.len(), scalar)
    };
    if carry == 0 {
        return;
    }
    // SAFETY: dst has max(left.len(), right.len()) + 2 initialized limbs,
    // and src is the shorter source. At least the two guards remain here.
    let (first, rest) = unsafe {
        dst.get_unchecked_mut(src.len()..)
            .split_first_mut()
            .unwrap_unchecked()
    };
    let (sum, overflow) = first.overflowing_add(carry);
    *first = sum;
    if overflow {
        let tail_overflow = Addition::propagate_carry(rest, 1);
        debug_assert_eq!(
            tail_overflow, 0,
            "two product guards contain the complete matrix sum"
        );
    }
}
