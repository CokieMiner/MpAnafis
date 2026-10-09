//! Half-GCD transition-matrix state and its exact reduction application.

#![expect(
    unsafe_code,
    reason = "matrix partitions bound source prefixes and materialized product sizes; write guards initialize outputs before exposure"
)]

use core::{
    cmp::{max, min},
    mem::{MaybeUninit, swap},
    ptr::{copy_nonoverlapping, write_bytes},
    slice::from_raw_parts_mut,
};

use super::{Addition, DivScratch, InternalMpUint, Limb, Multiplication};

/// A non-negative 2x2 Euclidean transition matrix.
///
/// If `M` is this matrix and `(a, b)` is the reduced pair, the input pair is
/// `M * (a, b)`. Euclidean quotient matrices have determinant `-1`, so the
/// sign of the accumulated determinant is tracked separately.
#[derive(Debug, Eq, PartialEq)]
pub struct HgcdMatrix {
    pub m00: InternalMpUint,
    pub m01: InternalMpUint,
    pub m10: InternalMpUint,
    pub m11: InternalMpUint,
    pub positive_det: bool,
}

impl Default for HgcdMatrix {
    fn default() -> Self {
        Self {
            m00: InternalMpUint::one(),
            m01: InternalMpUint::zero(),
            m10: InternalMpUint::zero(),
            m11: InternalMpUint::one(),
            positive_det: true,
        }
    }
}

impl HgcdMatrix {
    pub fn reset(&mut self) {
        self.m00.set_limb(1);
        self.m01.clear();
        self.m10.clear();
        self.m11.set_limb(1);
        self.positive_det = true;
    }

    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.m00.capacity() < capacity {
            self.m00
                .reserve(capacity.saturating_sub(self.m00.limbs().len()));
        }
        if self.m01.capacity() < capacity {
            self.m01
                .reserve(capacity.saturating_sub(self.m01.limbs().len()));
        }
        if self.m10.capacity() < capacity {
            self.m10
                .reserve(capacity.saturating_sub(self.m10.limbs().len()));
        }
        if self.m11.capacity() < capacity {
            self.m11
                .reserve(capacity.saturating_sub(self.m11.limbs().len()));
        }
    }

    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.positive_det
            && self.m00.is_one()
            && self.m01.is_zero()
            && self.m10.is_zero()
            && self.m11.is_one()
    }

    /// Exchanges the current reduced pair while preserving `input = M * pair`.
    pub const fn swap_columns(&mut self) {
        swap(&mut self.m00, &mut self.m01);
        swap(&mut self.m10, &mut self.m11);
        self.positive_det = !self.positive_det;
    }

    /// Right-multiplies this accumulated matrix by another HGCD matrix.
    ///
    /// Both inputs remain immutable until all four output entries have been
    /// formed in `next`. The two additional values are reusable scratch owners;
    /// multiplication's deeper workspace comes from the shared division pool.
    pub fn multiply_right(
        &mut self,
        rhs: &Self,
        next: &mut Self,
        product_a: &mut InternalMpUint,
        product_b: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        if rhs.is_identity() {
            return;
        }
        Self::assign_product_slice_two_by_one(
            &mut next.m00,
            &mut next.m10,
            &self.m00,
            &self.m10,
            rhs.m00.limbs(),
            scratch,
        );
        Self::assign_product_slice_two_by_one(
            product_a,
            product_b,
            &self.m01,
            &self.m11,
            rhs.m10.limbs(),
            scratch,
        );
        next.m00.add_assign(product_a);
        next.m10.add_assign(product_b);

        Self::assign_product_slice_two_by_one(
            &mut next.m01,
            &mut next.m11,
            &self.m00,
            &self.m10,
            rhs.m01.limbs(),
            scratch,
        );
        Self::assign_product_slice_two_by_one(
            product_a,
            product_b,
            &self.m01,
            &self.m11,
            rhs.m11.limbs(),
            scratch,
        );
        next.m01.add_assign(product_a);
        next.m11.add_assign(product_b);

        next.positive_det = self.positive_det == rhs.positive_det;
        swap(self, next);
    }

    /// Reconstructs the reduced pair into `next_u`/`next_v` from borrowed sources.
    ///
    /// This implements Möller's `mpn_hgcd_matrix_adjust` algorithm:
    /// `(u', v') = (high_u', high_v') * B^p + M^-1 * (u_low, v_low)`, with the
    /// multiplication strictly on `p` limbs instead of full operand length.
    /// The caller moves the outputs into place on success. Since the inputs
    /// are only borrowed, the recursive driver never copies the input pair
    /// and only clones when the child makes no progress or the
    /// reconstruction is rejected.
    /// The partition `p` is at most the larger source's limb count.
    #[expect(
        clippy::too_many_arguments,
        reason = "borrowed vector adjustment carries partition point, high parts, four scratch buffers, and division scratch pool."
    )]
    pub fn adjust_vector_into(
        &self,
        u_limbs: &[Limb],
        v_limbs: &[Limb],
        p: usize,
        high_u: &[Limb],
        high_v: &[Limb],
        next_u: &mut InternalMpUint,
        next_v: &mut InternalMpUint,
        product_a: &mut InternalMpUint,
        product_b: &mut InternalMpUint,
        sum_a: &mut InternalMpUint,
        sum_b: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) -> bool {
        let u_len = u_limbs.len();
        let v_len = v_limbs.len();
        debug_assert!(
            p <= max(u_len, v_len),
            "HGCD partition exceeds the source pair"
        );
        // SAFETY: min(p, u_len) <= u_len by construction.
        let u_low = unsafe { u_limbs.get_unchecked(..min(p, u_len)) };
        // SAFETY: min(p, v_len) <= v_len by construction.
        let v_low = unsafe { v_limbs.get_unchecked(..min(p, v_len)) };

        // Compute t0 = m11 * u_low, t1 = m10 * u_low into product_a, product_b
        Self::assign_product_slice_two_by_one(
            product_a, product_b, &self.m11, &self.m10, u_low, scratch,
        );
        // Compute s0 = m01 * v_low, s1 = m00 * v_low into sum_a, sum_b
        Self::assign_product_slice_two_by_one(sum_a, sum_b, &self.m01, &self.m00, v_low, scratch);

        let (add_u, sub_u) = if self.positive_det {
            (&*product_a, &*sum_a)
        } else {
            (&*sum_a, &*product_a)
        };
        if !assemble_adjusted_component(next_u, p, high_u, add_u, sub_u) {
            return false;
        }

        let (add_v, sub_v) = if self.positive_det {
            (&*sum_b, &*product_b)
        } else {
            (&*product_b, &*sum_b)
        };
        assemble_adjusted_component(next_v, p, high_v, add_v, sub_v)
    }

    /// Writes the pair of products `(matrix_entry_a * low, matrix_entry_b * low)`.
    ///
    /// The two products share the operand `low`, so they are formed by a single
    /// paired kernel rather than two independent multiplications.
    pub fn assign_product_slice_two_by_one(
        out_a: &mut InternalMpUint,
        out_b: &mut InternalMpUint,
        matrix_entry_a: &InternalMpUint,
        matrix_entry_b: &InternalMpUint,
        low: &[Limb],
        scratch: &mut DivScratch,
    ) {
        let m_a = matrix_entry_a.limbs();
        let m_b = matrix_entry_b.limbs();
        if low.is_empty() || (m_a.is_empty() && m_b.is_empty()) {
            out_a.clear();
            out_b.clear();
            return;
        }

        // HGCD matrices are sparse while the transition is being built, so one
        // entry is often zero. When it is, the corresponding product is exactly
        // zero and the other is a single ordinary multiplication. Taking that
        // path avoids deriving a second product length, preparing a destination
        // that would only be cleared, and running the paired kernel for one live
        // operand.
        if m_a.is_empty() || m_b.is_empty() {
            let (zeroed, live, entry) = if m_a.is_empty() {
                (out_a, out_b, m_b)
            } else {
                (out_b, out_a, m_a)
            };
            zeroed.clear();
            // SAFETY: each valid Limb slice occupies at most isize::MAX bytes;
            // their sum fits usize on every supported pointer width.
            let product_len = unsafe { entry.len().unchecked_add(low.len()) };
            let mut write = live.prepare_limb_write(product_len);
            // SAFETY: the branch supplies two nonempty initialized operands;
            // the write guard reserves their exact product width disjoint from
            // both. Multiplication initializes every limb before commit.
            unsafe {
                let _ = Multiplication::mul_nonempty_distinct_into_uninit(
                    entry,
                    low,
                    from_raw_parts_mut(write.as_mut_ptr().cast::<MaybeUninit<Limb>>(), product_len),
                    &mut scratch.mul_scratch,
                );
                let _ = write.commit();
            }
            live.normalize();
            return;
        }

        // SAFETY: each valid Limb slice occupies at most isize::MAX bytes; their sum fits usize.
        let product_len_a = unsafe { m_a.len().unchecked_add(low.len()) };
        let dst_a = out_a.ensure_capacity_set_len_get_limbs(product_len_a);
        // SAFETY: each valid Limb slice occupies at most isize::MAX bytes; their sum fits usize.
        let product_len_b = unsafe { m_b.len().unchecked_add(low.len()) };
        let dst_b = out_b.ensure_capacity_set_len_get_limbs(product_len_b);

        // The slice-based multiplication interface requires initialized limbs
        // on entry. Only newly exposed suffixes need initialization; reused
        // prefixes already contain valid limbs and are overwritten directly.
        Multiplication::mul_two_by_one(m_a, m_b, low, dst_a, dst_b, &mut scratch.mul_scratch);
        out_a.normalize();
        out_b.normalize();
    }
}

fn assemble_adjusted_component(
    dst: &mut InternalMpUint,
    p: usize,
    high: &[Limb],
    add: &InternalMpUint,
    sub: &InternalMpUint,
) -> bool {
    let add_limbs = add.limbs();
    let sub_limbs = sub.limbs();
    let upper_add_len = add_limbs.len().saturating_sub(p);
    let high_len = high.len();
    let max_high = max(high_len, upper_add_len);

    // SAFETY: adjust_vector_into receives p bounded by a materialized source
    // length. Each high/add slice also occupies at most isize::MAX bytes, with
    // at least two bytes per limb. Thus p+max_high+2 <= isize::MAX+2 < usize::MAX.
    let mut alloc_len = unsafe { p.unchecked_add(max_high).unchecked_add(2) };
    if alloc_len <= sub_limbs.len() {
        // SAFETY: the materialized Limb slice occupies at most isize::MAX
        // bytes, with at least two bytes per limb; one guard fits usize.
        alloc_len = unsafe { sub_limbs.len().unchecked_add(1) };
    }

    let mut write = dst.prepare_limb_write(alloc_len);
    let dst_ptr = write.as_mut_ptr();

    // SAFETY: the guard reserves alloc_len aligned writable limbs, disjoint
    // from the initialized sources. The bounded sizing establishes
    // p + max(high_len, upper_add_len) + 2 <= alloc_len and sub_len < alloc_len;
    // the copies and zero fills initialize every limb before reads or commit.
    unsafe {
        let low_copy = min(p, add_limbs.len());
        if low_copy > 0 {
            copy_nonoverlapping(add_limbs.as_ptr(), dst_ptr, low_copy);
        }
        if p > low_copy {
            write_bytes(dst_ptr.add(low_copy), 0, p.unchecked_sub(low_copy));
        }

        if high_len > 0 {
            copy_nonoverlapping(high.as_ptr(), dst_ptr.add(p), high_len);
        }
        let after_high = p.unchecked_add(high_len);
        if alloc_len > after_high {
            write_bytes(
                dst_ptr.add(after_high),
                0,
                alloc_len.unchecked_sub(after_high),
            );
        }

        let dst_slice = from_raw_parts_mut(dst_ptr, alloc_len);

        if upper_add_len > 0 {
            // SAFETY: alloc_len > p + upper_add_len and add_limbs.len() >= p + upper_add_len by construction.
            let dst_upper = dst_slice.get_unchecked_mut(p..p.unchecked_add(upper_add_len));
            let add_upper = add_limbs.get_unchecked(p..);
            let carry = Addition::add_slice_in_place(dst_upper, add_upper);
            // SAFETY: alloc_len > p + upper_add_len guarantees space for carry propagation.
            let dst_carry = dst_slice.get_unchecked_mut(p.unchecked_add(upper_add_len)..);
            let overflow = Addition::propagate_carry(dst_carry, carry);
            debug_assert_eq!(overflow, 0, "alloc_len guarantees space for carry");
        }

        if !sub_limbs.is_empty() {
            let sub_len = sub_limbs.len();
            // SAFETY: alloc_len > sub_len by construction of alloc_len.
            let dst_sub = dst_slice.get_unchecked_mut(..sub_len);
            let borrow = Addition::sub_slice_in_place(dst_sub, sub_limbs);
            // SAFETY: alloc_len > sub_len guarantees the tail exists.
            let dst_borrow = dst_slice.get_unchecked_mut(sub_len..);
            let underflow = Addition::propagate_borrow(dst_borrow, borrow);
            if underflow != 0 {
                return false;
            }
        }

        let _ = write.commit();
    }
    dst.normalize();
    true
}
