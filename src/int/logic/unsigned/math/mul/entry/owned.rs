//! Allocating, destination-reusing, and in-place products on owned integers.
//!
//! Nonzero products reserve their complete width before execution. Normalized
//! factors produce either `m+n` or `m+n-1` limbs, so one guard check determines
//! the result length without scanning the magnitude.

#![expect(
    unsafe_code,
    reason = "Owned drivers reserve complete products before raw initialization and bound every stack workspace prefix"
)]

use core::{
    mem::MaybeUninit,
    ptr::{copy_nonoverlapping, eq},
    slice::{from_raw_parts, from_raw_parts_mut},
};

use alloc::{vec, vec::Vec};

use super::{
    ArchKernels, INLINE_LIMBS, InternalMpUint, KARATSUBA_THRESHOLD, Limb, MulScratch,
    Multiplication, SQR_KARATSUBA_THRESHOLD, Schoolbook, ScratchBuffer, UintRepr,
};

impl InternalMpUint {
    /// Computes `self * other`.
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        if eq(self, other) {
            return self.square();
        }
        let a_limbs = self.limbs();
        let b_limbs = other.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();
        if a_len == 0 || b_len == 0 {
            return Self::zero();
        }
        // SAFETY: the exact length test proves index zero exists.
        if a_len == 1 && unsafe { *a_limbs.get_unchecked(0) == 1 } {
            return other.clone();
        }
        // SAFETY: the exact length test proves index zero exists.
        if b_len == 1 && unsafe { *b_limbs.get_unchecked(0) == 1 } {
            return self.clone();
        }
        // SAFETY: both valid slices span at most isize::MAX bytes, and Limb
        // occupies at least two bytes on 16/32/64-bit targets. Their combined
        // limb count is at most isize::MAX < usize::MAX.
        let result_len = unsafe { a_len.unchecked_add(b_len) };
        if result_len <= INLINE_LIMBS {
            let mut result = Self::zero();
            result.write_nonzero_product(a_limbs, b_limbs);
            return result;
        }

        // This fresh owned result has one known representation and exact
        // capacity. Constructing it directly avoids sending the allocation
        // through the general inline-to-heap growth state machine.
        if a_len < KARATSUBA_THRESHOLD || b_len < KARATSUBA_THRESHOLD {
            let mut product = Vec::with_capacity(result_len);
            // SAFETY: spare capacity covers the proven complete width. The
            // initialized nonempty operands are disjoint from this allocation;
            // every output limb is written before reading the guard. The
            // product has result_len or result_len-1 limbs, so the binary guard
            // test gives its exact positive length with one metadata commit.
            return unsafe {
                initialize_basecase_product(product.as_mut_ptr(), a_limbs, b_limbs);
                let guard = *product.as_ptr().add(result_len.unchecked_sub(1));
                product.set_len(result_len.unchecked_sub(usize::from(guard == 0)));
                Self::from_limbs_normalized(product)
            };
        }
        let mut limbs = vec![0; result_len];
        Multiplication::multiply_nonzero_owned(a_limbs, b_limbs, &mut limbs);
        // SAFETY: multiplication initialized the complete result_len-limb
        // vector; both factors are nonempty, so result_len >= 2.
        let top_is_zero = unsafe { *limbs.get_unchecked(result_len.unchecked_sub(1)) == 0 };
        // A normalized nonzero m-by-n product has at least m+n-1 limbs, so at
        // most the single guard limb can be removed.
        if top_is_zero {
            // SAFETY: the product lower bound proves result_len-1 is its
            // nonempty normalized prefix, wholly inside the initialized vector.
            unsafe {
                limbs.set_len(result_len.unchecked_sub(1));
            }
        }
        // The complete result already owns a heap allocation. Its normalized
        // prefix remains valid heap storage at either possible product width;
        // retaining it avoids an inline copy even with reduced tuning thresholds.
        Self {
            repr: UintRepr::Heap(limbs),
        }
    }

    /// Computes the square `self * self`, avoiding the redundant work of a
    /// general product.
    pub fn square(&self) -> Self {
        let a_len = self.limbs().len();
        if a_len == 0 {
            return Self::zero();
        }
        // SAFETY: a valid limb slice has at most isize::MAX/size_of::<Limb>()
        // elements; Limb occupies at least two bytes on every supported target.
        let result_len = unsafe { a_len.unchecked_mul(2) };
        if result_len > INLINE_LIMBS && a_len < SQR_KARATSUBA_THRESHOLD {
            let mut limbs = Vec::with_capacity(result_len);
            // SAFETY: a_len > 0 and the fresh allocation covers the proven
            // 2*a_len width. The raw square initializes every output limb
            // before reading the guard. Its exact normalized length is 2n or
            // 2n-1, permitting one positive length commit after the binary test.
            return unsafe {
                ArchKernels::sqr_basecase_unchecked(
                    limbs.as_mut_ptr(),
                    self.limbs().as_ptr(),
                    a_len,
                );
                let guard = *limbs.as_ptr().add(result_len.unchecked_sub(1));
                limbs.set_len(result_len.unchecked_sub(usize::from(guard == 0)));
                // This branch has n >= 3. The normalized square has at least
                // 2n-1 >= 5 > INLINE_LIMBS limbs, so no inline conversion exists.
                Self {
                    repr: UintRepr::Heap(limbs),
                }
            };
        }
        let mut res = Self::with_capacity(result_len);
        res.write_nonzero_square(self.limbs());
        res
    }

    /// Computes `self = a * b`, reusing `self`'s existing allocation.
    ///
    /// Rust's exclusive destination borrow prevents aliasing either operand.
    #[inline]
    pub fn assign_product(&mut self, a: &Self, b: &Self) {
        if eq(a, b) {
            self.assign_square(a);
            return;
        }
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();
        if a_len > 1 && b_len > 1 {
            self.write_nonzero_product(a_limbs, b_limbs);
            return;
        }
        if a_len == 0 || b_len == 0 {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }
        // SAFETY: the exact length test proves index zero exists.
        if a_len == 1 && unsafe { *a_limbs.get_unchecked(0) == 1 } {
            self.clone_from(b);
            return;
        }
        // SAFETY: the exact length test proves index zero exists.
        if b_len == 1 && unsafe { *b_limbs.get_unchecked(0) == 1 } {
            self.clone_from(a);
            return;
        }
        self.write_nonzero_product(a_limbs, b_limbs);
    }

    /// Computes `self = a * a`, reusing `self`'s existing allocation.
    ///
    /// Rust's exclusive destination borrow prevents aliasing the operand.
    pub fn assign_square(&mut self, a: &Self) {
        let a_limbs = a.limbs();
        if a_limbs.is_empty() {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }
        self.write_nonzero_square(a_limbs);
    }

    /// Computes `self = a * b` using a caller-owned scratch pool.
    pub fn assign_product_with_scratch(&mut self, a: &Self, b: &Self, scratch: &mut MulScratch) {
        if eq(a, b) {
            self.assign_square_with_scratch(a, scratch);
            return;
        }
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        if a_limbs.is_empty() || b_limbs.is_empty() {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }

        // SAFETY: each valid slice spans at most isize::MAX bytes and Limb >= 2
        // bytes, so their combined limb count is at most isize::MAX.
        let res_len = unsafe { a_limbs.len().unchecked_add(b_limbs.len()) };
        let mut pending = self.prepare_limb_write(res_len);
        // SAFETY: preparation reserves res_len native-aligned elements, and
        // MaybeUninit<Limb> permits the not-yet-initialized capacity. Normalized
        // nonempty operands have distinct owners and are disjoint from self.
        // The initializer returns only after writing every product element.
        let top_is_zero = unsafe {
            let output = from_raw_parts_mut(pending.as_mut_ptr().cast(), res_len);
            let result = Multiplication::mul_nonempty_distinct_into_uninit(
                a_limbs, b_limbs, output, scratch,
            );
            *result.get_unchecked(res_len.unchecked_sub(1)) == 0
        };
        // SAFETY: normalized nonzero m-by-n factors give m+n or m+n-1
        // initialized limbs, so the binary guard test permits one length commit.
        unsafe { self.set_len(res_len.unchecked_sub(usize::from(top_is_zero))) }
    }

    /// Computes `self = a * a` using a caller-owned scratch pool.
    pub fn assign_square_with_scratch(&mut self, a: &Self, scratch: &mut MulScratch) {
        let a_limbs = a.limbs();
        if a_limbs.is_empty() {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }

        // SAFETY: the valid slice byte bound and Limb >= 2 bytes prove
        // twice this limb count is at most isize::MAX < usize::MAX.
        let res_len = unsafe { a_limbs.len().unchecked_mul(2) };
        // SAFETY: the limb dispatcher fills the complete result slice before it is read.
        let result = self.ensure_capacity_set_len_get_limbs(res_len);
        Multiplication::sqr_limbs_with_scratch(a_limbs, result, scratch);
        self.trim_product_guard(res_len);
    }

    /// Multiplies in place by `other`.
    #[inline]
    pub fn mul_assign(&mut self, other: &Self) {
        let a_len = self.limbs().len();
        let b_len = other.limbs().len();
        if a_len == 0 || b_len == 0 {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }
        if other.is_one() {
            return;
        }
        if self.is_one() {
            self.clone_from(other);
            return;
        }
        if b_len == 1 {
            // The one-limb multiplier is independent of the destination. Growing
            // by one preserves the initialized prefix; the scalar kernel may then
            // overwrite that prefix in place because it consumes limbs low to high.
            // SAFETY: this branch proves b_len == 1.
            let scalar = unsafe { *other.limbs().get_unchecked(0) };
            // SAFETY: a_len <= isize::MAX/size_of::<Limb>() and Limb >= 2 bytes,
            // so a_len+1 < usize::MAX on every supported target.
            let res_len = unsafe { a_len.unchecked_add(1) };
            let mut pending = self.prepare_limb_write(res_len);
            let result = pending.as_mut_ptr();
            // SAFETY: preparation preserves the initialized a_len-limb prefix
            // and reserves a_len+1 writable limbs. The raw kernel supports exact
            // aliasing and initializes the complete product before reading its
            // guard. Nonzero normalized factors give res_len or res_len-1 limbs;
            // res_len >= 2 bounds both subtractions and the sole length commit.
            unsafe {
                Schoolbook::mul_limb_unchecked(result, result, a_len, scalar);
                let guard = *result.add(res_len.unchecked_sub(1));
                self.set_len(res_len.unchecked_sub(usize::from(guard == 0)));
            }
            return;
        }
        if a_len == 1 {
            // SAFETY: this branch proves a_len == 1.
            let scalar = unsafe { *self.limbs().get_unchecked(0) };
            // SAFETY: the valid source slice byte bound with Limb >= 2 bytes
            // proves b_len+1 < usize::MAX on every supported target.
            let res_len = unsafe { b_len.unchecked_add(1) };
            let mut pending = self.prepare_limb_write(res_len);
            let result = pending.as_mut_ptr();
            // SAFETY: preparation reserves b_len+1 writable limbs, and the
            // separately borrowed source contains b_len initialized limbs. The
            // kernel initializes every limb before reading the guard. Nonzero
            // normalized factors give res_len or res_len-1 limbs; res_len >= 2
            // bounds both subtractions and the sole positive length commit.
            unsafe {
                Schoolbook::mul_limb_unchecked(result, other.limbs().as_ptr(), b_len, scalar);
                let guard = *result.add(res_len.unchecked_sub(1));
                self.set_len(res_len.unchecked_sub(usize::from(guard == 0)));
            }
            return;
        }
        // The destination is also an operand, so the multiplicand has to be
        // preserved before `result` overwrites it. Narrow values stage through an
        // inline array; wider ones through a pooled buffer.
        // SAFETY: each valid limb slice spans at most isize::MAX bytes and
        // Limb >= 2 bytes, so this combined count is at most isize::MAX.
        let res_len = unsafe { a_len.unchecked_add(b_len) };
        if a_len <= INLINE_LIMBS {
            let mut saved_a = [MaybeUninit::<Limb>::uninit(); INLINE_LIMBS];
            // SAFETY: a_len <= INLINE_LIMBS and self's a_len-limb prefix is
            // initialized. MaybeUninit<Limb> has Limb's alignment and layout;
            // this stack allocation is disjoint from the source representation.
            // Only the copied prefix is subsequently exposed as initialized limbs.
            let slice_a = unsafe {
                copy_nonoverlapping(
                    self.limbs().as_ptr(),
                    saved_a.as_mut_ptr().cast::<Limb>(),
                    a_len,
                );
                from_raw_parts(saved_a.as_ptr().cast::<Limb>(), a_len)
            };
            // SAFETY: the limb dispatcher fills the complete result slice before it is read.
            let result = self.ensure_capacity_set_len_get_limbs(res_len);
            Multiplication::multiply_nonzero_owned(slice_a, other.limbs(), result);
        } else {
            let mut saved_a = ScratchBuffer::acquire(a_len);
            // SAFETY: arena acquisition returns empty, exclusive storage with
            // capacity >= a_len. The live multiplicand has a_len initialized
            // limbs in a disjoint allocation; both pointers are aligned. The
            // copy initializes the entire prefix before committing its length.
            unsafe {
                copy_nonoverlapping(self.limbs().as_ptr(), saved_a.as_mut_ptr(), a_len);
                saved_a.set_len(a_len);
            }
            // SAFETY: the limb dispatcher fills the complete result slice before it is read.
            let result = self.ensure_capacity_set_len_get_limbs(res_len);
            Multiplication::multiply_nonzero_owned(&saved_a, other.limbs(), result);
        }
        self.trim_product_guard(res_len);
    }

    /// Consumes both operands, multiplying into whichever already owns the larger
    /// allocation.
    #[inline]
    pub fn mul_into(mut self, mut other: Self) -> Self {
        let reuse_other = match (&self.repr, &other.repr) {
            (UintRepr::Inline { .. }, UintRepr::Inline { .. }) => false,
            (UintRepr::Inline { .. }, UintRepr::Heap(right)) => right.capacity() > INLINE_LIMBS,
            (UintRepr::Heap(left), UintRepr::Inline { .. }) => left.capacity() < INLINE_LIMBS,
            (UintRepr::Heap(left), UintRepr::Heap(right)) => right.capacity() > left.capacity(),
        };
        if reuse_other {
            other.mul_assign(&self);
            other
        } else {
            self.mul_assign(&other);
            self
        }
    }

    /// Writes a nonzero product.
    ///
    /// Nonzero normalized `m`- and `n`-limb operands have a product of at least
    /// `m + n - 1` limbs, so only the highest allocated limb can be zero.
    #[inline]
    fn write_nonzero_product(&mut self, a_limbs: &[Limb], b_limbs: &[Limb]) {
        // SAFETY: valid slice byte bounds and Limb >= 2 bytes prove the
        // combined limb count is at most isize::MAX on every supported target.
        let result_len = unsafe { a_limbs.len().unchecked_add(b_limbs.len()) };
        if a_limbs.len() < KARATSUBA_THRESHOLD || b_limbs.len() < KARATSUBA_THRESHOLD {
            let mut pending = self.prepare_limb_write(result_len);
            let result = pending.as_mut_ptr();
            // SAFETY: preparation reserves result_len writable limbs without
            // exposing uninitialized storage as a slice. Both normalized inputs
            // are nonempty and disjoint; the raw basecase initializes every limb.
            // Their product has result_len or result_len-1 limbs, so the guard
            // gives its exact positive length. result_len >= 2 bounds the read
            // and both subtractions before the sole length commit.
            unsafe {
                initialize_basecase_product(result, a_limbs, b_limbs);
                let guard = *result.add(result_len.unchecked_sub(1));
                self.set_len(result_len.unchecked_sub(usize::from(guard == 0)));
            }
            return;
        }
        if let Some(pending) = self.try_prepare_reused_heap_limbs(result_len) {
            let limbs = pending.initialize_suffix_with_zeroes();
            Multiplication::multiply_nonzero_owned(a_limbs, b_limbs, limbs);
            self.trim_product_guard(result_len);
            return;
        }
        let result = self.ensure_capacity_set_len_get_limbs(result_len);
        Multiplication::multiply_nonzero_owned(a_limbs, b_limbs, result);
        self.trim_product_guard(result_len);
    }

    /// Writes a nonzero square.
    ///
    /// A nonzero normalized `n`-limb value has a square of `2n` or `2n - 1`
    /// limbs, so at most one high guard limb can be zero.
    fn write_nonzero_square(&mut self, a_limbs: &[Limb]) {
        // SAFETY: the valid slice byte bound and Limb >= 2 bytes prove
        // twice this count is at most isize::MAX on every supported target.
        let result_len = unsafe { a_limbs.len().unchecked_mul(2) };
        if a_limbs.len() < SQR_KARATSUBA_THRESHOLD {
            let mut pending = self.prepare_limb_write(result_len);
            let result = pending.as_mut_ptr();
            // SAFETY: preparation reserves 2*n writable limbs, disjoint from
            // the nonempty source. The square kernel initializes every limb
            // before reading the guard. The normalized square has 2n or 2n-1
            // limbs; result_len >= 2 bounds both subtractions and the read,
            // yielding a positive initialized prefix for the sole length commit.
            unsafe {
                ArchKernels::sqr_basecase_unchecked(result, a_limbs.as_ptr(), a_limbs.len());
                let guard = *result.add(result_len.unchecked_sub(1));
                self.set_len(result_len.unchecked_sub(usize::from(guard == 0)));
            }
            return;
        }
        if let Some(pending) = self.try_prepare_reused_heap_limbs(result_len) {
            let result = pending.initialize_suffix_with_zeroes();
            Multiplication::square_nonzero_owned(a_limbs, result);
            self.trim_product_guard(result_len);
            return;
        }
        let result = self.ensure_capacity_set_len_get_limbs(result_len);
        Multiplication::square_nonzero_owned(a_limbs, result);
        self.trim_product_guard(result_len);
    }

    /// Drops the single guard limb a product or square may leave zero.
    ///
    /// The nonzero-product lower bound limits normalization to this one limb.
    #[inline]
    fn trim_product_guard(&mut self, result_len: usize) {
        // SAFETY: every caller has initialized and committed the exact
        // result_len-limb product or square of nonzero inputs. Thus result_len
        // >= 2 and result_len-1 is a valid initialized index.
        let top_is_zero = unsafe { *self.limbs().get_unchecked(result_len.unchecked_sub(1)) == 0 };
        if top_is_zero {
            // SAFETY: both nonzero factors have at least one limb, so
            // result_len >= 2. The product lower bound proves result_len-1
            // is a nonzero normalized prefix; the existing full length is
            // already correct when no high zero is present.
            unsafe {
                self.set_len(result_len.unchecked_sub(1));
            }
        }
    }
}

/// Initializes a raw complete product without requiring a typed destination slice.
///
/// # Safety
///
/// Both inputs are nonempty and disjoint from `dst`, which is aligned and
/// writable for their combined lengths. The inputs may overlap one another.
#[inline]
unsafe fn initialize_basecase_product(dst: *mut Limb, a: &[Limb], b: &[Limb]) {
    let (outer, inner) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    // SAFETY: the nonempty shorter input supplies the scalar or at least two
    // rows. The scalar kernel and paired-row basecase write every output limb
    // before accumulation can read it, throughout the caller's reserved span.
    unsafe {
        if outer.len() == 1 {
            Schoolbook::mul_limb_unchecked(
                dst,
                inner.as_ptr(),
                inner.len(),
                *outer.get_unchecked(0),
            );
        } else {
            ArchKernels::mul_basecase_unchecked(
                dst,
                outer.as_ptr(),
                outer.len(),
                inner.as_ptr(),
                inner.len(),
            );
        }
    }
}
