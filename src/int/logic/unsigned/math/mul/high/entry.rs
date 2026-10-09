//! Exact high products with certified carries from omitted low diagonals.
//!
//! Write P=T+E, where T retains terms with i+j>=c=skip-2 and E contains
//! the omitted terms. For r=min(c,m,n), every omitted column has at most r
//! products, so E<=r*(B-1)^2*sum(B^j,j<c)<r*(B-1)*B^c<r*B^(c+1).
//! T/B^c is an integer. Adding E can carry at most r into its second digit;
//! a second digit <=B-1-r certifies every digit at and above skip. Otherwise
//! compute E's two high digits and add them to T/B^c. Both paths are exact;
//! an ambiguous carry computes each scalar product once across T and E.
//!
//! Recursive products omit the low block and low cross-product prefixes.
//! Two guard digits bound every recursive high approximation within one
//! unit; the root's retained guards certify the exact high result.

#![expect(
    unsafe_code,
    reason = "validated diagonal widths establish first-write initialization, disjoint kernel spans and exact high-carry reconstruction"
)]

use core::{mem::MaybeUninit, slice::from_raw_parts_mut};

use super::{
    Addition, ArchKernels, KARATSUBA_THRESHOLD, Limb, MulScratch, Multiplication, ScratchBuffer,
    Widths,
};

/// Namespace for retained high products and their certification workspaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HighProduct;

impl HighProduct {
    /// Recursive crossover with the six-limb certification minimum applied.
    /// The full-block Karatsuba crossover remains the empirical input.
    pub const RECURSIVE_THRESHOLD: usize = Widths::new(KARATSUBA_THRESHOLD, 6).larger;

    /// Returns `floor(a*b/B^skip)` with one reserved high carry slot.
    ///
    /// Nonempty initialized operands and both workspaces are pairwise disjoint;
    /// `skip<a.len()+b.len()`. Schoolbook widths with skip>=3 omit the low
    /// triangle and certify its carry using two retained low digits. Wider
    /// products with `skip>=max(a.len(),b.len())` omit low blocks recursively;
    /// narrow retained suffixes use a full product of shortened input suffixes.
    /// Partial products certify the omitted carry before exposing the exact result.
    /// Unconsumed prefixes remain workspace; the returned suffix is exact.
    /// The carry slot remains outside `output.len()` until the consumer writes it.
    pub fn mul<'output>(
        a: &[Limb],
        b: &[Limb],
        skip: usize,
        output: &'output mut ScratchBuffer,
        carry_product: &mut ScratchBuffer,
        mul_scratch: &mut MulScratch,
    ) -> &'output mut [Limb] {
        debug_assert!(
            !a.is_empty() && !b.is_empty(),
            "the high product receives two nonempty operands"
        );
        // SAFETY: each materialized limb slice occupies at most isize::MAX
        // bytes, with at least two bytes per limb. Their sum plus one fits usize.
        let total = unsafe { a.len().unchecked_add(b.len()) };
        debug_assert!(skip < total, "the retained high-product suffix is nonempty");
        let smaller = a.len().min(b.len());
        let larger = a.len().max(b.len());
        // SAFETY: skip<total proves the retained suffix has positive width.
        let retained = unsafe { total.unchecked_sub(skip) };
        // retained>=1 and retained<floor(larger/2) imply larger>=4 and
        // skip=smaller+larger-retained>=4, supplying both guard digits.
        if smaller >= KARATSUBA_THRESHOLD && retained < larger.div_euclid(2) {
            return Self::trimmed_high_product(a, b, skip, total, output, mul_scratch);
        }
        if smaller >= Self::RECURSIVE_THRESHOLD && skip >= a.len().max(b.len()) {
            // At depth j a child has at most 2*floor(smaller/3^j)+2j
            // digits. Summing its geometric widths gives <smaller. For d
            // levels, smaller>=6*3^(d-1) gives d*(d+1)<=smaller, by induction.
            // Therefore 2*smaller covers every sequential child/descendant
            // arena without a sizing traversal or any scratch initialization.
            // Each input occupies at most isize::MAX bytes, with at least two
            // bytes per limb. Both limb counts fit on every target.
            let work_len = Self::scratch_len(if a.len() <= b.len() { a } else { b });
            // SAFETY: the two slice byte bounds prove total<=isize::MAX;
            // its one carry slot therefore fits usize on every pointer width.
            let capacity = unsafe { total.unchecked_add(1) };
            output.reset_with_capacity(capacity);
            carry_product.reset_with_capacity(work_len);
            // SAFETY: the root reserves total output limbs and the proved
            // maximum child/descendant arena. All four owners are disjoint.
            let (mut start, mut end) = unsafe {
                Self::high_product_blocks(
                    a,
                    b,
                    skip,
                    output.spare_capacity_mut().get_unchecked_mut(..total),
                    carry_product
                        .spare_capacity_mut()
                        .get_unchecked_mut(..work_len),
                    mul_scratch,
                )
            };
            // SAFETY: the recursive root retains two initialized guard digits
            // immediately before start. Its omitted contribution is at most
            // seven units; overflow from both guards is the only ambiguity.
            let ambiguous = unsafe {
                let digits = output.spare_capacity_mut();
                let (_, carry) = digits
                    .get_unchecked(start.unchecked_sub(2))
                    .assume_init()
                    .overflowing_add(7);
                carry && digits.get_unchecked(start.unchecked_sub(1)).assume_init() == Limb::MAX
            };
            if ambiguous {
                // SAFETY: total disjoint writable limbs cover the exact full
                // product. Overwriting the approximation initializes all of them.
                unsafe {
                    let _ = Multiplication::mul_nonempty_distinct_into_uninit(
                        a,
                        b,
                        output.spare_capacity_mut().get_unchecked_mut(..total),
                        mul_scratch,
                    );
                }
                start = skip;
                end = total;
            }
            // SAFETY: the block kernel initializes through end; a complete
            // fallback initializes total instead. The certified suffix is exact.
            unsafe {
                output.set_len(end);
                return output.get_unchecked_mut(start..);
            }
        }
        if smaller >= KARATSUBA_THRESHOLD || skip < 3 {
            // SAFETY: the materialized input bound also covers one high guard.
            let capacity = unsafe { total.unchecked_add(1) };
            output.reset_with_capacity(capacity);
            // SAFETY: reservation covers the complete disjoint product and its
            // carry slot. The product writer initializes every exposed limb.
            unsafe {
                let digits = output.spare_capacity_mut().get_unchecked_mut(..total);
                let _ =
                    Multiplication::mul_nonempty_distinct_into_uninit(a, b, digits, mul_scratch);
                output.set_len(total);
                return output.get_unchecked_mut(skip..);
            }
        }
        // SAFETY: 3<=skip<total gives c>=1 and total-c>=3; one guard fits usize.
        let (cut, width, capacity) = unsafe {
            let cut = skip.unchecked_sub(2);
            let width = total.unchecked_sub(cut);
            (cut, width, width.unchecked_add(1))
        };
        output.reset_with_capacity(capacity);
        // SAFETY: the disjoint nonempty operands and reserved width=total-cut
        // span satisfy the certified kernel's domain. It initializes all width limbs.
        unsafe {
            Self::certified_high_product(
                a,
                b,
                cut,
                output.spare_capacity_mut().get_unchecked_mut(..width),
                carry_product,
            );
        }
        // SAFETY: the certified product initialized exactly width limbs.
        // The reserved carry slot remains outside the initialized slice.
        unsafe {
            output.set_len(width);
            output.get_unchecked_mut(2..)
        }
    }

    /// Initializes the retained diagonals and resolves their omitted carry.
    ///
    /// Both high-product consumers share this kernel; output guard allocation
    /// and the complete multiplication tower remain in the enclosing driver.
    ///
    /// # Safety
    /// Nonempty initialized operands, output and carry storage are disjoint.
    /// `0<cut<a.len()+b.len()-2`; output contains `a.len()+b.len()-cut` writable
    /// limbs. The initialized result is T/B^cut, adjusted by E/B^cut only when
    /// required to make its suffix after two low guards equal to the exact product.
    unsafe fn certified_high_product(
        a: &[Limb],
        b: &[Limb],
        cut: usize,
        output: &mut [MaybeUninit<Limb>],
        carry_product: &mut ScratchBuffer,
    ) {
        // SAFETY: the inherited nonempty disjoint operands and exact output
        // width satisfy the diagonal writer's domain and first-write contract.
        unsafe {
            Self::high_product_diagonals(a, b, cut, output);
        }
        let bound = cut.min(a.len()).min(b.len());
        // SAFETY: bound<=each materialized limb count<=Limb::MAX. The kernel
        // initialized width>=3 limbs, including the upper low guard at index one.
        let certified =
            unsafe { *output.get_unchecked(1).assume_init_ref() <= Limb::MAX.unchecked_sub(bound) };
        if certified {
            return;
        }
        // SAFETY: cut<total-2 bounds the omitted polynomial's cut+2-limb width.
        let capacity = unsafe { cut.unchecked_add(2) };
        carry_product.reset_with_capacity(capacity);
        // SAFETY: cut>=1 and capacity=cut+2 reserve the omitted polynomial and
        // its two high digits. All buffers and initialized inputs are disjoint.
        // The low kernel initializes every element before its length is exposed.
        unsafe {
            Self::low_product_diagonals(
                a,
                b,
                cut,
                carry_product
                    .spare_capacity_mut()
                    .get_unchecked_mut(..capacity),
            );
            carry_product.set_len(capacity);
            let initialized = from_raw_parts_mut(output.as_mut_ptr().cast::<Limb>(), output.len());
            let (low_guards, higher) = initialized.split_at_mut_unchecked(2);
            let carry =
                Addition::add_slice_in_place(low_guards, carry_product.get_unchecked(cut..));
            let overflow = Addition::propagate_carry(higher, carry);
            debug_assert_eq!(overflow, 0, "(T+E)/B^c fits the retained product width");
        }
    }

    /// Writes `sum(a[i]*b[j]*B^(i+j-cut), i+j>=cut)` without a zero pass.
    ///
    /// # Safety
    /// Nonempty initialized operands and output are disjoint; `cut<a.len()+b.len()-2`.
    /// Output has exactly `a.len()+b.len()-cut` writable limbs. The first retained
    /// row initializes its low prefix and carry; each following row consumes that
    /// prefix and writes the next untouched high carry slot.
    pub unsafe fn high_product_diagonals(
        a: &[Limb],
        b: &[Limb],
        cut: usize,
        output: &mut [MaybeUninit<Limb>],
    ) {
        // SAFETY: b is nonempty. cut<total-2 ensures first<a.len(); the first
        // retained row begins at column<=b.len()-1 and has destination offset zero.
        let (first, first_column, first_len) = unsafe {
            let first = cut.saturating_sub(b.len().unchecked_sub(1));
            let column = cut.unchecked_sub(first);
            (first, column, b.len().unchecked_sub(column))
        };
        // SAFETY: first<a.len() selects the first retained scalar.
        let scalar = unsafe { *a.get_unchecked(first) };
        let mut first_carry = 0;
        let mut index = 0_usize;
        loop {
            // SAFETY: first_column<b.len() gives first_len>0. This loop stops
            // when its successor reaches first_len; first_column+index<b.len()
            // and index+1<=first_len<output.len() hold on every iteration.
            // The product high limb is <=B-2; adding a binary carry fits a Limb.
            unsafe {
                let (low, high) = ArchKernels::mul_limb_lo_hi(
                    scalar,
                    *b.get_unchecked(first_column.unchecked_add(index)),
                );
                let (digit, carry) = low.overflowing_add(first_carry);
                let _ = output.get_unchecked_mut(index).write(digit);
                first_carry = high.unchecked_add(Limb::from(carry));
                index = index.unchecked_add(1);
            }
            if index == first_len {
                break;
            }
        }
        // SAFETY: the first retained row has first_len+1 writable product limbs.
        unsafe {
            let _ = output.get_unchecked_mut(first_len).write(first_carry);
        }
        // SAFETY: first<a.len() bounds the successor by a.len().
        let mut row = unsafe { first.unchecked_add(1) };
        if row == a.len() {
            return;
        }
        let add_mul = ArchKernels::selected_add_mul_limbs_unchecked();
        while row < a.len() {
            let column = cut.saturating_sub(row);
            let offset = row.saturating_sub(cut);
            // SAFETY: column<b.len(). Earlier rows initialized exactly through
            // row+b.len()-cut-1, which covers this accumulation's complete span.
            // Its outgoing carry receives the next untouched slot within output.
            unsafe {
                let width = b.len().unchecked_sub(column);
                let carry = add_mul(
                    output.as_mut_ptr().cast::<Limb>().add(offset),
                    b.as_ptr().add(column),
                    width,
                    *a.get_unchecked(row),
                );
                let _ = output
                    .get_unchecked_mut(offset.unchecked_add(width))
                    .write(carry);
                row = row.unchecked_add(1);
            }
        }
    }

    /// Writes `sum(a[i]*b[j]*B^(i+j), i+j<cut)`, including both carry digits.
    ///
    /// # Safety
    /// Nonempty initialized operands and output are disjoint; cut>0 and
    /// output has cut+2 writable limbs. At most min(cut,m,n)<B products occur
    /// per column, so the complete omitted polynomial fits this width.
    unsafe fn low_product_diagonals(
        a: &[Limb],
        b: &[Limb],
        cut: usize,
        output: &mut [MaybeUninit<Limb>],
    ) {
        let first_len = cut.min(b.len());
        // SAFETY: cut>0 and both inputs are nonempty. The scalar row has
        // first_len>0 source limbs and fits the cut+2 destination, including carry.
        let first_width = unsafe { first_len.unchecked_add(1) };
        // SAFETY: a is nonempty, so its first scalar exists.
        let scalar = unsafe { *a.get_unchecked(0) };
        let mut first_carry = 0;
        let mut index = 0_usize;
        loop {
            // SAFETY: cut>0 and b.len()>0 give first_len>0. This loop stops
            // when its successor reaches first_len; index<first_len<=b.len()
            // and index+1<=first_len<output.len() hold on every iteration.
            // The product high limb is <=B-2; adding a binary carry fits a Limb.
            unsafe {
                let (low, high) = ArchKernels::mul_limb_lo_hi(scalar, *b.get_unchecked(index));
                let (digit, carry) = low.overflowing_add(first_carry);
                let _ = output.get_unchecked_mut(index).write(digit);
                first_carry = high.unchecked_add(Limb::from(carry));
                index = index.unchecked_add(1);
            }
            if index == first_len {
                break;
            }
        }
        // SAFETY: the first row fits first_width limbs, including carry.
        // The remaining zeros initialize every destination before accumulation.
        let initialized = unsafe {
            let _ = output.get_unchecked_mut(first_len).write(first_carry);
            output
                .get_unchecked_mut(first_width..)
                .fill(MaybeUninit::new(0));
            from_raw_parts_mut(output.as_mut_ptr().cast::<Limb>(), output.len())
        };
        let rows = cut.min(a.len());
        if rows == 1 {
            return;
        }
        let add_mul = ArchKernels::selected_add_mul_limbs_unchecked();
        let mut row = 1_usize;
        while row < rows {
            // SAFETY: row<rows<=cut bounds cut-row; the retained row ends
            // at or below cut. All accumulation and carry digits are initialized.
            unsafe {
                let width = b.len().min(cut.unchecked_sub(row));
                let carry = add_mul(
                    initialized.as_mut_ptr().add(row),
                    b.as_ptr(),
                    width,
                    *a.get_unchecked(row),
                );
                let carry_at = row.unchecked_add(width);
                let low = initialized.get_unchecked_mut(carry_at);
                let (sum, overflow) = low.overflowing_add(carry);
                *low = sum;
                let higher = initialized.get_unchecked_mut(carry_at.unchecked_add(1)..);
                let escaped = Addition::propagate_carry(higher, Limb::from(overflow));
                debug_assert_eq!(escaped, 0, "the omitted polynomial fits cut+2 limbs");
                row = row.unchecked_add(1);
            }
        }
    }
}
