//! Hardware-accelerated x86-64 BMI2 schoolbook squaring kernels.
//!
//! Uses `mulx` to evaluate the upper triangle of off-diagonal products, doubles
//! the triangle in-place with an unrolled `adcq` carry-chain, and accumulates
//! diagonal squares with independent carry propagation. The only extension
//! instruction is BMI2's `mulx`; this backend requires BMI2.

#![expect(
    clippy::inline_always,
    reason = "Inlining removes call boundaries between the square's assembly stages"
)]

use core::arch::asm;

use super::{ArchKernels, Limb};

/// Writes the complete square using BMI2 triangle products and scalar carries.
///
/// # Safety
///
/// BMI2 must be available, and `len > 8`. `a` must cover `len` aligned
/// initialized readable limbs. `dst` must cover `2 * len` aligned writable
/// limbs, disjoint from `a`; its contents may be uninitialized. The complete
/// output byte span must fit in `isize::MAX`.
pub unsafe fn sqr_basecase_unchecked(dst: *mut Limb, a: *const Limb, len: usize) {
    debug_assert!(len > 8, "small widths use the portable fixed kernels");
    let add_mul_limbs = ArchKernels::selected_add_mul_limbs_unchecked();

    // SAFETY: len > 8 and the complete 2*len output span fits in isize::MAX.
    // Row 0 initializes dst[0..=len]; later rows read only that initialized
    // prefix and extend it through 2*len-2. All indices stay within the aligned
    // disjoint spans. Each MULX has BMI2 available and early output registers.
    // For B=2^64, (B-1)^2+(B-1) < B^2 bounds each row's high carry.
    unsafe {
        let last = len.unchecked_sub(1);
        *dst = 0;
        let mut carry = 0;
        let scalar = *a;
        for column in 1..len {
            let low: Limb;
            let high: Limb;
            asm!(
                "mulxq {src}, {low}, {high}",
                src = in(reg) *a.add(column),
                in("rdx") scalar,
                low = out(reg) low,
                high = out(reg) high,
                options(nostack, att_syntax)
            );
            let (sum, overflow) = low.overflowing_add(carry);
            *dst.add(column) = sum;
            carry = high.unchecked_add(Limb::from(overflow));
        }
        *dst.add(len) = carry;
        for row in 1..last {
            let column = row.unchecked_mul(2).unchecked_add(1);
            let row_carry = add_mul_limbs(
                dst.add(column),
                a.add(row.unchecked_add(1)),
                last.unchecked_sub(row),
                *a.add(row),
            );
            *dst.add(row.unchecked_add(len)) = row_carry;
        }

        // a^2 = 2*sum_{i<j} a[i]*a[j]*B^(i+j) + sum_i a[i]^2*B^(2*i).
        // Doubling initializes the previously unwritten last output limb.
        let triangle_limbs = len.unchecked_mul(2).unchecked_sub(2);
        double_triangle_asm(dst, triangle_limbs);
        add_diagonal_squares_bmi2(dst, a, len);
    }
}

/// In-place doubling of the off-diagonal triangle `dst[1..=count]` using an
/// unrolled `adcq` chain without modifying the carry flag across branches.
///
/// # Safety
///
/// `dst` must cover `count + 2` aligned writable limbs. Limbs `1..=count`
/// must be initialized. Limb `count + 1` is written without being read.
/// The complete byte span must fit in `isize::MAX`.
#[inline(always)]
unsafe fn double_triangle_asm(dst: *mut Limb, count: usize) {
    if count == 0 {
        return;
    }
    // SAFETY: count > 0, and the caller provides the aligned writable span.
    let ptr = unsafe { dst.add(1) };
    let chunks = count >> 2;
    let rem = count & 3;

    // SAFETY: the caller guarantees dst[1..=count] is valid for reads and writes
    // and dst[count + 1] is writable. The asm block performs an unrolled adcq
    // carry chain. Exactly count limbs are read before the final carry write;
    // the address bound keeps the chunk counter below isize::MAX. Branches
    // use jrcxz and decq to preserve CF across loop
    // boundaries. `rem` is declared `inout` because the remainder loop rewrites
    // `%rcx`; its final value is discarded, which the `_` output makes explicit.
    unsafe {
        asm!(
            "xorl %eax, %eax", // Clear CF
            "decq {chunks}",
            "js 2f",

            // 4-way unrolled doubling loop
            "1:",
            "movq 0({ptr}), %r8",
            "adcq %r8, %r8",
            "movq %r8, 0({ptr})",

            "movq 8({ptr}), %r9",
            "adcq %r9, %r9",
            "movq %r9, 8({ptr})",

            "movq 16({ptr}), %r10",
            "adcq %r10, %r10",
            "movq %r10, 16({ptr})",

            "movq 24({ptr}), %r11",
            "adcq %r11, %r11",
            "movq %r11, 24({ptr})",

            "leaq 32({ptr}), {ptr}",
            "decq {chunks}",
            "jns 1b",

            "2:",
            "jrcxz 4f",

            "3:",
            "movq 0({ptr}), %r8",
            "adcq %r8, %r8",
            "movq %r8, 0({ptr})",
            "leaq 8({ptr}), {ptr}",
            "decq %rcx",
            "jnz 3b",

            "4:",
            // Write the final carry out to dst[count + 1]
            "movq $0, %r8",
            "adcq $0, %r8",
            "movq %r8, 0({ptr})",

            ptr = inout(reg) ptr => _,
            chunks = inout(reg) chunks => _,
            inout("rcx") rem => _,
            out("rax") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
}

/// Accumulates diagonal squares `a[i]^2` into `dst[2*i .. 2*i+2]` with 128-bit
/// carry propagation.
///
/// # Safety
///
/// BMI2 must be available. `a` must cover `len` aligned initialized readable
/// limbs. `dst` must cover `2 * len` aligned initialized writable limbs,
/// disjoint from `a`; its byte span must fit in `isize::MAX`. On entry it must
/// hold the doubled strict triangle, so adding the diagonals closes the square.
#[inline(always)]
unsafe fn add_diagonal_squares_bmi2(dst: *mut Limb, a: *const Limb, len: usize) {
    let mut carry = 0;
    // SAFETY: every i<len addresses one initialized input and two initialized
    // output limbs. The complete output bound proves 2*i+1 representable.
    // BMI2 is available and MULX outputs are early, distinct registers. Each
    // non-modular carry sum is at most two; digit additions return carry bits.
    unsafe {
        for i in 0..len {
            let u = *a.add(i);
            let lo: Limb;
            let hi: Limb;
            asm!(
                "mulxq %rdx, {lo}, {hi}",
                in("rdx") u,
                lo = out(reg) lo,
                hi = out(reg) hi,
                options(nostack, att_syntax)
            );
            let p_lo = dst.add(i.unchecked_mul(2));
            let p_hi = p_lo.add(1);
            let (partial_lo, c0) = (*p_lo).overflowing_add(lo);
            let (sum_lo, c1) = partial_lo.overflowing_add(carry);
            let incoming = Limb::from(c0).unchecked_add(Limb::from(c1));
            let (partial_hi, c2) = (*p_hi).overflowing_add(hi);
            let (sum_hi, c3) = partial_hi.overflowing_add(incoming);
            *p_lo = sum_lo;
            *p_hi = sum_hi;
            carry = Limb::from(c2).unchecked_add(Limb::from(c3));
        }
    }
    // The exact square of a len-limb operand fits in 2*len limbs, so the
    // diagonal accumulation must close with a zero carry.
    debug_assert_eq!(carry, 0, "diagonal squares must not carry past 2*len limbs");
}
