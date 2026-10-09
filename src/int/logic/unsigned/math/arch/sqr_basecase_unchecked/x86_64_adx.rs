//! Hardware-accelerated x86-64 ADX schoolbook squaring kernels.
//!
//! Uses `mulx` for 128-bit products and the ADX `adcx`/`adox` pair to run two
//! independent carry chains at once: the off-diagonal triangle is doubled on
//! `CF` while the diagonal squares accumulate on `OF`. Both chains advance in
//! the same sweep, so doubling and diagonal addition share one traversal.

#![expect(
    clippy::inline_always,
    reason = "Inlining removes call boundaries between the square's fixed assembly stages"
)]

use core::arch::asm;

use super::Limb;

/// Writes the complete square using paired triangle rows and an ADX sweep.
///
/// # Safety
///
/// ADX and BMI2 must be available, and `len > 8`. `a` must cover `len` aligned
/// initialized readable limbs. `dst` must cover `2 * len` aligned writable
/// limbs, disjoint from `a`; its contents may be uninitialized. The complete
/// output byte span must fit in `isize::MAX`.
pub unsafe fn sqr_basecase_unchecked(dst: *mut Limb, a: *const Limb, len: usize) {
    debug_assert!(len > 8, "small widths use the portable fixed kernels");

    // SAFETY: len > 8 and the complete output span fits in isize::MAX. Row 0
    // initializes dst[0..=len] using only a[0..len]. For B=2^64,
    // (B-1)^2+(B-1) < B^2 bounds the high product plus its carry bit.
    // Each MULX has its BMI2 prerequisite and disjoint early output registers.
    unsafe {
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
    }

    // Adjacent rows share a[index+2..len]; their cross term is peeled first.
    // SAFETY: index starts at one and increases by two below len-1. Complete
    // 2*len span bounds make every doubled index and increment representable.
    // Preceding rows initialize all digits read, including column 2*index+1;
    // add_triangle_pair writes its two new high digits. The peeled product
    // plus an existing digit is below B^2, with incoming carry at most s0.
    unsafe {
        let last = len.unchecked_sub(1);
        let mut index = 1_usize;
        while index < last {
            let s0 = *a.add(index);
            let s1 = *a.add(index.unchecked_add(1));
            let column = index.unchecked_mul(2).unchecked_add(1);
            let low: Limb;
            let high: Limb;
            asm!(
                "mulxq {s1}, {low}, {high}",
                s1 = in(reg) s1,
                in("rdx") s0,
                low = out(reg) low,
                high = out(reg) high,
                options(nostack, att_syntax)
            );
            let (cross, overflow) = (*dst.add(column)).overflowing_add(low);
            *dst.add(column) = cross;
            let carry = high.unchecked_add(Limb::from(overflow));
            let remaining = len.unchecked_sub(index).unchecked_sub(2);
            add_triangle_pair(
                dst.add(column.unchecked_add(1)),
                a.add(index.unchecked_add(2)),
                remaining,
                s0,
                s1,
                carry,
            );
            index = index.unchecked_add(2);
        }
    }

    // SAFETY: the rows establish dst[0]=0 and the complete initialized strict
    // triangle through 2*len-2. Original disjoint spans and CPU features hold.
    unsafe {
        double_triangle_and_diagonals_adx(dst, a, len);
    }
}

/// Doubles the off-diagonal triangle in place and accumulates the diagonal
/// squares in the same sweep, using independent `CF`/`OF` carry chains.
///
/// On entry `dst[0]` is zero and `dst[1..=2*len-2]` holds the strict upper
/// triangle. The sweep writes `dst[0] += a[0]^2` low limb, doubles every
/// triangle limb while folding the matching diagonal half into it, and closes
/// `dst[2*len-1]` with the final doubling carry plus the final diagonal high
/// limb and diagonal carry.
///
/// # Safety
///
/// - ADX and BMI2 must be available.
/// - `dst` must cover `2 * len` aligned writable limbs. The triangle prefix
///   through `2*len-2` must be initialized; the final limb is never read.
/// - `a` must cover `len` aligned initialized readable limbs, disjoint from `dst`.
/// - `len >= 9` (shorter lengths are handled by the portable fixed kernels).
/// - The complete output byte span must fit in `isize::MAX`.
#[inline(always)]
unsafe fn double_triangle_and_diagonals_adx(dst: *mut Limb, a: *const Limb, len: usize) {
    // The main loop processes pairs i = 0..len-3: limb 2i+1 takes the diagonal
    // high half and limb 2i+2 the next diagonal low half. The last pair
    // (i = len-2) is peeled into the tail, and the very last high half closes
    // the buffer after the tail. Register roles: r8/r10 = current/next
    // diagonal low, r9/r11 = current/next diagonal high, rax = doubling
    // scratch, rdx = `mulx` implicit operand.
    // SAFETY: this stage receives len >= 9 from the complete-square driver.
    let iterations = unsafe { len.unchecked_sub(2) };

    // SAFETY: the caller guarantees the complete spans and len >= 9, so the
    // peeled loop bound is non-empty and every indexed limb exists. The asm
    // performs an unrolled dual-chain carry sweep; the closing `loop` touches
    // no flags (a `decq`/`jnz` pair would clobber OF and destroy the adox
    // diagonal chain), and `xorl` established both chains as zero. ADX
    // instructions take a register destination only, so each limb is loaded
    // once, carried through both chains, and stored once. The caller establishes
    // ADX/BMI2 and aligned disjoint spans. Every temporary and changed pointer
    // is an early output, so live inputs cannot share its register.
    unsafe {
        asm!(
            "xorl %eax, %eax",              // CF = OF = 0
            // Diagonal pair 0: low half folds into dst[0], high half feeds limb 1.
            "movq 0({a_ptr}), %rdx",
            "mulxq %rdx, %r8, %r9",         // r9:r8 = a[0]^2
            "movq %r8, 0({dst})",           // dst[0] = lo0 (dst[0] is zero)
            // Preload diagonal pair 1 for the first loop iteration and advance
            // the source pointer so each loop reload walks to the next pair.
            "movq 8({a_ptr}), %rdx",
            "mulxq %rdx, %r10, %r11",       // r11:r10 = a[1]^2
            "leaq 8({a_ptr}), {a_ptr}",
            "movq {iterations}, %rcx",

            // Pair i: limb 2i+1 doubles and takes hi_i; limb 2i+2 doubles and
            // takes lo_{i+1}. The adcx chain threads the doubling carry on CF
            // while the adox chain threads the diagonal carry on OF; the two
            // chains never touch each other's flag.
            "1:",
            "movq 0({ptr}), %rax",
            "adcxq %rax, %rax",
            "adoxq %r9, %rax",
            "movq %rax, 0({ptr})",
            "movq 8({ptr}), %rax",
            "adcxq %rax, %rax",
            "adoxq %r10, %rax",
            "movq %rax, 8({ptr})",
            "leaq 16({ptr}), {ptr}",
            "leaq 8({a_ptr}), {a_ptr}",
            "movq %r11, %r9",               // hi_{i+1} becomes current
            "movq 0({a_ptr}), %rdx",
            "mulxq %rdx, %r10, %r11",       // preload pair i+2
            "loop 1b",

            // Tail pair i = len-2: limbs 2len-3 and 2len-2.
            "movq 0({ptr}), %rax",
            "adcxq %rax, %rax",
            "adoxq %r9, %rax",
            "movq %rax, 0({ptr})",
            "movq 8({ptr}), %rax",
            "adcxq %rax, %rax",
            "adoxq %r10, %rax",
            "movq %rax, 8({ptr})",
            "leaq 16({ptr}), {ptr}",

            // Close dst[2len-1] with the final doubling carry (CF), the final
            // diagonal carry (OF), and hi_{len-1}.
            "movq $0, %rax",
            "adcxq %rax, %rax",
            "adoxq %r11, %rax",
            "movq %rax, 0({ptr})",

            ptr = inout(reg) dst.add(1) => _,
            a_ptr = inout(reg) a => _,
            dst = in(reg) dst,
            iterations = in(reg) iterations,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
}

/// Adds `src * (s0 + B*s1) + carry` to an initialized destination prefix.
///
/// CF adds the low product to the two-word accumulator; OF adds the existing
/// destination digit and the shifted high product. Each source digit is read
/// once, and each destination digit is read and written once. If P=s0+B*s1
/// and the incoming two-word carry C <= P, the next carry satisfies
/// floor(((B-1)*P + C + (B-1))/B) <= P < B^2. Both carry chains therefore
/// terminate in the second high word, without a third output carry.
///
/// # Safety
/// ADX and BMI2 are available. Source covers len readable limbs, destination
/// covers len initialized limbs followed by two writable limbs, and the spans
/// do not overlap. All addresses are aligned Limb pointers. The initial carry
/// is at most s0, as produced by the peeled cross term; thus carry <= P.
#[inline(always)]
unsafe fn add_triangle_pair(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
    carry: Limb,
) {
    // SAFETY: the loop advances through exactly len initialized disjoint source
    // and destination digits, then writes two new high digits. The bound above
    // proves each ADCX/ADOX high carry absorption ends with its flag clear.
    // DEC acts on a positive address-bounded count and also leaves OF clear;
    // it preserves CF. RAX remains zero and no stack memory is addressed.
    unsafe {
        asm!(
            "xorl %eax, %eax",
            "movq $0, {carry_hi}",
            "testq {count}, {count}",
            "jz 3f",
            "2:",
            "movq ({src}), %rdx",
            "mulxq {s0}, %r8, %r9",
            "mulxq {s1}, %r10, %r11",
            "adcxq %r8, {carry_lo}",
            "adoxq ({dst}), {carry_lo}",
            "adcxq %r9, {carry_hi}",
            "adoxq %r10, {carry_hi}",
            "adcxq %rax, %r11",
            "adoxq %rax, %r11",
            "movq {carry_lo}, ({dst})",
            "movq {carry_hi}, {carry_lo}",
            "movq %r11, {carry_hi}",
            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {count}",
            "jnz 2b",
            "3:",
            "movq {carry_lo}, ({dst})",
            "movq {carry_hi}, 8({dst})",
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            count = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry_lo = inout(reg) carry => _,
            carry_hi = out(reg) _,
            out("rax") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
}
