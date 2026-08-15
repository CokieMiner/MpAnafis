//! x86-64 dual addition using the independent ADX carry chains.
//!
//! Evaluates two completely independent addition streams (`dst_a += src_a` and `dst_b += src_b`)
//! concurrently in a single unrolled pass using dual ADX carry flags (`adcxq` on CF and `adoxq` on OF).

use core::arch::asm;

use super::Limb;

/// Add two independent source spans into two destination spans.
///
/// Computes simultaneously:
///
/// ```text
///   (carry_a, dst_a[0..len]) = dst_a[0..len] + src_a[0..len]
///   (carry_b, dst_b[0..len]) = dst_b[0..len] + src_b[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// `adcxq` carries the first sum through CF while `adoxq` carries the second sum through OF.
/// Pointer and counter updates preserve both flags (`leaq`, `jrcxz`), retiring both independent
/// vector additions in a single pass without extra memory traffic.
///
/// # Safety
///
/// - Every pointer must cover `len` readable limbs.
/// - Both destination pointers must cover `len` writable limbs.
/// - No destination span may overlap any other span.
/// - The caller must ensure the CPU supports ADX.
pub unsafe fn add_two_limbs_unchecked(
    dst_a: *mut Limb,
    src_a: *const Limb,
    dst_b: *mut Limb,
    src_b: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let block_count = len >> 2;
    let tail_count = len & 3;
    let carry_a: u8;
    let carry_b: u8;

    // SAFETY:
    // 1. All pointers are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl %eax, %eax",                           // Zero %eax (clears both CF and OF)
            "jrcxz 1f",                                  // If block_count == 0, jump to remainder (1f)
            "jmp 2f",                                    // Jump to 4-way loop (2f)
            "1:",                                        // Jump stub label
            "jmp 3f",                                    // Jump to tail entry (3f)

            ".p2align 4",                                // Align loop header
            // Main 4-way unrolled dual-addition loop
            "2:",                                        // Loop head label
            // [Limb 0: Stream A on CF, Stream B on OF]
            "movq ({dst_a}), %r8",                       // Load dst_a[0]
            "movq ({src_a}), %r9",                       // Load src_a[0]
            "adcxq %r9, %r8",                            // %r8 = dst_a[0] + src_a[0] + CF (updates CF)
            "movq %r8, ({dst_a})",                       // Store updated dst_a[0]
            "movq ({dst_b}), %r10",                      // Load dst_b[0]
            "movq ({src_b}), %r11",                      // Load src_b[0]
            "adoxq %r11, %r10",                          // %r10 = dst_b[0] + src_b[0] + OF (updates OF)
            "movq %r10, ({dst_b})",                      // Store updated dst_b[0]

            // [Limb 1]
            "movq 8({dst_a}), %r8",                      // Load dst_a[1]
            "movq 8({src_a}), %r9",                      // Load src_a[1]
            "adcxq %r9, %r8",                            // Add with CF
            "movq %r8, 8({dst_a})",                      // Store dst_a[1]
            "movq 8({dst_b}), %r10",                     // Load dst_b[1]
            "movq 8({src_b}), %r11",                     // Load src_b[1]
            "adoxq %r11, %r10",                          // Add with OF
            "movq %r10, 8({dst_b})",                     // Store dst_b[1]

            // [Limb 2]
            "movq 16({dst_a}), %r8",                     // Load dst_a[2]
            "movq 16({src_a}), %r9",                     // Load src_a[2]
            "adcxq %r9, %r8",                            // Add with CF
            "movq %r8, 16({dst_a})",                     // Store dst_a[2]
            "movq 16({dst_b}), %r10",                    // Load dst_b[2]
            "movq 16({src_b}), %r11",                    // Load src_b[2]
            "adoxq %r11, %r10",                          // Add with OF
            "movq %r10, 16({dst_b})",                    // Store dst_b[2]

            // [Limb 3]
            "movq 24({dst_a}), %r8",                     // Load dst_a[3]
            "movq 24({src_a}), %r9",                     // Load src_a[3]
            "adcxq %r9, %r8",                            // Add with CF
            "movq %r8, 24({dst_a})",                     // Store dst_a[3]
            "movq 24({dst_b}), %r10",                    // Load dst_b[3]
            "movq 24({src_b}), %r11",                    // Load src_b[3]
            "adoxq %r11, %r10",                          // Add with OF
            "movq %r10, 24({dst_b})",                    // Store dst_b[3]

            // Advance all 4 pointers by 32 bytes (flag-free)
            "leaq 32({dst_a}), {dst_a}",                 // Advance dst_a (+32)
            "leaq 32({src_a}), {src_a}",                 // Advance src_a (+32)
            "leaq 32({dst_b}), {dst_b}",                 // Advance dst_b (+32)
            "leaq 32({src_b}), {src_b}",                 // Advance src_b (+32)
            "leaq -1(%rcx), %rcx",                       // Decrement block counter (flag-free)
            "jrcxz 3f",                                  // If rcx == 0, jump to remainder
            "jmp 2b",                                    // Repeat loop

            // Remainder entry point
            "3:",                                        // Tail entry label
            "movq {tail_count}, %rcx",                   // Load tail count
            "jrcxz 5f",                                  // If tail == 0, skip to finish (5f)

            // 1-limb tail loop
            "4:",                                        // Tail loop label
            "movq ({dst_a}), %r8",                       // Load single dst_a limb
            "movq ({src_a}), %r9",                       // Load single src_a limb
            "adcxq %r9, %r8",                            // Add stream A with CF
            "movq %r8, ({dst_a})",                       // Store single dst_a limb
            "movq ({dst_b}), %r10",                      // Load single dst_b limb
            "movq ({src_b}), %r11",                      // Load single src_b limb
            "adoxq %r11, %r10",                          // Add stream B with OF
            "movq %r10, ({dst_b})",                      // Store single dst_b limb
            "leaq 8({dst_a}), {dst_a}",                  // Advance pointers
            "leaq 8({src_a}), {src_a}",
            "leaq 8({dst_b}), {dst_b}",
            "leaq 8({src_b}), {src_b}",
            "leaq -1(%rcx), %rcx",                       // Decrement tail counter
            "jrcxz 5f",                                  // If rcx == 0, exit
            "jmp 4b",                                    // Repeat tail

            // Capture final carries for stream A and stream B
            "5:",                                        // Exit label
            "setc {carry_a}",                            // carry_a = 1 if CF == 1
            "seto {carry_b}",                            // carry_b = 1 if OF == 1

            dst_a = inout(reg) dst_a => _,
            src_a = inout(reg) src_a => _,
            dst_b = inout(reg) dst_b => _,
            src_b = inout(reg) src_b => _,
            inout("rcx") block_count => _,
            tail_count = in(reg) tail_count,
            carry_a = lateout(reg_byte) carry_a,
            carry_b = lateout(reg_byte) carry_b,
            out("rax") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    (Limb::from(carry_a), Limb::from(carry_b))
}
