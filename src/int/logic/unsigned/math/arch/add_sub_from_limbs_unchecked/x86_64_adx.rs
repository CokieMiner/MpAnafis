//! x86-64 ADX shared-source simultaneous addition and subtraction.
//!
//! Evaluates `sum = sum + source` and `diff = sum - source` simultaneously
//! using dual-carry ADX instructions (`adcxq` for addition via CF, `notq` + `adoxq` for subtraction via OF).

use core::arch::asm;

use super::Limb;

/// Replace `sum` with `sum_original + source` and write
/// `sum_original - source` to `difference`, returning carry and borrow.
///
/// Computes simultaneously:
///
/// ```text
///   (carry, sum[0..len]) = sum[0..len] + source[0..len]
///   (borrow, difference[0..len]) = sum[0..len] - source[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// ADX runs parallel addition (CF) and subtraction (OF) from shared operand loads.
/// 8-way unrolled loop keeps registers saturated and minimizes loop branch latency.
///
/// # Safety
///
/// - All three pointers must cover `len` limbs.
/// - `difference` may equal `source` exactly, but otherwise must not overlap an input span.
/// - `sum` and `source` must not overlap.
/// - The caller must ensure the CPU supports ADX.
#[allow(
    clippy::too_many_lines,
    reason = "Eight-limb ADX unrolling retained a repeatable 1-1.5% Fermat-transform win"
)]
pub unsafe fn add_sub_from_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    source: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let sum_ptr = sum;
    let difference_ptr = difference;
    let source_ptr = source;
    let block_count = len >> 3;
    let tail_count = len & 7;
    let sum_carry: u8;
    let difference_carry: u8;

    // SAFETY:
    // 1. `sum_ptr`, `difference_ptr`, and `source_ptr` are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping except where documented.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            // Seed CF=0 and OF=1 (OF=1 seeds two's complement borrow-to-carry inversion)
            "movabsq $0x7fffffffffffffff, %r8",           // %r8 = MAX_INT
            "addq $1, %r8",                              // Clears CF (0), sets OF (1)
            "jrcxz 1f",                                  // If block_count == 0, jump to remainder (1f)
            "jmp 2f",                                    // Jump to 8-way loop (2f)
            "1:",                                        // Jump stub label
            "jmp 3f",                                    // Jump to tail entry (3f)

            ".p2align 4",                                // Align loop header
            // Main 8-way unrolled butterfly loop
            "2:",                                        // Loop head label
            // [Limb 0]
            "movq ({sum_ptr}), %r8",                     // %r8 = sum[0]
            "movq ({source_ptr}), %r9",                  // %r9 = source[0]
            "movq %r8, %r10",                            // %r10 = sum[0]
            "adcxq %r9, %r10",                           // %r10 = sum[0] + source[0] + CF (updates CF)
            "notq %r9",                                  // %r9 = ~source[0] (preserves all flags)
            "adoxq %r9, %r8",                            // %r8 = sum[0] + ~source[0] + OF (updates OF)
            "movq %r10, ({sum_ptr})",                    // Store updated sum[0]
            "movq %r8, ({difference_ptr})",              // Store difference[0]

            // [Limb 1]
            "movq 8({sum_ptr}), %r8",                    // %r8 = sum[1]
            "movq 8({source_ptr}), %r9",                 // %r9 = source[1]
            "movq %r8, %r10",                            // %r10 = sum[1]
            "adcxq %r9, %r10",                           // %r10 = sum[1] + source[1] + CF
            "notq %r9",                                  // %r9 = ~source[1]
            "adoxq %r9, %r8",                            // %r8 = sum[1] + ~source[1] + OF
            "movq %r10, 8({sum_ptr})",                   // Store sum[1]
            "movq %r8, 8({difference_ptr})",             // Store difference[1]

            // [Limb 2]
            "movq 16({sum_ptr}), %r8",                   // Load sum[2]
            "movq 16({source_ptr}), %r9",                // Load source[2]
            "movq %r8, %r10",                            // Copy sum[2]
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source[2]
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, 16({sum_ptr})",                  // Store sum[2]
            "movq %r8, 16({difference_ptr})",            // Store difference[2]

            // [Limb 3]
            "movq 24({sum_ptr}), %r8",                   // Load sum[3]
            "movq 24({source_ptr}), %r9",                // Load source[3]
            "movq %r8, %r10",                            // Copy sum[3]
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source[3]
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, 24({sum_ptr})",                  // Store sum[3]
            "movq %r8, 24({difference_ptr})",            // Store difference[3]

            // [Limb 4]
            "movq 32({sum_ptr}), %r8",                   // Load sum[4]
            "movq 32({source_ptr}), %r9",                // Load source[4]
            "movq %r8, %r10",                            // Copy sum[4]
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source[4]
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, 32({sum_ptr})",                  // Store sum[4]
            "movq %r8, 32({difference_ptr})",            // Store difference[4]

            // [Limb 5]
            "movq 40({sum_ptr}), %r8",                   // Load sum[5]
            "movq 40({source_ptr}), %r9",                // Load source[5]
            "movq %r8, %r10",                            // Copy sum[5]
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source[5]
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, 40({sum_ptr})",                  // Store sum[5]
            "movq %r8, 40({difference_ptr})",            // Store difference[5]

            // [Limb 6]
            "movq 48({sum_ptr}), %r8",                   // Load sum[6]
            "movq 48({source_ptr}), %r9",                // Load source[6]
            "movq %r8, %r10",                            // Copy sum[6]
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source[6]
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, 48({sum_ptr})",                  // Store sum[6]
            "movq %r8, 48({difference_ptr})",            // Store difference[6]

            // [Limb 7]
            "movq 56({sum_ptr}), %r8",                   // Load sum[7]
            "movq 56({source_ptr}), %r9",                // Load source[7]
            "movq %r8, %r10",                            // Copy sum[7]
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source[7]
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, 56({sum_ptr})",                  // Store sum[7]
            "movq %r8, 56({difference_ptr})",            // Store difference[7]

            "leaq 64({sum_ptr}), {sum_ptr}",             // Advance sum pointer by 64 bytes (flag-free)
            "leaq 64({difference_ptr}), {difference_ptr}", // Advance difference pointer by 64 bytes
            "leaq 64({source_ptr}), {source_ptr}",       // Advance source pointer by 64 bytes
            "leaq -1(%rcx), %rcx",                       // Decrement block counter (flag-free)
            "jrcxz 3f",                                  // If rcx == 0, jump to remainder
            "jmp 2b",                                    // Repeat loop

            // Remainder entry point
            "3:",                                        // Tail entry label
            "movq {tail_count}, %rcx",                   // Load tail count
            "jrcxz 5f",                                  // If tail == 0, skip to finish (5f)

            // 1-limb tail loop
            "4:",                                        // Tail loop label
            "movq ({sum_ptr}), %r8",                     // Load sum limb
            "movq ({source_ptr}), %r9",                  // Load source limb
            "movq %r8, %r10",                            // Copy sum limb
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r9",                                  // Invert source limb
            "adoxq %r9, %r8",                            // Subtract with OF
            "movq %r10, ({sum_ptr})",                    // Store sum
            "movq %r8, ({difference_ptr})",              // Store difference
            "leaq 8({sum_ptr}), {sum_ptr}",              // Advance sum pointer
            "leaq 8({difference_ptr}), {difference_ptr}",// Advance diff pointer
            "leaq 8({source_ptr}), {source_ptr}",        // Advance source pointer
            "leaq -1(%rcx), %rcx",                       // Decrement tail counter
            "jrcxz 5f",                                  // If rcx == 0, exit
            "jmp 4b",                                    // Repeat tail

            // Capture final carry and borrow
            "5:",                                        // Exit label
            "setc {sum_carry}",                          // sum_carry = 1 if CF == 1 (addition carry)
            "seto {difference_carry}",                   // difference_carry = 1 if OF == 1 (inverted borrow)

            sum_ptr = inout(reg) sum_ptr => _,
            difference_ptr = inout(reg) difference_ptr => _,
            source_ptr = inout(reg) source_ptr => _,
            inout("rcx") block_count => _,
            tail_count = in(reg) tail_count,
            sum_carry = lateout(reg_byte) sum_carry,
            difference_carry = lateout(reg_byte) difference_carry,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            options(nostack, att_syntax)
        );
    }
    (Limb::from(sum_carry), Limb::from(difference_carry ^ 1))
}
