//! x86-64 ADX simultaneous addition and reverse subtraction.
//!
//! Evaluates butterfly operations (`sum = sum + diff`, `diff = diff - sum`) simultaneously
//! using dual-carry ADX instructions (`adcxq` for addition via CF, `notq` + `adoxq` for subtraction via OF).

use core::arch::asm;

use super::Limb;

/// Replace `sum` with `sum + difference` and `difference` with
/// `difference_original - sum_original`, returning carry and borrow.
///
/// Computes simultaneously:
///
/// ```text
///   (carry, sum[0..len]) = sum[0..len] + difference[0..len]
///   (borrow, difference[0..len]) = difference[0..len] - sum[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// ADX allows running addition on CF and subtraction on OF in parallel across shared loads.
/// Subtraction is converted to addition with negated bits (`notq` + `adoxq`) seeded with OF=1.
///
/// # Safety
///
/// - `sum` and `difference` must each point to readable and writable buffers of at least `len` 64-bit limbs.
/// - `sum` and `difference` buffers must not overlap in memory (non-aliasing invariant).
/// - The caller must ensure the CPU supports ADX.
pub unsafe fn add_reverse_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let sum_ptr = sum;
    let difference_ptr = difference;
    let block_count = len >> 2;
    let tail_count = len & 3;
    let sum_carry: u8;
    let difference_carry: u8;

    // SAFETY:
    // 1. `sum_ptr` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. `difference_ptr` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            // Seed CF=0 and OF=1 (OF=1 seeds two's complement borrow-to-carry inversion)
            "movabsq $0x7fffffffffffffff, %r8",           // %r8 = MAX_INT
            "addq $1, %r8",                              // Clears CF (0), sets OF (1)
            "jrcxz 1f",                                  // If block_count == 0, jump to remainder (1f)
            "jmp 2f",                                    // Jump to 4-way loop (2f)
            "1:",                                        // Jump stub label
            "jmp 3f",                                    // Jump to tail entry (3f)

            ".p2align 4",                                // Align loop header
            // Main 4-way unrolled butterfly loop
            "2:",                                        // Loop head label
            // [Limb 0]
            "movq ({sum_ptr}), %r8",                     // %r8 = sum[0]
            "movq ({difference_ptr}), %r9",              // %r9 = diff[0]
            "movq %r8, %r10",                            // %r10 = sum[0]
            "adcxq %r9, %r10",                           // %r10 = sum[0] + diff[0] + CF (updates CF)
            "notq %r8",                                  // %r8 = ~sum[0] (preserves all flags)
            "adoxq %r8, %r9",                            // %r9 = diff[0] + ~sum[0] + OF (updates OF)
            "movq %r10, ({sum_ptr})",                    // Store updated sum[0]
            "movq %r9, ({difference_ptr})",              // Store updated diff[0]

            // [Limb 1]
            "movq 8({sum_ptr}), %r8",                    // %r8 = sum[1]
            "movq 8({difference_ptr}), %r9",             // %r9 = diff[1]
            "movq %r8, %r10",                            // %r10 = sum[1]
            "adcxq %r9, %r10",                           // %r10 = sum[1] + diff[1] + CF
            "notq %r8",                                  // %r8 = ~sum[1]
            "adoxq %r8, %r9",                            // %r9 = diff[1] + ~sum[1] + OF
            "movq %r10, 8({sum_ptr})",                   // Store updated sum[1]
            "movq %r9, 8({difference_ptr})",             // Store updated diff[1]

            // [Limb 2]
            "movq 16({sum_ptr}), %r8",                   // %r8 = sum[2]
            "movq 16({difference_ptr}), %r9",            // %r9 = diff[2]
            "movq %r8, %r10",                            // %r10 = sum[2]
            "adcxq %r9, %r10",                           // %r10 = sum[2] + diff[2] + CF
            "notq %r8",                                  // %r8 = ~sum[2]
            "adoxq %r8, %r9",                            // %r9 = diff[2] + ~sum[2] + OF
            "movq %r10, 16({sum_ptr})",                  // Store updated sum[2]
            "movq %r9, 16({difference_ptr})",            // Store updated diff[2]

            // [Limb 3]
            "movq 24({sum_ptr}), %r8",                   // %r8 = sum[3]
            "movq 24({difference_ptr}), %r9",            // %r9 = diff[3]
            "movq %r8, %r10",                            // %r10 = sum[3]
            "adcxq %r9, %r10",                           // %r10 = sum[3] + diff[3] + CF
            "notq %r8",                                  // %r8 = ~sum[3]
            "adoxq %r8, %r9",                            // %r9 = diff[3] + ~sum[3] + OF
            "movq %r10, 24({sum_ptr})",                  // Store updated sum[3]
            "movq %r9, 24({difference_ptr})",            // Store updated diff[3]

            "leaq 32({sum_ptr}), {sum_ptr}",             // Advance sum pointer by 32 bytes (flag-free)
            "leaq 32({difference_ptr}), {difference_ptr}", // Advance diff pointer by 32 bytes (flag-free)
            "leaq -1(%rcx), %rcx",                       // Decrement block count (flag-free)
            "jrcxz 3f",                                  // If rcx == 0, jump to remainder
            "jmp 2b",                                    // Repeat loop

            // Remainder entry point
            "3:",                                        // Tail entry label
            "movq {tail_count}, %rcx",                   // Load tail count
            "jrcxz 5f",                                  // If tail == 0, skip to finish (5f)

            // 1-limb tail loop
            "4:",                                        // Tail loop label
            "movq ({sum_ptr}), %r8",                     // Load sum limb
            "movq ({difference_ptr}), %r9",              // Load diff limb
            "movq %r8, %r10",                            // Copy sum limb
            "adcxq %r9, %r10",                           // Add with CF
            "notq %r8",                                  // Invert sum limb
            "adoxq %r8, %r9",                            // Subtract with OF
            "movq %r10, ({sum_ptr})",                    // Store sum
            "movq %r9, ({difference_ptr})",              // Store diff
            "leaq 8({sum_ptr}), {sum_ptr}",              // Advance sum pointer
            "leaq 8({difference_ptr}), {difference_ptr}",// Advance diff pointer
            "leaq -1(%rcx), %rcx",                       // Decrement tail counter
            "jrcxz 5f",                                  // If rcx == 0, exit
            "jmp 4b",                                    // Repeat tail

            // Capture final carry and borrow
            "5:",                                        // Exit label
            "setc {sum_carry}",                          // sum_carry = 1 if CF == 1 (addition carry)
            "seto {difference_carry}",                   // difference_carry = 1 if OF == 1 (inverted borrow)

            sum_ptr = inout(reg) sum_ptr => _,
            difference_ptr = inout(reg) difference_ptr => _,
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
