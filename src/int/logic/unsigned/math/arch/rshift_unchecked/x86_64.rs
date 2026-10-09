//! `x86_64` in-place multi-limb right-shift kernel (inline assembly).
//!
//! Uses hardware `shrdq` (Double Precision Shift Right) to fuse bit shifting and neighbor-limb
//! bit merging into single instructions, traversing forwards from bottom to top.

use core::arch::asm;

use super::Limb;

/// Right-shift `len` limbs in-place by `shift` bits ($0 < \text{shift} < 64$).
/// Returns the bits shifted out of the bottom limb.
///
/// Computes:
///
/// ```text
///   (carry_out, limbs[0..len]) = limbs[0..len] >> shift
/// ```
///
/// In-place right shifting moves from low to high indices. Instead of separate shift and OR
/// instructions, `shrdq` shifts the destination register right by `%cl` bits while pulling the
/// bottom `%cl` bits from the adjacent high limb.
/// The loop is 4-way unrolled (32 bytes per iteration).
///
/// # Safety
///
/// - For nonzero `len`, `limbs` covers `len` aligned, initialized, writable
///   limbs within `isize::MAX` bytes.
/// - `shift` must satisfy $0 < \text{shift} < 64$.
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the normalization leaf at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;

    // SAFETY: the aligned initialized writable span contains len > 0 limbs.
    // The first load is at zero. Each four-limb block and (len-1) % 4 tail
    // loads higher source limbs before replacing lower ones. The final store
    // reaches len-1. Shift counts are in 1..64, all clobbers are outputs, and
    // no stack access occurs.
    unsafe {
        asm!(
            // [Extract Carry-Out: bottom_limb << (64 - shift)]
            "movq ({limbs}), {carry_out}",               // Load bottom limb (limbs[0])
            "movl {shift:e}, %ecx",                      // %ecx = shift
            "negb %cl",                                  // %cl = (64 - shift) mod 64
            "shlq %cl, {carry_out}",                     // carry_out = bottom_limb << (64 - shift)
            "movl {shift:e}, %ecx",                      // Restore %cl = shift

            "leaq -8({limbs},{len},8), {end}",           // end = &limbs[len-1]
            "leaq ({limbs}), {ptr}",                     // ptr = &limbs[0]
            "movq ({ptr}), %rax",                        // %rax = limbs[0]

            "movq {len}, %rdx",                          // %rdx = len
            "subq $1, %rdx",                             // %rdx = len - 1
            "jz 2f",                                     // If len == 1, jump directly to top limb shift (2f)

            "movq %rdx, %r8",                            // %r8 = len - 1
            "shrq $2, %r8",                              // %r8 = 4-limb chunks
            "jz 3f",                                     // If chunks == 0, go to scalar remainder loop (3f)

            // Main 4-way unrolled ascending loop body
            "4:",
            "movq 8({ptr}), %r9",                        // Load limbs[i+1]
            "movq 16({ptr}), %r10",                      // Load limbs[i+2]
            "movq 24({ptr}), %r11",                      // Load limbs[i+3]
            "shrdq %cl, %r9, %rax",                      // %rax = (%rax >> cl) | (%r9 << (64-cl))

            "movq 32({ptr}), %r12",                      // Load limbs[i+4]
            "shrdq %cl, %r10, %r9",                      // %r9 = (%r9 >> cl) | (%r10 << (64-cl))
            "shrdq %cl, %r11, %r10",                     // %r10 = (%r10 >> cl) | (%r11 << (64-cl))
            "shrdq %cl, %r12, %r11",                     // %r11 = (%r11 >> cl) | (%r12 << (64-cl))
            "movq %rax, ({ptr})",                        // Store updated limbs[i]
            "movq %r9, 8({ptr})",                        // Store updated limbs[i+1]
            "movq %r10, 16({ptr})",                      // Store updated limbs[i+2]
            "movq %r11, 24({ptr})",                      // Store updated limbs[i+3]

            "movq %r12, %rax",                           // Propagate %r12 as next bottom limb
            "leaq 32({ptr}), {ptr}",                     // Advance pointer by 4 limbs (32 bytes)
            "decq %r8",                                  // Decrement chunk counter
            "jnz 4b",                                    // Repeat while chunks != 0

            // Remainder processing (0 to 3 limbs)
            "3:",
            "andq $3, %rdx",                             // %rdx = (len - 1) & 3
            "jz 2f",                                     // If 0 remainder, jump to top limb (2f)

            // 1-limb unrolled tail loop
            "1:",
            "movq 8({ptr}), %r9",                        // Load limbs[i+1]
            "shrdq %cl, %r9, %rax",                      // Merge shifted bits
            "movq %rax, ({ptr})",                        // Store updated limb
            "movq %r9, %rax",                            // Update running limb
            "leaq 8({ptr}), {ptr}",                      // Advance pointer by 1 limb (8 bytes)
            "decq %rdx",                                 // Decrement remainder
            "jnz 1b",

            // [Shift top-most limb (no source above)]
            "2:",
            "shrq %cl, %rax",                            // limbs[len-1] = limbs[len-1] >> shift
            "movq %rax, ({end})",                        // Store top limb

            carry_out = out(reg) carry_out,
            limbs = in(reg) limbs,
            len = in(reg) len,
            shift = in(reg) shift,
            ptr = out(reg) _,
            end = out(reg) _,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            out("r12") _,
            options(nostack, att_syntax)
        );
    }
    carry_out
}
