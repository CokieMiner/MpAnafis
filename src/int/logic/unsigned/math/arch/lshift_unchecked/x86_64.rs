//! `x86_64` in-place multi-limb left-shift kernel (inline assembly).
//!
//! Uses hardware `shldq` (Double Precision Shift Left) to fuse bit shifting and neighbor-limb
//! bit merging into single instructions, traversing backwards from top to bottom.

use core::arch::asm;

use super::Limb;

/// Left-shift `len` limbs in-place by `shift` bits ($0 < \text{shift} < 64$).
/// Returns the bits shifted out of the top limb.
///
/// Computes:
///
/// ```text
///   (carry_out, limbs[0..len]) = limbs[0..len] << shift
/// ```
///
/// # Microarchitectural Strategy
///
/// In-place left shifting moves from high to low indices to avoid overwriting bits before they are read.
/// Instead of separate `shl`, `shr`, and `or` operations, x86 `shldq` shifts the destination register
/// left by `%cl` bits while pulling the top `%cl` bits from the source register in a single 3-cycle operation.
/// The loop is 4-way unrolled (32 bytes per iteration).
///
/// # Safety
///
/// - `limbs` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `shift` must satisfy $0 < \text{shift} < 64$.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in bit-shift hot paths"
)]
#[inline(always)]
pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;

    // SAFETY:
    // 1. `limbs` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. `shift` is strictly within `1..=63`.
    // 3. Pointer offsets (`-8`, `-16`, `-24`, `-32`) remain within `len * 8` bytes.
    unsafe {
        asm!(
            // [Extract Carry-Out: top_limb >> (64 - shift)]
            "movq -8({limbs},{len},8), {carry_out}",     // Load top limb (limbs[len-1])
            "movl {shift:e}, %ecx",                      // %ecx = shift
            "negb %cl",                                  // %cl = (64 - shift) mod 64
            "shrq %cl, {carry_out}",                     // carry_out = top_limb >> (64 - shift)
            "movl {shift:e}, %ecx",                      // Restore %cl = shift

            "leaq -8({limbs},{len},8), {ptr}",           // ptr = &limbs[len-1]
            "movq ({ptr}), %rax",                        // %rax = top limb

            "movq {len}, %rdx",                          // %rdx = len
            "subq $1, %rdx",                             // %rdx = len - 1 (iterations count)
            "jz 2f",                                     // If len == 1, jump directly to bottom limb shift (2f)

            "movq %rdx, %r8",                            // %r8 = len - 1
            "shrq $2, %r8",                              // %r8 = 4-limb chunks
            "jz 3f",                                     // If chunks == 0, go to scalar remainder loop (3f)

            // Main 4-way unrolled descending loop body
            "4:",
            "movq -8({ptr}), %r9",                       // Load limbs[i-1]
            "movq -16({ptr}), %r10",                     // Load limbs[i-2]
            "movq -24({ptr}), %r11",                     // Load limbs[i-3]
            "movq -32({ptr}), %r12",                     // Load limbs[i-4]

            "shldq %cl, %r9, %rax",                      // %rax = (%rax << cl) | (%r9 >> (64-cl))
            "movq %rax, ({ptr})",                        // Store updated limbs[i]
            "shldq %cl, %r10, %r9",                      // %r9 = (%r9 << cl) | (%r10 >> (64-cl))
            "movq %r9, -8({ptr})",                       // Store updated limbs[i-1]
            "shldq %cl, %r11, %r10",                     // %r10 = (%r10 << cl) | (%r11 >> (64-cl))
            "movq %r10, -16({ptr})",                     // Store updated limbs[i-2]
            "shldq %cl, %r12, %r11",                     // %r11 = (%r11 << cl) | (%r12 >> (64-cl))
            "movq %r11, -24({ptr})",                     // Store updated limbs[i-3]

            "movq %r12, %rax",                           // Propagate %r12 as next top limb
            "leaq -32({ptr}), {ptr}",                    // Decrement pointer by 4 limbs (32 bytes)
            "decq %r8",                                  // Decrement chunk counter
            "jnz 4b",                                    // Repeat while chunks != 0

            // Remainder processing (0 to 3 limbs)
            "3:",
            "andq $3, %rdx",                             // %rdx = remainder count (len - 1) & 3
            "jz 2f",                                     // If 0 remainder, jump to bottom limb (2f)

            // 1-limb unrolled tail loop
            "1:",
            "movq -8({ptr}), %r9",                       // Load limbs[i-1]
            "shldq %cl, %r9, %rax",                      // Merge shifted bits
            "movq %rax, ({ptr})",                        // Store updated limb
            "movq %r9, %rax",                            // Update running limb
            "leaq -8({ptr}), {ptr}",                     // Decrement pointer by 1 limb (8 bytes)
            "decq %rdx",                                 // Decrement remainder
            "jnz 1b",

            // [Shift bottom-most limb (no bits pulled below)]
            "2:",
            "shlq %cl, %rax",                            // limbs[0] = limbs[0] << shift
            "movq %rax, ({limbs})",                      // Store limbs[0]

            carry_out = out(reg) carry_out,
            limbs = in(reg) limbs,
            len = in(reg) len,
            shift = in(reg) shift,
            ptr = out(reg) _,
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
