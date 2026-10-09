//! x86-64 BMI2 shifted-high subtraction kernel.
//!
//! Subtracts `src >> (64 - shift)` using flag-preserving BMI2 shifts and LEA
//! merges across an uninterrupted SBB borrow chain.

use core::arch::asm;

use super::Limb;

/// Subtract a cross-limb shifted source span from `dst`, including `borrow`.
///
/// For B = 2^64, `dst_new - borrow_out * B^len = dst_old - K - borrow`,
/// where `K = src >> (64 - shift)` with a zero-extended top limb.
///
/// # Safety
///
/// - Both pointers must cover `len` aligned, initialized limbs in disjoint spans.
/// - `dst` requires exclusive access; each span's byte length must fit in `isize`.
/// - Zero length permits null pointers; the CPU must support BMI2.
/// - `0 < shift < 64`.
/// - `borrow <= 1`.
#[expect(
    clippy::as_conversions,
    reason = "shift is strictly below 64 and therefore fits the x86-64 shift-count register"
)]
#[expect(
    clippy::inline_always,
    reason = "inlining exposes the shifted subtraction loop to its quotient-estimation caller"
)]
#[inline(always)]
pub unsafe fn sub_shifted_high_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
    borrow: Limb,
) -> Limb {
    debug_assert!(
        shift > 0 && shift < Limb::BITS,
        "the cross-limb shift must be strictly inside one limb"
    );
    debug_assert!(borrow <= 1, "a subtraction borrow is one bit");
    if len == 0 {
        return borrow;
    }

    let dst_ptr = dst;
    let src_ptr = src;
    // Leave one final limb outside the paired loop. An even length leaves one
    // additional ordinary limb before that final zero-extended limb.
    let even_extra = usize::from(len.is_multiple_of(2));
    let borrow_out: u8;

    // SAFETY: len > 0 makes len - 1 exact. Each pair consumes two lower limbs
    // and reads their successors; even_extra consumes any remaining lower
    // limb before the top limb. All accesses stay in the aligned initialized
    // disjoint spans. Both shift counts lie in 1..64. BMI2 shifts, LEA, MOV,
    // JRCXZ and JMP preserve CF; the disjoint bit ranges make LEA's merge
    // exact. Early clobbers preserve inputs and SETC writes its late output last.
    unsafe {
        let pair_count = len.unchecked_sub(1) >> 1;
        let left_shift = shift as usize;
        let right_shift = (Limb::BITS as usize).unchecked_sub(left_shift);
        asm!(
            "movq ({src_ptr}), %r8",                     // %r8 = src[0]
            "btq $0, {borrow}",                          // Set CF = borrow bit (0 or 1)
            "jrcxz 2f",                                  // If pair_count == 0, skip paired loop (2f)

            // Main 2-limb paired unrolled loop body
            "1:",
            "movq 8({src_ptr}), %r9",                    // Load src[i+1] before its variable-count shift
            "movq ({dst_ptr}), %r13",                    // Load dst[i]
            "movq 16({src_ptr}), %r10",                  // Load src[i+2]
            "shlxq {left_shift}, %r9, %r12",             // %r12 = src[i+1] << shift (flag-preserving)
            "leaq -1(%rcx), %rcx",                       // Decrement pair counter without changing CF
            "shrxq {right_shift}, %r8, %r11",            // %r11 = src[i] >> (64 - shift)
            "leaq (%r11,%r12), %r11",                    // %r11 = merged cross-limb word (flag-preserving)
            "sbbq %r11, %r13",                           // dst[i] - r11 - CF -> updates CF (borrow)
            "leaq 16({src_ptr}), {src_ptr}",             // Advance after loading both following limbs
            "shlxq {left_shift}, %r10, %r12",            // %r12 = src[i+2] << shift
            "movq %r13, ({dst_ptr})",                    // Store updated dst[i]
            "movq %r10, %r8",                            // Carry src[i+2] into the next iteration
            "movq 8({dst_ptr}), %r13",                   // Load dst[i+1]
            "shrxq {right_shift}, %r9, %r11",            // %r11 = src[i+1] >> (64 - shift)

            "leaq (%r11,%r12), %r11",                    // %r11 = merged cross-limb word
            "sbbq %r11, %r13",                           // dst[i+1] - r11 - CF -> updates CF
            "movq %r13, 8({dst_ptr})",                   // Store updated dst[i+1]
            "leaq 16({dst_ptr}), {dst_ptr}",             // Advance dst by 2 limbs (16 bytes)
            "jrcxz 2f",                                  // If %rcx == 0, exit loop
            "jmp 1b",                                    // Repeat loop

            // Even-length extra limb handling
            "2:",
            "movq {even_extra}, %rcx",                   // %rcx = even_extra
            "jrcxz 3f",                                  // If 0, skip to final limb (3f)
            "movq 8({src_ptr}), %r9",                    // Load src[i+1]
            "shrxq {right_shift}, %r8, %r11",            // %r11 = src[i] >> (64 - shift)
            "shlxq {left_shift}, %r9, %r12",             // %r12 = src[i+1] << shift
            "leaq (%r11,%r12), %r11",                    // Merged cross-limb word
            "movq ({dst_ptr}), %r13",                    // Load dst[i]
            "sbbq %r11, %r13",                           // Subtract with borrow
            "movq %r13, ({dst_ptr})",                    // Store updated dst[i]
            "movq %r9, %r8",                             // Update %r8
            "leaq 8({src_ptr}), {src_ptr}",              // Advance src by 1 limb
            "leaq 8({dst_ptr}), {dst_ptr}",              // Advance dst by 1 limb

            // [Final zero-extended top limb]
            "3:",
            "shrxq {right_shift}, %r8, %r11",            // %r11 = top src limb >> (64 - shift)
            "movq ({dst_ptr}), %r13",                    // Load top dst limb
            "sbbq %r11, %r13",                           // Final subtraction with borrow
            "movq %r13, ({dst_ptr})",                    // Store final dst limb
            "setc {borrow_out}",                         // Extract final borrow bit (1 if CF=1, else 0)

            src_ptr = inout(reg) src_ptr => _,
            dst_ptr = inout(reg) dst_ptr => _,
            inout("rcx") pair_count => _,
            even_extra = in(reg) even_extra,
            left_shift = in(reg) left_shift,
            right_shift = in(reg) right_shift,
            borrow = in(reg) borrow,
            borrow_out = lateout(reg_byte) borrow_out,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            out("r12") _,
            out("r13") _,
            options(nostack, att_syntax)
        );
    }
    Limb::from(borrow_out)
}
