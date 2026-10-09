//! PowerPC 32-bit multiply-subtract with four-limb blocks.
//!
//! Uses product halves (`mullw`/`mulhwu`), addition with carry (`addc`/`addze`)
//! for the high product row, and subtraction with borrow (`subfic`/`subfe`) for destination updates.

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, subtract the result from `dst`,
/// and return the final `(carry, borrow)` pair.
///
/// For B = 2^32, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// `subfic` restores CA = 1 - borrow from the saved zero or all-ones mask.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and four-limb assembly paths share one call boundary"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len < 4 {
        if len == 0 {
            return (0, 0);
        }
        let carry: Limb;
        let borrow: Limb;
        // SAFETY: the caller's aligned initialized disjoint spans contain 1..=3 limbs.
        // CTR bounds the accesses and pointer advances are four bytes. XER.CA
        // is restored from the saved borrow before subtraction, separately
        // from product carry; both modified special registers are clobbered.
        unsafe {
            asm!(
                "mtctr {count}",
                "lwz {s}, 0({src})",
                "lwz {d}, 0({dst})",
                "mullw {lo}, {s}, {scalar}",
                "mulhwu {carry}, {s}, {scalar}",
                "subfc {d}, {lo}, {d}",
                "subfe {borrow}, {borrow}, {borrow}",
                "neg {borrow}, {borrow}",
                "stw {d}, 0({dst})",
                "bdz 3f",
                "2:",
                "addi {src}, {src}, 4",
                "addi {dst}, {dst}, 4",
                "lwz {s}, 0({src})",
                "lwz {d}, 0({dst})",
                "mullw {lo}, {s}, {scalar}",
                "mulhwu {hi}, {s}, {scalar}",
                "addc {lo}, {lo}, {carry}",
                "addze {carry}, {hi}",
                "subfic {s}, {borrow}, 0",
                "subfe {d}, {lo}, {d}",
                "subfe {borrow}, {borrow}, {borrow}",
                "neg {borrow}, {borrow}",
                "stw {d}, 0({dst})",
                "bdnz 2b",
                "3:",
                src = inout(reg_nonzero) src => _,
                dst = inout(reg_nonzero) dst => _,
                count = in(reg) len,
                scalar = in(reg) scalar,
                carry = out(reg) carry,
                borrow = out(reg) borrow,
                s = out(reg) _,
                d = out(reg) _,
                lo = out(reg) _,
                hi = out(reg) _,
                out("ctr") _,
                out("xer") _,
                options(nostack),
            );
        }
        return (carry, borrow);
    }
    let mut carry_hi: Limb = 0;
    let mut borrow_reg: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len >= 4 gives at least one four-limb block and a shorter tail
    // in the aligned initialized disjoint spans. CTR bounds every block and
    // tail access. Product plus carry is below B^2; SUBFIC restores CA from
    // the saved mask after ADDZE consumes product carry. Early outputs are
    // distinct from live inputs; CTR, XER and CR0 modifications are declared.
    unsafe {
        asm!(
            "mtctr {chunks}",             // Load loop counter into hardware CTR register

            ".p2align 4",
            // Main 4-way unrolled loop body
            "2:",

            // [Load 4 Source and 4 Destination Limbs (32 bits each)]
            "lwz {src_v0}, 0({src})",     // Load src[0]
            "lwz {src_v1}, 4({src})",     // Load src[1]
            "lwz {src_v2}, 8({src})",     // Load src[2]
            "lwz {src_v3}, 12({src})",    // Load src[3]
            "lwz {dst_v0}, 0({dst})",     // Load dst[0]
            "lwz {dst_v1}, 4({dst})",     // Load dst[1]
            "lwz {dst_v2}, 8({dst})",     // Load dst[2]
            "lwz {dst_v3}, 12({dst})",    // Load dst[3]

            // Form four independent product pairs.
            "mullw {p_lo0}, {src_v0}, {scalar}",          // Low 32 bits of src[0] * scalar
            "mulhwu {p_hi0}, {src_v0}, {scalar}",         // High 32 bits of src[0] * scalar
            "mullw {p_lo1}, {src_v1}, {scalar}",          // Low 32 bits of src[1] * scalar
            "mulhwu {p_hi1}, {src_v1}, {scalar}",         // High 32 bits of src[1] * scalar
            "mullw {p_lo2}, {src_v2}, {scalar}",          // Low 32 bits of src[2] * scalar
            "mulhwu {p_hi2}, {src_v2}, {scalar}",         // High 32 bits of src[2] * scalar
            "mullw {p_lo3}, {src_v3}, {scalar}",          // Low 32 bits of src[3] * scalar
            "mulhwu {p_hi3}, {src_v3}, {scalar}",         // High 32 bits of src[3] * scalar

            // [Limb 0 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo0}, {p_lo0}, {carry_hi}",          // p_lo0 += carry_hi, set CA
            "addze {carry_hi}, {p_hi0}",                  // carry_hi = p_hi0 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",          // dst_v0 = dst_v0 - p_lo0 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "stw {dst_v0}, 0({dst})",                     // Store updated dst[0]

            // [Limb 1 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo1}, {p_lo1}, {carry_hi}",          // p_lo1 += carry_hi
            "addze {carry_hi}, {p_hi1}",                  // carry_hi = p_hi1 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v1}, {p_lo1}, {dst_v1}",          // dst_v1 = dst_v1 - p_lo1 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "stw {dst_v1}, 4({dst})",                     // Store updated dst[1]

            // [Limb 2 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo2}, {p_lo2}, {carry_hi}",          // p_lo2 += carry_hi
            "addze {carry_hi}, {p_hi2}",                  // carry_hi = p_hi2 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v2}, {p_lo2}, {dst_v2}",          // dst_v2 = dst_v2 - p_lo2 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "stw {dst_v2}, 8({dst})",                     // Store updated dst[2]

            // [Limb 3 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo3}, {p_lo3}, {carry_hi}",          // p_lo3 += carry_hi
            "addze {carry_hi}, {p_hi3}",                  // carry_hi = p_hi3 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v3}, {p_lo3}, {dst_v3}",          // dst_v3 = dst_v3 - p_lo3 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "stw {dst_v3}, 12({dst})",                    // Store updated dst[3]

            // Advance pointers by 4 limbs (16 bytes) and loop via CTR
            "addi {src}, {src}, 16",
            "addi {dst}, {dst}, 16",
            "bdnz 2b",                                    // Decrement CTR and loop if != 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",
            "cmpwi {rem}, 0",
            "beq 3f",
            "mtctr {rem}",                                // Load remainder count into CTR
            "addi {src}, {src}, -4",
            "addi {dst}, {dst}, -4",

            ".p2align 4",
            // 1-limb unrolled tail loop
            "4:",
            "lwzu {src_v0}, 4({src})",                    // Load src limb and update pointer (+4)
            "lwzu {dst_v0}, 4({dst})",                    // Load dst limb and update pointer (+4)
            "mullw {p_lo0}, {src_v0}, {scalar}",          // Low 32-bit product
            "mulhwu {p_hi0}, {src_v0}, {scalar}",         // High 32-bit product
            "addc {p_lo0}, {p_lo0}, {carry_hi}",          // Add carry_hi
            "addze {carry_hi}, {p_hi0}",                  // Update carry_hi
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",          // Subtract product + borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Update borrow mask
            "stw {dst_v0}, 0({dst})",                     // Store updated limb
            "bdnz 4b",                                    // Loop if != 0

            // Tail completion
            "3:",
            "neg {borrow_reg}, {borrow_reg}",             // Convert mask (-1 -> 1, 0 -> 0)

            carry_hi = inout(reg) carry_hi,
            borrow_reg = inout(reg) borrow_reg,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
            scalar = in(reg) scalar,
            src_v0 = out(reg) _,
            src_v1 = out(reg) _,
            src_v2 = out(reg) _,
            src_v3 = out(reg) _,
            dst_v0 = out(reg) _,
            dst_v1 = out(reg) _,
            dst_v2 = out(reg) _,
            dst_v3 = out(reg) _,
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            p_lo2 = out(reg) _,
            p_hi2 = out(reg) _,
            p_lo3 = out(reg) _,
            p_hi3 = out(reg) _,
            temp = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (carry_hi, borrow_reg)
}
