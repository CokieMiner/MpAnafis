//! `AArch64` (ARMv8-A / ARMv9-A) fused multiply-subtract limb kernel.
//!
//! Uses 64x64->128-bit unsigned multipliers (`mul`/`umulh`), paired memory operations
//! (`ldp`/`stp`), and condition-code borrow propagation (`cmp`/`sbcs`/`cset cc`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^64, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Two-limb blocks use `cmp xzr, borrow` to restore C = 1 - borrow before `sbcs`.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and unrolled assembly paths share one call boundary in the arithmetic hot path"
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
        // SAFETY: the caller's aligned initialized disjoint spans contain 1..=3
        // limbs. Each iteration advances both pointers by one limb. The first
        // product needs no carry addition; later iterations restore C from
        // the saved borrow before sbcs, independently of product carry.
        unsafe {
            asm!(
                "ldr {s}, [{src}], #8",
                "ldr {d}, [{dst}]",
                "mul {lo}, {s}, {scalar}",
                "umulh {carry}, {s}, {scalar}",
                "subs {d}, {d}, {lo}",
                "cset {borrow}, cc",
                "str {d}, [{dst}], #8",
                "subs {count}, {count}, #1",
                "b.eq 3f",
                "2:",
                "ldr {s}, [{src}], #8",
                "ldr {d}, [{dst}]",
                "mul {lo}, {s}, {scalar}",
                "umulh {hi}, {s}, {scalar}",
                "adds {lo}, {lo}, {carry}",
                "adc {carry}, {hi}, xzr",
                "cmp xzr, {borrow}",
                "sbcs {d}, {d}, {lo}",
                "cset {borrow}, cc",
                "str {d}, [{dst}], #8",
                "subs {count}, {count}, #1",
                "b.ne 2b",
                "3:",
                src = inout(reg) src => _,
                dst = inout(reg) dst => _,
                count = inout(reg) len => _,
                scalar = in(reg) scalar,
                carry = out(reg) carry,
                borrow = out(reg) borrow,
                s = out(reg) _,
                d = out(reg) _,
                lo = out(reg) _,
                hi = out(reg) _,
                options(nostack),
            );
        }
        return (carry, borrow);
    }
    let mut carry_hi: Limb = 0;
    let mut borrow: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY: len >= 4 gives at least two complete pairs, followed by at most
    // one limb, within the aligned initialized disjoint spans. Product plus
    // carry is below B^2. ADC consumes product carry before CMP restores
    // C = 1 - borrow for SBCS. All temporaries are early outputs, distinct
    // from the live inputs; default asm options declare flag modification.
    unsafe {
        asm!(
            // Main 2-way unrolled loop body
            "2:",

            // Load two source and destination limbs.
            "ldp {src_val0}, {src_val1}, [{src}], #16",  // Load src[0..2] and advance pointer by 16 bytes
            "ldp {dst_val0}, {dst_val1}, [{dst}]",        // Load dst[0..2]

            // [Limb 0 Multiply-Subtract]
            "mul {p_lo0}, {src_val0}, {scalar}",          // p_lo0 = low 64 bits of src[0] * scalar
            "umulh {p_hi0}, {src_val0}, {scalar}",        // p_hi0 = high 64 bits of src[0] * scalar
            "adds {p_lo0}, {p_lo0}, {carry_hi}",          // p_lo0 += carry_hi, set C flag
            "adc {carry_hi}, {p_hi0}, xzr",              // carry_hi = p_hi0 + C
            "cmp xzr, {borrow}",                         // C = 1 - incoming borrow
            "sbcs {dst_val0}, {dst_val0}, {p_lo0}",       // dst[0] -= p_lo0 + borrow
            "cset {borrow}, cc",                         // Capture the combined subtraction borrow

            // [Limb 1 Multiply-Subtract]
            "mul {p_lo1}, {src_val1}, {scalar}",          // Low 64 bits of src[1] * scalar
            "umulh {p_hi1}, {src_val1}, {scalar}",        // High 64 bits of src[1] * scalar
            "adds {p_lo1}, {p_lo1}, {carry_hi}",          // p_lo1 += carry_hi
            "adc {carry_hi}, {p_hi1}, xzr",              // carry_hi = p_hi1 + C
            "cmp xzr, {borrow}",                         // Restore the incoming subtraction borrow
            "sbcs {dst_val1}, {dst_val1}, {p_lo1}",       // dst[1] -= p_lo1 + borrow
            "cset {borrow}, cc",                         // Capture the combined subtraction borrow

            // Store two result limbs.
            "stp {dst_val0}, {dst_val1}, [{dst}], #16",  // Store updated limbs and advance dst pointer
            "sub {chunks}, {chunks}, #1",                 // Decrement chunk counter
            "cbnz {chunks}, 2b",                          // Loop if chunks != 0

            // Remainder processing (0 or 1 limb)
            "1:",
            "cbz {rem}, 3f",                              // If rem == 0, skip to end (3f)

            // 1-limb tail
            "ldr {src_val0}, [{src}], #8",                // Load single src limb
            "ldr {dst_val0}, [{dst}]",                    // Load single dst limb
            "mul {p_lo0}, {src_val0}, {scalar}",          // Low 64 bits
            "umulh {p_hi0}, {src_val0}, {scalar}",        // High 64 bits
            "adds {p_lo0}, {p_lo0}, {carry_hi}",          // Add carry_hi
            "adc {carry_hi}, {p_hi0}, xzr",              // Propagate product carry
            "cmp xzr, {borrow}",                         // Restore the incoming subtraction borrow
            "sbcs {dst_val0}, {dst_val0}, {p_lo0}",       // Subtract product and incoming borrow
            "cset {borrow}, cc",                         // Capture the combined subtraction borrow
            "str {dst_val0}, [{dst}], #8",                // Store updated limb

            // Tail completion
            "3:",

            carry_hi = inout(reg) carry_hi,
            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_hi, borrow)
}
