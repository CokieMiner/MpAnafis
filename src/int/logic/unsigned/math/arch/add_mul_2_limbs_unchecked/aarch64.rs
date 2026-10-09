//! `AArch64` (ARMv8-A / ARMv9-A) fused dual-row multiply-add kernel.
//!
//! One source traversal accumulates `src * s0` and `src * s1 * 2^64`.
//! The updated next destination limb remains in a register between iterations.

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
#[expect(
    clippy::inline_always,
    reason = "Keep both hardware row recurrences in the selected multiplication caller"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;

    if len == 0 {
        return (0, 0);
    }

    // SAFETY: len > 0 bounds each iteration's source[j], dst[j], and dst[j+1]
    // accesses by the initialized spans. Eight-byte advances preserve alignment
    // and stay within their live allocations. Destination writes cannot alias src.
    unsafe {
        asm!(
            "ldr {d_cur}, [{dst}]",                      // Load the initial destination limb

            // Main dual-row accumulation loop
            "1:",
            "ldr {src_val}, [{src}], #8",                // Load src[j] and advance src pointer
            "ldr {d_next}, [{dst}, #8]",                 // Pre-load dst[j+1] for row 1 accumulation

            // Low and high halves of the two scalar products.
            "mul {lo0}, {src_val}, {s0}",                // Low 64 bits of src[j] * s0
            "mul {lo1}, {src_val}, {s1}",                // Low 64 bits of src[j] * s1
            "umulh {hi0}, {src_val}, {s0}",              // High 64 bits of src[j] * s0
            "umulh {hi1}, {src_val}, {s1}",              // High 64 bits of src[j] * s1

            // [Row 0 Carry Chain: Finalize dst[j]]
            "adds {lo0}, {lo0}, {c0}",                   // lo0 += c0, set C flag
            "adc {hi0}, {hi0}, xzr",                     // hi0 += C flag + 0
            "adds {d_cur}, {d_cur}, {lo0}",              // d_cur += lo0, set C flag
            "adc {c0}, {hi0}, xzr",                      // c0 = hi0 + C flag (carry for next row 0 limb)
            "str {d_cur}, [{dst}], #8",                  // Store finalized dst[j] and advance dst pointer

            // [Row 1 Carry Chain: Compute dst[j+1] and carry-forward in d_cur]
            "adds {lo1}, {lo1}, {c1}",                   // lo1 += c1, set C flag
            "adc {hi1}, {hi1}, xzr",                     // hi1 += C flag + 0
            "adds {d_cur}, {d_next}, {lo1}",             // d_cur = d_next + lo1, set C flag
            "adc {c1}, {hi1}, xzr",                      // c1 = hi1 + C flag (carry for next row 1 limb)

            "subs {len}, {len}, #1",                     // Decrement remaining limbs
            "b.ne 1b",                                   // Loop while len != 0

            // [Final Store: Write high accumulated limb dst[len]]
            "str {d_cur}, [{dst}]",                      // Store carry-forwarded dst[len]

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            src_val = out(reg) _,
            d_cur = out(reg) _,
            d_next = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
