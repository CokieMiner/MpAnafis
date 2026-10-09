//! PowerPC 64-bit write-only dual-row multiplication kernel.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! 64x64->128-bit multipliers (`mulld`/`mulhdu`) and hardware CTR loop branching (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] x (s0 + s1 x 2^64)
/// ```
///
/// Computes both 64x64->128-bit products with four `mulld`/`mulhdu` instructions,
/// merges row 0 and row 1 carry chains using `addc`/`addze`, and utilizes load/store-update
/// (`ldu`/`stdu`). `bdnz` decrements CTR and branches on a nonzero count.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
#[expect(
    clippy::inline_always,
    reason = "inlining joins the two-row initializer to complete basecase multiplication"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    let carry0: Limb = 0;
    let carry1: Limb = 0;
    let pending1: Limb = 0;

    // SAFETY: disjoint aligned spans provide len initialized inputs and len+2
    // writable outputs. The preadjusted assembly addresses are incremented
    // before dereferencing; CTR permits exactly len source loads and stores,
    // followed by two in-range final stores. Products plus two limbs are <=
    // B^2-1; the final high carry is <= s1. Output is never read. General
    // register clobbers, CTR, and XER are declared as outputs.
    unsafe {
        asm!(
            "addi {src}, {src}, -8",                     // Adjust pointer for ldu pre-increment
            "addi {dst}, {dst}, -8",                     // Adjust pointer for stdu pre-increment
            "mtctr {len}",                               // Load loop count into hardware CTR register

            ".p2align 4",
            // Main dual-row multiplication loop
            "1:",
            "ldu {value}, 8({src})",                     // Load src[j] and update src pointer (+8)

            // Two products split into four low/high multiplication instructions.
            "mulld {lo0}, {value}, {s0}",                // Low 64 bits of src[j] * s0
            "mulhdu {hi0}, {value}, {s0}",               // High 64 bits of src[j] * s0
            "mulld {lo1}, {value}, {s1}",                // Low 64 bits of src[j] * s1
            "mulhdu {hi1}, {value}, {s1}",               // High 64 bits of src[j] * s1

            // [Row 0 Carry Chain & Row 1 Pending Merge]
            "addc {sum0}, {lo0}, {carry0}",              // sum0 = lo0 + carry0, set CA bit in XER
            "addze {hi0}, {hi0}",                        // hi0 += CA bit
            "addc {out}, {sum0}, {pending1}",            // out = sum0 + pending1, set CA bit
            "addze {carry0}, {hi0}",                     // carry0 = hi0 + CA bit (row 0 carry)

            // [Row 1 Carry Chain: Compute next pending1]
            "addc {pending1}, {lo1}, {carry1}",          // pending1 = lo1 + carry1, set CA bit
            "addze {carry1}, {hi1}",                     // carry1 = hi1 + CA bit (row 1 carry)

            "stdu {out}, 8({dst})",                      // Store finalized dst[j] and update dst pointer (+8)
            "bdnz 1b",                                   // Decrement CTR and branch if != 0

            // [Epilogue: Flush trailing high row 1 limb + remaining carry]
            "addc {pending1}, {pending1}, {carry0}",     // pending1 += carry0, set CA bit
            "addze {carry1}, {carry1}",                  // carry1 += CA bit
            "std {pending1}, 8({dst})",                  // Store dst[len]
            "std {carry1}, 16({dst})",                   // Store final high limb dst[len+1]

            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            value = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            sum0 = out(reg) _,
            out = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            options(nostack)
        );
    }
}
