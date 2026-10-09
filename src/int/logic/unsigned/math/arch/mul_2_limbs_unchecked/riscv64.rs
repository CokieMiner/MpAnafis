//! RISC-V 64-bit write-only dual-row multiplication kernel.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! 64x64->128-bit multipliers (`mul`/`mulhu`) and branchless `sltu` carry capture.

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
/// Evaluates two simultaneous multiplication rows in registers without memory reads of `dst`.
/// Both products (`src[j] * s0` and `src[j] * s1`) are computed via `mul`/`mulhu`, and
/// carries are merged branchlessly using `sltu`.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
/// - The CPU must support the M extension.
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
    // writable outputs. The decreasing count permits exactly len source loads
    // and stores, followed by two in-range final stores. Products plus two limbs
    // are <= B^2-1; the final high carry is <= s1. Output is never read. Every
    // modified register is an output, and the backend gate enables the M extension.
    unsafe {
        asm!(
            // Main dual-row multiplication loop
            "1:",
            "ld {value}, 0({src})",                      // Load src[j]

            // [Row 0 Multiplication & Carry Merge]
            "mul {lo0}, {value}, {s0}",                  // Low 64 bits of src[j] * s0
            "mulhu {hi0}, {value}, {s0}",                // High 64 bits of src[j] * s0
            "add {sum0}, {lo0}, {carry0}",               // sum0 = lo0 + carry0
            "sltu {carry_bit0}, {sum0}, {carry0}",       // carry_bit0 = 1 if addition wrapped
            "add {hi0}, {hi0}, {carry_bit0}",            // hi0 += carry_bit0
            "add {out}, {sum0}, {pending1}",             // out = sum0 + pending1 (finalized limb)
            "sltu {carry_bit1}, {out}, {pending1}",      // carry_bit1 = 1 if second addition wrapped
            "add {carry0}, {hi0}, {carry_bit1}",         // carry0 = hi0 + carry_bit1 (row 0 carry)
            "sd {out}, 0({dst})",                        // Store finalized dst[j]

            // [Row 1 Multiplication & Pending Carry Accumulation]
            "mul {lo1}, {value}, {s1}",                  // Low 64 bits of src[j] * s1
            "mulhu {hi1}, {value}, {s1}",                // High 64 bits of src[j] * s1
            "add {pending1}, {lo1}, {carry1}",           // pending1 = lo1 + carry1
            "sltu {carry_bit2}, {pending1}, {carry1}",   // carry_bit2 = 1 if addition wrapped
            "add {carry1}, {hi1}, {carry_bit2}",         // carry1 = hi1 + carry_bit2 (row 1 carry)

            // Advance pointers and loop
            "addi {src}, {src}, 8",                      // Advance src pointer by 8 bytes
            "addi {dst}, {dst}, 8",                      // Advance dst pointer by 8 bytes
            "addi {len}, {len}, -1",                     // Decrement remaining count
            "bnez {len}, 1b",                            // Repeat while len != 0

            // [Epilogue: Store trailing high limbs]
            "add {out}, {pending1}, {carry0}",           // out = pending1 + carry0
            "sltu {carry_bit0}, {out}, {pending1}",      // Detect final wrap
            "add {carry1}, {carry1}, {carry_bit0}",      // carry1 += carry_bit0
            "sd {out}, 0({dst})",                        // Store dst[len]
            "sd {carry1}, 8({dst})",                     // Store final high limb dst[len+1]

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            value = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            sum0 = out(reg) _,
            carry_bit0 = out(reg) _,
            out = out(reg) _,
            carry_bit1 = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            carry_bit2 = out(reg) _,
            options(nostack)
        );
    }
}
