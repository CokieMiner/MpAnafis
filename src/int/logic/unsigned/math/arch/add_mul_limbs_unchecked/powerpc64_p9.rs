//! `PowerPC64` POWER9 multiply-add limb kernel using `maddld`/`maddhdu`.
//!
//! On ISA 3.0 (POWER9+), the fused `maddld`/`maddhdu` instructions compute a
//! full 128‑bit multiply‑accumulate in two instructions instead of four,
//! reducing the per‑limb arithmetic from 6 operations to 4.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// This is the `PowerPC64` POWER9 inline assembly implementation utilizing the
/// `maddld` and `maddhdu` instructions for fused multiply‑add, and `adde`/`addze`
/// for carry propagation.
///
/// The loop is **4‑way unrolled** for optimal performance, utilizing the CTR
/// register for zero‑overhead loop control.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Caller guarantees dst and src are valid for `len` elements.
    unsafe {
        asm!(
            "cmpldi {chunks}, 0",
            "beq 1f",                     // skip chunk loop if chunks == 0
            "mtctr {chunks}",             // CTR = chunks

            "2:",                         // --- Unrolled Loop x4 ---

            // Load all four limb pairs
            "ld {src_v0}, 0({src})",
            "ld {src_v1}, 8({src})",
            "ld {src_v2}, 16({src})",
            "ld {src_v3}, 24({src})",
            "ld {dst_v0}, 0({dst})",
            "ld {dst_v1}, 8({dst})",
            "ld {dst_v2}, 16({dst})",
            "ld {dst_v3}, 24({dst})",

            // Limb 0: maddld → maddhdu → addc → addze chain
            "maddld {t0}, {src_v0}, {scalar}, {dst_v0}",
            "maddhdu {c0}, {src_v0}, {scalar}, {dst_v0}",
            "addc {dst_v0}, {t0}, {carry}",
            "addze {carry}, {c0}",
            "std {dst_v0}, 0({dst})",

            // Limb 1
            "maddld {t1}, {src_v1}, {scalar}, {dst_v1}",
            "maddhdu {c1}, {src_v1}, {scalar}, {dst_v1}",
            "addc {dst_v1}, {t1}, {carry}",
            "addze {carry}, {c1}",
            "std {dst_v1}, 8({dst})",

            // Limb 2
            "maddld {t2}, {src_v2}, {scalar}, {dst_v2}",
            "maddhdu {c2}, {src_v2}, {scalar}, {dst_v2}",
            "addc {dst_v2}, {t2}, {carry}",
            "addze {carry}, {c2}",
            "std {dst_v2}, 16({dst})",

            // Limb 3
            "maddld {t3}, {src_v3}, {scalar}, {dst_v3}",
            "maddhdu {c3}, {src_v3}, {scalar}, {dst_v3}",
            "addc {dst_v3}, {t3}, {carry}",
            "addze {carry}, {c3}",
            "std {dst_v3}, 24({dst})",

            "addi {src}, {src}, 32",
            "addi {dst}, {dst}, 32",
            "bdnz 2b",                    // --CTR; loop if CTR != 0

            "1:",                         // --- Remainder Loop ---
            "cmpldi {rem}, 0",
            "beq 3f",                     // skip tail if rem == 0
            "mtctr {rem}",                // CTR = rem

            "4:",
            "ld {src_v0}, 0({src})",
            "ld {dst_v0}, 0({dst})",
            "maddld {t0}, {src_v0}, {scalar}, {dst_v0}",
            "maddhdu {c0}, {src_v0}, {scalar}, {dst_v0}",
            "addc {dst_v0}, {t0}, {carry}",
            "addze {carry}, {c0}",
            "std {dst_v0}, 0({dst})",

            "addi {src}, {src}, 8",
            "addi {dst}, {dst}, 8",
            "bdnz 4b",

            "3:",                         // --- End ---

            carry = inout(reg) carry,
            dst = inout(reg_nonzero) dst => _,
            src = inout(reg_nonzero) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = in(reg) scalar,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _,
            c0 = out(reg) _, c1 = out(reg) _, c2 = out(reg) _, c3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    carry
}
