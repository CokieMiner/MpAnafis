//! `PowerPC64` multiply-add limb kernel.

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
/// This is the `PowerPC64` inline assembly implementation utilizing the `mulld` and
/// `mulhdu` instructions for 64x64->128-bit multiplication, and `addc`/`addze` for
/// accumulation and carry propagation.
///
/// The loop is **4-way unrolled** for optimal performance, utilizing the CTR register
/// for zero-overhead loop control.
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
            ".p2align 4",

            "2:",                         // --- Unrolled Loop x4 ---

            // Load all four limbs
            "ld {src_v0}, 0({src})",
            "ld {src_v1}, 8({src})",
            "ld {src_v2}, 16({src})",
            "ld {src_v3}, 24({src})",
            "ld {dst_v0}, 0({dst})",
            "ld {dst_v1}, 8({dst})",
            "ld {dst_v2}, 16({dst})",
            "ld {dst_v3}, 24({dst})",

            // Hoist all eight multiplies — four independent mulld/mulhdu pairs
            "mulld {p_lo0}, {src_v0}, {scalar}",
            "mulhdu {p_hi0}, {src_v0}, {scalar}",
            "mulld {p_lo1}, {src_v1}, {scalar}",
            "mulhdu {p_hi1}, {src_v1}, {scalar}",
            "mulld {p_lo2}, {src_v2}, {scalar}",
            "mulhdu {p_hi2}, {src_v2}, {scalar}",
            "mulld {p_lo3}, {src_v3}, {scalar}",
            "mulhdu {p_hi3}, {src_v3}, {scalar}",

            // Strictly serial CA chain — products already in flight
            "addc {p_lo0}, {p_lo0}, {carry}",
            "addze {p_hi0}, {p_hi0}",
            "addc {dst_v0}, {dst_v0}, {p_lo0}",
            "addze {carry}, {p_hi0}",
            "std {dst_v0}, 0({dst})",

            "addc {p_lo1}, {p_lo1}, {carry}",
            "addze {p_hi1}, {p_hi1}",
            "addc {dst_v1}, {dst_v1}, {p_lo1}",
            "addze {carry}, {p_hi1}",
            "std {dst_v1}, 8({dst})",

            "addc {p_lo2}, {p_lo2}, {carry}",
            "addze {p_hi2}, {p_hi2}",
            "addc {dst_v2}, {dst_v2}, {p_lo2}",
            "addze {carry}, {p_hi2}",
            "std {dst_v2}, 16({dst})",

            "addc {p_lo3}, {p_lo3}, {carry}",
            "addze {p_hi3}, {p_hi3}",
            "addc {dst_v3}, {dst_v3}, {p_lo3}",
            "addze {carry}, {p_hi3}",
            "std {dst_v3}, 24({dst})",

            "addi {src}, {src}, 32",
            "addi {dst}, {dst}, 32",
            "bdnz 2b",                    // --CTR; loop if CTR != 0

            "1:",                         // --- Remainder Loop ---
            "cmpldi {rem}, 0",
            "beq 3f",                     // skip tail if rem == 0
            "mtctr {rem}",                // CTR = rem
            "addi {src}, {src}, -8",
            "addi {dst}, {dst}, -8",

            ".p2align 4",
            "4:",
            "ldu {src_v0}, 8({src})",
            "ldu {dst_v0}, 8({dst})",
            "mulld {p_lo0}, {src_v0}, {scalar}",
            "mulhdu {p_hi0}, {src_v0}, {scalar}",
            "addc {p_lo0}, {p_lo0}, {carry}",
            "addze {p_hi0}, {p_hi0}",
            "addc {dst_v0}, {dst_v0}, {p_lo0}",
            "addze {carry}, {p_hi0}",
            "std {dst_v0}, 0({dst})",

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
            p_lo0 = out(reg) _, p_lo1 = out(reg) _, p_lo2 = out(reg) _, p_lo3 = out(reg) _,
            p_hi0 = out(reg) _, p_hi1 = out(reg) _, p_hi2 = out(reg) _, p_hi3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    carry
}
