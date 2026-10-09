//! `AArch64` subtraction kernels (inline assembly).
//!
//! Four-limb blocks use paired loads and stores. SBCS carries the inverted
//! borrow: C=1 means no borrow. CMP seeds C=1 and CSET extracts the final bit.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// `dst_after - borrow*B^len = dst_before - src`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// `dst` must cover `len` aligned initialized writable limbs. `src` must cover
/// `len` aligned initialized readable limbs. The spans must be identical or
/// disjoint, and each byte span must fit in `isize::MAX`. `len == 0` performs
/// no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the four-limb assembly borrow chain to callers")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: four-limb blocks and the len&3 tail access exactly len initialized
    // aligned limbs. Each block loads all source values before corresponding
    // writes, permitting exact alias. SUB and CB(N)Z preserve C between SBCS
    // instructions. All changed pointers and temporaries are early outputs.
    unsafe {
        asm!(
            "cmp xzr, xzr",                    // C = 1 (no initial borrow)
            "cbz {chunks}, 1f",                // skip chunk loop if len < 4
            ".p2align 4",                          // 16-byte loop alignment
            "2:",
            "ldp {src_v0}, {src_v1}, [{src}], #16",   // load src[0..1], src += 16
            "ldp {dst_v0}, {dst_v1}, [{dst}]",         // load dst[0..1]
            "ldp {src_v2}, {src_v3}, [{src}], #16",    // load src[2..3], src += 16
            "ldp {dst_v2}, {dst_v3}, [{dst}, #16]",    // load dst[2..3] (offset 16)
            "sbcs {dst_v0}, {dst_v0}, {src_v0}",       // dst[0] -= src[0] + !C
            "sbcs {dst_v1}, {dst_v1}, {src_v1}",       // dst[1] -= src[1] + !C
            "sbcs {dst_v2}, {dst_v2}, {src_v2}",       // dst[2] -= src[2] + !C
            "sbcs {dst_v3}, {dst_v3}, {src_v3}",       // dst[3] -= src[3] + !C
            "stp {dst_v0}, {dst_v1}, [{dst}], #16",    // store dst[0..1], dst += 16
            "stp {dst_v2}, {dst_v3}, [{dst}], #16",    // store dst[2..3], dst += 16
            "sub {chunks}, {chunks}, #1",              // --chunks
            "cbnz {chunks}, 2b",                       // repeat if chunks != 0

            // --- Tail ---
            "1:",
            "cbz {rem}, 3f",                   // skip tail if rem == 0
            ".p2align 4",                          // 16-byte loop alignment
            "4:",
            "ldr {src_v0}, [{src}], #8",       // load src[i], src += 8
            "ldr {dst_v0}, [{dst}]",            // load dst[i]
            "sbcs {dst_v0}, {dst_v0}, {src_v0}", // dst[i] -= src[i] + !C
            "str {dst_v0}, [{dst}], #8",        // store dst[i], dst += 8
            "sub {rem}, {rem}, #1",             // --rem
            "cbnz {rem}, 4b",                   // repeat if rem != 0
            "3:",
            "cset {borrow}, cc",                // borrow = (C == 0) ? 1 : 0
            borrow = inout(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
