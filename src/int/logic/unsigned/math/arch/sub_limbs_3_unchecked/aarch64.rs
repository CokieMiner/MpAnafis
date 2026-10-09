//! `AArch64` subtraction kernels (inline assembly).
//!
//! Four-limb blocks use paired loads and stores. SBCS carries the inverted
//! borrow: C=1 means no borrow. CMP seeds C=1 and CSET extracts the final bit.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs,
/// returning the final borrow.
///
/// # Safety
///
/// Both sources must cover `len` aligned initialized readable limbs and may
/// alias each other. `dst` must cover `len` aligned writable limbs, disjoint
/// from both sources; its contents may be uninitialized. Each byte span must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the fixed assembly borrow chain to callers")]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: four-limb blocks and the len&3 tail access exactly len aligned
    // initialized source limbs and disjoint writable output limbs. Sources
    // may coincide; both are loaded before each output store. Counter SUB and
    // CB(N)Z preserve C, so SBCS carries the borrow continuously. All changed
    // pointers and temporaries are early outputs; no stack memory is accessed.
    unsafe {
        asm!(
            "cmp xzr, xzr",                    // C = 1 (no initial borrow)
            "cbz {chunks}, 1f",                // skip chunk loop if len < 4
            ".p2align 4",                          // 16-byte loop alignment
            "2:",
            "ldp {src1_v0}, {src1_v1}, [{src1}], #16",  // load src1[0..1], src1 += 16
            "ldp {src2_v0}, {src2_v1}, [{src2}], #16",  // load src2[0..1], src2 += 16
            "ldp {src1_v2}, {src1_v3}, [{src1}], #16",  // load src1[2..3], src1 += 16
            "ldp {src2_v2}, {src2_v3}, [{src2}], #16",  // load src2[2..3], src2 += 16
            "sbcs {src1_v0}, {src1_v0}, {src2_v0}",     // src1[0] -= src2[0] + !C
            "sbcs {src1_v1}, {src1_v1}, {src2_v1}",     // src1[1] -= src2[1] + !C
            "sbcs {src1_v2}, {src1_v2}, {src2_v2}",     // src1[2] -= src2[2] + !C
            "sbcs {src1_v3}, {src1_v3}, {src2_v3}",     // src1[3] -= src2[3] + !C
            "stp {src1_v0}, {src1_v1}, [{dst}], #16",   // store dst[0..1], dst += 16
            "stp {src1_v2}, {src1_v3}, [{dst}], #16",   // store dst[2..3], dst += 16
            "sub {chunks}, {chunks}, #1",               // --chunks
            "cbnz {chunks}, 2b",                        // repeat if chunks != 0

            // --- Tail ---
            "1:",
            "cbz {rem}, 3f",                   // skip tail if rem == 0
            ".p2align 4",                          // 16-byte loop alignment
            "4:",
            "ldr {src1_v0}, [{src1}], #8",      // load src1[i], src1 += 8
            "ldr {src2_v0}, [{src2}], #8",      // load src2[i], src2 += 8
            "sbcs {src1_v0}, {src1_v0}, {src2_v0}", // src1[i] -= src2[i] + !C
            "str {src1_v0}, [{dst}], #8",        // store dst[i], dst += 8
            "sub {rem}, {rem}, #1",              // --rem
            "cbnz {rem}, 4b",                    // repeat if rem != 0
            "3:",
            "cset {borrow}, cc",                 // borrow = (C == 0) ? 1 : 0
            borrow = inout(reg) borrow,
            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_v0 = out(reg) _, src1_v1 = out(reg) _, src1_v2 = out(reg) _, src1_v3 = out(reg) _,
            src2_v0 = out(reg) _, src2_v1 = out(reg) _, src2_v2 = out(reg) _, src2_v3 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
