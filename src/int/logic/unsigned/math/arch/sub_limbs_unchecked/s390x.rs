//! `s390x` (IBM Z) subtraction kernels (inline assembly).
//!
//! Two-limb SLBGR blocks carry the borrow in the condition code (CC), followed
//! by a single-limb tail. CC=0/1 means borrow; CC=2/3 means no borrow.
//! SLGR of a register with itself seeds CC=2. LA and BRCTG preserve CC.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// `dst_after - borrow*B^len = dst_before - src`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both pointers must cover `len` aligned initialized limbs; `dst` must be
/// writable. Spans must be identical or disjoint and their byte widths must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the two-limb assembly borrow chain to callers")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: two-limb blocks and the len&1 tail cover exactly len aligned
    // initialized limbs. Each source is read before its matching write,
    // permitting exact alias. SLGR seeds no borrow; LA/BRCTG preserve CC.
    // The tail guard maps rem=0 to -1 (skip) and rem=1 to zero (execute).
    // Every changed pointer and temporary is an early output.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",          // skip main loop if chunks == 0

            // chunks > 0: seed no borrow before the main loop
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",          // CC = 2 (no borrow)

            ".p2align 4",                          // 16-byte loop alignment
            "2:",
            // Process limb 0
            "lg {src_val0}, 0({src})",          // load src[0]
            "lg {dst_val0}, 0({dst})",          // load dst[0]
            "slbgr {dst_val0}, {src_val0}",     // dst[0] = dst[0] - src[0] - borrow
            "stg {dst_val0}, 0({dst})",         // store dst[0]
            // Process limb 1
            "lg {src_val1}, 8({src})",          // load src[1]
            "lg {dst_val1}, 8({dst})",          // load dst[1]
            "slbgr {dst_val1}, {src_val1}",     // dst[1] = dst[1] - src[1] - borrow
            "stg {dst_val1}, 8({dst})",         // store dst[1]
            "la {src}, 16({src})",              // src += 16
            "la {dst}, 16({dst})",              // dst += 16
            "brctg {chunks}, 2b",               // --chunks; branch if != 0 (preserves CC)
            "j 4f",                              // skip CC re-init (main loop done, CC already set)

            "1:",                                 // chunks == 0: seed no borrow for tail
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",           // CC = 2 (no borrow)

            "4:",                                 // common: CC is ready
            "brctg {rem}, 3f",                  // if rem was 0 -> wrap to MAX, branch (skip tail); preserves CC
            // Process remainder limb
            "lg {src_val0}, 0({src})",
            "lg {dst_val0}, 0({dst})",
            "slbgr {dst_val0}, {src_val0}",
            "stg {dst_val0}, 0({dst})",
            "3:",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",         // borrow = -borrow_from_CC (0 or -1)
            "lcgr {borrow}, {borrow}",          // two's complement -> 0 or 1
            borrow = out(reg) borrow,
            dst = inout(reg_addr) dst => _,
            src = inout(reg_addr) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
