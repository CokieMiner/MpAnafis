//! x86-64 addition into a disjoint destination.
//!
//! `adcq` propagates CF through eight-limb blocks and a 4/2/1 tail.
//! `movq`, `leaq`, `decq`, and branches preserve CF between additions.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Writes the low `len` limbs of `src1 + src2` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// For nonempty spans, all pointers must be aligned and cover `len` limbs.
/// Each span must lie within one live allocation and have at most `isize::MAX` bytes.
/// The sources must be initialized and readable; the destination must be
/// writable and disjoint from both sources. Its previous contents are unused.
/// The two source spans may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep the hardware carry chain in the selected arithmetic caller"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Explicit assembly sequence unrolled across 8 limbs and 4/2/1 tail blocks"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized source spans.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 bounds the aligned, writable destination span.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: the caller guarantees all pointers cover `len` limbs, and
        // this branch proves 2 <= len <= 4; dst is disjoint from both sources.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 3;
    let any_tail = len & 7;
    let tail_4 = (len >> 2) & 1;
    let tail_2 = (len >> 1) & 1;
    let tail_1 = len & 1;

    // SAFETY: eight-limb blocks and the binary tail sum to len, bounding every
    // aligned read and write. Signed counters fit because each span is at most
    // isize::MAX bytes. Empty spans skip memory access. Both sources are
    // initialized and may overlap; dst is disjoint and only written. All
    // modified GPRs are declared; pointer advances stop at the span ends.
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}",                  // Clear CF and initialize carry register to 0
            // Main 8-way unrolled loop
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movq ({src1}), %rax",                       // Load src1[0]
            "movq 8({src1}), %rcx",                      // Load src1[1]
            "adcq ({src2}), %rax",                       // %rax = src1[0] + src2[0] + CF
            "adcq 8({src2}), %rcx",                      // %rcx = src1[1] + src2[1] + CF
            "movq 16({src1}), %rdx",                     // Load src1[2]
            "movq 24({src1}), %r8",                      // Load src1[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "adcq 16({src2}), %rdx",                     // %rdx = src1[2] + src2[2] + CF
            "adcq 24({src2}), %r8",                      // %r8 = src1[3] + src2[3] + CF
            "movq 32({src1}), %rax",                     // Load src1[4]
            "movq 40({src1}), %rcx",                     // Load src1[5]
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "adcq 32({src2}), %rax",                     // %rax = src1[4] + src2[4] + CF
            "adcq 40({src2}), %rcx",                     // %rcx = src1[5] + src2[5] + CF
            "movq 48({src1}), %rdx",                     // Load src1[6]
            "movq 56({src1}), %r8",                      // Load src1[7]
            "movq %rax, 32({dst})",                      // Store dst[4]
            "movq %rcx, 40({dst})",                      // Store dst[5]
            "adcq 48({src2}), %rdx",                     // %rdx = src1[6] + src2[6] + CF
            "adcq 56({src2}), %r8",                      // %r8 = src1[7] + src2[7] + CF
            "movq %rdx, 48({dst})",                      // Store dst[6]
            "movq %r8, 56({dst})",                       // Store dst[7]
            "leaq 64({dst}), {dst}",                     // Advance dst by 64 bytes (preserves CF)
            "leaq 64({src1}), {src1}",                   // Advance src1 by 64 bytes (preserves CF)
            "leaq 64({src2}), {src2}",                   // Advance src2 by 64 bytes (preserves CF)
            "decq {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail: descending 4/2/1 blocks
            "3:",
            "decq {any_tail}",                           // Test if any remainder limbs remain (preserves CF)
            "js 6f",                                     // If none, jump directly to exit (6f)
            "decq {tail_4}",                             // Test 4-limb tail block
            "js 4f",                                     // Skip to 2-limb test if absent (4f)
            "movq ({src1}), %rax",                       // Load src1[0]
            "movq 8({src1}), %rcx",                      // Load src1[1]
            "adcq ({src2}), %rax",                       // src1[0] + src2[0] + CF
            "adcq 8({src2}), %rcx",                      // src1[1] + src2[1] + CF
            "movq 16({src1}), %rdx",                     // Load src1[2]
            "movq 24({src1}), %r8",                      // Load src1[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "adcq 16({src2}), %rdx",                     // src1[2] + src2[2] + CF
            "adcq 24({src2}), %r8",                      // src1[3] + src2[3] + CF
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "leaq 32({dst}), {dst}",                     // Advance dst by 32 bytes (preserves CF)
            "leaq 32({src1}), {src1}",                   // Advance src1 by 32 bytes (preserves CF)
            "leaq 32({src2}), {src2}",                   // Advance src2 by 32 bytes (preserves CF)

            "4:",
            "decq {tail_2}",                             // Test 2-limb tail block
            "js 5f",                                     // Skip to 1-limb test if absent (5f)
            "movq ({src1}), %rax",                       // Load src1[0]
            "movq 8({src1}), %rcx",                      // Load src1[1]
            "adcq ({src2}), %rax",                       // src1[0] + src2[0] + CF
            "adcq 8({src2}), %rcx",                      // src1[1] + src2[1] + CF
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "leaq 16({dst}), {dst}",                     // Advance dst by 16 bytes (preserves CF)
            "leaq 16({src1}), {src1}",                   // Advance src1 by 16 bytes (preserves CF)
            "leaq 16({src2}), {src2}",                   // Advance src2 by 16 bytes (preserves CF)

            "5:",
            "decq {tail_1}",                             // Test 1-limb tail block
            "js 6f",                                     // Skip if absent (6f)
            "movq ({src1}), %rax",                       // Load single src1 limb
            "adcq ({src2}), %rax",                       // %rax = src1[0] + src2[0] + CF
            "movq %rax, ({dst})",                        // Store single dst limb

            // Exit: capture final carry
            "6:",
            "adcq {carry}, {carry}",                     // carry = 0 + 0 + CF (0 or 1)
            carry = out(reg) carry,
            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            any_tail = inout(reg) any_tail => _,
            tail_4 = inout(reg) tail_4 => _,
            tail_2 = inout(reg) tail_2 => _,
            tail_1 = inout(reg) tail_1 => _,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("r8") _,
            options(nostack, att_syntax)
        );
    }
    carry
}

/// Writes a sum with a straight carry chain for `2 <= len <= 4`.
///
/// # Safety
///
/// `len` must be in `2..=4`. Both aligned sources must contain `len`
/// initialized limbs. The aligned destination must cover `len` writable limbs
/// and be disjoint from both sources; the sources may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep fixed-size carry chains inside the selected kernel"
)]
#[inline(always)]
unsafe fn add_small_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: len == 2 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    "movq ({src1}), %rax",               // Load src1[0]
                    "movq 8({src1}), %rcx",              // Load src1[1]
                    "xorl {carry:e}, {carry:e}",         // Clear CF and carry register
                    "addq ({src2}), %rax",               // %rax = src1[0] + src2[0], set CF
                    "adcq 8({src2}), %rcx",              // %rcx = src1[1] + src2[1] + CF
                    "movq %rax, ({dst})",                // Store dst[0]
                    "adcq {carry}, {carry}",             // carry = 0 + CF (0 or 1)
                    "movq %rcx, 8({dst})",               // Store dst[1]
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    carry = out(reg) carry,
                    out("rax") _,
                    out("rcx") _,
                    options(nostack, att_syntax)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: len == 3 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",         // Clear CF and carry register
                    "movq ({src1}), %rax",               // Load src1[0]
                    "movq 8({src1}), %rcx",              // Load src1[1]
                    "movq 16({src1}), %rdx",             // Load src1[2]
                    "addq ({src2}), %rax",               // %rax = src1[0] + src2[0], set CF
                    "adcq 8({src2}), %rcx",              // %rcx = src1[1] + src2[1] + CF
                    "adcq 16({src2}), %rdx",             // %rdx = src1[2] + src2[2] + CF
                    "movq %rax, ({dst})",                // Store dst[0]
                    "movq %rcx, 8({dst})",               // Store dst[1]
                    "movq %rdx, 16({dst})",              // Store dst[2]
                    "adcq {carry}, {carry}",             // carry = 0 + CF
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    carry = out(reg) carry,
                    out("rax") _,
                    out("rcx") _,
                    out("rdx") _,
                    options(nostack, att_syntax)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: len == 4 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",         // Clear CF and carry register
                    "movq ({src1}), %rax",               // Load src1[0]
                    "movq 8({src1}), %rcx",              // Load src1[1]
                    "movq 16({src1}), %rdx",             // Load src1[2]
                    "movq 24({src1}), %r8",              // Load src1[3]
                    "addq ({src2}), %rax",               // %rax = src1[0] + src2[0], set CF
                    "adcq 8({src2}), %rcx",              // %rcx = src1[1] + src2[1] + CF
                    "adcq 16({src2}), %rdx",             // %rdx = src1[2] + src2[2] + CF
                    "adcq 24({src2}), %r8",              // %r8 = src1[3] + src2[3] + CF
                    "movq %rax, ({dst})",                // Store dst[0]
                    "movq %rcx, 8({dst})",               // Store dst[1]
                    "movq %rdx, 16({dst})",              // Store dst[2]
                    "movq %r8, 24({dst})",               // Store dst[3]
                    "adcq {carry}, {carry}",             // carry = 0 + CF
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    carry = out(reg) carry,
                    out("rax") _,
                    out("rcx") _,
                    out("rdx") _,
                    out("r8") _,
                    options(nostack, att_syntax)
                );
            }
            carry
        }
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
