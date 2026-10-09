//! `LoongArch64` addition into a disjoint destination.
//!
//! `add.d` computes each limb sum modulo `2^64`; `sltu` detects overflow from
//! adding the sources and then the incoming binary carry. Their OR is the
//! outgoing carry. Four-limb blocks precede a scalar tail.

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
    reason = "Keep the carry recurrence in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized source spans.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 bounds the aligned, writable destination span.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: prior returns and len <= 4 prove 2 <= len <= 4. The caller
        // supplies aligned sources and a disjoint writable destination.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len > 4 gives chunks > 0; 4 * chunks + rem == len bounds every
    // aligned read and write. Both sources are initialized and may overlap;
    // dst is disjoint and only written. All modified GPRs are declared, and
    // pointer advances stop at the span ends.
    unsafe {
        asm!(
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld.d {t0}, {src1}, 0",                      // Load src1[0]
            "ld.d {t1}, {src2}, 0",                      // Load src2[0]
            "add.d {t2}, {t0}, {t1}",                    // t2 = src1[0] + src2[0]
            "sltu {c0}, {t2}, {t0}",                     // c0 = 1 if addition wrapped
            "add.d {t2}, {t2}, {carry}",                 // t2 += carry
            "sltu {c1}, {t2}, {carry}",                  // c1 = 1 if addition with carry wrapped
            "or {carry}, {c0}, {c1}",                    // Combined carry for next limb
            "st.d {t2}, {dst}, 0",                       // Store dst[0]

            // [Limb 1]
            "ld.d {t0}, {src1}, 8",                      // Load src1[1]
            "ld.d {t1}, {src2}, 8",                      // Load src2[1]
            "add.d {t2}, {t0}, {t1}",                    // t2 = src1[1] + src2[1]
            "sltu {c0}, {t2}, {t0}",                     // c0 = wrap detection
            "add.d {t2}, {t2}, {carry}",                 // t2 += carry
            "sltu {c1}, {t2}, {carry}",                  // c1 = wrap detection
            "or {carry}, {c0}, {c1}",                    // Combined carry
            "st.d {t2}, {dst}, 8",                       // Store dst[1]

            // [Limb 2]
            "ld.d {t0}, {src1}, 16",                     // Load src1[2]
            "ld.d {t1}, {src2}, 16",                     // Load src2[2]
            "add.d {t2}, {t0}, {t1}",                    // t2 = src1[2] + src2[2]
            "sltu {c0}, {t2}, {t0}",                     // c0 = wrap detection
            "add.d {t2}, {t2}, {carry}",                 // t2 += carry
            "sltu {c1}, {t2}, {carry}",                  // c1 = wrap detection
            "or {carry}, {c0}, {c1}",                    // Combined carry
            "st.d {t2}, {dst}, 16",                      // Store dst[2]

            // [Limb 3]
            "ld.d {t0}, {src1}, 24",                     // Load src1[3]
            "ld.d {t1}, {src2}, 24",                     // Load src2[3]
            "add.d {t2}, {t0}, {t1}",                    // t2 = src1[3] + src2[3]
            "sltu {c0}, {t2}, {t0}",                     // c0 = wrap detection
            "add.d {t2}, {t2}, {carry}",                 // t2 += carry
            "sltu {c1}, {t2}, {carry}",                  // c1 = wrap detection
            "or {carry}, {c0}, {c1}",                    // Combined carry
            "st.d {t2}, {dst}, 24",                      // Store dst[3]

            // Advance pointers by 32 bytes and loop
            "addi.d {src1}, {src1}, 32",                 // Advance src1
            "addi.d {src2}, {src2}, 32",                 // Advance src2
            "addi.d {dst}, {dst}, 32",                   // Advance dst
            "addi.d {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld.d {t0}, {src1}, 0",                      // Load single src1 limb
            "ld.d {t1}, {src2}, 0",                      // Load single src2 limb
            "add.d {t2}, {t0}, {t1}",                    // Add limbs
            "sltu {c0}, {t2}, {t0}",                     // Detect wrap
            "add.d {t2}, {t2}, {carry}",                 // Add carry
            "sltu {c1}, {t2}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combined carry
            "st.d {t2}, {dst}, 0",                       // Store dst limb
            "addi.d {src1}, {src1}, 8",                  // Advance src1
            "addi.d {src2}, {src2}, 8",                  // Advance src2
            "addi.d {dst}, {dst}, 8",                    // Advance dst
            "addi.d {rem}, {rem}, -1",                   // Decrement rem
            "bnez {rem}, 3b",                            // Repeat while rem != 0

            // Exit
            "4:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
            options(nostack)
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
    clippy::too_many_lines,
    reason = "Keep the three fixed-size carry chains in one inlined kernel"
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
                    // Limb 0 (carry-in = 0)
                    "ld.d {t0}, {src1}, 0",              // Load src1[0]
                    "ld.d {t1}, {src2}, 0",              // Load src2[0]
                    "add.d {t1}, {t1}, {t0}",            // t1 = src1[0] + src2[0]
                    "sltu {carry}, {t1}, {t0}",          // carry = 1 if wrap
                    "st.d {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.d {t0}, {src1}, 8",              // Load src1[1]
                    "ld.d {t1}, {src2}, 8",              // Load src2[1]
                    "add.d {t1}, {t1}, {t0}",            // t1 = src1[1] + src2[1]
                    "sltu {c0}, {t1}, {t0}",             // c0 = 1 if wrap
                    "add.d {t1}, {t1}, {carry}",         // t1 += carry
                    "sltu {c1}, {t1}, {carry}",          // c1 = 1 if wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.d {t1}, {dst}, 8",               // Store dst[1]
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
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
                    // Limb 0 (carry-in = 0)
                    "ld.d {t0}, {src1}, 0",              // Load src1[0]
                    "ld.d {t1}, {src2}, 0",              // Load src2[0]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 0
                    "sltu {carry}, {t1}, {t0}",          // Detect wrap
                    "st.d {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.d {t0}, {src1}, 8",              // Load src1[1]
                    "ld.d {t1}, {src2}, 8",              // Load src2[1]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.d {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.d {t1}, {dst}, 8",               // Store dst[1]
                    // Limb 2
                    "ld.d {t0}, {src1}, 16",             // Load src1[2]
                    "ld.d {t1}, {src2}, 16",             // Load src2[2]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 2
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.d {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.d {t1}, {dst}, 16",              // Store dst[2]
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
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
                    // Limb 0 (carry-in = 0)
                    "ld.d {t0}, {src1}, 0",              // Load src1[0]
                    "ld.d {t1}, {src2}, 0",              // Load src2[0]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 0
                    "sltu {carry}, {t1}, {t0}",          // Detect wrap
                    "st.d {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.d {t0}, {src1}, 8",              // Load src1[1]
                    "ld.d {t1}, {src2}, 8",              // Load src2[1]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.d {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.d {t1}, {dst}, 8",               // Store dst[1]
                    // Limb 2
                    "ld.d {t0}, {src1}, 16",             // Load src1[2]
                    "ld.d {t1}, {src2}, 16",             // Load src2[2]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 2
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.d {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.d {t1}, {dst}, 16",              // Store dst[2]
                    // Limb 3
                    "ld.d {t0}, {src1}, 24",             // Load src1[3]
                    "ld.d {t1}, {src2}, 24",             // Load src2[3]
                    "add.d {t1}, {t1}, {t0}",            // Add limb 3
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.d {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.d {t1}, {dst}, 24",              // Store dst[3]
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
