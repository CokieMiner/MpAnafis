//! `LoongArch` 32-bit addition kernels (inline assembly).
//!
//! Evaluates `dst = src1 + src2` using 4-way unrolled loops with branchless `sltu` carry tracking.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs, returning
/// the final carry.
///
/// # Microarchitectural Strategy
///
/// `LoongArch32` uses `add.w` and `sltu` (set-less-than unsigned) to detect arithmetic wrap-around.
/// The 4-way unrolled loop loads and adds 4 limbs per iteration, chaining carries branchlessly with `or`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    // SAFETY: The caller guarantees both pointers cover `len` elements.
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees all pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst`, `src1`, `src2` valid for `len in 2..=4`.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst`, `src1`, `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld.w {t0}, {src1}, 0",                      // Load src1[0]
            "ld.w {t1}, {src2}, 0",                      // Load src2[0]
            "add.w {t2}, {t1}, {t0}",                    // t2 = src2[0] + src1[0]
            "sltu {c0}, {t2}, {t0}",                     // c0 = 1 if addition wrapped
            "add.w {t2}, {t2}, {carry}",                 // t2 += carry
            "sltu {c1}, {t2}, {carry}",                  // c1 = 1 if addition with carry wrapped
            "or {carry}, {c0}, {c1}",                    // Combined carry for next limb
            "st.w {t2}, {dst}, 0",                       // Store dst[0]

            // [Limb 1]
            "ld.w {t0}, {src1}, 4",                      // Load src1[1]
            "ld.w {t1}, {src2}, 4",                      // Load src2[1]
            "add.w {t2}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t2}, {t0}",                     // Detect wrap
            "add.w {t2}, {t2}, {carry}",                 // Add carry
            "sltu {c1}, {t2}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t2}, {dst}, 4",                       // Store dst[1]

            // [Limb 2]
            "ld.w {t0}, {src1}, 8",                      // Load src1[2]
            "ld.w {t1}, {src2}, 8",                      // Load src2[2]
            "add.w {t2}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t2}, {t0}",                     // Detect wrap
            "add.w {t2}, {t2}, {carry}",                 // Add carry
            "sltu {c1}, {t2}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t2}, {dst}, 8",                       // Store dst[2]

            // [Limb 3]
            "ld.w {t0}, {src1}, 12",                     // Load src1[3]
            "ld.w {t1}, {src2}, 12",                     // Load src2[3]
            "add.w {t2}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t2}, {t0}",                     // Detect wrap
            "add.w {t2}, {t2}, {carry}",                 // Add carry
            "sltu {c1}, {t2}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t2}, {dst}, 12",                      // Store dst[3]

            // Advance pointers by 16 bytes and loop
            "addi.w {src1}, {src1}, 16",                 // Advance src1
            "addi.w {src2}, {src2}, 16",                 // Advance src2
            "addi.w {dst}, {dst}, 16",                   // Advance dst
            "addi.w {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld.w {t0}, {src1}, 0",                      // Load single src1 limb
            "ld.w {t1}, {src2}, 0",                      // Load single src2 limb
            "add.w {t2}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t2}, {t0}",                     // Detect wrap
            "add.w {t2}, {t2}, {carry}",                 // Add carry
            "sltu {c1}, {t2}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t2}, {dst}, 0",                       // Store dst limb
            "addi.w {src1}, {src1}, 4",                  // Advance src1
            "addi.w {src2}, {src2}, 4",                  // Advance src2
            "addi.w {dst}, {dst}, 4",                    // Advance dst
            "addi.w {rem}, {rem}, -1",                   // Decrement rem
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

/// Straight-line `dst[i] = src1[i] + src2[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 2 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src1}, 0",              // Load src1[0]
                    "ld.w {t1}, {src2}, 0",              // Load src2[0]
                    "add.w {t1}, {t1}, {t0}",            // t1 = src2[0] + src1[0]
                    "sltu {carry}, {t1}, {t0}",          // carry = 1 if wrap
                    "st.w {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.w {t0}, {src1}, 4",              // Load src1[1]
                    "ld.w {t1}, {src2}, 4",              // Load src2[1]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.w {t1}, {dst}, 4",               // Store dst[1]
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 3 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src1}, 0",              // Load src1[0]
                    "ld.w {t1}, {src2}, 0",              // Load src2[0]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 0
                    "sltu {carry}, {t1}, {t0}",          // Detect wrap
                    "st.w {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.w {t0}, {src1}, 4",              // Load src1[1]
                    "ld.w {t1}, {src2}, 4",              // Load src2[1]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.w {t1}, {dst}, 4",               // Store dst[1]
                    // Limb 2
                    "ld.w {t0}, {src1}, 8",              // Load src1[2]
                    "ld.w {t1}, {src2}, 8",              // Load src2[2]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 2
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.w {t1}, {dst}, 8",               // Store dst[2]
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 4 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src1}, 0",              // Load src1[0]
                    "ld.w {t1}, {src2}, 0",              // Load src2[0]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 0
                    "sltu {carry}, {t1}, {t0}",          // Detect wrap
                    "st.w {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.w {t0}, {src1}, 4",              // Load src1[1]
                    "ld.w {t1}, {src2}, 4",              // Load src2[1]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.w {t1}, {dst}, 4",               // Store dst[1]
                    // Limb 2
                    "ld.w {t0}, {src1}, 8",              // Load src1[2]
                    "ld.w {t1}, {src2}, 8",              // Load src2[2]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 2
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.w {t1}, {dst}, 8",               // Store dst[2]
                    // Limb 3
                    "ld.w {t0}, {src1}, 12",             // Load src1[3]
                    "ld.w {t1}, {src2}, 12",             // Load src2[3]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 3
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.w {t1}, {dst}, 12",              // Store dst[3]
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
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
