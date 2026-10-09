//! `LoongArch32` addition kernels (inline assembly).
//!
//! Evaluates `dst += src` using 4-way unrolled loops with branchless `sltu` carry tracking.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Adds `src` to `dst` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must be aligned, initialized, and cover `len` limbs in
/// live allocations of at most `isize::MAX` bytes. The destination must be
/// writable. The two spans must be disjoint or exactly identical.
#[expect(
    clippy::inline_always,
    reason = "Keep the carry recurrence in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst` and `src` are valid for `len in 2..=4`.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len > 4 proves chunks > 0; 4 * chunks + rem == len bounds
    // every aligned, initialized access. Each input pair is read before its
    // result is written, permitting exact aliasing.
    unsafe {
        asm!(
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld.w {t0}, {src}, 0",                      // Load src[0]
            "ld.w {t1}, {dst}, 0",                      // Load dst[0]
            "add.w {t1}, {t1}, {t0}",                    // t1 = dst[0] + src[0]
            "sltu {c0}, {t1}, {t0}",                     // c0 = 1 if addition wrapped
            "add.w {t1}, {t1}, {carry}",                 // t1 += carry
            "sltu {c1}, {t1}, {carry}",                  // c1 = 1 if addition with carry wrapped
            "or {carry}, {c0}, {c1}",                    // Combined carry for next limb
            "st.w {t1}, {dst}, 0",                       // Store updated dst[0]

            // [Limb 1]
            "ld.w {t0}, {src}, 4",                       // Load src[1]
            "ld.w {t1}, {dst}, 4",                       // Load dst[1]
            "add.w {t1}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t1}, {t0}",                     // Detect wrap
            "add.w {t1}, {t1}, {carry}",                 // Add carry
            "sltu {c1}, {t1}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t1}, {dst}, 4",                       // Store dst[1]

            // [Limb 2]
            "ld.w {t0}, {src}, 8",                       // Load src[2]
            "ld.w {t1}, {dst}, 8",                       // Load dst[2]
            "add.w {t1}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t1}, {t0}",                     // Detect wrap
            "add.w {t1}, {t1}, {carry}",                 // Add carry
            "sltu {c1}, {t1}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t1}, {dst}, 8",                       // Store dst[2]

            // [Limb 3]
            "ld.w {t0}, {src}, 12",                      // Load src[3]
            "ld.w {t1}, {dst}, 12",                      // Load dst[3]
            "add.w {t1}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t1}, {t0}",                     // Detect wrap
            "add.w {t1}, {t1}, {carry}",                 // Add carry
            "sltu {c1}, {t1}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t1}, {dst}, 12",                      // Store dst[3]

            // Advance pointers by 16 bytes and loop
            "addi.w {src}, {src}, 16",                   // Advance src pointer
            "addi.w {dst}, {dst}, 16",                   // Advance dst pointer
            "addi.w {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld.w {t0}, {src}, 0",                      // Load single src limb
            "ld.w {t1}, {dst}, 0",                      // Load single dst limb
            "add.w {t1}, {t1}, {t0}",                    // Add limbs
            "sltu {c0}, {t1}, {t0}",                     // Detect wrap
            "add.w {t1}, {t1}, {carry}",                 // Add carry
            "sltu {c1}, {t1}, {carry}",                  // Detect wrap
            "or {carry}, {c0}, {c1}",                    // Combine carry
            "st.w {t1}, {dst}, 0",                       // Store dst limb
            "addi.w {src}, {src}, 4",                    // Advance src
            "addi.w {dst}, {dst}, 4",                    // Advance dst
            "addi.w {rem}, {rem}, -1",                   // Decrement rem
            "bnez {rem}, 3b",                            // Repeat while rem != 0

            // Exit
            "4:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
            options(nostack)
        );
    }
    carry
}

/// Straight-line `dst[i] = dst[i] + src[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// The pointer contract of [`add_limbs_unchecked`] applies and `2 <= len <= 4`.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: len == 2 bounds the aligned, initialized spans.
            // Each input pair is read before its store, permitting exact aliasing.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src}, 0",              // Load src[0]
                    "ld.w {t1}, {dst}, 0",              // Load dst[0]
                    "add.w {t1}, {t1}, {t0}",            // t1 = dst[0] + src[0]
                    "sltu {carry}, {t1}, {t0}",          // carry = 1 if wrap
                    "st.w {t1}, {dst}, 0",               // Store updated dst[0]
                    // Limb 1
                    "ld.w {t0}, {src}, 4",              // Load src[1]
                    "ld.w {t1}, {dst}, 4",              // Load dst[1]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.w {t1}, {dst}, 4",               // Store updated dst[1]
                    src = in(reg) src,
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
            // SAFETY: len == 3 bounds the aligned, initialized spans.
            // Each input pair is read before its store, permitting exact aliasing.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src}, 0",              // Load src[0]
                    "ld.w {t1}, {dst}, 0",              // Load dst[0]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 0
                    "sltu {carry}, {t1}, {t0}",          // Detect wrap
                    "st.w {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.w {t0}, {src}, 4",              // Load src[1]
                    "ld.w {t1}, {dst}, 4",              // Load dst[1]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.w {t1}, {dst}, 4",               // Store dst[1]
                    // Limb 2
                    "ld.w {t0}, {src}, 8",              // Load src[2]
                    "ld.w {t1}, {dst}, 8",              // Load dst[2]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 2
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.w {t1}, {dst}, 8",               // Store dst[2]
                    src = in(reg) src,
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
            // SAFETY: len == 4 bounds the aligned, initialized spans.
            // Each input pair is read before its store, permitting exact aliasing.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src}, 0",              // Load src[0]
                    "ld.w {t1}, {dst}, 0",              // Load dst[0]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 0
                    "sltu {carry}, {t1}, {t0}",          // Detect wrap
                    "st.w {t1}, {dst}, 0",               // Store dst[0]
                    // Limb 1
                    "ld.w {t0}, {src}, 4",              // Load src[1]
                    "ld.w {t1}, {dst}, 4",              // Load dst[1]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 1
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.w {t1}, {dst}, 4",               // Store dst[1]
                    // Limb 2
                    "ld.w {t0}, {src}, 8",              // Load src[2]
                    "ld.w {t1}, {dst}, 8",              // Load dst[2]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 2
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Combine carry
                    "st.w {t1}, {dst}, 8",               // Store dst[2]
                    // Limb 3
                    "ld.w {t0}, {src}, 12",             // Load src[3]
                    "ld.w {t1}, {dst}, 12",             // Load dst[3]
                    "add.w {t1}, {t1}, {t0}",            // Add limb 3
                    "sltu {c0}, {t1}, {t0}",             // Detect wrap
                    "add.w {t1}, {t1}, {carry}",         // Add carry
                    "sltu {c1}, {t1}, {carry}",          // Detect wrap
                    "or {carry}, {c0}, {c1}",            // Final carry
                    "st.w {t1}, {dst}, 12",              // Store dst[3]
                    src = in(reg) src,
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
