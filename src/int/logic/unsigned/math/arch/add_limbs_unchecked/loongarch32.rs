//! `LoongArch32` addition kernels (inline assembly).
//!
//! Evaluates `dst += src` using 4-way unrolled loops with branchless `sltu` carry tracking.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Add `len` limbs from `src` into `dst` and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// `LoongArch32` uses `add.w` and `sltu` (set-less-than unsigned) to detect arithmetic wrap-around.
/// The 4-way unrolled loop loads and adds 4 limbs per iteration, chaining carries branchlessly with `or`.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    // SAFETY: The caller guarantees both pointers cover `len` elements.
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

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
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
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 2 limbs.
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
            // SAFETY: Caller guarantees `dst` and `src` are valid for 3 limbs.
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
            // SAFETY: Caller guarantees `dst` and `src` are valid for 4 limbs.
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
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
