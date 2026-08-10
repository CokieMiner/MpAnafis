//! PowerPC 64-bit addition kernels (inline assembly).
//!
//! Uses the XER[CA] carry bit via `adde` (add extended) instructions.
//! The loop is **4‑way unrolled** (`len >> 2`) with the CTR register
//! for counting (`bdnz` branch‑decrement‑non‑zero).  Any remaining
//! limbs (len & 3) are handled by a tail loop.
//!
//! ## Carry tracking
//!
//! `addic {carry}, {carry}, 0` clears XER[CA] (add immediate carrying).
//! Each `adde` instruction adds its two operands plus XER[CA] and
//! writes the new XER[CA] (carry out).  The final carry is extracted
//! with `li {carry}, 0` followed by `addze {carry}, {carry}` which
//! sets `{carry} = 0 + 0 + XER[CA]`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Add `len` limbs from `src` into `dst` and return the final carry.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
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
    let mut carry: Limb;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "addic {carry}, {rem}, 0",              // XER[CA] = 0 (add immediate carrying)
            // ── 4‑way unrolled loop (using CTR) ────────────────────────
            "mtctr {chunks}",                     // CTR = chunks (loop counter)
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ld {src_v0}, 0({src})",            // load src[0]
            "ld {src_v1}, 8({src})",            // load src[1]
            "ld {src_v2}, 16({src})",           // load src[2]
            "ld {src_v3}, 24({src})",           // load src[3]
            "ld {dst_v0}, 0({dst})",            // load dst[0]
            "ld {dst_v1}, 8({dst})",            // load dst[1]
            "ld {dst_v2}, 16({dst})",           // load dst[2]
            "ld {dst_v3}, 24({dst})",           // load dst[3]
            "adde {dst_v0}, {dst_v0}, {src_v0}", // dst[0] += src[0] + XER[CA]
            "adde {dst_v1}, {dst_v1}, {src_v1}", // dst[1] += src[1] + XER[CA]
            "adde {dst_v2}, {dst_v2}, {src_v2}", // dst[2] += src[2] + XER[CA]
            "adde {dst_v3}, {dst_v3}, {src_v3}", // dst[3] += src[3] + XER[CA]
            "std {dst_v0}, 0({dst})",           // store dst[0]
            "std {dst_v1}, 8({dst})",           // store dst[1]
            "std {dst_v2}, 16({dst})",          // store dst[2]
            "std {dst_v3}, 24({dst})",          // store dst[3]
            "addi {src}, {src}, 32",            // advance src by 32 bytes
            "addi {dst}, {dst}, 32",            // advance dst by 32 bytes
            "bdnz 2b",                           // CTR--, branch if CTR != 0
            // ── Tail: single‑limb remainder loop ───────────────────────
            "1:",
            "cmpldi {rem}, 0",                   // compare rem with 0
            "beq 3f",                             // skip tail if rem == 0
            "mtctr {rem}",                        // CTR = rem
            "addi {src}, {src}, -8",
            "addi {dst}, {dst}, -8",
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldu {src_v0}, 8({src})",           // load src limb, advance src
            "ldu {dst_v0}, 8({dst})",           // load dst limb, advance dst
            "adde {dst_v0}, {dst_v0}, {src_v0}", // dst += src + XER[CA]
            "std {dst_v0}, 0({dst})",           // store result
            "bdnz 4b",                            // CTR--, branch if CTR != 0
            "3:",
            "li {carry}, 0",                      // carry = 0
            "addze {carry}, {carry}",             // carry = 0 + 0 + XER[CA]
            carry = out(reg) carry,
            dst = inout(reg_nonzero) dst => _,
            src = inout(reg_nonzero) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
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
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 2 limbs.
            unsafe {
                asm!(
                    "ld {s0}, 0({src})",
                    "ld {s1}, 8({src})",
                    "ld {d0}, 0({dst})",
                    "ld {d1}, 8({dst})",
                    "addc {d0}, {d0}, {s0}",
                    "adde {d1}, {d1}, {s1}",
                    "std {d0}, 0({dst})",
                    "std {d1}, 8({dst})",
                    "addze {carry}, {zero}",
                    src = inout(reg_nonzero) src => _,
                    dst = inout(reg_nonzero) dst => _,
                    zero = inout(reg) 0_usize => _,
                    s0 = out(reg) _, s1 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _,
                    carry = out(reg) carry,
                    out("xer") _,
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
                    "ld {s0}, 0({src})",
                    "ld {s1}, 8({src})",
                    "ld {s2}, 16({src})",
                    "ld {d0}, 0({dst})",
                    "ld {d1}, 8({dst})",
                    "ld {d2}, 16({dst})",
                    "addc {d0}, {d0}, {s0}",
                    "adde {d1}, {d1}, {s1}",
                    "adde {d2}, {d2}, {s2}",
                    "std {d0}, 0({dst})",
                    "std {d1}, 8({dst})",
                    "std {d2}, 16({dst})",
                    "addze {carry}, {zero}",
                    src = inout(reg_nonzero) src => _,
                    dst = inout(reg_nonzero) dst => _,
                    zero = inout(reg) 0_usize => _,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _,
                    carry = out(reg) carry,
                    out("xer") _,
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
                    "ld {s0}, 0({src})",
                    "ld {s1}, 8({src})",
                    "ld {s2}, 16({src})",
                    "ld {s3}, 24({src})",
                    "ld {d0}, 0({dst})",
                    "ld {d1}, 8({dst})",
                    "ld {d2}, 16({dst})",
                    "ld {d3}, 24({dst})",
                    "addc {d0}, {d0}, {s0}",
                    "adde {d1}, {d1}, {s1}",
                    "adde {d2}, {d2}, {s2}",
                    "adde {d3}, {d3}, {s3}",
                    "std {d0}, 0({dst})",
                    "std {d1}, 8({dst})",
                    "std {d2}, 16({dst})",
                    "std {d3}, 24({dst})",
                    "addze {carry}, {zero}",
                    src = inout(reg_nonzero) src => _,
                    dst = inout(reg_nonzero) dst => _,
                    zero = inout(reg) 0_usize => _,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _, d3 = out(reg) _,
                    carry = out(reg) carry,
                    out("xer") _,
                    options(nostack)
                );
            }
            carry
        }
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
