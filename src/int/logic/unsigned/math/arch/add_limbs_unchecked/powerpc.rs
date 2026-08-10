//! PowerPC 32-bit addition kernels (inline assembly).
//!
//! Uses the XER[CA] carry bit via `adde` (add extended) instructions.
//! The loop is **4‑way unrolled** (`len >> 2`) with the CTR register
//! for counting (`bdnz` branch‑decrement‑non‑zero).  Any remaining
//! limbs (len & 3) are handled by a tail loop.

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
            "mtctr {chunks}",                    // CTR = chunks (loop counter)
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lwz {src_v0}, 0({src})",           // load src[0]
            "lwz {src_v1}, 4({src})",           // load src[1]
            "lwz {src_v2}, 8({src})",           // load src[2]
            "lwz {src_v3}, 12({src})",          // load src[3]
            "lwz {dst_v0}, 0({dst})",           // load dst[0]
            "lwz {dst_v1}, 4({dst})",           // load dst[1]
            "lwz {dst_v2}, 8({dst})",           // load dst[2]
            "lwz {dst_v3}, 12({dst})",          // load dst[3]
            "adde {dst_v0}, {dst_v0}, {src_v0}", // dst[0] += src[0] + XER[CA]
            "adde {dst_v1}, {dst_v1}, {src_v1}", // dst[1] += src[1] + XER[CA]
            "adde {dst_v2}, {dst_v2}, {src_v2}", // dst[2] += src[2] + XER[CA]
            "adde {dst_v3}, {dst_v3}, {src_v3}", // dst[3] += src[3] + XER[CA]
            "stw {dst_v0}, 0({dst})",           // store dst[0]
            "stw {dst_v1}, 4({dst})",           // store dst[1]
            "stw {dst_v2}, 8({dst})",           // store dst[2]
            "stw {dst_v3}, 12({dst})",          // store dst[3]
            "addi {src}, {src}, 16",            // advance src by 16 bytes
            "addi {dst}, {dst}, 16",            // advance dst by 16 bytes
            "bdnz 2b",                          // CTR--, branch if CTR != 0
            // ── Tail: single‑limb remainder loop ───────────────────────
            "1:",
            "cmpwi {rem}, 0",                   // compare rem with 0
            "beq 3f",                           // skip tail if rem == 0
            "mtctr {rem}",                      // CTR = rem
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "lwz {src_v0}, 0({src})",           // load src limb
            "lwz {dst_v0}, 0({dst})",           // load dst limb
            "adde {dst_v0}, {dst_v0}, {src_v0}", // dst += src + XER[CA]
            "stw {dst_v0}, 0({dst})",           // store result
            "addi {src}, {src}, 4",             // advance src by 4 bytes
            "addi {dst}, {dst}, 4",             // advance dst by 4 bytes
            "bdnz 4b",                          // CTR--, branch if CTR != 0
            "3:",
            "li {carry}, 0",                    // carry = 0
            "addze {carry}, {carry}",           // carry = 0 + 0 + XER[CA]
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
                    "lwz {s0}, 0({src})",
                    "lwz {s1}, 4({src})",
                    "lwz {d0}, 0({dst})",
                    "lwz {d1}, 4({dst})",
                    "addc {d0}, {d0}, {s0}",
                    "adde {d1}, {d1}, {s1}",
                    "stw {d0}, 0({dst})",
                    "stw {d1}, 4({dst})",
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
                    "lwz {s0}, 0({src})",
                    "lwz {s1}, 4({src})",
                    "lwz {s2}, 8({src})",
                    "lwz {d0}, 0({dst})",
                    "lwz {d1}, 4({dst})",
                    "lwz {d2}, 8({dst})",
                    "addc {d0}, {d0}, {s0}",
                    "adde {d1}, {d1}, {s1}",
                    "adde {d2}, {d2}, {s2}",
                    "stw {d0}, 0({dst})",
                    "stw {d1}, 4({dst})",
                    "stw {d2}, 8({dst})",
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
                    "lwz {s0}, 0({src})",
                    "lwz {s1}, 4({src})",
                    "lwz {s2}, 8({src})",
                    "lwz {s3}, 12({src})",
                    "lwz {d0}, 0({dst})",
                    "lwz {d1}, 4({dst})",
                    "lwz {d2}, 8({dst})",
                    "lwz {d3}, 12({dst})",
                    "addc {d0}, {d0}, {s0}",
                    "adde {d1}, {d1}, {s1}",
                    "adde {d2}, {d2}, {s2}",
                    "adde {d3}, {d3}, {s3}",
                    "stw {d0}, 0({dst})",
                    "stw {d1}, 4({dst})",
                    "stw {d2}, 8({dst})",
                    "stw {d3}, 12({dst})",
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
