//! x86‑64 (AMD64) `add_n` kernel.
//!
//! Uses the CF (carry flag) via `adcq` (add with carry, 64‑bit). Lengths above
//! four run an **8-way unrolled** loop from index zero, followed by a 4/2/1
//! descending tail for the `len & 7` limbs it cannot cover.
//!
//! The ordering is load-bearing in both directions.
//!
//! The unrolled loop must come *first*, not after an optional 4-limb prefix.
//! A prefix advances the index by four before the loop starts, which offsets
//! every 64-byte loop block by 32 bytes so each straddles two cache lines
//! instead of one. The signature was that 28 limbs ran slower in absolute
//! terms than 32, and 44 slower than 48: adding work made the routine faster,
//! which a straddle produces and a throughput limit never does. Running the
//! loop first is worth 9 to 16 points of ratio against GMP at the lengths that
//! had a prefix -- 28 to 31 limbs moved from 0.74-0.77x to 0.85-0.92x.
//!
//! The tail is 4/2/1 blocks rather than a single-limb loop, which costs a
//! branch per remainder limb on lengths where the whole call is under ten
//! nanoseconds. One `any_tail` guard covers all three, so a length that is a
//! multiple of eight -- 24, 32, 40, 48 -- skips the dispatch entirely instead
//! of running it three times to discover it has nothing to do.
//!
//! There is deliberately **no `.p2align` before the loop body**. Aligning a
//! loop header helps a long run amortise instruction fetch, but the padding
//! sits on the fall-through path and is *executed* on every call, which at
//! three iterations costs more than the alignment returns.
//!
//! Both were measured against the aligned, three-dispatch form in one process
//! and are worth a few percent from 23 limbs up. A four-limb block width was
//! measured at the same time and is *slower*, which is what identifies the cost
//! as fixed per-call overhead rather than per-iteration: doubling the iteration
//! count did not recover it. Do not re-try a narrower block on this evidence.
//!
//! Against GMP the remaining shape is a shallow dip around 24-32 limbs,
//! roughly 0.89x to 0.95x, with parity from 40 limbs and a lead from 44.
//!
//! Every block is entered by `decq` plus `js`, never `testq` or `cmpq`:
//! `dec` leaves CF untouched by design, so the carry established by limb i
//! survives the dispatch and is consumed by limb i+1. `leaq` and the
//! conditional jumps preserve it too. Substituting a flag-setting compare
//! anywhere in this routine silently drops a carry.

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
        // SAFETY: the caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: the caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: the caller guarantees both pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 3;
    let any_tail = len & 7;
    let tail_4 = (len >> 2) & 1;
    let tail_2 = (len >> 1) & 1;
    let tail_1 = len & 1;
    let idx = 0_usize;
    // Binary decomposition gives `len = 8*chunks + 4*tail_4 + 2*tail_2 + tail_1`
    // with each tail selector at most one. The blocks run in increasing index
    // order, so the carry chain is contiguous across every boundary.
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}",        // carry = 0, also clears CF
            // ── 8-way unrolled loop (do-while with decq/jns) ───────────
            // Runs first and from index zero so its 64-byte blocks keep the
            // destination's own alignment instead of inheriting a prefix's
            // 32-byte offset.
            "decq {chunks}",                      // decrement: jump if -1 (chunks == 0)
            "js 3f",                              // skip main loop if chunks == 0
            "2:",
            "movq ({src}, {idx}, 8), %rax",      // load first 4 limbs of src
            "movq 8({src}, {idx}, 8), %rcx",
            "movq 16({src}, {idx}, 8), %rdx",
            "movq 24({src}, {idx}, 8), %r8",
            "adcq %rax, ({dst}, {idx}, 8)",      // add first 4 limbs to dst + CF
            "adcq %rcx, 8({dst}, {idx}, 8)",
            "adcq %rdx, 16({dst}, {idx}, 8)",
            "adcq %r8, 24({dst}, {idx}, 8)",

            "movq 32({src}, {idx}, 8), %rax",    // load next 4 limbs of src
            "movq 40({src}, {idx}, 8), %rcx",
            "movq 48({src}, {idx}, 8), %rdx",
            "movq 56({src}, {idx}, 8), %r8",
            "adcq %rax, 32({dst}, {idx}, 8)",    // add next 4 limbs to dst + CF
            "adcq %rcx, 40({dst}, {idx}, 8)",
            "adcq %rdx, 48({dst}, {idx}, 8)",
            "adcq %r8, 56({dst}, {idx}, 8)",

            "leaq 8({idx}), {idx}",              // advance idx by 8
            "decq {chunks}",                      // decrement chunk counter
            "jns 2b",                             // loop back if chunks >= 0
            // ── Tail: descending 4/2/1 blocks, at most one of each ─────
            "3:",
            "decq {any_tail}",                    // no remainder at all: skip
            "js 6f",                              // all three dispatches below
            "decq {tail_4}",                      // decrement: jump if -1 (absent)
            "js 4f",
            "movq ({src}, {idx}, 8), %rax",
            "movq 8({src}, {idx}, 8), %rcx",
            "movq 16({src}, {idx}, 8), %rdx",
            "movq 24({src}, {idx}, 8), %r8",
            "adcq %rax, ({dst}, {idx}, 8)",
            "adcq %rcx, 8({dst}, {idx}, 8)",
            "adcq %rdx, 16({dst}, {idx}, 8)",
            "adcq %r8, 24({dst}, {idx}, 8)",
            "leaq 4({idx}), {idx}",
            "4:",
            "decq {tail_2}",
            "js 5f",
            "movq ({src}, {idx}, 8), %rax",
            "movq 8({src}, {idx}, 8), %rcx",
            "adcq %rax, ({dst}, {idx}, 8)",
            "adcq %rcx, 8({dst}, {idx}, 8)",
            "leaq 2({idx}), {idx}",
            "5:",
            "decq {tail_1}",
            "js 6f",
            "movq ({src}, {idx}, 8), %rax",
            "adcq %rax, ({dst}, {idx}, 8)",
            "6:",
            "adcq {carry}, {carry}",             // carry = 0 + 0 + CF (extract final carry)
            carry = out(reg) carry,
            idx = inout(reg) idx => _,
            dst = in(reg) dst,
            src = in(reg) src,
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 2 limbs.
            unsafe {
                asm!(
                        "xorl {carry:e}, {carry:e}",
                        "movq ({src}), %rax",
                        "movq 8({src}), %rcx",
                        "addq %rax, ({dst})",
                        "adcq %rcx, 8({dst})",
                        "adcq {carry}, {carry}",
                        src = in(reg) src,
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 3 limbs.
            unsafe {
                asm!(
                        "xorl {carry:e}, {carry:e}",
                        "movq ({src}), %rax",
                        "movq 8({src}), %rcx",
                        "movq 16({src}), %rdx",
                        "addq %rax, ({dst})",
                        "adcq %rcx, 8({dst})",
                        "adcq %rdx, 16({dst})",
                        "adcq {carry}, {carry}",
                        src = in(reg) src,
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 4 limbs.
            unsafe {
                asm!(
                        "xorl {carry:e}, {carry:e}",
                        "movq ({src}), %rax",
                        "movq 8({src}), %rcx",
                        "movq 16({src}), %rdx",
                        "movq 24({src}), %r8",
                        "addq %rax, ({dst})",
                        "adcq %rcx, 8({dst})",
                        "adcq %rdx, 16({dst})",
                        "adcq %r8, 24({dst})",
                        "adcq {carry}, {carry}",
                        src = in(reg) src,
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
        // SAFETY: The caller guarantees `2 <= len <= 4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
