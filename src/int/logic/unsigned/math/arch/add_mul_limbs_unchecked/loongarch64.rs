//! `LoongArch64` architecture-specific `addmul_1` / `submul_1` kernels.
//!
//! Uses `mul.d`/`mulh.du` for efficient 64×64→128 multiplication and
//! `add.d`/`sub.d` with `sltu` for carry/borrow tracking.
//!
//! The instruction set is nearly identical to RISC-V 64 for this purpose.
//!
//! ## Why inline assembly?
//!
//! The `LoongArch` ISA provides `mul.d` (multiply low) and `mulh.du` (multiply
//! high unsigned) instructions that together produce a 128-bit result from
//! two 64-bit operands.  Without asm the compiler would emit `__multi3`
//! calls for 128-bit multiplication, which are far slower than a single-pair
//! multiply-add chain.
//!
//! ## Register naming in asm!
//!
//! LLVM's `LoongArch64` assembler uses the following conventions inside
//! Rust's `asm!` macro:
//!   - `{name}` refers to a Rust-managed operand (register allocated by
//!     the compiler via the constraint list).
//!   - Integer registers are named `$r0`–`$r31` in standalone assembly,
//!     but `asm!` uses named operands to avoid manual allocation.
//!   - Immediates are written as signed integers (e.g. `-1`, `0`, `8`).
//!
//! ## Carry tracking strategy
//!
//! Neither `LoongArch` nor RISC-V have a carry flag.  We track carries
//! explicitly with `sltu` (set-less-than unsigned):
//!
//! ```text
//!   add.d result, a, b        // result = a + b (may wrap)
//!   sltu   carry, result, a   // carry = 1 if result < a (i.e. wrap occurred)
//! ```
//!
//! For the addmul loop each limb requires:
//!   1. `mul.d`  / `mulh.du`  → 128-bit product
//!   2. `add.d`  / `sltu`     → add carry to low part, detect overflow
//!   3. `add.d`               → add overflow to high part for next carry
//!   4. `add.d`  / `sltu`     → add product low to destination limb
//!   5. `add.d`               → accumulate second overflow into carry
//!
//! For submul the subtraction replaces the final addition with a
//! double-subtraction chain (subtract product+carry, then subtract borrow).
//!
//! ## Loop structure
//!
//! The loop is **2-way unrolled** (`len >> 1`): two source limbs and two
//! destination limbs are loaded together and processed back-to-back before
//! advancing pointers.  This reduces the overhead of load/store address
//! generation and gives the CPU more independent instructions to schedule.
//! Any odd trailing limb is handled by a single-iteration tail block.

use core::arch::asm;

use super::Limb;

// ── addmul_1 ───────────────────────────────────────────────────────────────

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry out.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// where the multiplication produces a `2×len`-limb intermediate and the
/// addition writes into the low `len` limbs of `dst`, returning the
/// overflow in `carry`.
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
/// The caller is responsible for ensuring `dst` has enough capacity for
/// the carry limb (i.e. `dst.len() > len` or the caller handles the
/// returned carry separately).
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry_in: Limb = 0;
    let chunks = len >> 1; // number of 2-limb blocks
    let rem = len & 1; // remainder (0 or 1)
    // SAFETY: Caller guarantees dst and src hold at least `len` valid Limb values.
    unsafe {
        asm!(
            // ── Check for empty input ──────────────────────────────────
            "beqz {chunks}, 2f",     // if chunks == 0, skip loop

            // ── 2-way unrolled loop ────────────────────────────────────
            "1:",
            // Load two source limbs and two destination limbs
            "ld.d {s0}, {src}, 0",   // s0 = src[0], 8-byte load with 0-byte offset
            "ld.d {s1}, {src}, 8",   // s1 = src[1], 8-byte load with 8-byte offset
            "ld.d {d0}, {dst}, 0",   // d0 = dst[0]
            "ld.d {d1}, {dst}, 8",   // d1 = dst[1]

            // ── Limb 0: product = s0 × scalar ──────────────────────────
            // p_lo0 = low 64 bits of product
            "mul.d {p_lo0}, {s0}, {scalar}",
            // p_hi0 = high 64 bits of product (unsigned)
            "mulh.du {p_hi0}, {s0}, {scalar}",

            // Add previous carry to the low part of the product.
            // Conceptually: (hi:lo) = (src[i] × scalar) + carry_in
            // We add carry to LO:  sum = lo + carry_in
            // Overflow from that addition (c1) is added to HI.
            "add.d {t_lo}, {p_lo0}, {carry_in}",
            // ca = 1 if sum wrapped (t_lo < p_lo0 means carry_in caused overflow)
            "sltu {ca}, {t_lo}, {p_lo0}",
            // hi_total = p_hi0 + overflow_bit  →  this is the carry for next limb
            "add.d {p_hi0}, {p_hi0}, {ca}",

            // Add the (lo + carry) to the destination limb.
            // result = dst[0] + (lo + carry_in)
            "add.d {t0}, {t_lo}, {d0}",
            // cb = 1 if this addition wrapped
            "sltu {cb}, {t0}, {d0}",
            // carry_in for next limb = hi_total + any overflow from dst addition
            "add.d {carry_in}, {p_hi0}, {cb}",
            // Store updated destination limb
            "st.d {t0}, {dst}, 0",

            // ── Limb 1: product = s1 × scalar ──────────────────────────
            "mul.d {p_lo1}, {s1}, {scalar}",
            "mulh.du {p_hi1}, {s1}, {scalar}",

            // Add carry (from limb 0) to the low part
            "add.d {t_lo}, {p_lo1}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo1}",
            // hi_total for next iteration
            "add.d {p_hi1}, {p_hi1}, {ca}",

            // Add to destination
            "add.d {t0}, {t_lo}, {d1}",
            "sltu {cb}, {t0}, {d1}",
            // Final carry for next 2-limb block
            "add.d {carry_in}, {p_hi1}, {cb}",
            "st.d {t0}, {dst}, 8",

            // Advance source and destination pointers by 16 bytes (2 limbs)
            "addi.d {src}, {src}, 16",
            "addi.d {dst}, {dst}, 16",
            // Decrement chunk counter and loop if not zero
            "addi.d {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // ── Single-limb remainder (if len is odd) ──────────────────
            "2:",
            "beqz {rem}, 4f",        // if rem == 0, we are done
            "3:",
            "ld.d {s0}, {src}, 0",   // load final source limb
            "ld.d {d0}, {dst}, 0",   // load final destination limb

            "mul.d {p_lo0}, {s0}, {scalar}",
            "mulh.du {p_hi0}, {s0}, {scalar}",

            "add.d {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add.d {p_hi0}, {p_hi0}, {ca}",

            "add.d {t0}, {t_lo}, {d0}",
            "sltu {cb}, {t0}, {d0}",
            "add.d {carry_in}, {p_hi0}, {cb}",
            "st.d {t0}, {dst}, 0",
            "4:",
            // ── End: carry_in holds the final carry ─────────────────────

            // ── Operand constraints ────────────────────────────────────
            carry_in = inout(reg) carry_in,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            s0 = out(reg) _,
            s1 = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo = out(reg) _,
            t0 = out(reg) _,
            ca = out(reg) _,
            cb = out(reg) _,
            options(nostack)
        );
    }
    carry_in
}
