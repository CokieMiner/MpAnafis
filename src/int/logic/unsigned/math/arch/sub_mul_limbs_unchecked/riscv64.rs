//! RISC-V 64-bit fused multiply-subtract limb kernel.
//!
//! Uses 64×64→128-bit unsigned multipliers (`mul`/`mulhu`) and branchless
//! overflow/underflow detection using `sltu` (Set Less Than Unsigned).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// RISC-V is an orthogonal load-store architecture without implicit flags registers.
/// Multiply-subtract calculates 128-bit products via `mul`/`mulhu`, accumulates the multiplication
/// carry branchlessly with `sltu`, and performs two-stage subtraction borrow propagation using
/// `sltu` and `or`. The loop is 2-way unrolled (16 bytes per iteration).
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 64-bit RISC-V multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_in: Limb = 0;
    let mut borrow_out: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder loop (2f)

            // Main 2-way unrolled loop body
            "1:",

            // [Load 2 Source and 2 Destination Limbs]
            "ld {s0}, 0({src})",                         // Load src[0]
            "ld {s1}, 8({src})",                         // Load src[1]
            "ld {d0}, 0({dst})",                         // Load dst[0]
            "ld {d1}, 8({dst})",                         // Load dst[1]

            // [Limb 0 Multiply-Subtract]
            "mul {p_lo0}, {s0}, {scalar}",               // p_lo0 = low 64 bits of src[0] * scalar
            "mulhu {p_hi0}, {s0}, {scalar}",             // p_hi0 = high 64 bits of src[0] * scalar
            "add {t_lo}, {p_lo0}, {carry_in}",           // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if addition wrapped, else 0
            "add {carry_in}, {p_hi0}, {ca}",             // carry_in = p_hi0 + ca
            "sub {t0}, {d0}, {t_lo}",                    // t0 = d0 - t_lo
            "sltu {b0}, {d0}, {t_lo}",                   // b0 = 1 if first subtraction underflowed
            "sub {t1}, {t0}, {borrow_out}",              // t1 = t0 - borrow_out
            "sltu {b1}, {t0}, {borrow_out}",             // b1 = 1 if second subtraction underflowed
            "or {borrow_out}, {b0}, {b1}",               // borrow_out = b0 | b1 (combined borrow)
            "sd {t1}, 0({dst})",                         // Store updated dst[0]

            // [Limb 1 Multiply-Subtract]
            "mul {p_lo1}, {s1}, {scalar}",               // Low 64 bits of src[1] * scalar
            "mulhu {p_hi1}, {s1}, {scalar}",             // High 64 bits of src[1] * scalar
            "add {t_lo}, {p_lo1}, {carry_in}",           // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if addition wrapped
            "add {carry_in}, {p_hi1}, {ca}",             // carry_in = p_hi1 + ca
            "sub {t0}, {d1}, {t_lo}",                    // t0 = d1 - t_lo
            "sltu {b0}, {d1}, {t_lo}",                   // First borrow
            "sub {t1}, {t0}, {borrow_out}",              // t1 = t0 - borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combined borrow
            "sd {t1}, 8({dst})",                         // Store updated dst[1]

            // Advance pointers by 2 limbs (16 bytes) and loop
            "addi {src}, {src}, 16",
            "addi {dst}, {dst}, 16",
            "addi {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // Remainder processing (0 or 1 limb)
            "2:",
            "beqz {rem}, 4f",

            // 1-limb tail
            "3:",
            "ld {s0}, 0({src})",                         // Load single src limb
            "ld {d0}, 0({dst})",                         // Load single dst limb
            "mul {p_lo0}, {s0}, {scalar}",               // Low 64-bit product
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 64-bit product
            "add {t_lo}, {p_lo0}, {carry_in}",           // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "add {carry_in}, {p_hi0}, {ca}",             // Propagate carry
            "sub {t0}, {d0}, {t_lo}",                    // Subtract product
            "sltu {b0}, {d0}, {t_lo}",                   // First borrow
            "sub {t1}, {t0}, {borrow_out}",              // Subtract previous borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combine borrow bits
            "sd {t1}, 0({dst})",                         // Store updated limb

            // Tail completion
            "4:",

            carry_in = inout(reg) carry_in,
            borrow_out = inout(reg) borrow_out,
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
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo = out(reg) _,
            ca = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_in, borrow_out)
}
