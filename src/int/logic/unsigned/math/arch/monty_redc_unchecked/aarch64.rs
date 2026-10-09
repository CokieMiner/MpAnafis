//! `AArch64` Montgomery reduction step kernel.
//!
//! Implements Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
//! using `AArch64` inline assembly (`mul`, `umulh`, `adds`, `adc`).

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `AArch64` inline assembly (`mul`, `umulh`, `adds`, `adc`).
///
/// For step `i`, this computes:
///
/// ```text
///   (out[0..len] + a_i * b[0..len] + q * m[0..len]) / 2^64
/// ```
///
/// where `q = ((out[0] + a_i * b[0]) * m_inv) mod 2^64`.
///
/// Stores the shifted result into `out[0..len-1]`, stores the combined low carry into
/// `out[len-1]`, and returns the top overflow carry (either 0 or 1).
///
/// Pass 1 computes `out += a_i * b` using `mul`/`umulh` with single carry-chain accumulation.
/// Pass 2 derives the quotient multiplier `q = out[0] * m_inv`.
/// Pass 3 evaluates `(out + q * m) >> 64` with offset pointer loads and stores to perform the
/// one-limb right shift during the reduction pass.
///
/// # Safety
///
/// - For nonzero `len`, all pointers must cover `len` aligned, initialized
///   limbs within `isize::MAX` bytes; `out` must be writable.
/// - `out` must be disjoint from `b` and `m`; the two inputs may overlap.
/// - `m` must be odd and `m_inv * m[0] = -1 mod 2^64`.
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the reduction step inside Montgomery multiplication"
)]
#[inline(always)]
pub unsafe fn monty_redc_step_unchecked(
    out: *mut Limb,
    b: *const Limb,
    m: *const Limb,
    len: usize,
    mut a_i: Limb,
    mut m_inv: Limb,
) -> Limb {
    if len == 0 {
        return 0;
    }

    // SAFETY: the caller provides aligned, initialized spans and a writable
    // output disjoint from the inputs. Pass one accesses exactly len limbs;
    // pass three loads limb j before writing j-1, preserving unread data.
    // Each product plus two limbs is <= (B-1)^2+2(B-1)=B^2-1, so high carries
    // fit one limb. The inverse cancels the discarded low word. The byte-span
    // bound makes len*8 exact, and every modified register is an output.
    unsafe {
        asm!(
            // --- Pass 1: out = out + a_i * b ---
            "mov {carry}, xzr",                          // Zero carry register
            "mov {offset}, xzr",                         // Zero byte offset counter

            "1:",
            "ldr {val_m_b}, [{b}, {offset}]",            // Load b[j]
            "ldr {val_out}, [{out}, {offset}]",          // Load out[j]
            "mul {p_lo}, {val_m_b}, {a_i}",              // Low 64 bits of b[j] * a_i
            "umulh {p_hi}, {val_m_b}, {a_i}",            // High 64 bits of b[j] * a_i
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry, set C flag
            "adc {p_hi}, {p_hi}, xzr",                   // p_hi += C flag + 0
            "adds {val_out}, {val_out}, {p_lo}",         // out[j] += p_lo, set C flag
            "adc {carry}, {p_hi}, xzr",                  // carry = p_hi + C flag
            "str {val_out}, [{out}, {offset}]",          // Store updated out[j]
            "add {offset}, {offset}, #8",                // Advance offset by 8 bytes
            "cmp {offset}, {len}, lsl #3",               // Check if offset < len * 8
            "b.lo 1b",                                   // Repeat while offset < len * 8
            "mov {a_i}, {carry}",                        // Save carry_b from Pass 1 into a_i

            // --- Pass 2: q = out[0] * m_inv ---
            "ldr {val_out}, [{out}]",                    // Load updated out[0]
            "mul {q}, {val_out}, {m_inv}",               // q = (out[0] * m_inv) mod 2^64

            // --- Pass 3: out = (out + q * m) >> 64 ---
            "mov {carry}, xzr",                          // Reset carry for reduction pass

            // Step 0 (j=0): compute q * m[0] + out[0], result is 0 mod 2^64, capture carry
            "ldr {val_m_b}, [{m}]",                      // Load m[0]
            "ldr {val_out}, [{out}]",                    // Load out[0]
            "mul {p_lo}, {val_m_b}, {q}",                // Low 64 bits of m[0] * q
            "umulh {p_hi}, {val_m_b}, {q}",              // High 64 bits of m[0] * q
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry
            "adc {p_hi}, {p_hi}, xzr",                   // p_hi += C flag
            "adds {val_out}, {val_out}, {p_lo}",         // Low word is 0 (discarded by shift)
            "adc {carry}, {p_hi}, xzr",                  // Capture reduction carry

            "mov {loops}, {len}",
            "sub {loops}, {loops}, #1",                  // loops = len - 1
            "cbz {loops}, 3f",                           // If len == 1, skip loop (3f)

            "add {m_ptr}, {m}, #8",                      // m_ptr = m + 8 (read m[1] first iteration)
            "add {out_read}, {out}, #8",                 // out_read = out + 8 (read out[1] first iteration)
            "mov {out_write}, {out}",                    // out_write = out (write out[0] first iteration)

            "2:",                                        // Loop for j = 1 to len-1
            "ldr {val_m_b}, [{m_ptr}], #8",              // Load m[j] and advance m_ptr
            "ldr {val_out}, [{out_read}], #8",           // Load out[j] and advance read pointer
            "mul {p_lo}, {val_m_b}, {q}",                // Low 64 bits of m[j] * q
            "umulh {p_hi}, {val_m_b}, {q}",              // High 64 bits of m[j] * q
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry
            "adc {p_hi}, {p_hi}, xzr",                   // p_hi += C flag
            "adds {val_out}, {val_out}, {p_lo}",         // out[j] += p_lo
            "adc {carry}, {p_hi}, xzr",                  // Update carry
            "str {val_out}, [{out_write}], #8",          // Store shifted limb into out[j-1]
            "subs {loops}, {loops}, #1",                 // Decrement remaining limbs
            "b.ne 2b",                                   // Repeat while loops != 0

            "3:",
            "mov {m_inv}, {carry}",                      // Return carry_m in m_inv register

            out = in(reg) out,
            b = in(reg) b,
            m = in(reg) m,
            len = in(reg) len,
            a_i = inout(reg) a_i,                        // Outputs carry_b
            m_inv = inout(reg) m_inv,                    // Outputs carry_m
            carry = out(reg) _,
            offset = out(reg) _,
            m_ptr = out(reg) _,
            out_read = out(reg) _,
            out_write = out(reg) _,
            val_m_b = out(reg) _,
            val_out = out(reg) _,
            p_lo = out(reg) _,
            p_hi = out(reg) _,
            q = out(reg) _,
            loops = out(reg) _,
            options(nostack)
        );
    }

    let carry_b = a_i;
    let carry_m = m_inv;
    // SAFETY: len > 0 makes len-1 exact and selects the writable top limb.
    unsafe {
        let (final_sum, final_carry) = carry_b.overflowing_add(carry_m);
        *out.add(len.unchecked_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
