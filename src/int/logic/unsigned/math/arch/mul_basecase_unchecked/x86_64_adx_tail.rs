//! ADX fixed rows beyond the compact-kernel block.

use core::arch::asm;

use super::Limb;

macro_rules! define_pipelined_fixed_add_mul {
    ($name:ident, $len:literal, $(($even:literal, $odd:literal)),+ $(,)?) => {
        #[doc = concat!(
            "Multiply exactly ",
            stringify!($len),
            " source limbs by one scalar and add into the destination."
        )]
        ///
        /// # Safety
        ///
        #[doc = concat!(
            "`src` and `dst` must cover exactly ",
            stringify!($len),
            " aligned initialized source and writable destination limbs. ",
            "The spans must not overlap; the CPU must support ADX and BMI2."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed-width rows remove generic loop control inside the complete basecase kernel"
        )]
        #[inline(always)]
        pub unsafe fn $name(dst: *mut Limb, src: *const Limb, scalar: Limb) -> Limb {
            let carry_hi: Limb;
            // CF carries the preceding product high; OF carries the
            // destination addition. Each second `mulx` starts before the
            // first result is stored.
            // SAFETY: aligned initialized disjoint spans cover every literal
            // offset, and the caller guarantees ADX/BMI2. Products plus two
            // limbs are <= B^2-1. Each output consumes both incoming carries;
            // flushing CF and OF gives the high limb. All clobbers are outputs.
            unsafe {
                asm!(
                    "xorl %r10d, %r10d",                          // Clear r10 (previous high product) and OF/CF flags
                    $(
                        concat!("mulxq ", stringify!($even), "({src}), %r8, %r9"),   // (%r9:%r8) = src[even] * scalar
                        "adcxq %r10, %r8",                        // r8 += previous high + CF
                        concat!("adoxq ", stringify!($even), "({dst}), %r8"),        // r8 += dst[even] + OF
                        concat!("mulxq ", stringify!($odd), "({src}), %r11, %r10"),  // (%r10:%r11) = src[odd] * scalar
                        concat!("movq %r8, ", stringify!($even), "({dst})"),         // Store updated dst[even]
                        "adcxq %r9, %r11",                        // r11 += high(even) + CF
                        concat!("adoxq ", stringify!($odd), "({dst}), %r11"),        // r11 += dst[odd] + OF
                        concat!("movq %r11, ", stringify!($odd), "({dst})"),         // Store updated dst[odd]
                    )+
                    "movq $0, %r11",                              // Zero r11 for carry flush
                    "adcxq %r11, %r10",                           // Flush CF into r10
                    "adoxq %r11, %r10",                           // Flush OF into r10
                    "movq %r10, {carry_hi}",                      // Store final carry-out
                    carry_hi = out(reg) carry_hi,
                    src = in(reg) src,
                    dst = in(reg) dst,
                    in("rdx") scalar,
                    out("r8") _,
                    out("r9") _,
                    out("r10") _,
                    out("r11") _,
                    options(nostack, att_syntax)
                );
            }
            carry_hi
        }
    };
}

macro_rules! define_pipelined_fixed_add_mul_odd {
    ($name:ident, $len:literal, $(($even:literal, $odd:literal)),+; $last:literal $(,)?) => {
        #[doc = concat!(
            "Multiply exactly ",
            stringify!($len),
            " source limbs by one scalar and add into the destination."
        )]
        ///
        /// # Safety
        ///
        #[doc = concat!(
            "`src` and `dst` must cover exactly ",
            stringify!($len),
            " aligned initialized source and writable destination limbs. ",
            "The spans must not overlap; the CPU must support ADX and BMI2."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed-width rows remove generic loop control inside the complete basecase kernel"
        )]
        #[inline(always)]
        pub unsafe fn $name(dst: *mut Limb, src: *const Limb, scalar: Limb) -> Limb {
            let carry_hi: Limb;
            // The paired body leaves the preceding high limb in r10. The
            // final odd product consumes it and leaves its high in r9 for the
            // exact two-chain flush.
            // SAFETY: aligned initialized disjoint spans cover every literal
            // offset, and the caller guarantees ADX/BMI2. Products plus two
            // limbs are <= B^2-1. The odd tail consumes the preceding high limb;
            // flushing CF and OF gives the high limb. All clobbers are outputs.
            unsafe {
                asm!(
                    "xorl %r10d, %r10d",                          // Clear r10 (previous high product) and OF/CF flags
                    $(
                        concat!("mulxq ", stringify!($even), "({src}), %r8, %r9"),   // (%r9:%r8) = src[even] * scalar
                        "adcxq %r10, %r8",                        // r8 += previous high + CF
                        concat!("adoxq ", stringify!($even), "({dst}), %r8"),        // r8 += dst[even] + OF
                        concat!("mulxq ", stringify!($odd), "({src}), %r11, %r10"),  // (%r10:%r11) = src[odd] * scalar
                        concat!("movq %r8, ", stringify!($even), "({dst})"),         // Store updated dst[even]
                        "adcxq %r9, %r11",                        // r11 += high(even) + CF
                        concat!("adoxq ", stringify!($odd), "({dst}), %r11"),        // r11 += dst[odd] + OF
                        concat!("movq %r11, ", stringify!($odd), "({dst})"),         // Store updated dst[odd]
                    )+
                    concat!("mulxq ", stringify!($last), "({src}), %r8, %r9"),       // Final odd limb multiply
                    "adcxq %r10, %r8",                            // r8 += previous high + CF
                    concat!("adoxq ", stringify!($last), "({dst}), %r8"),            // r8 += dst[last] + OF
                    concat!("movq %r8, ", stringify!($last), "({dst})"),            // Store dst[last]
                    "movq $0, %r11",                              // Zero r11 for carry flush
                    "adcxq %r11, %r9",                            // Flush CF into r9
                    "adoxq %r11, %r9",                            // Flush OF into r9
                    "movq %r9, {carry_hi}",                       // Store final carry-out
                    carry_hi = out(reg) carry_hi,
                    src = in(reg) src,
                    dst = in(reg) dst,
                    in("rdx") scalar,
                    out("r8") _,
                    out("r9") _,
                    out("r10") _,
                    out("r11") _,
                    options(nostack, att_syntax)
                );
            }
            carry_hi
        }
    };
}

define_pipelined_fixed_add_mul!(
    add_mul_14_limbs_unchecked,
    14,
    (0, 8),
    (16, 24),
    (32, 40),
    (48, 56),
    (64, 72),
    (80, 88),
    (96, 104),
);
define_pipelined_fixed_add_mul_odd!(
    add_mul_15_limbs_unchecked,
    15,
    (0, 8),
    (16, 24),
    (32, 40),
    (48, 56),
    (64, 72),
    (80, 88),
    (96, 104);
    112,
);
define_pipelined_fixed_add_mul!(
    add_mul_16_limbs_unchecked,
    16,
    (0, 8),
    (16, 24),
    (32, 40),
    (48, 56),
    (64, 72),
    (80, 88),
    (96, 104),
    (112, 120),
);
define_pipelined_fixed_add_mul_odd!(
    add_mul_17_limbs_unchecked,
    17,
    (0, 8),
    (16, 24),
    (32, 40),
    (48, 56),
    (64, 72),
    (80, 88),
    (96, 104),
    (112, 120);
    128,
);
