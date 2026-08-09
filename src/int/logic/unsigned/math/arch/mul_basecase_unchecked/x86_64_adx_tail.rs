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
            " readable source and readable/writable destination limbs."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed-width rows remove generic loop control from measured basecase hotspots"
        )]
        #[inline(always)]
        pub unsafe fn $name(dst: *mut Limb, src: *const Limb, scalar: Limb) -> Limb {
            let carry_hi: Limb;
            // CF carries the preceding product high; OF carries the
            // destination addition. Each second `mulx` starts before the
            // first result is stored, overlapping its multiplication latency.
            // SAFETY: the caller proves the spans encoded by these offsets.
            unsafe {
                asm!(
                    "xorl %r10d, %r10d",
                    $(
                        concat!("mulxq ", stringify!($even), "({src}), %r8, %r9"),
                        "adcxq %r10, %r8",
                        concat!("adoxq ", stringify!($even), "({dst}), %r8"),
                        concat!("mulxq ", stringify!($odd), "({src}), %r11, %r10"),
                        concat!("movq %r8, ", stringify!($even), "({dst})"),
                        "adcxq %r9, %r11",
                        concat!("adoxq ", stringify!($odd), "({dst}), %r11"),
                        concat!("movq %r11, ", stringify!($odd), "({dst})"),
                    )+
                    "movq $0, %r11",
                    "adcxq %r11, %r10",
                    "adoxq %r11, %r10",
                    "movq %r10, {carry_hi}",
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
            " readable source and readable/writable destination limbs."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed-width rows remove generic loop control from measured basecase hotspots"
        )]
        #[inline(always)]
        pub unsafe fn $name(dst: *mut Limb, src: *const Limb, scalar: Limb) -> Limb {
            let carry_hi: Limb;
            // The paired body leaves the preceding high limb in r10. The
            // final odd product consumes it and leaves its high in r9 for the
            // exact two-chain flush.
            // SAFETY: the caller proves the spans encoded by these offsets.
            unsafe {
                asm!(
                    "xorl %r10d, %r10d",
                    $(
                        concat!("mulxq ", stringify!($even), "({src}), %r8, %r9"),
                        "adcxq %r10, %r8",
                        concat!("adoxq ", stringify!($even), "({dst}), %r8"),
                        concat!("mulxq ", stringify!($odd), "({src}), %r11, %r10"),
                        concat!("movq %r8, ", stringify!($even), "({dst})"),
                        "adcxq %r9, %r11",
                        concat!("adoxq ", stringify!($odd), "({dst}), %r11"),
                        concat!("movq %r11, ", stringify!($odd), "({dst})"),
                    )+
                    concat!("mulxq ", stringify!($last), "({src}), %r8, %r9"),
                    "adcxq %r10, %r8",
                    concat!("adoxq ", stringify!($last), "({dst}), %r8"),
                    concat!("movq %r8, ", stringify!($last), "({dst})"),
                    "movq $0, %r11",
                    "adcxq %r11, %r9",
                    "adoxq %r11, %r9",
                    "movq %r9, {carry_hi}",
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
