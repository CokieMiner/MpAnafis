//! ADX fixed-width row kernels owned by complete basecase multiplication.

use core::arch::asm;

use super::Limb;

macro_rules! define_fixed_add_mul {
    ($name:ident, $len:literal, $($offset:literal),+ $(,)?) => {
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
            " readable source and readable/writable destination limbs, and the ",
            "two spans must not overlap, even partially (the row reads `src` ",
            "while it writes `dst`, so any overlap is a data race)."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed-width rows remove generic loop control from measured basecase hotspots"
        )]
        #[inline(always)]
        pub unsafe fn $name(dst: *mut Limb, src: *const Limb, scalar: Limb) -> Limb {
            let carry_hi: Limb;
            // The generic ADX invariant is specialized to constant offsets:
            // CF carries `dst + product_low`, OF carries the preceding product
            // high limb, and flushing both flags yields the exact closing limb.
            // SAFETY: the caller provides the exact fixed-width spans encoded
            // by the constant offset list below.
            unsafe {
                asm!(
                    "xorl %r10d, %r10d",
                    "xorl %eax, %eax",
                    $(
                        concat!("mulxq ", stringify!($offset), "({src}), %r8, %r9"),
                        concat!("movq ", stringify!($offset), "({dst}), %r11"),
                        "adcxq %r8, %r11",
                        "adoxq %r10, %r11",
                        concat!("movq %r11, ", stringify!($offset), "({dst})"),
                        "movq %r9, %r10",
                    )+
                    "movq $0, %r11",
                    "adcxq %r11, %r10",
                    "adoxq %r11, %r10",
                    "movq %r10, {carry_hi}",
                    carry_hi = out(reg) carry_hi,
                    src = in(reg) src,
                    dst = in(reg) dst,
                    in("rdx") scalar,
                    out("rax") _,
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

define_fixed_add_mul!(add_mul_4_limbs_unchecked, 4, 0, 8, 16, 24);
define_fixed_add_mul!(add_mul_5_limbs_unchecked, 5, 0, 8, 16, 24, 32);
define_fixed_add_mul!(add_mul_6_limbs_unchecked, 6, 0, 8, 16, 24, 32, 40);
define_fixed_add_mul!(add_mul_7_limbs_unchecked, 7, 0, 8, 16, 24, 32, 40, 48);
define_fixed_add_mul!(add_mul_8_limbs_unchecked, 8, 0, 8, 16, 24, 32, 40, 48, 56);
define_fixed_add_mul!(
    add_mul_9_limbs_unchecked,
    9,
    0,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
);
define_fixed_add_mul!(
    add_mul_10_limbs_unchecked,
    10,
    0,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
);
define_fixed_add_mul!(
    add_mul_11_limbs_unchecked,
    11,
    0,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
);
define_fixed_add_mul!(
    add_mul_12_limbs_unchecked,
    12,
    0,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
);
define_fixed_add_mul!(
    add_mul_13_limbs_unchecked,
    13,
    0,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
);

macro_rules! define_fixed_mul_two {
    ($name:ident, $len:literal, $close0:literal, $close1:literal, $(($offset:literal, $next:literal)),+ $(,)?) => {
        #[doc = concat!(
            "Write the first two rows of a basecase product with an exactly ",
            stringify!($len),
            "-limb inner operand."
        )]
        ///
        /// # Safety
        ///
        #[doc = concat!(
            "`src` must cover ",
            stringify!($len),
            " limbs and `dst` must cover the complete two-row output without overlap."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed two-row initialization removes generic loop control from measured basecase hotspots"
        )]
        #[inline(always)]
        pub unsafe fn $name(
            dst: *mut Limb,
            src: *const Limb,
            low_scalar: Limb,
            high_scalar: Limb,
        ) {
            // r8 and r9 are the carries of the low and high rows. At offset i,
            // dst[i] holds the high-row contribution from i-1; the low row
            // consumes it before the high row initializes dst[i+1]. The final
            // add merges the low-row carry and propagates at most one bit.
            // SAFETY: the caller proves the fixed spans encoded by the offsets.
            unsafe {
                asm!(
                    "movq 0({src}), %rdx",
                    "mulxq {low_scalar}, %r10, %r11",
                    "movq %r10, 0({dst})",
                    "movq %r11, %r8",
                    "mulxq {high_scalar}, %r10, %r11",
                    "movq %r10, 8({dst})",
                    "movq %r11, %r9",
                    $(
                        concat!("movq ", stringify!($offset), "({src}), %rdx"),
                        "mulxq {low_scalar}, %r10, %r11",
                        "addq %r8, %r10",
                        "adcq $0, %r11",
                        concat!("addq ", stringify!($offset), "({dst}), %r10"),
                        "adcq $0, %r11",
                        concat!("movq %r10, ", stringify!($offset), "({dst})"),
                        "movq %r11, %r8",
                        "mulxq {high_scalar}, %r10, %r11",
                        "addq %r9, %r10",
                        "adcq $0, %r11",
                        concat!("movq %r10, ", stringify!($next), "({dst})"),
                        "movq %r11, %r9",
                    )+
                    concat!("addq %r8, ", stringify!($close0), "({dst})"),
                    "adcq $0, %r9",
                    concat!("movq %r9, ", stringify!($close1), "({dst})"),
                    src = in(reg) src,
                    dst = in(reg) dst,
                    low_scalar = in(reg) low_scalar,
                    high_scalar = in(reg) high_scalar,
                    out("rdx") _,
                    out("r8") _,
                    out("r9") _,
                    out("r10") _,
                    out("r11") _,
                    options(nostack, att_syntax)
                );
            }
        }
    };
}

define_fixed_mul_two!(
    mul_2x4_limbs_unchecked,
    4,
    32,
    40,
    (8, 16),
    (16, 24),
    (24, 32),
);
define_fixed_mul_two!(
    mul_2x5_limbs_unchecked,
    5,
    40,
    48,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
);
define_fixed_mul_two!(
    mul_2x6_limbs_unchecked,
    6,
    48,
    56,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
);
define_fixed_mul_two!(
    mul_2x7_limbs_unchecked,
    7,
    56,
    64,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
);
define_fixed_mul_two!(
    mul_2x8_limbs_unchecked,
    8,
    64,
    72,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
    (56, 64),
);
define_fixed_mul_two!(
    mul_2x9_limbs_unchecked,
    9,
    72,
    80,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
    (56, 64),
    (64, 72),
);
define_fixed_mul_two!(
    mul_2x10_limbs_unchecked,
    10,
    80,
    88,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
    (56, 64),
    (64, 72),
    (72, 80),
);
define_fixed_mul_two!(
    mul_2x11_limbs_unchecked,
    11,
    88,
    96,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
    (56, 64),
    (64, 72),
    (72, 80),
    (80, 88),
);
define_fixed_mul_two!(
    mul_2x12_limbs_unchecked,
    12,
    96,
    104,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
    (56, 64),
    (64, 72),
    (72, 80),
    (80, 88),
    (88, 96),
);
define_fixed_mul_two!(
    mul_2x13_limbs_unchecked,
    13,
    104,
    112,
    (8, 16),
    (16, 24),
    (24, 32),
    (32, 40),
    (40, 48),
    (48, 56),
    (56, 64),
    (64, 72),
    (72, 80),
    (80, 88),
    (88, 96),
    (96, 104),
);
