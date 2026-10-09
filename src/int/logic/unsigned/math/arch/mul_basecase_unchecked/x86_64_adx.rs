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
            "two spans must not overlap. Pointers must be aligned and limbs ",
            "initialized. The CPU must support ADX and BMI2."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed-width rows remove generic loop control inside the complete basecase kernel"
        )]
        #[inline(always)]
        pub unsafe fn $name(dst: *mut Limb, src: *const Limb, scalar: Limb) -> Limb {
            let carry_hi: Limb;
            // The generic ADX invariant is specialized to constant offsets:
            // CF carries `dst + product_low`, OF carries the preceding product
            // high limb, and flushing both flags yields the exact closing limb.
            // SAFETY: the caller provides aligned initialized disjoint spans
            // and ADX/BMI2. Constant offsets access exactly the declared width.
            // Each product plus two limbs is <= B^2-1; flushing CF and OF gives
            // the closing high limb. Every modified register is an output.
            unsafe {
                asm!(
                    "xorl %r10d, %r10d",                          // Clear r10 (previous high product) and OF/CF flags
                    $(
                        concat!("mulxq ", stringify!($offset), "({src}), %r8, %r9"), // (%r9:%r8) = src[i] * scalar
                        concat!("adcxq ", stringify!($offset), "({dst}), %r8"),      // r8 += dst[i] + CF
                        "adoxq %r10, %r8",                        // r8 += previous high + OF
                        concat!("movq %r8, ", stringify!($offset), "({dst})"),       // Store updated dst[i]
                        "movq %r9, %r10",                         // Save high product for next limb
                    )+
                    "movq $0, %r8",                               // Zero r8 for carry flush
                    "adcxq %r8, %r10",                            // Flush CF into r10
                    "adoxq %r8, %r10",                            // Flush OF into r10
                    "movq %r10, {carry_hi}",                      // Store final carry-out
                    carry_hi = out(reg) carry_hi,
                    src = in(reg) src,
                    dst = in(reg) dst,
                    in("rdx") scalar,
                    out("r8") _,
                    out("r9") _,
                    out("r10") _,
                    options(nostack, att_syntax)
                );
            }
            carry_hi
        }
    };
}

define_fixed_add_mul!(add_mul_4_limbs_unchecked, 4, 0, 8, 16, 24);
define_fixed_add_mul!(add_mul_8_limbs_unchecked, 8, 0, 8, 16, 24, 32, 40, 48, 56);

// Other fixed widths belong to the runtime-selected complete operation.
select_arch_kernel!(@when not(all(target_feature = "adx", target_feature = "bmi2")) {
define_fixed_add_mul!(add_mul_5_limbs_unchecked, 5, 0, 8, 16, 24, 32);
define_fixed_add_mul!(add_mul_6_limbs_unchecked, 6, 0, 8, 16, 24, 32, 40);
define_fixed_add_mul!(add_mul_7_limbs_unchecked, 7, 0, 8, 16, 24, 32, 40, 48);
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
});

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
            " aligned initialized limbs and `dst` must cover ",
            stringify!($len),
            " + 2 aligned writable limbs, which may be uninitialized. ",
            "The spans must not overlap; the CPU must support BMI2."
        )]
        #[allow(
            clippy::inline_always,
            reason = "Fixed two-row initialization removes generic loop control inside the complete basecase kernel"
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
            // SAFETY: the caller provides disjoint aligned spans and BMI2.
            // Source offsets are below the fixed width; stores cover width+2.
            // Every destination read consumes a preceding row-one store.
            // Products plus two limbs are <= B^2-1 and the final high carry
            // is <= high_scalar. Every modified register is an output.
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

select_arch_kernel!(@when not(all(target_feature = "adx", target_feature = "bmi2")) {
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
});

/// Write the exact product of two four-limb operands with fixed ADX rows.
///
/// # Safety
///
/// `a` and `b` must each be readable for four limbs, `dst` must be writable
/// for eight limbs, and neither input span may overlap `dst`. The caller must
/// additionally prove that the executing CPU supports ADX and BMI2.
#[expect(
    clippy::inline_always,
    reason = "Inlining the fixed row primitives produces one call-free four-by-four kernel"
)]
#[inline(always)]
pub unsafe fn mul_4x4_adx_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // The first primitive initializes two complete rows. The following two
    // fixed add-mul rows close the exact eight-limb product without a loop,
    // width match, or generic add-mul call.
    // SAFETY: the caller proves both four-limb inputs, the disjoint eight-limb
    // destination, and the ADX+BMI2 CPU feature contract.
    unsafe {
        mul_2x4_limbs_unchecked(dst, b, *a, *a.add(1));
        let carry2 = add_mul_4_limbs_unchecked(dst.add(2), b, *a.add(2));
        *dst.add(6) = carry2;
        let carry3 = add_mul_4_limbs_unchecked(dst.add(3), b, *a.add(3));
        *dst.add(7) = carry3;
    }
}

/// Write the exact product of two eight-limb operands with fixed ADX rows.
///
/// # Safety
///
/// `a` and `b` must each be readable for eight limbs, `dst` must be writable
/// for sixteen limbs, and neither input span may overlap `dst`. The caller must
/// additionally prove that the executing CPU supports ADX and BMI2.
#[expect(
    clippy::inline_always,
    reason = "Inlining the fixed row primitives produces one call-free eight-by-eight kernel"
)]
#[inline(always)]
pub unsafe fn mul_8x8_adx_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // SAFETY: the caller proves both eight-limb inputs, the disjoint sixteen-
    // limb destination, and the ADX+BMI2 CPU feature contract. Every literal
    // shifted row and its closing carry remain inside that destination.
    unsafe {
        mul_2x8_limbs_unchecked(dst, b, *a, *a.add(1));
        let carry2 = add_mul_8_limbs_unchecked(dst.add(2), b, *a.add(2));
        *dst.add(10) = carry2;
        let carry3 = add_mul_8_limbs_unchecked(dst.add(3), b, *a.add(3));
        *dst.add(11) = carry3;
        let carry4 = add_mul_8_limbs_unchecked(dst.add(4), b, *a.add(4));
        *dst.add(12) = carry4;
        let carry5 = add_mul_8_limbs_unchecked(dst.add(5), b, *a.add(5));
        *dst.add(13) = carry5;
        let carry6 = add_mul_8_limbs_unchecked(dst.add(6), b, *a.add(6));
        *dst.add(14) = carry6;
        let carry7 = add_mul_8_limbs_unchecked(dst.add(7), b, *a.add(7));
        *dst.add(15) = carry7;
    }
}
