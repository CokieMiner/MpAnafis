//! Conservative architecture defaults and target dispatch.

use super::TuningProfile;

impl TuningProfile {
    /// Conservative architecture-independent profile.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            radix_decimal_recursive: 8,
            radix_small_recursive: 10,
            radix_large_recursive: 12,
            karatsuba: 18,
            toom_cook_3: 240,
            toom_cook_4: 640,
            toom_cook_6: 1_600,
            toom_cook_85: 1_900,
            toom85_paired_reconstruction_min_limbs: 256,
            toom8_full_guard_product_min_split_limbs: 384,
            sqr_karatsuba: 28,
            sqr_toom_cook_3: 64,
            sqr_toom_cook_4: 176,
            sqr_toom_cook_6: 376,
            sqr_toom_cook_85: 376,
            burnikel_ziegler: 160,
            newton_raphson: 2_880,
            burnikel_ziegler_block: 48,
            newton_reciprocal_basecase: 32,
            ntt: 0,
            ssa: 0,
            sqr_ssa: 0,
            transform_min_smaller_limbs: 1_024,
            transform_max_operand_ratio: 32,
            ssa_base_modulus_bits: 16_384,
            ssa_bnm1_basecase_limbs: 32,
            ssa_negacyclic_factor3: 48,
            ssa_negacyclic_factor5: 32,
            ssa_coefficient_visit_overhead: 16,
            ssa_basecase_cost_weight_16ths: 16,
            ssa_nested_cost_penalty_16ths: 16,
            ssa_direct_shift_max_limbs: 9,
        }
    }

    /// Measured on the documented Zen 5 host; the only measured architecture profile.
    const fn apply_x86_64(&mut self) {
        self.radix_decimal_recursive = 8;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 20;
        self.toom_cook_3 = 775;
        self.toom_cook_4 = 876;
        self.toom_cook_6 = 2_049;
        self.toom_cook_85 = 2_419;
        self.sqr_karatsuba = 56;
        self.sqr_toom_cook_3 = 202;
        self.sqr_toom_cook_4 = 320;
        self.sqr_toom_cook_6 = 398;
        self.sqr_toom_cook_85 = 398;
        self.burnikel_ziegler = 192;
        self.newton_raphson = 3_072;
        self.burnikel_ziegler_block = 64;
        self.newton_reciprocal_basecase = 40;
        self.ssa = 2_816;
        self.sqr_ssa = 2_304;
        self.transform_min_smaller_limbs = 1_100;
        self.ssa_nested_cost_penalty_16ths = 20;
        self.transform_max_operand_ratio = 32;
        self.ssa_base_modulus_bits = 65_536;
    }

    /// **Unmeasured.**  `AArch64` defaults reasoned from its ISA and cache family.
    const fn apply_aarch64(&mut self) {
        self.radix_decimal_recursive = 8;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 16;
        self.toom_cook_3 = 200;
        self.toom_cook_4 = 550;
        self.toom_cook_6 = 1_360;
        self.toom_cook_85 = 1_600;
        self.sqr_karatsuba = 28;
        self.sqr_toom_cook_3 = 50;
        self.sqr_toom_cook_4 = 150;
        self.sqr_toom_cook_6 = 320;
        self.sqr_toom_cook_85 = 320;
        self.burnikel_ziegler = 120;
        self.newton_raphson = 2_400;
        self.burnikel_ziegler_block = 40;
        self.newton_reciprocal_basecase = 24;
        self.ssa = 4_096;
        self.sqr_ssa = 4_096;
        self.ssa_base_modulus_bits = 32_768;
    }

    /// **Unmeasured.** POWER and s390x defaults, grouped until either is measured.
    const fn apply_power_s390x(&mut self) {
        self.radix_decimal_recursive = 8;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 20;
        self.sqr_karatsuba = 36;
        self.sqr_toom_cook_3 = 60;
        self.burnikel_ziegler = 160;
        self.newton_raphson = 2_800;
        self.burnikel_ziegler_block = 48;
        self.newton_reciprocal_basecase = 32;
        self.ssa = 4_096;
        self.sqr_ssa = 4_096;
        self.ssa_base_modulus_bits = 32_768;
    }

    /// **Unmeasured.** The conservative floor for other 64-bit targets.
    const fn apply_generic_64(&mut self) {
        self.radix_decimal_recursive = 8;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 16;
        self.toom_cook_3 = 192;
        self.toom_cook_4 = 550;
        self.toom_cook_6 = 1_360;
        self.toom_cook_85 = 1_600;
        self.sqr_karatsuba = 28;
        self.sqr_toom_cook_3 = 48;
        self.sqr_toom_cook_4 = 150;
        self.sqr_toom_cook_6 = 320;
        self.sqr_toom_cook_85 = 320;
        self.burnikel_ziegler = 112;
        self.newton_raphson = 2_300;
        self.burnikel_ziegler_block = 40;
        self.newton_reciprocal_basecase = 24;
        self.ssa = 4_096;
        self.sqr_ssa = 4_096;
        self.ssa_base_modulus_bits = 16_384;
    }

    /// **Unmeasured.** 32-bit hosts with an operating system.
    const fn apply_std32(&mut self) {
        self.radix_decimal_recursive = 16;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 28;
        self.toom_cook_3 = 320;
        self.toom_cook_4 = 900;
        self.toom_cook_6 = 2_400;
        self.toom_cook_85 = 2_800;
        self.sqr_karatsuba = 42;
        self.sqr_toom_cook_3 = 80;
        self.sqr_toom_cook_4 = 250;
        self.sqr_toom_cook_6 = 560;
        self.sqr_toom_cook_85 = 560;
        self.burnikel_ziegler = 230;
        self.newton_raphson = 3_600;
        self.burnikel_ziegler_block = 80;
        self.newton_reciprocal_basecase = 48;
        self.ssa = 8_192;
        self.sqr_ssa = 8_192;
    }

    /// **Unmeasured.** 32-bit embedded targets and wasm32 keep transforms disabled.
    const fn apply_embedded_wasm(&mut self) {
        self.radix_decimal_recursive = 16;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 14;
        self.toom_cook_3 = 160;
        self.toom_cook_4 = 450;
        self.toom_cook_6 = 1_200;
        self.toom_cook_85 = 1_400;
        self.sqr_karatsuba = 22;
        self.sqr_toom_cook_3 = 40;
        self.sqr_toom_cook_4 = 125;
        self.sqr_toom_cook_6 = 275;
        self.sqr_toom_cook_85 = 275;
        self.burnikel_ziegler = 96;
        self.newton_raphson = 1_900;
        self.burnikel_ziegler_block = 32;
        self.newton_reciprocal_basecase = 20;
    }

    /// **Unmeasured.** AVR, MSP430, and other 16-bit targets.
    const fn apply_16bit(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 32;
        self.toom_cook_3 = 384;
        self.toom_cook_4 = 1_100;
        self.toom_cook_6 = 2_700;
        self.toom_cook_85 = 3_200;
        self.sqr_karatsuba = 42;
        self.sqr_toom_cook_3 = 96;
        self.sqr_toom_cook_4 = 300;
        self.sqr_toom_cook_6 = 625;
        self.sqr_toom_cook_85 = 625;
        self.burnikel_ziegler = 270;
        self.newton_raphson = 4_300;
        self.burnikel_ziegler_block = 90;
        self.newton_reciprocal_basecase = 56;
    }
}

/// Select the conservative built-in profile for a target architecture.
#[must_use]
pub fn profile_for_target(target_arch: &str, pointer_width: &str) -> TuningProfile {
    let mut profile = TuningProfile::portable();
    match pointer_width {
        "64" => match target_arch {
            "x86_64" => profile.apply_x86_64(),
            "aarch64" | "arm64ec" => profile.apply_aarch64(),
            "powerpc64" | "powerpc64le" | "s390x" => profile.apply_power_s390x(),
            _ => profile.apply_generic_64(),
        },
        "32" => match target_arch {
            "wasm32" | "riscv32" | "loongarch32" | "xtensa" => profile.apply_embedded_wasm(),
            _ => profile.apply_std32(),
        },
        "16" => profile.apply_16bit(),
        _ => profile.apply_embedded_wasm(),
    }
    profile
}
