//! Backend-provider policy used by complete runtime-dispatched operations.

/// Emits paired-row component surfaces only when the direct basecase
/// composition is built. Runtime-dispatched x86-64 builds select these
/// components once as part of the complete basecase operation instead.
macro_rules! with_direct_basecase_components {
    ($($item:item)*) => {
        $(
            #[cfg(not(all(
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(all(target_feature = "adx", target_feature = "bmi2"))
            )))]
            $item
        )*
    };
}

macro_rules! select_arch_provider {
    (
        function: $function:ident;
        surface: composite;
        x86_64: [bmi2, adx_bmi2];
    ) => {
        use super::{DoubleLimb, Limb};
        #[cfg(not(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        )))]
        use super::ArchKernels;

        mod portable;
        #[cfg(not(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        )))]
        mod direct;
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        mod runtime_dispatch;
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(feature = "std", all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        pub mod x86_64_adx;
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        pub mod x86_64_adx_tail;

        pub use portable::{
            mul_2x2_portable_unchecked, mul_3x3_portable_unchecked,
        };

        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx",
            target_feature = "bmi2"
        ))]
        pub use x86_64_adx::{
            mul_4x4_adx_unchecked as mul_4x4_unchecked,
            mul_8x8_adx_unchecked as mul_8x8_unchecked,
        };

        #[cfg(not(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(feature = "std", all(target_feature = "adx", target_feature = "bmi2"))
        )))]
        pub use portable::{
            mul_4x4_portable_unchecked as mul_4x4_unchecked,
            mul_8x8_portable_unchecked as mul_8x8_unchecked,
        };

        #[cfg(not(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        )))]
        pub use direct::$function;

        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        pub use runtime_dispatch::{$function, mul_4x4_unchecked, mul_8x8_unchecked};

        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        use super::{
            X86Backend, add_mul_2_limbs_bmi2_backend, add_mul_2_limbs_vanilla_backend,
            add_mul_limbs_adx_backend, add_mul_limbs_bmi2_backend,
            add_mul_limbs_vanilla_backend, mul_2_limbs_bmi2_backend,
            mul_2_limbs_vanilla_backend, selected_x86_backend,
        };
    };
    (
        function: $function:ident;
        surface: composite;
        x86_64: [adx_bmi2, bmi2];
    ) => {
        use super::{ArchKernels, Limb};
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "bmi2")
        ))]
        use super::{X86Backend, selected_x86_backend};

        mod direct;
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "bmi2")
        ))]
        mod runtime_dispatch;
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                all(target_feature = "adx", target_feature = "bmi2"),
                all(feature = "std", not(target_feature = "bmi2"))
            )
        ))]
        pub mod x86_64_adx;
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2")),
            any(feature = "std", target_feature = "bmi2")
        ))]
        pub mod x86_64_bmi2;

        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "bmi2"
        ) {
            mod selected;
            pub use direct::$function as small_square;
            pub use selected::$function;
        });
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "bmi2",
            target_feature = "adx"
        ))]
        pub use x86_64_adx::$function as large_square;
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "bmi2",
            not(target_feature = "adx")
        ))]
        pub use x86_64_bmi2::$function as large_square;
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "bmi2")
        ))]
        pub use runtime_dispatch::$function;
        #[cfg(not(any(
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                all(target_feature = "adx", target_feature = "bmi2")
            ),
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                all(target_feature = "bmi2", not(target_feature = "adx"))
            ),
            all(
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = "bmi2")
            )
        )))]
        pub use direct::$function;
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        backends: [$($backend:ident => $availability:meta),* $(,)?];
        x86_64: $x86_policy:tt;
        powerpc64: $power_policy:tt;
        special_coverage: [$($special_coverage:meta),* $(,)?];
        fallback_imports: [$($fallback_import:ident),* $(,)?];
        runtime_backends: [$($runtime_alias:ident => $runtime_backend:ident),* $(,)?];
    ) => {
        select_arch_kernel! {
            function: $function;
            kernel: $kernel;
            surface: selectable;
            backends: [$($backend => $availability),*];
            x86_64: $x86_policy;
            powerpc64: $power_policy;
            special_coverage: [$($special_coverage),*];
            fallback_imports: [$($fallback_import),*];
        }
        $(
            #[cfg(all(
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(all(target_feature = "adx", target_feature = "bmi2"))
            ))]
            pub use self::$runtime_backend::$function as $runtime_alias;
        )*
    };
}
