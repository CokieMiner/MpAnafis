//! Backend-provider policy used by complete runtime-dispatched operations.

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
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(test, not(all(target_feature = "adx", target_feature = "bmi2")))
        ))]
        pub mod x86_64_adx;
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(test, not(all(target_feature = "adx", target_feature = "bmi2")))
        ))]
        pub mod x86_64_adx_tail;

        pub use portable::{mul_2x2_portable_unchecked, mul_3x3_portable_unchecked};
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
        pub use runtime_dispatch::$function;

        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        use super::{
            X86Backend,
            add_mul_2_limbs_unchecked::{
                add_mul_2_limbs_bmi2_backend, add_mul_2_limbs_vanilla_backend,
            },
            add_mul_limbs_unchecked::{
                add_mul_limbs_adx_backend, add_mul_limbs_bmi2_backend,
                add_mul_limbs_vanilla_backend,
            },
            mul_2_limbs_unchecked::{mul_2_limbs_bmi2_backend, mul_2_limbs_vanilla_backend},
            selected_x86_backend,
        };
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
        test_backends: [$($test_alias:ident => $test_backend:ident),* $(,)?];
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
        $(
            #[cfg(all(
                test,
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(all(target_feature = "adx", target_feature = "bmi2"))
            ))]
            pub use self::$test_backend::$function as $test_alias;
        )*
    };
}
