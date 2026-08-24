//! Declarative architecture-kernel availability and dispatch wiring.

/// Declares the backend coverage and feature policy for one architecture
/// kernel.
///
/// Ordinary backends pair a module name with their complete availability
/// predicate. The `x86_64` and `powerpc64` fields select the shared feature
/// policies for those targets. The fallback is derived from the same coverage
/// declarations, so adding a backend cannot leave a second stale predicate.
macro_rules! select_arch_kernel {
    (
        function: $function:ident;
        kernel: $kernel:ident;
        surface: provider;
        $($configuration:tt)*
    ) => {
        select_arch_provider! {
            function: $function;
            kernel: $kernel;
            $($configuration)*
        }
    };
    (
        function: $function:ident;
        surface: composite;
        x86_64: $x86_policy:tt;
    ) => {
        select_arch_provider! {
            function: $function;
            surface: composite;
            x86_64: $x86_policy;
        }
    };
    (
        @when $availability:meta {
            $($item:item)*
        }
    ) => {
        $(
            #[cfg($availability)]
            $item
        )*
    };
    (@import_limb ntt_float_f64) => {};
    (@import_limb $other:ident) => {
        use super::Limb;
    };
    (@import_kernel_types NttFloatKernel) => {
        use super::kernels::NttFloatKernels;
    };
    (@import_kernel_types $other:ident) => {};
    (
        function: $function:ident;
        surface: direct;
        backends: [$($backend:ident => $availability:meta),* $(,)?];
        x86_64: $x86_policy:tt;
        powerpc64: $power_policy:tt;
        special_coverage: [$($special_coverage:meta),* $(,)?];
        fallback_imports: [$($fallback_import:ident),* $(,)?];
    ) => {
        $(
            select_arch_kernel!(
                @direct_backend $function, $backend, $availability
            );
        )*
        select_arch_kernel!(@direct_x86 $function, $x86_policy);
        select_arch_kernel!(@direct_powerpc64 $function, $power_policy);
        select_arch_kernel!(
            @direct_fallback_policy
            $function,
            any(
                miri,
                not(any($($availability,)* $($special_coverage),*))
            ),
            [$($fallback_import),*],
            $x86_policy
        );
        pub use selected::$function;
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        surface: selector;
        backends: [$($backend:ident => $availability:meta),* $(,)?];
        x86_64: $x86_policy:tt;
        powerpc64: $power_policy:tt;
        special_coverage: [$($special_coverage:meta),* $(,)?];
        fallback_imports: [$($fallback_import:ident),* $(,)?];
        test_backends: [$($test_alias:ident => $test_backend:ident),* $(,)?];
    ) => {
        use super::kernels::$kernel;
        select_arch_kernel!(@import_kernel_types $kernel);
        select_arch_kernel!(@import_limb $function);
        $(
            select_arch_kernel!(
                @backend $function, $kernel, $backend, $availability
            );
        )*
        select_arch_kernel!(@powerpc64 $function, $kernel, $power_policy);
        select_x86_kernel! {
            function: $function;
            kernel: $kernel;
            policy: $x86_policy;
            generic: any(miri, not(any($($availability,)* $($special_coverage),*)));
            fallback_imports: [$($fallback_import),*];
            test_backends: [$($test_alias => $test_backend),*];
        }
        pub use selected::kernel;
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        surface: selectable;
        backends: [$($backend:ident => $availability:meta),* $(,)?];
        x86_64: $x86_policy:tt;
        powerpc64: $power_policy:tt;
        special_coverage: [$($special_coverage:meta),* $(,)?];
        fallback_imports: [$($fallback_import:ident),* $(,)?];
    ) => {
        $(
            select_arch_kernel!(
                @backend $function, $kernel, $backend, $availability
            );
        )*
        select_arch_kernel!(@x86 $function, $kernel, $x86_policy);
        select_arch_kernel!(@powerpc64 $function, $kernel, $power_policy);
        select_arch_kernel!(
            @fallback
            $function,
            $kernel,
            any(
                miri,
                not(any($($availability,)* $($special_coverage),*))
            ),
            [$($fallback_import),*]
        );
        select_arch_kernel!(@kernel_import $kernel, $x86_policy);
        select_arch_kernel!(@surface $function, $x86_policy);
    };

    (
        @direct_backend
        $function:ident,
        $backend:ident,
        $availability:meta
    ) => {
        #[cfg($availability)]
        mod $backend;
        #[cfg($availability)]
        mod selected {
            pub use super::$backend::$function;
        }
    };

    (@backend $function:ident, $kernel:ident, $backend:ident, $availability:meta) => {
        #[cfg($availability)]
        mod $backend;
        #[cfg($availability)]
        mod selected {
            use super::$kernel;

            pub use super::$backend::$function;

            #[inline]
            pub const fn kernel() -> $kernel {
                $function
            }
        }
    };

    (@direct_fallback $function:ident, $availability:meta, []) => {
        select_arch_kernel!(@direct_backend $function, fallback, $availability);
    };
    (
        @direct_fallback
        $function:ident,
        $availability:meta,
        [$($fallback_import:ident),+]
    ) => {
        #[cfg($availability)]
        use super::{$($fallback_import),+};
        select_arch_kernel!(@direct_backend $function, fallback, $availability);
    };
    (
        @direct_fallback_policy
        $function:ident,
        $availability:meta,
        $fallback_imports:tt,
        [adx]
    ) => {};
    (
        @direct_fallback_policy
        $function:ident,
        $availability:meta,
        $fallback_imports:tt,
        $x86_policy:tt
    ) => {
        select_arch_kernel!(
            @direct_fallback $function, $availability, $fallback_imports
        );
    };

    (@fallback $function:ident, $kernel:ident, $availability:meta, []) => {
        select_arch_kernel!(@backend $function, $kernel, fallback, $availability);
    };
    (
        @fallback
        $function:ident,
        $kernel:ident,
        $availability:meta,
        [$($fallback_import:ident),+]
    ) => {
        #[cfg($availability)]
        use super::{$($fallback_import),+};
        select_arch_kernel!(@backend $function, $kernel, fallback, $availability);
    };

    (@direct_x86 $function:ident, [baseline]) => {
        select_arch_kernel!(
            @direct_backend $function, x86_64,
            all(not(miri), target_arch = "x86_64", target_pointer_width = "64")
        );
    };
    (@direct_x86 $function:ident, [adx]) => {
        select_arch_kernel!(@when not(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx"
        )) {
            mod fallback;
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(feature = "std", target_feature = "adx")
        ) {
            mod x86_64_adx;
        });
        select_arch_kernel!(@when all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "adx")
        ) {
            pub mod runtime_dispatch;
            use super::{X86Backend, selected_x86_backend};
            mod selected {
                pub use super::runtime_dispatch::$function;
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx"
        ) {
            mod selected {
                pub use super::x86_64_adx::$function;
            }
        });
        select_arch_kernel!(@when not(any(
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                target_feature = "adx"
            ),
            all(
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = "adx")
            )
        )) {
            mod selected {
                pub use super::fallback::$function;
            }
        });
    };
    (@direct_x86 $function:ident, []) => {};

    (@x86 $function:ident, $kernel:ident, [baseline]) => {
        select_arch_kernel!(
            @backend $function, $kernel, x86_64,
            all(not(miri), target_arch = "x86_64", target_pointer_width = "64")
        );
    };
    (@x86 $function:ident, $kernel:ident, [bmi2, adx_bmi2]) => {
        select_arch_kernel!(
            @backend $function, $kernel, x86_64_adx,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                target_feature = "adx",
                target_feature = "bmi2"
            )
        );
        select_arch_kernel!(
            @backend $function, $kernel, x86_64,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = "bmi2"),
                not(feature = "std")
            )
        );
        select_arch_kernel!(
            @backend $function, $kernel, x86_64_bmi2,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                target_feature = "bmi2",
                not(target_feature = "adx"),
                not(feature = "std")
            )
        );
        select_arch_kernel!(@x86_runtime_modules $function, $kernel, adx_bmi2);
    };
    (@x86 $function:ident, $kernel:ident, [bmi2]) => {
        select_arch_kernel!(
            @backend $function, $kernel, x86_64_bmi2,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                target_feature = "adx",
                target_feature = "bmi2"
            )
        );
        select_arch_kernel!(
            @backend $function, $kernel, x86_64,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = "bmi2"),
                not(feature = "std")
            )
        );
        select_arch_kernel!(
            @backend $function, $kernel, x86_64_bmi2,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                target_feature = "bmi2",
                not(target_feature = "adx"),
                not(feature = "std")
            )
        );
        select_arch_kernel!(@x86_runtime_modules $function, $kernel, test_bmi2);
    };
    (@x86 $function:ident, $kernel:ident, []) => {};

    (@x86_runtime_modules $function:ident, $kernel:ident, adx_bmi2) => {
        select_arch_kernel!(@when all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ) {
            mod x86_64;
            mod x86_64_adx;
            mod x86_64_bmi2;
            mod runtime_dispatch;
            use super::{X86Backend, selected_x86_backend};
            mod selected {
                use super::$kernel;

                pub use super::runtime_dispatch::$function;

                #[inline]
                pub fn kernel() -> $kernel {
                    super::runtime_dispatch::selected_kernel()
                }
            }
        });
    };
    (@x86_runtime_modules $function:ident, $kernel:ident, test_bmi2) => {
        select_arch_kernel!(@when all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ) {
            mod x86_64;
            mod x86_64_bmi2;
        });
        select_arch_kernel!(@when all(
            test,
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ) {
            mod runtime_dispatch;
            use super::{X86Backend, selected_x86_backend};
            mod selected {
                use super::$kernel;

                #[inline]
                pub fn kernel() -> $kernel {
                    super::runtime_dispatch::selected_kernel()
                }
            }
        });
    };

    (@direct_powerpc64 $function:ident, [baseline]) => {
        select_arch_kernel!(
            @direct_backend $function, powerpc64,
            all(not(miri), target_arch = "powerpc64")
        );
    };
    (@direct_powerpc64 $function:ident, [power8, power9]) => {
        select_arch_kernel!(
            @direct_backend $function, powerpc64_p9,
            all(
                not(miri),
                target_arch = "powerpc64",
                target_feature = "power9-vector"
            )
        );
        select_arch_kernel!(
            @direct_backend $function, powerpc64,
            all(
                not(miri),
                target_arch = "powerpc64",
                not(target_feature = "power9-vector")
            )
        );
    };
    (@direct_powerpc64 $function:ident, []) => {};

    (@powerpc64 $function:ident, $kernel:ident, [baseline]) => {
        select_arch_kernel!(
            @backend $function, $kernel, powerpc64,
            all(not(miri), target_arch = "powerpc64")
        );
    };
    (@powerpc64 $function:ident, $kernel:ident, []) => {};
    (@powerpc64 $function:ident, $kernel:ident, [power8, power9]) => {
        select_arch_kernel!(
            @backend $function, $kernel, powerpc64_p9,
            all(
                not(miri),
                target_arch = "powerpc64",
                target_feature = "power9-vector"
            )
        );
        select_arch_kernel!(
            @backend $function, $kernel, powerpc64,
            all(
                not(miri),
                target_arch = "powerpc64",
                not(target_feature = "power9-vector")
            )
        );
    };
    (@powerpc64 $function:ident, $kernel:ident, []) => {};

    (@kernel_import $kernel:ident, [bmi2]) => {
        with_direct_basecase_components! {
            use super::kernels::$kernel;
        }
    };
    (@kernel_import $kernel:ident, $x86_policy:tt) => {
        use super::kernels::$kernel;
    };

    (@surface $function:ident, [bmi2]) => {
        with_direct_basecase_components! {
            pub use selected::kernel;
        }
    };
    (@surface $function:ident, $x86_policy:tt) => {
        pub use selected::$function;
        pub use selected::kernel;
    };
}
