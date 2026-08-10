//! Selector-only x86 kernel policies used by `select_arch_kernel!`.

macro_rules! select_x86_kernel {
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [fallback, adx];
        generic: $generic:meta;
        fallback_imports: $fallback_imports:tt;
        test_backends: [];
    ) => {
        select_x86_kernel!(
            @single
            $function,
            $kernel,
            x86_64_adx,
            "adx",
            $generic,
            $fallback_imports
        );
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [fallback, bmi2];
        generic: $generic:meta;
        fallback_imports: $fallback_imports:tt;
        test_backends: [];
    ) => {
        select_x86_kernel!(
            @single
            $function,
            $kernel,
            x86_64_bmi2,
            "bmi2",
            $generic,
            $fallback_imports
        );
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [fallback, bmi2, adx_bmi2];
        generic: $generic:meta;
        fallback_imports: [$($fallback_import:ident),*];
        test_backends: [$($test_alias:ident => $test_backend:ident),* $(,)?];
    ) => {
        #[cfg(any(
            $generic,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                any(
                    all(feature = "std", any(test, not(all(
                        target_feature = "adx",
                        target_feature = "bmi2"
                    )))),
                    all(not(feature = "std"), not(target_feature = "bmi2"))
                )
            )
        ))]
        mod fallback;
        #[cfg(any(
            $generic,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                any(
                    all(feature = "std", any(test, not(all(
                        target_feature = "adx",
                        target_feature = "bmi2"
                    )))),
                    all(not(feature = "std"), not(target_feature = "bmi2"))
                )
            )
        ))]
        use super::{$($fallback_import),*};
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                feature = "std",
                all(
                    not(feature = "std"),
                    target_feature = "adx",
                    target_feature = "bmi2"
                )
            )
        ) {
            mod x86_64_adx;
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                all(feature = "std", any(test, not(all(
                    target_feature = "adx",
                    target_feature = "bmi2"
                )))),
                all(
                    not(feature = "std"),
                    target_feature = "bmi2",
                    not(target_feature = "adx")
                )
            )
        ) {
            mod x86_64_bmi2;
        });
        select_arch_kernel!(@when all(
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
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx",
            target_feature = "bmi2"
        ) {
            mod selected {
                use super::$kernel;
                use super::x86_64_adx::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(feature = "std"),
            target_feature = "bmi2",
            not(target_feature = "adx")
        ) {
            mod selected {
                use super::$kernel;
                use super::x86_64_bmi2::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        select_arch_kernel!(@when any(
            $generic,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(feature = "std"),
                not(target_feature = "bmi2")
            )
        ) {
            mod selected {
                use super::$kernel;
                use super::fallback::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        $(
            #[cfg(all(
                test,
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64"
            ))]
            pub use self::$test_backend::$function as $test_alias;
        )*
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [sse2, avx2];
        generic: $generic:meta;
        fallback_imports: [$($fallback_import:ident),*];
        test_backends: [$($test_alias:ident => $test_backend:ident),* $(,)?];
    ) => {
        // The SSE2 module is the mandatory x86-64 baseline, so it covers all
        // real x86-64 builds that do not compile in AVX2; the pure Rust
        // fallback remains for miri and non-x86-64 targets. A compile-time
        // AVX2 build never references this module, so it is cfg'd out rather
        // than left dead.
        #[cfg($generic)]
        mod fallback;
        #[cfg($generic)]
        use super::{$($fallback_import),*};
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "avx2")
        ) {
            mod x86_64;
        });
        // AVX2 is usable directly at compile time when the target is built
        // with `avx2`, and as a runtime tier on std builds.
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(feature = "std", target_feature = "avx2")
        ) {
            mod x86_64_avx2;
        });
        select_arch_kernel!(@when all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "avx2")
        ) {
            mod runtime_dispatch;
            use super::{X86SimdTier, selected_x86_simd_tier};
            mod selected {
                use super::$kernel;

                #[inline]
                pub fn kernel() -> $kernel {
                    super::runtime_dispatch::selected_kernel()
                }
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "avx2"
        ) {
            mod selected {
                use super::$kernel;
                use super::x86_64_avx2::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(feature = "std"),
            not(target_feature = "avx2")
        ) {
            mod selected {
                use super::$kernel;
                use super::x86_64::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        select_arch_kernel!(@when $generic {
            mod selected {
                use super::$kernel;
                use super::fallback::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        $(
            #[cfg(all(
                test,
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = "avx2")
            ))]
            pub use self::$test_backend::$function as $test_alias;
        )*
    };
    (
        @single
        $function:ident,
        $kernel:ident,
        $backend:ident,
        $feature:literal,
        $generic:meta,
        []
    ) => {
        select_x86_kernel!(
            @single_items
            $function, $kernel, $backend, $feature, $generic
        );
    };
    (
        @single
        $function:ident,
        $kernel:ident,
        $backend:ident,
        $feature:literal,
        $generic:meta,
        [$($fallback_import:ident),+]
    ) => {
        #[cfg(any(
            $generic,
            all(
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = $feature)
            )
        ))]
        use super::{$($fallback_import),+};
        select_x86_kernel!(
            @single_items
            $function, $kernel, $backend, $feature, $generic
        );
    };
    (
        @single_items
        $function:ident,
        $kernel:ident,
        $backend:ident,
        $feature:literal,
        $generic:meta
    ) => {
        select_arch_kernel!(@when any(
            $generic,
            all(
                feature = "std",
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = $feature)
            ),
            all(
                not(feature = "std"),
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_feature = $feature)
            )
        ) {
            mod fallback;
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(feature = "std", target_feature = $feature)
        ) {
            mod $backend;
        });
        select_arch_kernel!(@when all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = $feature)
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
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = $feature
        ) {
            mod selected {
                use super::$kernel;
                use super::$backend::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
        select_arch_kernel!(@when any(
            $generic,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(feature = "std"),
                not(target_feature = $feature)
            )
        ) {
            mod selected {
                use super::$kernel;
                use super::fallback::$function;

                #[inline]
                pub const fn kernel() -> $kernel {
                    $function
                }
            }
        });
    };
}
