//! Selector-only x86 kernel policies used by `select_arch_kernel!`.

macro_rules! select_x86_kernel {
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [sse2, avx2, avx512, small];
        generic: $generic:meta;
        fallback_imports: [$($fallback_import:ident),*];
    ) => {
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                feature = "std",
                target_feature = "avx512f"
            )
        ) {
            mod x86_64_avx512;
            #[cfg(feature = "std")]
            pub use x86_64_avx512::$function as avx512_kernel;
        });
        select_x86_kernel! {
            @simd_pair
            function: $function;
            kernel: $kernel;
            generic: $generic;
            native_avx512: target_feature = "avx512f";
            small_kernel: true;
            fallback_imports: [$($fallback_import),*];
        }
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [fallback, adx];
        generic: $generic:meta;
        fallback_imports: $fallback_imports:tt;
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
    ) => {
        #[cfg(any(
            $generic,
            all(
                not(miri),
                target_arch = "x86_64",
                target_pointer_width = "64",
                any(
                    all(feature = "std", not(all(
                        target_feature = "adx",
                        target_feature = "bmi2"
                    ))),
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
                    all(feature = "std", not(all(
                        target_feature = "adx",
                        target_feature = "bmi2"
                    ))),
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
                all(feature = "std", not(all(
                    target_feature = "adx",
                    target_feature = "bmi2"
                ))),
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
            use runtime_dispatch::selected_kernel;
            pub use self::{
                fallback::$function as fallback_kernel,
                x86_64_adx::$function as adx_kernel,
                x86_64_bmi2::$function as bmi2_kernel,
            };
            mod selected {
                use super::$kernel;

                #[inline]
                pub fn kernel() -> $kernel {
                    super::selected_kernel()
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
            use x86_64_adx::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
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
            use x86_64_bmi2::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
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
            use fallback::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
                }
            }
        });
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [sse2, avx2];
        generic: $generic:meta;
        fallback_imports: [$($fallback_import:ident),*];
    ) => {
        select_x86_kernel! {
            @simd_pair
            function: $function;
            kernel: $kernel;
            generic: $generic;
            native_avx512: any();
            small_kernel: false;
            fallback_imports: [$($fallback_import),*];
        }
    };
    (
        function: $function:ident;
        kernel: $kernel:ident;
        policy: [sse2, avx2, avx512];
        generic: $generic:meta;
        fallback_imports: [$($fallback_import:ident),*];
    ) => {
        // `std` builds retain every safe tier for runtime and length-sensitive
        // dispatch. `no_std` builds compile only the tier guaranteed by their
        // target features.
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                feature = "std",
                target_feature = "avx512f"
            )
        ) {
            mod x86_64_avx512;
            #[cfg(feature = "std")]
            pub use x86_64_avx512::$function as avx512_kernel;
        });
        select_x86_kernel! {
            @simd_pair
            function: $function;
            kernel: $kernel;
            generic: $generic;
            native_avx512: target_feature = "avx512f";
            small_kernel: false;
            fallback_imports: [$($fallback_import),*];
        }
    };
    (
        @simd_pair
        function: $function:ident;
        kernel: $kernel:ident;
        generic: $generic:meta;
        native_avx512: $native_avx512:meta;
        small_kernel: $small_kernel:tt;
        fallback_imports: [$($fallback_import:ident),*];
    ) => {
        // The SSE2 module is the mandatory x86-64 baseline. `std` retains it
        // for length-sensitive dispatch even when a wider target feature is
        // guaranteed; the pure Rust fallback remains for miri and non-x86-64
        // targets.
        #[cfg($generic)]
        mod fallback;
        $(
            #[cfg($generic)]
            use super::$fallback_import;
        )*
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                feature = "std",
                not(target_feature = "avx2")
            )
        ) {
            mod x86_64;
        });
        // AVX2 is usable directly at compile time when the target is built
        // with `avx2`, and as a runtime tier on std builds.
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                feature = "std",
                all(target_feature = "avx2", not($native_avx512))
            )
        ) {
            mod x86_64_avx2;
        });
        select_arch_kernel!(@when all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ) {
            mod runtime_dispatch;
            use super::{X86SimdTier, selected_x86_simd_tier};
            pub use self::{
                x86_64::$function as sse2_kernel,
                x86_64_avx2::$function as avx2_kernel,
            };
            use runtime_dispatch::selected_kernel;
            select_x86_kernel!(@small_function $small_kernel, $function);
            mod selected {
                use super::$kernel;

                select_x86_kernel!(@small_kernel $small_kernel, $function, $kernel);

                #[inline]
                pub fn kernel() -> $kernel {
                    super::selected_kernel()
                }
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            $native_avx512,
            not(feature = "std")
        ) {
            use x86_64_avx512::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
                }
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "avx2",
            not($native_avx512),
            not(feature = "std")
        ) {
            use x86_64_avx2::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
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
            use x86_64::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
                }
            }
        });
        select_arch_kernel!(@when $generic {
            use fallback::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
                }
            }
        });
    };
    (@small_function true, $function:ident) => {
        use x86_64::$function as small_function;
    };
    (@small_function false, $function:ident) => {};
    (@small_kernel true, $function:ident, $kernel:ident) => {
        use super::small_function;

        #[inline]
        pub const fn small_kernel() -> $kernel {
            small_function
        }
    };
    (@small_kernel false, $function:ident, $kernel:ident) => {};
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
            use runtime_dispatch::selected_kernel;
            mod selected {
                use super::$kernel;

                #[inline]
                pub fn kernel() -> $kernel {
                    super::selected_kernel()
                }
            }
        });
        select_arch_kernel!(@when all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = $feature
        ) {
            use $backend::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
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
            use fallback::$function as backend_function;
            mod selected {
                use super::{$kernel, backend_function};

                #[inline]
                pub const fn kernel() -> $kernel {
                    backend_function
                }
            }
        });
    };
}
