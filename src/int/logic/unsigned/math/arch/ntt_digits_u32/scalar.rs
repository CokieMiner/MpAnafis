//! Scalar 16-bit NTT digit packing shared by portable and baseline providers.

#![allow(
    unsafe_code,
    reason = "The architecture facade supplies validated raw spans"
)]
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The shift result is masked to one exact 16-bit digit"
)]

/// Packs up to `dst_len` 16-bit digits from `len` 64-bit limbs.
///
/// # Safety
/// `limbs` is readable for `len` `u64`s and `dst` is writable for `dst_len`
/// `u32`s. A short destination is deliberately truncated.
#[cfg(any(
    test,
    all(
        not(all(target_arch = "aarch64", target_pointer_width = "64")),
        not(all(target_arch = "x86_64", target_feature = "avx2"))
    )
))]
pub unsafe fn limbs_to_digits_16_scalar(
    dst: *mut u32,
    limbs: *const u64,
    len: usize,
    dst_len: usize,
) -> usize {
    let mut count = 0_usize;
    for index in 0..len {
        // SAFETY: the caller guarantees the input span for `len` elements.
        let limb = unsafe { *limbs.add(index) };
        for shift in [0_u32, 16, 32, 48] {
            if count == dst_len {
                break;
            }
            // SAFETY: `count < dst_len` guarantees a writable output slot.
            unsafe {
                *dst.add(count) = (limb >> shift) as u32 & u32::from(u16::MAX);
            }
            count = count.wrapping_add(1);
        }
        if count == dst_len {
            break;
        }
    }

    while count != 0 {
        // SAFETY: `count` is a number of initialized output slots.
        let is_zero = unsafe { *dst.add(count.wrapping_sub(1)) == 0 };
        if !is_zero {
            break;
        }
        count = count.wrapping_sub(1);
    }
    count
}
