//! Build-time Eratosthenes sieve for the native-primality lookup.

use std::{fs, path::Path};

/// Writes one composite bit per odd integer in the target's lookup domain.
///
/// Sixteen-bit targets use 4 KiB for their entire native range; wider targets
/// use 64 KiB for integers below 2^20. Computation uses the build host's word
/// size, while the output contains only bytes and is independent of endianness.
///
/// # Panics
///
/// Panics if the output cannot be written. All arithmetic and bitmap bounds
/// are checked against the fixed maximum of 2^20 integers.
pub fn write_bitmap(destination: &Path, pointer_width: &str) {
    let bytes = if pointer_width == "16" {
        1_usize << 12
    } else {
        1_usize << 16
    };
    let maximum = bytes
        .checked_mul(16)
        .and_then(|limit| limit.checked_sub(1))
        .expect("the build host can represent the 20-bit sieve domain");
    let mut bitmap = vec![0_u8; bytes];
    *bitmap.first_mut().expect("the bitmap is nonempty") = 1;
    let mut prime = 3_usize;
    while prime <= maximum.checked_div(prime).expect("prime is at least three") {
        if bitmap.get(prime >> 4).expect("prime is within the bitmap") & (1 << ((prime >> 1) & 7))
            == 0
        {
            let square = prime.checked_mul(prime).expect("prime^2 <= maximum");
            for multiple in (square..=maximum).step_by(prime << 1) {
                *bitmap
                    .get_mut(multiple >> 4)
                    .expect("multiple <= maximum bounds the byte index") |=
                    1 << ((multiple >> 1) & 7);
            }
        }
        prime = prime
            .checked_add(2)
            .expect("prime <= 1023 before increment");
    }
    if fs::read(destination).ok().as_deref() != Some(&bitmap) {
        fs::write(destination, bitmap).expect("failed to write prime_bitmap.bin");
    }
}
