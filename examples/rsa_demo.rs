//! Educational textbook-RSA arithmetic demonstration.
//!
//! Demonstrates primality testing, large multiplication, modular inversion,
//! modular exponentiation, and signed extended GCD using `MpAnafis`.
//!
//! **This example is not suitable for production cryptography.** It uses
//! deterministic prime searches, textbook RSA without padding, and APIs that do not
//! claim constant-time execution.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::many_single_char_names,
    clippy::print_stdout,
    clippy::string_slice,
    reason = "example binary: single-char RSA variable names, string display formatting, and operator arithmetic are acceptable"
)]

use core::time::Duration;
use std::time::Instant;

use mp_anafis::{MpInt, MpUint, Precision};

fn main() {
    // Search deterministically for two 512-bit probable primes.
    let t_prime = Instant::now();
    let base = MpUint::from(1_u64).wrapping_shl(511);
    let p = (&base + MpUint::from(1_u64))
        .next_prime()
        .expect("prime not found");
    let q_seed = &base + MpUint::from(1_000_003_u64);
    let q = q_seed.next_prime().expect("prime not found");
    let prime_gen_time = t_prime.elapsed();

    assert_eq!(p.significant_bits(), 512, "p must be exactly 512 bits");
    assert_eq!(q.significant_bits(), 512, "q must be exactly 512 bits");
    assert_ne!(p, q, "p and q must be distinct");
    assert!(p.is_prime(), "p must pass probable-prime screening");
    assert!(q.is_prime(), "q must pass probable-prime screening");

    textbook_rsa_demo(&p, &q, prime_gen_time);
    number_theory_demo(&p, &q);
}

fn textbook_rsa_demo(p: &MpUint, q: &MpUint, prime_gen_time: Duration) {
    println!("=== Textbook RSA Demo ===\n");

    let t0 = Instant::now();
    let n = p * q;
    let one = MpUint::from(1_u64);
    let phi = (p - &one) * (q - &one);
    let e = MpUint::from(65537_u64);
    let d = e.invert(&phi).expect("e must be coprime to phi(n)");
    let derive_time = t0.elapsed();

    println!("p:          {} bits, probable prime", p.significant_bits());
    println!("q:          {} bits, probable prime", q.significant_bits());
    println!("n:          {} bits (modulus)", n.significant_bits());
    println!("d:          {} bits", d.significant_bits());
    println!(
        "prime gen:  {} (searching two 512-bit probable primes via next_prime)",
        format_duration(prime_gen_time)
    );
    println!(
        "key derive: {} (n = p*q, phi, d = e^-1 mod phi)",
        format_duration(derive_time)
    );
    println!(
        "keygen all: {} (prime gen + derive)",
        format_duration(prime_gen_time + derive_time)
    );

    let msg = MpUint::from(42_u64);

    let t1 = Instant::now();
    let c = msg.pow_mod(&e, &n).expect("encryption failed");
    let enc_time = t1.elapsed();

    let t2 = Instant::now();
    let m = c.pow_mod(&d, &n).expect("decryption failed");
    let dec_time = t2.elapsed();

    assert_eq!(msg, m, "RSA round-trip must recover the plaintext");
    println!("encrypt: {}", format_duration(enc_time));
    println!("decrypt: {}", format_duration(dec_time));
    println!("result:  [OK] round-trip verified\n");
}

fn number_theory_demo(p: &MpUint, q: &MpUint) {
    println!("=== Arithmetic & Number Theory ===\n");

    // Verify the signed Bezout identity p*x + q*y = gcd(p, q).
    let pi = MpInt::from(p.clone());
    let qi = MpInt::from(q.clone());
    let (bezout_gcd, x, y) = pi
        .extended_gcd(&qi)
        .expect("unlimited inputs admit coefficients");
    let lhs = &pi * &x + &qi * &y;
    assert_eq!(lhs, bezout_gcd, "Bezout identity: p*x + q*y = gcd(p,q)");
    assert_eq!(
        bezout_gcd,
        MpInt::from(1_u8),
        "the two selected candidates are coprime"
    );
    println!("extended_gcd:      [OK] p*x + q*y = 1");

    // Display factorial magnitudes in decimal scientific notation.
    for n in [50_u32, 100, 200] {
        let fact = MpUint::factorial(n, Precision::Unlimited);
        let bits = fact.significant_bits();
        let dec = fact.to_string_radix(10);
        println!(
            "{n:>4}!  {bits:>5} bits  {}.{}e{}",
            &dec[..1],
            &dec[1..8],
            dec.len() - 1
        );
    }

    // Evaluate 2^(2^100) modulo 10^9 + 7.
    let result = MpUint::from(2_u64)
        .pow_mod(
            &MpUint::from(1_u64).wrapping_shl(100),
            &MpUint::from(1_000_000_007_u64),
        )
        .expect("modular exponentiation failed");
    println!("\npow_mod:            [OK] 2^(2^100) mod 1e9+7 = {result}");

    // Native-size primality decisions are exact through u64::MAX.
    for &prime in &[2_u64, 3, 5, 7, 11, 97, 7919, 104_729, 982_451_653] {
        assert!(MpUint::from(prime).is_prime(), "{prime} should be prime");
    }
    for &comp in &[4_u64, 6, 8, 9, 10, 100, 1_001, 104_730] {
        assert!(!MpUint::from(comp).is_prime(), "{comp} should be composite");
    }
    println!("primality:          [OK] all 9 primes, 8 composites");

    // Verify gcd(a, b)*lcm(a, b) = a*b for positive inputs.
    let a = MpUint::from(123_456_789_u64);
    let b = MpUint::from(987_654_321_u64);
    let g = a.gcd(&b);
    let l = a.lcm(&b).expect("lcm failed");
    assert_eq!(
        g.clone() * l.clone(),
        a.clone() * b.clone(),
        "gcd * lcm must equal a * b",
    );
    println!("gcd/lcm:            [OK] gcd({a}, {b}) = {g}, lcm = {l}");
}

fn format_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1_000 {
        format!("{micros} us")
    } else {
        format!("{:.1} ms", d.as_secs_f64() * 1_000.0)
    }
}
