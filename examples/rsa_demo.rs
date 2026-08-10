//! Educational textbook-RSA arithmetic demonstration.
//!
//! Demonstrates primality testing, large multiplication, modular inversion,
//! modular exponentiation, and signed extended GCD using `arbi-anafis`.
//!
//! **This example is not suitable for production cryptography.** It uses
//! deterministic primes, textbook RSA without padding, and APIs that do not
//! claim constant-time execution.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::many_single_char_names,
    clippy::panic,
    clippy::print_stdout,
    clippy::shadow_unrelated,
    clippy::string_slice,
    clippy::unreadable_literal,
    reason = "example binary: single-char RSA variable names, string display formatting, and operator use are acceptable"
)]

use core::time::Duration;
use std::time::Instant;

use arbi_anafis::{ArbiInt, ArbiUint};

fn main() {
    // Generate two 512-bit primes once and measure their generation time.
    let t_prime = Instant::now();
    let base = ArbiUint::from(1_u64).wrapping_shl(511);
    let p = (&base + ArbiUint::from(1_u64))
        .next_prime()
        .expect("prime not found");
    let q_seed = &base + ArbiUint::from(1_000_003_u64);
    let q = q_seed.next_prime().expect("prime not found");
    let prime_gen_time = t_prime.elapsed();

    assert_eq!(p.significant_bits(), 512, "p must be exactly 512 bits");
    assert_eq!(q.significant_bits(), 512, "q must be exactly 512 bits");
    assert_ne!(p, q, "p and q must be distinct");
    assert!(p.is_prime(), "p must be prime");
    assert!(q.is_prime(), "q must be prime");

    textbook_rsa_demo(&p, &q, prime_gen_time);
    number_theory_demo(&p, &q);
}

fn format_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1_000 {
        format!("{micros} µs")
    } else {
        format!("{:.1} ms", d.as_secs_f64() * 1_000.0)
    }
}

fn textbook_rsa_demo(p: &ArbiUint, q: &ArbiUint, prime_gen_time: Duration) {
    println!("=== Textbook RSA Demo ===\n");

    let t0 = Instant::now();
    let n = p * q;
    let one = ArbiUint::from(1_u64);
    let phi = (p - &one) * (q - &one);
    let e = ArbiUint::from(65537_u64);
    let d = e.invert(&phi).expect("e must be coprime to phi(n)");
    let derive_time = t0.elapsed();

    println!("p:          {} bits, prime", p.significant_bits());
    println!("q:          {} bits, prime", q.significant_bits());
    println!("n:          {} bits (modulus)", n.significant_bits());
    println!("d:          {} bits", d.significant_bits());
    println!(
        "prime gen:  {} (searching two 512-bit primes via next_prime)",
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

    let msg = ArbiUint::from(42_u64);

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

fn number_theory_demo(p: &ArbiUint, q: &ArbiUint) {
    println!("=== Arithmetic & Number Theory ===\n");

    // -- Extended GCD with the actual RSA primes (coprime by construction) --
    let pi = ArbiInt::from(p.clone());
    let qi = ArbiInt::from(q.clone());
    if let Some((g, x, y)) = pi.extended_gcd(&qi) {
        let lhs = &pi * &x + &qi * &y;
        assert_eq!(lhs, g, "Bezout identity: p*x + q*y = gcd(p,q)");
        println!("extended_gcd:      [OK] p*x + q*y = 1");
    }

    // -- Factorials --
    for n in [50_u32, 100, 200] {
        let fact = ArbiUint::factorial(n, arbi_anafis::Precision::Unlimited);
        let bits = fact.significant_bits();
        let dec = fact.to_string_radix(10);
        println!(
            "{n:>4}!  {bits:>5} bits  {}.{}e{}",
            &dec[..1],
            &dec[1..8],
            dec.len() - 1
        );
    }

    // -- Modular exponentiation with large exponent --
    let result = ArbiUint::from(2_u64)
        .pow_mod(
            &ArbiUint::from(1_u64).wrapping_shl(100),
            &ArbiUint::from(1_000_000_007_u64),
        )
        .expect("modular exponentiation failed");
    println!("\npow_mod:            [OK] 2^(2^100) mod 1e9+7 = {result}");

    // -- Primality testing --
    for &prime in &[2_u64, 3, 5, 7, 11, 97, 7919, 104_729, 982_451_653] {
        assert!(ArbiUint::from(prime).is_prime(), "{prime} should be prime");
    }
    for &comp in &[4_u64, 6, 8, 9, 10, 100, 1_001, 104_730] {
        assert!(
            !ArbiUint::from(comp).is_prime(),
            "{comp} should be composite"
        );
    }
    println!("primality:          [OK] all 9 primes, 8 composites");

    // -- GCD / LCM --
    let a = ArbiUint::from(123_456_789_u64);
    let b = ArbiUint::from(987_654_321_u64);
    let g = a.gcd(&b);
    let l = a.lcm(&b).expect("lcm failed");
    assert_eq!(
        g.clone() * l.clone(),
        a.clone() * b.clone(),
        "gcd * lcm must equal a * b",
    );
    println!("gcd/lcm:            [OK] gcd({a}, {b}) = {g}, lcm = {l}");
}
