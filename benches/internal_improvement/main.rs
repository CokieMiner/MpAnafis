//! Multiplication and squaring experiments through the internal tuning interface.
//! `compare` measures production dispatch against GMP and FLINT; `crossovers`
//! measures selected tiers. Fixtures, reference checks, and numeric buffer
//! allocation precede timing. Planning remains timed in production rows.
//!
//! Case labels record operand sizes and worker budgets. CPU affinity and A/B/B/A
//! execution are configured externally; see `README.md` for the protocol.

#[cfg(all(
    target_arch = "x86_64",
    target_pointer_width = "64",
    target_os = "linux"
))]
mod compare;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod crossovers;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod shared;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn main() {
    divan::main();
}

#[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
fn main() {
    eprintln!("internal_improvement requires 64-bit x86 for its GMP reference implementation");
    std::process::exit(1);
}
