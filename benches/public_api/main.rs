//! Public integer operations and operand scenarios on identical paired inputs.
//! Cases are grouped by domain, category, function, scenario, and engine.
//! Rug/GMP comparisons require 64-bit x86; FLINT references have explicit gates.
//! The local README describes the source layout. `tools/bench.py` controls
//! checked plans, pinned execution, reports, and retained measurement records.

mod int;

fn main() {
    divan::main();
}
