//! Canonical values used for untimed comparison validation.

use mp_anafis::{MpInt, MpUint};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

/// Encodes values without engine-specific types or precision metadata.
pub trait Outcome {
    /// Returns a canonical comparison representation.
    fn encode(&self) -> String;
}

impl<T: Outcome + ?Sized> Outcome for &T {
    fn encode(&self) -> String {
        (*self).encode()
    }
}

impl Outcome for MpUint {
    fn encode(&self) -> String {
        self.to_string_radix(16)
    }
}

impl Outcome for MpInt {
    fn encode(&self) -> String {
        self.to_string_radix(16)
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
impl Outcome for Integer {
    fn encode(&self) -> String {
        self.to_string_radix(16)
    }
}

impl<T: Outcome> Outcome for Option<T> {
    fn encode(&self) -> String {
        self.as_ref().map_or_else(
            || "None".to_owned(),
            |value| format!("Some({})", value.encode()),
        )
    }
}

impl<A: Outcome, B: Outcome> Outcome for (A, B) {
    fn encode(&self) -> String {
        format!("({}, {})", self.0.encode(), self.1.encode())
    }
}

impl<A: Outcome, B: Outcome, C: Outcome> Outcome for (A, B, C) {
    fn encode(&self) -> String {
        format!(
            "({}, {}, {})",
            self.0.encode(),
            self.1.encode(),
            self.2.encode()
        )
    }
}

impl<A: Outcome, B: Outcome, C: Outcome, D: Outcome> Outcome for (A, B, C, D) {
    fn encode(&self) -> String {
        format!(
            "({}, {}, {}, {})",
            self.0.encode(),
            self.1.encode(),
            self.2.encode(),
            self.3.encode()
        )
    }
}

impl<A: Outcome, B: Outcome, C: Outcome, D: Outcome, E: Outcome> Outcome for (A, B, C, D, E) {
    fn encode(&self) -> String {
        format!(
            "({}, {}, {}, {}, {})",
            self.0.encode(),
            self.1.encode(),
            self.2.encode(),
            self.3.encode(),
            self.4.encode()
        )
    }
}

impl<T: Outcome, E: core::fmt::Debug> Outcome for Result<T, E> {
    fn encode(&self) -> String {
        match self {
            Ok(value) => format!("Ok({})", value.encode()),
            Err(error) => format!("Err({error:?})"),
        }
    }
}

impl<T: Outcome> Outcome for Vec<T> {
    fn encode(&self) -> String {
        self.iter()
            .map(Outcome::encode)
            .collect::<Vec<_>>()
            .join(",")
    }
}

macro_rules! scalar_outcomes {
    ($($kind:ty),+ $(,)?) => { $(
        impl Outcome for $kind {
            fn encode(&self) -> String { format!("{self:?}") }
        }
    )+ };
}

scalar_outcomes!(
    bool,
    u8,
    u32,
    u64,
    u128,
    usize,
    i8,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    String,
    core::cmp::Ordering
);
