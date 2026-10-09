//! Separate operand dimensions for production division measurements.

use core::mem::size_of;

use super::PRODUCTION_DIVISOR_LIMBS;

/// Additional boundary probes; the standard geometry is always retained.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DivisionGrid {
    pub divisor_widths: Vec<usize>,
    pub quotient_widths: Vec<usize>,
    pub scalar_quotients: Vec<usize>,
    pub block_ratios: Vec<usize>,
}

impl DivisionGrid {
    /// Decode a frozen worker geometry before allocation. The first CSV is the
    /// divisor dimension; named suffixes preserve the other independent units.
    pub fn parse(specification: &str) -> Result<Self, String> {
        let parse = |encoded: &str| -> Result<Vec<usize>, String> {
            if encoded.is_empty() {
                return Ok(Vec::new());
            }
            encoded
                .split(',')
                .map(|field| field.parse::<usize>().map_err(|error| error.to_string()))
                .collect()
        };
        let mut fields = specification.split(';');
        let mut grid = Self {
            divisor_widths: parse(fields.next().unwrap_or_default())?,
            ..Self::default()
        };
        let mut seen = Vec::new();
        for field in fields {
            let (name, encoded) = field
                .split_once('=')
                .ok_or("missing division dimension name")?;
            if seen.contains(&name) {
                return Err("duplicate division dimension".to_owned());
            }
            seen.push(name);
            let destination = match name {
                "q" => &mut grid.quotient_widths,
                "scalar" => &mut grid.scalar_quotients,
                "ratios" => &mut grid.block_ratios,
                _ => return Err("unknown division dimension".to_owned()),
            };
            *destination = parse(encoded)?;
        }
        if grid
            .divisor_widths
            .iter()
            .chain(&grid.quotient_widths)
            .chain(&grid.scalar_quotients)
            .chain(&grid.block_ratios)
            .any(|&value| value == 0)
        {
            return Err("division dimensions must be positive".to_owned());
        }
        let divisor = PRODUCTION_DIVISOR_LIMBS
            .iter()
            .chain(&grid.divisor_widths)
            .copied()
            .max()
            .expect("standard divisor grid");
        let quotient = divisor
            .checked_mul(3)
            .map(|width| width.max(grid.quotient_widths.iter().copied().max().unwrap_or(129)))
            .ok_or("division quotient span overflows")?;
        let bytes = divisor
            .checked_add(quotient)
            .and_then(|width| width.checked_add(2))
            .and_then(|width| width.checked_mul(size_of::<usize>()))
            .ok_or("division fixture span overflows")?;
        if isize::try_from(bytes).is_err() {
            return Err("division fixture exceeds the addressable span".to_owned());
        }
        Ok(grid)
    }

    /// Encode all units explicitly; both binaries and cached scores use the
    /// same dimension string as their worker geometry identity.
    #[must_use]
    pub fn render(&self) -> String {
        let mut fields = Vec::new();
        for (name, values) in [
            ("", &self.divisor_widths),
            ("q=", &self.quotient_widths),
            ("scalar=", &self.scalar_quotients),
            ("ratios=", &self.block_ratios),
        ] {
            fields.push(format!(
                "{name}{}",
                values
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        fields.join(";")
    }
}
