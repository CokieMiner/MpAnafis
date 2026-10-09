//! Render and parse complete generated Rust profiles through one field registry.

use core::fmt::Write;

use super::{Parameter, TuningProfile};

impl TuningProfile {
    /// Parses a complete rendered profile and returns its typed values.
    ///
    /// # Errors
    ///
    /// Accepts single-line `usize` declarations, line comments, and generated
    /// 16-bit exclusion attributes. Rendering reconstructs those attributes
    /// from the registry. Missing, duplicate, unknown, or malformed constants
    /// are rejected, including obsolete profiles lacking a new control.
    pub fn from_source(source: &str) -> Result<Self, String> {
        let mut constants = parse_declarations(source)?;
        let mut profile = Self::portable();
        for parameter in Parameter::ALL {
            let value = take_constant(&mut constants, parameter.name)?;
            (parameter.set)(&mut profile, value);
        }
        if let Some((name, _)) = constants.first() {
            return Err(format!("unknown tuning constant {name}"));
        }
        Ok(profile)
    }

    /// Renders a complete Rust source profile with `header` preceding declarations.
    #[must_use]
    pub fn render(self, header: &str) -> String {
        let mut output = header.to_owned();
        for parameter in Parameter::ALL {
            output.push('\n');
            if parameter.wide_only {
                output.push_str("#[cfg(not(target_pointer_width = \"16\"))]\n");
            }
            write!(
                output,
                "pub const {}: usize = {};",
                parameter.name,
                format_constant((parameter.get)(self)),
            )
            .expect("writing to String is infallible");
        }
        output
    }
}

fn parse_declarations(source: &str) -> Result<Vec<(&str, usize)>, String> {
    let mut constants = Vec::new();
    for line in source.lines() {
        let declaration = line.split_once("//").map_or(line, |(code, _)| code).trim();
        if declaration.is_empty() || declaration == "#[cfg(not(target_pointer_width = \"16\"))]" {
            continue;
        }
        let constant = declaration
            .strip_prefix("pub const ")
            .or_else(|| declaration.strip_prefix("const "))
            .ok_or_else(|| format!("unsupported tuning profile line: {declaration}"))?;
        let (raw_name, definition) = constant
            .split_once(':')
            .ok_or_else(|| format!("missing constant type: {declaration}"))?;
        let name = raw_name.trim();
        if constants.iter().any(|(existing, _)| *existing == name) {
            return Err(format!("duplicate tuning constant {name}"));
        }
        let (kind, value) = definition
            .split_once('=')
            .ok_or_else(|| format!("missing value for constant {name}"))?;
        if kind.trim() != "usize" {
            return Err(format!("constant {name} must have type usize"));
        }
        let expression = value
            .trim()
            .strip_suffix(';')
            .ok_or_else(|| format!("constant {name} has no terminating semicolon"))?;
        constants.push((name, parse_value(expression, name)?));
    }
    Ok(constants)
}

fn take_constant(constants: &mut Vec<(&str, usize)>, name: &str) -> Result<usize, String> {
    let index = constants
        .iter()
        .position(|(key, _)| *key == name)
        .ok_or_else(|| format!("missing constant {name}"))?;
    Ok(constants.swap_remove(index).1)
}

fn parse_value(expression: &str, name: &str) -> Result<usize, String> {
    let compact: String = expression
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact == "usize::MAX-1" {
        return Ok(usize::MAX - 1);
    }
    let literal = expression.trim();
    if !literal.starts_with(|character: char| character.is_ascii_digit())
        || !literal
            .chars()
            .all(|character| character.is_ascii_digit() || character == '_')
    {
        return Err(format!(
            "constant {name} is not a usize literal: {expression}"
        ));
    }
    literal
        .replace('_', "")
        .parse::<usize>()
        .map_err(|_error| format!("constant {name} is not a usize literal: {expression}"))
}

fn format_constant(value: usize) -> String {
    if value == usize::MAX - 1 {
        return "usize::MAX - 1".to_owned();
    }
    let s = value.to_string();
    if s.len() <= 4 {
        return s;
    }
    let mut result = String::new();
    let char_count = s.chars().count();
    for (i, ch) in s.chars().enumerate() {
        let rev_pos = char_count.saturating_sub(i);
        if i > 0 && rev_pos.is_multiple_of(3) {
            result.push('_');
        }
        result.push(ch);
    }
    result
}
