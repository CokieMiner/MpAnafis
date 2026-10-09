//! Complete run reports and machine-specific artifact destinations.

use core::fmt::Write as _;
use std::{
    fs::{File, create_dir_all, remove_file, rename},
    io::{Result as IoResult, Write},
    path::{Path, PathBuf},
};

use super::{MeasurementContext, ScoreStore, TuningProfile};

impl ScoreStore {
    /// Write and synchronize the complete report before profile publication.
    ///
    /// # Errors
    /// Returns directory, file, synchronization, or rename errors. A failed
    /// report cannot establish that a run preserved its validation evidence.
    pub fn write_report(
        path: &Path,
        cpu: &str,
        date: &str,
        profile: &TuningProfile,
        decisions: &[(String, String)],
    ) -> IoResult<()> {
        let rendered = profile.render("// Final tuned profile");
        let context = MeasurementContext::current().key();
        let decisions_json = decisions
            .iter()
            .map(|(knob, outcome)| {
                format!(
                    "{{\"knob\":{},\"outcome\":{}}}",
                    Self::json_string(knob),
                    Self::json_string(outcome)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            "{{\"cpu\":{},\"date\":{},\"measurement_context\":{},\"decisions\":[{}],\"profile_source\":{}}}\n",
            Self::json_string(cpu),
            Self::json_string(date),
            Self::json_string(&context),
            decisions_json,
            Self::json_string(&rendered),
        );
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        // Each session gives its report an exclusive timestamped directory.
        let temporary = path.with_extension("tmp");
        let mut file = File::create_new(&temporary)?;
        let result = (|| {
            file.write_all(source.as_bytes())?;
            file.sync_all()?;
            drop(file);
            rename(&temporary, path)
        })();
        if result.is_err() {
            drop(remove_file(&temporary));
        }
        result
    }

    /// Per-machine directory for all tuning artifacts.
    #[must_use]
    pub fn machine_dir(cpu: &str) -> PathBuf {
        let mut sanitized = String::new();
        let mut previous_underscore = false;
        for character in cpu.chars() {
            let keep = character.is_ascii_alphanumeric() || character == '-';
            if keep {
                sanitized.push(character);
                previous_underscore = false;
            } else if !previous_underscore {
                sanitized.push('_');
                previous_underscore = true;
            }
        }
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("tune")
            .join(sanitized)
    }

    /// Escape an owned JSON string, including control characters.
    pub fn json_string(value: &str) -> String {
        let mut escaped = String::with_capacity(
            value
                .len()
                .checked_add(2)
                .expect("quoted string capacity fits"),
        );
        escaped.push('"');
        for character in value.chars() {
            match character {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\u{08}' => escaped.push_str("\\b"),
                '\u{0c}' => escaped.push_str("\\f"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                control if control <= '\u{1f}' => {
                    write!(escaped, "\\u{:04x}", u32::from(control))
                        .expect("writing into a String cannot fail");
                }
                ordinary => escaped.push(ordinary),
            }
        }
        escaped.push('"');
        escaped
    }
}
