//! Hardware, toolchain, and source identity for reusable screening scores.

use core::num::NonZeroUsize;
use std::{
    env::{
        consts::{ARCH, FAMILY, OS},
        var, var_os, vars_os,
    },
    ffi::OsString,
    fs::{metadata, read, read_dir},
    path::{Path, PathBuf},
    process::Command,
    thread::available_parallelism,
};

use super::{Platform, ScoreStore};

const SCORE_SCHEMA_VERSION: &str = "mp-tune-score-v1";

/// Hardware and toolchain context that makes a timing reusable.
/// Owned strings preserve incomplete best-effort platform metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementContext {
    /// Stable CPU, affinity, and cache description.
    pub platform: String,
    /// Target architecture, system, and pointer width.
    pub target: String,
    /// Compiler version and identity.
    pub compiler: String,
    /// Build flags and executor environment.
    pub flags: String,
    /// Executor kind and available worker count.
    pub executor: String,
}

impl MeasurementContext {
    /// Build the context for the current tuner process.
    #[must_use]
    pub fn current() -> Self {
        let platform = Platform::platform_identity().key();
        let target = format!(
            "arch={};os={};family={};pointer_bits={}",
            ARCH,
            OS,
            FAMILY,
            usize::BITS,
        );
        let compiler = compiler_identity();
        let flags = ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RAYON_NUM_THREADS"]
            .iter()
            .filter_map(|name| var(name).ok().map(|value| format!("{name}={value}")))
            .collect::<Vec<_>>()
            .join(",");
        let workers = available_parallelism().map_or(1, NonZeroUsize::get);
        let executor = format!(
            "{};workers={workers}",
            if cfg!(feature = "rayon") {
                "rayon"
            } else {
                "sequential"
            },
        );
        Self {
            platform,
            target,
            compiler,
            flags,
            executor,
        }
    }

    /// Stable single-line rendering suitable for score keys and reports.
    #[must_use]
    pub fn key(&self) -> String {
        [
            ("platform", self.platform.as_str()),
            ("target", self.target.as_str()),
            ("compiler", self.compiler.as_str()),
            ("flags", self.flags.as_str()),
            ("executor", self.executor.as_str()),
        ]
        .into_iter()
        .map(|(name, value)| format!("{name}={}:{}", value.len(), compact_context(value)))
        .collect::<Vec<_>>()
        .join("|")
    }
}

impl ScoreStore {
    /// Identity of the exact source, toolchain, flags, and scoring schema.
    /// An unreadable workspace has no valid identity; construction and later
    /// context checks reject it before scoring or profile installation.
    #[must_use]
    pub fn measurement_context_hash() -> Option<u64> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        for path in ["src", "build_support", "tools/tune"] {
            collect_files(&root.join(path), &mut files)?;
        }
        for path in ["build.rs", "Cargo.toml", "Cargo.lock"] {
            files.push(root.join(path));
        }
        for path in [
            ".cargo/config.toml",
            ".cargo/config",
            "rust-toolchain.toml",
            "rust-toolchain",
        ] {
            let candidate = root.join(path);
            if candidate.try_exists().ok()? {
                files.push(candidate);
            }
        }
        files.sort_unstable();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SCORE_SCHEMA_VERSION.as_bytes());
        for variable in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC"] {
            bytes.extend_from_slice(variable.as_bytes());
            if let Some(value) = var_os(variable) {
                bytes.extend_from_slice(value.to_string_lossy().as_bytes());
            }
        }
        let mut cargo_env_vars: Vec<(String, OsString)> = vars_os()
            .filter_map(|(key, value)| {
                let name = key.to_string_lossy();
                (name.starts_with("CARGO_PROFILE_RELEASE_")
                    || name.starts_with("CARGO_TARGET_")
                    || name.starts_with("CARGO_BUILD_"))
                .then(|| (name.into_owned(), value))
            })
            .collect();
        cargo_env_vars.sort_by(|a, b| a.0.cmp(&b.0));
        for (key, value) in cargo_env_vars {
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(value.to_string_lossy().as_bytes());
        }
        bytes.extend_from_slice(MeasurementContext::current().key().as_bytes());
        for path in files {
            if path.ends_with("src/int/tuned_thresholds.rs") {
                continue;
            }
            let relative = path.strip_prefix(root).ok()?;
            bytes.extend_from_slice(relative.to_string_lossy().as_bytes());
            bytes.extend_from_slice(&read(path).ok()?);
        }
        Some(Self::fnv1a(&bytes))
    }
}

fn compiler_identity() -> String {
    let rustc = var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    Command::new(rustc)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |value| value.trim().to_owned())
}

fn compact_context(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\\' => "\\\\".to_owned(),
            '\n' => "\\n".to_owned(),
            '\r' => "\\r".to_owned(),
            '|' => "\\|".to_owned(),
            _ => character.to_string(),
        })
        .collect()
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) -> Option<()> {
    for entry in read_dir(directory).ok()? {
        let path = entry.ok()?.path();
        if metadata(&path).ok()?.is_dir() {
            collect_files(&path, files)?;
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "toml" | "lock")
        ) {
            files.push(path);
        }
    }
    Some(())
}
