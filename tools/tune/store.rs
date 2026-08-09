//! Persistent measurement store: candidate score cache and machine report.
//!
//! Everything the tuner measures is written under `target/tune/` so a run can
//! be resumed, two machines can be diffed, and a decision can be traced back
//! to the cells that produced it. The format is deliberately tiny hand-rolled
//! JSON: the store only carries numbers and short strings, and adding a
//! serialization dependency for that would outsize the code.

use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::tuning_profile::TuningProfile;

/// Machine-stable cache file, shared across runs on one host.
pub const SCORE_CACHE_NAME: &str = "score-cache.json";

const SCORE_SCHEMA_VERSION: &str = "mp-tune-score-v1";

/// FNV-1a 64-bit hash over the rendered profile source.
///
/// This is the identity of a candidate profile: two runs that render the same
/// source measure the same program, so their cell timings can be shared.
#[must_use]
pub fn profile_hash(profile: &TuningProfile) -> u64 {
    fnv1a(profile.render("// hash seed").as_bytes())
}

/// Identity of the code, toolchain, flags, and scoring schema behind a score.
///
/// Raw timings are meaningful only for the exact program that produced them.
/// Hashing the relevant source tree prevents a resumed run from silently
/// comparing cached measurements from an older checkout with fresh candidates.
/// If the workspace cannot be read, a time-based identity safely limits reuse
/// to the current process instead of trusting an incomplete fingerprint.
#[must_use]
pub fn measurement_context_hash() -> u64 {
    workspace_context_hash().unwrap_or_else(|| {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        fnv1a(&nonce.to_le_bytes())
    })
}

fn workspace_context_hash() -> Option<u64> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for path in ["src", "build_support", "tools/tune"] {
        collect_files(&root.join(path), &mut files)?;
    }
    for path in ["build.rs", "Cargo.toml", "Cargo.lock"] {
        files.push(root.join(path));
    }
    files.sort_unstable();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(SCORE_SCHEMA_VERSION.as_bytes());
    for variable in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC"] {
        bytes.extend_from_slice(variable.as_bytes());
        if let Some(value) = env::var_os(variable) {
            bytes.extend_from_slice(value.to_string_lossy().as_bytes());
        }
    }
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = Command::new(rustc).arg("-vV").output().ok()?;
    bytes.extend_from_slice(&version.stdout);
    for path in files {
        if path.ends_with("src/int/tuned_thresholds.rs") {
            continue;
        }
        let relative = path.strip_prefix(root).ok()?;
        bytes.extend_from_slice(relative.to_string_lossy().as_bytes());
        bytes.extend_from_slice(&fs::read(path).ok()?);
    }
    Some(fnv1a(&bytes))
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) -> Option<()> {
    for entry in fs::read_dir(directory).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
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

/// Per-cell timing cache keyed by candidate profile hash.
#[derive(Debug, Default)]
pub struct ScoreStore {
    entries: HashMap<u64, Vec<u128>>,
    path: PathBuf,
}

impl ScoreStore {
    /// Load the cache at `path`, tolerating a missing or malformed file.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let entries = fs::read_to_string(path)
            .ok()
            .and_then(|source| parse_scores(&source))
            .unwrap_or_default();
        Self {
            entries,
            path: path.to_owned(),
        }
    }

    /// Cached cell timings for a candidate profile, if measured before.
    #[must_use]
    pub fn get(&self, hash: u64) -> Option<&[u128]> {
        self.entries.get(&hash).map(Vec::as_slice)
    }

    /// Record and immediately checkpoint cell timings for a candidate profile.
    ///
    /// A compiled SSA or division candidate can take minutes. Persisting after
    /// each completed worker lets an interrupted run reuse every compatible
    /// result instead of losing the whole coordinate pass.
    pub fn insert(&mut self, hash: u64, cells: Vec<u128>) {
        drop(self.entries.insert(hash, cells));
        self.save();
    }

    /// Write the cache back to disk, best-effort.
    pub fn save(&self) {
        if let Some(parent) = self.path.parent() {
            drop(fs::create_dir_all(parent));
        }
        let mut file = match File::create(&self.path) {
            Ok(file) => file,
            Err(error) => {
                println!(
                    "Could not write the score cache {}: {error}",
                    self.path.display()
                );
                return;
            }
        };
        drop(file.write_all(encode_scores(&self.entries).as_bytes()));
    }
}

/// The per-machine directory all tuning artifacts live in.
#[must_use]
pub fn machine_dir(cpu: &str) -> PathBuf {
    let mut sanitized = String::new();
    let mut previous_underscore = false;
    for ch in cpu.chars() {
        let keep = ch.is_ascii_alphanumeric() || ch == '-';
        if keep {
            sanitized.push(ch);
            previous_underscore = false;
        } else if !previous_underscore {
            sanitized.push('_');
            previous_underscore = true;
        }
    }
    Path::new("target").join("tune").join(sanitized)
}

/// Write the human-readable JSON report beside the tuned profile.
///
/// `decisions` carries one (knob, outcome) line per tuning decision so the
/// report documents not only the final values but why each was chosen.
pub fn write_report(
    path: &Path,
    cpu: &str,
    date: &str,
    profile: &TuningProfile,
    decisions: &[(String, String)],
) {
    if let Some(parent) = path.parent() {
        drop(fs::create_dir_all(parent));
    }
    let mut file = match File::create(path) {
        Ok(file) => file,
        Err(error) => {
            println!(
                "Could not write the tuning report {}: {error}",
                path.display()
            );
            return;
        }
    };
    let rendered = profile.render("// Final tuned profile");
    let decisions_json = decisions
        .iter()
        .map(|(knob, outcome)| {
            format!(
                "{{\"knob\":{},\"outcome\":{}}}",
                json_string(knob),
                json_string(outcome)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    drop(
        file.write_all(
            format!(
                "{{\"cpu\":{},\"date\":{},\"decisions\":[{}],\"profile_source\":{}}}\n",
                json_string(cpu),
                json_string(date),
                decisions_json,
                json_string(&rendered),
            )
            .as_bytes(),
        ),
    );
}

fn json_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// FNV-1a 64-bit over arbitrary bytes.
#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn encode_scores(entries: &HashMap<u64, Vec<u128>>) -> String {
    let items = entries
        .iter()
        .map(|(hash, cells)| {
            format!(
                "{{\"h\":{hash},\"c\":[{}]}}",
                cells
                    .iter()
                    .map(u128::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"scores\":[{items}]}}\n")
}

fn parse_scores(source: &str) -> Option<HashMap<u64, Vec<u128>>> {
    let body = source
        .trim()
        .strip_prefix("{\"scores\":[")?
        .strip_suffix("]}")?;
    if body.is_empty() {
        return Some(HashMap::new());
    }
    let mut entries = HashMap::new();
    for item in body
        .split("},{")
        .map(|chunk| chunk.trim_start_matches('{').trim_end_matches('}'))
    {
        let (hash_part, cells_part) = item.split_once(",\"c\":[")?;
        let hash = hash_part.strip_prefix("\"h\":")?.parse::<u64>().ok()?;
        let cells = cells_part
            .trim_end_matches(']')
            .split(',')
            .map(str::parse::<u128>)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        drop(entries.insert(hash, cells));
    }
    Some(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_cache_round_trips() {
        let mut entries = HashMap::new();
        drop(entries.insert(1_u64, vec![10, 20, 30]));
        drop(entries.insert(u64::MAX, vec![378_125, 2_174_824_933]));
        let encoded = encode_scores(&entries);
        let parsed = parse_scores(&encoded).expect("round trip must parse");
        assert_eq!(parsed, entries);
    }

    #[test]
    fn fnv1a_matches_the_reference_vector() {
        assert_eq!(fnv1a(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xAF63_DC4C_8601_EC8C);
    }

    #[test]
    fn profile_hash_is_deterministic_and_distinguishes_profiles() {
        let first = TuningProfile::portable();
        let second = TuningProfile::portable();
        assert_eq!(profile_hash(&first), profile_hash(&second));
        let mut different = first;
        different.karatsuba = 99;
        assert_ne!(profile_hash(&first), profile_hash(&different));
    }

    #[test]
    fn machine_dir_sanitizes_the_cpu_name() {
        let dir = machine_dir("AMD Ryzen 9 7950X (16 cores)");
        let rendered = dir.to_string_lossy();
        assert!(
            rendered.contains("AMD_Ryzen_9_7950X_16_cores_"),
            "{rendered}"
        );
        assert!(
            !rendered.contains('(') && !rendered.contains(' '),
            "{rendered}"
        );
    }
}
