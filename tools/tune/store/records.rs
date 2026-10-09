//! Persistent screening scores keyed by profile and measurement context.

use std::{
    collections::HashMap,
    fs::{File, create_dir_all, read_to_string, remove_file, rename},
    io::Write,
    path::{Path, PathBuf},
    process::id as process_id,
    time::{SystemTime, UNIX_EPOCH},
};

use super::TuningProfile;

/// Machine-stable cache file, shared across runs on one host.
pub const SCORE_CACHE_NAME: &str = "score-cache.json";

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
        let entries = read_to_string(path)
            .ok()
            .and_then(|source| Self::parse_scores(&source))
            .unwrap_or_default();
        Self {
            entries,
            path: path.to_owned(),
        }
    }

    /// Return compatible cached cell timings for a candidate profile.
    #[must_use]
    pub fn get(&self, hash: u64) -> Option<&[u128]> {
        self.entries.get(&hash).map(Vec::as_slice)
    }

    /// Record cell timings and checkpoint the screening cache.
    pub fn insert(&mut self, hash: u64, cells: Vec<u128>) {
        drop(self.entries.insert(hash, cells));
        self.save();
    }

    /// Write the cache back to disk atomically, best-effort.
    pub fn save(&self) {
        if let Some(error) = self
            .path
            .parent()
            .and_then(|parent| create_dir_all(parent).err())
        {
            eprintln!(
                "Could not create cache directory for {}: {error}",
                self.path.display()
            );
            return;
        }
        let stamp = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_nanos(),
            Err(error) => {
                eprintln!(
                    "Could not timestamp score cache {}: {error}",
                    self.path.display()
                );
                return;
            }
        };
        // Exclusive creation prevents concurrent sessions from truncating each
        // other's temporary cache. A timestamp collision only skips this save.
        let tmp_path = self
            .path
            .with_extension(format!("{}-{stamp}.tmp", process_id()));
        let mut file = match File::create_new(&tmp_path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!(
                    "Could not create temporary score cache {}: {error}",
                    tmp_path.display()
                );
                return;
            }
        };
        if let Err(error) = file.write_all(Self::encode_scores(&self.entries).as_bytes()) {
            eprintln!(
                "Could not write the score cache {}: {error}",
                tmp_path.display()
            );
            drop(file);
            drop(remove_file(&tmp_path));
            return;
        }
        if let Err(error) = file.sync_all() {
            eprintln!(
                "Could not sync the score cache {}: {error}",
                tmp_path.display()
            );
            drop(file);
            drop(remove_file(&tmp_path));
            return;
        }
        drop(file);
        if let Err(error) = rename(&tmp_path, &self.path) {
            eprintln!(
                "Could not replace the score cache atomically {}: {error}",
                self.path.display()
            );
            drop(remove_file(&tmp_path));
        }
    }

    /// FNV-1a 64-bit hash over the rendered profile source.
    ///
    /// This identifies the tuning values. The separate measurement-context hash
    /// also distinguishes source, toolchain, flags, and worker configuration.
    #[must_use]
    pub fn profile_hash(profile: &TuningProfile) -> u64 {
        Self::fnv1a(profile.render("// hash seed").as_bytes())
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

    /// Render complete screening vectors in the cache's compact JSON protocol.
    pub fn encode_scores(entries: &HashMap<u64, Vec<u128>>) -> String {
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

    /// Decode the cache protocol, rejecting malformed hashes or cell timings.
    pub fn parse_scores(source: &str) -> Option<HashMap<u64, Vec<u128>>> {
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
}
