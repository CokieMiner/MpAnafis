//! Host metadata and measurement hygiene.

use core::cmp::Ordering;
use std::{fs, fs::File, io::BufRead, path::Path, process::Command};

/// Identity of the one logical CPU available to the tuner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityIdentity {
    description: String,
}

impl AffinityIdentity {
    /// Stable description suitable for reports and cache separation.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Stable, owned description of one cache level in the selected CPU's cache
/// hierarchy. Linux may omit individual fields; an absent value means that
/// the host did not expose or did not provide a parseable value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheIdentity {
    level: Option<u32>,
    kind: Option<String>,
    size_bytes: Option<u64>,
    coherency_line_bytes: Option<u32>,
    shared_cpu_list: Option<String>,
}

/// Best-effort host identity used to separate tuning results by hardware.
///
/// This value is intentionally independent of the score store: callers can
/// add its stable [`Self::key`] rendering to any future cache or report
/// schema without making platform discovery a tuning failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlatformIdentity {
    cpu_model: String,
    logical_cpu: Option<usize>,
    numa_node: Option<u32>,
    caches: Vec<CacheIdentity>,
}

impl PlatformIdentity {
    /// Stable compact rendering suitable for a score key or report field.
    #[must_use]
    pub fn key(&self) -> String {
        let caches = self
            .caches
            .iter()
            .map(cache_key)
            .collect::<Vec<_>>()
            .join(";");
        format!(
            "model={};cpu={};numa={};caches={caches}",
            self.cpu_model,
            option_string(self.logical_cpu),
            option_string(self.numa_node),
        )
    }
}

/// Best-effort processor model string.
pub fn cpu_model() -> String {
    let path = Path::new("/proc/cpuinfo");
    if let Ok(file) = File::open(path) {
        let reader = std::io::BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(value) = line.strip_prefix("model name\t: ") {
                return value.trim().to_owned();
            }
        }
    }
    if let Ok(output) = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        && let Ok(value) = String::from_utf8(output.stdout)
    {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    "unknown".to_owned()
}

/// Best-effort ISO calendar date.
pub fn today() -> String {
    for (program, arguments) in [
        ("date", &["+%Y-%m-%d"][..]),
        (
            "powershell",
            &["-Command", "Get-Date -Format yyyy-MM-dd"][..],
        ),
    ] {
        if let Ok(output) = Command::new(program).args(arguments).output()
            && let Ok(value) = String::from_utf8(output.stdout)
        {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return trimmed.to_owned();
            }
        }
    }
    "unknown".to_owned()
}

/// Collect host cache and topology metadata without making tuning fail.
///
/// Linux reads cache descriptors from sysfs for the one logical CPU selected
/// by the process affinity mask. Other hosts return the processor model and
/// unknown cache/topology fields; no platform-specific dependency is needed.
#[must_use]
pub fn platform_identity() -> PlatformIdentity {
    #[cfg(target_os = "linux")]
    {
        let logical_cpu = linux_allowed_cpu().ok().flatten();
        let (numa_node, caches) = logical_cpu.map_or((None, Vec::new()), |cpu| {
            (linux_numa_node(cpu), linux_caches(cpu))
        });
        PlatformIdentity {
            cpu_model: cpu_model(),
            logical_cpu,
            numa_node,
            caches,
        }
    }

    #[cfg(not(target_os = "linux"))]
    PlatformIdentity {
        cpu_model: cpu_model(),
        logical_cpu: None,
        numa_node: None,
        caches: Vec::new(),
    }
}

/// Identify the one logical CPU to which the launcher restricted this process.
///
/// Linux exposes the exact allowed CPU through procfs and useful core-class
/// metadata through sysfs; both are read with the standard library. Other
/// hosts retain the standard library's affinity-count check. The launcher
/// remains responsible for choosing an idle CPU (`taskset -c` on Linux or the
/// platform equivalent).
pub fn single_cpu_affinity() -> Result<AffinityIdentity, String> {
    #[cfg(target_os = "linux")]
    if let Some(cpu) = linux_allowed_cpu()? {
        let root = format!("/sys/devices/system/cpu/cpu{cpu}");
        let mut parts = vec![format!("logical-cpu={cpu}")];
        for (name, path) in [
            ("core", format!("{root}/topology/core_id")),
            ("capacity", format!("{root}/cpu_capacity")),
        ] {
            if let Ok(value) = fs::read_to_string(path) {
                parts.push(format!("{name}={}", value.trim()));
            }
        }
        for path in [
            format!("{root}/cpufreq/cpuinfo_max_freq"),
            format!("{root}/cpufreq/scaling_max_freq"),
        ] {
            if let Ok(value) = fs::read_to_string(path)
                && let Ok(khz) = value.trim().parse::<u64>()
            {
                parts.push(format!("max-mhz={}", khz.div_euclid(1_000)));
                break;
            }
        }
        return Ok(AffinityIdentity {
            description: parts.join(","),
        });
    }

    let available = std::thread::available_parallelism()
        .map_err(|error| format!("could not inspect process affinity: {error}"))?
        .get();
    if available != 1 {
        return Err(format!(
            "the tuner can run on {available} logical CPUs; launch it with single-CPU affinity"
        ));
    }

    Ok(AffinityIdentity {
        description: "one-logical-cpu".to_owned(),
    })
}

#[cfg(target_os = "linux")]
fn linux_allowed_cpu() -> Result<Option<usize>, String> {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return Ok(None);
    };
    let Some(allowed) = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(str::trim)
    else {
        return Ok(None);
    };
    if allowed.contains(',') {
        return Err(format!(
            "the Linux affinity mask allows CPUs {allowed}; launch the tuner with one CPU"
        ));
    }
    if let Some((start_text, end_text)) = allowed.split_once('-') {
        let start = start_text
            .parse::<usize>()
            .map_err(|error| format!("invalid Linux CPU affinity {allowed}: {error}"))?;
        let end = end_text
            .parse::<usize>()
            .map_err(|error| format!("invalid Linux CPU affinity {allowed}: {error}"))?;
        if start != end {
            return Err(format!(
                "the Linux affinity mask allows CPUs {allowed}; launch the tuner with one CPU"
            ));
        }
        return Ok(Some(start));
    }
    allowed
        .parse::<usize>()
        .map(Some)
        .map_err(|error| format!("invalid Linux CPU affinity {allowed}: {error}"))
}

#[cfg(target_os = "linux")]
fn linux_caches(cpu: usize) -> Vec<CacheIdentity> {
    let root = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.strip_prefix("index")
                        .is_some_and(|n| n.parse::<u32>().is_ok())
                })
        })
        .collect::<Vec<_>>();
    paths.sort_unstable();
    let mut caches = paths
        .iter()
        .map(|path| {
            let level = fs::read_to_string(path.join("level")).unwrap_or_default();
            let kind = fs::read_to_string(path.join("type")).ok();
            let size = fs::read_to_string(path.join("size")).ok();
            let line = fs::read_to_string(path.join("coherency_line_size")).ok();
            let shared = fs::read_to_string(path.join("shared_cpu_list")).ok();
            parse_cache_identity(
                &level,
                kind.as_deref(),
                size.as_deref(),
                line.as_deref(),
                shared.as_deref(),
            )
        })
        .collect::<Vec<_>>();
    caches.sort_by(cache_ordering);
    caches
}

#[cfg(target_os = "linux")]
fn linux_numa_node(cpu: usize) -> Option<u32> {
    let root = format!("/sys/devices/system/cpu/cpu{cpu}");
    if let Ok(entries) = fs::read_dir(root)
        && let Some(node) = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_prefix("node"))
                    .and_then(|node| node.parse::<u32>().ok())
            })
            .min()
    {
        return Some(node);
    }

    let entries = fs::read_dir("/sys/devices/system/node").ok()?;
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let node = entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_prefix("node"))
                .and_then(|node| node.parse::<u32>().ok())?;
            let cpulist = fs::read_to_string(entry.path().join("cpulist")).ok()?;
            cpu_list_contains(&cpulist, cpu).then_some(node)
        })
        .min()
}

fn cpu_list_contains(source: &str, cpu: usize) -> bool {
    source.split(',').any(|segment| {
        let trimmed = segment.trim();
        let Some((start_text, end_text)) = trimmed.split_once('-') else {
            return trimmed.parse::<usize>().is_ok_and(|value| value == cpu);
        };
        let (Ok(start), Ok(end)) = (
            start_text.trim().parse::<usize>(),
            end_text.trim().parse::<usize>(),
        ) else {
            return false;
        };
        start <= cpu && cpu <= end
    })
}

fn parse_cache_identity(
    level: &str,
    kind: Option<&str>,
    size: Option<&str>,
    coherency_line: Option<&str>,
    shared_cpu_list: Option<&str>,
) -> CacheIdentity {
    CacheIdentity {
        level: level.trim().parse::<u32>().ok(),
        kind: kind.and_then(non_empty_trimmed),
        size_bytes: size.and_then(|value| parse_size_bytes(value.trim())),
        coherency_line_bytes: coherency_line.and_then(|value| value.trim().parse::<u32>().ok()),
        shared_cpu_list: shared_cpu_list.and_then(non_empty_trimmed),
    }
}

fn parse_size_bytes(source: &str) -> Option<u64> {
    let trimmed = source.trim();
    let split = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, suffix) = trimmed.split_at(split);
    let number = digits.parse::<u64>().ok()?;
    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1_u64.checked_shl(10)?,
        "m" | "mb" | "mib" => 1_u64.checked_shl(20)?,
        "g" | "gb" | "gib" => 1_u64.checked_shl(30)?,
        _ => return None,
    };
    number.checked_mul(multiplier)
}

fn non_empty_trimmed(source: &str) -> Option<String> {
    let trimmed = source.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn cache_ordering(left: &CacheIdentity, right: &CacheIdentity) -> Ordering {
    left.level
        .cmp(&right.level)
        .then_with(|| left.kind.cmp(&right.kind))
        .then_with(|| left.size_bytes.cmp(&right.size_bytes))
        .then_with(|| left.coherency_line_bytes.cmp(&right.coherency_line_bytes))
        .then_with(|| left.shared_cpu_list.cmp(&right.shared_cpu_list))
}

fn cache_key(cache: &CacheIdentity) -> String {
    format!(
        "l={};t={};s={};line={};shared={}",
        option_string(cache.level),
        cache.kind.as_deref().unwrap_or("unknown"),
        option_string(cache.size_bytes),
        option_string(cache.coherency_line_bytes),
        cache.shared_cpu_list.as_deref().unwrap_or("unknown"),
    )
}

fn option_string<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "unknown".to_owned(), |item| item.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        cache_key, cache_ordering, cpu_list_contains, parse_cache_identity, parse_size_bytes,
    };

    #[test]
    fn parses_linux_cache_sizes_without_host_access() {
        assert_eq!(parse_size_bytes("32K"), Some(32 * 1024));
        assert_eq!(parse_size_bytes(" 2M\n"), Some(2 * 1024 * 1024));
        assert_eq!(parse_size_bytes("4096"), Some(4096));
        assert_eq!(parse_size_bytes("3.5M"), None);
        assert_eq!(parse_size_bytes("bad"), None);
    }

    #[test]
    fn parses_partial_cache_metadata() {
        let cache = parse_cache_identity(
            " 3\n",
            Some(" Unified\n"),
            Some("16M\n"),
            Some("64\n"),
            Some("0-3,8\n"),
        );
        assert_eq!(cache.level, Some(3));
        assert_eq!(cache.kind.as_deref(), Some("Unified"));
        assert_eq!(cache.size_bytes, Some(16 * 1024 * 1024));
        assert_eq!(cache.coherency_line_bytes, Some(64));
        assert_eq!(cache.shared_cpu_list.as_deref(), Some("0-3,8"));

        let partial = parse_cache_identity("unknown", Some("\n"), Some("bad"), None, None);
        assert_eq!(partial.level, None);
        assert_eq!(partial.kind, None);
        assert_eq!(partial.size_bytes, None);
    }

    #[test]
    fn cache_ordering_and_key_are_deterministic() {
        let high = parse_cache_identity("3", Some("Unified"), Some("32M"), Some("64"), Some("0-7"));
        let low = parse_cache_identity("1", Some("Data"), Some("32K"), Some("64"), Some("0"));
        assert!(cache_ordering(&low, &high).is_lt());
        assert_eq!(cache_key(&low), "l=1;t=Data;s=32768;line=64;shared=0");
    }

    #[test]
    fn parses_linux_cpu_lists_without_host_access() {
        assert!(cpu_list_contains("0-3,8,10-12", 0));
        assert!(cpu_list_contains("0-3,8,10-12", 11));
        assert!(!cpu_list_contains("0-3,8,10-12", 9));
        assert!(!cpu_list_contains("bad,8-x", 8));
    }
}
