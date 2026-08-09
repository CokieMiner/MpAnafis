//! Host metadata and measurement hygiene.

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
