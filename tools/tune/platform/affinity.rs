//! Explicit CPU-set validation for parallel policy measurements.

#[cfg(target_os = "linux")]
use std::fs::read_to_string;
use std::thread::available_parallelism;

use super::{AffinityIdentity, Platform};

impl Platform {
    /// Match the launcher's affinity mask to an explicit parallel CPU set.
    ///
    /// Linux records every selected CPU's exposed core and capacity metadata.
    /// Other hosts require an affinity-count match; exact IDs remain unavailable.
    pub fn parallel_cpu_affinity(specification: &str) -> Result<AffinityIdentity, String> {
        let cpus = Self::parse_cpu_list(specification)?;
        if cpus.len() < 2 {
            return Err("parallel tuning requires at least two selected CPUs".to_owned());
        }
        let available = available_parallelism()
            .map_err(|error| error.to_string())?
            .get();
        if available < cpus.len() {
            return Err(format!(
                "{} workers exceed the available parallelism budget {available}",
                cpus.len()
            ));
        }
        let cpu_list = cpus
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let selected = format!("selected-cpus={cpu_list};max-workers={}", cpus.len());
        #[cfg(target_os = "linux")]
        let description = {
            let mut parts = vec![selected];
            let status = read_to_string("/proc/self/status").map_err(|error| error.to_string())?;
            let allowed = status
                .lines()
                .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
                .ok_or("Linux did not expose the process affinity mask")?;
            if Self::parse_cpu_list(allowed)? != cpus {
                return Err(format!(
                    "launch with taskset -c {specification}; actual affinity is {}",
                    allowed.trim()
                ));
            }
            for cpu in cpus {
                for (name, field) in [
                    ("core", "topology/core_id"),
                    ("package", "topology/physical_package_id"),
                    ("capacity", "cpu_capacity"),
                    ("max-khz", "cpufreq/cpuinfo_max_freq"),
                ] {
                    if let Ok(value) =
                        read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{field}"))
                    {
                        parts.push(format!("cpu{cpu}-{name}={}", value.trim()));
                    }
                }
            }
            parts.join(";")
        };
        #[cfg(not(target_os = "linux"))]
        if available != cpus.len() {
            return Err("parallel tuning requires an exact affinity-count match".to_owned());
        }
        #[cfg(not(target_os = "linux"))]
        let description = selected;
        Ok(AffinityIdentity { description })
    }

    /// Expand and validate a Linux-style list of unique CPU IDs and ranges.
    pub fn parse_cpu_list(specification: &str) -> Result<Vec<usize>, String> {
        let mut cpus = Vec::new();
        for part in specification.trim().split(',').map(str::trim) {
            let (start, end) = if let Some((start, end)) = part.split_once('-') {
                (
                    start.parse::<usize>().map_err(|error| error.to_string())?,
                    end.parse::<usize>().map_err(|error| error.to_string())?,
                )
            } else {
                let cpu = part.parse::<usize>().map_err(|error| error.to_string())?;
                (cpu, cpu)
            };
            let count = end
                .checked_sub(start)
                .and_then(|span| span.checked_add(1))
                .ok_or("invalid CPU range")?;
            cpus.try_reserve(count).map_err(|error| error.to_string())?;
            cpus.extend(start..=end);
        }
        cpus.sort_unstable();
        if cpus.windows(2).any(|pair| pair.first() == pair.get(1)) {
            return Err("CPU set contains duplicates".to_owned());
        }
        Ok(cpus)
    }
}
