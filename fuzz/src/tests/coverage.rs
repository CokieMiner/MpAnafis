//! Stable inherent API inventory and fuzz call-site coverage checks.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[test]
#[cfg_attr(
    miri,
    ignore = "API inventory discovery reads the host filesystem, unavailable under Miri isolation"
)]
fn stable_inherent_inventory_matches_source_and_fuzz_call_sites() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let inventory = include_str!("../../../docs/int/api-inventory.md");
    for (kind, section, count) in [("uint", "## 2.", 159), ("int", "## 3.", 169)] {
        let section = inventory
            .split(section)
            .nth(1)
            .unwrap()
            .split("\n## ")
            .next()
            .unwrap();
        let documented = names(section.lines().filter(|line| line.starts_with("- `pub ")));
        let api = root.join("src/int/api");
        let source = rust_files(&api)
            .into_iter()
            .filter(|file| {
                file.strip_prefix(&api)
                    .unwrap()
                    .components()
                    .any(|component| component.as_os_str() == kind)
            })
            .map(|file| fs::read_to_string(file).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let implemented = names(
            source
                .lines()
                .map(str::trim_start)
                .filter(|line| line.starts_with("pub ")),
        );
        assert_eq!(documented.len(), count, "{kind} inventory count");
        assert_eq!(
            implemented, documented,
            "{kind} source and inventory disagree"
        );
        let folder = if kind == "uint" { "unsigned" } else { "signed" };
        let harness = rust_files(&root.join("fuzz/src").join(folder))
            .into_iter()
            .map(|file| fs::read_to_string(file).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let code = harness
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let identifiers: BTreeSet<_> = code
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .filter(|word| !word.is_empty())
            .collect();
        // Macro arguments name policy methods whose generic call sites are type checked.
        let missing: Vec<_> = documented
            .iter()
            .filter(|name| !identifiers.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "{kind} methods absent from fuzz cases: {missing:?}"
        );
    }
}

fn names<'a>(lines: impl Iterator<Item = &'a str>) -> BTreeSet<String> {
    lines
        .filter_map(|line| line.split_once("fn ").map(|(_, rest)| rest))
        .map(|rest| {
            rest.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                .next()
                .unwrap()
                .to_owned()
        })
        .collect()
}

fn rust_files(folder: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(folder).unwrap() {
        let path = entry.unwrap().path();
        if path
            .file_name()
            .is_some_and(|name| name == "tests" || name == "tests.rs")
        {
            continue;
        }
        if path.is_dir() {
            files.extend(rust_files(&path));
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    files.sort();
    files
}
