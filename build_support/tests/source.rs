//! Complete-profile round trips and accepted declaration syntax.

use super::super::TuningProfile;

// Miri limits field mutations; each candidate still parses all declarations.
const MUTATED_CONSTANT_LIMIT: usize = if cfg!(miri) { 4 } else { usize::MAX };

#[test]
fn every_constant_round_trips_independently() {
    let profile = TuningProfile::portable();
    let source = profile.render("// profile");
    assert_eq!(TuningProfile::from_source(&source), Ok(profile));
    for declaration in source
        .lines()
        .filter(|line| line.starts_with("pub const "))
        .take(MUTATED_CONSTANT_LIMIT)
    {
        let (name, _) = declaration.split_once('=').expect("rendered declaration");
        for value in [
            0,
            1,
            9_999,
            10_000,
            usize::MAX - 2,
            usize::MAX - 1,
            usize::MAX,
        ] {
            let literal = if value == usize::MAX - 1 {
                "usize::MAX - 1".to_owned()
            } else {
                value.to_string()
            };
            let changed = source.replace(declaration, &format!("{name}= {literal};"));
            let parsed = TuningProfile::from_source(&changed).expect("usize value parses");
            let rendered = parsed.render("// profile");
            let rendered_declaration = rendered
                .lines()
                .find(|line| line.starts_with(name))
                .expect("constant remains in the rendered profile");
            assert_eq!(
                rendered_declaration
                    .replace('_', "")
                    .split_once('=')
                    .expect("value")
                    .1
                    .trim(),
                format!("{literal};"),
                "{name} retains the input value"
            );
            let expected = source.replace(declaration, rendered_declaration);
            assert_eq!(rendered, expected, "only {name} changes");
            assert_eq!(TuningProfile::from_source(&rendered), Ok(parsed));
            if value == usize::MAX - 1 {
                assert!(rendered.contains(&format!("{name}= usize::MAX - 1;")));
            }
        }
    }
}

#[test]
fn comments_spacing_and_declaration_order_preserve_values() {
    let mut profile = TuningProfile::portable();
    profile.toom_cook_6 = usize::MAX - 1;
    profile.toom_cook_85 = usize::MAX - 1;
    let source = profile.render("// pub const KARATSUBA_THRESHOLD: usize = 999;");
    let reordered = source.lines().rev().collect::<Vec<_>>().join("\n");
    let spaced = source
        .replace("pub const ", " const ")
        .replace(": usize = ", " :usize= ")
        .replace("usize::MAX - 1", "usize :: MAX - 1")
        .replace(';', " ; // declaration");
    for variant in [&source, &reordered, &spaced] {
        assert_eq!(TuningProfile::from_source(variant), Ok(profile));
    }
}

#[test]
fn incomplete_duplicate_and_malformed_profiles_are_rejected() {
    let source = TuningProfile::portable().render("// profile");
    for declaration in source
        .lines()
        .filter(|line| line.starts_with("pub const "))
        .take(MUTATED_CONSTANT_LIMIT)
    {
        for replacement in [String::new(), format!("// {declaration}")] {
            let incomplete = source.replace(declaration, &replacement);
            assert!(
                TuningProfile::from_source(&incomplete).is_err(),
                "{declaration}"
            );
        }
        let duplicate = format!("{source}\n{declaration}");
        assert!(
            TuningProfile::from_source(&duplicate).is_err(),
            "{declaration}"
        );
    }
    let declaration = source
        .lines()
        .find(|line| line.starts_with("pub const "))
        .expect("profile has declarations");
    let (name, _) = declaration.split_once('=').expect("rendered declaration");
    for value in [
        "",
        "1 2",
        "-1",
        "+1",
        "_12",
        "1 + 2",
        "usize::MAX",
        "1; 2",
        "0x10",
        "1usize",
        "18446744073709551616",
    ] {
        let invalid = source.replace(declaration, &format!("{name}= {value};"));
        assert!(
            TuningProfile::from_source(&invalid).is_err(),
            "value {value:?}"
        );
    }
    for invalid in [
        declaration.replace(": usize", ""),
        declaration.replace("usize", "u64"),
        declaration.replace('=', ""),
        declaration.replace(';', ""),
    ] {
        assert!(TuningProfile::from_source(&source.replace(declaration, &invalid)).is_err());
    }
    for extra in [
        "pub const UNKNOWN_THRESHOLD: usize = 1;",
        "fn injected() {}",
        "#[cfg(any())]",
        "/* block comment */",
    ] {
        assert!(
            TuningProfile::from_source(&format!("{source}\n{extra}")).is_err(),
            "{extra}"
        );
    }
}
