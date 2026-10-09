//! Adjacent-tier argument ordering, domains, and rejected allocation widths.

use core::mem::size_of;

use super::{PairDomain, PairSpecification, ProbeQuality};

#[test]
fn complete_pair_protocols_preserve_fields_and_reject_invalid_domains_and_widths() {
    for (specification, domain, baseline, candidate, radix, len, quality, iterations, confidence) in [
        (
            "schoolbook,mulders,32,coarse,12,0,992499",
            PairDomain::Arithmetic,
            "schoolbook",
            "mulders",
            0,
            32,
            ProbeQuality::Coarse,
            12,
            0,
        ),
        (
            "mulders,full,259,precise,40,15,992499",
            PairDomain::Arithmetic,
            "mulders",
            "full",
            0,
            259,
            ProbeQuality::Precise,
            40,
            15,
        ),
        (
            "montgomery,barrett,96,precise,40,15,992499",
            PairDomain::Arithmetic,
            "montgomery",
            "barrett",
            0,
            96,
            ProbeQuality::Precise,
            40,
            15,
        ),
        (
            "schoolbook,recursive,10,64,precise,5,15,992499",
            PairDomain::Formatting,
            "schoolbook",
            "recursive",
            10,
            64,
            ProbeQuality::Precise,
            5,
            15,
        ),
    ] {
        let pair = PairSpecification::parse(specification, domain).expect("valid worker pair");
        assert_eq!(
            (
                pair.baseline,
                pair.candidate,
                pair.radix,
                pair.len,
                pair.quality,
                pair.iterations,
                pair.confidence_bits,
                pair.maximum_ratio
            ),
            (
                baseline, candidate, radix, len, quality, iterations, confidence, 992_499
            ),
        );
    }
    for specification in [
        "",
        "schoolbook,mulders,4",
        "schoolbook,mulders,invalid,coarse,12,0,992499",
        "schoolbook,mulders,0,coarse,12,0,992499",
        "schoolbook,mulders,4,unknown,12,0,992499",
        "schoolbook,mulders,4,coarse,0,0,992499",
        "schoolbook,mulders,4,coarse,12,15,992499",
        "schoolbook,mulders,4,precise,12,0,992499",
        "schoolbook,mulders,4,coarse,12,0,0",
        "schoolbook,mulders,4,coarse,12,0,1000000",
        "schoolbook,mulders,4,coarse,12,0,992499,extra",
    ] {
        assert!(PairSpecification::parse(specification, PairDomain::Arithmetic).is_err());
    }
    for width in [usize::MAX, usize::MAX.div_euclid(size_of::<usize>())] {
        let specification = format!("schoolbook,mulders,{width},coarse,1,0,992499");
        assert!(PairSpecification::parse(&specification, PairDomain::Arithmetic).is_err());
    }
    for specification in [
        "schoolbook,recursive,2,64,precise,5,15,992499",
        "schoolbook,recursive,8,64,precise,5,15,992499",
        "schoolbook,recursive,37,64,precise,5,15,992499",
        "schoolbook,recursive,10,64,precise,5,15,992499,extra",
    ] {
        assert!(PairSpecification::parse(specification, PairDomain::Formatting).is_err());
    }
}
