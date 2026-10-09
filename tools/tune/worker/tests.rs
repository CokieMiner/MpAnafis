//! Frozen catalog masks are validated before constructing worker operands.

use super::{CellSelection, PARSING_CHUNK_SIZES, ParsingWorker};

#[test]
fn parsing_worker_freezes_valid_chunk_dimensions_before_allocating() {
    assert_eq!(ParsingWorker::widths(""), Ok(PARSING_CHUNK_SIZES.to_vec()));
    assert_eq!(ParsingWorker::widths("1,19,64"), Ok(vec![1, 19, 64]));
    for invalid in ["0", "1,0", "1,", "invalid", "-1"] {
        assert!(ParsingWorker::widths(invalid).is_err(), "{invalid}");
    }
    assert!(ParsingWorker::widths(&usize::MAX.to_string()).is_err());
    assert_eq!(ParsingWorker::cell_weights(&[1, 4, 16]).len(), 21);
}

#[test]
fn cell_selection_round_trips_dense_and_sparse_catalogs() {
    for indices in [
        vec![0],
        vec![1, 3, 5, 10, 11, 12, 25],
        (0..100).collect(),
        (0..100).step_by(4).collect(),
    ] {
        assert_eq!(
            CellSelection::parse(&CellSelection::encode(&indices), 100),
            Ok(indices)
        );
    }
    assert_eq!(CellSelection::parse("", 4), Ok(vec![0, 1, 2, 3]));
    assert_eq!(CellSelection::parse("1-8/3,9", 10), Ok(vec![1, 4, 7, 9]));
    let sparse: Vec<_> = (0..100_000).step_by(4).collect();
    let encoded = CellSelection::encode(&sparse);
    assert!(encoded.len() < 32);
    assert_eq!(CellSelection::parse(&encoded, 100_000), Ok(sparse));
}

#[test]
fn cell_selection_rejects_ambiguous_and_invalid_indices() {
    for encoded in [
        "0,0",
        "2,1",
        "0-3,3",
        "3-1",
        "0-4",
        "0-3/0",
        "-1",
        "1,",
        "",
        "0/invalid",
        "0-3/1/2",
    ] {
        let count = if encoded.is_empty() { 0 } else { 4 };
        assert!(CellSelection::parse(encoded, count).is_err(), "{encoded}");
    }
    assert!(CellSelection::parse(&usize::MAX.to_string(), 4).is_err());
}
