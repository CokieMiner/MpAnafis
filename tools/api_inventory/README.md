# API inventory

`tools/api_inventory.py` generates a deterministic TSV from rustdoc JSON.
It follows public module and reexport paths, omits hidden items and private
members, and retains cfg traces from enclosing modules.

Generate JSON with private-item documentation so module traces remain available:

```sh
cargo +nightly rustdoc --locked --lib --all-features -- \
  -Z unstable-options --output-format json --document-private-items
python3 tools/api_inventory.py
python3 tools/api_inventory.py --check
```

Defaults use `target/doc/mp_anafis.json` and `tools/api_inventory.tsv`.
`--rustdoc-json` and `--output` select explicit files. `--check` compares without
writing: exit 0 means agreement, 1 means stale or missing output, and 2 means
invalid input or a generation failure. Unsupported rustdoc format versions fail
explicitly; supported versions are declared in `models.py`.

The package separates JSON validation and reachability (`inventory.py`), signature
normalization (`normalizer.py`), shared records (`models.py`), and CLI/output
handling (`renderer.py`).

The inventory describes the selected Cargo feature set. Source visibility audits
also inspect feature-gated branches unavailable in that compiled configuration.
