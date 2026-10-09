# Integer performance curves

Benchmark source stays in [benches/](../../../benches/public_api/README.md).
This directory is for reviewed **time-versus-size graphs**, grouped by suite,
experiment, category, and function. Git retains only the curves, a short README,
and a measurement manifest. No performance study is currently retained here.

```text
public_api/<record-name>/
  README.md
  manifest.json
  plots/public_api/int/{signed,unsigned}/<category>/<function>/<configuration>.png
```

Internal studies use `internal_improvement/<record-name>/` with the same intent.
The shared reporter supports internal captures; the checked publisher currently
exports public API runs only.

## Local evidence and validation

Complete runs stay under ignored `target/bench-results/`: raw stdout/stderr,
commands, plan, host/compiler/revision metadata, measurements, tables, and
reports. Single-size comparisons belong in those tables, not in bar charts.
Each plotted engine needs at least two numeric sizes. Missing sizes are never
fabricated, and distinct functions, configurations, and worker budgets remain
separate. Shape/worker arguments without a scalar size remain tabular.

Legacy captures and the single-size harness record are preserved locally in
`target/bench-results/retained-docs-before-size-sweeps/`. They are not committed
performance studies. Preserve valuable local runs outside disposable build
storage before cleaning `target/`.

## Exporting curves

Use the [shared driver](../../../tools/benchmark/README.md):

```sh
python3 tools/bench.py run \
  --case 'int::unsigned::arithmetic::operators::add' \
  --arg 256 --arg 1024 --arg 4096 --arg 65536 --rounds 3 \
  --output target/bench-results/add-review

python3 tools/bench.py publish target/bench-results/add-review \
  --name add-review-2026-09-20 \
  --description 'State the question, size units, batch size, conditions, and interpretation.'
```

Omit `--arg` to run the chosen function's entire registered ladder. Width limits
are operation-specific; the [driver guide](../../../tools/benchmark/README.md)
lists current ranges.

The exporter validates measurements against every raw capture before producing
curves and a manifest with measurement details and source hashes. It rejects incomplete,
single-size, or smoke runs and refuses existing record names. The manifest does
not contain the raw measurements: retain/share the complete run separately for
reproducibility. A dirty revision identifier alone cannot reconstruct sources.

Keep retained records immutable. Regenerate local reports in a working directory
or choose a new record name. `.gitignore` excludes other generated artifacts from
this documentation tree while permitting README files, manifests, and plot PNGs.
