# Python API

`import opendv` gives the functions below; they are stable across 3.x. Every other module is internal. The command line covers the same ground: `opendv standardize | batch | analyze | validate | review | ui | plugins` (`opendv <command> --help`).

```python
import opendv

opendv.run_batch("sources_manifest_no4tu.yaml", "out", workers=8)   # fetch + standardize
opendv.analyze("out/standardized", "results")                       # meta-analysis, contrasts, invariance, ...
queue = opendv.review_queue("out")                                   # columns needing a human decision
df = opendv.standardize("my_study.csv")                              # one table -> canonical DV names
scored = opendv.score(df)                                            # questionnaire scores (SUS, TLX, ...)
```

::: opendv.standardize

::: opendv.run_batch

::: opendv.analyze

::: opendv.review_queue

::: opendv.score

::: opendv.load_plugins

## Plugins

Another installed package can add source types, file formats and instruments through entry points in its `pyproject.toml`, without forking OpenDV-HCI. They are loaded once, on first use, by the batch runner, the file loader and the instrument registry; a plugin that fails to load is logged and skipped. `opendv plugins` lists what is installed.

| Entry-point group | Entry-point name | Object |
|---|---|---|
| `opendv.sources` | the manifest `source_type` | `materialize(source, target, refresh)` returning `(base_dir, provenance_id)` |
| `opendv.loaders` | file suffix without the dot | `load(path) -> pandas.DataFrame` |
| `opendv.instruments` | any | path of an instrument registry YAML (format of `schemas/instruments.yaml`), or a callable returning one |

```toml
[project.entry-points."opendv.sources"]
my_archive = "my_plugin:materialize"

[project.entry-points."opendv.loaders"]
sav2 = "my_plugin:load_sav2"

[project.entry-points."opendv.instruments"]
my_scales = "my_plugin:REGISTRY_PATH"
```
