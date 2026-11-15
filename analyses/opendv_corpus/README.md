# OpenDV corpus: 88 studies with open data

Every study the repository maps, analysed as one corpus with curated mappings only (no LLM). The write-up of the results is [docs/opendv_corpus.md](../../docs/opendv_corpus.md).

| File | What it is |
|---|---|
| `build_manifest.py` → `sources_manifest.yaml` | batch manifest of the 39 development and 49 held-out studies (each entry unchanged, plus `study_set`) |
| `corrections.py` → `mappings/` | corrected copies of six held-out mappings whose NASA-TLX scales were not declared (the frozen files are not changed; see `docs/opendv_corpus.md` §5) |
| `studies.py` → `corpus_studies.csv` | one row per study: set, venue, year, DOI, venue group (SIGCHI, other ACM, IEEE, other, no paper) |
| `analysis/` | `opendv analyze` on the corpus run |
| `relations/` | `construct_relations` on the corpus run, at the levels raw names, shared schema, curated mappings |
| `value.py` → `value/` | what the mapping adds: names per DV, measures and study pairs shared under raw names vs the mapping, ambiguous raw names, schema growth from the git history, pooled values with k ≥ 5, and a scale audit of every study × DV |

The held-out folders (`analyses/heldout/`, `analyses/heldout_2/`) are pre-registered records; this folder only reads their manifests and mappings.

## Reproduce

Data: the development studies as in `docs/reproducibility.md`; the held-out studies with

```bash
python -m opendv.analysis.confirmatory fetch --sources analyses/heldout/sources.yaml --data-dir data/raw/heldout
python -m opendv.analysis.confirmatory fetch --sources analyses/heldout_2/sources.yaml --data-dir data/raw/heldout_2
```

Then:

```bash
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg
python -m analyses.opendv_corpus.corrections
python -m analyses.opendv_corpus.build_manifest
python -m analyses.opendv_corpus.studies
opendv batch --manifest analyses/opendv_corpus/sources_manifest.yaml --output-dir data/processed/corpus88c \
  --disable-llm-deduction --workers 3 --cache-dir data/processed/cache --no-fair-export
opendv analyze --input-dir data/processed/corpus88c/standardized --output-dir analyses/opendv_corpus/analysis --exclude-llm-deduced
python -m opendv.analysis.construct_relations --input-dir data/processed/corpus88c/standardized --output-dir analyses/opendv_corpus/relations
python -m analyses.opendv_corpus.value --batch-dir data/processed/corpus88c
```

`value.py` reads the git history of `schemas/standard_dv_mapping.yaml` and of the mapping files, so it needs a git checkout.
