# OpenDV-HCI

OpenDV-HCI maps the dependent-variable columns of HCI study data (`TLX1`, `trust_overall`, `PerSafe03`, ...) to one canonical vocabulary, scores standard questionnaires, and synthesizes results across studies: pooled means, condition contrasts, measurement invariance and a multiverse of harmonization choices. Every mapping records where it came from (curated alias, schema alias, local LLM, human review), so each pooled number can be traced to the columns behind it.

## Install

```bash
pip install -e .                 # core
pip install -e .[llm,ui]         # + local-LLM mapping and the Streamlit app
```

For the exact environment behind the published results, see [Reproducing the results](reproducibility.md).

## Quick start

```bash
opendv batch --manifest sources_manifest_no4tu.yaml --output-dir out     # fetch + standardize 11 studies
opendv analyze --input-dir out/standardized --output-dir results         # cross-study analysis
opendv review queue out                                                   # what needs a human decision
```

or from Python, see the [Python API](api.md).

## Where to go next

- [Overview](overview.md) and [usage workflows](usage_workflows.md): what the tool does and the common workflows.
- [Multi-study analysis](multi_study_analysis_guide.md): the meta-analysis, contrasts, invariance and multiverse outputs.
- [Schema design](schema_design.md) and [dataset guidelines](dataset_guidelines.md): how DVs, aliases, instruments and dataset mappings are declared.
- [Architecture](architecture.md): pipeline stages, package layout and plugin entry points.
- [Research agenda](research_agenda.md): results so far and open questions.
