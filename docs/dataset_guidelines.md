# Dataset Upload & Anonymization Guidelines

## Sourcing Data

* Prioritize publicly available HCI datasets shared through ACM Digital Library, OSF, or repositories with DOI support.
* Ensure data contain relevant DV fields with some inconsistencies for transformation testing.

## Publishing data so the pipeline can read it

Deposits on Zenodo, figshare, 4TU.ResearchData or a Dataverse installation can be added to a manifest directly (`zenodo_record`, `figshare_article`, `fourtu_dataset`, `dataverse_dataset`; see the README). To make a deposit easy to standardize:
- upload tables as CSV/TSV/XLSX (or zip archives of them) rather than only as PDFs or proprietary formats;
- keep the raw column names and add a short mapping YAML (`raw column: canonical DV id`, optionally `value_labels` and `scales`) next to the data;
- state participant and condition columns, so the study can be added to `schemas/study_designs.yaml`.

## Upload Protocol

1. Use `data/raw/` for unprocessed datasets
2. Upload corresponding `schema.yaml` under `schemas/`
3. Add logs in `logs/` after running the tool

## Anonymization

* No personally identifiable information (PII) must be present in uploaded data
* Fields like `participant_id`, `email`, or `free_text_response` should be:

  * Removed
  * Pseudonymized
  * Or flagged in schema via `sensitive: true`

## Sample Workflow

```
raw_dataset.csv → anonymize → schema.yaml → convert → standardized_output.csv
```
