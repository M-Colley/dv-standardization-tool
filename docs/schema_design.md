# DV Schema Design

## Why YAML?
YAML was chosen for its:
- Human readability
- Nested hierarchy support
- Compatibility with version control systems

## Naming Convention Principles
Drawing from inconsistencies highlighted in CHI, AutomotiveUI, and CHI PLAY literature:
- Use **lower_snake_case** for all variable names
- Include **construct**, **metric**, and **unit** fields
- Support **aliases** for mapping legacy names

## Example YAML Structure

```yaml
dependent_variable:
  name: task_completion_time
  construct: efficiency
  metric: duration
  unit: seconds
  aliases:
    - time_taken
    - completion_time
    - total_time
````
**Schema Validation:** Implemented through the schema validator and visualized in error logs.

## Constructs, instruments and items

The DV schema (`schemas/standard_dv_mapping.yaml`) names **outcomes**, e.g. `usability` or `mental_demand`. How an outcome is measured lives in two further files:

- **Instrument registry** (`schemas/instruments.yaml`, `opendv/instruments.py`). Each standard questionnaire lists its items (raw column aliases and keying, +1 or −1), the candidate response codings, and the scoring rule of each score, with a citation. Registered: SUS, raw NASA-TLX, Van der Laan acceptance (AOA), UEQ-S, SSQ and Körber's TiA. The analysis computes every score whose items are all present, so adding a questionnaire needs no Python:

  ```yaml
  instruments:
    sus:
      citation: "Brooke (1996) ..."
      response_codings: [[1, 5]]
      items:
        - {id: SUS1, aliases: [sus1, sus_1], key: 1}
        - {id: SUS2, aliases: [sus2, sus_2], key: -1}
        # ...
      scores:
        sus_score: {dv: usability, items: [SUS1, ..., SUS10], centering: min, aggregate: sum, multiplier: 2.5}
  ```

  `centering` is `none`, `reflect` (reverse items become min + max − x), `min` (x − min / max − x) or `midpoint` (centred on the scale midpoint). When an instrument lists several codings (Van der Laan items are exported as 1..5 or −2..+2), the first coding that contains every observed answer is used, and an ambiguous match is logged. Answers outside every coding are not scored. `tests/test_instruments.py` checks each score against worked examples and checks that its possible range equals the DV's `canonical_range`.
- **Study designs** (`schemas/study_designs.yaml`): participant and condition columns per study, used for condition contrasts (see `docs/multi_study_analysis_guide.md`).

A DV should name one instrument's scale, or a construct whose instruments share a range. Single-item mental-effort ratings were split off `mental_demand` (NASA-TLX, 0–20) into `paas_mental_effort` (1–9) and `rsme` (0–150) for that reason.

Dataset-specific multi-item scales that are not standard instruments are declared in the dataset's mapping YAML under `scales:` (items, `reverse_items`, `scale_range`); undeclared item groups fall back to `opendv/item_composition.py`, which infers reverse-keyed items from correlations and reports Cronbach's α.

All four standard schemas, the registry and the design file are validated in CI with `python -m opendv.validate_schema <file>`.
