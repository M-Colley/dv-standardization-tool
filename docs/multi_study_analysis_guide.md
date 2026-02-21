# Multi-Study Analysis Guide (After DV Standardization)

This guide shows how to analyze **multiple standardized datasets** even when independent variables are unavailable, inconsistent, or poorly documented.

## Why these analyses are useful without IVs
When studies share standardized dependent variable (DV) names, you can still do meaningful cross-study work:

1. **Coverage/compatibility analysis**: quantify how much the studies can be compared (DV overlap).
2. **Outcome benchmarking**: compare study-level performance on shared outcomes.
3. **Cross-study latent construct extraction**: build a pooled composite score from common DVs.

## Included reproducible example datasets
Three synthetic standardized datasets are included in:

- `data/processed/multi_study_examples/study_a_standardized.csv`
- `data/processed/multi_study_examples/study_b_standardized.csv`
- `data/processed/multi_study_examples/study_c_standardized.csv`

They intentionally have:
- **Overlapping DVs** (`task_success_rate`, and subsets of timing/usability/workload variables)
- **Study-specific DVs** (e.g., `mental_demand_score`, `workload_score`)

## Python workflow
Script: `analyses/multi_study_analysis.py`

### Install latest packages
```bash
python -m pip install --upgrade pip
python -m pip install --upgrade pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Run
```bash
python analyses/multi_study_analysis.py \
  --input-dir data/processed/multi_study_examples \
  --output-dir analyses/output_python
```

### Outputs
- `analyses/output_python/dv_overlap_matrix.csv`
- `analyses/output_python/harmonized_dv_summary.csv`
- `analyses/output_python/cross_study_composite_summary.csv`
- `analyses/output_python/dv_overlap_heatmap.png`
- `analyses/output_python/dv_mean_shift.png`

## R workflow
Script: `analyses/multi_study_analysis.R`

### Install latest packages from CRAN
```r
install.packages(c("dplyr", "readr", "readxl", "ggplot2", "tidyr", "purrr"))
```

### Run
```bash
Rscript analyses/multi_study_analysis.R \
  data/processed/multi_study_examples \
  analyses/output_r
```

### Outputs
- `analyses/output_r/dv_overlap_matrix.csv`
- `analyses/output_r/harmonized_dv_summary.csv`
- `analyses/output_r/cross_study_composite_summary.csv`
- `analyses/output_r/dv_overlap_heatmap.png`
- `analyses/output_r/dv_mean_shift.png`

## Applying to your own standardized studies
1. Put each standardized dataset (`.csv` or `.xlsx`) into one folder.
2. Ensure DV columns are numeric and consistently named using the schema.
3. Run either script against that folder.
4. Inspect overlap first; if overlap is weak, restrict to the best-shared subset and rerun.

## Interpretation notes
- **High Jaccard overlap** indicates strong interoperability.
- **Mean z-shifts** highlight which studies are above/below pooled DV baselines.
- **Composite index** gives a common latent outcome for cross-study ranking/comparison.

This pattern is especially helpful for exploratory synthesis, preregistration planning, and power analysis preparation before full model-based meta-analysis.
