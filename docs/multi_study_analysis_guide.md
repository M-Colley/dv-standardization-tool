# Multi-Study Analysis Guide (After DV Standardization)

This guide shows how to analyze **multiple standardized datasets** even when independent variables are unavailable, inconsistent, or poorly documented.

## Why these analyses are useful without IVs
When studies share standardized dependent variable (DV) names, you can still do meaningful cross-study work:

1. **Coverage/compatibility analysis**: quantify how much the studies can be compared (DV overlap).
2. **Outcome benchmarking**: compare study-level performance on shared outcomes.
3. **Cross-study latent construct extraction**: build a pooled composite score from common DVs.
4. **Random-effects DV meta-analysis**: estimate pooled means and heterogeneity for shared DVs.

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

### Run on all latest standardized batch datasets (with fallback)
Use this helper command/script to automatically use all files from
`data/processed/batch_runs/latest/standardized` when available, and otherwise
fall back to the bundled multi-study examples.

```bash
bash analyses/run_latest_multi_study_analysis.sh
```

Optional custom output directory:

```bash
bash analyses/run_latest_multi_study_analysis.sh analyses/output_python_latest_standardized
```

### Outputs
- `analyses/output_python/dv_overlap_matrix.csv`
- `analyses/output_python/harmonized_dv_summary.csv`
- `analyses/output_python/meta_analysis_summary.csv`
- `analyses/output_python/cross_study_composite_summary.csv`
- `analyses/output_python/dv_overlap_heatmap.png`
- `analyses/output_python/dv_mean_shift.png`
- `analyses/output_python/dv_coverage_by_study.png`
- `analyses/output_python/cross_study_composite_distribution.png`

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


## Derived score auto-calculation (Python and R)
Both analysis scripts now auto-calculate common questionnaire scores when all required item-level columns are available (including mapped aliases):

- **NASA-TLX composite** (`nasa_tlx_score`) from six TLX subscales/items
- **SUS composite** (`sus_score`) from `sus1`..`sus10` (with alternating reverse coding)
- **van der Laan AOA subscales**:
  - `aoa_usefulness` from `aoa1, aoa3, aoa5, aoa7, aoa9`
  - `aoa_satisfying` from `aoa2, aoa4, aoa6, aoa8`

This uses standardized/mapped names when present (for example `mental_demand`, `physical_demand`, etc. for TLX items) so the calculations remain robust after DV standardization.

## Interpretation notes
- **High Jaccard overlap** indicates strong interoperability.
- **Mean z-shifts** highlight which studies are above/below pooled DV baselines.
- **Composite index** gives a common latent outcome for cross-study ranking/comparison.

This pattern is especially helpful for exploratory synthesis, preregistration planning, and power analysis preparation before full model-based meta-analysis.
