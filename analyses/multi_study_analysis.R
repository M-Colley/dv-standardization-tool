#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(readxl)
  library(ggplot2)
  library(tidyr)
  library(purrr)
})

id_like <- c(
  "id", "participant_id", "userid", "user_id", "subject_id",
  "session_id", "submitdate", "seed", "lastpage"
)

normalize_colname <- function(x) {
  gsub("[^a-z0-9]", "", tolower(x))
}

resolve_mapped_series <- function(df, candidates) {
  normalized_df <- normalize_colname(names(df))
  for (cand in candidates) {
    idx <- which(normalized_df == normalize_colname(cand))
    if (length(idx) > 0) {
      return(suppressWarnings(as.numeric(df[[idx[[1]]]])))
    }
  }
  NULL
}

add_derived_scale_scores <- function(df) {
  out <- df

  tlx_item_candidates <- list(
    c("tlx1", "nasa_tlx1", "mental_demand", "tlx_mental_demand", "nasa_tlx_mental"),
    c("tlx2", "nasa_tlx2", "physical_demand"),
    c("tlx3", "nasa_tlx3", "temporal_demand"),
    c("tlx4", "nasa_tlx4", "performance"),
    c("tlx5", "nasa_tlx5", "effort"),
    c("tlx6", "nasa_tlx6", "frustration")
  )
  tlx_items <- map(tlx_item_candidates, ~ resolve_mapped_series(out, .x))
  if (all(map_lgl(tlx_items, ~ !is.null(.x)))) {
    out$nasa_tlx_score <- rowMeans(as.data.frame(tlx_items), na.rm = FALSE)
  }

  sus_item_candidates <- list(
    c("sus1", "sus_1"), c("sus2", "sus_2"), c("sus3", "sus_3"),
    c("sus4", "sus_4"), c("sus5", "sus_5"), c("sus6", "sus_6"),
    c("sus7", "sus_7"), c("sus8", "sus_8"), c("sus9", "sus_9"),
    c("sus10", "sus_10")
  )
  sus_items <- map(sus_item_candidates, ~ resolve_mapped_series(out, .x))
  if (all(map_lgl(sus_items, ~ !is.null(.x)))) {
    out$sus_score <- (
      (sus_items[[1]] - 1) +
      (sus_items[[3]] - 1) +
      (sus_items[[5]] - 1) +
      (sus_items[[7]] - 1) +
      (sus_items[[9]] - 1) +
      (5 - sus_items[[2]]) +
      (5 - sus_items[[4]]) +
      (5 - sus_items[[6]]) +
      (5 - sus_items[[8]]) +
      (5 - sus_items[[10]])
    ) * 2.5
  }

  aoa_item_candidates <- list(
    c("aoa1", "aoa_1"), c("aoa2", "aoa_2"), c("aoa3", "aoa_3"),
    c("aoa4", "aoa_4"), c("aoa5", "aoa_5"), c("aoa6", "aoa_6"),
    c("aoa7", "aoa_7"), c("aoa8", "aoa_8"), c("aoa9", "aoa_9")
  )
  aoa_items <- map(aoa_item_candidates, ~ resolve_mapped_series(out, .x))
  if (all(map_lgl(aoa_items, ~ !is.null(.x)))) {
    out$aoa_usefulness <- (
      (3 - aoa_items[[1]]) +
      (-3 + aoa_items[[3]]) +
      (3 - aoa_items[[5]]) +
      (3 - aoa_items[[7]]) +
      (3 - aoa_items[[9]])
    ) / 5.0

    out$aoa_satisfying <- (
      (3 - aoa_items[[2]]) +
      (3 - aoa_items[[4]]) +
      (-3 + aoa_items[[6]]) +
      (-3 + aoa_items[[8]])
    ) / 4.0
  }

  out
}

args <- commandArgs(trailingOnly = TRUE)
input_dir <- ifelse(length(args) >= 1, args[[1]], "data/processed/multi_study_examples")
output_dir <- ifelse(length(args) >= 2, args[[2]], "analyses/output_r")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_study <- function(path) {
  ext <- tools::file_ext(path)
  if (ext == "csv") {
    read_csv(path, show_col_types = FALSE)
  } else {
    read_excel(path)
  }
}

files <- list.files(input_dir, pattern = "\\.(csv|xlsx)$", full.names = TRUE, recursive = TRUE)
if (length(files) == 0) {
  stop(paste("No CSV/XLSX files found in", input_dir))
}

studies <- set_names(map(files, ~ add_derived_scale_scores(read_study(.x))), nm = tools::file_path_sans_ext(basename(files)))

numeric_dvs <- function(df) {
  names(df)[sapply(df, is.numeric) & !(tolower(names(df)) %in% id_like)]
}

# 1) DV overlap matrix
sets <- map(studies, ~ unique(numeric_dvs(.x)))
study_names <- names(studies)
overlap <- matrix(NA_real_, nrow = length(studies), ncol = length(studies), dimnames = list(study_names, study_names))
for (a in study_names) {
  for (b in study_names) {
    union_set <- union(sets[[a]], sets[[b]])
    overlap[a, b] <- ifelse(length(union_set) == 0, NA_real_, length(intersect(sets[[a]], sets[[b]])) / length(union_set))
  }
}
write.csv(overlap, file.path(output_dir, "dv_overlap_matrix.csv"), row.names = TRUE)

# 2) Harmonized DV summary and mean shift without IVs
harmonized <- imap_dfr(studies, ~ {
  dvs <- numeric_dvs(.x)
  map_dfr(dvs, function(dv) {
    vals <- .x[[dv]]
    tibble(
      study = .y,
      dv = dv,
      n = sum(!is.na(vals)),
      mean = mean(vals, na.rm = TRUE),
      sd = sd(vals, na.rm = TRUE)
    )
  })
})

harmonized <- harmonized %>%
  group_by(dv) %>%
  mutate(
    global_mean = mean(mean, na.rm = TRUE),
    pooled_sd = sqrt(mean(sd^2, na.rm = TRUE)),
    mean_z_vs_global = (mean - global_mean) / pooled_sd
  ) %>%
  ungroup() %>%
  select(-global_mean, -pooled_sd)

write.csv(harmonized, file.path(output_dir, "harmonized_dv_summary.csv"), row.names = FALSE)

# 3) Cross-study composite index from DVs shared by all studies
dv_counts <- table(unlist(sets))
common_dvs <- names(dv_counts[dv_counts >= 2])
if (length(common_dvs) < 2) {
  stop("Need at least two DVs shared by at least two studies for PCA composite index.")
}

stacked <- bind_rows(lapply(names(studies), function(study_name) {
  studies[[study_name]] %>%
    mutate(study = study_name) %>%
    select(any_of(common_dvs), study)
}))

z <- stacked %>%
  group_by(study) %>%
  mutate(across(all_of(common_dvs), ~ {
    x <- .x
    x[is.na(x)] <- median(x, na.rm = TRUE)
    s <- sd(x, na.rm = TRUE)
    ifelse(is.na(s) | s == 0, 0, (x - mean(x, na.rm = TRUE)) / s)
  })) %>%
  ungroup()

pca <- prcomp(z %>% select(all_of(common_dvs)), center = FALSE, scale. = FALSE)
z$cross_study_composite <- pca$x[, 1]

composite <- z %>%
  group_by(study) %>%
  summarise(
    n = n(),
    mean = mean(cross_study_composite, na.rm = TRUE),
    sd = sd(cross_study_composite, na.rm = TRUE),
    explained_variance_ratio = summary(pca)$importance[2, 1],
    .groups = "drop"
  )

write.csv(composite, file.path(output_dir, "cross_study_composite_summary.csv"), row.names = FALSE)

# Plots
overlap_df <- as.data.frame(as.table(overlap))
colnames(overlap_df) <- c("study_a", "study_b", "jaccard")

p1 <- ggplot(overlap_df, aes(study_a, study_b, fill = jaccard)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", jaccard)), color = "black") +
  scale_fill_gradient(low = "#d6eaf8", high = "#154360", limits = c(0, 1)) +
  labs(title = "DV overlap across standardized studies (Jaccard)", x = NULL, y = NULL) +
  theme_minimal()

ggsave(file.path(output_dir, "dv_overlap_heatmap.png"), p1, width = 7, height = 5, dpi = 150)

shared_dvs <- harmonized %>% count(dv, name = "k") %>% filter(k >= 2) %>% pull(dv)
if (length(shared_dvs) > 0) {
  p2 <- harmonized %>%
    filter(dv %in% shared_dvs) %>%
    ggplot(aes(dv, mean_z_vs_global, fill = study)) +
    geom_col(position = "dodge") +
    geom_hline(yintercept = 0, color = "black") +
    labs(
      title = "Comparable DV-level differences without using IVs",
      x = NULL,
      y = "Study mean shift (z vs global DV mean)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))

  ggsave(file.path(output_dir, "dv_mean_shift.png"), p2, width = 9, height = 5, dpi = 150)
}

cat("Loaded studies:", paste(names(studies), collapse = ", "), "\n\n")
cat("DV overlap matrix:\n")
print(round(overlap, 2))
cat("\nTop harmonized summaries:\n")
print(head(harmonized, 12))
cat("\nComposite index by study:\n")
print(composite)
