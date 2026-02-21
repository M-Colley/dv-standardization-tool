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

files <- list.files(input_dir, pattern = "\\.(csv|xlsx)$", full.names = TRUE)
if (length(files) == 0) {
  stop(paste("No CSV/XLSX files found in", input_dir))
}

studies <- set_names(map(files, read_study), nm = tools::file_path_sans_ext(basename(files)))

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
