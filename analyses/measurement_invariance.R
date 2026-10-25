# Measurement invariance of one scale across studies (lavaan multi-group CFA).
#
# Called by analyses/measurement_invariance.py:
#   Rscript measurement_invariance.R <items.csv> <spec.json> <out.json>
#
# items.csv: one row per observation with columns `group` (study), `cluster`
# (participant; observations of one participant are not independent) and the
# item columns named in spec.json ({"scale": ..., "items": [...]}).
#
# One-factor CFA fitted per study at three nested levels (Meredith, 1993):
#   configural  same factor structure, all parameters free per study
#   metric      loadings equal across studies
#   scalar      loadings and intercepts equal across studies
# Estimator MLR with cluster-robust standard errors and test statistics
# (Yuan-Bentler), so repeated answers of one participant are handled.
# The Python caller applies the decision rules; this script only fits.

suppressPackageStartupMessages({
  library(lavaan)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1], check.names = FALSE)
spec <- fromJSON(args[2])
items <- spec$items

model <- paste0("F =~ ", paste(items, collapse = " + "))  # items are safe names x1..xk
levels <- list(configural = character(0), metric = "loadings", scalar = c("loadings", "intercepts"))
measures <- c("chisq.scaled", "df.scaled", "pvalue.scaled", "cfi.robust", "tli.robust",
              "rmsea.robust", "srmr", "aic", "bic")

fits <- list()
out <- list(scale = spec$scale, lavaan = as.character(packageVersion("lavaan")),
            n_obs = nrow(data), groups = as.list(table(data$group)),
            clusters = as.list(tapply(data$cluster, data$group, function(x) length(unique(x)))),
            levels = list())
for (level in names(levels)) {
  fit <- tryCatch(
    cfa(model, data = data, group = "group", cluster = "cluster", estimator = "MLR",
        group.equal = levels[[level]], std.lv = FALSE),
    error = function(e) e, warning = function(w) {
      # keep the fit but record the warning (e.g. negative variances)
      suppressWarnings(fit2 <- cfa(model, data = data, group = "group", cluster = "cluster",
                                   estimator = "MLR", group.equal = levels[[level]]))
      attr(fit2, "warning") <- conditionMessage(w)
      fit2
    })
  if (inherits(fit, "error")) {
    out$levels[[level]] <- list(error = conditionMessage(fit))
    next
  }
  fm <- fitMeasures(fit, measures)
  std <- standardizedSolution(fit)
  loadings <- std[std$op == "=~", c("group", "rhs", "est.std")]
  loadings$group <- lavInspect(fit, "group.label")[loadings$group]
  names(loadings) <- c("group", "item", "std_loading")
  entry <- c(as.list(round(fm, 6)),
             list(converged = lavInspect(fit, "converged"),
                  warning = if (is.null(attr(fit, "warning"))) "" else attr(fit, "warning"),
                  std_loadings = loadings))
  out$levels[[level]] <- entry
  fits[[level]] <- fit
}

# Scaled chi-square difference tests between adjacent levels (Satorra-Bentler 2001).
for (pair in list(c("configural", "metric"), c("metric", "scalar"))) {
  if (!is.null(fits[[pair[1]]]) && !is.null(fits[[pair[2]]])) {
    test <- tryCatch(lavTestLRT(fits[[pair[1]]], fits[[pair[2]]], method = "satorra.bentler.2001"),
                     error = function(e) NULL)
    if (!is.null(test)) {
      out$levels[[pair[2]]]$lrt_chisq_diff <- test[2, "Chisq diff"]
      out$levels[[pair[2]]]$lrt_df_diff <- test[2, "Df diff"]
      out$levels[[pair[2]]]$lrt_pvalue <- test[2, "Pr(>Chisq)"]
    }
  }
}

write(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), args[3])
