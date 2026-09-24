# Run Kyle Butts' R package (https://github.com/kylebutts/jive) on the shared data sets.
#
# Setup (once):
#   install.packages(c("remotes", "dreamerr", "stringmagic"))
#   remotes::install_github("lrberge/fixest")     # the dev version: jive needs sparse_model_matrix
#   remotes::install_github("kylebutts/jive")
#
# Run from the repo root:  Rscript validation/run_r.R
#
# Notes
# - jive() cannot handle a model with no fixed effects at all (it errors inside block_diag_hatvalues), so we add a
#   constant fixed effect `one`. That is equivalent to an intercept, which is what our estimators always include.
# - jive()  = Angrist, Imbens, Krueger (1999) JIVE1 with the controls partialled out  -> compare with our UJIVE1.
# - ujive() = Kolesar (2013) UJIVE                                                    -> no equivalent in this package.
# - ssc = FALSE (the default): no small-sample correction, so the SE is the plain robust sandwich.
suppressMessages(library(jive))

datasets <- c("n300_k10_nocontrols", "n300_k10_controls", "n60_k8_controls")
rows <- list()
for (nm in datasets) {
  d <- read.csv(file.path("validation", "data", paste0(nm, ".csv")))
  d$one <- 1
  k <- sum(grepl("^z[0-9]+$", names(d)))
  controls <- if ("w1" %in% names(d)) "w1 + w2" else "1"
  fml <- as.formula(sprintf("y ~ %s | one | t ~ %s", controls, paste0("z", seq_len(k), collapse = " + ")))
  for (est in c("jive", "ujive")) {
    r <- get(est)(fml, data = d)
    rows[[length(rows) + 1]] <- data.frame(dataset = nm, source = paste0("R ", est, "()"),
                                           beta = unname(r$beta), se = abs(unname(r$se)))
  }
}
out <- do.call(rbind, rows)
write.csv(out, file.path("validation", "results_r.csv"), row.names = FALSE)
print(out)
