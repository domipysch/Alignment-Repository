#!/usr/bin/env Rscript

# Run DOT alignment on a prepared dataset in the given folder.
# Saves predicted gene expression per spot (GEP) as CSV to output_path.

# Usage:
# Rscript run_dot_export.R /path/to/dataset_folder [LSO|HSO] [deterministic-mapping|probabilistic-mapping] <cellTypeKey> <output_path>
# - First arg: dataset_folder (required)
# - Second arg: "LSO" for low-resolution or "HSO" for high-resolution (required)
# - Third arg: mapping mode (required): either "deterministic-mapping" or "probabilistic-mapping"
# - Fourth arg: cellTypeKey (required): column name in scData_Cells.csv containing the cell type labels
# - Fifth arg: output_path (required): explicit output path for the CSV

library(DOTr)
library(utils)
library(stats)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Please supply five arguments: dataset_folder, mode (LSO|HSO), mapping_mode ('deterministic-mapping'|'probabilistic-mapping'), cellTypeKey, and output_path.")
}
dataset_folder <- args[1]
mode <- toupper(args[2])
mapping_mode <- args[3]
cellTypeKey <- args[4]
output_path <- args[5]

if (! mode %in% c("LSO", "HSO")) {
  stop(sprintf("Invalid mode '%s'. Use 'LSO' (low-resolution) or 'HSO' (high-resolution).", mode))
}

if (! mapping_mode %in% c("deterministic-mapping", "probabilistic-mapping")) {
  stop(sprintf("Invalid mapping_mode '%s'. Use 'deterministic-mapping' or 'probabilistic-mapping'.", mapping_mode))
}

# ---- Load input data ----
ref_counts <- read.csv(file.path(dataset_folder, "scData_GEP.csv"),
                       header = TRUE, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
ref_meta   <- read.csv(file.path(dataset_folder, "scData_Cells.csv"),
                       header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)

srt_counts <- read.csv(file.path(dataset_folder, "stData_GEP.csv"),
                       header = TRUE, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
srt_coords <- read.csv(file.path(dataset_folder, "stData_Spots.csv"),
                       header = TRUE, row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

# Verify that the provided cellTypeKey exists in the metadata
if (! cellTypeKey %in% colnames(ref_meta)) {
  stop(sprintf("cellTypeKey '%s' not found in scData_Cells.csv. Available columns: %s",
               cellTypeKey, paste(colnames(ref_meta), collapse = ", ")))
}

# ---- Build reference centroids per cell type (NO gene filtering here) ----
# ref_counts: genes x cells  => want to compute average expression for each cell type

# transpose to cells x genes for easier aggregation
cells_by_gene <- t(as.matrix(ref_counts))  # cells x genes
cell_types <- as.character(ref_meta[[cellTypeKey]])
celltype_levels <- unique(cell_types)
ref_centroids_list <- lapply(celltype_levels, function(ct) {
  idx <- which(cell_types == ct)
  if (length(idx) == 1) {
    # single cell: return its vector
    as.numeric(cells_by_gene[idx, , drop = TRUE])
  } else {
    colMeans(cells_by_gene[idx, , drop = FALSE])
  }
})
ref_centroids <- do.call(rbind, ref_centroids_list) # celltype x gene
rownames(ref_centroids) <- as.character(celltype_levels)
colnames(ref_centroids) <- colnames(cells_by_gene)
dim(ref_centroids)

# ---- Create DOT object and run decomposition ----
dot.srt <- setup.srt(srt_data = srt_counts, srt_coords = srt_coords)
dot.ref <- setup.ref(ref_data = ref_counts, ref_annotations = cell_types, 1)  # (..., 10)?
dot <- create.DOT(dot.srt, dot.ref)

# Run selected resolution
if (mode == "HSO") {
  cat("Running DOT in high-resolution mode (HSO)...\n")
  dot <- run.DOT.highresolution(dot)

  # If requested, convert probabilistic weights to deterministic (one-hot per spot)
  if (!is.null(mapping_mode) && identical(mapping_mode, "deterministic-mapping")) {
    cat("Applying deterministic mapping: converting dot@weights to one-hot per spot...\n")
    # ensure numeric matrix
    weights_mat <- as.matrix(dot@weights)
    # compute finite-value mask per entry
    finite_mask <- is.finite(weights_mat)
    # replace non-finite entries with -Inf for safe argmax computation
    weights_repl <- weights_mat
    weights_repl[!finite_mask] <- -Inf
    # find index of max per row (spot). max.col returns column index of max per row.
    # If all entries are -Inf (no finite values), we'll handle below to set row to zeros.
    argmax_idx <- max.col(weights_repl, ties.method = "first")
    # build one-hot matrix
    # one_hot <- matrix(0, nrow = nrow(weights_mat), ncol = ncol(weights_mat))
    one_hot <- matrix(0, nrow = nrow(weights_mat), ncol = ncol(weights_mat),
                  dimnames = dimnames(weights_mat))
    one_hot[cbind(seq_len(nrow(weights_mat)), argmax_idx)] <- 1
    # rows with no finite values -> keep all zeros
    rows_no_finite <- rowSums(finite_mask) == 0
    if (any(rows_no_finite)) {
      one_hot[rows_no_finite, ] <- 0
    }
    # assign back to dot@weights
    dot@weights <- one_hot
    cat("Deterministic mapping applied.\n")

  }

} else {
  cat("Running DOT in low-resolution mode (LSO)...\n")
  dot <- run.DOT.lowresolution(dot, ratios_weight = 0, max_spot_size = 20, verbose = FALSE)
}


# Now compute gene x spot reconstructed expression:
# ref_centroids : celltype x gene
# dot@weights   : spots x celltype
expr_genes_by_spots <- t(ref_centroids) %*% t(dot@weights)  # genes x spots (G x S)

# Convert to numeric matrix and set names
expr_genes_by_spots <- as.matrix(expr_genes_by_spots)
rownames(expr_genes_by_spots) <- colnames(ref_centroids)   # gene names
colnames(expr_genes_by_spots) <- rownames(dot@weights)           # spot IDs
dim(expr_genes_by_spots)

# Determine output file path (use output_path if provided, otherwise default location)
out_file <- if (!is.null(output_path) && nzchar(output_path)) {
  output_path
} else {
  file.path(dataset_folder, "results", "dot_GEP.csv")
}

# Ensure output directory exists
out_dir <- dirname(out_file)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

# ---- Write CSV: first column spot IDs, first row header gene names, top-left "GEP" ----
out_df <- data.frame(GEP = rownames(expr_genes_by_spots),
                     expr_genes_by_spots,
                     check.names = FALSE, stringsAsFactors = FALSE)


numeric_cols <- vapply(out_df, is.numeric, logical(1))
out_df[numeric_cols] <- lapply(out_df[numeric_cols], function(x) round(x, 4))
write.csv(out_df, file = out_file, row.names = FALSE, quote = FALSE)

cat("Done. Output written to:", out_file, "\n")
