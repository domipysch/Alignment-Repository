# README

This repository contains all used code for the Master Project WS25/26.

- `00_Playground`: A playground for testing and experimenting with code snippets.
- `01_Data_Prep`: Code used for preparation of downloaded datasets.
- `02_Aligners`: Code to run existing alignment methods.
- `03_Metrics`: Code to compute evaluation metrics for alignments.

In general, the workflow followed these steps:

1) Data Preparation: Downloading and preprocessing datasets to ensure they are in the same format.
2) Alignment: Running various alignment algorithms on the prepared datasets in several settings (prob. vs det. mapping, cell state granularity, etc.).
3) Evaluation: Computing metrics to evaluate the quality of the alignments produced by different methods.

When running the alignment methods, the results got written to the `results_xxx/` folder
within each datasets directory.
Evaluation metrics were then computed based on these results and stored in the `metrics_xxx/` folder
within each datasets directory.

## Installation & Running

Since every alignment method has its own dependencies and requirements,
different conda environments were created for each method.

### AlternativeIdea

```bash
conda env create -f alternative_idea/environment.yml
conda activate alternative-idea
```

### Tangram

```bash
conda env create -f aligners/environment_tangram.yml
conda activate tangram_env
```

### TACCO

```bash
conda env create -f aligners/environment_tacco.yml
conda activate tacco_env
```

### DOT

DOTr is not available on conda-forge and must be installed from GitHub after creating the environment:

```bash
conda env create -f aligners/environment_dot.yml
conda activate dot_env
Rscript -e "devtools::install_github('saezlab/DOT')"
```

## TODO

- Tests
- Build installable package when ready (maybe extract only our alignment method code for that? or also with evaluation?)
