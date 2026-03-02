# Spatial Transcriptomics Alignment

Alignment of scRNA-seq data with high-res spatial transcriptomics data.

## Overview

This repository contains code for scRNA-seq to spatial transcriptomics alignment.
The task consists of inferring gene expression at spatially resolved locations by mapping scRNA-seq reference data to spatial transcriptomics (ST) spots.

It consists of two main components:

- an evaluation framework with metrics for four objectives in `src/metrics`.
- a noval alignment method in `src/alternative_idea` that does not learn a cell-to-spot mapping directly, but instead also learns a cell-to-cell-state mapping, effectively mapping those cell-states to spots.

## Repository Structure

```
src/
├── aligners/           # Wrappers for baseline methods (Tangram, TACCO, DOT, CytoSPACE)
├── alternative_idea/   # Novel alignment method
├── metrics/            # Evaluation metrics for O1, O2, O4
├── data_preparation/   # Dataset preprocessing scripts
├── utils/              # Shared I/O utilities
└── run_experiment.py   # Experiment orchestrator (grid search over hyperparameters)
```

## Dataset Format

Each dataset folder must contain the following files (see `sample_dataset/` for a reference example).

### scRNA data

**`scData_Cells.csv`** — One row per cell with cell type labels (if available).
```
cellID,cellType,cellTypeMinor
CID4290A_ACGCAGCCACCACGTG,Endothelial,Endothelial ACKR1
CID4290A_ACGCAGCCACCACGTG,N/A,N/A
```

**`scData_Genes.csv`** — One gene ID per row.
```
geneID
BRCA1
TP53
```

**`scData_GEP.csv`** — Full gene expression matrix, rows: genes, columns: cells. Gene and cell order must match `scData_Genes.csv` and `scData_Cells.csv` respectively.

### Spatial data

**`stData_Spots.csv`** — One row per spot with 2D array coordinates.
```
spotID,cArray0,cArray1
TTAATGTAGACCAGGT-1,0,0
```

**`stData_Genes.csv`** — One gene ID per row (same format as `scData_Genes.csv`).

**`stData_GEP.csv`** — Full gene expression matrix, rows: genes, columns: spots. Gene and spot order must match `stData_Genes.csv` and `stData_Spots.csv` respectively.

> Note: Of course, the xxx_Genes.csv files do not provide any additional information, they are only there for convenience.
> They are also not used in the code, since the genes are also present in the GEP files.

## Installation & Running

Since every alignment method has its own dependencies and requirements,
different conda environments have to be created for each method.

### AlternativeIdea

```bash
conda env create -f alternative_idea/environment.yml
conda activate alternative_idea_env
```

### Tangram

```bash
conda env create -f src/aligners/environment_tangram.yml
conda activate tangram_env
```

### TACCO

```bash
conda env create -f src/aligners/environment_tacco.yml
conda activate tacco_env
```

### DOT

DOTr is not available on conda-forge and must be installed from GitHub after creating the environment:

```bash
conda env create -f src/aligners/environment_dot.yml
conda activate dot_env
Rscript -e "devtools::install_github('saezlab/DOT')"
```

## Usage

All commands are run from the `Code/` directory.

### Running a Baseline Aligner

```bash
conda activate tangram_env  # or tacco_env / dot_env
python -m src.aligners.run_tangram \
  -d <dataset_folder> \
  -o <output_path.csv>
```

Each aligner script follows the same pattern (`run_tacco.py`, `run_dot.py`).
For more detailed arguments and options, please refer to the respective script.

Example using `sample_dataset/`:

```bash
conda activate tangram_env
python -m src.aligners.run_tangram \
  -d sample_dataset \
  -o sample_dataset/results/tangram_GEP.csv

conda activate tacco_env
python -m src.aligners.run_tacco \
  -d sample_dataset \
  -o sample_dataset/results/tacco_GEP.csv

conda activate dot_env
python -m src.aligners.run_dot \
  -d sample_dataset \
  -o sample_dataset/results/dot_GEP.csv
```

### Computing Metrics

```bash
python -m src.metrics.run_all_metrics \
  -d <dataset_folder> \
  -r <result_gep.csv> \
  -m <metrics_output_folder>
```

Example using `sample_dataset/results/tangram_GEP.csv`:

```bash
python -m src.metrics.run_all_metrics \
  -d sample_dataset \
  -r sample_dataset/results/tangram_GEP.csv \
  -m sample_dataset/metrics
```

### Running our novel method

```bash
conda activate alternative_idea_env

python -m run_experiment \
  -d <dataset_folder> \
  -c experiment_config.yaml \
  -o <result_folder> \
  -m <metrics_folder> \
  [--save_result] \
  [--run_permutation_tests] \
  [--logging verbose]
```

The experiment config supports grid search: list values become search axes and the runner executes every combination.
Results are written to numbered subdirectories with a `summary.csv` tracking status and loss values.
For more detailed arguments and options, please refer to the script.

Example using `sample_dataset`:

```bash
conda activate alternative_idea_env
python -m run_experiment \
  -d sample_dataset \
  -c experiment_config.yaml \
  -o sample_dataset/results \
  -m sample_dataset/metrics
```

## TODO

- Tests
- Build installable package when ready (maybe extract only our alignment method code for that? or also with evaluation?)
