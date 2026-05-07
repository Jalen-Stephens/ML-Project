# Node Wars: The Link Awakens

**Auditing Stability in Drug Interaction GNNs**

## Overview

This project builds a heterogeneous biomedical knowledge graph from Stanford SNAP datasets and evaluates whether drug interaction predictions from graph neural networks are stable under small perturbations in learned representations. We compare a structure-only GCN against a multi-relational RGCN and measure prediction sensitivity using Gaussian noise injection and dimensional dropout.

## Hypothesis

Drug interaction predictions produced by GNNs are highly sensitive to small perturbations in learned embeddings — strong predictive accuracy may mask representational instability.

## Datasets

| Dataset | Edges | Description |
|---------|-------|-------------|
| ChCh-Miner | 48,514 | Drug–drug interactions |
| ChG-Miner | 15,138 | Drug–gene (target) interactions |
| DG-AssocMiner | 21,357 | Disease–gene associations |
| DCh-Miner | 466,656 | Disease–drug associations |

All sourced from [Stanford BioSNAP](https://snap.stanford.edu/biodata/).

## Project Structure

```
data/           Raw and processed datasets, train/val/test splits
src/            Python modules (models, training, evaluation, perturbation)
notebooks/      Jupyter notebooks (01–07) in build order
experiments/    Configs, checkpoints, and result logs
blog/           Final index.html and figure assets
```

## Setup

```bash
pip install -r requirements.txt
```

### Reproducibility — single notebook (Colab-ready)

`Node_Wars_Unified.ipynb` at the repo root is the canonical reproducibility
artifact: every figure and number in `blog/index.html` is produced by this one
notebook. Upload it to Google Colab (or open it locally) and click **Runtime →
Run all**. With the cached checkpoints and result JSONs committed to this repo,
it completes in ~2 minutes and renders every blog figure inline. To retrain
all models from scratch, set `RUN_FROM_SCRATCH = True` in the first toggle
cell.

The `notebooks/01_*…07_*.ipynb` files are the original development notebooks
and are kept for historical reference; the unified notebook supersedes them.

## Reproducibility

All experiments use seeds {42, 123, 456}. Configs are stored as YAML in `experiments/configs/`. Results are saved as JSON in `experiments/results/`.
