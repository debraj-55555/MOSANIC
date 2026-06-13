# MOSAIC

**M**ulti-**O**mic **S**patial **A**ttention for **I**ntercellular **C**ommunication — a heterogeneous graph transformer for cell–cell communication (CCC) inference from spatial transcriptomics.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

MOSAIC scores **ligand–receptor (LR)** and **metabolite–receptor (MR)** interactions, identifies **communication hubs**, detects **multi-hop relay chains**, and performs **in-silico knockout** — all as parameter-free read-outs of one trained graph-attention model.

---

## Installation

```bash
git clone https://github.com/<your-org>/MOSAIC.git
cd MOSAIC
pip install -e .
```

This single command installs MOSAIC together with its **bundled scFEA** flux estimator (no separate setup) and every required dependency (PyTorch, PyTorch-Geometric, scanpy, scVI-tools, ESM-2, ChemBERTa, MAGIC).

**System requirements**: Python 3.9+, CUDA-enabled GPU recommended for training (CPU works for inference / score read-outs).

---

## Quick start

```python
from mosaic import run_pipeline

# One call: preprocess → train → score → return analysis object
result = run_pipeline(
    "tissue.h5ad",
    technology="visium",     # "visium" | "xenium" | "merfish" | "slideseq"
    organism="human",        # "human" | "mouse"
    epochs=500,
)

# Ligand-receptor pairs ranked by intensity-weighted attention (paper Fig 2)
result.lr_pairs(top_k=20)

# Spatial activity of a specific LR pair
result.plot_spatial("TGFB1", "TGFBR2")

# Cell-type × cell-type communication matrix (paper Fig 3)
result.communication_matrix(channel="secreted")

# In-silico knockout of a candidate hub (paper Fig 6)
result.knockout_gene("SCARF1")

# Hub-score — paper's central scalar (Methods §4, Eq. 18)
result.hub_scores(top_k=30)                       # canonical: ε₂ + ε₄
result.hub_scores(top_k=30, channels="lr")        # LR-network hub (Fig 6a/b)
result.hub_scores(top_k=30, channels="mr")        # MR hub (Fig 4c/d)

# Hub fan-out: SCARF1-style multiplexer (paper Fig 6h)
result.hub_fanout("SCARF1", top_k=12)

# Benchmark against an independent LR database (paper Fig 2 evaluation)
result.evaluate_against("OmniPath_ligrec.csv")
```

For an end-to-end walkthrough covering all 11 application sections — niches, LR/MR ranking, spatial maps, hub-score, knockout, relay detection, multiplexer fan-out — see [`tutorial.ipynb`](tutorial.ipynb).

---

## Configurations — pick technology + organism, not per-dataset

MOSAIC ships with composable YAML configs in [`mosaic/configs/`](mosaic/configs/). The base [`default.yaml`](mosaic/configs/default.yaml) carries model + training defaults; per-technology and per-organism overrides are layered on top:

```
mosaic/configs/
├── default.yaml              # Base model + training (shared)
├── visium.yaml               # 10x Visium spatial parameters
├── xenium.yaml               # 10x Xenium spatial parameters
├── merfish.yaml              # MERFISH spatial parameters
├── slideseq.yaml             # Slide-seqV2 spatial parameters
└── organisms/
    ├── human.yaml            # Human LR + MR databases
    └── mouse.yaml            # Mouse LR + MR databases
```

You **do not** need a per-dataset config file. Calling `mosaic.run_pipeline(..., technology="visium", organism="human")` automatically composes `default + visium + organisms/human`. If you need to override anything (e.g., `epochs`, `lr`, `k_neighbors`), pass it as a keyword to `run_pipeline()` or use `mosaic.setup()` to write a final composed config to disk before training.

---

## Architecture overview

MOSAIC represents a tissue as a **heterogeneous graph** with three node types (cells, genes, metabolites) and **seven biologically-typed edges**:

**Cell–cell τ edges** (bidirectional, seeded from spatial k-NN):
- **τ₁** *secreted* — paracrine LR signalling within d_sec µm
- **τ₂** *metabolite-mediated* — scFEA flux-coupled cells within d_met µm
- **τ₃** *intracellular* — per-cell self-loop carrying receptor PCA + flux state

**Cross-type ε edges** (directed):
- **ε₁** *cell → gene* — top-K expression
- **ε₂** *gene → gene LR* — canonical ligand–receptor pairs (CellNEST human / NicheNet + CellTalkDB mouse)
- **ε₃** *cell → metabolite* — scFEA flux
- **ε₄** *metabolite → receptor* — MEBOCOST sensing edges

Node features come from frozen foundation models:
- **scVI** 128-d cell embeddings (auto-trained on first run)
- **ESM-2 650M** 1280-d gene embeddings
- **ChemBERTa-77M-MTR** 600-d metabolite embeddings

A two-block heterogeneous graph transformer trained on the single objective of held-out spatial-expression prediction yields attention scores from which every downstream analysis is read off **without additional parameters**.

See [`DEV_README.md`](DEV_README.md) for the full module-by-module reference.

---

## Tutorial coverage

The included [`tutorial.ipynb`](tutorial.ipynb) walks through:

1. Setup, preprocess, train
2. LR pair ranking + receptor-specific top ligands
3. MR pair ranking
4. Spatial communication maps (LR + MR)
5. Cell-type × cell-type communication matrices
6. Niche clustering from learned cell embeddings
7. **Hub-score — canonical + channel-restricted variants** (paper §4)
8. *In-silico* gene knockout + channel importance
9. Relay-chain detection (2-hop + cross-channel)
10. Hub fan-out / multiplexer trace (SCARF1-style)
11. Export tables for downstream analysis

---

## Reproducing the paper

All five datasets, trained checkpoints, evaluation results, supplementary materials and analysis notebooks are deposited at **Zenodo: [DOI: 10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)** (deposit pending; corresponds to the MOSAIC paper submission, 2026).

End-to-end retraining + evaluation for a single dataset:
```bash
mosaic preprocess --config tissue_config.yaml
mosaic train      --config tissue_config.yaml
mosaic evaluate   --config tissue_config.yaml
```

---

## Verbosity

By default the package logs at `WARNING` level — minimal output. Increase verbosity with:

```python
import mosaic
mosaic.set_verbosity("info")     # major lifecycle messages
mosaic.set_verbosity("debug")    # full per-step trace
mosaic.set_verbosity("silent")   # suppress everything
```

---

## Citation

If you use MOSAIC in your research, please cite:

```bibtex
@article{mosaic2026,
  title   = {MOSAIC: Multi-Omic Spatial Attention for Intercellular Communication},
  author  = {<author list>},
  journal = {<journal pending>},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

The trained checkpoints, processed datasets, and evaluation outputs are archived at the Zenodo DOI above.

---

## License

MIT — see [`LICENSE`](LICENSE). Pre-trained model weights inherit the same MIT licence.

The bundled training-time LR catalogues retain their original licences: CellNEST (MIT), NicheNet (Apache-2.0), CellTalkDB (CC-BY-4.0), MEBOCOST (CC-BY-4.0).
