# MOSANIC Developer Guide

> **MOSANIC**: Multi-mOdal Self-Attention Network for Intercellular Communication
>
> Internal reference for extending, debugging, and maintaining the package.

---

## Package Structure

```
MOSANIC/
├── mosanic/                         # Main Python package
│   ├── __init__.py                 # v1.0.0, exports MOSANIC class + build_model
│   ├── cli.py                      # Entry point: mosanic preprocess|train|evaluate|run
│   │
│   ├── models/                     # Neural network architecture
│   │   ├── encoder.py              # HetGTEncoder — heterogeneous graph transformer
│   │   ├── decoder.py              # ExpressionDecoder — MLP for gene expression
│   │   └── mosanic_model.py         # MOSANIC class (encoder + decoder) + build_model()
│   │
│   ├── data/                       # Data loading & preprocessing
│   │   ├── preprocessor.py         # 17-step pipeline (THE main preprocessing script)
│   │   ├── anndata_loader.py       # Load h5ad, extract scVI, spatial coords, QC
│   │   ├── lr_database.py          # Parse LR database (CellNEST / NicheNet)
│   │   ├── metabolite.py           # scFEA flux → ChemBERTa mapping
│   │   ├── spatial_graph.py        # k-NN spatial graph from coordinates
│   │   ├── spatial_cv.py           # Spatial cross-validation splits
│   │   ├── channel_classifier.py   # Classify LR pairs → contact/secreted/ECM
│   │   └── technology.py           # Detect technology (Visium/Xenium/MERFISH/Slide-seq)
│   │
│   ├── graph/                      # Heterogeneous graph construction
│   │   ├── assembler.py            # GraphAssembler → PyG HeteroData
│   │   ├── edge_builder.py         # Build all 7 edge types (EdgeBuilder class)
│   │   ├── node_features.py        # Load ESM-2 + ChemBERTa embeddings (NodeFeatureBuilder)
│   │   ├── typed_edge_builder.py   # Contact/secreted/metabolite edge builders
│   │   └── intracellular_edge_builder.py  # τ₃ self-loop edges (receptor PCA + flux)
│   │
│   ├── training/                   # Model training
│   │   ├── trainer.py              # MOSANICTrainer — full training loop
│   │   ├── losses.py               # MOSANICLoss — Huber + optional CCC auxiliary
│   │   └── callbacks.py            # EarlyStopping, ModelCheckpoint, TensorBoard
│   │
│   ├── evaluation/                 # Model evaluation & CCC scoring
│   │   ├── evaluator.py            # Full eval pipeline (L1/L2/L3 metrics)
│   │   ├── ccc_extractor.py        # CCCExtractor — 13 LR scoring variants
│   │   └── metrics.py              # R², AUROC, DES, ARI/NMI functions
│   │
│   ├── analysis/                   # Downstream biological analysis
│   │   ├── relay.py                # Multi-hop relay detection
│   │   └── knockout.py             # In silico perturbation / edge ablation
│   │
│   ├── configs/                    # YAML configuration files
│   │   ├── default.yaml            # Base defaults (model, training, eval)
│   │   ├── visium.yaml             # Visium spatial params
│   │   ├── xenium.yaml             # Xenium spatial params (smaller model)
│   │   ├── merfish.yaml            # MERFISH spatial params
│   │   ├── slideseq.yaml           # Slide-seqV2 spatial params
│   │   ├── organisms/
│   │   │   ├── human.yaml          # Human databases (CellNEST, M_R)
│   │   │   └── mouse.yaml          # Mouse databases (NicheNet, M_R_mouse)
│   │   └── examples/
│   │       └── breast_cancer_visium.yaml   # Complete example config
│   │
│   ├── databases/                  # Bundled LR + metabolite databases
│   │   ├── CellNEST_database.csv   # 14,909 human LR pairs (training DB)
│   │   ├── M_R.txt                 # Human metabolite-receptor pairs
│   │   ├── LR_database_mouse.csv   # Mouse LR pairs
│   │   └── M_R_mouse.txt           # Mouse metabolite-receptor pairs
│   │
│   └── external/scfea/             # Bundled scFEA for metabolite flux estimation
│
├── tests/
│   └── test_full_pipeline.py       # End-to-end test with real breast data
│
├── checkpoints/                    # Trained model weights (created by training)
├── data/processed/                 # Preprocessed graphs (created by preprocessing)
├── run_mosanic.py                   # Top-level runner script
├── setup.py                        # pip install -e .
├── pyproject.toml                  # Package metadata
└── DEV_README.md                   # THIS FILE
```

---

## Data Requirements

### Input h5ad format

MOSANIC expects an AnnData `.h5ad` file with:

| Slot | Shape | Description | Required? |
|------|-------|-------------|-----------|
| `.X` | [N, G] | Log-normalized expression (dense or sparse) | Yes |
| `.layers["raw_count"]` | [N, G] | Raw UMI counts | Yes (for label generation) |
| `.obsm["X_scvi"]` | [N, 128] | Pre-trained scVI latent embeddings | Yes |
| `.obsm["spatial"]` | [N, 2] | Spatial coordinates (pixels) | Yes |
| `.obs["cell_type"]` or `.obs["leiden"]` | [N] | Cell type labels | Optional (for ARI/NMI eval) |
| `.var_names` | [G] | Gene symbols | Yes |

### How to prepare scVI embeddings

```python
import scvi
adata = sc.read_h5ad("your_data.h5ad")
scvi.model.SCVI.setup_anndata(adata)
model = scvi.model.SCVI(adata, n_latent=128)
model.train(max_epochs=200)
adata.obsm["X_scvi"] = model.get_latent_representation()
adata.save("your_data_with_scvi.h5ad")
```

### Pre-computed embeddings

ESM-2 protein embeddings (1,280-dim) and ChemBERTa metabolite embeddings (600-dim)
must be pre-computed and stored as individual `.npy` files:

```
embeddings/
├── proteins/
│   ├── BRCA1.npy     # np.array shape (1280,)
│   ├── TP53.npy
│   └── ...           # ~15,000 genes
└── metabolites/
    ├── Cholesterol.npy  # np.array shape (600,)
    ├── Glucose.npy
    └── ...              # ~400 metabolites
```

Missing genes/metabolites fall back to zero vectors with a warning.

### scFEA flux

Run scFEA separately to produce a flux balance CSV:
```
scfea_balance_<dataset>.csv
  index: cell IDs (matching h5ad .obs_names)
  columns: metabolic module names
  values: predicted flux values
```

---

## Config System

### Path resolution

All paths in the config are resolved relative to `Path(root).parent`:

```
root: MOSANIC                        # → /path/to/CCC/MOSANIC
root.parent: /path/to/CCC           # base for path resolution
raw_adata: src5/data/raw/x.h5ad    # → /path/to/CCC/src5/data/raw/x.h5ad
```

### Creating a config for a new dataset

1. Copy `mosanic/configs/examples/breast_cancer_visium.yaml`
2. Update `dataset`, `technology`, `organism`
3. Update all paths under `paths:`
4. Adjust spatial thresholds if needed (or inherit from technology configs)
5. Adjust `model.hidden_dim` for very large datasets (use 128 for >50K cells)

### Technology-specific defaults

| Technology | k_neighbors | max_distance_um | contact_um | secreted_um | hidden_dim |
|-----------|------------|----------------|-----------|------------|-----------|
| Visium | 6 | 150 | 87 | 150 | 256 |
| Xenium | 10 | 50 | 20 | 50 | 128 |
| MERFISH | 6 | 100 | 15 | 100 | 256 |
| Slide-seqV2 | 6 | 50 | 15 | 50 | 256 |

---

## Architecture Reference

### Model: MOSANIC (mosanic/models/mosanic_model.py)

```
HeteroData (3 node types, 7 edge types)
        ↓
    HetGTEncoder                          # mosanic/models/encoder.py
    ├── Input projections (per node type)
    │   cell:       Linear(128→256) + LN + GELU
    │   gene:       Linear(1280→256) + LN + GELU
    │   metabolite: Linear(600→256) + LN + GELU
    ├── L=2 × HetGTBlock
    │   ├── 7× TransformerConv (per edge type, 4 heads × 64d)
    │   ├── Gate-weighted aggregation (softmax over edge types per dst)
    │   ├── Residual + LayerNorm
    │   └── FFN(256→1024→256) + Residual + LayerNorm
    └── Output: cell_embeddings [N, 256]
        ↓
    ExpressionDecoder                     # mosanic/models/decoder.py
    └── MLP: 256 → 256 → 200 genes
        ↓
    expression [N, 200]  ← Huber loss vs observed log1p expression
```

### Edge types (7 total)

| Edge | Relation | Src→Dst | Attr dim | Purpose |
|------|----------|---------|----------|---------|
| τ₁ | secreted | cell→cell | 2 | Paracrine LR signaling |
| τ₂ | metabolite | cell→cell | 3 | Metabolite-mediated CCC |
| τ₃ | intracellular | cell→cell (self) | 102 | Receptor PCA + flux |
| ε₁ | expresses | cell→gene | 3 | Expression links |
| **ε₂** | **interacts** | **gene→gene** | **3** | **LR pairs (CCC signal)** |
| ε₃ | flux | cell→metabolite | 2 | Metabolic flux |
| ε₄ | sensed_by | metabolite→gene | varies | MR binding |

**ε₂ is critical**: Ablation shows removing it drops AUROC from 0.740 → 0.500.
Everything else has < 0.003 AUROC impact.

### CCC scoring

```python
# Raw LR pair score (from attention)
s(L, R) = mean over heads and layers of α_{LR}^{(ε₂, h, l)}

# Per-cell communication intensity
I_cell(L, R) = s(L, R) × max(expr_L, 0) × max(expr_R, 0)
```

---

## How to Add New Functionality

### Adding a new edge type

1. **Define builder** in `mosanic/graph/edge_builder.py`:
   ```python
   def _step_new_edge(self, ...):
       # Return (edge_index [2, E], edge_attr [E, d])
   ```

2. **Register in assembler** (`mosanic/graph/assembler.py`):
   ```python
   data["src_type", "relation", "dst_type"].edge_index = new_ei
   data["src_type", "relation", "dst_type"].edge_attr = new_ea
   ```

3. **Add edge_dim to encoder defaults** (`mosanic/models/encoder.py`):
   ```python
   DEFAULT_EDGE_TYPE_DIMS[("src", "relation", "dst")] = attr_dim
   ```

4. **Update metadata** in assembler for `build_model()` auto-detection.

### Adding a new node type

1. Add input projection dim to `DEFAULT_NODE_IN_DIMS` in `encoder.py`
2. Add node features in `node_features.py`
3. Register in assembler
4. Define edges connecting the new node type

### Adding a new evaluation metric

1. Add function to `mosanic/evaluation/metrics.py`
2. Call it from `mosanic/evaluation/evaluator.py`
3. Store result in `ccc_eval_results.json`

### Adding a new downstream analysis

1. Create `mosanic/analysis/new_analysis.py`
2. Export from `mosanic/analysis/__init__.py`
3. Optionally add a CLI subcommand in `mosanic/cli.py`

### Adding a new technology

1. Create `mosanic/configs/new_tech.yaml` with spatial parameters
2. Add detection logic in `mosanic/data/technology.py`
3. Add distance threshold defaults

### Adding a new organism

1. Create `mosanic/configs/organisms/new_org.yaml`
2. Add LR database to `mosanic/databases/`
3. Add metabolite database if available

---

## Key Files Quick Reference

| What you want to do | File to edit |
|---------------------|-------------|
| Change model architecture | `mosanic/models/encoder.py` |
| Change loss function | `mosanic/training/losses.py` |
| Change training loop | `mosanic/training/trainer.py` |
| Change graph construction | `mosanic/graph/edge_builder.py` |
| Add preprocessing step | `mosanic/data/preprocessor.py` |
| Change CCC scoring | `mosanic/evaluation/ccc_extractor.py` |
| Add evaluation metric | `mosanic/evaluation/metrics.py` |
| Add CLI command | `mosanic/cli.py` |
| Add database | `mosanic/databases/` + organism config |

---

## Output Files (What a Biologist Gets)

After running the full pipeline, MOSANIC produces these user-facing outputs:

```
output/<dataset>/
├── lr_pair_rankings.csv              # PRIMARY: All LR pairs ranked, 5 scoring variants
├── top50_lr_pairs.txt                # Quick glance: top 50 pairs (human-readable)
├── lr_scores_all_variants.json       # Machine-readable: raw/filtered/enhanced/last_layer/cosine
├── comm_matrix_secreted.csv          # Cell-type × cell-type communication (secreted channel)
├── comm_matrix_metabolite.csv        # Cell-type × cell-type communication (metabolite channel)
├── comm_matrix_intracellular.csv     # Cell-type × cell-type communication (intracellular)
├── cell_embeddings.npy               # [N, 256] for UMAP/clustering
├── mosanic_results.json               # Summary: AUROC, ARI, NMI, training metrics
└── spatial_communication_intensity.csv  # Per-cell LR intensity for spatial visualization
```

### Scoring Variants Explained

| Variant | Method | Best For |
|---------|--------|----------|
| `raw` | Mean attention over heads × layers | Baseline ranking |
| `filtered` | Raw + homodimer removal + expr tie-breaking | Publication reporting |
| `enhanced` | Attention × log(receptor_degree) × √(expr_L × expr_R) | Fixing degree-1 saturation |
| `last_layer` | Final layer attention only (full context) | Highest AUROC typically |
| `cosine` | Cosine similarity of learned gene embeddings | Avoids softmax bias entirely |

### Verified Performance (breast_new, 500 epochs)

```
AUROC (vs OmniPath):
  raw=0.736, filtered=0.738, last_layer=0.738, cosine=0.680
Clustering: ARI=0.334, NMI=0.496
Expression: val R²=0.430
Training: 376 epochs, 145s on A100
```

---

## Testing

### End-to-end pipeline test (500 epochs, ~2.5 min)
```bash
cd MOSANIC
python tests/test_end_to_end.py --device cuda --epochs 500
```

### Quick smoke test (10 epochs, ~8 seconds)
```bash
python tests/test_full_pipeline.py --quick --device cuda
```

### Import-only test
```bash
python -c "from mosanic import MOSANIC, build_model; print('OK')"
```

---

## Naming Map (src5 → MOSANIC)

| src5 name | MOSANIC name | File |
|-----------|-------------|------|
| `HetGT5CCC` | `MOSANIC` | `models/mosanic_model.py` |
| `MultiTypeHetGTEncoder` | `HetGTEncoder` | `models/encoder.py` |
| `MultiTypeHetGTBlock` | `HetGTBlock` | `models/encoder.py` |
| `ExpressionDecoder` | `ExpressionDecoder` | `models/decoder.py` (unchanged) |
| `HetGT5Trainer` | `MOSANICTrainer` | `training/trainer.py` |
| `HetGTCCCLoss` | `MOSANICLoss` | `training/losses.py` |
| `HeteroGraphAssembler` | `GraphAssembler` | `graph/assembler.py` |
| `CCCExtractor` | `CCCExtractor` | `evaluation/ccc_extractor.py` (unchanged) |
| `preprocess.py` | `data/preprocessor.py` | — |
| `train.py` | `cli.py train` | — |
| `run_ccc_eval.py` | `evaluation/evaluator.py` | — |

---

## Dependencies

Core:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.4
- scanpy, anndata, scvi-tools (for scVI)
- numpy, scipy, pandas, scikit-learn
- pyyaml, matplotlib

Optional:
- tensorboard (for training visualization)
- scFEA (bundled in mosanic/external/scfea/)

Install: `pip install -e .` from the MOSANIC/ directory.
