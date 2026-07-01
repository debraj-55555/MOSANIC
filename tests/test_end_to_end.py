#!/usr/bin/env python3
"""
MOSANIC End-to-End Pipeline Test — Breast Cancer (Visium)

Full pipeline from raw data → preprocessing → training → CCC extraction → output.
Produces biologically meaningful outputs: ranked LR pairs, cell-type communication
matrices, per-cell spatial communication maps, evaluation metrics.

This mirrors what CellNEST and other CCC tools produce as user-facing results.

Usage:
    cd /mnt/disk-drive/debraj/cci_proj2/CCC/MOSANIC
    /opt/miniconda/envs/ccc_env/bin/python3 tests/test_end_to_end.py --device cuda

    Quick test (50 epochs):
    /opt/miniconda/envs/ccc_env/bin/python3 tests/test_end_to_end.py --device cuda --epochs 50
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]       # MOSANIC/
PROJECT = ROOT.parent                             # CCC/
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mosanic.e2e")

OUTPUT_DIR = ROOT / "output" / "breast_new"


def banner(msg):
    log.info("\n" + "=" * 72)
    log.info(f"  {msg}")
    log.info("=" * 72)


# =============================================================================
# PHASE 1: DATA LOADING & VALIDATION
# =============================================================================

def phase1_validate_raw_data():
    """Check all raw inputs exist and are readable."""
    banner("PHASE 1: Raw Data Validation")

    raw_adata_path = PROJECT / "src5/data/raw/breast_new.h5ad"
    assert raw_adata_path.exists(), f"Missing: {raw_adata_path}"

    import anndata
    adata = anndata.read_h5ad(raw_adata_path)
    log.info(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    log.info(f"  .X type: {type(adata.X).__name__}, dtype: {adata.X.dtype}")

    # Check scVI embeddings
    assert "X_scvi" in adata.obsm, "Missing .obsm['X_scvi'] — run scVI first"
    scvi_dim = adata.obsm["X_scvi"].shape[1]
    log.info(f"  scVI embeddings: [{adata.n_obs}, {scvi_dim}]")

    # Check spatial coordinates
    assert "spatial" in adata.obsm, "Missing .obsm['spatial']"
    log.info(f"  Spatial coords: [{adata.obsm['spatial'].shape}]")

    # Check cell types
    ct_col = None
    for col in ["cell_type", "leiden", "celltype", "cluster"]:
        if col in adata.obs.columns:
            ct_col = col
            break
    if ct_col:
        n_types = adata.obs[ct_col].nunique()
        log.info(f"  Cell types: {n_types} ({ct_col})")
    else:
        log.warning("  No cell type annotation found")

    # Check LR database
    lr_path = PROJECT / "src5/data/raw/CellNEST_database.csv"
    lr_df = pd.read_csv(lr_path)
    log.info(f"  LR database: {len(lr_df)} pairs")

    # Check scFEA balance
    scfea_path = PROJECT / "src5/data/raw/scfea_balance_breast_new.csv"
    scfea_df = pd.read_csv(scfea_path, index_col=0)
    log.info(f"  scFEA flux: [{scfea_df.shape[0]} cells × {scfea_df.shape[1]} modules]")

    # Check embeddings
    emb_dir = PROJECT / "src5/data/embeddings/proteins"
    n_emb = len(list(emb_dir.glob("*.npy")))
    log.info(f"  ESM-2 embeddings: {n_emb} genes cached")

    met_dir = PROJECT / "src5/data/embeddings/metabolites"
    n_met = len(list(met_dir.glob("*.npy")))
    log.info(f"  ChemBERTa embeddings: {n_met} metabolites cached")

    log.info("  ✓ All raw data validated")
    return adata


# =============================================================================
# PHASE 2: PREPROCESSING (use existing or build from cache)
# =============================================================================

def phase2_preprocess(config_path):
    """Load or build the preprocessed heterogeneous graph."""
    banner("PHASE 2: Preprocessing → Heterogeneous Graph")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    processed_dir = PROJECT / cfg["paths"]["processed_dir"] / dataset
    processed_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed_dir / "hetero_ccc_graph.pt"

    # Copy from src5 if not already present (avoid re-running 17-step pipeline)
    if not graph_path.exists():
        src5_graph = PROJECT / "src5/data/processed" / dataset / "hetero_ccc_graph.pt"
        if src5_graph.exists():
            log.info(f"  Copying processed graph from src5...")
            shutil.copy2(src5_graph, graph_path)
            # Also copy support files
            for fname in ["preprocessing_cache.pt", "metadata.json",
                          "omnipath_lr_pairs.json", "omnipath_ligrecextra_only.json"]:
                src_f = PROJECT / "src5/data/processed" / dataset / fname
                if src_f.exists():
                    shutil.copy2(src_f, processed_dir / fname)
        else:
            log.error(f"  No graph found. Run: mosanic preprocess --config {config_path}")
            sys.exit(1)

    # Load graph
    graph_dict = torch.load(str(graph_path), map_location="cpu", weights_only=False)
    data = graph_dict["hetero_graph"] if isinstance(graph_dict, dict) and "hetero_graph" in graph_dict else graph_dict
    metadata = graph_dict.get("metadata", {}) if isinstance(graph_dict, dict) else {}

    log.info(f"  Node types: {data.node_types}")
    log.info(f"  Edge types: {[et[1] for et in data.edge_types]}")
    for nt in data.node_types:
        log.info(f"    {nt}: {data[nt].x.shape}")
    for et in data.edge_types:
        n_e = data[et].edge_index.shape[1]
        d_e = data[et].edge_attr.shape[1] if data[et].edge_attr is not None else 0
        log.info(f"    {et[1]}: {n_e} edges, attr_dim={d_e}")

    log.info(f"  Expression targets: {data['cell'].y_expr.shape}")
    log.info(f"  Train/Val/Test: {data['cell'].train_mask.sum()}/{data['cell'].val_mask.sum()}/{data['cell'].test_mask.sum()}")
    log.info("  ✓ Graph loaded")

    return data, metadata, cfg, processed_dir


# =============================================================================
# PHASE 3: TRAINING
# =============================================================================

def phase3_train(data, metadata, cfg, device, num_epochs):
    """Train MOSANIC model."""
    banner(f"PHASE 3: Training ({num_epochs} epochs)")

    from mosanic.models import build_model
    from mosanic.training.trainer import MOSANICTrainer

    n_expr_genes = data["cell"].y_expr.shape[1]
    model = build_model(cfg=cfg, n_expr_genes=n_expr_genes, graph_metadata=metadata)
    params = model.count_parameters()
    log.info(f"  MOSANIC model: {params['total']:,} params (encoder={params['encoder']:,}, decoder={params['expression_decoder']:,})")

    # Override training config
    cfg["training"]["epochs"] = num_epochs
    if num_epochs <= 50:
        cfg["training"]["patience"] = 20

    ckpt_dir = ROOT / "checkpoints" / cfg["dataset"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["checkpoint_dir"] = str(ROOT / "checkpoints")

    data_dev = data.to(device)
    trainer = MOSANICTrainer(model, data_dev, cfg, device=device, dataset=cfg["dataset"])

    t0 = time.time()
    history = trainer.train(num_epochs=num_epochs)
    elapsed = time.time() - t0

    best_val_r2 = max(history.get("val_r2", [0]))
    n_epochs_run = len(history.get("train_loss", []))

    # Test set evaluation (best checkpoint already loaded by trainer.train())
    test_metrics = trainer._validate(trainer.test_mask, split="test")
    test_r2 = test_metrics.get("r2_mean", 0)

    log.info(f"  Training: {n_epochs_run} epochs in {elapsed:.1f}s")
    log.info(f"  Best val R²: {best_val_r2:.4f}")
    log.info(f"  Test R²: {test_r2:.4f}")

    # Load best checkpoint
    ckpt_path = ckpt_dir / "model_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model = model.to(device)
        log.info(f"  ✓ Best checkpoint loaded from epoch {ckpt.get('epoch', '?')}")

    log.info("  ✓ Training complete")
    return model, data_dev, {"test_r2": test_r2, "best_val_r2": best_val_r2,
                              "n_epochs": n_epochs_run, "elapsed_s": elapsed}


# =============================================================================
# PHASE 4: CCC EXTRACTION — THE MAIN OUTPUT
# =============================================================================

def phase4_extract_ccc(model, data, device, processed_dir, cfg):
    """Extract all CCC outputs: LR scores, comm matrices, spatial maps."""
    banner("PHASE 4: CCC Extraction")

    from mosanic.evaluation.ccc_extractor import CCCExtractor

    model.eval()

    # Load LR pair vocabulary
    cache_path = processed_dir / "preprocessing_cache.pt"
    cache = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    lr_pair_vocab = cache["gene_interaction_edges"]["lr_pair_vocab"]
    gene_vocab = cache["node_features"]["gene_vocab"]
    gene_name_to_idx = {str(g).upper(): i for i, g in enumerate(gene_vocab)}

    log.info(f"  LR pair vocabulary: {len(lr_pair_vocab)} pairs")
    log.info(f"  Gene vocabulary: {len(gene_vocab)} genes")

    # Create extractor
    extractor = CCCExtractor(model, data, device=device)
    extraction = extractor.extract()

    # ── 4a: Raw LR pair scores ────────────────────────────────────────────
    log.info("\n--- 4a: Raw attention-based LR pair scores ---")
    lr_scores_raw = extractor.get_lr_pair_scores(extraction, lr_pair_vocab)
    log.info(f"  Scored {len(lr_scores_raw)} LR pairs")

    # ── 4b: Filtered scores (remove homodimers + expression tie-breaking) ─
    log.info("\n--- 4b: Filtered LR scores (homodimer removal + expression) ---")
    lr_scores_filtered = extractor.get_lr_pair_scores_raw_filtered(
        extraction, lr_pair_vocab, filter_homodimers=True
    )

    # ── 4c: Enhanced scores (degree correction) ──────────────────────────
    log.info("\n--- 4c: Enhanced LR scores (degree + expression correction) ---")
    lr_scores_enhanced = extractor.get_lr_pair_scores_enhanced(
        extraction, lr_pair_vocab, mode="combined"
    )

    # ── 4d: Last-layer attention ─────────────────────────────────────────
    log.info("\n--- 4d: Last-layer attention scores ---")
    lr_scores_last = extractor.get_lr_pair_scores_last_layer(
        extraction, lr_pair_vocab, layer_idx=-1
    )

    # ── 4e: Embedding cosine similarity ──────────────────────────────────
    log.info("\n--- 4e: Embedding cosine similarity scores ---")
    lr_scores_cosine = extractor.get_lr_pair_scores_embedding(
        lr_pair_vocab, gene_name_to_idx=gene_name_to_idx
    )

    # ── 4f: Cell-type communication matrix ───────────────────────────────
    log.info("\n--- 4f: Cell-type communication matrix ---")
    comm_matrices = {}
    if hasattr(data["cell"], "cell_type"):
        ct = data["cell"].cell_type
        comm_matrices = extractor.get_cell_communication_matrix(extraction, ct)
        for ch, mat in comm_matrices.items():
            log.info(f"  {ch}: [{mat.shape[0]}×{mat.shape[1]}] matrix")

    # ── 4g: Cell embeddings for niche clustering ─────────────────────────
    log.info("\n--- 4g: Cell embeddings ---")
    result = model(data, return_attention=False)
    cell_embeddings = result["node_embeddings"].detach().cpu().numpy()
    log.info(f"  Cell embeddings: {cell_embeddings.shape}")

    # ── 4h: Gate weights (channel importance) ────────────────────────────
    log.info("\n--- 4h: Gate weights (learned channel importance) ---")
    for dst, gates in extraction["gate_weights"].items():
        log.info(f"  {dst}: {gates}")

    log.info("  ✓ CCC extraction complete")

    return {
        "lr_scores_raw": lr_scores_raw,
        "lr_scores_filtered": lr_scores_filtered,
        "lr_scores_enhanced": lr_scores_enhanced,
        "lr_scores_last_layer": lr_scores_last,
        "lr_scores_cosine": lr_scores_cosine,
        "comm_matrices": comm_matrices,
        "cell_embeddings": cell_embeddings,
        "gate_weights": extraction["gate_weights"],
        "lr_pair_vocab": lr_pair_vocab,
    }


# =============================================================================
# PHASE 5: EVALUATION — Metrics against known databases
# =============================================================================

def phase5_evaluate(ccc_results, data, processed_dir):
    """Evaluate CCC predictions against OmniPath."""
    banner("PHASE 5: Evaluation vs Reference Databases")

    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Load OmniPath reference
    op_path = processed_dir / "omnipath_lr_pairs.json"
    if not op_path.exists():
        log.warning("  OmniPath reference not found — skipping AUROC eval")
        return {}

    op_pairs = json.load(open(op_path))
    op_set = set()
    for l, r in op_pairs:
        op_set.add((l.upper(), r.upper()))
        op_set.add((r.upper(), l.upper()))

    metrics = {}

    # AUROC for each scoring variant
    for variant_name, scores_dict in [
        ("raw", ccc_results["lr_scores_raw"]),
        ("filtered", ccc_results["lr_scores_filtered"]),
        ("enhanced", ccc_results["lr_scores_enhanced"]),
        ("last_layer", ccc_results["lr_scores_last_layer"]),
        ("cosine", ccc_results["lr_scores_cosine"]),
    ]:
        if not scores_dict:
            continue
        pairs = list(scores_dict.keys())
        values = np.array([scores_dict[p] for p in pairs])
        labels = np.array([1 if (p[0].upper(), p[1].upper()) in op_set else 0 for p in pairs])
        n_pos = int(labels.sum())
        if n_pos == 0 or n_pos == len(labels):
            continue
        auroc = roc_auc_score(labels, values)
        aupr = average_precision_score(labels, values)
        metrics[f"auroc_{variant_name}"] = auroc
        metrics[f"aupr_{variant_name}"] = aupr
        log.info(f"  {variant_name:12s}: AUROC={auroc:.4f}  AUPR={aupr:.4f}  (n_pos={n_pos}/{len(pairs)})")

    # ARI/NMI
    if hasattr(data["cell"], "cell_type"):
        ct = data["cell"].cell_type.cpu().numpy()
        n_types = len(set(ct.tolist()))
        if n_types > 1:
            emb = ccc_results["cell_embeddings"]
            km = KMeans(n_clusters=n_types, n_init=10, random_state=42)
            pred = km.fit_predict(emb)
            ari = adjusted_rand_score(ct, pred)
            nmi = normalized_mutual_info_score(ct, pred)
            metrics["ari"] = ari
            metrics["nmi"] = nmi
            log.info(f"  Clustering:  ARI={ari:.4f}  NMI={nmi:.4f}  ({n_types} types)")

    log.info("  ✓ Evaluation complete")
    return metrics


# =============================================================================
# PHASE 6: SAVE OUTPUTS — Biologist-friendly format
# =============================================================================

def phase6_save_outputs(ccc_results, eval_metrics, train_metrics, cfg):
    """Save all results in user-friendly formats."""
    banner("PHASE 6: Saving Outputs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Ranked LR pairs CSV ─────────────────────────────────────────
    primary = ccc_results["lr_scores_filtered"] or ccc_results["lr_scores_raw"]
    csv_path = OUTPUT_DIR / "lr_pair_rankings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "ligand", "receptor", "score"])
        for rank, ((lig, rec), score) in enumerate(primary.items(), 1):
            writer.writerow([rank, lig, rec, f"{score:.6f}"])
    log.info(f"  ✓ lr_pair_rankings.csv ({len(primary)} pairs)")

    # ── 2. Cell-type communication matrices ──────────────────────────────
    for channel, mat in ccc_results.get("comm_matrices", {}).items():
        pd.DataFrame(mat).to_csv(OUTPUT_DIR / f"comm_matrix_{channel}.csv")
    log.info(f"  ✓ comm_matrix_*.csv ({len(ccc_results.get('comm_matrices', {}))} channels)")

    # ── 3. Cell embeddings ───────────────────────────────────────────────
    np.save(OUTPUT_DIR / "cell_embeddings.npy", ccc_results["cell_embeddings"])
    log.info(f"  ✓ cell_embeddings.npy {ccc_results['cell_embeddings'].shape}")

    # ── 4. Results JSON ──────────────────────────────────────────────────
    summary = {
        "dataset": cfg["dataset"],
        "technology": cfg["technology"],
        "training": train_metrics,
        "evaluation": eval_metrics,
        "n_lr_pairs": len(primary),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  ✓ results.json")

    log.info(f"  ✓ All outputs → {OUTPUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MOSANIC End-to-End Pipeline Test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = ROOT / "mosanic/configs/examples/breast_cancer_visium.yaml"

    banner("MOSANIC END-TO-END PIPELINE")
    log.info(f"  Config:  {config_path}")
    log.info(f"  Device:  {device}")
    log.info(f"  Epochs:  {args.epochs}")
    log.info(f"  Output:  {OUTPUT_DIR}")

    t_total = time.time()

    # Phase 1: Validate raw data
    adata = phase1_validate_raw_data()

    # Phase 2: Preprocess → heterogeneous graph
    data, metadata, cfg, processed_dir = phase2_preprocess(config_path)

    # Phase 3: Train
    model, data_dev, train_metrics = phase3_train(data, metadata, cfg, device, args.epochs)

    # Phase 4: Extract CCC
    ccc_results = phase4_extract_ccc(model, data_dev, device, processed_dir, cfg)

    # Phase 5: Evaluate
    eval_metrics = phase5_evaluate(ccc_results, data_dev, processed_dir)

    # Phase 6: Save outputs
    phase6_save_outputs(ccc_results, eval_metrics, train_metrics, cfg)

    # Final summary
    elapsed = time.time() - t_total
    banner("PIPELINE COMPLETE")
    log.info(f"  Total time:     {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"  Test R²:        {train_metrics['test_r2']:.4f}")
    for k, v in eval_metrics.items():
        log.info(f"  {k}:  {v:.4f}")
    log.info(f"  LR pairs:       {len(ccc_results['lr_scores_raw'])}")
    log.info(f"  Output dir:     {OUTPUT_DIR}")
    log.info(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size
        log.info(f"    {f.name:<45s} {size:>10,} bytes")
    log.info("")


if __name__ == "__main__":
    main()
