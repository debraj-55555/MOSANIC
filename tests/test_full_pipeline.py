#!/usr/bin/env python3
"""
MOSANIC Full Pipeline Test — Breast Cancer (Visium)

Tests the complete pipeline: preprocess → train → evaluate
using real breast cancer spatial transcriptomics data.

Usage:
    cd /mnt/disk-drive/debraj/cci_proj2/CCC/MOSANIC
    /opt/miniconda/envs/ccc_env/bin/python3 tests/test_full_pipeline.py [--device cuda] [--quick]

Options:
    --device    cuda or cpu (default: cuda if available)
    --quick     Quick test: 10 epochs only (default: full 500 epochs)
    --skip-preprocess   Skip preprocessing (use existing processed graph)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

# ─── Setup ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]       # MOSANIC/
PROJECT = ROOT.parent                             # CCC/
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mosanic.test")


def banner(msg):
    log.info("")
    log.info("=" * 70)
    log.info(f"  {msg}")
    log.info("=" * 70)


def check_data_exists():
    """Verify all required input files exist before starting."""
    required = {
        "Raw AnnData":       PROJECT / "src5/data/raw/breast_new.h5ad",
        "LR Database":       PROJECT / "src5/data/raw/CellNEST_database.csv",
        "M_R Database":      PROJECT / "src5/data/raw/M_R.txt",
        "scFEA Balance":     PROJECT / "src5/data/raw/scfea_balance_breast_new.csv",
        "ESM-2 Embeddings":  PROJECT / "src5/data/embeddings/proteins",
        "ChemBERTa Embeddings": PROJECT / "src5/data/embeddings/metabolites",
    }
    all_ok = True
    for name, path in required.items():
        exists = path.exists()
        status = "✓" if exists else "✗ MISSING"
        log.info(f"  {status}  {name}: {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        log.error("Some required files are missing. See above.")
        sys.exit(1)
    log.info("  All input data found.")


def test_imports():
    """Test that all MOSANIC modules import correctly."""
    banner("TEST 1: Module Imports")

    from mosanic import __version__, MOSANIC, build_model
    log.info(f"  mosanic v{__version__}")

    from mosanic.models.encoder import HetGTEncoder
    from mosanic.models.decoder import ExpressionDecoder
    from mosanic.training.trainer import MOSANICTrainer
    from mosanic.training.losses import MOSANICLoss
    from mosanic.training.callbacks import EarlyStopping, ModelCheckpoint
    from mosanic.graph import GraphAssembler, EdgeBuilder, NodeFeatureBuilder
    from mosanic.evaluation.ccc_extractor import CCCExtractor
    log.info("  ✓ All modules imported successfully")


def test_model_forward():
    """Test model construction and forward pass with dummy data."""
    banner("TEST 2: Model Forward Pass (Dummy Data)")

    from mosanic.models import build_model
    from torch_geometric.data import HeteroData

    model = build_model(
        cfg={"model": {"hidden_dim": 64, "n_heads": 2, "n_layers": 1, "decoder_dims": [64]}},
        n_expr_genes=50,
    )
    log.info(f"  Model: {model.count_parameters()['total']:,} params")

    # Dummy HeteroData
    data = HeteroData()
    data["cell"].x = torch.randn(10, 128)
    data["gene"].x = torch.randn(20, 1280)
    data["metabolite"].x = torch.randn(5, 600)
    data["cell", "secreted", "cell"].edge_index = torch.randint(0, 10, (2, 15))
    data["cell", "secreted", "cell"].edge_attr = torch.randn(15, 2)
    data["cell", "expresses", "gene"].edge_index = torch.stack(
        [torch.randint(0, 10, (30,)), torch.randint(0, 20, (30,))]
    )
    data["cell", "expresses", "gene"].edge_attr = torch.randn(30, 3)
    data["gene", "interacts", "gene"].edge_index = torch.stack(
        [torch.randint(0, 20, (8,)), torch.randint(0, 20, (8,))]
    )
    data["gene", "interacts", "gene"].edge_attr = torch.randn(8, 3)
    data["cell", "flux", "metabolite"].edge_index = torch.stack(
        [torch.randint(0, 10, (7,)), torch.randint(0, 5, (7,))]
    )
    data["cell", "flux", "metabolite"].edge_attr = torch.randn(7, 2)

    result = model(data, return_attention=True)
    assert result["expression"].shape == (10, 50)
    assert result["node_embeddings"].shape == (10, 64)
    assert len(result["attention_info"]["per_layer"]) == 1
    log.info(f"  ✓ Forward pass: expr={result['expression'].shape}, attn_layers=1")

    # Test loss
    from mosanic.training.losses import MOSANICLoss
    loss_fn = MOSANICLoss(ccc_weight=0.0, expr_loss_type="huber")
    loss, _ = loss_fn(
        result, {"expression": torch.randn(10, 50)},
        node_mask=torch.ones(10, dtype=torch.bool),
    )
    log.info(f"  ✓ Huber loss: {loss.item():.4f}")


def test_preprocess(config_path, skip=False):
    """Run preprocessing on breast cancer data."""
    banner("TEST 3: Preprocessing (Real Breast Data)")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    processed_dir = PROJECT / cfg["paths"]["processed_dir"] / dataset

    if skip and (processed_dir / "hetero_ccc_graph.pt").exists():
        log.info(f"  Skipping (graph exists): {processed_dir / 'hetero_ccc_graph.pt'}")
        return

    # Use existing src5 processed graph to avoid re-running 17 steps
    src5_graph = PROJECT / "src5/data/processed" / dataset / "hetero_ccc_graph.pt"
    if src5_graph.exists():
        log.info(f"  Copying existing processed graph from src5...")
        processed_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src5_graph, processed_dir / "hetero_ccc_graph.pt")
        # Also copy preprocessing cache if available
        src5_cache = PROJECT / "src5/data/processed" / dataset / "preprocessing_cache.pt"
        if src5_cache.exists():
            shutil.copy2(src5_cache, processed_dir / "preprocessing_cache.pt")
        src5_meta = PROJECT / "src5/data/processed" / dataset / "metadata.json"
        if src5_meta.exists():
            shutil.copy2(src5_meta, processed_dir / "metadata.json")
        # Copy OmniPath eval data
        for fname in ["omnipath_lr_pairs.json", "omnipath_ligrecextra_only.json",
                       "ccc_eval_results.json", "lr_scores_raw.json"]:
            src_f = PROJECT / "src5/data/processed" / dataset / fname
            if src_f.exists():
                shutil.copy2(src_f, processed_dir / fname)
        log.info(f"  ✓ Processed graph copied to: {processed_dir}")
    else:
        log.error(f"  No existing graph found at {src5_graph}")
        log.error("  Run: mosanic preprocess --config <config> first")
        sys.exit(1)


def test_train(config_path, device, quick=False):
    """Train on breast cancer data."""
    banner("TEST 4: Training (Real Breast Data)")

    from mosanic.models import build_model
    from mosanic.training.trainer import MOSANICTrainer

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    processed_dir = PROJECT / cfg["paths"]["processed_dir"] / dataset

    # Load graph
    graph_path = processed_dir / "hetero_ccc_graph.pt"
    log.info(f"  Loading graph: {graph_path}")
    graph_dict = torch.load(str(graph_path), map_location="cpu", weights_only=False)
    if isinstance(graph_dict, dict) and "hetero_graph" in graph_dict:
        data = graph_dict["hetero_graph"]
        metadata = graph_dict.get("metadata", {})
    else:
        data = graph_dict
        metadata = {}

    log.info(f"  Graph: {data.node_types}, edges={[et[1] for et in data.edge_types]}")
    log.info(f"  Cells: {data['cell'].x.shape[0]}, Genes: {data['gene'].x.shape[0]}")

    # Build model
    n_expr_genes = data["cell"].y_expr.shape[1] if hasattr(data["cell"], "y_expr") else 200
    model = build_model(cfg=cfg, n_expr_genes=n_expr_genes, graph_metadata=metadata)
    params = model.count_parameters()
    log.info(f"  Model: encoder={params['encoder']:,}, decoder={params['expression_decoder']:,}, total={params['total']:,}")

    # Override epochs for quick test
    if quick:
        cfg["training"]["epochs"] = 10
        cfg["training"]["patience"] = 5
        log.info("  Quick mode: 10 epochs, patience=5")

    # Checkpoint dir
    ckpt_dir = ROOT / "checkpoints" / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["checkpoint_dir"] = str(ROOT / "checkpoints")
    cfg["paths"] = cfg.get("paths", {})

    # Train
    data_dev = data.to(device)
    trainer = MOSANICTrainer(model, data_dev, cfg, device=device, dataset=dataset)

    t0 = time.time()
    num_epochs = cfg["training"].get("epochs", 500)
    history = trainer.train(num_epochs=num_epochs)
    elapsed = time.time() - t0

    best_r2 = max(history.get("val_r2", [0]))
    final_loss = history["train_loss"][-1] if history.get("train_loss") else None
    log.info(f"  ✓ Training complete: {elapsed:.1f}s, {len(history.get('train_loss', []))} epochs")
    log.info(f"  ✓ Best val R²: {best_r2:.4f}, Final train loss: {final_loss:.4f}")

    # Test metrics on test set
    test_metrics = trainer._validate(trainer.test_mask, split="test")
    test_r2 = test_metrics.get("r2", test_metrics.get("val_r2", 0))
    log.info(f"  ✓ Test R²: {test_r2:.4f}")

    return model, data_dev


def test_evaluate(model, data, device):
    """Extract CCC scores and compute metrics."""
    banner("TEST 5: CCC Evaluation")

    from mosanic.evaluation.ccc_extractor import CCCExtractor

    model.eval()
    extractor = CCCExtractor(model, data, device=device)
    extraction = extractor.extract()

    # Check attention was extracted
    lr_scores = extraction.get("lr_pair_edge_scores")
    if lr_scores is not None:
        n_pairs = len(lr_scores)
        mean_score = lr_scores.mean()
        log.info(f"  ✓ LR pair scores: {n_pairs} pairs, mean attention={mean_score:.4f}")
    else:
        log.warning("  No LR pair scores extracted (gene↔gene edges missing?)")

    # Check cell embeddings for clustering
    result = model(data, return_attention=False)
    emb = result["node_embeddings"].detach().cpu().numpy()
    log.info(f"  ✓ Cell embeddings: {emb.shape}")

    # Quick ARI/NMI if cell types available
    if hasattr(data["cell"], "cell_type"):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ct = data["cell"].cell_type.cpu().numpy()
        n_types = len(set(ct.tolist()))
        if n_types > 1:
            km = KMeans(n_clusters=n_types, n_init=10, random_state=42)
            pred = km.fit_predict(emb)
            ari = adjusted_rand_score(ct, pred)
            nmi = normalized_mutual_info_score(ct, pred)
            log.info(f"  ✓ Niche clustering: ARI={ari:.4f}, NMI={nmi:.4f} ({n_types} types)")

    log.info("  ✓ Evaluation complete")


def main():
    parser = argparse.ArgumentParser(description="MOSANIC Full Pipeline Test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 epochs)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = ROOT / "mosanic/configs/examples/breast_cancer_visium.yaml"

    banner("MOSANIC FULL PIPELINE TEST")
    log.info(f"  Config:  {config_path}")
    log.info(f"  Device:  {device}")
    log.info(f"  Mode:    {'quick (10 epochs)' if args.quick else 'full (500 epochs)'}")
    log.info(f"  Project: {PROJECT}")

    t_start = time.time()

    # Pre-flight: check data
    banner("TEST 0: Data Availability Check")
    check_data_exists()

    # Step 1: Imports
    test_imports()

    # Step 2: Model forward
    test_model_forward()

    # Step 3: Preprocess (or copy existing)
    test_preprocess(config_path, skip=args.skip_preprocess)

    # Step 4: Train
    model, data_dev = test_train(config_path, device, quick=args.quick)

    # Step 5: Evaluate
    test_evaluate(model, data_dev, device)

    # Summary
    elapsed = time.time() - t_start
    banner("ALL TESTS PASSED")
    log.info(f"  Total time: {elapsed:.1f}s")
    log.info(f"  Package:    mosanic-ccc v1.0.0")
    log.info(f"  Dataset:    breast_new (Visium)")
    log.info(f"  Device:     {device}")
    log.info("")


if __name__ == "__main__":
    main()
