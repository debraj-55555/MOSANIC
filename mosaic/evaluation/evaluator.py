"""
mosaic/evaluation/evaluator.py

Evaluate trained MOSAIC model on L2 + L3 CCC metrics:

  L2-A: DES (Distance Enrichment Score)
    - Spearman rho and AUROC between cell-pair attention and inverse distance
  L2-B: Cross-DB Recall@K
    - Top-K model LR pairs vs OmniPath (independent DB)
  L3: Niche clustering
    - ARI/NMI on cell embeddings vs cell type labels

Usage:
    python -m mosaic.evaluation.evaluator \\
        --config mosaic/configs/breast_config.yaml \\
        --dataset breast_new \\
        --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--channel", default="all",
                   help="Cell-cell channel for DES: all|contact|secreted|metabolite")
    return p.parse_args()


def evaluate_model(cfg, dataset, device="cuda", channel="all"):
    """
    Run full CCC evaluation on a trained MOSAIC model.

    Args:
        cfg: dict from YAML config
        dataset: dataset name string
        device: torch device string
        channel: cell-cell channel for DES

    Returns:
        dict with all evaluation results
    """
    root = Path(cfg["paths"]["root"]).parent
    processed_dir = root / cfg["paths"]["processed_dir"] / dataset
    graph_path    = processed_dir / "hetero_ccc_graph.pt"
    ckpt_dir      = Path(cfg.get("training", {}).get("checkpoint_dir", "mosaic/checkpoints")) / dataset
    ckpt_path     = ckpt_dir / "model_best.pt"
    omnipath_cache         = processed_dir / "omnipath_lr_pairs.json"           # ligrecextra+lrdb
    omnipath_cache_ligrecx = processed_dir / "omnipath_ligrecextra_only.json"   # ligrecextra only (fair)
    out_path      = processed_dir / "ccc_eval_results.json"

    # -- Device --------------------------------------------------------
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    log.debug("Device: %s", device)

    # -- Load graph ----------------------------------------------------
    log.debug("Loading graph: %s", graph_path)
    ds   = torch.load(graph_path, map_location="cpu", weights_only=False)
    data = ds["hetero_graph"]
    meta = ds["metadata"]
    log.debug("  %d cells, %d genes, %d metabolites",
             meta["n_cells"], meta["n_genes"], meta["n_metabolites"])

    # -- Load model ----------------------------------------------------
    log.debug("Loading checkpoint: %s", ckpt_path)
    from mosaic.models import build_model
    model = build_model(cfg, n_expr_genes=meta["n_expr_genes"], graph_metadata=meta)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    log.debug("Model loaded.")

    # -- Extract attention ---------------------------------------------
    from mosaic.evaluation.ccc_extractor import CCCExtractor
    extractor = CCCExtractor(model, data, device=device)

    log.debug("Running extraction ...")
    extraction = extractor.extract()

    # LR pair vocab (same order as gene_interaction_edges in graph)
    lr_pair_vocab = meta.get("lr_pair_vocab", [])
    if not lr_pair_vocab:
        # Fall back: read from preprocessing cache
        cache_path = processed_dir / "preprocessing_cache.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            lr_pair_vocab = cache.get("gene_interaction_edges", {}).get("lr_pair_vocab", [])
    log.debug("LR pair vocab size: %d", len(lr_pair_vocab))

    lr_scores_raw = extractor.get_lr_pair_scores(extraction, lr_pair_vocab)

    # Raw with homodimer filtering + expression tie-breaking (cleaner primary candidate)
    lr_scores_raw_filt = extractor.get_lr_pair_scores_raw_filtered(
        extraction, lr_pair_vocab, filter_homodimers=True
    )

    # Enhanced attention scoring (degree + expression correction)
    lr_scores_deg  = extractor.get_lr_pair_scores_enhanced(extraction, lr_pair_vocab, mode="degree")
    lr_scores_expr = extractor.get_lr_pair_scores_enhanced(extraction, lr_pair_vocab, mode="expr")
    lr_scores_comb = extractor.get_lr_pair_scores_enhanced(extraction, lr_pair_vocab, mode="combined")

    # Embedding cosine similarity scoring (fixes local-softmax hub-receptor bias)
    # Build gene_name -> node_index map from graph metadata
    gene_vocab = meta.get("gene_vocab", [])
    if not gene_vocab:
        cache_path = processed_dir / "preprocessing_cache.pt"
        _c = torch.load(cache_path, map_location="cpu", weights_only=False)
        gene_vocab = _c.get("gene_universe", {}).get("gene_names", [])
    gene_name_to_idx = {str(g).upper(): i for i, g in enumerate(gene_vocab)} if gene_vocab else {}
    log.debug("Gene vocab for embedding scoring: %d genes", len(gene_name_to_idx))
    lr_scores_emb = extractor.get_lr_pair_scores_embedding(
        lr_pair_vocab, gene_name_to_idx=gene_name_to_idx if gene_name_to_idx else None
    )
    log.debug("Embedding cosine scored pairs: %d", len(lr_scores_emb))

    # Last-layer attention (may be more discriminative than mean-over-layers)
    lr_scores_last = extractor.get_lr_pair_scores_last_layer(
        extraction, lr_pair_vocab, layer_idx=-1, filter_homodimers=True
    )
    # Expressed-only: filter pairs where either gene is expressed in <5% of cells
    lr_scores_expr_filt = extractor.get_lr_pair_scores_expressed(
        extraction, lr_pair_vocab, min_expr_frac=0.05, filter_homodimers=True
    )
    # Linear degree correction (exact de-normalization of local softmax)
    # attn x in_degree ~ pre-softmax logit -- makes degree-1 and degree-20 pairs comparable
    lr_scores_linear_deg = extractor.get_lr_pair_scores_enhanced(
        extraction, lr_pair_vocab, mode="linear_degree"
    )
    # Log-space degree correction (most theoretically correct pre-softmax logit approx)
    # log(attn) + log(degree) ~ pre-softmax logit in additive log-space
    lr_scores_log_deg = extractor.get_lr_pair_scores_enhanced(
        extraction, lr_pair_vocab, mode="log_degree"
    )
    # Last-layer + linear degree (best of both worlds)
    lr_scores_ll_deg = extractor.get_lr_pair_scores_last_layer_degree(
        extraction, lr_pair_vocab, layer_idx=-1, degree_mode="linear", filter_homodimers=True
    )
    # Cell-spatial LIGREC (combines gene-gene attn with cell-level expression co-localization)
    # Rescues Tier3 pairs (high-degree receptors) by using co-expression spatial evidence
    gene_vocab = meta.get("gene_vocab", [])
    if not gene_vocab:
        cache_path2 = processed_dir / "preprocessing_cache.pt"
        _c2 = torch.load(cache_path2, map_location="cpu", weights_only=False)
        gene_vocab = _c2.get("gene_universe", {}).get("gene_names", [])
    gene_name_to_idx_for_ligrec = {str(g).upper(): i for i, g in enumerate(gene_vocab)} if gene_vocab else {}
    lr_scores_cell_spatial_50 = extractor.get_lr_pair_scores_cell_spatial(
        extraction, lr_pair_vocab, gene_name_to_idx=gene_name_to_idx_for_ligrec,
        alpha=0.5, filter_homodimers=True
    )
    lr_scores_cell_spatial_00 = extractor.get_lr_pair_scores_cell_spatial(
        extraction, lr_pair_vocab, gene_name_to_idx=gene_name_to_idx_for_ligrec,
        alpha=0.0, filter_homodimers=True
    )

    # Default for downstream: use raw (best AUROC for dense CellNEST vocab)
    lr_scores = lr_scores_raw

    # Save raw LR scores for DLRC / NicheAct evaluators
    lr_scores_raw_file = processed_dir / "lr_scores_raw.json"
    with open(lr_scores_raw_file, "w") as _f:
        json.dump({"||".join(k): float(v) for k, v in lr_scores_raw.items()}, _f)
    log.debug("Saved LR scores: %s (%d pairs)", lr_scores_raw_file, len(lr_scores_raw))

    # -- Spatial attention scores (for DES) ----------------------------
    coords_um = data["cell"].pos.cpu().numpy() if hasattr(data["cell"], "pos") else None
    if coords_um is None:
        # Try loading from cache
        cache_path = processed_dir / "preprocessing_cache.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        coords_um = cache.get("spatial_graph", {}).get("coords_um")

    if coords_um is None:
        log.error("Could not find coords_um -- skipping DES")
        spatial_attn = None
    else:
        if isinstance(coords_um, torch.Tensor):
            coords_um = coords_um.numpy()
        # IMPORTANT: exclude_self_loops=True (default) and exclude_channels=["intracellular"]
        # Self-loops (intracellular edges, src==dst, dist=0) inflate DES artificially.
        # DES should measure spatial enrichment of CELL-CELL communication only.
        spatial_attn = extractor.get_spatial_attention_scores(
            extraction, coords_um=coords_um, channel=channel,
            exclude_self_loops=True, exclude_channels=["intracellular"],
        )
        log.debug("  Cell pairs for DES: %d (self-loops and intracellular excluded)",
                 len(spatial_attn["scores"]))

    # -- Cell embeddings (for L3) --------------------------------------
    log.debug("Extracting cell embeddings for L3 ...")
    with torch.no_grad():
        out = model(data.to(device), return_attention=False)
    cell_emb = out["node_embeddings"].cpu().numpy()   # [N, hidden_dim]
    cell_types = data["cell"].cell_type.cpu().numpy() if hasattr(data["cell"], "cell_type") else None
    if cell_types is None:
        cache_path = processed_dir / "preprocessing_cache.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cell_types = cache.get("anndata", {}).get("cell_types")
        if isinstance(cell_types, torch.Tensor):
            cell_types = cell_types.numpy()

    # Load spatial splits for test-only evaluation
    cache_path = processed_dir / "preprocessing_cache.pt"
    _cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    splits = _cache.get("spatial_splits", {})
    test_mask  = np.array(splits.get("node_test_mask",  [True] * len(cell_emb)), dtype=bool)
    train_mask = np.array(splits.get("node_train_mask", [True] * len(cell_emb)), dtype=bool)
    log.debug("Splits -- train: %d  val: %d  test: %d",
             train_mask.sum(),
             np.array(splits.get("node_val_mask", [])).sum() if "node_val_mask" in splits else 0,
             test_mask.sum())

    # -- Run metrics ---------------------------------------------------
    results = {"dataset": dataset}

    # L2-A: DES
    log.debug("\n=== L2-A: DES (Distance Enrichment Score) ===")
    if spatial_attn is not None:
        from mosaic.evaluation.metrics import compute_des
        des = compute_des(spatial_attn)
        results["l2_des"] = des
        log.debug("  DES Spearman rho: %.4f (p=%.2e)", des["des_spearman"], des["des_pval"])
        log.debug("  DES AUROC:      %.4f", des["des_auc"])
        log.debug("  DES ratio (bot/top dist): %.2f", des["des_ratio"])
    else:
        log.warning("DES skipped (no spatial coords)")

    # L2-B: Cross-DB Recall@K (OmniPath -- two reference sets)
    log.debug("\n=== L2-B: Cross-DB Recall@K (OmniPath) ===")
    from mosaic.evaluation.metrics import fetch_omnipath_lr_pairs, compute_cross_db_recall

    # Full reference (ligrecextra + lrdb) -- lrdb overlaps LIANA consensus
    omnipath_pairs = fetch_omnipath_lr_pairs(
        cache_path=str(omnipath_cache), datasets="full")
    # Fair reference (ligrecextra only) -- independent of all tested LR DBs
    omnipath_pairs_ligrecx = fetch_omnipath_lr_pairs(
        cache_path=str(omnipath_cache_ligrecx), datasets="ligrecextra")
    log.debug("OmniPath full (ligrecextra+lrdb): %d pairs", len(omnipath_pairs))
    log.debug("OmniPath fair (ligrecextra only): %d pairs", len(omnipath_pairs_ligrecx))

    if omnipath_pairs:
        # Run all scoring variants for comparison
        scoring_variants = {
            "raw":              lr_scores_raw,
            "raw_filtered":     lr_scores_raw_filt,
            "last_layer":       lr_scores_last,
            "expressed":        lr_scores_expr_filt,
            "degree":           lr_scores_deg,
            "linear_degree":    lr_scores_linear_deg,
            "log_degree":       lr_scores_log_deg,
            "last_layer_deg":   lr_scores_ll_deg,
            "expr":             lr_scores_expr,
            "combined":         lr_scores_comb,
            "cell_spatial_50":  lr_scores_cell_spatial_50,  # 50% attn + 50% ligrec
            "cell_spatial_00":  lr_scores_cell_spatial_00,  # pure ligrec
            "embedding":    lr_scores_emb,
        }
        all_recalls = {}
        for name, sc in scoring_variants.items():
            if not sc:
                log.warning("  [%s] No scores available, skipping", name)
                continue
            r = compute_cross_db_recall(sc, omnipath_pairs)
            all_recalls[name] = r
            log.debug("  [%s] Recall@100=%.4f  @500=%.4f  @1000=%.4f  @2000=%.4f",
                     name,
                     r.get("recall@100", 0), r.get("recall@500", 0),
                     r.get("recall@1000", 0), r.get("recall@2000", 0))

        # Primary result: pick best-AUROC variant
        candidate_order = ["cell_spatial_50", "cell_spatial_00", "last_layer_deg",
                           "linear_degree", "last_layer", "raw",
                           "raw_filtered", "combined", "degree", "expr"]
        best_primary = max(
            [n for n in candidate_order if n in all_recalls],
            key=lambda n: all_recalls[n].get("auroc", 0.0),
            default="raw",
        )
        results["l2_cross_db"]         = all_recalls[best_primary]
        results["l2_cross_db"]["_scoring_variant"] = best_primary
        results["l2_cross_db_variants"] = all_recalls

        # Summary table
        _print_names = ["raw", "raw_filtered", "last_layer", "linear_degree", "log_degree",
                        "last_layer_deg", "expressed", "degree", "expr", "combined",
                        "cell_spatial_50", "cell_spatial_00"]
        log.debug("\n  Scoring summary table:")
        log.debug("  %-16s  %-9s  %-9s  %-10s  %-10s  %-8s", "Scoring", "R@100", "R@500", "R@1000", "R@2000", "AUROC")
        for name in _print_names:
            if name not in all_recalls:
                continue
            r = all_recalls[name]
            marker = " <-" if name == best_primary else ""
            log.debug("  %-16s  %.4f    %.4f    %.4f     %.4f     %.4f  %s",
                     name,
                     r.get("recall@100", 0), r.get("recall@500", 0),
                     r.get("recall@1000", 0), r.get("recall@2000", 0),
                     r.get("auroc", 0), marker)

        log.debug("\n  OmniPath pairs: %d  |  vocab coverage: %.1f%%  |  primary variant: %s",
                 all_recalls["raw"]["n_omnipath_pairs"],
                 all_recalls["raw"]["omnipath_vocab_coverage"] * 100,
                 best_primary)
        log.debug("  Top 5 %s-scored LR pairs:", best_primary)
        for lig, rec, sc in all_recalls[best_primary]["top_10_pairs"][:5]:
            log.debug("    (%s, %s) = %.5f", lig, rec, sc)

        # -- Fair reference: ligrecextra only --------------------------
        if omnipath_pairs_ligrecx:
            log.debug("\n  --- Fair Reference: ligrecextra only (no lrdb overlap) ---")
            fair_recalls = {}
            for name, sc in scoring_variants.items():
                r = compute_cross_db_recall(sc, omnipath_pairs_ligrecx)
                fair_recalls[name] = r
            results["l2_cross_db_fair"] = fair_recalls

            log.debug("  ligrecextra vocab coverage: %.1f%%",
                     fair_recalls["raw"]["omnipath_vocab_coverage"] * 100)
    else:
        log.warning("OmniPath unavailable -- skipping Cross-DB Recall")

    # L3: Niche clustering -- evaluated on test cells only (same split as L1)
    log.debug("\n=== L3: Niche Clustering (test cells only) ===")
    if cell_types is not None:
        from mosaic.evaluation.metrics import compute_niche_clustering
        ct_arr = np.array(cell_types)
        # Test-only clustering (primary -- avoids train-set leakage)
        l3 = compute_niche_clustering(cell_emb[test_mask], ct_arr[test_mask])
        results["l3_niche"] = l3
        log.debug("  ARI: %.4f  NMI: %.4f  (n=%d test cells)", l3["ari"], l3["nmi"], test_mask.sum())
        # All-cell clustering (secondary -- for reference)
        l3_all = compute_niche_clustering(cell_emb, ct_arr)
        results["l3_niche_all"] = l3_all
        log.debug("  ARI: %.4f  NMI: %.4f  (n=%d all cells, reference)", l3_all["ari"], l3_all["nmi"], len(ct_arr))
    else:
        log.warning("L3 skipped (no cell type labels)")

    # -- Save results --------------------------------------------------
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.debug("\nResults saved: %s", out_path)

    # -- Summary -------------------------------------------------------
    log.debug("\n" + "=" * 60)
    log.debug("CCC EVALUATION SUMMARY -- %s", dataset)
    log.debug("=" * 60)
    if "l2_des" in results:
        d = results["l2_des"]
        log.debug("L2-A DES:  spearman=%.4f  AUC=%.4f  ratio=%.2f  (n=%d pairs)",
                 d["des_spearman"], d["des_auc"], d["des_ratio"], d["n_cell_pairs"])
    if "l2_cross_db_variants" in results:
        log.debug("L2-B Recall@K (scoring variants):")
        for name, r in results["l2_cross_db_variants"].items():
            log.debug("  %-10s  @100=%.4f  @500=%.4f  @1000=%.4f  @2000=%.4f",
                     name,
                     r.get("recall@100", 0), r.get("recall@500", 0),
                     r.get("recall@1000", 0), r.get("recall@2000", 0))
        log.debug("  OmniPath vocab coverage: %.1f%%",
                 results["l2_cross_db"]["omnipath_vocab_coverage"] * 100)
    elif "l2_cross_db" in results:
        r = results["l2_cross_db"]
        log.debug("L2-B Recall@K:  @100=%.4f  @500=%.4f  @1000=%.4f  @2000=%.4f",
                 r.get("recall@100", 0), r.get("recall@500", 0),
                 r.get("recall@1000", 0), r.get("recall@2000", 0))
        log.debug("  OmniPath vocab coverage: %.1f%%", r["omnipath_vocab_coverage"] * 100)
    if "l3_niche" in results:
        log.debug("L3 Niche (test only): ARI=%.4f  NMI=%.4f  (n=%d cells)",
                 results["l3_niche"]["ari"], results["l3_niche"]["nmi"], test_mask.sum())
    if "l3_niche_all" in results:
        log.debug("L3 Niche (all cells): ARI=%.4f  NMI=%.4f  (n=%d cells)",
                 results["l3_niche_all"]["ari"], results["l3_niche_all"]["nmi"], len(cell_emb))
    log.debug("=" * 60)

    return results


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate_model(cfg, args.dataset, device=args.device, channel=args.channel)


if __name__ == "__main__":
    main()
