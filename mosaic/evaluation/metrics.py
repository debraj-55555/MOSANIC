"""
mosaic/evaluation/metrics.py

Unified evaluation metrics for MOSAIC:

  L1 Expression Prediction:
    - compute_expression_metrics: R^2, Pearson, Spearman, NRMSE per gene
    - compute_delta_r2: deltaR^2 = R^2(model) - R^2(baseline)

  L2 CCC Recovery:
    - compute_des: Distance Enrichment Score (Spearman rho + AUROC)
    - compute_cross_db_recall: Precision@K + AUROC vs OmniPath
    - fetch_omnipath_lr_pairs: Fetch/cache LR pairs from OmniPath REST API

  L3 Downstream Utility:
    - compute_niche_clustering: ARI/NMI on cell embeddings vs cell type labels
    - compute_clustering_metrics: Wrapper for ARI/NMI with resolution sweep
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


# =========================================================================
# L1: Expression Prediction Metrics
# =========================================================================

def compute_expression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    gene_names: list = None,
) -> dict:
    """
    Compute all L1 expression prediction metrics.

    Args:
        y_pred: [N_eval, G] predicted expression
        y_true: [N_eval, G] observed expression
        gene_names: optional list of G gene names

    Returns:
        dict with R^2, Pearson, Spearman, NRMSE per gene and summary stats
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_squared_error

    N, G = y_true.shape

    r2_per_gene = []
    pearson_per_gene = []
    spearman_per_gene = []
    nrmse_per_gene = []
    mse_per_gene = []
    valid_gene_idx = []

    for g in range(G):
        yt = y_true[:, g]
        yp = y_pred[:, g]

        # Skip constant genes or genes with NaN predictions
        if yt.std() < 1e-8 or np.any(np.isnan(yp)) or np.any(np.isinf(yp)):
            continue

        valid_gene_idx.append(g)

        # R^2
        r2 = r2_score(yt, yp)
        r2_per_gene.append(r2)

        # Pearson r
        r, _ = pearsonr(yt, yp)
        pearson_per_gene.append(r if not np.isnan(r) else 0.0)

        # Spearman rho
        rho, _ = spearmanr(yt, yp)
        spearman_per_gene.append(rho if not np.isnan(rho) else 0.0)

        # NRMSE
        rmse = np.sqrt(mean_squared_error(yt, yp))
        data_range = yt.max() - yt.min()
        nrmse = rmse / data_range if data_range > 1e-8 else 1.0
        nrmse_per_gene.append(nrmse)

        # MSE
        mse_per_gene.append(mean_squared_error(yt, yp))

    r2_arr = np.array(r2_per_gene)
    pearson_arr = np.array(pearson_per_gene)
    spearman_arr = np.array(spearman_per_gene)
    nrmse_arr = np.array(nrmse_per_gene)
    mse_arr = np.array(mse_per_gene)

    # Gene-level ranking
    gene_ranking = np.argsort(r2_arr)[::-1]

    result = {
        # Summary statistics
        'r2_mean': float(r2_arr.mean()),
        'r2_median': float(np.median(r2_arr)),
        'r2_std': float(r2_arr.std()),
        'pearson_mean': float(pearson_arr.mean()),
        'pearson_median': float(np.median(pearson_arr)),
        'spearman_mean': float(spearman_arr.mean()),
        'spearman_median': float(np.median(spearman_arr)),
        'nrmse_mean': float(nrmse_arr.mean()),
        'nrmse_median': float(np.median(nrmse_arr)),
        'mse_mean': float(mse_arr.mean()),
        'n_valid_genes': len(valid_gene_idx),
        'n_positive_r2': int((r2_arr > 0).sum()),
        'frac_positive_r2': float((r2_arr > 0).mean()),

        # Per-gene arrays
        'r2_per_gene': r2_arr.tolist(),
        'pearson_per_gene': pearson_arr.tolist(),
        'spearman_per_gene': spearman_arr.tolist(),
        'nrmse_per_gene': nrmse_arr.tolist(),
        'mse_per_gene': mse_arr.tolist(),
        'valid_gene_idx': valid_gene_idx,
    }

    # Top/bottom genes
    if gene_names:
        valid_names = [gene_names[i] for i in valid_gene_idx]
        top20_idx = gene_ranking[:20]
        bot20_idx = gene_ranking[-20:]
        result['top20_genes'] = [
            {'gene': valid_names[i], 'r2': float(r2_arr[i]),
             'pearson': float(pearson_arr[i])}
            for i in top20_idx
        ]
        result['bottom20_genes'] = [
            {'gene': valid_names[i], 'r2': float(r2_arr[i]),
             'pearson': float(pearson_arr[i])}
            for i in bot20_idx
        ]

    return result


def compute_delta_r2(
    model_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    """
    Compute deltaR^2 = R^2(model) - R^2(baseline) per gene and overall.

    Args:
        model_metrics: L1 metrics from the spatial model
        baseline_metrics: L1 metrics from cell-type-only baseline

    Returns:
        dict with delta_r2 metrics
    """
    model_r2 = np.array(model_metrics['r2_per_gene'])
    baseline_r2 = np.array(baseline_metrics['r2_per_gene'])

    # Align genes (both should have same valid genes)
    n = min(len(model_r2), len(baseline_r2))
    delta_r2 = model_r2[:n] - baseline_r2[:n]

    return {
        'delta_r2_mean': float(delta_r2.mean()),
        'delta_r2_median': float(np.median(delta_r2)),
        'delta_r2_std': float(delta_r2.std()),
        'n_improved_genes': int((delta_r2 > 0).sum()),
        'frac_improved': float((delta_r2 > 0).mean()),
        'delta_r2_per_gene': delta_r2.tolist(),
        'model_r2_mean': model_metrics['r2_mean'],
        'baseline_r2_mean': baseline_metrics['r2_mean'],
    }


# =========================================================================
# L2-A: DES (Distance Enrichment Score)
# =========================================================================

def compute_des(spatial_attn: Dict) -> Dict:
    """
    Distance Enrichment Score.

    Tests whether high-attention cell pairs are at shorter spatial distances.
    Uses all cell-cell attention scores across channels (sum).

    Args:
        spatial_attn: output of CCCExtractor.get_spatial_attention_scores()
                      keys: 'scores' [E], 'distances' [E], 'cell_pairs'

    Returns:
        dict with:
          des_spearman   Spearman rho(attention, -distance)  in [-1, 1]
          des_pval       p-value for spearman correlation
          des_auc        AUROC: predict d<d_median from attention score in [0.5, 1]
          des_top10_mean mean distance of top-10% pairs (um)
          des_bot10_mean mean distance of bottom-10% pairs (um)
          des_ratio      bot10_mean / top10_mean (>1 = spatially enriched)
          n_cell_pairs   total number of scored cell pairs
          mean_dist_um   mean distance across all pairs (um)
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    scores = np.array(spatial_attn["scores"], dtype=np.float32)
    dists  = np.array(spatial_attn["distances"], dtype=np.float32)

    if len(scores) == 0:
        log.warning("DES: no cell pairs to score")
        return {"des_spearman": 0.0, "des_pval": 1.0, "des_auc": 0.5,
                "n_cell_pairs": 0}

    # 1. Spearman rho(score, -distance)
    rho, pval = spearmanr(scores, -dists)

    # 2. AUROC: predict d < d_median from attention score
    d_median = np.median(dists)
    labels = (dists < d_median).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, scores))

    # 3. Top vs bottom decile mean distance
    n = len(scores)
    n10 = max(1, n // 10)
    ranked_idx = np.argsort(-scores)   # descending attention
    top10_mean = float(dists[ranked_idx[:n10]].mean())
    bot10_mean = float(dists[ranked_idx[-n10:]].mean())
    ratio = bot10_mean / max(top10_mean, 1e-8)

    log.info("DES: spearman=%.4f (p=%.2e)  AUC=%.4f  ratio=%.2f  n=%d",
             rho, pval, auc, ratio, n)

    return {
        "des_spearman":   float(rho),
        "des_pval":       float(pval),
        "des_auc":        float(auc),
        "des_top10_mean": top10_mean,
        "des_bot10_mean": bot10_mean,
        "des_ratio":      float(ratio),
        "n_cell_pairs":   int(n),
        "mean_dist_um":   float(np.mean(dists)),
    }


# =========================================================================
# L2-B: Cross-DB Recall@K (OmniPath)
# =========================================================================

OMNIPATH_BASE = "https://omnipathdb.org/interactions"

# Two reference sets:
#   "full"         ligrecextra + lrdb  -- broad but lrdb overlaps LIANA consensus
#   "ligrecextra"  ligrecextra only    -- independent of all tested LR DBs (use for fair comparison)
OMNIPATH_DATASETS = {
    "full":        "ligrecextra,lrdb",
    "ligrecextra": "ligrecextra",
}


def fetch_omnipath_lr_pairs(
    cache_path: Optional[str] = None,
    datasets: str = "full",
) -> Set[Tuple[str, str]]:
    """
    Fetch LR pairs from OmniPath REST API.

    Args:
        cache_path: JSON file to cache results (avoids re-downloading).
        datasets:   "full" (ligrecextra+lrdb) or "ligrecextra" (independent reference).
                    Use "ligrecextra" for fair comparison against LIANA+ which includes lrdb.

    Returns set of (source_genesymbol, target_genesymbol) upper-cased.
    """
    if cache_path and Path(cache_path).exists():
        log.info("OmniPath: loading from cache %s", cache_path)
        with open(cache_path) as f:
            raw = json.load(f)
        return {(str(l).upper(), str(r).upper()) for l, r in raw}

    ds_str = OMNIPATH_DATASETS.get(datasets, datasets)
    url = f"{OMNIPATH_BASE}?datasets={ds_str}&genesymbols=1&format=tsv"
    import requests
    log.info("OmniPath: fetching from %s", url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log.error("OmniPath fetch failed: %s", e)
        return set()

    pairs: Set[Tuple[str, str]] = set()
    lines = resp.text.strip().split("\n")
    if not lines:
        return pairs

    header = lines[0].split("\t")
    try:
        src_idx = header.index("source_genesymbol")
        tgt_idx = header.index("target_genesymbol")
    except ValueError:
        log.error("OmniPath response missing source_genesymbol/target_genesymbol columns")
        log.error("Header: %s", header)
        return pairs

    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) <= max(src_idx, tgt_idx):
            continue
        src = cols[src_idx].strip().upper()
        tgt = cols[tgt_idx].strip().upper()
        if src and tgt and src != "NA" and tgt != "NA":
            pairs.add((src, tgt))

    log.info("OmniPath: %d LR pairs fetched", len(pairs))

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(list(pairs), f)
        log.info("OmniPath: cached to %s", cache_path)

    return pairs


def compute_cross_db_recall(
    lr_scores: Dict[Tuple[str, str], float],
    omnipath_pairs: Set[Tuple[str, str]],
    k_values: Tuple[int, ...] = (100, 500, 1000, 2000),
) -> Dict:
    """
    Cross-DB LR pair recovery: Precision@K + AUROC + AUPR vs OmniPath.

    Standard protocol (Liu et al. 2022 GB, CellNEST 2025):
      - Score all pairs in model's vocab (independent of OmniPath)
      - Binary label: is (L,R) in OmniPath ligrecextra?
      - Precision@K (called "Recall@K" in CCC literature):
            hits@K / K  where hits = |top_K intersect OmniPath| (bidirectional)
      - Lift@K = Precision@K / random_hit_rate  (corrects for DB size/overlap)
      - AUROC = roc_auc_score(labels, scores)  within model's vocab
      - AUPR  = average_precision_score(labels, scores)  within model's vocab
      - Random AUPR = random_hit_rate  (AUPR of random ranking)

    Checks both orientations (A->B and B->A) since OmniPath direction may differ.

    Args:
        lr_scores:      {(ligand, receptor): score} from get_lr_pair_scores()
        omnipath_pairs: set of (source, target) from fetch_omnipath_lr_pairs()
        k_values:       K values to evaluate

    Returns:
        dict with Precision@K, Lift@K, AUROC, AUPR for each K
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not lr_scores:
        log.warning("Cross-DB Recall: no LR scores provided")
        return {}
    if not omnipath_pairs:
        log.warning("Cross-DB Recall: no OmniPath pairs available")
        return {}

    # Build bidirectional lookup for OmniPath
    omnipath_bi = set()
    for l, r in omnipath_pairs:
        omnipath_bi.add((l, r))
        omnipath_bi.add((r, l))

    # Rank model LR pairs by score (descending)
    ranked = sorted(lr_scores.items(), key=lambda x: -x[1])
    n_total = len(ranked)

    results = {
        "n_model_pairs":    n_total,
        "n_omnipath_pairs": len(omnipath_pairs),
    }

    # Coverage: how many OmniPath pairs are in our vocab at all
    our_vocab_bi = set()
    for l, r in lr_scores:
        our_vocab_bi.add((l, r))
        our_vocab_bi.add((r, l))
    n_covered = sum(1 for p in omnipath_pairs if p in our_vocab_bi)
    results["omnipath_vocab_coverage"] = n_covered / len(omnipath_pairs) if omnipath_pairs else 0.0
    results["n_omnipath_in_vocab"] = n_covered

    # Random hit rate: fraction of model's pairs in OmniPath = random Precision@K = random AUPR
    n_model_in_omni = sum(1 for (l, r) in lr_scores if (l, r) in omnipath_bi)
    random_hit_rate = n_model_in_omni / n_total if n_total > 0 else 0.0
    results["random_hit_rate"] = float(random_hit_rate)
    results["n_model_in_omnipath"] = n_model_in_omni

    # Precision@K (called Recall@K in CCC literature: hits@K / K)
    for k in k_values:
        k_eff = min(k, n_total)
        top_k = [pair for pair, _ in ranked[:k_eff]]
        hits = sum(1 for (l, r) in top_k if (l, r) in omnipath_bi)
        recall = hits / k_eff if k_eff > 0 else 0.0
        lift = recall / random_hit_rate if random_hit_rate > 0 else 0.0
        results[f"recall@{k}"] = float(recall)
        results[f"hits@{k}"]   = int(hits)
        results[f"lift@{k}"]   = float(lift)
        log.info("  Precision@%d: %.3f  (%d/%d hits)  lift=%.2fx", k, recall, hits, k_eff, lift)

    # AUROC + AUPR (binary classification: is this pair in OmniPath?)
    # Computed over ALL model vocab pairs -- standard AUROC/AUPR formulation
    labels = np.array([1 if (l, r) in omnipath_bi else 0 for (l, r), _ in ranked])
    scores = np.array([s for _, s in ranked])
    if labels.sum() > 0 and labels.sum() < len(labels):
        auroc = float(roc_auc_score(labels, scores))
        aupr  = float(average_precision_score(labels, scores))
    else:
        auroc = float("nan")
        aupr  = float("nan")
    results["auroc"]       = auroc
    results["aupr"]        = aupr
    results["aupr_random"] = float(random_hit_rate)   # baseline AUPR = random_hit_rate
    log.info("  AUROC: %.4f  AUPR: %.4f  (random AUPR: %.4f)", auroc, aupr, random_hit_rate)

    # Top-10 pairs (for inspection)
    results["top_10_pairs"] = [(str(l), str(r), float(s)) for (l, r), s in ranked[:10]]

    return results


def compute_auroc(
    lr_scores: Dict[Tuple[str, str], float],
    reference_pairs: Set[Tuple[str, str]],
) -> Dict:
    """
    Compute AUROC for LR pair recovery against a reference database.

    Convenience wrapper around compute_cross_db_recall that returns
    just the AUROC and key summary statistics.

    Args:
        lr_scores:       {(ligand, receptor): score}
        reference_pairs: set of known (ligand, receptor) pairs

    Returns:
        dict with auroc, aupr, n_pairs, coverage
    """
    result = compute_cross_db_recall(lr_scores, reference_pairs)
    return {
        "auroc": result.get("auroc", float("nan")),
        "aupr": result.get("aupr", float("nan")),
        "n_model_pairs": result.get("n_model_pairs", 0),
        "n_reference_pairs": result.get("n_omnipath_pairs", 0),
        "coverage": result.get("omnipath_vocab_coverage", 0.0),
    }


# =========================================================================
# L3: Niche Clustering (ARI, NMI)
# =========================================================================

def compute_niche_clustering(
    cell_embeddings: np.ndarray,
    cell_types: np.ndarray,
    n_clusters: int = None,
) -> Dict:
    """
    ARI/NMI niche clustering on cell embeddings.

    Args:
        cell_embeddings: [N, hidden_dim] numpy array
        cell_types:      [N] integer cell type labels (ground truth)
        n_clusters:      number of clusters (default: n_unique_cell_types)

    Returns:
        dict with ari, nmi, n_clusters
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n_types = len(np.unique(cell_types))
    if n_clusters is None:
        n_clusters = n_types

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred_labels = km.fit_predict(cell_embeddings)

    ari = float(adjusted_rand_score(cell_types, pred_labels))
    nmi = float(normalized_mutual_info_score(cell_types, pred_labels))

    log.info("L3 Niche clustering: ARI=%.4f  NMI=%.4f  (k=%d)", ari, nmi, n_clusters)
    return {"ari": ari, "nmi": nmi, "n_clusters": n_clusters}


def compute_clustering_metrics(
    cell_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    resolutions: List[float] = None,
    method: str = 'leiden',
) -> Dict[str, float]:
    """
    Cluster cells by CCC embeddings and compare to reference annotations.

    Sweeps over resolutions/k values and returns the best ARI/NMI.

    Args:
        cell_embeddings: [N, D] -- model node embeddings
        reference_labels: [N] -- integer labels (cell type or spatial domain)
        resolutions: Leiden resolution parameters to sweep
        method: 'leiden' or 'kmeans'

    Returns:
        {'best_ari': float, 'best_nmi': float, 'best_resolution': float, 'method': str}
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if resolutions is None:
        resolutions = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    best_ari = -1.0
    best_nmi = -1.0
    best_res = 0.0

    if method == 'leiden':
        try:
            import scanpy as sc
            import anndata as ad

            adata = ad.AnnData(cell_embeddings)
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)

            for res in resolutions:
                sc.tl.leiden(adata, resolution=res, key_added='cluster')
                pred_labels = adata.obs['cluster'].astype(int).values

                ari = adjusted_rand_score(reference_labels, pred_labels)
                nmi = normalized_mutual_info_score(
                    reference_labels, pred_labels, average_method='arithmetic'
                )

                if ari > best_ari:
                    best_ari = ari
                    best_nmi = nmi
                    best_res = res

        except ImportError:
            log.warning("scanpy not available -- falling back to kmeans")
            method = 'kmeans'

    if method == 'kmeans':
        from sklearn.cluster import KMeans

        n_clusters_ref = len(np.unique(reference_labels))
        for n_k in [n_clusters_ref, n_clusters_ref * 2, max(2, n_clusters_ref // 2)]:
            km = KMeans(n_clusters=n_k, n_init=10, random_state=42)
            pred_labels = km.fit_predict(cell_embeddings)

            ari = adjusted_rand_score(reference_labels, pred_labels)
            nmi = normalized_mutual_info_score(
                reference_labels, pred_labels, average_method='arithmetic'
            )

            if ari > best_ari:
                best_ari = ari
                best_nmi = nmi
                best_res = float(n_k)

    return {
        'best_ari': float(best_ari),
        'best_nmi': float(best_nmi),
        'best_resolution': float(best_res),
        'method': method,
    }
