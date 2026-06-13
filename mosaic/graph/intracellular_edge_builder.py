"""
intracellular_edge_builder.py — Build τ₃ Intracellular Self-Loop Edges

Creates self-loop edges for each cell with features encoding the cell's
intracellular signaling state. These capture "what signaling cascades are
primed to activate when communication occurs" — critical for predicting
how a cell's gene expression responds to its neighborhood.

Edge structure:
  edge_index: [2, N] — self-loops (cell i → cell i)
  edge_attr:  [N, d₄] — intracellular signaling features

Feature composition (d₄ = receptor_pca_dim + n_scfea_modules):
  1. Receptor expression profile (PCA-compressed): Which receptors does this
     cell express above the gene-mean threshold? Compressed via PCA to reduce
     the ~800 receptor dimensions to a manageable size.
  2. scFEA metabolic module activity: Normalized flux per module capturing
     the cell's metabolic state (glycolysis, TCA, amino acid metabolism, etc.)

Non-circularity:
  - Receptor expression: raw count > gene_mean (binary, same logic as
    binary_label_generator but captures RECEPTOR side only)
  - scFEA flux: external tool output, not derived from model features
  - Both are structural features (cell identity), not prediction targets

Usage:
    python -m src4.graph.intracellular_edge_builder \
        --config src4/configs/breast_config.yaml \
        --dataset breast_new
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml

logger = logging.getLogger(__name__)

DEFAULT_RECEPTOR_PCA_DIM = 32


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_intracellular_edges(
    n_cells: int,
    receptor_features: np.ndarray,
    scfea_flux: np.ndarray,
    receptor_pca_dim: int = DEFAULT_RECEPTOR_PCA_DIM,
) -> tuple:
    """
    Build τ₃ intracellular self-loop edges.

    Args:
        n_cells: Number of cells
        receptor_features: [N, n_receptors] binary or continuous receptor expression
        scfea_flux: [N, n_modules] scFEA flux values
        receptor_pca_dim: PCA dimensionality for receptor compression

    Returns:
        edge_index: torch.Tensor [2, N] — self-loops
        edge_attr:  torch.Tensor [N, d₄] — intracellular features
    """
    # Self-loops: each cell connects to itself
    idx = torch.arange(n_cells, dtype=torch.long)
    edge_index = torch.stack([idx, idx])  # [2, N]

    # ── 1. Compress receptor expression via PCA ─────────────────────────
    receptor_pca = _pca_compress(
        receptor_features, n_components=receptor_pca_dim
    )  # [N, receptor_pca_dim]
    logger.debug(
        f"Receptor features: {receptor_features.shape} → PCA → {receptor_pca.shape}"
    )

    # ── 2. Normalize scFEA flux per module ──────────────────────────────
    flux_norm = _normalize_flux(scfea_flux)  # [N, n_modules]
    logger.debug(f"scFEA flux: {scfea_flux.shape} → normalized → {flux_norm.shape}")

    # ── 3. Concatenate ──────────────────────────────────────────────────
    edge_features = np.concatenate(
        [receptor_pca, flux_norm], axis=1
    ).astype(np.float32)  # [N, receptor_pca_dim + n_modules]

    edge_attr = torch.from_numpy(edge_features)
    logger.debug(f"τ₃ intracellular: {n_cells} self-loops, edge_dim={edge_attr.shape[1]}")

    return edge_index, edge_attr


def compute_receptor_features(
    adata,
    lr_df,
    filtered_genes: list,
) -> np.ndarray:
    """
    Compute binary receptor expression features for each cell.

    For each receptor gene in the LR database:
      feature = 1 if raw_count[cell, receptor] > gene_mean[receptor], else 0

    Args:
        adata: AnnData with layers['raw_count']
        lr_df: LR database DataFrame with 'receptor' column
        filtered_genes: genes present in the scVI-filtered dataset

    Returns:
        receptor_features: [N, n_receptors] float32 binary matrix
    """
    gene_set_upper = {str(g).upper() for g in filtered_genes}

    # Get unique receptor genes in dataset
    all_receptors = sorted(set(str(r).upper() for r in lr_df['receptor'].unique()))
    receptors_in_data = [r for r in all_receptors if r in gene_set_upper]
    logger.debug(
        f"Receptors: {len(all_receptors)} total → {len(receptors_in_data)} in dataset"
    )

    if not receptors_in_data:
        logger.warning("No receptor genes found in dataset — returning zeros")
        return np.zeros((adata.n_obs, 1), dtype=np.float32)

    # Get raw count expression
    X_raw = None
    for layer_name in ('raw_count', 'raw_counts', 'counts'):
        if layer_name in adata.layers:
            X_raw = adata.layers[layer_name]
            break
    if X_raw is None:
        logger.warning("No raw count layer — using adata.X")
        X_raw = adata.X

    gene_to_idx = {str(g).upper(): i for i, g in enumerate(adata.var_names)}

    # Extract receptor expression [N, n_receptors]
    rec_indices = [gene_to_idx[r] for r in receptors_in_data]
    if sp.issparse(X_raw):
        rec_expr = X_raw[:, rec_indices].toarray().astype(np.float32)
    else:
        rec_expr = np.array(X_raw[:, rec_indices], dtype=np.float32)

    # Binarize: above gene-mean threshold
    gene_means = rec_expr.mean(axis=0)  # [n_receptors]
    receptor_binary = (rec_expr > gene_means[None, :]).astype(np.float32)

    logger.debug(
        f"Receptor binary features: {receptor_binary.shape}, "
        f"mean_active_frac={receptor_binary.mean():.3f}"
    )

    return receptor_binary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pca_compress(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA compression with truncated SVD (handles high-dim sparse data)."""
    from sklearn.decomposition import PCA

    n_features = X.shape[1]
    if n_features <= n_components:
        logger.debug(
            f"PCA skipped: n_features={n_features} <= n_components={n_components}"
        )
        return X.astype(np.float32)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    logger.debug(
        f"PCA: {n_features}→{n_components} dims, "
        f"explained variance={explained:.3f}"
    )
    return X_pca.astype(np.float32)


def _normalize_flux(flux: np.ndarray) -> np.ndarray:
    """Normalize scFEA flux per module to [0, 1] range."""
    flux = flux.astype(np.float32)
    # Per-module min-max normalization
    mins = flux.min(axis=0, keepdims=True)
    maxs = flux.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    return (flux - mins) / ranges


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(config_path: str, dataset: str, receptor_pca_dim: int = DEFAULT_RECEPTOR_PCA_DIM):
    import pandas as pd

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg['paths']['project_root'])
    processed_dir = project_root / cfg['paths']['processed_dir'] / dataset

    # ── Load AnnData ───────────────────────────────────────────────────
    import anndata
    adata_path = project_root / cfg['paths']['source_adata']
    logger.debug(f"Loading AnnData: {adata_path}")
    adata = anndata.read_h5ad(adata_path)
    n_cells = adata.n_obs
    logger.debug(f"  Cells: {n_cells}")

    # ── Load LR database ──────────────────────────────────────────────
    lr_path = processed_dir / 'databases' / 'processed_lr_database.csv'
    lr_df = pd.read_csv(lr_path)

    # ── Load filtered genes ───────────────────────────────────────────
    with open(processed_dir / 'filtered_genes.json') as f:
        filtered_genes = json.load(f)

    # ── Compute receptor features ─────────────────────────────────────
    receptor_features = compute_receptor_features(adata, lr_df, filtered_genes)

    # ── Load scFEA flux ───────────────────────────────────────────────
    scfea_path = project_root / cfg['paths']['source_scfea']
    import pandas as pd
    scfea_df = pd.read_csv(scfea_path, index_col=0)
    scfea_flux = scfea_df.values.astype(np.float32)
    logger.debug(f"scFEA flux: {scfea_flux.shape}")

    # Align flux rows to cell order if needed
    if scfea_flux.shape[0] != n_cells:
        logger.warning(
            f"scFEA cells ({scfea_flux.shape[0]}) != AnnData cells ({n_cells}). "
            f"Truncating/padding."
        )
        if scfea_flux.shape[0] > n_cells:
            scfea_flux = scfea_flux[:n_cells]
        else:
            pad = np.zeros(
                (n_cells - scfea_flux.shape[0], scfea_flux.shape[1]),
                dtype=np.float32,
            )
            scfea_flux = np.vstack([scfea_flux, pad])

    # ── Build edges ───────────────────────────────────────────────────
    edge_index, edge_attr = build_intracellular_edges(
        n_cells=n_cells,
        receptor_features=receptor_features,
        scfea_flux=scfea_flux,
        receptor_pca_dim=receptor_pca_dim,
    )

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = processed_dir / 'typed_edges'
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(edge_index, out_dir / 'intracellular_edge_index.pt')
    torch.save(edge_attr, out_dir / 'intracellular_edge_attr.pt')

    meta = {
        'n_cells': n_cells,
        'n_self_loops': int(edge_index.shape[1]),
        'edge_dim': int(edge_attr.shape[1]),
        'receptor_pca_dim': receptor_pca_dim,
        'n_receptor_genes': int(receptor_features.shape[1]),
        'n_scfea_modules': int(scfea_flux.shape[1]),
        'feature_components': {
            'receptor_pca': list(range(receptor_pca_dim)),
            'scfea_flux': list(
                range(receptor_pca_dim, receptor_pca_dim + scfea_flux.shape[1])
            ),
        },
    }
    with open(out_dir / 'intracellular_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logger.debug("=" * 60)
    logger.info("τ₃ Intracellular Edge Construction Complete")
    logger.debug(f"  Self-loops:    {edge_index.shape[1]}")
    logger.debug(f"  Edge dim:      {edge_attr.shape[1]}")
    logger.debug(f"  Components:    receptor_pca({receptor_pca_dim}) + "
                f"scfea_flux({scfea_flux.shape[1]})")
    logger.info(f"  Saved to:      {out_dir}")
    logger.debug("=" * 60)

    return edge_index, edge_attr, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build τ₃ intracellular self-loop edges'
    )
    parser.add_argument('--config', required=True, help='Config YAML path')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument(
        '--receptor_pca_dim', type=int, default=DEFAULT_RECEPTOR_PCA_DIM,
        help=f'PCA dimensions for receptor features (default: {DEFAULT_RECEPTOR_PCA_DIM})',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    run(
        config_path=args.config,
        dataset=args.dataset,
        receptor_pca_dim=args.receptor_pca_dim,
    )


if __name__ == '__main__':
    main()
