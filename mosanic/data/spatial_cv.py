"""
spatial_cv.py — Region-Based Spatial Cross-Validation Splits

Prevents spatial autocorrelation leakage by assigning complete spatial regions
to train / val / test sets, not individual edges.

Method: k-means clustering on cell coordinates → each cluster = a spatial region.
Edges are assigned to a split based on which region both endpoints belong to:
  - Both in train region   → train edge
  - At least one in val    → val edge   (boundary edges go to the harder split)
  - At least one in test   → test edge

Default split ratios: 70% train / 15% val / 15% test (by edges, approximately)

Outputs:
  labels/train_mask.npy   [n_edges] bool
  labels/val_mask.npy     [n_edges] bool
  labels/test_mask.npy    [n_edges] bool
  labels/cv_statistics.json

Usage:
    python -m mosanic.data.spatial_cv \
        --config configs/breast_config.yaml \
        --dataset breast_new
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

def make_spatial_splits(
    coords: np.ndarray,         # [N, 2] cell coordinates (µm)
    edge_index: np.ndarray,     # [2, E]
    val_frac:  float = 0.15,
    test_frac: float = 0.15,
    n_clusters: int = None,     # if None, auto-select
    seed: int = 42,
) -> dict:
    """
    Build train/val/test edge masks using region-based k-means splitting.

    Returns:
        dict with train_mask, val_mask, test_mask (all [E] bool arrays)
        and statistics dict
    """
    from sklearn.cluster import KMeans

    n_cells = coords.shape[0]
    n_edges = edge_index.shape[1]

    # Auto-select number of clusters: roughly sqrt(n_cells / 5) → ~10-30 regions
    if n_clusters is None:
        n_clusters = max(10, min(30, int(np.sqrt(n_cells / 5))))
    logger.debug(f"k-means clustering: n_cells={n_cells}, n_clusters={n_clusters}")

    # Cluster cells into spatial regions
    rng = np.random.RandomState(seed)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cell_region = km.fit_predict(coords)   # [N] integer region id
    logger.debug(f"Cluster sizes: min={np.bincount(cell_region).min()}, "
                f"max={np.bincount(cell_region).max()}, "
                f"mean={np.bincount(cell_region).mean():.1f}")

    # Assign regions to splits (by region, so no leakage)
    region_ids = np.arange(n_clusters)
    shuffled = rng.permutation(region_ids)

    n_val_regions  = max(1, int(n_clusters * val_frac))
    n_test_regions = max(1, int(n_clusters * test_frac))
    n_train_regions = n_clusters - n_val_regions - n_test_regions

    train_regions = set(shuffled[:n_train_regions].tolist())
    val_regions   = set(shuffled[n_train_regions: n_train_regions + n_val_regions].tolist())
    test_regions  = set(shuffled[n_train_regions + n_val_regions:].tolist())

    logger.debug(f"Regions: train={len(train_regions)}, val={len(val_regions)}, test={len(test_regions)}")

    # Map cells to split label: 0=train, 1=val, 2=test
    cell_split = np.zeros(n_cells, dtype=np.int8)
    for r in val_regions:
        cell_split[cell_region == r] = 1
    for r in test_regions:
        cell_split[cell_region == r] = 2

    # Assign edges: hardest split wins (test > val > train)
    src_split = cell_split[edge_index[0]]   # [E]
    dst_split = cell_split[edge_index[1]]   # [E]
    edge_split = np.maximum(src_split, dst_split)   # [E]  0/1/2

    train_mask = (edge_split == 0)
    val_mask   = (edge_split == 1)
    test_mask  = (edge_split == 2)

    # Log split quality
    n_train = train_mask.sum()
    n_val   = val_mask.sum()
    n_test  = test_mask.sum()
    total   = n_edges

    logger.debug(f"Edge split: train={n_train} ({n_train/total*100:.1f}%), "
                f"val={n_val} ({n_val/total*100:.1f}%), "
                f"test={n_test} ({n_test/total*100:.1f}%)")

    # Warn if any split is empty or too small
    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        if mask.sum() < 100:
            logger.warning(f"Split '{name}' has only {mask.sum()} edges — consider fewer clusters")

    stats = {
        'n_cells':       int(n_cells),
        'n_edges':       int(n_edges),
        'n_clusters':    int(n_clusters),
        'n_train_edges': int(n_train),
        'n_val_edges':   int(n_val),
        'n_test_edges':  int(n_test),
        'train_frac':    float(n_train / total),
        'val_frac':      float(n_val / total),
        'test_frac':     float(n_test / total),
        'n_train_regions': len(train_regions),
        'n_val_regions':   len(val_regions),
        'n_test_regions':  len(test_regions),
        'seed':           seed,
    }

    return {
        'train_mask':    train_mask,
        'val_mask':      val_mask,
        'test_mask':     test_mask,
        'cell_region':   cell_region,
        'cell_split':    cell_split,
        'stats':         stats,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(config_path: str, dataset: str, n_clusters: int = None, seed: int = 42):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = cfg['paths']['project_root']
    processed_dir = Path(root) / cfg['paths']['processed_dir'] / dataset
    labels_dir = processed_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    coords = np.load(processed_dir / 'spatial_coords_um.npy')
    edge_index = np.load(processed_dir / 'spatial_edge_index.npy')
    logger.debug(f"Coords: {coords.shape}, Edges: {edge_index.shape}")

    # Build splits
    result = make_spatial_splits(
        coords=coords,
        edge_index=edge_index,
        n_clusters=n_clusters,
        seed=seed,
    )

    # Save edge-level masks
    np.save(labels_dir / 'train_mask.npy', result['train_mask'])
    np.save(labels_dir / 'val_mask.npy',   result['val_mask'])
    np.save(labels_dir / 'test_mask.npy',  result['test_mask'])
    np.save(labels_dir / 'cell_region.npy', result['cell_region'])
    np.save(labels_dir / 'cell_split.npy',  result['cell_split'])

    # Save node-level masks (for HetGT expression prediction)
    cell_split = result['cell_split']
    np.save(labels_dir / 'node_train_mask.npy', (cell_split == 0))
    np.save(labels_dir / 'node_val_mask.npy',   (cell_split == 1))
    np.save(labels_dir / 'node_test_mask.npy',  (cell_split == 2))
    n_node_train = (cell_split == 0).sum()
    n_node_val = (cell_split == 1).sum()
    n_node_test = (cell_split == 2).sum()
    logger.debug(f"Node split: train={n_node_train}, val={n_node_val}, test={n_node_test}")

    with open(labels_dir / 'cv_statistics.json', 'w') as f:
        json.dump(result['stats'], f, indent=2)

    stats = result['stats']
    logger.debug("=" * 60)
    logger.info("Spatial CV Split Complete")
    logger.debug(f"  Clusters (regions): {stats['n_clusters']}")
    logger.debug(f"  Train: {stats['n_train_edges']} edges ({stats['train_frac']*100:.1f}%)")
    logger.debug(f"  Val:   {stats['n_val_edges']}   edges ({stats['val_frac']*100:.1f}%)")
    logger.debug(f"  Test:  {stats['n_test_edges']}  edges ({stats['test_frac']*100:.1f}%)")
    logger.info(f"  Saved to: {labels_dir}")
    logger.debug("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Build spatial CV splits')
    parser.add_argument('--config',     required=True, help='Config YAML path')
    parser.add_argument('--dataset',    required=True, help='Dataset name')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of spatial regions (default: auto)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    run(config_path=args.config, dataset=args.dataset,
        n_clusters=args.n_clusters, seed=args.seed)


if __name__ == '__main__':
    main()
