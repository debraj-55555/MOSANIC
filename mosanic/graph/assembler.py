"""
mosanic/graph/assembler.py

Assembles the final PyG HeteroData from the preprocessing cache.

Graph structure:
  Node types:
    'cell'        x=[N, 128]    scVI latent embeddings
    'gene'        x=[G, 1280]   ESM-2 protein embeddings
    'metabolite'  x=[M, 600]    ChemBERTa SMILES embeddings

  Edge types (6):
    ('cell', 'secreted',       'cell')       τ₁  LR-secreted
    ('cell', 'metabolite',     'cell')       τ₂  scFEA flux-mediated  [SEPARATE from LR]
    ('cell', 'intracellular',  'cell')       τ₃  receptor PCA + flux self-loops
    ('cell', 'expresses',      'gene')       ε₁  top-K expression edges
    ('gene', 'interacts',      'gene')       ε₂  LR pair edges (KEY for CCC scoring)
    ('cell', 'flux',           'metabolite') ε₃  scFEA flux edges  [SEPARATE from LR]
  NOTE: τ₁ contact removed — τ₁⊆τ₁ (100% topological overlap); absent in intestinal
        cancer with no performance cost; distance-only fallback in mouse. Redundant.

Primary target: y_expr [N, 200] log-normalized expression on 'cell' nodes.
Auxiliary stored at graph level: y_lr, y_metab, lr_vocab, lr_pair_vocab, etc.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class GraphAssembler:
    """
    Assembles a PyG HeteroData object from the preprocessing cache dict.

    Args:
        cfg: config dict (used for metadata)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def assemble(self, cache: dict) -> Tuple:
        """
        Build HeteroData from preprocessing cache.

        Args:
            cache: dict populated by preprocessor.py (keys: anndata, spatial_graph, …)

        Returns:
            (HeteroData, metadata_dict)
        """
        from torch_geometric.data import HeteroData

        # ── Unpack cache ──────────────────────────────────────────────────
        s1  = cache["anndata"]
        s2  = cache["spatial_graph"]
        s6  = cache["node_features"]
        s7  = cache["cell_cell_lr_edges"]
        s8  = cache["cell_cell_met_edges"]
        s9  = cache["cell_gene_edges"]
        s10 = cache["gene_interaction_edges"]
        s11 = cache["cell_metabolite_edges"]
        s11b = cache.get("metabolite_gene_edges", None)
        s12 = cache["intracellular_edges"]
        s13 = cache["expr_labels"]
        s14 = cache["lr_labels"]
        s15 = cache["met_labels"]
        s16 = cache["spatial_splits"]

        # ── Node features ─────────────────────────────────────────────────
        cell_x    = _to_float_tensor(s1["cell_features"])          # [N, 128]
        gene_x    = _to_float_tensor(s6["gene_feats"])             # [G, 1280]
        met_x     = _to_float_tensor(s6["met_feats"])              # [M, 600]

        gene_vocab = s6["gene_vocab"]
        met_vocab  = s6["met_vocab"]

        n_cells = int(cell_x.shape[0])
        n_genes = int(gene_x.shape[0])
        n_mets  = int(met_x.shape[0])

        log.info("  Nodes: cell=%d, gene=%d, metabolite=%d", n_cells, n_genes, n_mets)

        # ── Expression labels + masks ─────────────────────────────────────
        y_expr       = _to_float_tensor(s13["y_expr"])             # [N, 200]
        target_genes = s13["target_genes"]

        train_mask = _to_bool_tensor(s16["node_train_mask"])       # [N]
        val_mask   = _to_bool_tensor(s16["node_val_mask"])         # [N]
        test_mask  = _to_bool_tensor(s16["node_test_mask"])        # [N]

        # Cell type labels
        cell_type = _cell_type_tensor(s1.get("cell_types"))

        # ── Cell-cell edges ───────────────────────────────────────────────
        # τ₁ contact removed: τ₁⊆τ₁ (100% overlap), absent in intestinal, fallback in mouse
        secreted_ei = s7["secreted_ei"]
        secreted_ea = _to_float_tensor(s7["secreted_ea"])
        met_cc_ei   = s8["met_cc_ei"]
        met_cc_ea   = _to_float_tensor(s8["met_cc_ea"])
        intra_ei    = s12["intra_ei"]
        intra_ea    = _to_float_tensor(s12["intra_ea"])

        # ── Cross-type edges ──────────────────────────────────────────────
        cell_gene_ei = s9["cell_gene_ei"]
        cell_gene_ea = _to_float_tensor(s9["cell_gene_ea"])
        lr_gene_ei   = s10["lr_gene_ei"]
        lr_gene_ea   = _to_float_tensor(s10["lr_gene_ea"])
        cell_met_ei  = s11["cell_met_ei"]
        cell_met_ea  = _to_float_tensor(s11["cell_met_ea"])

        # ── Auxiliary labels (stored at graph level) ──────────────────────
        base_ei   = _to_long_tensor(s2["edge_index"])              # [2, E]
        dist_um   = _to_float_tensor(s2["dist_um"])                # [E]

        y_lr     = _load_sparse_label(s14, "edge_lr_multilabel")   # [E, n_lr]
        y_metab  = _load_sparse_label(s15, "y_metab")              # [E, n_mod]
        lr_vocab = s14.get("lr_vocab", [])
        metab_vocab = s15.get("metab_vocab", [])
        lr_pair_vocab = s10.get("lr_pair_vocab", [])

        # ── Assemble HeteroData ───────────────────────────────────────────
        data = HeteroData()

        # Cell nodes
        data["cell"].x          = cell_x
        data["cell"].y_expr     = y_expr
        data["cell"].train_mask = train_mask
        data["cell"].val_mask   = val_mask
        data["cell"].test_mask  = test_mask
        data["cell"].num_nodes  = n_cells
        if cell_type is not None:
            data["cell"].cell_type = cell_type

        # Gene nodes
        data["gene"].x         = gene_x
        data["gene"].num_nodes = n_genes

        # Metabolite nodes
        data["metabolite"].x         = met_x
        data["metabolite"].num_nodes = n_mets

        # τ₁ secreted
        data["cell", "secreted", "cell"].edge_index = secreted_ei
        data["cell", "secreted", "cell"].edge_attr  = secreted_ea

        # τ₂ metabolite cell-cell (flux-mediated, SEPARATE from LR)
        data["cell", "metabolite", "cell"].edge_index = met_cc_ei
        data["cell", "metabolite", "cell"].edge_attr  = met_cc_ea

        # τ₃ intracellular self-loops
        data["cell", "intracellular", "cell"].edge_index = intra_ei
        data["cell", "intracellular", "cell"].edge_attr  = intra_ea

        # ε₁ cell → gene expression
        data["cell", "expresses", "gene"].edge_index = cell_gene_ei
        data["cell", "expresses", "gene"].edge_attr  = cell_gene_ea

        # ε₂ gene ↔ gene LR interaction  (KEY for direct CCC scoring)
        data["gene", "interacts", "gene"].edge_index = lr_gene_ei
        data["gene", "interacts", "gene"].edge_attr  = lr_gene_ea

        # ε₃ cell → metabolite flux
        data["cell", "flux", "metabolite"].edge_index = cell_met_ei
        data["cell", "flux", "metabolite"].edge_attr  = cell_met_ea

        # ε₄ metabolite → gene (sensed_by)
        if s11b is not None and s11b["met_gene_ei"].shape[1] > 0:
            data["metabolite", "sensed_by", "gene"].edge_index = s11b["met_gene_ei"]
            data["metabolite", "sensed_by", "gene"].edge_attr  = _to_float_tensor(s11b["met_gene_ea"])

        # ── Auxiliary graph-level attributes ─────────────────────────────
        data.base_edge_index  = base_ei
        data.base_edge_dist   = dist_um
        if y_lr is not None:
            data.y_lr    = y_lr
        if y_metab is not None:
            data.y_metab = y_metab

        # ── Metadata ──────────────────────────────────────────────────────
        metadata = {
            "dataset":       self.cfg.get("dataset", "unknown"),
            "graph_type":    "HeteroData",
            "model_target":  "MOSANIC",
            "version":       "mosanic",

            # Node counts
            "n_cells":        n_cells,
            "n_genes":        n_genes,
            "n_metabolites":  n_mets,
            "cell_feature_dim":  int(cell_x.shape[1]),
            "gene_feature_dim":  int(gene_x.shape[1]),
            "met_feature_dim":   int(met_x.shape[1]),

            # Primary task
            "n_expr_genes":   int(y_expr.shape[1]),
            "target_genes":   target_genes,

            # Edge counts + dims  (τ₁ contact removed)
            "n_secreted_edges":      int(secreted_ei.shape[1]),
            "n_met_cc_edges":        int(met_cc_ei.shape[1]),
            "n_intra_edges":         int(intra_ei.shape[1]),
            "n_cell_gene_edges":     int(cell_gene_ei.shape[1]),
            "n_lr_gene_edges":       int(lr_gene_ei.shape[1]),
            "n_cell_met_edges":      int(cell_met_ei.shape[1]),
            "secreted_edge_dim":     int(secreted_ea.shape[1]),
            "met_cc_edge_dim":       int(met_cc_ea.shape[1]),
            "intra_edge_dim":        int(intra_ea.shape[1]),
            "cell_gene_edge_dim":    int(cell_gene_ea.shape[1]),
            "lr_gene_edge_dim":      int(lr_gene_ea.shape[1]),
            "cell_met_edge_dim":     int(cell_met_ea.shape[1]),
            "n_met_gene_edges":      int(s11b["met_gene_ei"].shape[1]) if s11b is not None and s11b["met_gene_ei"].shape[1] > 0 else 0,
            "met_gene_edge_dim":     int(s11b["met_gene_ea"].shape[1]) if s11b is not None and s11b["met_gene_ei"].shape[1] > 0 else 0,

            # CV splits
            "node_train": int(train_mask.sum()),
            "node_val":   int(val_mask.sum()),
            "node_test":  int(test_mask.sum()),

            # Vocabularies
            "gene_vocab":       gene_vocab,
            "met_vocab":        met_vocab,
            "lr_vocab":         lr_vocab,
            "metab_vocab":      metab_vocab,
            "lr_pair_vocab":    lr_pair_vocab,
            "cell_type_names":  s1.get("cell_type_names"),  # list[str] or None

            # Auxiliary label sizes
            "n_lr_classes":    int(y_lr.shape[1]) if y_lr is not None else 0,
            "n_metab_classes": int(y_metab.shape[1]) if y_metab is not None else 0,

            # Edge type descriptions
            "edge_types": {
                "secreted":      "τ₁ paracrine LR-secreted edges, dist≤150µm",
                "metabolite":    "τ₂ flux-mediated cell-cell, SEPARATE from LR",
                "intracellular": "τ₃ self-loops: receptor PCA + scFEA flux",
                "expresses":     "ε₁ cell→gene top-K expression edges",
                "interacts":     "ε₂ gene↔gene LR pairs — direct CCC score via attention",
                "flux":          "ε₃ cell→metabolite scFEA flux>q50, SEPARATE from LR",
            },
        }

        # Log summary
        self._log_summary(metadata)

        return data, metadata

    def _log_summary(self, m: dict):
        log.info("=" * 65)
        log.info("HeteroData assembled: %s", m["dataset"])
        log.info("  Node types:")
        log.info("    cell:        %5d  (dim=%d,  scVI latent)",      m["n_cells"],        m["cell_feature_dim"])
        log.info("    gene:        %5d  (dim=%d, ESM-2)",             m["n_genes"],        m["gene_feature_dim"])
        log.info("    metabolite:  %5d  (dim=%d, ChemBERTa)",         m["n_metabolites"],  m["met_feature_dim"])
        log.info("  Edge types:")
        log.info("    τ₁ secreted:     %6d  (dim=%d)", m["n_secreted_edges"],   m["secreted_edge_dim"])
        log.info("    τ₂ met cell-cell:%6d  (dim=%d)", m["n_met_cc_edges"],     m["met_cc_edge_dim"])
        log.info("    τ₃ intracellular:%6d  (dim=%d)", m["n_intra_edges"],      m["intra_edge_dim"])
        log.info("    ε₁ cell→gene:    %6d  (dim=%d)", m["n_cell_gene_edges"],  m["cell_gene_edge_dim"])
        log.info("    ε₂ gene↔gene LR: %6d  (dim=%d)", m["n_lr_gene_edges"],   m["lr_gene_edge_dim"])
        log.info("    ε₃ cell→met:     %6d  (dim=%d)", m["n_cell_met_edges"],   m["cell_met_edge_dim"])
        if m.get("n_met_gene_edges", 0) > 0:
            log.info("    ε₄ met→gene:     %6d  (dim=%d)", m["n_met_gene_edges"],  m["met_gene_edge_dim"])
        log.info("  Expression target: %d genes", m["n_expr_genes"])
        log.info("  Node split: train=%d / val=%d / test=%d",
                 m["node_train"], m["node_val"], m["node_test"])
        log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    return torch.tensor(x, dtype=torch.float32)


def _to_long_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.int64))
    return torch.tensor(x, dtype=torch.long)


def _to_bool_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.bool()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(bool))
    return torch.tensor(x, dtype=torch.bool)


def _cell_type_tensor(cell_types) -> Optional[torch.Tensor]:
    if cell_types is None:
        return None
    if isinstance(cell_types, torch.Tensor):
        return cell_types.long()
    arr = np.array(cell_types)
    if arr.dtype.kind in ("U", "S", "O"):
        # String → integer encoding
        uniq, inv = np.unique(arr, return_inverse=True)
        return torch.from_numpy(inv.astype(np.int64))
    return torch.from_numpy(arr.astype(np.int64))


def _load_sparse_label(step_dict: dict, key: str) -> Optional[torch.Tensor]:
    """Load a label array that may be scipy sparse or dense numpy/tensor."""
    import scipy.sparse as sp_mod

    val = step_dict.get(key)
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.float()
    if sp_mod.issparse(val):
        return torch.from_numpy(val.toarray().astype(np.float32))
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val.astype(np.float32))
    return None
