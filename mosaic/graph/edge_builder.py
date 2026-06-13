"""
mosaic/graph/edge_builder.py

Builds all 7 edge types for the heterogeneous MOSAIC graph.
LR database and metabolite database are kept SEPARATE throughout.

Edge types:
  (cell, contact,    cell)      τ₁  LR-contact, dist≤87µm
  (cell, secreted,   cell)      τ₁  LR-secreted, dist≤150µm
  (cell, metabolite, cell)      τ₂  scFEA flux-mediated, dist≤150µm  [SEPARATE from LR]
  (cell, intracellular, cell)   τ₃  self-loops: receptor PCA + flux
  (cell, expresses,  gene)      ε₁  top-K expression per cell
  (gene, interacts,  gene)      ε₂  LR pair edges — KEY for direct CCC scoring
  (cell, flux,       metabolite)ε₃  scFEA flux > q50 [SEPARATE from LR]

Edge attr dimensions:
  τ₁, τ₁     [E, 2]   [dist_norm, gaussian_weight]
  τ₂         [E, 3]   [n_active_modules_norm, mean_flux_norm, dist_norm]
  τ₃         [N, d4]  [receptor_pca(32), scfea_flux(n_modules)]
  ε₁         [E, 3]   [expr_norm, is_ligand, is_receptor]
  ε₂         [E, 3]   [is_contact, is_secreted, is_ecm]
  ε₃         [E, 2]   [flux_norm, module_activity_frac]
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

# Reuse src4 typed edge builders — no duplication
from mosaic.graph.typed_edge_builder import (
    build_contact_edges,
    build_secreted_edges,
    build_metabolite_edges,
)


# ─────────────────────────────────────────────────────────────────────────────
# EdgeBuilder class
# ─────────────────────────────────────────────────────────────────────────────

class EdgeBuilder:
    """
    Builds all 7 edge types for the MOSAIC heterogeneous graph.

    Args:
        cfg: config dict (spatial section used for thresholds)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sp = cfg.get("spatial", {})
        self.contact_thr    = float(sp.get("contact_threshold_um",    87.0))
        self.secreted_thr   = float(sp.get("secreted_threshold_um",  150.0))
        self.met_thr        = float(sp.get("metabolite_threshold_um", 150.0))
        self.flux_q_edge    = float(cfg.get("metabolite_database", {})
                                       .get("flux_quantile_edge", 0.50))
        self.req_gene       = bool(sp.get("require_gene_presence", True))

    # ─────────────────────────────────────────────────────────────────────────
    # τ₁ + τ₁  Cell-cell LR edges  (contact + secreted)
    # ─────────────────────────────────────────────────────────────────────────

    def build_cell_cell_edges(
        self,
        edge_index: torch.Tensor,
        dist_um: np.ndarray,
        lr_df: pd.DataFrame,
        gene_set: Set[str],
    ) -> dict:
        """
        Build τ₁ contact and τ₁ secreted cell-cell edges using the LR database.

        Returns dict with:
            contact_ei  [2, E_c]    contact edge index
            contact_ea  [E_c, 2]   [dist_norm, gaussian_weight]
            secreted_ei [2, E_s]   secreted edge index
            secreted_ea [E_s, 2]   [dist_norm, gaussian_weight]
        """
        gene_set_upper = {g.upper() for g in gene_set}

        log.info("  Building τ₁ contact edges (LR contact, dist≤%.0fµm)…", self.contact_thr)
        contact_ei, contact_ea = build_contact_edges(
            edge_index, dist_um, lr_df, gene_set_upper,
            threshold_um=self.contact_thr,
            require_gene_presence=self.req_gene,
        )

        log.info("  Building τ₁ secreted edges (LR secreted, dist≤%.0fµm)…", self.secreted_thr)
        secreted_ei, secreted_ea = build_secreted_edges(
            edge_index, dist_um, lr_df, gene_set_upper,
            threshold_um=self.secreted_thr,
            require_gene_presence=self.req_gene,
        )

        log.info("  τ₁ contact: %d edges  τ₁ secreted: %d edges",
                 contact_ei.shape[1], secreted_ei.shape[1])

        return {
            "contact_ei":  contact_ei,
            "contact_ea":  contact_ea,
            "secreted_ei": secreted_ei,
            "secreted_ea": secreted_ea,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # τ₂  Cell-cell metabolite edges  (flux-based, SEPARATE from LR)
    # ─────────────────────────────────────────────────────────────────────────

    def build_metabolite_cell_edges(
        self,
        edge_index: torch.Tensor,
        dist_um: np.ndarray,
        flux_matrix: np.ndarray,
        module_names: List[str],
        module_receptor_map: Dict[str, List[str]],
        met_gene_set: Set[str],
    ) -> dict:
        """
        Build τ₂ metabolite-mediated cell-cell edges using scFEA flux.
        SEPARATE from LR database.

        Returns dict with:
            met_cc_ei  [2, E_m]   edge index
            met_cc_ea  [E_m, 3]   [n_active_modules_norm, mean_flux_norm, dist_norm]
        """
        gene_set_upper = {g.upper() for g in met_gene_set}

        log.info("  Building τ₂ metabolite cell-cell edges (flux>q%.0f, dist≤%.0fµm)…",
                 self.flux_q_edge * 100, self.met_thr)

        met_ei, met_ea = build_metabolite_edges(
            edge_index=edge_index,
            dist_um=dist_um,
            scfea_flux=flux_matrix,
            module_receptor_map=module_receptor_map,
            module_names=module_names,
            expr_gene_set=gene_set_upper,
            threshold_um=self.met_thr,
            flux_quantile=self.flux_q_edge,
        )

        log.info("  τ₂ metabolite cell-cell: %d edges", met_ei.shape[1])
        return {"met_cc_ei": met_ei, "met_cc_ea": met_ea}

    # ─────────────────────────────────────────────────────────────────────────
    # τ₃  Intracellular self-loop edges
    # ─────────────────────────────────────────────────────────────────────────

    def build_intracellular_edges(
        self,
        expr_matrix: np.ndarray,
        gene_names: List[str],
        receptor_genes: List[str],
        scfea_flux: np.ndarray,
        receptor_pca_dim: int = 32,
    ) -> dict:
        """
        Build τ₃ intracellular self-loop edges.
        Features: receptor binary expression (PCA-compressed) + scFEA flux.

        Returns dict with:
            intra_ei  [2, N]    self-loops
            intra_ea  [N, d4]   [receptor_pca(32), flux_norm(n_modules)]
        """
        from mosaic.graph.intracellular_edge_builder import (
            build_intracellular_edges as _build_intra,
            _pca_compress,
            _normalize_flux,
        )
        import scipy.sparse as sp

        n_cells = expr_matrix.shape[0] if hasattr(expr_matrix, "shape") else len(expr_matrix)

        # Build binary receptor expression matrix
        gene_to_idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
        rec_upper = [r.upper() for r in receptor_genes if r.upper() in gene_to_idx]
        if not rec_upper:
            log.warning("  τ₃: no receptor genes found in dataset — using zero receptor features")
            rec_feats = np.zeros((n_cells, 1), dtype=np.float32)
        else:
            rec_indices = [gene_to_idx[r] for r in rec_upper]
            if sp.issparse(expr_matrix):
                rec_expr = expr_matrix[:, rec_indices].toarray().astype(np.float32)
            else:
                rec_expr = np.asarray(expr_matrix[:, rec_indices], dtype=np.float32)
            gene_means = rec_expr.mean(axis=0)
            rec_feats = (rec_expr > gene_means[None, :]).astype(np.float32)
            log.info("  τ₃ receptor features: %d cells × %d receptors, "
                     "mean_active=%.3f", n_cells, rec_feats.shape[1], rec_feats.mean())

        edge_index, edge_attr = _build_intra(
            n_cells=n_cells,
            receptor_features=rec_feats,
            scfea_flux=scfea_flux,
            receptor_pca_dim=receptor_pca_dim,
        )

        log.info("  τ₃ intracellular: %d self-loops, attr_dim=%d",
                 edge_index.shape[1], edge_attr.shape[1])
        return {"intra_ei": edge_index, "intra_ea": edge_attr}

    # ─────────────────────────────────────────────────────────────────────────
    # ε₁  Cell→gene expression edges
    # ─────────────────────────────────────────────────────────────────────────

    def build_cell_gene_edges(
        self,
        expr_matrix: np.ndarray,
        gene_names_dataset: List[str],
        gene_vocab: List[str],
        top_k: int = 500,
        min_expr_quantile: float = 0.5,
    ) -> dict:
        """
        Build (cell, expresses, gene) edges.

        For each cell, keep the top-K expressed genes that are in gene_vocab
        AND above the gene's mean expression (non-circular — uses presence, not level).

        Args:
            expr_matrix:         [N, G_dataset] expression (log-normalized or raw)
            gene_names_dataset:  list of G_dataset gene names
            gene_vocab:          list of interaction gene names (subset; node indices)
            top_k:               max genes per cell
            min_expr_quantile:   minimum expression quantile to include (per-gene)

        Returns dict with:
            cell_gene_ei  [2, E_cg]   cell_idx, gene_vocab_idx
            cell_gene_ea  [E_cg, 3]   [expr_norm, is_ligand, is_receptor]
            gene_vocab:   list[str]   (echo back for cross-checks)
        """
        import scipy.sparse as sp

        dataset_gene_upper = {str(g).upper(): i for i, g in enumerate(gene_names_dataset)}
        vocab_upper = [g.upper() for g in gene_vocab]

        # Indices of gene_vocab genes within dataset columns
        vocab_col_indices = []  # position in dataset matrix → gene_vocab_idx
        vocab_in_dataset  = []  # which gene_vocab genes are actually in dataset
        for vi, g in enumerate(vocab_upper):
            col = dataset_gene_upper.get(g)
            if col is not None:
                vocab_col_indices.append((col, vi))
                vocab_in_dataset.append(vi)

        if not vocab_col_indices:
            log.warning("  ε₁: no gene_vocab genes found in dataset — empty edges")
            return {
                "cell_gene_ei": torch.zeros((2, 0), dtype=torch.long),
                "cell_gene_ea": torch.zeros((0, 3), dtype=torch.float),
                "gene_vocab": gene_vocab,
            }

        col_idx_arr  = np.array([c for c, _ in vocab_col_indices])  # [V_present]
        vocab_idx_arr = np.array([v for _, v in vocab_col_indices])  # [V_present]

        # Extract expression submatrix [N, V_present]
        if sp.issparse(expr_matrix):
            sub_expr = expr_matrix[:, col_idx_arr].toarray().astype(np.float32)
        else:
            sub_expr = np.asarray(expr_matrix[:, col_idx_arr], dtype=np.float32)

        # Per-gene threshold: expression > gene mean (above-average expression)
        gene_means = sub_expr.mean(axis=0)   # [V_present]
        global_max = sub_expr.max() if sub_expr.max() > 0 else 1.0

        n_cells     = sub_expr.shape[0]
        n_vocab_present = sub_expr.shape[1]

        cell_indices = []
        gene_indices = []
        expr_vals    = []

        for cell_i in range(n_cells):
            row = sub_expr[cell_i]  # [V_present]

            # Require expression > gene mean (gene-level threshold, non-circular)
            above_mean = row > gene_means    # [V_present] bool

            if not above_mean.any():
                continue

            # Among qualifying genes, take top-K by expression value
            qualifying = np.where(above_mean)[0]
            if len(qualifying) > top_k:
                vals = row[qualifying]
                order = np.argpartition(-vals, top_k)[:top_k]
                qualifying = qualifying[order]

            cell_indices.extend([cell_i] * len(qualifying))
            gene_indices.extend(vocab_idx_arr[qualifying].tolist())
            expr_vals.extend(row[qualifying].tolist())

        if not cell_indices:
            log.warning("  ε₁: no cell-gene edges generated (all cells below threshold)")
            return {
                "cell_gene_ei": torch.zeros((2, 0), dtype=torch.long),
                "cell_gene_ea": torch.zeros((0, 3), dtype=torch.float),
                "gene_vocab": gene_vocab,
            }

        cell_gene_ei = torch.tensor(
            [cell_indices, gene_indices], dtype=torch.long
        )

        # Edge attributes: [expr_norm, is_ligand, is_receptor]
        # is_ligand / is_receptor are per gene-vocab node, broadcast from gene metadata
        expr_arr  = np.array(expr_vals, dtype=np.float32)
        expr_norm = expr_arr / global_max

        # We'll fill is_ligand/is_receptor as zeros here — they are graph-level
        # node properties set in assembler; keep 3-dim for compatibility
        edge_attr = torch.tensor(
            np.stack([expr_norm,
                      np.zeros_like(expr_norm),
                      np.zeros_like(expr_norm)], axis=1),
            dtype=torch.float,
        )

        log.info("  ε₁ cell→gene: %d edges (%d cells × ~%.1f genes avg)",
                 cell_gene_ei.shape[1], n_cells, cell_gene_ei.shape[1] / max(n_cells, 1))

        return {
            "cell_gene_ei": cell_gene_ei,
            "cell_gene_ea": edge_attr,
            "gene_vocab":   gene_vocab,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ε₂  Gene↔gene LR interaction edges  (KEY for direct CCC scoring)
    # ─────────────────────────────────────────────────────────────────────────

    def build_gene_interaction_edges(
        self,
        lr_df: pd.DataFrame,
        gene_vocab: List[str],
    ) -> dict:
        """
        Build (gene, interacts, gene) edges — one per LR pair in the database.

        This is the KEY edge type: attention on these edges directly gives
        LR pair CCC scores without any post-hoc correction.

        Args:
            lr_df:       LR database with columns: ligand, receptor, channel_type
            gene_vocab:  ordered list of gene names (gene node indices)

        Returns dict with:
            lr_gene_ei  [2, n_lr]   ligand_gene_idx, receptor_gene_idx
            lr_gene_ea  [n_lr, 3]   [is_contact, is_secreted, is_ecm]
            lr_pair_vocab  list[tuple]  (ligand, receptor) in edge order
        """
        vocab_upper = {g.upper(): i for i, g in enumerate(gene_vocab)}

        src_list  = []
        dst_list  = []
        attr_list = []
        pair_vocab = []

        channel_types = sorted(lr_df["channel_type"].dropna().unique())
        log.info("  ε₂ LR pairs — channel types in db: %s", channel_types)

        for _, row in lr_df.iterrows():
            lig = str(row["ligand"]).upper()
            rec = str(row["receptor"]).upper()
            ct  = str(row.get("channel_type", "secreted")).lower()

            lig_idx = vocab_upper.get(lig)
            rec_idx = vocab_upper.get(rec)

            if lig_idx is None or rec_idx is None:
                continue  # LR pair not in gene vocab (gene absent from dataset)

            is_contact  = float("contact"  in ct)
            is_secreted = float("secreted" in ct)
            is_ecm      = float("ecm"      in ct)

            src_list.append(lig_idx)
            dst_list.append(rec_idx)
            attr_list.append([is_contact, is_secreted, is_ecm])
            pair_vocab.append((str(row["ligand"]), str(row["receptor"])))

        if not src_list:
            log.warning("  ε₂: no LR pairs mapped to gene vocab — empty edges")
            return {
                "lr_gene_ei":  torch.zeros((2, 0), dtype=torch.long),
                "lr_gene_ea":  torch.zeros((0, 3), dtype=torch.float),
                "lr_pair_vocab": [],
            }

        lr_gene_ei = torch.tensor([src_list, dst_list], dtype=torch.long)
        lr_gene_ea = torch.tensor(attr_list, dtype=torch.float)

        log.info("  ε₂ gene→gene (LR): %d interaction edges (%d/%d pairs mapped to vocab)",
                 lr_gene_ei.shape[1], lr_gene_ei.shape[1], len(lr_df))

        return {
            "lr_gene_ei":   lr_gene_ei,
            "lr_gene_ea":   lr_gene_ea,
            "lr_pair_vocab": pair_vocab,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ε₃  Cell→metabolite flux edges  (SEPARATE from LR)
    # ─────────────────────────────────────────────────────────────────────────

    def build_cell_metabolite_edges(
        self,
        flux_matrix: np.ndarray,
        module_names: List[str],
        met_vocab: List[str],
        flux_quantile: float = 0.50,
    ) -> dict:
        """
        Build (cell, flux, metabolite) edges.

        A cell→metabolite edge exists when the cell's scFEA flux for that module
        exceeds the module's q50 threshold.

        Args:
            flux_matrix:    [N, M_scfea] scFEA flux values
            module_names:   list of scFEA module names (length M_scfea)
            met_vocab:      list of metabolite node names (metabolite node indices)
            flux_quantile:  per-module threshold (default 0.50)

        Returns dict with:
            cell_met_ei  [2, E_cm]   cell_idx, met_vocab_idx
            cell_met_ea  [E_cm, 2]   [flux_norm, module_activity_frac]
        """
        met_vocab_upper = {m.upper(): i for i, m in enumerate(met_vocab)}
        mod_upper = [m.upper() for m in module_names]

        # Map scFEA module → met_vocab index
        mod_to_vocab = {}
        for m_idx, m in enumerate(mod_upper):
            vi = met_vocab_upper.get(m)
            if vi is not None:
                mod_to_vocab[m_idx] = vi

        if not mod_to_vocab:
            log.warning("  ε₃: no scFEA modules map to met_vocab — empty edges")
            return {
                "cell_met_ei": torch.zeros((2, 0), dtype=torch.long),
                "cell_met_ea": torch.zeros((0, 2), dtype=torch.float),
            }

        mapped_mod_indices = np.array(list(mod_to_vocab.keys()))    # [M_mapped]
        mapped_vocab_indices = np.array(list(mod_to_vocab.values())) # [M_mapped]

        flux_sub = flux_matrix[:, mapped_mod_indices]  # [N, M_mapped]

        # Per-module threshold
        thresholds = np.quantile(flux_sub, flux_quantile, axis=0)  # [M_mapped]

        # Global max for normalization
        global_max = flux_sub.max() if flux_sub.max() > 0 else 1.0

        n_cells   = flux_sub.shape[0]
        n_modules = flux_sub.shape[1]

        cell_indices = []
        met_indices  = []
        flux_vals    = []
        act_fracs    = []  # fraction of mapped modules active per cell

        for cell_i in range(n_cells):
            row = flux_sub[cell_i]
            above = row > thresholds

            if not above.any():
                continue

            active_m_indices = np.where(above)[0]
            act_frac = float(above.sum()) / n_modules

            for mi in active_m_indices:
                cell_indices.append(cell_i)
                met_indices.append(int(mapped_vocab_indices[mi]))
                flux_vals.append(float(row[mi]))
                act_fracs.append(act_frac)

        if not cell_indices:
            log.warning("  ε₃: no cell-metabolite edges above threshold")
            return {
                "cell_met_ei": torch.zeros((2, 0), dtype=torch.long),
                "cell_met_ea": torch.zeros((0, 2), dtype=torch.float),
            }

        cell_met_ei = torch.tensor([cell_indices, met_indices], dtype=torch.long)

        flux_arr  = np.array(flux_vals, dtype=np.float32)
        flux_norm = flux_arr / global_max
        act_arr   = np.array(act_fracs, dtype=np.float32)

        cell_met_ea = torch.tensor(
            np.stack([flux_norm, act_arr], axis=1),
            dtype=torch.float,
        )

        log.info("  ε₃ cell→metabolite: %d edges (%d cells × ~%.1f modules avg)",
                 cell_met_ei.shape[1], n_cells, cell_met_ei.shape[1] / max(n_cells, 1))

        return {
            "cell_met_ei": cell_met_ei,
            "cell_met_ea": cell_met_ea,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ε₄  metabolite → gene  (sensed_by)
    # ─────────────────────────────────────────────────────────────────────────

    def build_metabolite_gene_edges(
        self,
        module_names: List[str],
        module_receptor_map: dict,
        met_vocab: List[str],
        gene_vocab: List[str],
    ) -> dict:
        """
        Build (metabolite, sensed_by, gene) edges from M_R database.

        Each edge connects a metabolite module node to a receptor gene node
        that senses that metabolite, enabling direct metabolite→gene
        information flow through message passing.

        Args:
            module_names:        list of scFEA module names (length M_scfea)
            module_receptor_map: dict {module_name: [receptor_gene, ...]}
            met_vocab:           list of metabolite node names (node indices)
            gene_vocab:          list of gene node names (node indices)

        Returns dict with:
            met_gene_ei  [2, E_mg]   met_vocab_idx, gene_vocab_idx
            met_gene_ea  [E_mg, 1]   [n_receptors_norm]
        """
        met_vocab_upper = {m.upper(): i for i, m in enumerate(met_vocab)}
        gene_vocab_upper = {g.upper(): i for i, g in enumerate(gene_vocab)}

        met_indices = []
        gene_indices = []

        for mod_name, receptors in module_receptor_map.items():
            if not receptors:
                continue
            mi = met_vocab_upper.get(mod_name.upper())
            if mi is None:
                continue
            for rec in receptors:
                gi = gene_vocab_upper.get(rec.upper())
                if gi is not None:
                    met_indices.append(mi)
                    gene_indices.append(gi)

        if not met_indices:
            log.warning("  ε₄: no metabolite-gene edges — empty M_R overlap")
            return {
                "met_gene_ei": torch.zeros((2, 0), dtype=torch.long),
                "met_gene_ea": torch.zeros((0, 1), dtype=torch.float),
            }

        met_gene_ei = torch.tensor([met_indices, gene_indices], dtype=torch.long)

        # Edge attribute: number of receptors per metabolite (normalized)
        # Counts how many receptors this metabolite connects to (connectivity degree)
        from collections import Counter
        met_counts = Counter(met_indices)
        max_count = max(met_counts.values())
        attr_vals = [met_counts[mi] / max_count for mi in met_indices]
        met_gene_ea = torch.tensor(attr_vals, dtype=torch.float).unsqueeze(1)

        n_unique_met = len(set(met_indices))
        n_unique_gene = len(set(gene_indices))
        log.info("  ε₄ metabolite→gene (sensed_by): %d edges (%d metabolites → %d receptors)",
                 met_gene_ei.shape[1], n_unique_met, n_unique_gene)

        return {
            "met_gene_ei": met_gene_ei,
            "met_gene_ea": met_gene_ea,
        }
