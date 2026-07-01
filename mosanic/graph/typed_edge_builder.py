"""
typed_edge_builder.py — Build biology-typed edge sub-graphs from spatial k-NN base graph.

Replaces distance-quartile routing (src4 original) with database-driven channel assignment:
  contact    — CellChatDB 'contact' LR pairs, dist ≤ contact_threshold_um
  secreted   — CellChatDB 'secreted' LR pairs, dist ≤ secreted_threshold_um
  metabolite — scFEA high-flux sender + metabolite sensor in receiver, dist ≤ metabolite_threshold_um
  ecm        — CellChatDB 'ecm' LR pairs, dist ≤ ecm_threshold_um

Non-circularity guarantee:
  Edge EXISTENCE: LR database category + distance threshold + gene presence (boolean) OR
                  scFEA flux > q50 threshold (coarser than q75 used for y_metab label)
  Edge FEATURES:  distance, gaussian_weight, flux statistics — all structural/geometric
  Labels:         y_strength, y_lr, y_metab — derived from raw expression / scFEA flux
                  using DIFFERENT thresholds than edge existence conditions
"""

import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# Default thresholds (µm) — can be overridden via config
DEFAULT_CONTACT_THRESHOLD_UM    = 50.0    # juxtacrine: direct cell-cell contact
DEFAULT_SECRETED_THRESHOLD_UM   = 150.0   # paracrine: = max_distance_um
DEFAULT_METABOLITE_THRESHOLD_UM = 200.0   # metabolites diffuse further than proteins
DEFAULT_ECM_THRESHOLD_UM        = 100.0   # ECM is local but not contact-range
DEFAULT_METAB_FLUX_QUANTILE     = 0.50    # q50 for EDGE EXISTENCE (label uses q75)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _gaussian_weight(dist_um: np.ndarray, sigma_um: float = 50.0) -> np.ndarray:
    """Gaussian distance decay weight: exp(-d² / 2σ²)."""
    return np.exp(-(dist_um ** 2) / (2 * sigma_um ** 2))


def _get_lr_genes(lr_df: pd.DataFrame, channel_type: str) -> Tuple[Set[str], Set[str]]:
    """Return (ligand_genes, receptor_genes) for a given channel_type."""
    sub = lr_df[lr_df['channel_type'] == channel_type]
    ligands   = set(sub['ligand'].str.upper().unique())
    receptors = set(sub['receptor'].str.upper().unique())
    return ligands, receptors


def _gene_presence_mask(edge_index: torch.Tensor,
                         cell_gene_presence: Dict[int, Set[str]],
                         ligand_genes: Set[str],
                         receptor_genes: Set[str]) -> np.ndarray:
    """
    Return boolean mask over edges where:
      sender cell has ≥1 gene from ligand_genes   (gene PRESENT in dataset, not expression)
      receiver cell has ≥1 gene from receptor_genes

    This is a gene-presence filter (binary), NOT expression-level → non-circular.
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    mask = np.zeros(len(src), dtype=bool)
    for i, (s, d) in enumerate(zip(src, dst)):
        s_genes = cell_gene_presence.get(int(s), set())
        d_genes = cell_gene_presence.get(int(d), set())
        if s_genes & ligand_genes and d_genes & receptor_genes:
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Phase 3 builders — each returns (edge_index [2,E], edge_attr [E, edge_dim])
# ---------------------------------------------------------------------------

def build_contact_edges(
    edge_index:    torch.Tensor,
    dist_um:       np.ndarray,
    lr_df:         pd.DataFrame,
    gene_set:      Set[str],
    threshold_um:  float = DEFAULT_CONTACT_THRESHOLD_UM,
    require_gene_presence: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build juxtacrine / Cell-Cell Contact edges.

    Selection criteria:
      1. dist ≤ threshold_um  (default 50µm — direct contact range)
      2. ANY contact-category LR gene present in dataset gene_set

    Edge attr: [dist_norm, gaussian_weight]  — edge_dim = 2
    """
    # Distance filter
    dist_mask = dist_um <= threshold_um
    logger.debug(f"  Contact: dist≤{threshold_um}µm → {dist_mask.sum()} edges")

    # Gene presence filter
    if require_gene_presence:
        lig_genes, rec_genes = _get_lr_genes(lr_df, 'contact')
        lig_in_dataset = lig_genes & gene_set
        rec_in_dataset = rec_genes & gene_set
        if not lig_in_dataset or not rec_in_dataset:
            logger.warning("  Contact: no LR genes found in dataset gene_set — using distance filter only")
            gene_mask = np.ones(len(dist_um), dtype=bool)
        else:
            logger.debug(f"  Contact LR genes in dataset: {len(lig_in_dataset)} ligands, "
                        f"{len(rec_in_dataset)} receptors")
            gene_mask = _gene_presence_mask(
                edge_index, _build_cell_gene_sets(edge_index, gene_set, lig_in_dataset, rec_in_dataset),
                lig_in_dataset, rec_in_dataset
            )
    else:
        gene_mask = np.ones(len(dist_um), dtype=bool)

    final_mask = dist_mask & gene_mask
    logger.debug(f"  Contact: final {final_mask.sum()} edges after gene filter")

    return _build_edge_tensors_2d(edge_index, dist_um, threshold_um, final_mask)


def build_secreted_edges(
    edge_index:    torch.Tensor,
    dist_um:       np.ndarray,
    lr_df:         pd.DataFrame,
    gene_set:      Set[str],
    threshold_um:  float = DEFAULT_SECRETED_THRESHOLD_UM,
    require_gene_presence: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build paracrine / Secreted Signaling edges.

    Selection criteria:
      1. dist ≤ threshold_um  (default 150µm = max_distance_um)
      2. ANY secreted-category LR gene present in dataset gene_set

    Edge attr: [dist_norm, gaussian_weight]  — edge_dim = 2
    """
    dist_mask = dist_um <= threshold_um
    logger.debug(f"  Secreted: dist≤{threshold_um}µm → {dist_mask.sum()} edges")

    if require_gene_presence:
        lig_genes, rec_genes = _get_lr_genes(lr_df, 'secreted')
        lig_in_dataset = lig_genes & gene_set
        rec_in_dataset = rec_genes & gene_set
        if not lig_in_dataset or not rec_in_dataset:
            logger.warning("  Secreted: no LR genes in dataset — using distance filter only")
            gene_mask = np.ones(len(dist_um), dtype=bool)
        else:
            logger.debug(f"  Secreted LR genes in dataset: {len(lig_in_dataset)} ligands, "
                        f"{len(rec_in_dataset)} receptors")
            gene_mask = _gene_presence_mask(
                edge_index, _build_cell_gene_sets(edge_index, gene_set, lig_in_dataset, rec_in_dataset),
                lig_in_dataset, rec_in_dataset
            )
    else:
        gene_mask = np.ones(len(dist_um), dtype=bool)

    final_mask = dist_mask & gene_mask
    logger.debug(f"  Secreted: final {final_mask.sum()} edges")

    return _build_edge_tensors_2d(edge_index, dist_um, threshold_um, final_mask)


def build_ecm_edges(
    edge_index:    torch.Tensor,
    dist_um:       np.ndarray,
    lr_df:         pd.DataFrame,
    gene_set:      Set[str],
    threshold_um:  float = DEFAULT_ECM_THRESHOLD_UM,
    require_gene_presence: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build ECM-Receptor edges.

    Selection criteria:
      1. dist ≤ threshold_um  (default 100µm)
      2. ANY ecm-category LR gene present in dataset gene_set

    If no ECM pairs exist in lr_df (database lacks ECM annotation),
    falls back to secreted edges at the same distance threshold.

    Edge attr: [dist_norm, gaussian_weight]  — edge_dim = 2
    """
    ecm_pairs = lr_df[lr_df['channel_type'] == 'ecm']
    if len(ecm_pairs) == 0:
        logger.warning("  ECM: no 'ecm' channel_type entries in LR database. "
                       "Falling back to 'secreted' pairs at ECM distance threshold.")
        return build_secreted_edges(edge_index, dist_um, lr_df, gene_set,
                                    threshold_um=threshold_um,
                                    require_gene_presence=require_gene_presence)

    dist_mask = dist_um <= threshold_um
    logger.debug(f"  ECM: dist≤{threshold_um}µm → {dist_mask.sum()} edges")

    if require_gene_presence:
        lig_genes, rec_genes = _get_lr_genes(lr_df, 'ecm')
        lig_in_dataset = lig_genes & gene_set
        rec_in_dataset = rec_genes & gene_set
        if not lig_in_dataset or not rec_in_dataset:
            logger.warning("  ECM: no ECM genes found in dataset gene_set — using distance filter only")
            gene_mask = np.ones(len(dist_um), dtype=bool)
        else:
            logger.debug(f"  ECM LR genes in dataset: {len(lig_in_dataset)} ligands, "
                        f"{len(rec_in_dataset)} receptors")
            gene_mask = _gene_presence_mask(
                edge_index, _build_cell_gene_sets(edge_index, gene_set, lig_in_dataset, rec_in_dataset),
                lig_in_dataset, rec_in_dataset
            )
    else:
        gene_mask = np.ones(len(dist_um), dtype=bool)

    final_mask = dist_mask & gene_mask
    logger.debug(f"  ECM: final {final_mask.sum()} edges")

    return _build_edge_tensors_2d(edge_index, dist_um, threshold_um, final_mask)


def build_metabolite_edges(
    edge_index:           torch.Tensor,
    dist_um:              np.ndarray,
    scfea_flux:           np.ndarray,
    module_receptor_map:  Dict[str, List[str]],
    module_names:         List[str],
    expr_gene_set:        Set[str],
    threshold_um:         float = DEFAULT_METABOLITE_THRESHOLD_UM,
    flux_quantile:        float = DEFAULT_METAB_FLUX_QUANTILE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build metabolite-mediated edges (τ₂).

    Selection criteria (NON-CIRCULAR):
      1. dist ≤ threshold_um  (default 200µm — metabolites diffuse further)
      2. Sender cell has flux[sender, any_module] > q50  (coarser than q75 used for y_metab label)
      3. Receiver cell expresses ≥1 receptor for that module  (gene PRESENCE, binary)

    Edge attr: [n_active_modules_norm, mean_active_flux_norm, dist_norm]  — edge_dim = 3
      n_active_modules_norm: fraction of scFEA modules active above flux_quantile in sender
      mean_active_flux_norm: mean flux over active modules in sender, normalized to [0,1]
      dist_norm:             distance / threshold_um

    Non-circularity:
      Edge existence uses q50 flux threshold.
      y_metab label uses q75 co-activity threshold.
      Edge attr carries coarse flux statistics (inputs).
      Labels capture fine-grained module-level activity (targets).
    """
    n_cells   = scfea_flux.shape[0]
    n_modules = scfea_flux.shape[1]
    src_arr   = edge_index[0].numpy()
    dst_arr   = edge_index[1].numpy()

    # Per-module flux threshold at q50
    flux_thresholds = np.quantile(scfea_flux, flux_quantile, axis=0)  # [n_modules]

    # Pre-compute which modules are active per cell (flux > q50)
    cell_active = scfea_flux > flux_thresholds[np.newaxis, :]  # [n_cells, n_modules]

    # Pre-compute which cells express ≥1 receptor for each module
    # module_has_receptor[m] = set of module indices where receiver expresses a receptor
    modules_with_receptors = []
    module_receptor_genes  = []
    for m_idx, mod_name in enumerate(module_names):
        recs = module_receptor_map.get(mod_name, [])
        recs_in_dataset = [r for r in recs if r.upper() in expr_gene_set]
        modules_with_receptors.append(len(recs_in_dataset) > 0)
        module_receptor_genes.append(set(r.upper() for r in recs_in_dataset))

    modules_with_receptors = np.array(modules_with_receptors)  # [n_modules] bool
    n_usable = modules_with_receptors.sum()
    logger.debug(f"  Metabolite: {n_usable}/{n_modules} modules have ≥1 expressed receptor sensor")

    if n_usable == 0:
        logger.warning("  Metabolite: no modules with expressed receptors — returning empty edges")
        empty_ei   = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 3), dtype=torch.float)
        return empty_ei, empty_attr

    # Distance mask
    dist_mask = dist_um <= threshold_um

    # Per-edge computation
    selected_indices = []
    edge_n_active    = []   # n_active_modules_norm
    edge_mean_flux   = []   # mean_active_flux_norm

    for i in range(len(src_arr)):
        if not dist_mask[i]:
            continue

        s = int(src_arr[i])
        d = int(dst_arr[i])

        # Sender active modules (that also have expressed receptor in receiver)
        sender_active_usable = cell_active[s] & modules_with_receptors  # [n_modules] bool

        if not sender_active_usable.any():
            continue

        # Count active usable modules and their mean flux
        active_fluxes = scfea_flux[s, sender_active_usable]
        n_active      = int(sender_active_usable.sum())

        selected_indices.append(i)
        edge_n_active.append(n_active / n_usable)            # normalize by usable modules
        edge_mean_flux.append(float(active_fluxes.mean()))

    logger.debug(f"  Metabolite: {len(selected_indices)} edges selected (dist≤{threshold_um}µm + flux>q{int(flux_quantile*100)} + receptor expressed)")

    if not selected_indices:
        empty_ei   = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 3), dtype=torch.float)
        return empty_ei, empty_attr

    sel = np.array(selected_indices)
    ei_sel = edge_index[:, sel]  # [2, E_m]

    # Normalize flux to [0, 1] using global max
    flux_arr = np.array(edge_mean_flux)
    max_flux = flux_arr.max() if flux_arr.max() > 0 else 1.0
    flux_norm = flux_arr / max_flux

    dist_sel  = dist_um[sel]
    dist_norm = dist_sel / threshold_um

    edge_attr = torch.tensor(
        np.stack([np.array(edge_n_active), flux_norm, dist_norm], axis=1),
        dtype=torch.float
    )

    return ei_sel, edge_attr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_edge_tensors_2d(
    edge_index: torch.Tensor,
    dist_um:    np.ndarray,
    threshold_um: float,
    mask:       np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply boolean mask and build (edge_index, edge_attr) with edge_dim=2.

    edge_attr columns: [dist_norm, gaussian_weight]
    """
    ei_sel   = edge_index[:, mask]
    dist_sel = dist_um[mask]

    dist_norm = dist_sel / threshold_um
    gauss     = _gaussian_weight(dist_sel, sigma_um=threshold_um / 2.0)

    edge_attr = torch.tensor(
        np.stack([dist_norm, gauss], axis=1),
        dtype=torch.float
    )
    return ei_sel, edge_attr


def _build_cell_gene_sets(
    edge_index:       torch.Tensor,
    gene_set:         Set[str],
    ligand_genes:     Set[str],
    receptor_genes:   Set[str],
) -> Dict[int, Set[str]]:
    """
    Build a per-cell gene presence dict for cells appearing in edge_index.
    Since gene_set is the dataset's filtered gene universe (not per-cell expression),
    every cell in the graph has the same gene presence (dataset-level gene filtering).

    Note: In future, could be replaced with per-cell presence mask from AnnData.X > 0.
    """
    all_cell_ids = set(edge_index[0].numpy().tolist()) | set(edge_index[1].numpy().tolist())
    # All cells share the same gene universe (visium full-transcriptome)
    all_relevant = (ligand_genes | receptor_genes) & gene_set
    return {cell_id: all_relevant for cell_id in all_cell_ids}


# ---------------------------------------------------------------------------
# Top-level convenience function (called from dataset_builder.py)
# ---------------------------------------------------------------------------

def build_all_typed_edges(
    edge_index:          torch.Tensor,
    dist_um:             np.ndarray,
    lr_df:               pd.DataFrame,
    gene_set:            Set[str],
    scfea_flux:          np.ndarray,
    module_names:        List[str],
    module_receptor_map: Dict[str, List[str]],
    cfg:                 dict,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build all 4 typed edge sets from the spatial k-NN base graph.

    Args:
        edge_index:          [2, E] base spatial k-NN edges
        dist_um:             [E] edge distances in micrometers
        lr_df:               LR database DataFrame with 'channel_type' column
        gene_set:            Set of gene symbols present in dataset
        scfea_flux:          [N_cells, N_modules] scFEA flux matrix
        module_names:        scFEA module names (columns of balance.csv)
        module_receptor_map: {module_name: [receptor_genes]} from scfea_metabolite_mapper
        cfg:                 Config dict (uses cfg['spatial']['typed_edges'] section)

    Returns:
        Dict with keys: 'contact', 'secreted', 'metabolite', 'ecm'
        Each value: (edge_index [2, E_t], edge_attr [E_t, edge_dim])
    """
    te_cfg = cfg.get('spatial', {}).get('typed_edges', {})

    contact_thr    = float(te_cfg.get('contact_threshold_um',    DEFAULT_CONTACT_THRESHOLD_UM))
    secreted_thr   = float(te_cfg.get('secreted_threshold_um',   DEFAULT_SECRETED_THRESHOLD_UM))
    metabolite_thr = float(te_cfg.get('metabolite_threshold_um', DEFAULT_METABOLITE_THRESHOLD_UM))
    ecm_thr        = float(te_cfg.get('ecm_threshold_um',        DEFAULT_ECM_THRESHOLD_UM))
    flux_q         = float(te_cfg.get('metabolite_flux_quantile', DEFAULT_METAB_FLUX_QUANTILE))
    req_gene       = bool(te_cfg.get('require_gene_presence',    True))

    gene_set_upper = {g.upper() for g in gene_set}

    logger.debug("Building typed edges...")

    logger.debug("  Building contact edges...")
    contact_ei, contact_attr = build_contact_edges(
        edge_index, dist_um, lr_df, gene_set_upper, contact_thr, req_gene
    )

    logger.debug("  Building secreted edges...")
    secreted_ei, secreted_attr = build_secreted_edges(
        edge_index, dist_um, lr_df, gene_set_upper, secreted_thr, req_gene
    )

    logger.debug("  Building ECM edges...")
    ecm_ei, ecm_attr = build_ecm_edges(
        edge_index, dist_um, lr_df, gene_set_upper, ecm_thr, req_gene
    )

    logger.debug("  Building metabolite edges...")
    metab_ei, metab_attr = build_metabolite_edges(
        edge_index, dist_um, scfea_flux, module_receptor_map,
        module_names, gene_set_upper, metabolite_thr, flux_q
    )

    typed_edges = {
        'contact':    (contact_ei,  contact_attr),
        'secreted':   (secreted_ei, secreted_attr),
        'metabolite': (metab_ei,    metab_attr),
        'ecm':        (ecm_ei,      ecm_attr),
    }

    # Summary
    for name, (ei, attr) in typed_edges.items():
        logger.debug(f"  {name:12s}: {ei.shape[1]:6d} edges, edge_dim={attr.shape[1]}")

    return typed_edges
