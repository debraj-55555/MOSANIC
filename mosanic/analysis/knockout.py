"""
mosanic/analysis/knockout.py

Perturbation-based CCC scorer for MOSANIC.

Measures the causal effect of edge removal on expression prediction:
  1. Per-edge-type ablation: remove all edges of one type, measure delta-R^2
  2. Per-gene channel importance: per-gene delta-R^2 when ablating each channel

This is more robust than attention: it directly tests what the model
actually uses, not just where it "looks."
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def channel_ablation(
    model: torch.nn.Module,
    data,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """
    Remove all edges of each type and measure expression prediction change.

    For each edge type tau:
      1. Zero out all tau edges (replace edge_index with empty)
      2. Run model forward
      3. Compare expression predictions to full-model predictions

    Returns:
        dict with:
            'channel_importance': {edge_type: delta_mse}
            'full_mse': float  (MSE of full model)
            'predictions_full': [N, G]
            'predictions_ablated': {edge_type: [N, G]}
    """
    model.eval()
    data = data.to(device)

    # Full model prediction
    full_result = model(data)
    full_pred = full_result['expression'].cpu()
    y_true = data['cell'].y_expr.cpu()
    test_mask = data['cell'].test_mask.cpu()

    full_pred_test = np.clip(full_pred[test_mask].numpy(), 0, None)
    y_true_test = y_true[test_mask].numpy()
    full_mse = float(np.mean((full_pred_test - y_true_test) ** 2))

    channel_importance = {}
    predictions_ablated = {}

    for ablate_type in data.edge_types:
        # Create modified data with this edge type removed
        # We do this by making a copy of edge_index/attr dicts
        x_dict, edge_index_dict, edge_attr_dict = model._extract_graph_data(data)

        # Replace the ablated edge type with empty tensors
        empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_dim = edge_attr_dict[ablate_type].shape[1] if ablate_type in edge_attr_dict else 2
        empty_ea = torch.zeros(0, edge_dim, device=device)

        edge_index_dict_mod = dict(edge_index_dict)
        edge_attr_dict_mod = dict(edge_attr_dict)
        edge_index_dict_mod[ablate_type] = empty_ei
        edge_attr_dict_mod[ablate_type] = empty_ea

        # Forward through encoder + decoder manually
        node_emb = model.encoder(x_dict, edge_index_dict_mod, edge_attr_dict_mod)
        ablated_pred = model.expression_decoder(node_emb).cpu()

        ablated_pred_test = np.clip(ablated_pred[test_mask].numpy(), 0, None)
        ablated_mse = float(np.mean((ablated_pred_test - y_true_test) ** 2))

        delta_mse = ablated_mse - full_mse  # positive = this channel helps
        channel_importance[ablate_type] = {
            'delta_mse': delta_mse,
            'ablated_mse': ablated_mse,
            'relative_importance': delta_mse / max(full_mse, 1e-8),
        }

        predictions_ablated[ablate_type] = ablated_pred

        logger.info(
            "  Ablate %15s: MSE %.4f (delta=%+.4f, %+.1f%%)",
            ablate_type[1], ablated_mse,
            delta_mse, delta_mse / max(full_mse, 1e-8) * 100
        )

    logger.info("  Full model MSE: %.4f", full_mse)

    return {
        'channel_importance': channel_importance,
        'full_mse': full_mse,
        'predictions_full': full_pred,
        'predictions_ablated': predictions_ablated,
    }


@torch.no_grad()
def per_gene_channel_importance(
    model: torch.nn.Module,
    data,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """
    Per-gene delta-R^2 when ablating each channel.

    For each gene g and each edge type tau:
      delta-R^2(g, tau) = R^2_full(g) - R^2_ablated(g, tau)

    Positive = this channel helps predict this gene.

    Returns:
        {edge_type: [n_genes] delta_r2_per_gene}
    """
    from sklearn.metrics import r2_score

    model.eval()
    data = data.to(device)

    test_mask = data['cell'].test_mask.cpu()
    y_true = data['cell'].y_expr.cpu()[test_mask].numpy()

    # Full model
    full_pred = model(data)['expression'].cpu()
    full_pred_test = np.clip(full_pred[test_mask].numpy(), 0, None)

    n_genes = y_true.shape[1]

    # Full R^2 per gene
    full_r2 = np.zeros(n_genes)
    for g in range(n_genes):
        if y_true[:, g].std() < 1e-8:
            full_r2[g] = 0.0
        else:
            full_r2[g] = r2_score(y_true[:, g], full_pred_test[:, g])

    result = {}
    for ablate_type in data.edge_types:
        x_dict, edge_index_dict, edge_attr_dict = model._extract_graph_data(data)

        empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_dim = edge_attr_dict[ablate_type].shape[1]
        empty_ea = torch.zeros(0, edge_dim, device=device)

        edge_index_dict_mod = dict(edge_index_dict)
        edge_attr_dict_mod = dict(edge_attr_dict)
        edge_index_dict_mod[ablate_type] = empty_ei
        edge_attr_dict_mod[ablate_type] = empty_ea

        node_emb = model.encoder(x_dict, edge_index_dict_mod, edge_attr_dict_mod)
        ablated_pred = model.expression_decoder(node_emb).cpu()
        ablated_pred_test = np.clip(ablated_pred[test_mask].numpy(), 0, None)

        delta_r2 = np.zeros(n_genes)
        for g in range(n_genes):
            if y_true[:, g].std() < 1e-8:
                delta_r2[g] = 0.0
            else:
                abl_r2 = r2_score(y_true[:, g], ablated_pred_test[:, g])
                delta_r2[g] = full_r2[g] - abl_r2

        result[ablate_type] = delta_r2
        n_helped = (delta_r2 > 0.01).sum()
        logger.info(
            "  %15s: mean delta-R^2=%.4f, genes helped (delta-R^2>0.01): %d/%d",
            ablate_type[1], delta_r2.mean(), n_helped, n_genes
        )

    return {
        'full_r2_per_gene': full_r2,
        'delta_r2_per_gene': result,
    }


def run_knockout(
    model: torch.nn.Module,
    data,
    device: torch.device = torch.device('cpu'),
    per_gene: bool = False,
) -> Dict:
    """
    High-level API: run knockout/perturbation analysis on a trained MOSANIC model.

    Args:
        model: trained MOSANIC model
        data: HeteroData graph
        device: torch device
        per_gene: if True, also compute per-gene channel importance (slower)

    Returns:
        dict with channel_importance and optionally per_gene_importance
    """
    logger.info("Running channel ablation analysis...")
    result = channel_ablation(model, data, device)

    # Remove large tensor predictions from result for serialization
    output = {
        'channel_importance': result['channel_importance'],
        'full_mse': result['full_mse'],
    }

    if per_gene:
        logger.info("Running per-gene channel importance...")
        gene_result = per_gene_channel_importance(model, data, device)
        output['per_gene_importance'] = {
            'full_r2_per_gene': gene_result['full_r2_per_gene'].tolist(),
            'delta_r2_per_gene': {
                str(k): v.tolist() for k, v in gene_result['delta_r2_per_gene'].items()
            },
        }

    # Summary
    logger.info("\nKnockout Analysis Summary:")
    logger.info("  Full model MSE: %.4f", output['full_mse'])
    for et, imp in output['channel_importance'].items():
        logger.info("  %s: delta_mse=%+.4f (%.1f%%)",
                     et, imp['delta_mse'], imp['relative_importance'] * 100)

    return output
