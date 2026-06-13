"""
mosaic/models/mosaic_model.py

MOSAIC: Heterogeneous Graph Transformer for Cell-Cell Communication.

Wraps HetGTEncoder (cell + gene + metabolite nodes, 7 edge types)
and ExpressionDecoder (primary task: cell gene expression prediction).

Pipeline:
    HeteroData (3 node types, 7 edge types)
        |
    HetGTEncoder
        cell [N, 128] -> [N, hidden_dim]
        gene [G, 1280] -> [G, hidden_dim]    (message passing through LR pair edges)
        metabolite [M, 600] -> [M, hidden_dim]
        |
    cell_embeddings [N, hidden_dim]
        |
    ExpressionDecoder
        |
    expression [N, n_genes]   <- PRIMARY prediction target

Key improvement over src4:
    - Attention on (gene, interacts, gene) edges = direct LR pair CCC score
    - No post-hoc gene-weighting correction needed
    - Gene identity (ESM-2 embeddings) visible to GNN -> richer semantics
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import HetGTEncoder
from .decoder import ExpressionDecoder


class MOSAIC(nn.Module):
    """
    MOSAIC model: HetGTEncoder + ExpressionDecoder.

    Args:
        node_in_dims:   {node_type: in_dim}  (default: cell=128, gene=1280, met=600)
        n_expr_genes:   number of expression target genes (default 200)
        edge_type_dims: {edge_type_tuple: edge_attr_dim} (uses encoder defaults if None)
        config:         model config dict with keys:
                          model.hidden_dim, model.n_heads, model.n_layers,
                          model.dropout, model.ffn_ratio,
                          model.decoder_dims, model.decoder_dropout
    """

    def __init__(
        self,
        node_in_dims: Optional[Dict[str, int]] = None,
        n_expr_genes: int = 200,
        edge_type_dims: Optional[Dict[Tuple[str, str, str], int]] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__()

        cfg = config or {}
        model_cfg = cfg.get("model", cfg)  # support both flat and nested configs

        hidden_dim      = int(model_cfg.get("hidden_dim",      256))
        n_heads         = int(model_cfg.get("n_heads",         4))
        n_layers        = int(model_cfg.get("n_layers",        2))
        dropout         = float(model_cfg.get("dropout",       0.1))
        ffn_ratio       = float(model_cfg.get("ffn_ratio",     4.0))
        decoder_dims    = list(model_cfg.get("decoder_dims",   [256]))
        decoder_dropout = float(model_cfg.get("decoder_dropout", 0.2))

        # -- Encoder ----------------------------------------------------
        self.encoder = HetGTEncoder(
            node_in_dims=node_in_dims,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            edge_type_dims=edge_type_dims,
            dropout=dropout,
            ffn_ratio=ffn_ratio,
        )

        # -- Expression Decoder (primary task) --------------------------
        self.expression_decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            n_genes=n_expr_genes,
            decoder_dims=decoder_dims,
            dropout=decoder_dropout,
        )

        self.hidden_dim   = hidden_dim
        self.n_expr_genes = n_expr_genes

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        data,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass on HeteroData.

        Args:
            data: PyG HeteroData with:
                data['cell'].x:       [N, 128]
                data['gene'].x:       [G, 1280]
                data['metabolite'].x: [M, 600]
                data[edge_type].edge_index / edge_attr for each of 7 edge types
            return_attention: if True, return attention weights and gate weights

        Returns:
            {
                'expression':      [N, n_genes]    <- PRIMARY prediction
                'node_embeddings': [N, hidden_dim] <- for L3 clustering
                'attention_info':  dict            <- only if return_attention
                    {
                        'per_layer':    list of {edge_type: [E_t, n_heads]}
                        'gate_weights': list of {dst_node_type: [n_edge_types]}
                    }
            }
        """
        x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(data)

        # Encode
        if return_attention:
            cell_emb, attention_info = self.encoder(
                x_dict, edge_index_dict, edge_attr_dict,
                return_attention=True,
            )
        else:
            cell_emb = self.encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Predict expression
        expr_pred = self.expression_decoder(cell_emb)   # [N, n_genes]

        result = {
            "expression":      expr_pred,
            "node_embeddings": cell_emb,
        }

        if return_attention:
            result["attention_info"] = attention_info

        return result

    def encode(self, data, return_attention: bool = False):
        """Get cell embeddings only (for inference / evaluation)."""
        x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(data)
        return self.encoder(
            x_dict, edge_index_dict, edge_attr_dict,
            return_attention=return_attention,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_graph_data(data):
        """Extract x_dict, edge_index_dict, edge_attr_dict from HeteroData."""
        x_dict = {}
        for nt in data.node_types:
            x_dict[nt] = data[nt].x

        edge_index_dict = {}
        edge_attr_dict  = {}
        for et in data.edge_types:
            store = data[et]
            edge_index_dict[et] = store.edge_index
            ea = getattr(store, "edge_attr", None)
            edge_attr_dict[et]  = ea

        return x_dict, edge_index_dict, edge_attr_dict

    def count_parameters(self) -> Dict[str, int]:
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.expression_decoder.parameters())
        return {
            "encoder":            enc_params,
            "expression_decoder": dec_params,
            "total":              enc_params + dec_params,
        }

    def __repr__(self):
        param_counts = self.count_parameters()
        return (
            f"MOSAIC(\n"
            f"  encoder:  HetGTEncoder(hidden={self.hidden_dim}, "
            f"layers={self.encoder.n_layers})\n"
            f"  decoder:  ExpressionDecoder(->{self.n_expr_genes} genes)\n"
            f"  params:   encoder={param_counts['encoder']:,}  "
            f"decoder={param_counts['expression_decoder']:,}  "
            f"total={param_counts['total']:,}\n"
            f")"
        )


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

def build_model(cfg: dict, n_expr_genes: int, graph_metadata: dict = None) -> "MOSAIC":
    """
    Instantiate MOSAIC from config dict.

    Args:
        cfg:           config dict (supports both flat and nested 'model' key)
        n_expr_genes:  number of expression target genes from preprocessing
        graph_metadata: optional metadata dict to auto-infer edge_type_dims

    Returns:
        MOSAIC model (not yet moved to device)
    """
    edge_type_dims = None

    if graph_metadata is not None:
        # Build edge_type_dims from graph metadata
        key_map = {
            ("cell", "contact",       "cell"):       "contact_edge_dim",
            ("cell", "secreted",      "cell"):       "secreted_edge_dim",
            ("cell", "metabolite",    "cell"):       "met_cc_edge_dim",
            ("cell", "intracellular", "cell"):       "intra_edge_dim",
            ("cell", "expresses",     "gene"):       "cell_gene_edge_dim",
            ("gene", "interacts",     "gene"):       "lr_gene_edge_dim",
            ("cell", "flux",          "metabolite"): "cell_met_edge_dim",
            ("metabolite", "sensed_by", "gene"):       "met_gene_edge_dim",
        }
        # Only include edge types that have >0 edges in the graph
        n_edge_keys = {
            ("cell", "contact",       "cell"):       "n_contact_edges",
            ("cell", "secreted",      "cell"):       "n_secreted_edges",
            ("cell", "metabolite",    "cell"):       "n_met_cc_edges",
            ("cell", "intracellular", "cell"):       "n_intra_edges",
            ("cell", "expresses",     "gene"):       "n_cell_gene_edges",
            ("gene", "interacts",     "gene"):       "n_lr_gene_edges",
            ("cell", "flux",          "metabolite"): "n_cell_met_edges",
            ("metabolite", "sensed_by", "gene"):     "n_met_gene_edges",
        }
        dims = {}
        for et, meta_key in key_map.items():
            d = graph_metadata.get(meta_key)
            if d is not None and int(d) > 0:
                # Check if this edge type actually has edges
                n_key = n_edge_keys.get(et)
                n_edges = int(graph_metadata.get(n_key, 1)) if n_key else 1
                if n_edges > 0:
                    dims[et] = int(d)
        if dims:
            edge_type_dims = dims

    node_features = cfg.get("node_features", {})
    n_mets = int(graph_metadata.get("n_metabolites", 1)) if graph_metadata else 1
    node_in_dims = {
        "cell":       int(node_features.get("cell_dim",       128)),
        "gene":       int(node_features.get("gene_dim",       1280)),
    }
    if n_mets > 0:
        node_in_dims["metabolite"] = int(node_features.get("metabolite_dim", 600))

    model = MOSAIC(
        node_in_dims=node_in_dims,
        n_expr_genes=n_expr_genes,
        edge_type_dims=edge_type_dims,
        config=cfg,
    )

    return model
