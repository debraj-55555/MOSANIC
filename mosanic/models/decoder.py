"""
Expression Decoder for MOSANIC.

Predicts gene expression from node embeddings -- the PRIMARY task.

For each cell i:
    y_expr[i] = MLP(h_i)

where h_i incorporates spatial CCC context from the HetGT encoder.
The model must learn: spatial_neighborhood(scVI_embeddings) -> gene_expression

Architecture:
    MLP: hidden_dim -> 512 -> 256 -> n_genes
    With LayerNorm, GELU, Dropout at each layer
"""

import torch
import torch.nn as nn
from typing import List


class ExpressionDecoder(nn.Module):
    """
    MLP decoder for node-level gene expression prediction.

    Args:
        hidden_dim: Input dimension from encoder
        n_genes: Number of target genes (output dimension)
        decoder_dims: Hidden layer dimensions for the MLP
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_genes: int = 200,
        decoder_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if decoder_dims is None:
            decoder_dims = [512, 256]

        self.n_genes = n_genes

        layers = []
        in_dim = hidden_dim
        for h_dim in decoder_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final output layer -- no activation (expression can be any non-negative value)
        layers.append(nn.Linear(in_dim, n_genes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict gene expression for each node.

        Args:
            node_embeddings: [N, hidden_dim] from HetGT encoder

        Returns:
            predicted_expression: [N, n_genes]
        """
        return self.mlp(node_embeddings)
