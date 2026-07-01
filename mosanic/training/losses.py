"""
Loss functions for MOSANIC model.

Primary: Expression prediction loss (MSE on node-level gene expression)
Auxiliary: Edge-level CCC losses (strength, LR pairs, metabolites)

L_total = L_expr + ccc_weight * (a*L_strength + b*L_lr + g*L_metab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MOSANICLoss(nn.Module):
    """
    Combined loss for MOSANIC: expression prediction (primary) + CCC (auxiliary).

    Args:
        ccc_weight: Overall weight for auxiliary CCC losses relative to expression
        task_weights: Per-task weights for CCC losses
            {'strength': 1.0, 'lr_pairs': 0.5, 'metabolites': 0.3}
        expr_loss_type: 'mse' or 'huber' for expression prediction
        use_focal_loss: Use focal loss for LR/metabolite tasks
    """

    def __init__(
        self,
        ccc_weight: float = 0.1,
        task_weights: Optional[Dict[str, float]] = None,
        expr_loss_type: str = 'mse',
        use_focal_loss: bool = False,
    ):
        super().__init__()

        self.ccc_weight = ccc_weight

        if task_weights is None:
            task_weights = {
                'strength': 1.0,
                'lr_pairs': 0.5,
                'metabolites': 0.3,
            }
        self.task_weights = task_weights

        # Expression loss
        if expr_loss_type == 'huber':
            self.expr_loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)
        else:
            self.expr_loss_fn = nn.MSELoss(reduction='mean')

        # CCC task losses
        if use_focal_loss:
            from .losses import FocalLoss
            self.lr_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
            self.metab_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.lr_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            self.metab_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            predictions: Model output dict with keys:
                'expression': [N, G]
                'strength': [E, 1]  (optional)
                'lr_pairs': [E, n_lr]  (optional)
                'metabolites': [E, n_metab]  (optional)
            targets: Ground truth dict with keys:
                'expression': [N, G]
                'strength': [E]  (optional)
                'lr_pairs': [E, n_lr]  (optional)
                'metabolites': [E, n_metab]  (optional)
            node_mask: [N] bool -- which cells to evaluate expression on
            edge_mask: [E] bool -- which edges to evaluate CCC on

        Returns:
            total_loss: scalar
            loss_dict: {loss_name: value} for logging
        """
        loss_dict = {}
        device = predictions['expression'].device

        # -- PRIMARY: Expression prediction loss (node-level) -----------
        expr_pred = predictions['expression']
        expr_true = targets['expression']

        if node_mask is not None:
            expr_pred = expr_pred[node_mask]
            expr_true = expr_true[node_mask]

        loss_expr = self.expr_loss_fn(expr_pred, expr_true)
        loss_dict['loss_expr'] = loss_expr.item()

        # -- AUXILIARY: CCC losses (edge-level) -------------------------
        loss_ccc = torch.tensor(0.0, device=device)

        # Strength (regression)
        if 'strength' in predictions and targets.get('strength') is not None:
            s_pred = predictions['strength']
            s_true = targets['strength']
            if edge_mask is not None:
                s_pred = s_pred[edge_mask]
                s_true = s_true[edge_mask]
            s_pred = s_pred.squeeze()
            s_true = s_true.squeeze()
            loss_s = F.mse_loss(s_pred, s_true)
            loss_dict['loss_strength'] = loss_s.item()
            loss_ccc = loss_ccc + self.task_weights.get('strength', 1.0) * loss_s

        # LR pairs (multi-label classification)
        if 'lr_pairs' in predictions and targets.get('lr_pairs') is not None:
            lr_pred = predictions['lr_pairs']
            lr_true = targets['lr_pairs']
            if edge_mask is not None:
                lr_pred = lr_pred[edge_mask]
                lr_true = lr_true[edge_mask]
            loss_lr = self.lr_loss_fn(lr_pred, lr_true)
            loss_dict['loss_lr_pairs'] = loss_lr.item()
            loss_ccc = loss_ccc + self.task_weights.get('lr_pairs', 0.5) * loss_lr

        # Metabolites (multi-label classification)
        if 'metabolites' in predictions and targets.get('metabolites') is not None:
            m_pred = predictions['metabolites']
            m_true = targets['metabolites']
            if edge_mask is not None:
                m_pred = m_pred[edge_mask]
                m_true = m_true[edge_mask]
            loss_m = self.metab_loss_fn(m_pred, m_true)
            loss_dict['loss_metabolites'] = loss_m.item()
            loss_ccc = loss_ccc + self.task_weights.get('metabolites', 0.3) * loss_m

        loss_dict['loss_ccc'] = loss_ccc.item()

        # -- TOTAL: Expression (primary) + CCC (auxiliary) --------------
        total_loss = loss_expr + self.ccc_weight * loss_ccc
        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict
