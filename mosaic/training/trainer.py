"""
mosaic/training/trainer.py

Trainer for MOSAIC model.

Adapted from src4/training/hetgt_trainer.py.
Key changes vs src4:
  - No auxiliary edge-level tasks (MOSAIC drops EdgePredictorMultiTask)
  - Expression loss only (Huber by default, as per breast_config.yaml)
  - Model is MOSAIC with 3 node types + 7 edge types
  - No edge_train_mask / edge_val_mask needed for training
  - Checkpoint dir: mosaic/checkpoints/<dataset>/

Training objective: predict y_expr [N, n_genes] from graph structure.
Model learns to use (gene, interacts, gene) attention as CCC signal.

Optional auxiliary:
  lambda_spatial > 0 -> add spatial attention regularization:
    For each secreted cell-cell edge, penalize positive correlation between
    attention and normalized distance. This encourages high-attention cell
    pairs to be spatially proximate, directly optimizing DES.
    Loss: lambda_spatial * Pearson(attn, dist_norm)   [want negative]
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from mosaic.training.losses import MOSAICLoss
from mosaic.training.callbacks import EarlyStopping, ModelCheckpoint

log = logging.getLogger(__name__)


class MOSAICTrainer:
    """
    Full-graph trainer for MOSAIC (expression prediction primary task).

    Args:
        model:   MOSAIC model
        data:    PyG HeteroData (MOSAIC format, full graph on device)
        config:  config dict (uses 'training' section)
        device:  torch.device
        dataset: dataset name (for checkpoint path)
    """

    def __init__(
        self,
        model: nn.Module,
        data: Any,
        config: Dict[str, Any],
        device: torch.device,
        dataset: str = "breast_new",
    ):
        self.model   = model.to(device)
        self.data    = data.to(device)
        self.config  = config
        self.device  = device
        self.dataset = dataset

        train_cfg = config.get("training", config)

        # -- Node masks (expression task) ---------------------------------
        self.train_mask = self.data["cell"].train_mask.to(device)
        self.val_mask   = self.data["cell"].val_mask.to(device)
        self.test_mask  = self.data["cell"].test_mask.to(device)

        # -- Spatial attention regularization -----------------------------
        # lambda_spatial: encourages cell-cell attention to be spatially enriched
        # (high attention <-> short distance). Directly optimizes DES metric.
        self.lambda_spatial = float(train_cfg.get("lambda_spatial", 0.0))
        if self.lambda_spatial > 0:
            # Pre-cache secreted edge distance (edge_attr[:, 0] = dist_norm, col 1 = gaussian_weight)
            # IMPORTANT: use col 0 (dist_norm), NOT col 1 (gaussian_weight).
            # gaussian_weight = exp(-dist^2/2sigma^2) DECREASES with distance -- using it would invert the loss!
            sec_et = ("cell", "secreted", "cell")
            if sec_et in data.edge_types:
                self._secreted_dist = self.data[sec_et].edge_attr[:, 0].to(device)  # [E_s] = dist_norm
                log.info("Spatial regularization: lambda=%.4f, secreted edges=%d",
                         self.lambda_spatial, len(self._secreted_dist))
            else:
                log.warning("Secreted edge type not found -- disabling spatial regularization")
                self.lambda_spatial = 0.0
                self._secreted_dist = None
        else:
            self._secreted_dist = None

        # -- Loss --------------------------------------------------------
        self.criterion = MOSAICLoss(
            ccc_weight=0.0,          # no auxiliary tasks in MOSAIC
            expr_loss_type=train_cfg.get("expr_loss", "huber"),
        )

        # -- Optimizer ---------------------------------------------------
        lr = float(train_cfg.get("lr", 1e-3))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # -- Scheduler ---------------------------------------------------
        sched = train_cfg.get("scheduler", "plateau")
        if sched == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=float(train_cfg.get("scheduler_factor", 0.5)),
                patience=int(train_cfg.get("scheduler_patience", 15)),
                verbose=True,
            )
        else:
            self.scheduler = None

        # -- Callbacks ---------------------------------------------------
        patience = int(train_cfg.get("patience", 50))
        self.early_stopping = EarlyStopping(patience=patience, mode="max", verbose=True)

        ckpt_dir = Path(train_cfg.get("checkpoint_dir", "mosaic/checkpoints")) / dataset
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=str(ckpt_dir),
            monitor="val_r2",
            mode="max",
            verbose=True,
        )

        # -- Gradient clipping ------------------------------------------
        self.grad_clip = float(train_cfg.get("grad_clip", 5.0))

        # -- Mixed precision --------------------------------------------
        use_amp = train_cfg.get("use_amp", device.type == "cuda")
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # -- Training state ---------------------------------------------
        self.current_epoch = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_r2": [],
            "val_pearson": [],
        }

    # -----------------------------------------------------------------
    # Main train loop
    # -----------------------------------------------------------------

    def train(self, num_epochs: int) -> Dict[str, list]:
        n_params    = sum(p.numel() for p in self.model.parameters())
        n_train     = int(self.train_mask.sum())
        n_val       = int(self.val_mask.sum())
        n_genes     = int(self.data["cell"].y_expr.shape[1])
        edge_types  = [et[1] for et in self.data.edge_types]

        # One-line user-facing header (always printed)
        print(f"[mosaic] training {self.dataset}: {n_params/1e6:.1f}M params, "
              f"{n_train} train / {n_val} val cells, {n_genes} target genes "
              f"on {self.device}")
        print(f"[mosaic] epoch     train_loss   val_loss     val_R2    val_r    time")

        log_every = 20      # print every 20 epochs (user-requested cadence)
        best_val_r2 = -float("inf")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self._train_epoch()
            val_metrics   = self._validate(self.val_mask, split="val")

            self.history["train_loss"].append(train_metrics["loss_expr"])
            self.history["val_loss"].append(val_metrics["loss_expr"])
            self.history["val_r2"].append(val_metrics["r2_mean"])
            self.history["val_pearson"].append(val_metrics["pearson_median"])
            best_val_r2 = max(best_val_r2, val_metrics["r2_mean"])

            if self.scheduler is not None:
                self.scheduler.step(val_metrics["r2_mean"])

            self.checkpoint(
                epoch,
                {"val_r2": val_metrics["r2_mean"]},
                self.model,
                self.optimizer,
            )

            # User-facing single line every `log_every` epochs (and the first + last)
            is_last = (epoch + 1) == num_epochs
            if epoch % log_every == 0 or is_last:
                print(f"[mosaic] {epoch+1:5d}/{num_epochs:<4d} "
                      f"{train_metrics['loss_expr']:10.4f}  "
                      f"{val_metrics['loss_expr']:10.4f}  "
                      f"{val_metrics['r2_mean']:7.4f}  "
                      f"{val_metrics['pearson_median']:7.4f}  "
                      f"{time.time()-t0:5.1f}s",
                      flush=True)

            if self.early_stopping(val_metrics["r2_mean"]):
                print(f"[mosaic] early stopping at epoch {epoch+1} "
                      f"(best val_R2={best_val_r2:.4f})", flush=True)
                break

        # Final test evaluation — reload best checkpoint
        best_ckpt_path = self.checkpoint.checkpoint_dir / f"{self.checkpoint.filename_prefix}_best.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            log.info("Restored best model (epoch %d)", ckpt.get("epoch", -1))
        test_metrics = self._validate(self.test_mask, split="test")
        self._print_report(test_metrics, "TEST")

        return self.history

    # -----------------------------------------------------------------
    # Single epoch
    # -----------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        use_spatial = self.lambda_spatial > 0 and self._secreted_dist is not None
        with autocast(enabled=self.use_amp):
            predictions = self.model(self.data, return_attention=use_spatial)

            total_loss, loss_dict = self.criterion(
                predictions,
                {"expression": self.data["cell"].y_expr},
                node_mask=self.train_mask,
                edge_mask=None,
            )

            # -- Spatial attention regularization -------------------------
            if use_spatial:
                sec_et = ("cell", "secreted", "cell")
                attn_info = predictions.get("attention_info", {})
                per_layer  = attn_info.get("per_layer", [])
                if per_layer:
                    last_layer_attn = per_layer[-1].get(sec_et)   # [E_s, n_heads] or None
                    if last_layer_attn is not None:
                        # Mean over attention heads: [E_s]
                        attn_mean = last_layer_attn.float().mean(dim=-1)
                        dist_norm = self._secreted_dist

                        # Normalize to z-scores for scale-invariant Pearson correlation
                        z_attn = (attn_mean - attn_mean.mean()) / (attn_mean.std() + 1e-8)
                        z_dist = (dist_norm - dist_norm.mean()) / (dist_norm.std() + 1e-8)

                        # Pearson(attn, dist): positive = attn up when dist up (BAD)
                        # We MINIMIZE this -> encourages attn up for SHORT distances
                        spatial_loss = (z_attn * z_dist).mean()
                        total_loss = total_loss + self.lambda_spatial * spatial_loss
                        loss_dict["loss_spatial"] = float(spatial_loss.detach())

        self.scaler.scale(total_loss).backward()

        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _validate(
        self,
        node_mask: torch.Tensor,
        split: str = "val",
    ) -> Dict[str, Any]:
        self.model.eval()

        predictions = self.model(self.data)

        _, loss_dict = self.criterion(
            predictions,
            {"expression": self.data["cell"].y_expr},
            node_mask=node_mask,
            edge_mask=None,
        )

        # Expression metrics
        expr_pred = predictions["expression"][node_mask].cpu().numpy()
        expr_true = self.data["cell"].y_expr[node_mask].cpu().numpy()

        # Clamp to non-negative at eval (targets are log1p >= 0)
        expr_pred = np.clip(expr_pred, 0, None)

        n_genes = expr_pred.shape[1]
        r2_per_gene      = []
        pearson_per_gene = []
        nrmse_per_gene   = []

        for g in range(n_genes):
            yt = expr_true[:, g]
            yp = expr_pred[:, g]
            if yt.std() < 1e-8 or yp.std() < 1e-8:
                continue

            r2 = r2_score(yt, yp)
            r2_per_gene.append(r2)

            r, _ = pearsonr(yt, yp)
            pearson_per_gene.append(float(r) if not np.isnan(r) else 0.0)

            rng = yt.max() - yt.min()
            if rng > 1e-8:
                nrmse_per_gene.append(np.sqrt(np.mean((yt - yp) ** 2)) / rng)

        r2_mean         = float(np.mean(r2_per_gene))        if r2_per_gene      else 0.0
        pearson_median  = float(np.median(pearson_per_gene)) if pearson_per_gene else 0.0
        nrmse_median    = float(np.median(nrmse_per_gene))   if nrmse_per_gene   else 1.0
        r2_positive     = int(sum(r > 0 for r in r2_per_gene))

        return {
            **loss_dict,
            "r2_mean":       r2_mean,
            "pearson_median": pearson_median,
            "nrmse_median":  nrmse_median,
            "r2_positive":   r2_positive,
            "n_genes_eval":  len(r2_per_gene),
            "r2_per_gene":   r2_per_gene,
        }

    def _print_report(self, metrics: Dict, label: str = "TEST"):
        log.info("--- %s Expression Prediction ---", label)
        log.info("  R2 mean:       %.4f", metrics["r2_mean"])
        log.info("  Pearson r:     %.4f  (median)", metrics["pearson_median"])
        log.info("  NRMSE:         %.4f  (median)", metrics["nrmse_median"])
        log.info("  Loss (expr):   %.4f", metrics["loss_expr"])
        log.info("  R2>0:          %d / %d genes",
                 metrics["r2_positive"], metrics["n_genes_eval"])

        if metrics.get("r2_per_gene"):
            arr  = np.array(metrics["r2_per_gene"])
            top5 = arr.argsort()[-5:][::-1]
            log.info("  Top 5 R2:  %s", [f"{arr[i]:.3f}" for i in top5])
