"""
Training callbacks for model training.

Implements:
- EarlyStopping: Stop training if validation metric doesn't improve
- ModelCheckpoint: Save best model based on validation metric
- TensorBoardLogger: Log metrics and visualizations to TensorBoard
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    """
    Stop training if validation performance doesn't improve for a given patience.

    Monitors a validation metric and stops training if it doesn't improve
    (increase for 'max' mode, decrease for 'min' mode) for `patience` epochs.

    Args:
        patience (int): Number of epochs to wait before stopping (default: 10)
        min_delta (float): Minimum change to qualify as improvement (default: 0.001)
        mode (str): 'max' or 'min' - whether higher or lower metric is better
        verbose (bool): Print messages when improvement occurs

    Example:
        early_stopping = EarlyStopping(patience=10, mode='max')
        for epoch in range(num_epochs):
            val_f1 = validate(...)
            if early_stopping(val_f1):
                print("Early stopping triggered!")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # Internal state
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Comparison function
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        elif mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

    def __call__(self, val_metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_metric (float): Current validation metric value

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_metric
            return False

        if self.is_better(val_metric, self.best_score):
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Save the best model based on a validation metric.

    Monitors a validation metric and saves the model checkpoint whenever
    the metric improves.

    Args:
        checkpoint_dir (str): Directory to save checkpoints
        monitor (str): Metric name to monitor (e.g., 'val_f1', 'val_loss')
        mode (str): 'max' or 'min' - whether higher or lower is better
        save_best_only (bool): Only save when metric improves
        verbose (bool): Print messages when saving
        filename_prefix (str): Prefix for checkpoint filenames

    Example:
        checkpoint = ModelCheckpoint(
            checkpoint_dir='checkpoints/',
            monitor='val_f1',
            mode='max'
        )

        for epoch in range(num_epochs):
            metrics = train_and_validate(...)
            checkpoint(epoch, metrics, model)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        monitor: str = 'val_f1',
        mode: str = 'max',
        save_best_only: bool = True,
        verbose: bool = True,
        filename_prefix: str = 'model'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.filename_prefix = filename_prefix

        # Internal state
        self.best_score = None

        # Comparison function
        if mode == 'max':
            self.is_better = lambda current, best: current > best
        elif mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Save checkpoint if metric improved.

        Args:
            epoch (int): Current epoch number
            metrics (dict): Dictionary of metrics {'metric_name': value}
            model (nn.Module): Model to save
            optimizer (Optimizer, optional): Optimizer state to save
        """
        if self.monitor not in metrics:
            if self.verbose:
                print(f"[Checkpoint] Warning: Metric '{self.monitor}' not found in metrics")
            return

        current_score = metrics[self.monitor]

        # Decide whether to save
        should_save = False
        if self.best_score is None:
            should_save = True
            self.best_score = current_score
        elif self.is_better(current_score, self.best_score):
            should_save = True
            self.best_score = current_score
        elif not self.save_best_only:
            should_save = True

        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'best_score': self.best_score
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            if self.save_best_only:
                filename = f"{self.filename_prefix}_best.pt"
            else:
                filename = f"{self.filename_prefix}_epoch{epoch:03d}.pt"

            filepath = self.checkpoint_dir / filename
            torch.save(checkpoint, filepath)

    def load_best_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load the best saved model.

        Args:
            model (nn.Module): Model to load weights into
            optimizer (Optimizer, optional): Optimizer to load state into

        Returns:
            dict: Checkpoint dictionary with metrics, epoch, etc.
        """
        filepath = self.checkpoint_dir / f"{self.filename_prefix}_best.pt"

        if not filepath.exists():
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.verbose:
            print(f"[Checkpoint] Loaded best model from epoch {checkpoint['epoch']}")

        return checkpoint


class TensorBoardLogger:
    """
    Log metrics and visualizations to TensorBoard.

    Provides methods to log scalars, histograms, and custom visualizations
    like attention weight heatmaps.

    Args:
        log_dir (str): Directory for TensorBoard logs
        comment (str, optional): Comment to append to log directory name

    Example:
        logger = TensorBoardLogger(log_dir='runs/experiment1')

        for epoch in range(num_epochs):
            # Training
            logger.log_scalar('train/loss', loss, epoch)
            logger.log_scalar('train/f1', f1, epoch)

            # Validation
            logger.log_scalar('val/loss', val_loss, epoch)

            # Attention weights
            logger.log_attention_weights(attention_weights, epoch)
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = ''
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)
        print(f"[TensorBoard] Logging to: {self.log_dir}")

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """
        Log a scalar value.

        Args:
            tag (str): Name of the scalar (e.g., 'train/loss', 'val/f1')
            value (float): Scalar value
            step (int): Global step (usually epoch or iteration)
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int
    ):
        """
        Log multiple scalars at once (useful for comparing metrics).

        Args:
            main_tag (str): Parent tag (e.g., 'losses', 'metrics')
            tag_scalar_dict (dict): {metric_name: value}
            step (int): Global step
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: int
    ):
        """
        Log a histogram of values.

        Args:
            tag (str): Name of the histogram (e.g., 'weights/layer1')
            values (Tensor or ndarray): Values to histogram
            step (int): Global step
        """
        self.writer.add_histogram(tag, values, step)

    def log_attention_weights(
        self,
        attention_weights: torch.Tensor,
        step: int,
        pathway_names: Optional[list] = None
    ):
        """
        Visualize pathway attention weights as a heatmap.

        Creates a heatmap showing which pathways are attended to for each node.

        Args:
            attention_weights (Tensor): Attention weights [N, num_pathways]
            step (int): Global step
            pathway_names (list, optional): Names of pathways for axis labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if pathway_names is None:
            pathway_names = ['Juxtacrine', 'Paracrine', 'Metabolite', 'Combined']

        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        # Average across nodes for summary visualization
        avg_attention = attention_weights.mean(axis=0)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Average attention per pathway (bar plot)
        axes[0].bar(pathway_names, avg_attention)
        axes[0].set_ylabel('Average Attention Weight')
        axes[0].set_title('Pathway Importance (Averaged)')
        axes[0].set_ylim([0, 1])

        # Plot 2: Attention distribution (heatmap of first 100 nodes)
        sns.heatmap(
            attention_weights[:100, :].T,
            ax=axes[1],
            cmap='viridis',
            yticklabels=pathway_names,
            xticklabels=False,
            cbar_kws={'label': 'Attention Weight'}
        )
        axes[1].set_xlabel('Nodes (sample of 100)')
        axes[1].set_ylabel('Pathway')
        axes[1].set_title('Attention Weights per Node')

        plt.tight_layout()

        # Log to TensorBoard
        self.writer.add_figure('attention/pathway_weights', fig, step)
        plt.close(fig)

    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_data: Any
    ):
        """
        Log the model computational graph.

        Args:
            model (nn.Module): Model to visualize
            input_data: Sample input to the model
        """
        self.writer.add_graph(model, input_data)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print(f"[TensorBoard] Logger closed")
