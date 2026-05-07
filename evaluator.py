# evaluator.py

from scipy.stats import pearsonr
import torch
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Optional
from logger import *


class Evaluator:
    """
    Evaluator class for protein-ligand binding affinity prediction models.

    Provides comprehensive evaluation metrics including RMSE, Pearson correlation,
    Concordance Index (CI), and visualization capabilities. Tracks model
    performance across training epochs.

    Attributes:
        model: The model to evaluate.
        device: Device for evaluation computations.

    Example:
        >>> evaluator = EValuator(model, device='cuda')
        >>> rmse, pearson, ci, preds, targets = evaluator.evaluate(val_loader)
        >>> print(f"RMSE: {rmse:.3f}, Pearson: {pearson:.3f}, CI: {ci:.3f}")
    """

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        """
        Initialize the evaluator.

        Args:
            model: Model instance to evaluate.
            device: Device for computations ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device

    @staticmethod
    def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Vectorized Concordance Index calculation using NumPy.

        Measures the fraction of pairs where the predicted ordering agrees
        with the true ordering. Higher values indicate better ranking performance.

        Args:
            y_true: True binding affinity values.
            y_pred: Predicted binding affinity values.

        Returns:
            Concordance Index between 0 and 1.

        Example:
            >>> true_vals = np.array([1.0, 2.0, 3.0])
            >>> pred_vals = np.array([1.1, 2.1, 2.9])
            >>> ci = EValuator.concordance_index(true_vals, pred_vals)
            >>> print(f"CI: {ci:.3f}")  # CI: 1.000
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Create difference matrices: diff[i, j] = y[i] - y[j]
        # This creates NxN matrices
        true_diff = y_true[:, np.newaxis] - y_true[np.newaxis, :]
        pred_diff = y_pred[:, np.newaxis] - y_pred[np.newaxis, :]
        
        # We need only pairs where y_true[i] > y_true[j]
        mask = true_diff > 0
        valid_pairs = mask.sum()
        
        if valid_pairs == 0:
            return 0.5
        
        # Count ordering agreements
        # 1.0 if prediction also >
        # 0.5 if prediction equals
        concordant = (pred_diff[mask] > 0).sum()
        ties = (pred_diff[mask] == 0).sum()
        
        return (concordant + 0.5 * ties) / valid_pairs

    def _run_loader(
        self,
        loader,
        progress: float = 1.0,
        criterion: Optional[torch.nn.Module] = None,
    ) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
        """
        Shared inference pass over a loader.

        Optionally accumulates average loss while collecting predictions and targets.
        """
        self.model.eval()
        preds, targets = [], []
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in loader:
                batch = [
                    inp.to(self.device) if hasattr(inp, 'to')
                    else {k: v.to(self.device) for k, v in inp.items()}
                    for inp in batch
                ]
                y_hat = self.model(tuple(batch), progress=progress).view(-1)
                target = batch[-1].view(-1)

                preds.extend(y_hat.cpu().tolist())
                targets.extend(target.cpu().tolist())

                if criterion is not None:
                    batch_loss = criterion(y_hat, target)
                    total_loss += float(batch_loss.item())
                    batch_count += 1

        avg_loss = (total_loss / batch_count) if criterion is not None and batch_count > 0 else None
        return avg_loss, np.array(preds), np.array(targets)

    def evaluate(self, loader, progress: float = 1.0) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model performance on a dataset.

        Computes RMSE, Pearson correlation, and Concordance Index.

        Args:
            loader: DataLoader containing evaluation data.

        Returns:
            Tuple of (rmse, pearson_r, ci, predictions, targets).

        Example:
            >>> rmse, pearson, ci, preds, targets = evaluator.evaluate(test_loader)
            >>> logger.info(f"[EVALUATION] RMSE: {rmse:.3f}, Pearson: {pearson:.3f}, CI: {ci:.3f}")
        """
        _, preds, targets = self._run_loader(loader, progress=progress, criterion=None)

        finite_mask = np.isfinite(preds) & np.isfinite(targets)
        preds = preds[finite_mask]
        targets = targets[finite_mask]

        if preds.size == 0 or targets.size == 0:
            return float('nan'), float('nan'), 0.0, preds, targets
        
        rmse = np.sqrt(np.mean((preds - targets)**2))
        if preds.size < 2 or np.std(preds) == 0 or np.std(targets) == 0:
            r_val = float('nan')
        else:
            r_val, _ = pearsonr(preds, targets)
        ci_val = self.concordance_index(targets, preds)  # Calculate CI

        return rmse, r_val, ci_val, preds, targets

    def evaluate_with_loss(
        self,
        loader,
        criterion: torch.nn.Module,
        progress: float = 1.0,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model performance and validation loss in a single loader pass.

        Returns:
            Tuple of (avg_loss, rmse, pearson_r, ci, predictions, targets).
        """
        avg_loss, preds, targets = self._run_loader(loader, progress=progress, criterion=criterion)

        finite_mask = np.isfinite(preds) & np.isfinite(targets)
        preds = preds[finite_mask]
        targets = targets[finite_mask]

        if preds.size == 0 or targets.size == 0:
            return float('nan'), float('nan'), float('nan'), 0.0, preds, targets

        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        if preds.size < 2 or np.std(preds) == 0 or np.std(targets) == 0:
            r_val = float('nan')
        else:
            r_val, _ = pearsonr(preds, targets)
        ci_val = self.concordance_index(targets, preds)

        if avg_loss is None:
            avg_loss = float('nan')

        return avg_loss, rmse, r_val, ci_val, preds, targets

    def plot_history(self, exp_dir: str, history: dict, show: bool = True, save: bool = True) -> None:
        """
        Plot training history and save to file.

        Creates comprehensive plots showing training loss, validation metrics,
        and prediction vs true value scatter plots.

        Args:
            exp_dir: Directory to save plots.
            show: Whether to display plots.
            save: Whether to save plots to files.

        Example:
            >>> evaluator.plot_history('experiments/exp_001', show=False, save=True)
        """
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(15, 15))

        # 1. Losses
        plt.subplot(3, 2, 1)
        if not len(history['train_loss']) == len(history['val_loss']) == len(epochs):
            log_error("Loss history lengths must match number of epochs")

        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, history['val_loss'], 'k--', label='Val Loss')
        plt.title('Learning Curve (Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # 2. RMSE
        plt.subplot(3, 2, 2)
        if not len(history['train_rmse']) == len(history['val_rmse']) == len(epochs):
            log_error("RMSE history lengths must match number of epochs")
        plt.plot(epochs, history['train_rmse'], 'b-', label='Train RMSE')
        plt.plot(epochs, history['val_rmse'], 'k--', label='Val RMSE')
        plt.title('Validation RMSE')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend()

        # 3. Pearson R
        plt.subplot(3, 2, 3)
        if not len(history['train_pearson']) == len(history['val_pearson']) == len(epochs):
            log_error("Pearson R history lengths must match number of epochs")
        plt.plot(epochs, history['train_pearson'], 'b-', label='Train Pearson')
        plt.plot(epochs, history['val_pearson'], 'k--', label='Val Pearson')
        plt.title('Correlation (Pearson R)')
        plt.xlabel('Epochs')
        plt.ylabel('R')
        plt.grid(True)
        plt.legend()

        # 4. CI
        plt.subplot(3, 2, 4)
        if not len(history['train_ci']) == len(history['val_ci']) == len(epochs):
            log_error("CI history lengths must match number of epochs")
        plt.plot(epochs, history['train_ci'], 'b-', label='Train CI')
        plt.plot(epochs, history['val_ci'], 'k--', label='Val CI')
        plt.title('Ranking Accuracy (CI)')
        plt.xlabel('Epochs')
        plt.ylabel('CI')
        plt.grid(True)
        plt.legend()

        if history.get('best_y_true') is None or history.get('best_y_pred') is None:
            plt.tight_layout()
            if save:
                plt.savefig(f'{exp_dir}/model_performance_report.png')
                log_info(f"Performance report saved to {exp_dir}/model_performance_report.png", stage="EVALUATOR")
            if show:
                plt.show()
            return

        y_true_np = np.array(history['best_y_true'])
        y_pred_np = np.array(history['best_y_pred'])

        plt.subplot(3, 2, 5)
        plt.scatter(y_true_np, y_pred_np, alpha=0.5, color='teal')
        # Ideal prediction line (diagonal)
        lims = [min(min(y_true_np), min(y_pred_np)), max(max(y_true_np), max(y_pred_np))]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        plt.title(f'Actual vs Predicted (Epoch {len(history["val_pearson"])})')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.grid(True)

        # 6. Error distribution (Histogram)
        plt.subplot(3, 2, 6)
        errors = y_pred_np - y_true_np
        plt.hist(errors, bins=20, color='salmon', edgecolor='black')
        plt.axvline(0, color='black', linestyle='--')
        plt.title('Error Distribution (Residuals)')
        plt.xlabel('Prediction Error')
        plt.grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(f'{exp_dir}/model_performance_report.png')
            log_info(f"Performance report saved to {exp_dir}/model_performance_report.png", stage="EVALUATOR")
        if show:
            plt.show()
