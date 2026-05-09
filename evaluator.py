# evaluator.py

from scipy.stats import pearsonr
import torch
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Optional
import json
from logger import *


class Evaluator:
    """
    Evaluate binding-affinity regressors and summarize prediction geometry.

    The evaluator is shared by training-time validation and final held-out
    testing. Besides standard scalar metrics such as RMSE, Pearson correlation,
    and Concordance Index (CI), it also provides scatter-oriented diagnostics
    used to understand calibration, bias, and diagonal agreement.

    Attributes:
        model: The model to evaluate.
        device: Device for evaluation computations.

    Example:
        >>> evaluator = Evaluator(model, device='cuda')
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
            >>> ci = Evaluator.concordance_index(true_vals, pred_vals)
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

    @staticmethod
    def _filter_finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        return y_true[mask], y_pred[mask]

    @staticmethod
    def lin_concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = Evaluator._filter_finite_pairs(y_true, y_pred)
        if y_true.size < 2:
            return float('nan')

        mean_true = float(np.mean(y_true))
        mean_pred = float(np.mean(y_pred))
        var_true = float(np.var(y_true))
        var_pred = float(np.var(y_pred))
        cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))

        denom = var_true + var_pred + (mean_true - mean_pred) ** 2
        if denom <= 0:
            return float('nan')
        return 2.0 * cov / denom

    @staticmethod
    def orthogonal_rmse_to_diagonal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = Evaluator._filter_finite_pairs(y_true, y_pred)
        if y_true.size == 0:
            return float('nan')
        return float(np.sqrt(np.mean(((y_pred - y_true) / np.sqrt(2.0)) ** 2)))

    @staticmethod
    def scatter_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute extended scatter-geometry diagnostics for one prediction set.

        This intentionally goes beyond one-number performance summaries and is
        used to describe whether predictions:

        - align with the ideal diagonal `y = x`,
        - show systematic offset,
        - compress or stretch the target range,
        - deviate in orientation from the ideal agreement line.
        """
        y_true, y_pred = Evaluator._filter_finite_pairs(y_true, y_pred)
        diagnostics = {
            "num_points": int(y_true.size),
            "bias": float('nan'),
            "mae": float('nan'),
            "rmse": float('nan'),
            "pearson_r": float('nan'),
            "ccc": float('nan'),
            "ols_slope": float('nan'),
            "ols_intercept": float('nan'),
            "orthogonal_rmse_to_diagonal": float('nan'),
            "pca_slope": float('nan'),
            "pca_intercept": float('nan'),
            "delta_angle_deg": float('nan'),
        }
        if y_true.size == 0:
            return diagnostics

        residuals = y_pred - y_true
        diagnostics["bias"] = float(np.mean(residuals))
        diagnostics["mae"] = float(np.mean(np.abs(residuals)))
        diagnostics["rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
        diagnostics["orthogonal_rmse_to_diagonal"] = Evaluator.orthogonal_rmse_to_diagonal(y_true, y_pred)
        diagnostics["ccc"] = Evaluator.lin_concordance_correlation_coefficient(y_true, y_pred)

        if y_true.size >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            r_val, _ = pearsonr(y_true, y_pred)
            diagnostics["pearson_r"] = float(r_val)

            ols_slope, ols_intercept = np.polyfit(y_true, y_pred, deg=1)
            diagnostics["ols_slope"] = float(ols_slope)
            diagnostics["ols_intercept"] = float(ols_intercept)

            center = np.array([np.mean(y_true), np.mean(y_pred)], dtype=float)
            cov = np.cov(np.stack([y_true, y_pred], axis=0))
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal_vec = eigvecs[:, int(np.argmax(eigvals))]
            if principal_vec[0] < 0:
                principal_vec = -principal_vec

            if abs(principal_vec[0]) > 1e-12:
                pca_slope = float(principal_vec[1] / principal_vec[0])
                pca_intercept = float(center[1] - pca_slope * center[0])
                diagnostics["pca_slope"] = pca_slope
                diagnostics["pca_intercept"] = pca_intercept
                diagnostics["delta_angle_deg"] = float(np.degrees(np.arctan(pca_slope) - np.pi / 4.0))

        return diagnostics

    @staticmethod
    def _bootstrap_pca_band(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        x_grid: np.ndarray,
        num_bootstrap: int = 200,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_true, y_pred = Evaluator._filter_finite_pairs(y_true, y_pred)
        if y_true.size < 3:
            return np.full_like(x_grid, np.nan, dtype=float), np.full_like(x_grid, np.nan, dtype=float)

        rng = np.random.default_rng(42)
        lines = []
        n = y_true.size
        for _ in range(num_bootstrap):
            idx = rng.integers(0, n, size=n)
            sample_true = y_true[idx]
            sample_pred = y_pred[idx]
            diag = Evaluator.scatter_diagnostics(sample_true, sample_pred)
            slope = diag["pca_slope"]
            intercept = diag["pca_intercept"]
            if np.isfinite(slope) and np.isfinite(intercept):
                lines.append(slope * x_grid + intercept)

        if not lines:
            return np.full_like(x_grid, np.nan, dtype=float), np.full_like(x_grid, np.nan, dtype=float)

        line_stack = np.stack(lines, axis=0)
        lower = np.quantile(line_stack, alpha / 2.0, axis=0)
        upper = np.quantile(line_stack, 1.0 - alpha / 2.0, axis=0)
        return lower, upper

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
        scatter_diag = self.scatter_diagnostics(y_true_np, y_pred_np)
        diagnostics_path = f'{exp_dir}/best_validation_scatter_diagnostics.json'
        with open(diagnostics_path, 'w') as f:
            json.dump(scatter_diag, f, indent=4)

        plt.subplot(3, 2, 5)
        plt.scatter(y_true_np, y_pred_np, alpha=0.5, color='teal')
        # Ideal prediction line (diagonal)
        lims = [min(min(y_true_np), min(y_pred_np)), max(max(y_true_np), max(y_pred_np))]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        x_grid = np.linspace(lims[0], lims[1], 200)
        pca_slope = scatter_diag["pca_slope"]
        pca_intercept = scatter_diag["pca_intercept"]
        if np.isfinite(pca_slope) and np.isfinite(pca_intercept):
            band_low, band_high = self._bootstrap_pca_band(y_true_np, y_pred_np, x_grid)
            if np.isfinite(band_low).any() and np.isfinite(band_high).any():
                plt.fill_between(
                    x_grid, band_low, band_high,
                    color='royalblue', alpha=0.18,
                    label='PCA axis 90% band'
                )
            plt.plot(
                x_grid, pca_slope * x_grid + pca_intercept,
                color='royalblue', linewidth=2.0,
                label='PCA major axis'
            )
        plt.title(f'Actual vs Predicted (Epoch {len(history["val_pearson"])})')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.grid(True)
        plt.legend(loc='lower right')
        summary_text = (
            f"slope={scatter_diag['pca_slope']:.3f}\n"
            f"bias={scatter_diag['bias']:.3f}\n"
            f"CCC={scatter_diag['ccc']:.3f}\n"
            f"orthoRMSE={scatter_diag['orthogonal_rmse_to_diagonal']:.3f}\n"
            f"d_angle={scatter_diag['delta_angle_deg']:.2f} deg"
        )
        plt.text(
            0.03, 0.97, summary_text,
            transform=plt.gca().transAxes,
            va='top', ha='left',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='lightgray')
        )

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
            log_info(f"Scatter diagnostics saved to {diagnostics_path}", stage="EVALUATOR")
        if show:
            plt.show()
