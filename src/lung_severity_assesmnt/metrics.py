import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
from lung_severity_assesmnt.registry import METRICS

@METRICS.register('lung_severity_assesmnt_metrics')
class AgeEstimationMetrics(nn.Module):
    """
    A module to compute multiple age estimation metrics.

    Supported metrics:
    - 'mae': Mean Absolute Error.
    - 'pc': Pearson Correlation Coefficient.
    - 'cs_L': Cumulative Score for a specific integer threshold L (e.g., 'cs_5', 'cs_10').
    - {'name': 'cs', 'step': T}: Generates CS metrics for thresholds from 0 to 15
      with a step of T.
    """
    def __init__(self, metrics: List[Union[str, Dict]], dim_names=None):
        """
        Args:
            metrics (List[Union[str, Dict]]): A list specifying which metrics to compute.
                Example 1: ['mae', 'pc', 'cs_5', 'cs_10']
                Example 2: ['mae', {'name': 'cs', 'step': 5}]
        """
        super().__init__()
        self.metric_info = []
        self.dim_names = dim_names
        self._initialize_metrics(metrics)

    def _initialize_metrics(self, metrics: List[Union[str, Dict]]):
        """Parses metric names and prepares the calculation functions."""
        for config in metrics:
            if isinstance(config, str):
                # Handle simple string-based metrics
                if config == 'mae':
                    self.metric_info.append(('mae', self._calculate_mae))
                elif config == 'pc':
                    self.metric_info.append(('pc', self._calculate_pc))
                elif config.startswith('cs_'):
                    # Handle individual CS metrics like 'cs_5'
                    try:
                        threshold = float(config.split('_')[1])
                        metric_fn = lambda p, t, th=threshold: self._calculate_cs(p, t, threshold=th)
                        self.metric_info.append((config, metric_fn))
                    except (ValueError, IndexError):
                        raise ValueError(f"Invalid CS format: '{config}'. Expected 'cs_L' where L is an integer.")
                else:
                    raise ValueError(f"Unknown or invalid metric string: '{config}'.")

            elif isinstance(config, dict):
                # Handle dictionary-based metric configurations
                name = config.get('name')
                if name == 'cs':
                    step = config.get('step')
                    if not isinstance(step, int) or step <= 0:
                        raise ValueError("CS metric requires a positive integer 'step' in its config.")
                    
                    max_threshold = 15
                    for threshold in range(0, max_threshold + 1, step):
                        cs_name = f'cs_{threshold}'
                        metric_fn = lambda p, t, th=threshold: self._calculate_cs(p, t, threshold=th)
                        self.metric_info.append((cs_name, metric_fn))
                else:
                    raise ValueError(f"Unknown metric config name: '{name}'.")
            else:
                raise TypeError(f"Metric configuration must be str or dict, not {type(config)}.")

    @staticmethod
    def _calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Mean Absolute Error."""
        return F.l1_loss(predictions, targets)

    @staticmethod
    def _calculate_pc(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Pearson Correlation Coefficient."""
        vx = predictions - predictions.mean()
        vy = targets - targets.mean()
        denominator = torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)) + 1e-8
        return torch.sum(vx * vy) / denominator

    @staticmethod
    def _calculate_cs(predictions: torch.Tensor, targets: torch.Tensor, threshold: float) -> torch.Tensor:
        """Computes Cumulative Score for a given error threshold."""
        abs_error = torch.abs(predictions - targets)
        correct_predictions = (abs_error <= threshold).float()
        return torch.mean(correct_predictions)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the configured metrics."""
        predictions = predictions.squeeze()
        targets = targets.squeeze().float()

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions and targets must have same shape, got {predictions.shape} and {targets.shape}"
            )

        metrics_dict = {}
        for name, metric_fn in self.metric_info:
            metrics_dict[name] = metric_fn(predictions, targets)
            
        return metrics_dict

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = predictions.float()
        targets = targets.float()

        # squeeze only last singleton dims, but keep (N,2)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        if targets.ndim > 1 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions and targets must have same shape, got {predictions.shape} and {targets.shape}"
            )

        metrics_dict = {}

        # ----- scalar case (original behavior) -----
        if predictions.ndim == 1:
            for name, metric_fn in self.metric_info:
                metrics_dict[name] = metric_fn(predictions, targets)
            return metrics_dict

        # ----- multi-output case: (N, D) -----
        if predictions.ndim == 2:
            D = predictions.shape[1]

            # overall metrics (keep same key names)
            # mae/cs -> average over all entries; pc -> average of per-dim PC (more meaningful)
            for name, metric_fn in self.metric_info:
                if name == "pc":
                    pcs = []
                    for d in range(D):
                        pcs.append(metric_fn(predictions[:, d], targets[:, d]))
                    metrics_dict[name] = torch.stack(pcs).mean()
                else:
                    metrics_dict[name] = metric_fn(predictions, targets)

            # per-dimension metrics
            dim_names = getattr(self, "dim_names", None)
            for d in range(D):
                suffix = dim_names[d] if dim_names and d < len(dim_names) else f"d{d}"
                for name, metric_fn in self.metric_info:
                    metrics_dict[f"{name}_{suffix}"] = metric_fn(predictions[:, d], targets[:, d])

            return metrics_dict

        # any other shape: flatten (fallback)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        for name, metric_fn in self.metric_info:
            metrics_dict[name] = metric_fn(predictions, targets)
        return metrics_dict

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(predictions, targets)
