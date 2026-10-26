import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from lung_severity_assesmnt.registry import LOSSES
def RegressionCenterLoss(*args, **kwargs):
    raise NotImplementedError("Center loss is not included; use the supplied MAE/MSE medical configs.")


@LOSSES.register('multi_loss')
class MultiLoss(nn.Module):
     def __init__(self, losses: List[Dict[str, Any]], feat_dim: int = 224):
          super(MultiLoss, self).__init__()
          self.loss_modules = nn.ModuleList()
          self.loss_info = []
          self.feat_dim = feat_dim

          loss_map = {
               'mse_loss': nn.MSELoss,
               'mae_loss': nn.L1Loss,
               'smooth_l1_loss': nn.SmoothL1Loss,
               'center_loss': lambda: RegressionCenterLoss(feat_dim=feat_dim),
          }

          for loss_config in losses:
               name = loss_config['name']
               weight = loss_config['weight']
               
               if name not in loss_map:
                    raise ValueError(
                         f"Unknown loss name: '{name}'. "
                         f"Supported losses are: {list(loss_map.keys())}"
                    )

               self.loss_modules.append(loss_map[name]())
               self.loss_info.append((name, weight))

     def forward(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
          
          if predictions.dim() == 1:
               predictions = predictions.unsqueeze(1)

          if targets.dim() == 1:
               targets = targets.unsqueeze(1)
          
          total_loss = torch.tensor(0.0, device=predictions.device)
          loss_dict = {}

          for (name, weight), loss_fn in zip(self.loss_info, self.loss_modules):
               if name == 'center_loss':
                    if features is None:
                         raise ValueError("Features are required for center_loss but were not provided")
                    
                    # Convert 4D feature maps to 2D features for center loss
                    if features.dim() == 4:  # (B, C, H, W)
                         # Global average pooling to get (B, C) features
                         features_2d = torch.mean(features, dim=(2, 3))  # (B, C)
                    else:
                         features_2d = features
                    
                    # Flatten targets for center loss
                    targets_flat = targets.squeeze(1) if targets.dim() == 2 else targets
                    unweighted_loss = loss_fn(features_2d, targets_flat) 
               else:
                    unweighted_loss = loss_fn(predictions, targets)
                    
               loss_dict[name] = unweighted_loss
               total_loss += weight * unweighted_loss
          
          loss_dict['total_loss'] = total_loss
          return loss_dict

@LOSSES.register("ralo_multitask_loss")
class RaloMultiTaskLoss(nn.Module):
    def __init__(self, lambda_lo: float = 0.3, base: str = "smooth_l1"):
        super().__init__()
        self.lambda_lo = float(lambda_lo)
        if base == "l1":
            self.base = nn.L1Loss()
        elif base == "mse":
            self.base = nn.MSELoss()
        else:
            self.base = nn.SmoothL1Loss()

    def forward(self, predictions, targets, features=None):
        # predictions/targets: [B,2]
        if predictions.ndim != 2 or predictions.size(1) != 2:
            raise ValueError(f"Expected predictions [B,2], got {predictions.shape}")
        if targets.ndim != 2 or targets.size(1) != 2:
            raise ValueError(f"Expected targets [B,2], got {targets.shape}")

        ge_p, lo_p = predictions[:, 0], predictions[:, 1]
        ge_t, lo_t = targets[:, 0], targets[:, 1]

        loss_ge = self.base(ge_p, ge_t)
        loss_lo = self.base(lo_p, lo_t)
        total = loss_ge + self.lambda_lo * loss_lo

        return {"loss_ge": loss_ge, "loss_lo": loss_lo, "total_loss": total}
