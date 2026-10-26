import torch
from age_estimation.registry import OPTIMIZERS

@OPTIMIZERS.register('Adam')
def load_adam_optimizer(**kwargs):
     return torch.optim.Adam(**kwargs)

@OPTIMIZERS.register('AdamW')
def load_adamw_optimizer(**kwargs):
     return torch.optim.AdamW(**kwargs)
