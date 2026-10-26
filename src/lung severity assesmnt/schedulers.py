import torch
from age_estimation.registry import SCHEDULERS

@SCHEDULERS.register('CosineAnnealingLR')
def load_cosine_annealing_lr_scheduler(**kwargs):
     scheduler_info = dict(
          level = 'step',
     )
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(**kwargs)
     return scheduler, scheduler_info


@SCHEDULERS.register('WarmupCosineAnnealingLR')
def load_warmup_cosine_annealing_lr_scheduler(**kwargs):
    """
    Combines a linear warmup (LambdaLR) with CosineAnnealingLR using SequentialLR.
    Required kwargs: optimizer, T_max (for cosine). Optional: warmup_steps (int).
    """
    # pop optimizer so we don't pass it twice
    optimizer = kwargs.pop('optimizer', None)
    if optimizer is None:
        raise ValueError("WarmupCosineAnnealingLR requires 'optimizer' in kwargs")

    warmup_steps = int(kwargs.pop('warmup_steps', 0))
    # remaining kwargs are for CosineAnnealingLR (e.g. T_max, eta_min)
    cosine_kwargs = dict(kwargs)

    # build cosine scheduler (uses optimizer)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cosine_kwargs)

    if warmup_steps > 0:
        # linear warmup from 0 -> 1 multiplier over warmup_steps
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: float(step + 1) / float(max(1, warmup_steps))
        )
        # SequentialLR needs PyTorch >= 1.11
        try:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_steps]
            )
        except AttributeError:
            raise RuntimeError("SequentialLR not available in your torch version; upgrade torch to >=1.11")
    else:
        scheduler = cosine_sched

    scheduler_info = dict(level='step')
    return scheduler, scheduler_info

@SCHEDULERS.register('OneCycleLR')
def load_one_cycle_lr_scheduler(**kwargs):
    """
    Wrapper for torch.optim.lr_scheduler.OneCycleLR.
    Required kwargs: optimizer, max_lr. Provide total_steps or (epochs and steps_per_epoch) via kwargs.
    """
    # pop optimizer so we don't pass it twice
    # optimizer = kwargs.pop('optimizer', None)
    # if optimizer is None:
    #     raise ValueError("OneCycleLR requires 'optimizer' in kwargs")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(**kwargs)
    scheduler_info = dict(level='step')
    return scheduler, scheduler_info

@SCHEDULERS.register('ReduceLROnPlateau')
def load_reduce_on_plateau_scheduler(**kwargs):
    """
    Wrapper for ReduceLROnPlateau. This scheduler is metric-driven.
    Required kwargs: optimizer.
    """
    # pop optimizer so we don't pass it twice
    # optimizer = kwargs.pop('optimizer', None)
    # if optimizer is None:
    #     raise ValueError("ReduceLROnPlateau requires 'optimizer' in kwargs")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**kwargs)
    scheduler_info = dict(level='metric')  # caller should pass metric to scheduler.step(metric)
    return scheduler, scheduler_info

@SCHEDULERS.register('StepLR')
def load_step_lr_scheduler(**kwargs):
    optimizer = kwargs.pop('optimizer', None)
    if optimizer is None:
        raise ValueError("StepLR requires 'optimizer' in kwargs")
    scheduler = torch.optim.lr_scheduler.StepLR(**kwargs)
    scheduler_info = dict(level='epoch')
    return scheduler, scheduler_info

@SCHEDULERS.register('ExponentialLR')
def load_exponential_lr_scheduler(**kwargs):
    optimizer = kwargs.pop('optimizer', None)
    if optimizer is None:
        raise ValueError("ExponentialLR requires 'optimizer' in kwargs")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(**kwargs)
    scheduler_info = dict(level='epoch')
    return scheduler, scheduler_info

# def load_cosine_annealing_warm_restarts_scheduler(**kwargs):
#     optimizer = kwargs.pop('optimizer', None)
#     if optimizer is None:
#         raise ValueError("CosineAnnealingWarmRestarts requires 'optimizer' in kwargs")
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(**kwargs)
#     scheduler_info = dict(level='step')
#     return scheduler, scheduler_info

# import torch
@SCHEDULERS.register('CosineAnnealingWarmRestarts')
def load_cosine_annealing_warm_restarts_scheduler(**kwargs):
    """
    CosineAnnealingWarmRestarts wrapper matching your loader convention.

    Required kwargs:
      - optimizer
      - T_0

    Optional kwargs:
      - T_mult (default 1)
      - eta_min (default 0.0)
      - last_epoch (default -1)

    IMPORTANT:
      - scheduler_info is level='step' so T_0 is interpreted in *optimizer steps*,
        assuming your trainer calls scheduler.step() every optimizer step.
    """
    optimizer = kwargs.pop("optimizer", None)
    if optimizer is None:
        raise ValueError("CosineAnnealingWarmRestarts requires 'optimizer' in kwargs")

    # build scheduler with remaining kwargs (T_0, T_mult, eta_min, last_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, **kwargs
    )

    scheduler_info = dict(level="step")
    return scheduler, scheduler_info