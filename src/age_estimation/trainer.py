import os
import math
import time
import json
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset
import datetime

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")

from age_estimation.registry import DATASETS, MODELS, OPTIMIZERS, SCHEDULERS, LOSSES, METRICS


class CheckpointManager:
    """Manages model checkpoints with top-k saving capability."""
    
    def __init__(
        self,
        checkpoints_dir,
        save_top_k=5,
        monitor='mae',
        mode='min',
        filename='checkpoint-{epoch:02d}-{monitor:.4f}',
        save_last=True,
        save_best=True,
        save_weights_only=True
    ):
        self.checkpoints_dir = checkpoints_dir
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.save_last = save_last
        self.save_best = save_best
        self.save_weights_only = save_weights_only
        
        self.best_checkpoints = []  # List of (metric_value, checkpoint_path)
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    def _is_better(self, current, best):
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best

    def _cast_state_dict(self, sd, dtype=None):
        if dtype is None:
            return sd
        out = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                out[k] = v.to(dtype).cpu()
            else:
                out[k] = v
        return out

    def save(self, model, optimizer, epoch, train_losses, val_losses, val_metrics,
            lr_scheduler=None, accelerator=None, global_step=0, storage_cast=None):
        """
        storage_cast: None | torch.float16 | torch.bfloat16  (applied to weights-only saves)
        """
        metric_value = val_losses.get(self.monitor, val_losses.get('val_loss', float('inf')))
        
        # always collect model weights
        state_dict = accelerator.get_state_dict(model) if accelerator is not None else model.state_dict()

        # -------- full resume file (last.pth) --------
        if self.save_last:
            full = {
                'epoch': epoch,
                'global_step': global_step,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if lr_scheduler is not None:
                full['scheduler_state_dict'] = lr_scheduler.state_dict()
            last_path = os.path.join(self.checkpoints_dir, 'last.pth')
            torch.save(full, last_path)
            logger.info(f"Saved latest checkpoint to {last_path}")

        # -------- tiny files (best + top-k) --------
        tiny_sd = self._cast_state_dict(state_dict, dtype=storage_cast) if self.save_weights_only else state_dict
        tiny_meta = {
            'epoch': epoch,
            'global_step': global_step,
            self.monitor: metric_value,
            'model_state_dict': tiny_sd,
        }

        # best.pth
        if self._is_better(metric_value, self.best_metric_value):
            self.best_metric_value = metric_value
            if self.save_best:
                best_path = os.path.join(self.checkpoints_dir, 'best.pth')
                torch.save(tiny_meta if self.save_weights_only else {**tiny_meta, 'optimizer_state_dict': optimizer.state_dict(),
                                                                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None},
                        best_path)
                logger.info(f"Saved best checkpoint to {best_path} ({self.monitor}={metric_value:.4f})")

        # top-k
        if self.save_top_k > 0:
            fname = f'epoch_{epoch:03d}-{self.monitor}_{metric_value:.4f}.pth'
            path = os.path.join(self.checkpoints_dir, fname)
            torch.save(tiny_meta if self.save_weights_only else {**tiny_meta, 'optimizer_state_dict': optimizer.state_dict(),
                                                                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None},
                    path)
            self.best_checkpoints.append((metric_value, path))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
            if len(self.best_checkpoints) > self.save_top_k:
                _, remove_path = self.best_checkpoints.pop()
                if os.path.exists(remove_path):
                    os.remove(remove_path)
                    logger.info(f"Removed checkpoint: {remove_path}")

    # def save(self, model, optimizer, epoch, train_losses, val_losses, val_metrics, 
    #          lr_scheduler=None, accelerator=None, global_step=0):
    #     """Save checkpoint and manage top-k checkpoints."""
        
    #     # Extract monitor metric value
    #     metric_value = val_losses.get(self.monitor, val_losses.get('val_loss', float('inf')))
        
    #     # Prepare checkpoint data
    #     checkpoint = {
    #         'epoch': epoch,
    #         'global_step': global_step,
    #         'train_losses': train_losses,
    #         'val_losses': val_losses,
    #         'val_metrics': val_metrics,
    #         'optimizer_state_dict': optimizer.state_dict(),
    #     }
        
    #     # Add model state
    #     if accelerator is not None:
    #         checkpoint['model_state_dict'] = accelerator.get_state_dict(model)
    #     else:
    #         checkpoint['model_state_dict'] = model.state_dict()
        
    #     # Add scheduler state
    #     if lr_scheduler is not None:
    #         checkpoint['scheduler_state_dict'] = lr_scheduler.state_dict()
        
    #     # Save last checkpoint
    #     if self.save_last:
    #         last_path = os.path.join(self.checkpoints_dir, 'last.pth')
    #         torch.save(checkpoint, last_path)
    #         logger.info(f"Saved latest checkpoint to {last_path}")
        
    #     # Save best checkpoint
    #     if self._is_better(metric_value, self.best_metric_value):
    #         self.best_metric_value = metric_value
    #         if self.save_best:
    #             best_path = os.path.join(self.checkpoints_dir, 'best.pth')
    #             torch.save(checkpoint, best_path)
    #             logger.info(f"Saved best checkpoint to {best_path} ({self.monitor}={metric_value:.4f})")
        
    #     # Save top-k checkpoints
    #     if self.save_top_k > 0:
    #         # Format filename with epoch and metric
    #         filename = f'checkpoint-epoch_{epoch:03d}-{self.monitor}_{metric_value:.4f}.pth'
    #         checkpoint_path = os.path.join(self.checkpoints_dir, filename)
    #         torch.save(checkpoint, checkpoint_path)
            
    #         # Add to best checkpoints list
    #         self.best_checkpoints.append((metric_value, checkpoint_path))
            
    #         # Sort by metric value
    #         self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
            
    #         # Remove excess checkpoints
    #         if len(self.best_checkpoints) > self.save_top_k:
    #             _, remove_path = self.best_checkpoints.pop()
    #             if os.path.exists(remove_path):
    #                 os.remove(remove_path)
    #                 logger.info(f"Removed checkpoint: {remove_path}")


class AgeEstimationTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.use_mlflow = self.configs.get('use_mlflow', False) and MLFLOW_AVAILABLE
        self.setup()
    # print("i'm in the new trainer")
    def setup(self):
        self.setup_project()
        self.setup_dataloaders()
        self.setup_model()
        self.setup_loss()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_metrics()
        self.prepare_accelerator()
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_mae = float('inf')
        self.best_val_metrics = {}
        self.current_epoch = 0
        self.global_step = 0
        self.val_global_step = 0
        self.patience_counter = 0

        if self.main_process:
            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(self.configs.to_dict(), f, indent=4)
            
            # Initialize MLflow if enabled
            if self.use_mlflow:
                self.setup_mlflow()

    def setup_project(self):
        base_dir = self.configs.get('project_dir', 'outputs')
        use_timestamp = self.configs.get('use_timestamp', True)
        
        if use_timestamp:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join(base_dir, ts)
        else:
            self.output_dir = base_dir
        
        self.checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=self.logs_dir
        )
        
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        use_wandb = self.configs.get('use_wandb', False)
        log_with = 'wandb' if use_wandb else 'tensorboard'

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.configs.get('gradient_accumulation_steps', 1),
            mixed_precision=self.configs.get('mixed_precision', 'no'),
            log_with=log_with,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )   
        
        self.main_process = self.accelerator.is_main_process
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if self.accelerator.mixed_precision == "bf16" else \
                      torch.float16 if self.accelerator.mixed_precision == "fp16" else \
                      torch.float32

        if self.main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            logger.info(f"Project directory: {self.output_dir}")
            logger.info(f"Checkpoints directory: {self.checkpoints_dir}")
            logger.info(f"Logs directory: {self.logs_dir}")

    def setup_mlflow(self):
        """Initialize MLflow tracking and log configuration."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping MLflow setup")
            return
        
        # Set MLflow tracking URI
        mlflow_dir = os.path.join(self.output_dir, 'mlruns')
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow_dir_abs = os.path.abspath(mlflow_dir)
        mlflow.set_tracking_uri(f"file://{mlflow_dir_abs}")
        
        # Set experiment name
        experiment_name = self.configs.get('project_name', 'age_estimation')
        mlflow.set_experiment(experiment_name)
        
        # End any existing run
        if mlflow.active_run() is not None:
            mlflow.end_run()
        
        # Start new run
        run_name = f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow experiment: {experiment_name}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        # Log configuration parameters
        config_dict = self.configs.to_dict()
        flattened_params = self._flatten_dict(config_dict)
        mlflow.log_params(flattened_params)
        
        # Log system information
        mlflow.set_tag("gpu_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
        mlflow.set_tag("device", str(self.device))
        mlflow.set_tag("dtype", str(self.dtype))
        mlflow.set_tag("mixed_precision", self.configs.get('mixed_precision', 'no'))
        mlflow.set_tag("num_processes", self.accelerator.num_processes)
        
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        mlflow.log_param("model_total_params", total_params)
        mlflow.log_param("model_trainable_params", trainable_params)
        
        # Log config file as artifact
        config_path = os.path.join(self.output_dir, "config.json")
        mlflow.log_artifact(config_path, "config")
        
        logger.info("MLflow setup completed")

    def _flatten_dict(self, d, parent_key="", sep="."):
        """Flatten nested dictionary for MLflow logging."""
        allowed_types = (int, float, str, bool)
        items = {}
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, (list, tuple)) or v is None:
                items[new_key] = json.dumps(v)
            elif isinstance(v, allowed_types):
                items[new_key] = v
            else:
                try:
                    items[new_key] = float(v) if isinstance(v, (int, float)) else str(v)
                except Exception:
                    items[new_key] = str(v)
        
        return items

    def setup_dataloaders(self):
        train_dataset = DATASETS.build(self.configs.train_dataset.to_dict())
        valid_dataset = DATASETS.build(self.configs.valid_dataset.to_dict())
        test_dataset = DATASETS.build(self.configs.test_dataset.to_dict())
        
        
        # inside setup_dataloaders(), right after datasets are built
        debug_overfit_n = int(self.configs.trainer.get("debug_overfit_n", 0))
        if debug_overfit_n > 0:
            base_train = train_dataset
            idx = list(range(min(debug_overfit_n, len(base_train))))
            train_dataset = Subset(base_train, idx)
            valid_dataset = Subset(base_train, idx)
            test_dataset = Subset(base_train, idx)
            if self.main_process:
                logger.warning(f"OVERFIT DEBUG: using same {len(idx)} samples for train/val/test")


        # Get dataloader configuration
        num_workers = self.configs.trainer.get('num_workers', 4)
        pin_memory = self.configs.trainer.get('pin_memory', True)

        bs  = self.configs.trainer.batch_size
        vbs = self.configs.trainer.get('valid_batch_size', bs)
        tbs = self.configs.trainer.get('test_batch_size',  bs)

        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=vbs, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=tbs, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        if self.main_process:
            logger.info(f"Train dataloader: {len(train_dataset)} samples")
            logger.info(f"Valid dataloader: {len(valid_dataset)} samples")
            logger.info(f"Test dataloader: {len(test_dataset)} samples")

    def setup_model(self):
        # Put this in trainer_new.setup_model() BEFORE self.model.to(...)
        from collections import Counter

        self.model = MODELS.build(self.configs.model.to_dict())

        pre = getattr(self.configs.trainer, "pretrained_from", None)
        if pre:
            pre = os.path.expanduser(str(pre))
            if not os.path.exists(pre):
                logger.warning(f"pretrained_from not found: {pre}")
            else:
                ckpt = torch.load(pre, map_location="cpu")
                raw = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt.get("state_dict") or ckpt

                # --- candidate transforms ---
                def id_map(k): return k
                def strip_module(k): return k[7:] if k.startswith("module.") else k
                def strip_model(k):  return k[6:] if k.startswith("model.") else k
                def strip_backbone(k): return k[9:] if k.startswith("backbone.") else k
                def add_model(k):    return "model."+k
                def add_backbone(k): return "backbone."+k

                trans = [
                    ("identity", id_map),
                    ("strip_module", strip_module),
                    ("strip_model",  strip_model),
                    ("strip_backbone", strip_backbone),
                    ("add_model",    add_model),
                    ("add_backbone", add_backbone),
                ]

                msd = self.model.state_dict()
                best_name, best_map, best_overlap = None, None, -1

                for name, fn in trans:
                    cand = {fn(k): v for k, v in raw.items()}
                    overlap = sum(1 for k, v in cand.items() if k in msd and msd[k].shape == v.shape)
                    if overlap > best_overlap:
                        best_name, best_map, best_overlap = name, cand, overlap

                logger.info(f"Best key mapping: {best_name} with overlap {best_overlap}/{len(msd)}")

                # If overlap still tiny, try suffix-based rescue (longest matching suffix with same shape)
                if best_overlap < 100:
                    suffix_map = {}
                    msd_items = list(msd.items())
                    for ck, v in raw.items():
                        parts = ck.split('.')
                        best_key, best_len = None, 0
                        for mk, mv in msd_items:
                            # match by suffix segments and shape
                            for i in range(len(parts)):
                                if mk.endswith('.'.join(parts[i:])) and mv.shape == v.shape and (len(parts)-i) > best_len:
                                    best_key, best_len = mk, (len(parts)-i)
                        if best_key and best_key not in suffix_map:
                            suffix_map[best_key] = v
                    # prefer suffix map if it improves overlap
                    if len(suffix_map) > best_overlap:
                        logger.info(f"Using suffix-match map with overlap {len(suffix_map)}/{len(msd)}")
                        best_map = suffix_map
                        best_overlap = len(suffix_map)

                missing, unexpected = self.model.load_state_dict(best_map, strict=False)
                logger.info(f"Loaded pretrained weights from {pre}")
                logger.info(f"Final overlap: {best_overlap}/{len(msd)} | Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        # now move to device
        self.model.to(device=self.device, dtype=self.dtype)

        
        # raise SystemExit(0)
    # def setup_model(self):
    #     self.model = MODELS.build(self.configs.model.to_dict())
    #     self.model.to(self.device, self.dtype)
        
    #     if self.main_process:
    #         total_params = sum(p.numel() for p in self.model.parameters())
    #         trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #         logger.info(f"Model parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")

    def setup_optimizer(self):
        optimizer_cfg = self.configs.trainer.optimizer.to_dict()

        # Honor model-defined param grouping when available
        use_param_groups = optimizer_cfg.pop("use_param_groups", True)
        lr = optimizer_cfg.get("lr", 1e-4)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)

        if use_param_groups and hasattr(self.model, "param_groups"):
            params = self.model.param_groups(lr_low=lr, lr_high=lr*3, wd=weight_decay)
        else:
            params = self.model.parameters()

        # Materialize for inspection / augmentation
        params = list(params)

        # Normalize to a list of param-group dicts so optimizer never sees bare dict inside a tensor list
        if params and isinstance(params[0], dict):
            param_groups = params  # model already returned grouped dicts
        else:
            param_groups = [{"params": params, "lr": lr, "weight_decay": weight_decay}]

        # Also optimize loss parameters (e.g., MediAge prototypes)
        loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        if loss_params:
            param_groups.append({"params": loss_params, "lr": lr, "weight_decay": weight_decay})

        optimizer_cfg["params"] = param_groups
        self.optimizer = OPTIMIZERS.build(optimizer_cfg)
        # if self.main_process:
        #     for i, g in enumerate(self.optimizer.param_groups):
        #         n = sum(p.numel() for p in g["params"])
                # logger.info(f"[DEBUG] group{i}: lr={g['lr']}, wd={g.get('weight_decay', None)}, nparams={n}")

    def setup_scheduler(self):
        scheduler_cfg = self.configs.trainer.scheduler.to_dict()
        scheduler_cfg["optimizer"] = self.optimizer
        self.scheduler, self.scheduler_info = SCHEDULERS.build(scheduler_cfg)

        # Determine stepping level from scheduler_info first, with safe fallback.
        level = str(getattr(self, "scheduler_info", {}).get("level", "")).lower().strip()
        if level not in {"step", "epoch", "metric"}:
            cfg_name = (scheduler_cfg.get("type") or scheduler_cfg.get("name") or "").lower()
            actual_sched = getattr(self.scheduler, "scheduler", self.scheduler)
            class_name = actual_sched.__class__.__name__.lower()
            if "reducelronplateau" in cfg_name or "reducelronplateau" in class_name:
                level = "metric"
            elif "onecyclelr" in cfg_name or "onecyclelr" in class_name:
                level = "step"
            else:
                level = "epoch"

        self._sched_batch = (level == "step")
        self._sched_needs_metric = (level == "metric")
        self._sched_epoch = (level == "epoch")

        # if self.main_process:
            # logger.info(f"[DEBUG] scheduler level={level}")

    def setup_loss(self):
        loss_cfg = self.configs.trainer.loss.to_dict()
        
        # Get feature dimension from model for center loss
        if 'center_loss' in [loss['name'] for loss in loss_cfg.get('losses', [])]:
            if hasattr(self.model, 'model_size'):
                if self.model.model_size == 'S':
                    feat_dim = 224
                elif self.model.model_size == 'B':
                    feat_dim = 384
                elif self.model.model_size == 'L':
                    feat_dim = 512
                else:
                    feat_dim = 224
            else:
                feat_dim = 224
            loss_cfg['feat_dim'] = feat_dim
        
        self.loss_fn = LOSSES.build(loss_cfg)
        # keep loss parameters (e.g., prototypes) on same device/dtype as the model
        self.loss_fn.to(self.device, dtype=self.dtype)

    def setup_metrics(self):
        self.metrics_fn = METRICS.build(self.configs.trainer.metrics.to_dict())

    def setup_checkpoint_manager(self):
        """Setup checkpoint manager with configuration."""
        checkpoint_cfg = self.configs.trainer.get('checkpoints', {})
        
        save_top_k = checkpoint_cfg.get('save_top_k', 5)
        monitor = checkpoint_cfg.get('monitor', 'val_loss')
        mode = checkpoint_cfg.get('mode', 'min')
        filename = checkpoint_cfg.get('filename', 'checkpoint-{epoch:02d}-{monitor:.4f}')
        save_last = checkpoint_cfg.get('save_last', True)
        save_best = checkpoint_cfg.get('save_best', True)
        save_weights_only = checkpoint_cfg.get('save_weights_only', True)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoints_dir=self.checkpoints_dir,
            save_top_k=save_top_k,
            monitor=monitor,
            mode=mode,
            filename=filename,
            save_last=save_last,
            save_best=save_best,
            save_weights_only=save_weights_only
        )

    def prepare_accelerator(self):
        """Prepare model, optimizer, scheduler and dataloaders with accelerator"""
        self.model, self.loss_fn, self.optimizer, self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.accelerator.prepare(
            self.model, self.loss_fn, self.optimizer, self.train_dataloader, self.valid_dataloader, self.test_dataloader
        )
        
        # Prepare scheduler
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)
        
        # Setup checkpoint manager
        self.setup_checkpoint_manager()
        
        # Initialize trackers
        self.accelerator.init_trackers(
            project_name=self.configs.get('project_name', 'age_estimation'),
            config=self._flatten_dict(self.configs.to_dict())
        )
               
    def save_checkpoint(self, epoch, step, val_loss=None, mae=None, is_best=False,
                        val_metrics=None, test_metrics=None):
        if not self.main_process:
            return
        # build the small dicts the manager expects
        train_losses = {}  # you don't need to store history in ckpt
        val_losses = {'val_loss': val_loss if val_loss is not None else float('inf')}
        if isinstance(val_metrics, dict):
            val_losses.update({k: v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        # choose optional on-disk cast for tiny files
        storage_cast = torch.float16  # or torch.bfloat16, or None to keep fp32
        
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_metrics or {},
            lr_scheduler=self.scheduler,
            accelerator=self.accelerator,
            global_step=step,
            storage_cast=storage_cast
        )

    # def save_checkpoint(self, epoch, step, val_loss=None, mae=None, is_best=False,
    #                val_metrics=None, test_metrics=None):
    #     """Save checkpoint with training state"""
    #     if not self.main_process:
    #         return
            
    #     checkpoint = {
    #         'epoch': epoch,
    #         'global_step': step,
    #         'model_state_dict': self.accelerator.get_state_dict(self.model),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
    #         'configs': self.configs.to_dict(),
    #         'best_val_loss': self.best_val_loss,
    #         'best_mae': self.best_mae,
    #         'val_loss': val_loss,
    #         'mae': mae,
    #     }

    #     if isinstance(val_metrics, dict):
    #         checkpoint['val_metrics'] = val_metrics
    #     if isinstance(test_metrics, dict):
    #         checkpoint['test_metrics'] = test_metrics

    #     # Save latest checkpoint
    #     latest_path = os.path.join(self.checkpoints_dir, 'latest.pth')
    #     torch.save(checkpoint, latest_path)
    #     logger.info(f"Saved latest checkpoint to {latest_path}")
        
    #     # Save epoch checkpoint
    #     epoch_path = os.path.join(self.checkpoints_dir, f'epoch_{epoch}.pth')
    #     torch.save(checkpoint, epoch_path)
        
    #     # Save best checkpoint
    #     if is_best:
    #         best_path = os.path.join(self.checkpoints_dir, 'best.pth')
    #         torch.save(checkpoint, best_path)
    #         logger.info(f"Saved best checkpoint to {best_path}")
            
    #         # Log to MLflow
    #         # if self.use_mlflow:
    #         #     mlflow.log_artifact(best_path, "checkpoints")
               
    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint and resume training"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoints_dir, 'last.pth')
            
        if not os.path.exists(checkpoint_path):
            logger.info(f"No checkpoint found at {checkpoint_path}")
            return False
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_mae = checkpoint.get('best_mae', float('inf'))
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        return True
    
    def __call__(self):
        self.train()
        self.cleanup()

    def train(self):
        """Main training loop with proper checkpointing and evaluation"""
        # Load checkpoint if resuming
        resume_from_checkpoint = self.configs.trainer.get('resume_from_checkpoint', None)
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        elif self.configs.trainer.get('auto_resume', True):
            self.load_checkpoint()
            
        self.model.train()
        
        num_epochs = self.configs.trainer.get('max_epochs', 10)
        eval_steps = self.configs.trainer.get('eval_steps', 1000)
        log_steps = self.configs.trainer.get('log_steps', 100)
        save_steps = self.configs.trainer.get('save_steps', 1000)
        gradient_accumulation_steps = self.configs.trainer.get('gradient_accumulation_steps', 1)
        early_stopping_patience = self.configs.trainer.get('early_stopping_patience', 5)
        max_grad_norm = self.configs.trainer.get('max_grad_norm', 1.0)
        
        # Calculate total steps
        num_updates_per_epoch = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps)
        max_train_steps = num_updates_per_epoch * num_epochs
        
        running_loss = 0.0
        
        # Start training from current epoch
        for epoch in range(self.current_epoch, num_epochs):
            if self.main_process:
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
            # --- optional staged unfreeze of ViT ---
            unfreeze_epoch = self.configs.trainer.get("unfreeze_vit_epoch", None)
            if unfreeze_epoch is not None and epoch == int(unfreeze_epoch):
                m = self.accelerator.unwrap_model(self.model)
                n_last = int(self.configs.trainer.get("unfreeze_vit_last_n", 4))
                use_vit_flag = getattr(m, "use_vit", None)
                if use_vit_flag is False:
                    if self.main_process:
                        logger.info("Skipping ViT unfreeze: model.use_vit is False")
                elif hasattr(m, "unfreeze_vit_last_n"):
                    m.unfreeze_vit_last_n(n_last)
                    if self.main_process:
                        logger.info(f"Unfroze last {n_last} ViT blocks at epoch {epoch+1}")
                else:
                    if self.main_process:
                        logger.warning("Model has no unfreeze_vit_last_n() method")

            epoch_loss = 0.0
            num_epoch_batches = 0
            
            # Create progress bar for main process
            progress_bar = None
            if self.main_process:
                progress_bar = tqdm(
                    total=len(self.train_dataloader),
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    ncols=100,
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=1.0,
                    maxinterval=5.0,
                    miniters=1,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                )
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    step_result = self.train_step(batch, epoch, step, max_grad_norm)
                    
                running_loss += step_result['loss']
                epoch_loss += step_result['loss']
                num_epoch_batches += 1
                self.global_step += 1
                
                # Update progress bar
                if self.main_process and progress_bar:
                    progress_bar.update(1)
                
                # Logging
                if self.global_step % log_steps == 0:
                    avg_loss = running_loss / log_steps
                    if self.main_process:
                        if progress_bar:
                            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                        # logger.info(f"Epoch {epoch + 1}/{num_epochs}, Step {self.global_step}, Loss: {avg_loss:.4f}")
                        
                        # Log to accelerator
                        self.accelerator.log({
                            "train/loss": avg_loss,
                            "train/epoch": epoch + 1,
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        }, step=self.global_step)
                        
                        # Log to MLflow
                        if self.use_mlflow:
                            mlflow.log_metrics({
                                "train_step/loss": avg_loss,
                                "train_step/learning_rate": self.optimizer.param_groups[0]['lr']
                            }, step=self.global_step)
                        
                    running_loss = 0.0
                
                # Evaluation
                # if self.global_step % eval_steps == 0:
                #     val_results = self.evaluate()
                #     test_results = self.test()
                if self.global_step > 0 and self.global_step % eval_steps == 0:
                    val_results = self.evaluate()
                    test_results = self.test()
                    self.model.train()   # IMPORTANT: restore train mode

                    # Check if this is the best model
                    val_loss = val_results.get('val_loss', float('inf'))
                    val_mae = val_results.get('mae', float('inf'))
                    is_best = val_loss < self.best_val_loss
                    
                    if is_best:
                        self.best_val_loss = val_loss
                        self.best_mae = val_mae
                        self.best_val_metrics = val_results
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                    # Save checkpoint
                    self.save_checkpoint(epoch, self.global_step, val_loss, val_mae, is_best,
                                       val_metrics=val_results, test_metrics=test_results)
                    
                    # Early stopping
                    if early_stopping_patience > 0 and self.patience_counter >= early_stopping_patience:
                        if self.main_process:
                            logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                        break
                    
                    self.model.train()
                
                # Save intermediate checkpoints
                elif self.global_step % save_steps == 0:
                    self.save_checkpoint(epoch, self.global_step)
            
            if self.main_process and progress_bar:
                progress_bar.close()
            
            # End of epoch - save checkpoint and evaluate
            avg_epoch_loss = epoch_loss / num_epoch_batches if num_epoch_batches > 0 else 0.0
            self.current_epoch = epoch + 1
            
            if self.main_process:
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Final evaluation of the epoch
            val_results = self.evaluate()
            test_results = self.test()
            
            val_loss = val_results.get('val_loss', float('inf'))
            val_mae = val_results.get('mae', float('inf'))
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.best_mae = val_mae
                self.best_val_metrics = val_results
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Step scheduler
            if self.scheduler:
                if self._sched_needs_metric:
                    metric_name = self.configs.trainer.get("scheduler_monitor", "val_loss")
                    metric_value = val_results.get(metric_name, val_loss)
                    self.scheduler.step(metric_value)
                elif self._sched_epoch:
                    self.scheduler.step()

            # Save end-of-epoch checkpoint
            self.save_checkpoint(self.current_epoch, self.global_step, val_loss, val_mae, is_best,
                               val_metrics=val_results, test_metrics=test_results)
            
            # Log to MLflow (epoch-level)
            if self.main_process and self.use_mlflow:
                mlflow.log_metrics({
                    "train_epoch/loss": avg_epoch_loss,
                    "val_epoch/loss": val_loss,
                    "val_epoch/mae": val_mae
                }, step=epoch)
            
            # Early stopping check
            if early_stopping_patience > 0 and self.patience_counter >= early_stopping_patience:
                if self.main_process:
                    logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        if self.main_process:
            logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}, Best MAE: {self.best_mae:.4f}")

    def _align_output_label_shapes(self, outputs, labels):
        """Normalize common singleton-target layouts: (B,1) <-> (B,)."""
        if outputs.shape == labels.shape:
            return outputs, labels

        if (
            outputs.ndim == labels.ndim + 1
            and outputs.shape[-1] == 1
            and outputs.shape[:-1] == labels.shape
        ):
            return outputs, labels.unsqueeze(-1)

        if (
            labels.ndim == outputs.ndim + 1
            and labels.shape[-1] == 1
            and labels.shape[:-1] == outputs.shape
        ):
            return outputs.unsqueeze(-1), labels

        return outputs, labels
    def _debug_regression_stats(self, outputs, labels):
        p = outputs.detach().float().view(-1)
        y = labels.detach().float().view(-1)

        p_std = p.std(unbiased=False).item()
        y_std = y.std(unbiased=False).item()
        pc = float("nan")
        if p.numel() > 1 and p_std > 1e-8 and y_std > 1e-8:
            pc = torch.corrcoef(torch.stack([p, y]))[0, 1].item()

        return {
            "pred_mean": p.mean().item(),
            "pred_std": p_std,
            "pred_min": p.min().item(),
            "pred_max": p.max().item(),
            "target_mean": y.mean().item(),
            "target_std": y_std,
            "target_min": y.min().item(),
            "target_max": y.max().item(),
            "batch_mae": (p - y).abs().mean().item(),
            "batch_pc": pc,
            "outside_0_8": ((p < 0) | (p > 8)).float().mean().item(),
        }

    def train_step(self, batch, epoch, step, max_grad_norm=1.0):
        """Single training step with gradient accumulation and clipping"""
        # if self.main_process and epoch == 0 and step < 3:
            # logger.info(f"[DEBUG] model.training = {self.model.training}")

        images, labels = batch
        # if self.main_process and epoch == 0 and step < 3:
            # b = images.shape[0]
            # per-sample intensity stats
            # m = images.float().mean(dim=(1,2,3))
            # s = images.float().std(dim=(1,2,3))
            # logger.info(f"[DEBUG] batch img mean range: {m.min().item():.4f}..{m.max().item():.4f}")
            # logger.info(f"[DEBUG] batch img std  range: {s.min().item():.4f}..{s.max().item():.4f}")

        # Forward pass
        outputs, features = self.model(images)
        outputs, labels = self._align_output_label_shapes(outputs, labels)

        # if self.main_process and epoch == 0 and step < 5:
        #     logger.info(f"[DEBUG] {self._debug_regression_stats(outputs, labels)}")

        # HARD ASSERT: prevents silent broadcasting bugs
        if outputs.shape != labels.shape:
            raise RuntimeError(
                f"Shape mismatch: outputs {tuple(outputs.shape)} vs labels {tuple(labels.shape)}. "
                "This will silently broadcast and destroy learning."
            )

        # quick stats once
        # if self.main_process and epoch == 0 and step == 0:
            # logger.info(f"[DEBUG] labels mean/std per-dim: mean={labels.float().mean(0)}, std={labels.float().std(0)}")
        loss_dict = self.loss_fn(outputs, labels, features)
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        self.accelerator.backward(loss)
        if self.main_process and epoch == 0 and step < 3:
            m = self.accelerator.unwrap_model(self.model)
            # pick a couple of key params
            probes = []
            for n, p in m.named_parameters():
                if ("head" in n or "fapms" in n) and p.requires_grad:
                    if p.grad is not None:
                        probes.append((n, p.grad.detach().float().norm().item()))
                    else:
                        probes.append((n, None))
                    if len(probes) >= 6:
                        break
            # logger.info(f"[DEBUG] grad norms (head/fapm): {probes}")
        # Only step optimizer and scheduler if not accumulating gradients
        if self.accelerator.sync_gradients:
            # Gradient clipping
            if max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            # DEBUG: confirm optimizer steps actually happen
            # if self.main_process and epoch == 0 and step < 5:
                # logger.info(f"[DEBUG] OPTIMIZER STEP at epoch={epoch} step={step} global_step={self.global_step}")
                
            # Step scheduler per-batch only if it's batch-based
            if self.scheduler and self._sched_batch:
                self.scheduler.step()

            self.optimizer.zero_grad()

        return {
            'loss': loss.item(),
            'epoch': epoch,
            'step': step,
            'loss_dict': {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        }

    def evaluate(self):
        """Evaluate model on validation set"""
        was_training = self.model.training
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, desc="Validation", disable=not self.main_process):
                images, labels = batch
                
                outputs, features = self.model(images)
                outputs, labels = self._align_output_label_shapes(outputs, labels)
                loss_dict = self.loss_fn(outputs, labels, features)
                
                # Gather predictions and targets
                predictions = self.accelerator.gather_for_metrics(outputs)
                targets = self.accelerator.gather_for_metrics(labels)
                
                if self.main_process:
                    batch_size = predictions.size(0)
                    total_samples += batch_size
                    
                    loss = loss_dict['total_loss']
                    total_loss += loss.item() * batch_size
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())

        m = self.accelerator.unwrap_model(self.model)
        if hasattr(m, "cleanup_eval_backbone"):
            m.cleanup_eval_backbone()
    
        # Calculate final metrics
        results = {}
        if self.main_process:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            
            # Concatenate all predictions and targets
            if all_predictions:
                all_predictions = torch.cat(all_predictions, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                
                # Calculate metrics
                metrics_dict = self.metrics_fn(all_predictions, all_targets)
                
                # Convert to scalars
                metrics_results = {}
                for metric_name, metric_value in metrics_dict.items():
                    metrics_results[metric_name] = metric_value.item() if torch.is_tensor(metric_value) else metric_value
                
                results = {
                    'val_loss': avg_loss,
                    'num_samples': total_samples,
                    **metrics_results
                }
                
                # Log
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_results.items()])
                logger.info(f"Validation Results - Loss: {avg_loss:.4f}, {metrics_str}")
                
                # Log to accelerator
                log_dict = {"val/loss": avg_loss}
                for metric_name, metric_value in metrics_results.items():
                    log_dict[f"val/{metric_name}"] = metric_value
                
                self.accelerator.log(log_dict, step=self.global_step)
                
                # Log to MLflow
                if self.use_mlflow:
                    mlflow_dict = {}
                    for k, v in metrics_results.items():
                        mlflow_dict[f"val/{k}"] = v
                    mlflow.log_metrics(mlflow_dict, step=self.global_step)
            else:
                results = {'val_loss': avg_loss, 'num_samples': total_samples}
                logger.info(f"Validation Results - Loss: {avg_loss:.4f}")
                self.accelerator.log({"val/loss": avg_loss}, step=self.global_step)
        if was_training:
            self.model.train()

        return results

    def test(self):
        """Evaluate model on test set"""
        was_training = self.model.training
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing", disable=not self.main_process):
                images, labels = batch
                
                outputs, features = self.model(images)
                outputs, labels = self._align_output_label_shapes(outputs, labels)
                loss_dict = self.loss_fn(outputs, labels, features)
                
                # Gather predictions and targets
                predictions = self.accelerator.gather_for_metrics(outputs)
                targets = self.accelerator.gather_for_metrics(labels)
                
                if self.main_process:
                    batch_size = predictions.size(0)
                    total_samples += batch_size
                    
                    loss = loss_dict['total_loss']
                    total_loss += loss.item() * batch_size
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())

        m = self.accelerator.unwrap_model(self.model)
        if hasattr(m, "cleanup_eval_backbone"):
            m.cleanup_eval_backbone()
            
        # Calculate final metrics
        results = {}
        if self.main_process:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            
            # Concatenate all predictions and targets
            if all_predictions:
                all_predictions = torch.cat(all_predictions, dim=0)
                all_targets = torch.cat(all_targets, dim=0)

                # DEBUG: variance + dumb baseline (mean predictor)
                pred = all_predictions.float()
                tgt  = all_targets.float()

                # logger.info(f"[DEBUG] pred mean/std: mean={pred.mean(0)}, std={pred.std(0)}")
                # logger.info(f"[DEBUG] tgt  mean/std: mean={tgt.mean(0)}, std={tgt.std(0)}")

                mu = tgt.mean(0, keepdim=True)
                baseline_mae = (tgt - mu).abs().mean(0)
                model_mae = (tgt - pred).abs().mean(0)
                # logger.info(f"[DEBUG] baseline MAE per-dim = {baseline_mae}")
                # logger.info(f"[DEBUG] model    MAE per-dim = {model_mae}")

                # Calculate metrics
                metrics_dict = self.metrics_fn(all_predictions, all_targets)
                
                # Convert to scalars
                metrics_results = {}
                for metric_name, metric_value in metrics_dict.items():
                    metrics_results[metric_name] = metric_value.item() if torch.is_tensor(metric_value) else metric_value
                
                results = {
                    'test_loss': avg_loss,
                    'num_samples': total_samples,
                    **{f"test_{k}": v for k, v in metrics_results.items()}
                }
                
                # Log
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_results.items()])
                logger.info(f"Test Results - Loss: {avg_loss:.4f}, {metrics_str}")
                
                # Log to accelerator
                log_dict = {"test/loss": avg_loss}
                for metric_name, metric_value in metrics_results.items():
                    log_dict[f"test/{metric_name}"] = metric_value
                
                self.accelerator.log(log_dict, step=self.global_step)
                
                # Log to MLflow
                if self.use_mlflow:
                    mlflow_dict = {}
                    for k, v in metrics_results.items():
                        mlflow_dict[f"test/{k}"] = v
                    mlflow.log_metrics(mlflow_dict, step=self.global_step)
            else:
                results = {'test_loss': avg_loss, 'num_samples': total_samples}
                logger.info(f"Test Results - Loss: {avg_loss:.4f}")
                self.accelerator.log({"test/loss": avg_loss}, step=self.global_step)

        if was_training:
            self.model.train()

        return results

    def predict(self, dataloader):
        """Run inference on a given dataloader"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction", disable=not self.main_process):
                images, labels = batch
                
                outputs, _ = self.model(images)
                outputs, labels = self._align_output_label_shapes(outputs, labels)
                
                # Gather predictions and targets
                predictions = self.accelerator.gather(outputs)
                targets = self.accelerator.gather(labels)
                
                if self.main_process:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

        m = self.accelerator.unwrap_model(self.model)
        if hasattr(m, "cleanup_eval_backbone"):
            m.cleanup_eval_backbone()

        return all_predictions, all_targets
    
    def load_best_model(self):
        """Load the best model checkpoint"""
        best_checkpoint_path = os.path.join(self.checkpoints_dir, 'best.pth')
        if os.path.exists(best_checkpoint_path):
            logger.info("Loading best model checkpoint...")
            self.load_checkpoint(best_checkpoint_path)
            return True
        else:
            logger.warning("No best model checkpoint found")
            return False
    
    def get_training_stats(self):
        """Get current training statistics"""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_mae': self.best_mae,
            'patience_counter': self.patience_counter,
        }

    def cleanup(self):
        """Clean up and finalize training"""
        if self.main_process:
            logger.info("Cleaning up training...")
            self.save_checkpoint(self.current_epoch, self.global_step, is_best=False)
            
            # Log final metrics to MLflow
            if self.use_mlflow:
                if self.best_val_loss != float('inf'):
                    mlflow.log_metric("final_best_val_loss", self.best_val_loss)
                    mlflow.log_metric("final_best_mae", self.best_mae)
                    
                    # Log best validation metrics
                    for metric_name, metric_value in self.best_val_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"final_best_{metric_name}", metric_value)
                
                mlflow.log_metric("total_train_steps", self.global_step)
                mlflow.log_metric("final_epoch", self.current_epoch)
                
                # Log final checkpoints
                best_checkpoint = os.path.join(self.checkpoints_dir, 'best.pth')
                if os.path.exists(best_checkpoint):
                    mlflow.log_artifact(best_checkpoint, "final_checkpoints")
                
                # End MLflow run
                if mlflow.active_run() is not None:
                    mlflow.end_run()
                    logger.info("MLflow run ended")
            
            # Rename output directory with MAE
            mae_str = f"{self.best_mae:.3f}"
            new_output_dir = f"{self.output_dir}_mae{mae_str}"
            if self.output_dir != new_output_dir and not os.path.exists(new_output_dir):
                os.rename(self.output_dir, new_output_dir)
                logger.info(f"Renamed output directory to {new_output_dir}")
        
        self.accelerator.end_training()
