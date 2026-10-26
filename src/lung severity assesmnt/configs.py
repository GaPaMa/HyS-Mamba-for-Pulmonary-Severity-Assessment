from typing import Any, Dict

import os
import importlib.util
import argparse

class Configs:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict

        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Configs(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # return self.config_dict[key]
        try:
            return self.config_dict[key]
        except KeyError:
            # MUST raise AttributeError so getattr(..., default) works
            raise AttributeError(key)

    def __repr__(self) -> str:
        return f"Configs({self.config_dict})"
    
    def get(self, key: str, default: Any = None) -> Any:
        value = self.config_dict.get(key, default)
        setattr(self, key, value)
        self.config_dict[key] = value
        return value

    def to_dict(self) -> Dict[str, Any]:

        result = {}
        for key, value in self.config_dict.items():
            if isinstance(value, Configs):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    

def parse_args(training=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="model config file path")
    parser.add_argument("--seed", default=None, type=int, help="seed for reproducibility")
    parser.add_argument("--num_workers", default=None, type=int, help="seed for reproducibility")
    parser.add_argument("--dtype", default=None, type=str, help="data type: fp16 bf16 ...")
    parser.add_argument(
        "--project_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions, checkpoints and logs will be written.",
    )

    if not training:
        parser.add_argument("--save_dir", default=None, type=str, help="path to save generated samples")
    else:
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="tensorboard",
            help='Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument(
            "--resume-from-checkpoint",
            type=str,
            default=None,
            help="If the training should continue from a checkpoint folder."
        )

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    """Merge command line arguments into config."""
    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is not None:
            cfg[k] = v
        elif k not in cfg:
            cfg[k] = v
    return cfg


def read_config(config_path):
    """Read Python config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Convert config to dictionary
    cfg = {}
    for key in dir(config_module):
        if not key.startswith('__'):
            value = getattr(config_module, key)
            if isinstance(value, dict):
                cfg[key] = value
            else:
                cfg[key] = value
    
    if 'project_dir' not in cfg:
        cfg['project_dir'] = 'outputs'

    if 'description' not in cfg:
        cfg['description'] = 'desc'

    if 'os' in cfg:
        del cfg['os']
    
    return cfg


def parse_configs(training=False):
    """Parse both command line arguments and config file."""
    args = parse_args(training)
    cfg = read_config(args.config_file)
    cfg = merge_args(cfg, args, training)

    return Configs(cfg)
