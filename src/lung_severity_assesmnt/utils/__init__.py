"""Compatibility imports for the original training entry point."""
from ..configs import Configs, parse_configs, read_config


def __getattr__(name):
    if name == "AgeEstimationTrainer":
        from ..trainer import AgeEstimationTrainer
        return AgeEstimationTrainer
    raise AttributeError(name)
