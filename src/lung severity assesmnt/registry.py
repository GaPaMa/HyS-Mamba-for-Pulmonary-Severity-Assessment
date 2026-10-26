
import sys
import importlib
import torch.nn as nn
from typing import Dict, Type, Any, Callable, Optional, Union, List
from copy import deepcopy


class Registry:
    """A registry to map strings to classes or functions.
    
    Args:
        name (str): Registry name.
        locations (List[str]): List of module paths to search for registered items.
    """
    def __init__(self, name: str, locations: Optional[List[str]] = None):
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}
        self._locations = locations or []
        self._imported = False

    def _import_modules(self):
        """Import modules from specified locations."""
        if self._imported:
            return
            
        for location in self._locations:
            if location not in sys.path:
                sys.path.append(location)
            try:
                importlib.import_module(location)
            except ImportError:
                pass
        self._imported = True

    def register(self, name: str) -> Callable:
        """Register a module.
        
        Args:
            name (str): Module name to be registered.
            
        Returns:
            callable: A decorator to register the module.
        """
        def _register(cls: Type[Any]) -> Type[Any]:
            if name in self._module_dict:
                raise KeyError(f'{name} is already registered in {self._name}')
            self._module_dict[name] = cls
            return cls
        return _register

    def get(self, name: str) -> Type[Any]:
        """Get the registered module by name.
        
        Args:
            name (str): Module name to get.
            
        Returns:
            Type[Any]: The registered module.
            
        Raises:
            KeyError: If the module is not registered.
        """
        self._import_modules()
        if name not in self._module_dict:
            raise KeyError(f'{name} is not registered in {self._name}')
        return self._module_dict[name]

    def build(self, cfg: Union[Dict, Type[Any]], **kwargs) -> Any:
        """Build a module from config or return the module itself.
        
        Args:
            cfg (Union[Dict, Type[Any]]): Config dict or module class.
            **kwargs: Arguments passed to the module constructor.
            
        Returns:
            Any: The built module.
            
        Raises:
            TypeError: If cfg is neither a dict nor a module.
        """
        if cfg is None:
            return None
            
        if isinstance(cfg, dict):
            # Handle special case for optimizer params which contains a generator
            if 'params' in cfg:
                params = cfg.pop('params')
                cfg = deepcopy(cfg)
                cfg['params'] = params
            else:
                cfg = deepcopy(cfg)
            for k, v in kwargs.items():
                cfg[k] = v
                
            if 'type' not in cfg:
                raise KeyError('`type` must be specified in config dict')
                
            module_type = cfg.pop('type')
            module = self.get(module_type)
            return module(**cfg)
            
        elif isinstance(cfg, nn.Module):
            return cfg
            
        else:
            raise TypeError(f'Only support dict and nn.Module, but got {type(cfg)}')

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._module_dict

    def __getitem__(self, name: str) -> Type[Any]:
        """Get the registered module by name."""
        return self.get(name)

    def __len__(self) -> int:
        """Get the number of registered modules."""
        return len(self._module_dict)

    def __repr__(self) -> str:
        """Get the string representation of the registry."""
        return f'{self._name}({list(self._module_dict.keys())})'


DATASETS = Registry('dataset', locations=['age_estimation.datasets'])
MODELS = Registry('model', locations=['age_estimation.models'])
OPTIMIZERS = Registry('optimizer', locations=['age_estimation.optimizers'])
SCHEDULERS = Registry('scheduler', locations=['age_estimation.schedulers'])
LOSSES = Registry('loss', locations=['age_estimation.losses'])
METRICS = Registry('metric', locations=['age_estimation.metrics'])