from dataclasses import dataclass
from typing import Tuple, List
from core.constraints import BaseConstraint
from omegaconf import OmegaConf
import yaml

# Define a custom resolver for the !tuple tag
def tuple_constructor(loader, node):
    res = tuple(loader.construct_sequence(node))
    print(">", node, res)
    return res

# yaml.SafeLoader.add_constructor('!tuple', tuple_constructor)
OmegaConf.register_new_resolver("to_tuple", lambda *args: tuple(args))


@dataclass
class RlCommonConfig:
    total_timesteps: int
    train_freq: tuple

@dataclass
class RlConfig:
    common: RlCommonConfig

@dataclass
class TaskConfig:
    action_constraint: BaseConstraint
    state_bounds: Tuple[float, float]
    state_indexes: List[float] # State indexs from the environment state
    var_count: int
    action_low: float
    action_high: float
    env: dict
    rl: RlConfig