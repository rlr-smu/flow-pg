from dataclasses import dataclass
from experiments.common.setup_experiment import get_value_logger
# from experiments.problems import all_problems, BaseProblem
from core.flow.real_nvp import RealNvp
from omegaconf import DictConfig, OmegaConf
import yaml
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from core.constraints import BaseConstraint
import torch as th
import hydra
import gym
from hydra.utils import instantiate
from experiments.config import TaskConfig

import pybulletgym

@dataclass
class Cfg:
    task: TaskConfig
    cv_error_margin: float


@hydra.main(version_base=None, config_path="./conf", config_name="train_rl")
def main(cfg: Cfg):
    print(OmegaConf.to_yaml(cfg))

    value_logger = get_value_logger()
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    constraint:BaseConstraint = instantiate(cfg.task.action_constraint, _convert_="all")
    device = "cuda:1"
    constraint = constraint.to(device)
    env = gym.make(**cfg.task.env)
    extra_kwargs = {
        "env": env,
        "constraint": constraint,
        "state_indexes": cfg.task.state_indexes,
        "cv_error_margin": cfg.cv_error_margin
    }
    # Construct action noise
    if "action_noise" in cfg.agent:
        n_actions = env.action_space.shape[-1]
        extra_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions), sigma= cfg.agent.action_noise * np.ones(n_actions))

    model = instantiate(cfg.agent,  _convert_="all", **extra_kwargs)
    print(model)
    model.set_logger(value_logger)
    model.learn(total_timesteps=cfg.task.total_timesteps)


if __name__ == "__main__":
    main()
    



