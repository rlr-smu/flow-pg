from dataclasses import dataclass
from experiments.common.setup_experiment import setup_experiment, flush_logs, get_value_logger
# from experiments.problems import all_problems, BaseProblem
# from core.rl.ddpg_flow import DDPGFlow, DDPGProj
from core.flow.real_nvp import RealNvp
from omegaconf import DictConfig, OmegaConf
from core.constraints import BaseConstraint
import torch as th
import hydra
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="./conf", config_name="train_rl")
def main(cfg: DictConfig):
    logger = get_value_logger(log_dir)
    print(cfg)
    constraint:BaseConstraint = instantiate(cfg.constraint, _convert_="all")
    device = "cuda:1"
    constraint.to(device)
    print(constraint)
    print( constraint.get_cv(th.tensor([[1, 2], [3, 4]], dtype=th.float, device=device), None))
    exit()

    problem: BaseProblem = all_problems[cfg.problem]
    env = problem.get_env()
    if cfg.agent == "ddpgflow":
        flow = RealNvp.load_module(cfg.agent_configs.ddpgflow.flow_file)
        model = DDPGFlow(flow, problem, env=env)
    elif cfg.agent == "ddpgproj":
        model = DDPGProj(problem, policy="MlpPolicy", env=env)
        

    model.set_logger(logger)
    model.learn(total_timesteps=problem.n_timesteps)


if __name__ == "__main__":
    main()
    



