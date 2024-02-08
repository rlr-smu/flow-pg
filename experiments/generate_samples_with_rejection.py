import seaborn
import pandas as pd
import numpy as np
from typing import List
import time
# from experiments.problems import all_problems, BaseProblem, BaseConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig

from core.constraints import BaseConstraint
# from experiments.common.setup_experiment import setup_experiment, flush_logs
import torch as th
from experiments.common.setup_experiment import get_log_dir
import hydra
from hydra.utils import instantiate
import logging

logger = logging.getLogger(__name__)

# def get_samples_with_rejection(constraint: BaseProblem, sample_count: int):

    

# def plot_3d(df):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection = '3d')
#     dimNames = [f"Dim {i}" for i in range(3)]
#     x, y, z = [df[n] for n in dimNames]


#     ax.set_xlabel(dimNames[0])
#     ax.set_ylabel(dimNames[1])
#     ax.set_zlabel(dimNames[2])
#     ax.scatter(x, y, z, s=0.1, alpha=0.1) 
#     return fig


@hydra.main(version_base=None, config_path="./conf", config_name="generate_samples_with_rejection")
def main(cfg: DictConfig):
    log_dir = get_log_dir()
    cfg_const = cfg.task.constraint
    constraint:BaseConstraint  = instantiate(cfg_const.action_constraint, _convert_="all")
    s_c = 0
    batch_size = 100000
    valid_samples = []
    state_bounds = cfg_const.state_bounds
    while s_c <= cfg.count:
        if constraint.conditional_param_count > 0:
            state = th.rand((batch_size, constraint.conditional_param_count))*(state_bounds[1] - state_bounds[0]) + state_bounds[0]
        else:
            state = None
        actions = th.rand((batch_size, constraint.var_count))*2 - 1
        values = th.concat([actions,state], dim=1)
        validity = constraint.is_feasible(actions, state, 0)
        valid = values[validity]
        valid_samples.append(valid)
        s_c += len(valid)
    
    data = th.concat(valid_samples, dim=0)[:cfg.count]
    if cfg.plot:
        df = pd.DataFrame({f"Dim {i}": data[:, i] for i in range(data.shape[1])})
        seaborn.pairplot(df, plot_kws={"s": 1}).savefig(f"{log_dir}/plot.png")
    np.save(f"{log_dir}/data.npy", data)
    logger.info(f"Done, count:{len(data)}")


if __name__ == "__main__":
    main()